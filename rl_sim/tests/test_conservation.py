"""Packet-accounting tests (spec item J, conservation and hop limit)."""

from __future__ import annotations

import pytest

from fanet_sim import config
from fanet_sim.envs.packet import DROP_REASON_LABELS

from .conftest import EpisodeTrace, drop_counts_by_reason


def test_every_packet_has_exactly_one_outcome(episode: EpisodeTrace) -> None:
    """No packet is both delivered and dropped, and none is counted twice."""
    env = episode.env
    for pkt in env.all_packets:
        assert not (pkt.delivered and pkt.dropped), f"{pkt} both delivered and dropped"

    delivered_ids = {p.packet_id for p in env.delivered}
    dropped_ids = {p.packet_id for p in env.dropped}
    assert len(delivered_ids) == len(env.delivered), "a packet was delivered twice"
    assert len(dropped_ids) == len(env.dropped), "a packet was dropped twice"
    assert delivered_ids.isdisjoint(dropped_ids)


def test_conservation_created_equals_delivered_dropped_and_queued(
    episode: EpisodeTrace,
) -> None:
    """J: created = delivered + dropped + still waiting in link queues."""
    env = episode.env
    in_queues = sum(d.total_queued() for d in env.drones)

    assert len(env.all_packets) == len(env.delivered) + len(env.dropped) + in_queues

    # And the same identity as a set, so nothing is lost or double-counted.
    accounted = (
        {p.packet_id for p in env.delivered}
        | {p.packet_id for p in env.dropped}
        | {p.packet_id for d in env.drones for p in d.queued_packets()}
    )
    assert accounted == {p.packet_id for p in env.all_packets}


def test_drops_split_cleanly_by_reason(episode: EpisodeTrace) -> None:
    """Every drop carries one of the five reasons, and the parts sum to the whole."""
    counts = drop_counts_by_reason(episode.env)
    assert set(counts).issubset(set(DROP_REASON_LABELS)), (
        f"unexpected drop reason: {set(counts) - set(DROP_REASON_LABELS)}"
    )
    assert sum(counts.values()) == len(episode.env.dropped)


def test_packets_never_exceed_the_hop_limit(episode: EpisodeTrace) -> None:
    """A packet takes at most MAX_HOPS hops (the off-by-one fix)."""
    env = episode.env
    for pkt in env.all_packets:
        assert pkt.hop_count <= config.MAX_HOPS, f"{pkt} took {pkt.hop_count} hops"
        # path is [source, hop1, hop2, ...], so it holds hop_count + 1 entries.
        assert len(pkt.path) == pkt.hop_count + 1


def test_hop_limit_drops_really_sit_at_the_limit(episode: EpisodeTrace) -> None:
    """A packet dropped as hop_limit has used up exactly its hop budget."""
    for pkt in episode.env.dropped:
        if pkt.drop_reason is not None and pkt.drop_reason.label == "hop_limit":
            assert pkt.hop_count == config.MAX_HOPS


def test_delivered_packets_beat_the_ttl(episode: EpisodeTrace) -> None:
    """Delivery always happens strictly inside the packet's lifetime."""
    for pkt in episode.env.delivered:
        delay = pkt.delay()
        assert delay is not None
        assert 0 <= delay < config.PACKET_TTL


# ---------------------------------------------------------------------------
# Unit-level checks for the two expiry paths.
#
# At the Phase-1A saturating load (one packet per M-drone per step, N = 1) a
# packet is always resolved within a few steps, so a full run produces no "ttl"
# or "hop_limit" drops at all. These exercise those paths directly.
# ---------------------------------------------------------------------------

import numpy as np

from fanet_sim.envs.drone import Drone
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import DropReason, PacketFactory


def _packet(created_at: int = 0):
    """Return a fresh packet using the configured TTL and hop limit."""
    factory = PacketFactory(
        ttl=config.PACKET_TTL, max_hops=config.MAX_HOPS, size_bytes=512
    )
    return factory.create(source_id=0, created_at=created_at)


def test_is_alive_ttl_boundary() -> None:
    """A packet lives for exactly TTL steps: alive at TTL-1, dead at TTL."""
    pkt = _packet(created_at=0)
    assert pkt.is_alive(config.PACKET_TTL - 1) is True
    assert pkt.is_alive(config.PACKET_TTL) is False


def test_is_alive_hop_boundary() -> None:
    """A packet may take MAX_HOPS hops, and is dead once it has taken them."""
    pkt = _packet()
    for hop in range(config.MAX_HOPS):
        assert pkt.is_alive(0) is True, f"died early at hop {hop}"
        pkt.relay_to(hop + 1)
    assert pkt.hop_count == config.MAX_HOPS
    assert pkt.is_alive(0) is False


def _env_for_expiry() -> FANETEnv:
    """A reset environment used only to drive _expire_packet."""
    import os

    env = FANETEnv(routing="greedy", log_path=os.devnull)
    env.reset()
    return env


def test_expiry_classifies_a_timed_out_packet_as_ttl() -> None:
    """A packet that ran out of time, with hops to spare, is a "ttl" drop."""
    env = _env_for_expiry()
    pkt = _packet(created_at=0)
    pkt.relay_to(1)  # one hop used, far below the limit
    env._expire_packet(pkt, holder=env.drones[0])
    assert pkt.drop_reason is DropReason.TTL


def test_expiry_classifies_a_hop_exhausted_packet_as_hop_limit() -> None:
    """A packet that used up its hop budget is a "hop_limit" drop."""
    env = _env_for_expiry()
    pkt = _packet(created_at=0)
    for hop in range(config.MAX_HOPS):
        pkt.relay_to(hop + 1)
    env._expire_packet(pkt, holder=env.drones[0])
    assert pkt.drop_reason is DropReason.HOP_LIMIT


def test_queued_packet_past_its_ttl_is_expired_by_the_env() -> None:
    """The env's queue sweep really does retire a stale packet."""
    env = _env_for_expiry()
    drone = env.drones[0]
    stale = _packet(created_at=0)
    drone.link_queues[7] = [stale]
    env.step_count = config.PACKET_TTL

    env._expire_queued_packets()

    assert drone.link_queues[7] == []
    assert stale.dropped and stale.drop_reason is DropReason.TTL
    assert stale in env.dropped
    # The link it was waiting on is charged for it.
    assert drone.link_stats[7]["expired_in_queue"] == 1
    assert env.link_stats[(drone.drone_id, 7)]["expired_in_queue"] == 1


def test_traffic_interval_controls_how_many_packets_exist() -> None:
    """Halving the interval roughly doubles the packets created."""
    import os

    from .conftest import run_episode

    prev = config.PACKET_INTERVAL_STEPS
    try:
        config.PACKET_INTERVAL_STEPS = 10
        light = len(run_episode(steps=200, placement_seed=5, run_seed=5).env.all_packets)
        config.PACKET_INTERVAL_STEPS = 5
        heavy = len(run_episode(steps=200, placement_seed=5, run_seed=5).env.all_packets)
    finally:
        config.PACKET_INTERVAL_STEPS = prev

    n_sources = config.NUM_M_DRONES
    assert light == pytest.approx(200 / 10 * n_sources, rel=0.1)
    assert heavy == pytest.approx(200 / 5 * n_sources, rel=0.1)


def test_one_packet_per_interval_per_source() -> None:
    """Each M-drone creates exactly one packet per interval, on its own offset."""
    from .conftest import run_episode

    env = run_episode(steps=120, placement_seed=5, run_seed=5).env
    interval = config.PACKET_INTERVAL_STEPS

    per_source: dict[int, list[int]] = {}
    for pkt in env.all_packets:
        per_source.setdefault(pkt.source_id, []).append(pkt.created_at)

    for source_id, steps in per_source.items():
        offset = env._traffic_offset[source_id]
        assert all(s % interval == offset for s in steps)
        assert steps == sorted(steps)
        assert len(set(steps)) == len(steps), "two packets on one step"
