"""ACK-bookkeeping tests (spec item J, ACK-vs-ground-truth).

ACKs are ideal in Phase 1A — instant and never lost — so every counter a drone
builds from ACK information must match the simulator's own view exactly.
"""

from __future__ import annotations

from fanet_sim import config

from .conftest import EpisodeTrace, drop_counts_by_reason


def test_created_counts_match(episode: EpisodeTrace) -> None:
    """Sources agree with the simulator on how many packets were created."""
    env = episode.env
    assert sum(d.packets_created for d in env.drones) == len(env.all_packets)
    # Only M-drones create packets.
    assert all(d.packets_created == 0 for d in env.drones if d.drone_type == "C")


def test_gs_acks_match_deliveries(episode: EpisodeTrace) -> None:
    """Every delivered packet produced exactly one end-to-end ACK at its source."""
    env = episode.env
    assert sum(d.packets_gs_acked for d in env.drones) == len(env.delivered)

    # Per source, too — not just in total.
    per_source: dict[int, int] = {}
    for pkt in env.delivered:
        per_source[pkt.source_id] = per_source.get(pkt.source_id, 0) + 1
    for drone in env.drones:
        assert drone.packets_gs_acked == per_source.get(drone.drone_id, 0)


def test_per_link_channel_losses_match(episode: EpisodeTrace) -> None:
    """Hop-by-hop ACKs account for every channel drop, on the right link."""
    env = episode.env
    counts = drop_counts_by_reason(env)
    acked = sum(
        r["lost_channel"] for d in env.drones for r in d.link_stats.values()
    )
    assert acked == counts.get("channel", 0)


def test_per_link_queue_full_losses_match(episode: EpisodeTrace) -> None:
    """Every queue_full drop is charged to the link whose queue refused it."""
    env = episode.env
    counts = drop_counts_by_reason(env)
    on_link = sum(
        r["dropped_queue_full"] for d in env.drones for r in d.link_stats.values()
    )
    assert on_link == counts.get("queue_full", 0)


def test_per_link_expiries_match(episode: EpisodeTrace) -> None:
    """Every ttl / hop_limit drop is accounted for exactly once.

    A packet that died waiting is charged to the link queue it waited on. One
    that arrived already out of TTL or hops never waited anywhere, so no link
    is charged; it lands in the env's expired_on_arrival instead.
    """
    env = episode.env
    counts = drop_counts_by_reason(env)
    expired_in_queue = sum(
        r["expired_in_queue"] for d in env.drones for r in d.link_stats.values()
    )
    total_expired = counts.get("ttl", 0) + counts.get("hop_limit", 0)

    assert expired_in_queue + env.expired_on_arrival == total_expired


def test_link_counters_are_internally_consistent(episode: EpisodeTrace) -> None:
    """Per link the books balance, both on offer and on transmission.

    offered = refused + sent + expired-while-waiting + still waiting
    sent    = acked + lost_channel
    """
    for drone in episode.env.drones:
        for key, row in drone.link_stats.items():
            waiting = drone.queue_len(key)
            assert row["offered"] == (
                row["dropped_queue_full"]
                + row["sent"]
                + row["expired_in_queue"]
                + waiting
            ), f"drone {drone.drone_id} link {key} offer book does not balance"
            assert row["sent"] == row["acked"] + row["lost_channel"], (
                f"drone {drone.drone_id} link {key} send book does not balance"
            )


def test_acked_sends_match_accepted_transmissions(episode: EpisodeTrace) -> None:
    """Total positive ACKs equal the transmissions the simulator accepted."""
    acked = sum(
        r["acked"] for d in episode.env.drones for r in d.link_stats.values()
    )
    assert acked == episode.tx_total


def test_source_side_timer_matches_unacked_packets(episode: EpisodeTrace) -> None:
    """A source gives up on exactly those own packets the GS never acked in time.

    The timer last ran at the start of the final step, so it has been evaluated
    up to step MAX_STEPS - 1.
    """
    env = episode.env
    last_timer_step = env.step_count - 1

    expected = sum(
        1
        for pkt in env.all_packets
        if not pkt.delivered
        and last_timer_step - pkt.created_at >= config.PACKET_TTL
    )
    assert sum(d.packets_lost_no_ack for d in env.drones) == expected


def test_hop_acks_only_report_ok_or_channel() -> None:
    """A send either arrives or is lost: the receiver never refuses it.

    Uses a standalone drone — mutating the shared episode's counters here would
    leave a link in one book and not the other.
    """
    import numpy as np
    import pytest

    from fanet_sim.envs.drone import Drone

    drone = Drone(
        drone_id=0,
        drone_type="M",
        initial_position=np.array([10.0, 10.0]),
        speed=10.0,
        gs_position=np.array(config.GS_POSITION, dtype=np.float64),
    )
    drone.note_hop_ack(1, "ok")
    drone.note_hop_ack(1, "channel")
    assert drone.link_stats[1]["acked"] == 1
    assert drone.link_stats[1]["lost_channel"] == 1

    with pytest.raises(ValueError, match="unknown hop-ACK outcome"):
        drone.note_hop_ack(1, "queue_full")


def test_source_accounting_closes(episode: EpisodeTrace) -> None:
    """Per source: created = acked + given-up-on + still waiting."""
    for drone in episode.env.drones:
        assert drone.packets_created == (
            drone.packets_gs_acked
            + drone.packets_lost_no_ack
            + len(drone.outstanding)
        ), f"drone {drone.drone_id} source accounting does not close"


def test_gs_link_is_counted_as_a_link(episode: EpisodeTrace) -> None:
    """Deliveries to the GS register on the sender's "GS" link, not nowhere."""
    env = episode.env
    gs_acked = sum(
        d.link_stats.get("GS", {}).get("acked", 0) for d in env.drones
    )
    assert gs_acked == len(env.delivered)


def test_ack_link_books_match_the_simulator_per_link(
    episode: EpisodeTrace,
) -> None:
    """Every drone's own link book equals the simulator's, field by field."""
    from fanet_sim.envs.drone import LINK_FIELDS

    env = episode.env
    ack_rows = {
        (d.drone_id, next_hop): dict(row)
        for d in env.drones
        for next_hop, row in d.link_stats.items()
    }
    truth_rows = {link: dict(row) for link, row in env.link_stats.items()}

    assert set(ack_rows) == set(truth_rows)
    for link in truth_rows:
        for field in LINK_FIELDS:
            assert ack_rows[link][field] == truth_rows[link][field], (
                f"link {link} field {field}: ACK {ack_rows[link][field]} "
                f"vs simulator {truth_rows[link][field]}"
            )
