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
    acked = sum(sum(d.link_lost_channel.values()) for d in env.drones)
    assert acked == counts.get("channel", 0)


def test_per_link_queue_full_losses_match(episode: EpisodeTrace) -> None:
    """queue_full drops split into link refusals plus packets born into a full queue."""
    env = episode.env
    counts = drop_counts_by_reason(env)
    on_link = sum(sum(d.link_lost_queue_full.values()) for d in env.drones)
    at_birth = sum(d.own_queue_full_drops for d in env.drones)
    assert on_link + at_birth == counts.get("queue_full", 0)


def test_link_counters_are_internally_consistent(episode: EpisodeTrace) -> None:
    """Per link: sent = acked + lost_channel + lost_queue_full."""
    for drone in episode.env.drones:
        keys = (
            set(drone.link_sent)
            | set(drone.link_acked)
            | set(drone.link_lost_channel)
            | set(drone.link_lost_queue_full)
        )
        for key in keys:
            assert drone.link_sent[key] == (
                drone.link_acked[key]
                + drone.link_lost_channel[key]
                + drone.link_lost_queue_full[key]
            ), f"drone {drone.drone_id} link {key} does not balance"


def test_acked_sends_match_accepted_transmissions(episode: EpisodeTrace) -> None:
    """Total positive ACKs equal the transmissions the simulator accepted."""
    acked = sum(sum(d.link_acked.values()) for d in episode.env.drones)
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
    gs_acked = sum(d.link_acked.get("GS", 0) for d in env.drones)
    assert gs_acked == len(env.delivered)
