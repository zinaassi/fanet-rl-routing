"""Per-link queue tests (spec item 8, queue bullets).

Each drone keeps one FIFO queue per outgoing link. Every queue holds at most Q
packets and releases at most N per step, but the queues are independent: a
drone with several links sends on all of them in the same step.
"""

from __future__ import annotations

import numpy as np

from fanet_sim import config
from fanet_sim.envs.drone import Drone
from fanet_sim.envs.packet import PacketFactory

from .conftest import EpisodeTrace


def _drone() -> Drone:
    """Return a lone M-drone for unit-level queue tests."""
    return Drone(
        drone_id=0,
        drone_type="M",
        initial_position=np.array([10.0, 10.0]),
        speed=10.0,
        gs_position=np.array(config.GS_POSITION, dtype=np.float64),
    )


def _factory() -> PacketFactory:
    """Return a packet factory using the configured TTL and hop limit."""
    return PacketFactory(
        ttl=config.PACKET_TTL, max_hops=config.MAX_HOPS, size_bytes=512
    )


def test_enqueue_refuses_past_capacity_per_link() -> None:
    """One link queue accepts exactly Q packets and then refuses."""
    drone = _drone()
    factory = _factory()

    for i in range(config.QUEUE_CAPACITY):
        assert drone.enqueue(1, factory.create(source_id=0, created_at=i)) is True

    assert drone.queue_len(1) == config.QUEUE_CAPACITY
    assert drone.queue_is_full(1)
    for i in range(5):
        assert drone.enqueue(1, factory.create(source_id=0, created_at=i)) is False
    assert drone.queue_len(1) == config.QUEUE_CAPACITY


def test_link_queues_are_independent() -> None:
    """Filling one link's queue leaves the others empty and accepting."""
    drone = _drone()
    factory = _factory()

    for i in range(config.QUEUE_CAPACITY):
        drone.enqueue(1, factory.create(source_id=0, created_at=i))

    assert drone.queue_is_full(1)
    assert not drone.queue_is_full(2)
    assert not drone.queue_is_full("GS")
    assert drone.enqueue(2, factory.create(source_id=0, created_at=0)) is True
    assert drone.enqueue("GS", factory.create(source_id=0, created_at=0)) is True
    assert drone.total_queued() == config.QUEUE_CAPACITY + 2


def test_dequeue_head_takes_the_oldest_first() -> None:
    """A link queue is FIFO: the head is the oldest packet on that link."""
    drone = _drone()
    factory = _factory()
    packets = [factory.create(source_id=0, created_at=i) for i in range(5)]
    for pkt in packets:
        drone.enqueue(1, pkt)

    batch = drone.dequeue_head(1, 3)
    assert [p.packet_id for p in batch] == [p.packet_id for p in packets[:3]]
    assert [p.packet_id for p in drone.link_queues[1]] == [
        p.packet_id for p in packets[3:]
    ]


def test_dequeue_head_defaults_to_the_per_link_budget() -> None:
    """With no argument a link releases at most MAX_TX_PER_STEP packets."""
    drone = _drone()
    factory = _factory()
    for i in range(config.QUEUE_CAPACITY):
        drone.enqueue(1, factory.create(source_id=0, created_at=i))

    assert len(drone.dequeue_head(1)) == config.MAX_TX_PER_STEP


def test_every_link_queue_stays_within_capacity(episode: EpisodeTrace) -> None:
    """J: every link queue holds at most Q packets."""
    assert episode.max_queue_seen <= config.QUEUE_CAPACITY
    for drone in episode.env.drones:
        for next_hop, queue in drone.link_queues.items():
            assert len(queue) <= config.QUEUE_CAPACITY, (
                f"drone {drone.drone_id} link {next_hop} overfilled"
            )


def test_each_link_queue_sends_at_most_n_per_step(episode: EpisodeTrace) -> None:
    """J: each link queue releases at most N packets per step."""
    assert episode.max_released_step <= config.MAX_TX_PER_STEP


def test_a_drone_can_send_on_several_links_in_one_step(
    episode: EpisodeTrace,
) -> None:
    """J: there is no per-drone send limit — parallel links really are used."""
    assert episode.multi_link_sends > 0, (
        "no drone ever sent on two links in the same step"
    )
    # And the per-drone total can exceed the per-link budget.
    assert episode.max_sent_step > config.MAX_TX_PER_STEP


def test_the_queues_actually_fill_up(episode: EpisodeTrace) -> None:
    """Guard: the run must exercise the cap, or the cap tests are vacuous."""
    assert episode.max_queue_seen == config.QUEUE_CAPACITY
