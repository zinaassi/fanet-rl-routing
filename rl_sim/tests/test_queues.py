"""Finite-queue tests (spec item J, first two bullets)."""

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


def test_enqueue_refuses_past_capacity() -> None:
    """The queue accepts exactly QUEUE_CAPACITY packets and then refuses."""
    drone = _drone()
    factory = PacketFactory(ttl=50, max_hops=10, size_bytes=512)

    for i in range(config.QUEUE_CAPACITY):
        assert drone.enqueue(factory.create(source_id=0, created_at=i)) is True

    assert len(drone.queue) == config.QUEUE_CAPACITY
    assert drone.queue_is_full()
    # Every further offer is refused and the queue does not grow.
    for i in range(5):
        assert drone.enqueue(factory.create(source_id=0, created_at=i)) is False
    assert len(drone.queue) == config.QUEUE_CAPACITY


def test_dequeue_for_send_takes_the_oldest_first() -> None:
    """The send batch is the head of the FIFO, in creation order."""
    drone = _drone()
    factory = PacketFactory(ttl=50, max_hops=10, size_bytes=512)
    packets = [factory.create(source_id=0, created_at=i) for i in range(5)]
    for pkt in packets:
        drone.enqueue(pkt)

    batch = drone.dequeue_for_send(3)
    assert [p.packet_id for p in batch] == [p.packet_id for p in packets[:3]]
    assert [p.packet_id for p in drone.queue] == [p.packet_id for p in packets[3:]]


def test_dequeue_for_send_defaults_to_the_step_budget() -> None:
    """With no argument the batch is at most MAX_TX_PER_STEP packets."""
    drone = _drone()
    factory = PacketFactory(ttl=50, max_hops=10, size_bytes=512)
    for i in range(config.QUEUE_CAPACITY):
        drone.enqueue(factory.create(source_id=0, created_at=i))

    assert len(drone.dequeue_for_send()) == config.MAX_TX_PER_STEP


def test_queue_never_exceeds_capacity_during_a_run(episode: EpisodeTrace) -> None:
    """J: a queue never holds more than Q packets."""
    assert episode.max_queue_seen <= config.QUEUE_CAPACITY
    for drone in episode.env.drones:
        assert len(drone.queue) <= config.QUEUE_CAPACITY


def test_no_drone_sends_more_than_n_per_step(episode: EpisodeTrace) -> None:
    """J: a drone never sends more than N packets per step.

    Checked two ways: the batch released from the queue (which also covers
    failed sends and no_route drops), and the transmissions actually attempted.
    """
    assert episode.max_released_step <= config.MAX_TX_PER_STEP
    assert episode.max_sent_step <= config.MAX_TX_PER_STEP


def test_the_queue_actually_fills_up(episode: EpisodeTrace) -> None:
    """Guard: the run must exercise the cap, or the two tests above are vacuous."""
    assert episode.max_queue_seen == config.QUEUE_CAPACITY
