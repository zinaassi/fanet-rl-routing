"""End-to-end ACK, deadlines and per-link delivery rates (Phase 1b)."""

from __future__ import annotations

import os

import pytest

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import PacketFactory

from agents import make_router
from agents.tracker import Tracker


def _tracker() -> Tracker:
    """Return an empty tracker with the configured window."""
    return Tracker()


# ---------------------------------------------------------------------------
# Delivery rates
# ---------------------------------------------------------------------------

def test_delivery_rate_starts_at_one_minus_channel_loss() -> None:
    """Before any outcome, a link is guessed at 1 - channel_loss."""
    tracker = _tracker()
    guess = 1.0 - channel.p_loss(200.0)
    assert tracker.delivery_rate(0, 1, default=guess) == pytest.approx(guess)


def test_delivery_rate_is_the_mean_of_resolved_outcomes() -> None:
    """Once outcomes exist, the guess is replaced by their mean."""
    tracker = _tracker()
    for delivered in (True, True, False, True):
        tracker._record_outcome(0, 1, delivered)
    assert tracker.delivery_rate(0, 1, default=0.9) == pytest.approx(3 / 4)


def test_delivery_rate_keeps_only_the_last_w_outcomes() -> None:
    """The window slides: older outcomes fall out."""
    tracker = _tracker()
    window = config.DELIVERY_RATE_WINDOW
    for _ in range(window):
        tracker._record_outcome(0, 1, False)
    assert tracker.delivery_rate(0, 1, default=1.0) == pytest.approx(0.0)

    for _ in range(window):
        tracker._record_outcome(0, 1, True)
    assert tracker.delivery_rate(0, 1, default=0.0) == pytest.approx(1.0)
    assert len(tracker.rates[(0, 1)]) == window


def test_rates_are_kept_per_drone_and_per_link() -> None:
    """One drone's measurements never leak into another's."""
    tracker = _tracker()
    tracker._record_outcome(0, 1, True)
    tracker._record_outcome(5, 1, False)
    assert tracker.delivery_rate(0, 1, default=0.5) == pytest.approx(1.0)
    assert tracker.delivery_rate(5, 1, default=0.5) == pytest.approx(0.0)
    assert tracker.delivery_rate(9, 1, default=0.42) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _record_path(tracker: Tracker, packet_id: int, hops) -> None:
    """Record one decision per hop and mark each as actually sent."""
    for drone_id, next_hop in hops:
        tracker.record_decision(
            packet_id=packet_id, created_at=0, drone_id=drone_id,
            next_hop=next_hop, features=(0.5, 0.0, 0.1),
        )
        tracker.mark_sent(packet_id, drone_id, next_hop)


def test_a_delivered_packet_credits_every_drone_on_its_path() -> None:
    """The ACK walks back: each forwarder learns "delivered" for ITS link."""
    tracker = _tracker()
    _record_path(tracker, 7, [(3, 8), (8, 2), (2, "GS")])

    examples = tracker.resolve(7, delivered=True)

    assert len(examples) == 3
    assert all(target == 1.0 for _, target in examples)
    assert tracker.delivery_rate(3, 8, default=0.0) == pytest.approx(1.0)
    assert tracker.delivery_rate(8, 2, default=0.0) == pytest.approx(1.0)
    assert tracker.delivery_rate(2, "GS", default=0.0) == pytest.approx(1.0)
    # And only those links moved.
    assert set(tracker.rates) == {(3, 8), (8, 2), (2, "GS")}


def test_a_lost_packet_is_recorded_at_its_deadline_and_not_before() -> None:
    """Nothing resolves until created_at + TTL, then every forwarder does."""
    tracker = _tracker()
    created_at, ttl = 10, config.PACKET_TTL
    for drone_id, next_hop in [(1, 4), (4, 6)]:
        tracker.record_decision(
            packet_id=99, created_at=created_at, drone_id=drone_id,
            next_hop=next_hop, features=(0.5, 0.0, 0.1), ttl=ttl,
        )
        tracker.mark_sent(99, drone_id, next_hop)

    # Every step before the deadline resolves nothing.
    for step in range(created_at, created_at + ttl):
        assert tracker.resolve_deadlines(step) == []
    assert 99 in tracker.decisions

    examples = tracker.resolve_deadlines(created_at + ttl)
    assert len(examples) == 2
    assert all(target == 0.0 for _, target in examples)
    assert tracker.delivery_rate(1, 4, default=1.0) == pytest.approx(0.0)
    assert tracker.delivery_rate(4, 6, default=1.0) == pytest.approx(0.0)


def test_a_delivered_packet_is_not_resolved_again_at_its_deadline() -> None:
    """The ACK settles it; the deadline then finds nothing left."""
    tracker = _tracker()
    _record_path(tracker, 5, [(1, "GS")])
    assert len(tracker.resolve(5, delivered=True)) == 1
    assert tracker.resolve_deadlines(config.PACKET_TTL) == []
    assert tracker.delivery_rate(1, "GS", default=0.0) == pytest.approx(1.0)


def test_a_packet_that_revisits_a_drone_keeps_both_decisions() -> None:
    """The loop guard blocks only the previous hop, so A can decide twice."""
    tracker = _tracker()
    _record_path(tracker, 11, [(1, 2), (2, 3), (3, 1), (1, 4)])

    examples = tracker.resolve(11, delivered=False)

    assert len(examples) == 4, "every decision must resolve, including repeats"
    assert all(target == 0.0 for _, target in examples)


def test_an_unsent_decision_trains_but_does_not_move_the_rate() -> None:
    """A queue_full decision is a training example, not a delivery outcome."""
    tracker = _tracker()
    tracker.record_decision(
        packet_id=3, created_at=0, drone_id=1, next_hop=2,
        features=(0.5, 1.0, 0.1),
    )
    # No mark_sent: the queue refused it, so it was never transmitted.
    examples = tracker.resolve(3, delivered=False)

    assert len(examples) == 1 and examples[0][1] == 0.0
    assert (1, 2) not in tracker.rates
    assert tracker.delivery_rate(1, 2, default=0.77) == pytest.approx(0.77)


def test_reset_clears_everything() -> None:
    """A new run starts with no rates and no pending decisions."""
    tracker = _tracker()
    _record_path(tracker, 1, [(0, 1)])
    tracker.resolve(1, delivered=True)
    tracker.reset()
    assert tracker.rates == {} and tracker.decisions == {}
    assert tracker.deadlines == {}


# ---------------------------------------------------------------------------
# In a real run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["rate", "rl"])
def test_every_decision_resolves_or_is_still_in_flight(rule: str) -> None:
    """Over a run, decisions are settled; only recent ones remain pending."""
    router = make_router(rule, run_seed=5, training=False, init_seed=0)
    env = FANETEnv(routing=rule, log_path=os.devnull,
                   placement_seed=5, run_seed=5, router=router)
    env.reset()
    for _ in range(200):
        env.step()
    env.close_logger()

    # Anything still pending must be younger than one TTL.
    for packet_id in router.tracker.decisions:
        packet = next(p for p in env.all_packets if p.packet_id == packet_id)
        assert env.step_count - packet.created_at <= config.PACKET_TTL
    assert router.tracker.rates, "no link ever measured a delivery rate"
