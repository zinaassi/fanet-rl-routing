"""
test_2a_mobility.py — Moving M-drones, and the links that break under them.

Phase 2a step 1. The checks here fall into three groups:

* movement off is still Phase 1, to the last bit;
* the flight itself — speed, delay, the border, stopping at B;
* what happens to packets queued on a link that goes away.

Every test that needs movement turns it on through the ``moving`` fixture,
which also restores ``config.STATIC_MODE`` afterwards, so one test leaving
movement on cannot quietly change the next.
"""

from __future__ import annotations

import math
import os
import sys
from contextlib import contextmanager
from typing import Iterator, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.channel import are_connected
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import DropReason


@contextmanager
def moving(max_start_delay_s: float = config.MAX_START_DELAY_S) -> Iterator[None]:
    """Run the body with movement on, then put the config back."""
    static, delay = config.STATIC_MODE, config.MAX_START_DELAY_S
    config.STATIC_MODE = False
    config.MAX_START_DELAY_S = max_start_delay_s
    try:
        yield
    finally:
        config.STATIC_MODE = static
        config.MAX_START_DELAY_S = delay


@contextmanager
def load_ms(value: int) -> Iterator[None]:
    """Run the body at *value* ms per packet, then put the config back."""
    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(round(value / 1000.0 / config.TIMESTEP))
    try:
        yield
    finally:
        config.PACKET_INTERVAL_STEPS = previous


def _run(rule: str, seed: int, steps: int = config.MAX_STEPS) -> FANETEnv:
    """Run one episode and return the finished environment."""
    env = FANETEnv(routing=rule, log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    for _ in range(steps):
        env.step()
    env.close_logger()
    return env


def _m_drones(env: FANETEnv) -> List:
    """Return the M-drones, in id order."""
    return [d for d in sorted(env.drones, key=lambda d: d.drone_id)
            if d.drone_type == "M"]


# ---------------------------------------------------------------------------
# Movement off is still Phase 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ("greedy", "random"))
def test_movement_off_reproduces_the_static_reference(rule: str) -> None:
    """With STATIC_MODE on, the numbers are the frozen Phase-1 ones.

    The reference was recorded before any Phase-2a code existed, so this is
    the check that the whole of step 1 is inert while nothing moves.
    """
    from scripts.metrics_1a import compute_metrics
    from scripts.stamp import read_stamped_csv

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "out", "2a_reference.csv")
    if not os.path.exists(path):
        pytest.skip("out/2a_reference.csv is missing")
    expected = next(
        r for r in read_stamped_csv(path)
        if r["rule"] == rule and int(r["placement_seed"]) == 3
    )

    assert config.STATIC_MODE, "this check is about the static world"
    with load_ms(200):
        env = _run(rule, 3)
    truth = compute_metrics(env)["ground_truth"]

    assert (truth["total_loss"] or 0.0) == pytest.approx(
        float(expected["total_loss"]), abs=1e-12
    )
    for cause in ("channel", "queue_full", "no_route", "dead_end",
                  "ttl", "hop_limit"):
        assert truth["loss_rate_by_reason"][cause] == pytest.approx(
            float(expected[cause]), abs=1e-12
        )


def test_movement_off_breaks_no_links_and_rescues_nothing() -> None:
    """Static means static: no re-routes, no link_broken drops."""
    assert config.STATIC_MODE
    with load_ms(200):
        env = _run("greedy", 4)
    assert env.reroutes == 0
    assert env.link_broken_drops == 0
    assert all(p.drop_reason is not DropReason.LINK_BROKEN
               for p in env.all_packets if p.dropped)


def test_zero_speed_moves_nobody() -> None:
    """Movement ON but speed 0 leaves every drone where it started.

    A different switch from STATIC_MODE, and it has to agree with it.
    """
    with moving():
        previous = (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX)
        config.DRONE_SPEED_MIN = config.DRONE_SPEED_MAX = 0.0
        try:
            env = _run("greedy", 2, steps=200)
        finally:
            config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX = previous

    for drone in env.drones:
        assert np.allclose(drone.position, drone.start_point), (
            f"drone {drone.drone_id} moved at speed 0"
        )
    assert env.reroutes == 0
    assert env.link_broken_drops == 0


def test_the_layout_is_the_same_with_movement_on_and_off() -> None:
    """Turning movement on must not move anybody's STARTING position.

    This is what the separate mobility stream buys: the direction, speed and
    delay draws cannot shift the placement stream, so the layout a seed names
    is the same layout either way.
    """
    still = _run("greedy", 6, steps=0)
    still_positions = np.array([d.start_point for d in
                                sorted(still.drones, key=lambda d: d.drone_id)])
    with moving():
        flying = _run("greedy", 6, steps=0)
    flying_positions = np.array([d.start_point for d in
                                 sorted(flying.drones, key=lambda d: d.drone_id)])

    assert np.array_equal(still_positions, flying_positions)
    assert still.placement_draws == flying.placement_draws


# ---------------------------------------------------------------------------
# The flight
# ---------------------------------------------------------------------------

def test_each_step_moves_exactly_speed_times_dt() -> None:
    """Until it lands on B, an M-drone covers speed * 0.1 s every step.

    The one exception is the final step, which is short because the drone
    snaps to B rather than overshooting it.
    """
    with moving(max_start_delay_s=0.0):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=8, run_seed=8)
        env.reset()
        previous = {d.drone_id: d.position.copy() for d in _m_drones(env)}
        short_steps = {d.drone_id: 0 for d in _m_drones(env)}
        landed = {d.drone_id: False for d in _m_drones(env)}

        for _ in range(400):
            env.step()
            for drone in _m_drones(env):
                moved = float(np.linalg.norm(
                    drone.position - previous[drone.drone_id]
                ))
                expected = drone.speed * config.TIMESTEP
                if landed[drone.drone_id]:
                    # Already hovering at B: it must not move again.
                    assert moved == pytest.approx(0.0, abs=1e-12), (
                        f"drone {drone.drone_id} moved {moved} after landing"
                    )
                elif drone.arrived:
                    # The landing step itself: shorter than a full one, and
                    # there is exactly one of them.
                    assert 0.0 <= moved <= expected + 1e-9
                    short_steps[drone.drone_id] += 1
                    landed[drone.drone_id] = True
                else:
                    assert moved == pytest.approx(expected, abs=1e-9), (
                        f"drone {drone.drone_id} moved {moved}, expected "
                        f"{expected}"
                    )
                previous[drone.drone_id] = drone.position.copy()
        env.close_logger()

    assert all(n <= 1 for n in short_steps.values()), (
        "a drone took more than one short step, so it is not snapping to B"
    )


def test_a_drone_never_leaves_the_area_and_stops_at_b() -> None:
    """B is inside the arena, and the drone ends exactly on it."""
    with moving(max_start_delay_s=0.0):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=9, run_seed=9)
        env.reset()
        targets = {d.drone_id: d.end_point.copy() for d in _m_drones(env)}
        for _ in range(config.MAX_STEPS):
            env.step()
            for drone in env.drones:
                assert 0.0 <= drone.position[0] <= config.WIDTH
                assert 0.0 <= drone.position[1] <= config.HEIGHT
        env.close_logger()

    for drone in _m_drones(env):
        assert drone.arrived, f"drone {drone.drone_id} never reached B"
        assert np.array_equal(drone.position, targets[drone.drone_id]), (
            f"drone {drone.drone_id} stopped near B, not on it"
        )
        # And B itself is inside the border margin.
        for axis, extent in ((0, config.WIDTH), (1, config.HEIGHT)):
            assert -1e-9 <= targets[drone.drone_id][axis] <= extent + 1e-9


def test_a_drone_waits_out_its_start_delay() -> None:
    """Nobody moves before their delay, and everybody moves after it."""
    with moving(max_start_delay_s=20.0):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=11, run_seed=11)
        env.reset()
        delays = {d.drone_id: d.start_delay_s for d in _m_drones(env)}
        start = {d.drone_id: d.position.copy() for d in _m_drones(env)}
        assert max(delays.values()) > 0.0, "no delay was drawn at all"

        moved_at = {d.drone_id: None for d in _m_drones(env)}
        for step in range(400):
            env.step()
            now = step * config.TIMESTEP
            for drone in _m_drones(env):
                if moved_at[drone.drone_id] is not None:
                    continue
                if not np.array_equal(drone.position, start[drone.drone_id]):
                    moved_at[drone.drone_id] = now
        env.close_logger()

    for drone_id, first in moved_at.items():
        if first is None:
            continue
        assert first >= delays[drone_id] - 1e-9, (
            f"drone {drone_id} moved at {first}s, before its "
            f"{delays[drone_id]}s delay"
        )
        # It set off on the first step after the delay, not much later.
        assert first <= delays[drone_id] + config.TIMESTEP + 1e-9


def test_speeds_and_delays_are_inside_their_ranges() -> None:
    """Every drawn speed and delay respects the configured bounds."""
    with moving():
        for seed in (1, 2, 3):
            env = _run("greedy", seed, steps=0)
            for drone in _m_drones(env):
                assert config.DRONE_SPEED_MIN <= drone.speed <= config.DRONE_SPEED_MAX
                assert 0.0 <= drone.start_delay_s <= config.MAX_START_DELAY_S


def test_zero_max_delay_starts_everyone_together() -> None:
    """MAX_START_DELAY_S = 0 means no drone waits."""
    with moving(max_start_delay_s=0.0):
        env = _run("greedy", 12, steps=0)
        assert all(d.start_delay_s == 0.0 for d in _m_drones(env))


def test_the_trajectories_do_not_depend_on_the_routing_rule() -> None:
    """greedy and random fly the same layout identically.

    The mobility stream is keyed on the placement seed, so the flight is a
    property of the layout, not of what the packets are doing.
    """
    with moving():
        greedy = _run("greedy", 14, steps=300)
        random_ = _run("random", 14, steps=300)

    for a, b in zip(_m_drones(greedy), _m_drones(random_)):
        assert a.drone_id == b.drone_id
        assert np.array_equal(a.start_point, b.start_point)
        assert np.array_equal(a.end_point, b.end_point)
        assert a.speed == b.speed
        assert a.start_delay_s == b.start_delay_s
        assert np.array_equal(a.position, b.position), (
            f"drone {a.drone_id} is in a different place under the two rules"
        )


def test_c_drones_hover() -> None:
    """C-drones do not move in step 1, however long the run."""
    with moving():
        env = _run("greedy", 15, steps=config.MAX_STEPS)
    for drone in env.drones:
        if drone.drone_type == "C":
            assert np.array_equal(drone.position, drone.start_point)


# ---------------------------------------------------------------------------
# Broken links
# ---------------------------------------------------------------------------

def test_packets_add_up_while_the_network_moves() -> None:
    """created = delivered + dropped + still in flight, with movement on.

    The broken-link rescue moves packets between queues, which is exactly the
    kind of change that loses or duplicates one.
    """
    with moving(), load_ms(200):
        env = _run("greedy", 5)

    created = len(env.all_packets)
    delivered = sum(1 for p in env.all_packets if p.delivered)
    dropped = sum(1 for p in env.all_packets if p.dropped)
    in_flight = sum(1 for p in env.all_packets
                    if not p.delivered and not p.dropped)
    assert created == delivered + dropped + in_flight
    assert created > 0 and delivered > 0

    # And every dropped packet carries exactly one reason.
    assert all(p.drop_reason is not None
               for p in env.all_packets if p.dropped)


def test_a_broken_link_leaves_no_queue_behind() -> None:
    """After a step, no drone holds a queue for a hop it cannot reach."""
    with moving(), load_ms(200):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=5, run_seed=5)
        env.reset()
        for _ in range(600):
            env.step()
            for drone in env.drones:
                for key, queue in drone.link_queues.items():
                    if not queue:
                        continue
                    if key == "GS":
                        assert are_connected(drone.position, env.gs_position), (
                            f"drone {drone.drone_id} still queues for the GS "
                            f"while out of range"
                        )
                    else:
                        assert key in drone.neighbors, (
                            f"drone {drone.drone_id} still queues for {key}, "
                            f"which is no longer a neighbour"
                        )
        env.close_logger()


def test_the_moving_world_actually_breaks_links() -> None:
    """The rescue is exercised: some seed re-routes and drops link_broken.

    Without this, every other broken-link check could pass by never running.
    """
    with moving(), load_ms(200):
        env = _run("greedy", 5)
    assert env.reroutes > 0, "no link broke in a whole moving run"
    assert env.link_broken_drops > 0
    assert any(p.drop_reason is DropReason.LINK_BROKEN
               for p in env.all_packets if p.dropped)


def test_the_two_views_still_agree_while_moving() -> None:
    """ACK-derived counters match ground truth with links breaking."""
    from scripts.metrics_1a import compute_metrics

    with moving(), load_ms(200):
        for seed in (2, 5):
            env = _run("greedy", seed)
            metrics = compute_metrics(env)
            assert not metrics["discrepancies"], metrics["discrepancies"]


def test_nothing_crashes_at_the_top_speed() -> None:
    """200 runs at 30 m/s, the fastest the fleet can fly."""
    previous = (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX)
    with moving():
        config.DRONE_SPEED_MIN = config.DRONE_SPEED_MAX = 30.0
        try:
            for seed in range(1, 201):
                env = FANETEnv(routing="greedy", log_path=os.devnull,
                               placement_seed=seed, run_seed=seed)
                env.reset()
                for _ in range(60):
                    env.step()
                env.close_logger()
        finally:
            config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX = previous


# ---------------------------------------------------------------------------
# The new measures
# ---------------------------------------------------------------------------

def test_time_connected_is_a_share_and_is_one_when_static() -> None:
    """Every M-drone is connected the whole time in the static world.

    The Phase-1 layout filter guarantees it, so this is also a check that the
    measure is reading the right network.
    """
    from scripts.metrics_1a import compute_metrics

    with load_ms(200):
        env = _run("greedy", 7)
    truth = compute_metrics(env)["ground_truth"]
    assert truth["pct_time_connected"] == pytest.approx(1.0, abs=1e-12)
    assert env.measured_steps == config.MAX_STEPS - config.WARMUP_STEPS - config.DRAIN_STEPS


def test_loss_while_connected_equals_total_loss_when_always_connected() -> None:
    """With nobody ever cut off, the two loss figures are the same number."""
    from scripts.metrics_1a import compute_metrics

    with load_ms(200):
        env = _run("greedy", 7)
    truth = compute_metrics(env)["ground_truth"]
    assert truth["created_while_connected"] == truth["created"]
    assert truth["loss_while_connected"] == pytest.approx(
        truth["total_loss"], abs=1e-12
    )


def test_loss_while_connected_ignores_packets_born_cut_off() -> None:
    """A packet created with no path to the GS is left out of that figure."""
    from scripts.metrics_1a import compute_metrics

    with moving(), load_ms(200):
        env = _run("greedy", 5)
    truth = compute_metrics(env)["ground_truth"]

    assert truth["created_while_connected"] <= truth["created"]
    assert 0.0 <= (truth["pct_time_connected"] or 0.0) <= 1.0
    # Whatever was set aside, what is left is still a rate.
    assert 0.0 <= (truth["loss_while_connected"] or 0.0) <= 1.0
