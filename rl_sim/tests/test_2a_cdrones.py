"""
test_2a_cdrones.py — The C-drones start moving.

Phase 2a step 3. C-drones carry no traffic of their own, so where they sit is
the only thing they contribute. Four rules decide that: hover, a random walk,
flying toward the busiest neighbour, and parking at the midpoint of the
longest gap to the GS. On top of those sit two options — announcing targets so
they do not pile up, and steering around each other.

Every rule decides from the drone's OWN neighbour table. Several tests below
build a table by hand, which is both the only way to know what the right
answer is and a check that nothing else is being read.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.drone import Hello
from fanet_sim.envs.fanet_env import FANETEnv


@contextmanager
def world(rule: str = "hover", avoid: bool = False, share: bool = False,
          spread: float = 150.0, turn: float = 5.0,
          static: bool = False, ideal: bool = False,
          load: int = 200) -> Iterator[None]:
    """Run the body in one configuration, then put every setting back."""
    saved = (config.C_RULE, config.AVOID, config.SHARE_TARGETS,
             config.C_TARGET_SPREAD_M, config.C_TURN_INTERVAL,
             config.STATIC_MODE, config.IDEAL_HELLO,
             config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL)
    config.C_RULE = rule
    config.AVOID = avoid
    config.SHARE_TARGETS = share
    config.C_TARGET_SPREAD_M = spread
    config.C_TURN_INTERVAL = turn
    config.STATIC_MODE = static
    config.IDEAL_HELLO = ideal
    config.HELLO_INTERVAL = 1.0
    config.PACKET_INTERVAL_STEPS = int(round(load / 1000.0 / config.TIMESTEP))
    try:
        yield
    finally:
        (config.C_RULE, config.AVOID, config.SHARE_TARGETS,
         config.C_TARGET_SPREAD_M, config.C_TURN_INTERVAL,
         config.STATIC_MODE, config.IDEAL_HELLO,
         config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL) = saved


def _env(seed: int = 5, rule: str = "greedy") -> FANETEnv:
    """Return a freshly reset environment."""
    env = FANETEnv(routing=rule, log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    return env


def _run(rule: str, seed: int, steps: int = config.MAX_STEPS) -> FANETEnv:
    """Run one episode and return the finished environment."""
    env = _env(seed, rule)
    for _ in range(steps):
        env.step()
    env.close_logger()
    return env


def _c_drones(env: FANETEnv) -> List:
    """Return the C-drones, in id order."""
    return [d for d in sorted(env.drones, key=lambda d: d.drone_id)
            if d.drone_type == "C"]


def _hello(sender_id, kind: str, position, *, velocity=(0.0, 0.0),
           sent_at: int = 0, fill: int = 0, relayed: int = 0,
           target: Optional[np.ndarray] = None) -> Hello:
    """Build one table entry by hand."""
    return Hello(
        sender_id=sender_id, sender_type=kind,
        position=np.array(position, dtype=np.float64),
        velocity=np.array(velocity, dtype=np.float64),
        sent_at=sent_at, queue_fill=fill, relayed=relayed, target=target,
    )


# ---------------------------------------------------------------------------
# hover is still step 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ("greedy", "random"))
def test_hover_reproduces_step_2(rule: str) -> None:
    """C_RULE = hover gives the 1 s-hello step-2 numbers back exactly."""
    from scripts.metrics_1a import compute_metrics
    from scripts.stamp import read_stamped_csv

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "out", "2a_hello_runs.csv")
    if not os.path.exists(path):
        pytest.skip("out/2a_hello_runs.csv is missing")
    rows = [r for r in read_stamped_csv(path)
            if r["rule"] == rule and int(r["placement_seed"]) == 3
            and r["ideal_hello"] == "False"
            and abs(float(r["hello_interval_s"]) - 1.0) < 1e-9]
    if not rows:
        pytest.skip("no 1 s-hello row for this rule")
    expected = rows[0]

    saved = config.MAX_START_DELAY_S
    config.MAX_START_DELAY_S = 50.0
    try:
        with world(rule="hover"):
            env = _run(rule, 3)
    finally:
        config.MAX_START_DELAY_S = saved
    truth = compute_metrics(env)["ground_truth"]

    assert (truth["total_loss"] or 0.0) == pytest.approx(
        float(expected["total_loss"]), abs=1e-12
    ), f"{rule} no longer reproduces its step-2 result"
    for cause in ("channel", "queue_full", "no_route", "dead_end",
                  "ttl", "hop_limit", "link_broken", "stale_link"):
        assert truth["loss_rate_by_reason"][cause] == pytest.approx(
            float(expected[cause]), abs=1e-12
        )
    assert truth["c_distance_flown_mean"] == 0.0


def test_hover_draws_nothing_from_the_c_drone_stream() -> None:
    """A hovering run leaves the C-drone stream untouched."""
    with world(rule="hover"):
        env = _env()
        before = env._rng_cdrone.bit_generator.state["state"]["state"]
        for _ in range(200):
            env.step()
        after = env._rng_cdrone.bit_generator.state["state"]["state"]
        env.close_logger()
    assert before == after


def test_c_movement_leaves_the_other_streams_alone() -> None:
    """Layouts, M-drone flights and hello schedules do not move."""
    with world(rule="hover"):
        still = _env(6)
    with world(rule="random_walk"):
        walking = _env(6)

    for a, b in zip(sorted(still.drones, key=lambda d: d.drone_id),
                    sorted(walking.drones, key=lambda d: d.drone_id)):
        assert np.array_equal(a.start_point, b.start_point)
        assert np.array_equal(a.end_point, b.end_point)
        assert a.start_delay_s == b.start_delay_s
        assert a.next_hello_step == b.next_hello_step
        if a.drone_type == "M":
            assert a.speed == b.speed
    assert (still._rng_chan.bit_generator.state["state"]["state"]
            == walking._rng_chan.bit_generator.state["state"]["state"])
    assert (still._rng_hello.bit_generator.state["state"]["state"]
            == walking._rng_hello.bit_generator.state["state"]["state"])


# ---------------------------------------------------------------------------
# The physical limits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ("random_walk", "busiest", "midpoint"))
def test_c_drones_keep_to_their_speed_and_stay_inside(rule: str) -> None:
    """Never more than 30 m/s, never outside the arena."""
    cap = config.C_DRONE_SPEED * config.TIMESTEP
    with world(rule=rule, avoid=True):
        env = _env(7)
        previous = {d.drone_id: d.position.copy() for d in _c_drones(env)}
        for _ in range(400):
            env.step()
            for drone in _c_drones(env):
                moved = float(np.linalg.norm(
                    drone.position - previous[drone.drone_id]
                ))
                assert moved <= cap + 1e-9, (
                    f"C-drone {drone.drone_id} moved {moved} m in a step"
                )
                assert 0.0 <= drone.position[0] <= config.WIDTH
                assert 0.0 <= drone.position[1] <= config.HEIGHT
                previous[drone.drone_id] = drone.position.copy()
        env.close_logger()


# ---------------------------------------------------------------------------
# random_walk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("turn", (5.0, 3.0))
def test_random_walk_turns_on_schedule(turn: float) -> None:
    """A new direction every C_TURN_INTERVAL, and not in between.

    Between turns the heading only ever changes by a bounce, which flips a
    component rather than replacing the direction, so a mid-interval change
    is always a sign flip.
    """
    every = int(round(turn / config.TIMESTEP))
    with world(rule="random_walk", turn=turn):
        env = _env(8)
        headings = {}
        turned_at = {d.drone_id: [] for d in _c_drones(env)}
        for _ in range(300):
            env.step()
            for drone in _c_drones(env):
                now = drone.heading.copy()
                was = headings.get(drone.drone_id)
                headings[drone.drone_id] = now
                if was is None:
                    continue
                if np.allclose(was, now):
                    continue
                # A bounce flips a component; a turn replaces the direction.
                if np.isclose(np.abs(was), np.abs(now)).all():
                    continue
                turned_at[drone.drone_id].append(env.step_count)
        env.close_logger()

    for drone_id, steps in turned_at.items():
        assert len(steps) >= 2, (
            f"C-drone {drone_id} turned {len(steps)} times in 300 steps at a "
            f"{turn}s interval"
        )
        gaps = {b - a for a, b in zip(steps, steps[1:])}
        assert gaps == {every}, (
            f"C-drone {drone_id} turned after {sorted(gaps)} steps, "
            f"expected exactly {every}"
        )


def test_random_walk_bounces_off_the_border() -> None:
    """A drone aimed at a wall turns away instead of leaving or sticking."""
    with world(rule="random_walk"):
        env = _env(9)
        drone = _c_drones(env)[0]
        drone.position = np.array([config.WIDTH - 1.0, 400.0])
        drone.heading = np.array([1.0, 0.0])          # straight at the wall
        drone.next_decision_step = 10 ** 9            # no scheduled turn
        heading = env._random_walk_heading(drone)
        env.close_logger()
    assert heading[0] < 0.0, "it did not turn away from the wall"
    assert heading[1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# busiest and midpoint, on tables built by hand
# ---------------------------------------------------------------------------

def test_busiest_picks_the_busiest_neighbour() -> None:
    """The most relayed wins; ties go to the fuller queue."""
    with world(rule="busiest"):
        env = _env()
        drone = _c_drones(env)[0]
        drone.hello_table = {
            1: _hello(1, "M", (100.0, 100.0), relayed=2, fill=9),
            2: _hello(2, "M", (200.0, 200.0), relayed=7, fill=0),
            3: _hello(3, "M", (300.0, 300.0), relayed=7, fill=5),
            4: _hello(4, "M", (400.0, 400.0), relayed=0, fill=10),
        }
        target = env._choose_target(drone)
        env.close_logger()
    # 2 and 3 tie on relayed; 3 has the fuller queue.
    assert np.array_equal(target, np.array([300.0, 300.0]))


def test_busiest_hovers_when_nobody_relayed() -> None:
    """No relaying anywhere means no target at all."""
    with world(rule="busiest"):
        env = _env()
        drone = _c_drones(env)[0]
        drone.hello_table = {
            1: _hello(1, "M", (100.0, 100.0), relayed=0, fill=4),
        }
        assert env._choose_target(drone) is None
        env.close_logger()


def test_busiest_stops_short_of_its_target() -> None:
    """Within C_BUSIEST_STOP_M it holds position rather than closing in."""
    with world(rule="busiest"):
        env = _env()
        drone = _c_drones(env)[0]
        drone.position = np.array([500.0, 500.0])
        drone.target = np.array([550.0, 500.0])        # 50 m away
        drone.next_decision_step = 10 ** 9
        assert env._target_heading(drone) is None

        drone.target = np.array([500.0 + config.C_BUSIEST_STOP_M + 50.0, 500.0])
        heading = env._target_heading(drone)
        env.close_logger()
    assert heading is not None
    assert np.allclose(heading, np.array([1.0, 0.0]))


def test_midpoint_targets_the_middle_of_the_longest_gap() -> None:
    """The farthest M-drone from the GS, halfway back to it."""
    with world(rule="midpoint"):
        env = _env()
        drone = _c_drones(env)[0]
        gs = env.gs_position
        near = gs + np.array([100.0, 0.0])
        far = gs + np.array([400.0, 0.0])
        drone.hello_table = {
            1: _hello(1, "M", near),
            2: _hello(2, "M", far),
            3: _hello(3, "C", gs + np.array([800.0, 0.0])),   # not an M-drone
        }
        target = env._choose_target(drone)
        env.close_logger()
    assert np.allclose(target, gs + np.array([200.0, 0.0]))


def test_midpoint_hovers_when_no_m_drone_is_heard() -> None:
    """Only M-drones count; a table of C-drones gives no target."""
    with world(rule="midpoint"):
        env = _env()
        drone = _c_drones(env)[0]
        drone.hello_table = {
            3: _hello(3, "C", (100.0, 100.0)),
            "GS": _hello("GS", "GS", env.gs_position),
        }
        assert env._choose_target(drone) is None
        env.close_logger()


def test_c_drones_read_only_their_table() -> None:
    """An empty table means no target, whatever is really out there."""
    for rule in ("busiest", "midpoint"):
        with world(rule=rule):
            env = _env()
            drone = _c_drones(env)[0]
            assert len(env.drones) > 1          # there IS something out there
            drone.hello_table = {}
            assert env._choose_target(drone) is None
            env.close_logger()


# ---------------------------------------------------------------------------
# Target sharing
# ---------------------------------------------------------------------------

def test_sharing_pushes_two_c_drones_onto_different_targets() -> None:
    """Told the same best target, the second one takes the next best."""
    with world(rule="midpoint", share=True, spread=150.0):
        env = _env()
        gs = env.gs_position
        far = gs + np.array([400.0, 0.0])
        second = gs + np.array([0.0, 300.0])

        first, other = _c_drones(env)[:2]
        table = {
            1: _hello(1, "M", far),
            2: _hello(2, "M", second),
        }
        first.hello_table = dict(table)
        first.target = env._choose_target(first)

        # The second one hears the first one's announcement.
        other.hello_table = dict(table)
        other.hello_table["C1"] = _hello("C1", "C", first.position,
                                         target=first.target)
        other.target = env._choose_target(other)
        env.close_logger()

    assert first.target is not None and other.target is not None
    gap = float(np.linalg.norm(first.target - other.target))
    assert gap >= config.C_TARGET_SPREAD_M, (
        f"two C-drones chose targets {gap:.1f} m apart, inside the "
        f"{config.C_TARGET_SPREAD_M} m spread"
    )


def test_sharing_leaves_nothing_when_everything_is_taken() -> None:
    """With only one candidate and it already claimed, the drone hovers."""
    with world(rule="midpoint", share=True, spread=150.0):
        env = _env()
        gs = env.gs_position
        only = gs + np.array([400.0, 0.0])
        drone = _c_drones(env)[0]
        drone.hello_table = {
            1: _hello(1, "M", only),
            "C9": _hello("C9", "C", (0.0, 0.0),
                         target=(only + gs) / 2.0),
        }
        assert env._choose_target(drone) is None
        env.close_logger()


def test_without_sharing_they_may_pick_the_same_point() -> None:
    """The control: sharing off, both take the same best target."""
    with world(rule="midpoint", share=False):
        env = _env()
        gs = env.gs_position
        far = gs + np.array([400.0, 0.0])
        first, other = _c_drones(env)[:2]
        first.hello_table = {1: _hello(1, "M", far)}
        other.hello_table = {
            1: _hello(1, "M", far),
            "C1": _hello("C1", "C", first.position,
                         target=(far + gs) / 2.0),
        }
        assert np.allclose(env._choose_target(first),
                           env._choose_target(other))
        env.close_logger()


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

def test_two_c_drones_flying_head_on_keep_their_distance() -> None:
    """With fresh, accurate hellos they never close inside D_SAFE."""
    with world(rule="midpoint", avoid=True):
        env = _env()
        left, right = _c_drones(env)[:2]
        for other in _c_drones(env)[2:]:
            other.position = np.array([10.0, 10.0])   # parked far away
            other.target = None

        left.position = np.array([300.0, 500.0])
        right.position = np.array([500.0, 500.0])
        left.target = np.array([900.0, 500.0])
        right.target = np.array([0.0, 500.0])

        closest = float("inf")
        for _ in range(60):
            # Refresh both tables with the truth, so the prediction is exact.
            left.hello_table = {right.drone_id: _hello(
                right.drone_id, "C", right.position,
                velocity=right.velocity, sent_at=env.step_count)}
            right.hello_table = {left.drone_id: _hello(
                left.drone_id, "C", left.position,
                velocity=left.velocity, sent_at=env.step_count)}
            left.next_decision_step = 10 ** 9
            right.next_decision_step = 10 ** 9
            env._move_c_drones()
            env.step_count += 1
            closest = min(closest, float(np.linalg.norm(
                left.position - right.position
            )))
        env.close_logger()

    assert closest >= config.C_SAFE_DISTANCE_M - 1e-9, (
        f"they closed to {closest:.2f} m, inside the "
        f"{config.C_SAFE_DISTANCE_M} m limit"
    )


def test_a_boxed_in_c_drone_waits() -> None:
    """Surrounded on every heading, it stays where it is and says so."""
    with world(rule="midpoint", avoid=True):
        env = _env()
        drone = _c_drones(env)[0]
        drone.position = np.array([450.0, 450.0])
        drone.target = np.array([900.0, 450.0])
        drone.next_decision_step = 10 ** 9
        # A ring of neighbours just inside the safe distance, all round.
        drone.hello_table = {
            i: _hello(i, "M",
                      drone.position + 10.0 * np.array(
                          [np.cos(a), np.sin(a)]),
                      sent_at=env.step_count)
            for i, a in enumerate(np.linspace(0, 2 * np.pi, 24, endpoint=False))
        }
        was = drone.position.copy()
        waits = drone.avoid_waits
        env._move_c_drones()
        env.close_logger()

    assert np.array_equal(drone.position, was), "it moved into the ring"
    assert drone.avoid_waits == waits + 1


# ---------------------------------------------------------------------------
# The books
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ("random_walk", "busiest", "midpoint"))
def test_packets_add_up_and_views_agree(rule: str) -> None:
    """created = delivered + dropped + in flight, and both books match."""
    from scripts.metrics_1a import compute_metrics

    with world(rule=rule, avoid=True, share=True):
        env = _run("greedy", 5)
    metrics = compute_metrics(env)
    assert not metrics["discrepancies"], metrics["discrepancies"]

    created = len(env.all_packets)
    delivered = sum(1 for p in env.all_packets if p.delivered)
    dropped = sum(1 for p in env.all_packets if p.dropped)
    in_flight = sum(1 for p in env.all_packets
                    if not p.delivered and not p.dropped)
    assert created == delivered + dropped + in_flight
    assert created > 0


def test_nothing_crashes_over_two_hundred_runs() -> None:
    """200 short runs across the three moving rules."""
    for index, rule in enumerate(("random_walk", "busiest", "midpoint")):
        with world(rule=rule, avoid=True, share=True):
            for seed in range(1 + index * 67, 1 + (index + 1) * 67):
                env = _env(seed)
                for _ in range(40):
                    env.step()
                env.close_logger()


def test_the_c_drones_actually_move_and_are_measured() -> None:
    """The rules do something, and the measures see it."""
    from scripts.metrics_1a import compute_metrics

    with world(rule="midpoint", avoid=True, share=True):
        env = _run("greedy", 5)
    truth = compute_metrics(env)["ground_truth"]
    assert truth["c_distance_flown_mean"] > 0.0
    assert len(truth["c_distance_from_gs"]) == int(
        config.MAX_STEPS * config.TIMESTEP
    )
    assert truth["close_passes"] == (truth["close_passes_cc"]
                                     + truth["close_passes_cm"]
                                     + truth["close_passes_mm"])
