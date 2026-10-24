"""
test_2c_interface.py — Phase 2c M0: the C-drone agent's interface.

The interface has to be a faithful re-expression of what Phase 2a already
did, or nothing trained through it can be compared with anything measured
before it. Three groups of checks:

* the old rules, put through the new interface, give the old runs back;
* the observation carries only what the drone could know — never another
  drone's true position;
* the action and the reward mean what they say.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs import c_interface
from fanet_sim.envs.c_scripted import HoverAgent, MidpointAgent, ReplayAgent
from fanet_sim.envs.fanet_env import FANETEnv


@contextmanager
def m0_world(c_rule: str = "rl", load: int = 200) -> Iterator[None]:
    """The world M0-M3 use: static mission, C-drones flying, hellos on."""
    saved = (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
             config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
             config.SHARE_TARGETS, config.AVOID)
    config.C_RULE = c_rule
    config.STATIC_MODE = True
    config.IDEAL_HELLO = False
    config.HELLO_INTERVAL = 1.0
    config.SHARE_TARGETS = True
    config.AVOID = True
    config.PACKET_INTERVAL_STEPS = int(round(load / 1000.0 / config.TIMESTEP))
    try:
        yield
    finally:
        (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
         config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
         config.SHARE_TARGETS, config.AVOID) = saved


def _run(seed: int, agent_factory=None, steps: int = config.MAX_STEPS):
    """Run one episode and return the finished environment."""
    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    if agent_factory is not None:
        env.c_agent = agent_factory(env)
    for _ in range(steps):
        env.step()
    env.close_logger()
    return env


def _loss(env) -> float:
    """Return one run's total loss."""
    from scripts.metrics_1a import compute_metrics
    return compute_metrics(env)["ground_truth"]["total_loss"] or 0.0


def _c_drones(env) -> List:
    """Return the C-drones, in id order."""
    return [d for d in sorted(env.drones, key=lambda d: d.drone_id)
            if d.drone_type == "C"]


# ---------------------------------------------------------------------------
# The old rules through the new interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", (301, 305))
def test_hover_through_the_interface_is_hover(seed: int) -> None:
    """A waypoint of "where I already am" reproduces hovering exactly."""
    with m0_world(c_rule="hover"):
        native = _loss(_run(seed))
    with m0_world(c_rule="rl"):
        through = _run(seed, HoverAgent)
    assert _loss(through) == native
    assert all(d.distance_flown == 0.0 for d in _c_drones(through))


@pytest.mark.parametrize("seed", (301, 305, 310))
def test_midpoint_through_the_interface_is_midpoint(seed: int) -> None:
    """The Phase-2a rule, expressed as actions, gives the same run back.

    Bit-for-bit, including how far each C-drone flew. Encoding a waypoint as
    a normalised action and decoding it loses the last bit of about a
    quarter of all points, some 6e-14 m, and this says that never grows into
    anything the simulator notices.
    """
    with m0_world(c_rule="midpoint"):
        native_env = _run(seed)
        native = _loss(native_env)
    with m0_world(c_rule="rl"):
        through_env = _run(seed, MidpointAgent)

    assert _loss(through_env) == native
    for a, b in zip(_c_drones(native_env), _c_drones(through_env)):
        assert a.distance_flown == b.distance_flown
        assert np.array_equal(a.position, b.position)


def test_the_interface_refuses_to_run_without_an_agent() -> None:
    """C_RULE = "rl" with nothing injected is an error, not a silent hover."""
    with m0_world(c_rule="rl"):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=301, run_seed=301)
        env.reset()
        with pytest.raises(RuntimeError, match="no c_agent"):
            env.step()
        env.close_logger()


# ---------------------------------------------------------------------------
# The observation leaks nothing
# ---------------------------------------------------------------------------

def test_no_true_position_reaches_the_observation() -> None:
    """Teleport every other drone; the observation must not move.

    The strongest form of the check: the drones that are NOT in the table
    are moved to the far corner, and the ones that are keep their table
    entries. Anything reading a live position would change.
    """
    with m0_world(c_rule="rl"):
        env = _run(301, MidpointAgent, steps=120)
        drone = _c_drones(env)[0]
        before = c_interface.build_observation(env, drone)

        for other in env.drones:
            if other.drone_id == drone.drone_id:
                continue
            other.position = np.array([config.WIDTH, config.HEIGHT],
                                      dtype=np.float64)
            other.velocity = np.array([7.0, -7.0], dtype=np.float64)
            other.end_point = np.zeros(2, dtype=np.float64)

        after = c_interface.build_observation(env, drone)
        env.close_logger()

    assert np.array_equal(before["self"], after["self"])
    assert np.array_equal(before["neighbours"], after["neighbours"])
    assert np.array_equal(before["mask"], after["mask"])


def test_the_observation_follows_the_table() -> None:
    """Dropping a table entry drops a row; nothing else is consulted."""
    with m0_world(c_rule="rl"):
        env = _run(301, MidpointAgent, steps=120)
        drone = _c_drones(env)[0]
        if not drone.hello_table:
            pytest.skip("that drone heard nobody")
        before = c_interface.build_observation(env, drone)
        drone.hello_table.pop(sorted(drone.hello_table, key=str)[0])
        after = c_interface.build_observation(env, drone)
        env.close_logger()

    assert len(after["mask"]) == len(before["mask"]) - 1


def test_the_observation_has_the_shape_it_claims() -> None:
    """Widths match the constants, and nothing is NaN or infinite."""
    with m0_world(c_rule="rl"):
        env = _run(302, MidpointAgent, steps=200)
        for drone in _c_drones(env):
            observation = c_interface.build_observation(env, drone)
            assert observation["self"].shape == (c_interface.SELF_FEATURES,)
            assert observation["neighbours"].shape == (
                len(drone.hello_table), c_interface.NEIGHBOUR_FEATURES
            )
            assert observation["mask"].shape == (len(drone.hello_table),)
            assert np.isfinite(observation["self"]).all()
            assert np.isfinite(observation["neighbours"]).all()
        env.close_logger()


def test_the_observation_does_not_depend_on_table_order() -> None:
    """Rebuilding the table in another order gives the same rows back."""
    with m0_world(c_rule="rl"):
        env = _run(303, MidpointAgent, steps=150)
        drone = next((d for d in _c_drones(env) if len(d.hello_table) > 1),
                     None)
        if drone is None:
            pytest.skip("no C-drone heard more than one neighbour")
        before = c_interface.build_observation(env, drone)
        drone.hello_table = dict(reversed(list(drone.hello_table.items())))
        after = c_interface.build_observation(env, drone)
        env.close_logger()
    assert np.array_equal(before["neighbours"], after["neighbours"])


def test_padding_keeps_the_rows_and_marks_the_gaps() -> None:
    """A batch of different-sized tables stacks without losing anything."""
    small = {"self": np.arange(c_interface.SELF_FEATURES, dtype=np.float64),
             "neighbours": np.ones((1, c_interface.NEIGHBOUR_FEATURES)),
             "mask": np.ones(1)}
    big = {"self": np.zeros(c_interface.SELF_FEATURES),
           "neighbours": np.full((3, c_interface.NEIGHBOUR_FEATURES), 2.0),
           "mask": np.ones(3)}
    batch = c_interface.pad_observations([small, big])

    assert batch["neighbours"].shape == (2, 3, c_interface.NEIGHBOUR_FEATURES)
    assert batch["mask"].tolist() == [[1, 0, 0], [1, 1, 1]]
    assert np.array_equal(batch["neighbours"][0, 0],
                          np.ones(c_interface.NEIGHBOUR_FEATURES))
    assert np.array_equal(batch["neighbours"][0, 1],
                          np.zeros(c_interface.NEIGHBOUR_FEATURES))


# ---------------------------------------------------------------------------
# The action and the reward
# ---------------------------------------------------------------------------

def test_the_action_names_a_point_in_the_arena() -> None:
    """Decoding is GS-relative, scaled by 450 m, and clipped to the arena."""
    gs = np.array([450.0, 450.0])
    assert np.array_equal(
        c_interface.waypoint_from_action(np.zeros(2), gs), gs
    )
    assert np.array_equal(
        c_interface.waypoint_from_action(np.array([1.0, -1.0]), gs),
        np.array([900.0, 0.0])
    )
    # Outside [-1, 1] is clipped by the arena, not rescaled.
    assert np.array_equal(
        c_interface.waypoint_from_action(np.array([5.0, 5.0]), gs),
        np.array([config.WIDTH, config.HEIGHT])
    )


def test_the_drone_flies_to_the_waypoint_it_was_given() -> None:
    """A fixed action sends the drone there, and it stops within 10 m.

    Steering is off here. With it on, seven drones sent to the SAME point
    cannot all reach it — the 23 m clearance holds them in a ring around it,
    which is the steering doing its job, not the flight failing. That
    interaction is checked in test_crowding_is_the_steering_not_the_flight.
    """
    gs = np.array([450.0, 450.0])
    corner = np.array([0.6, 0.6])
    with m0_world(c_rule="rl"):
        config.AVOID = False
        env = _run(304, lambda e: ReplayAgent([corner]), steps=600)
        target = c_interface.waypoint_from_action(corner, gs)
        for drone in _c_drones(env):
            gap = float(np.linalg.norm(drone.position - target))
            assert gap <= config.C_MIDPOINT_STOP_M + 1e-6, (
                f"C-drone {drone.drone_id} stopped {gap:.1f} m from its "
                f"waypoint"
            )
        env.close_logger()


def test_crowding_is_the_steering_not_the_flight() -> None:
    """Sent to one point with steering on, they ring it instead of stacking.

    Worth pinning because a policy that gives every C-drone the same
    waypoint will meet this, and the result should be a spread-out ring, not
    seven drones on one spot or seven drones stuck far away.
    """
    gs = np.array([450.0, 450.0])
    corner = np.array([0.6, 0.6])
    with m0_world(c_rule="rl"):
        config.AVOID = True
        env = _run(304, lambda e: ReplayAgent([corner]), steps=600)
        target = c_interface.waypoint_from_action(corner, gs)
        drones = _c_drones(env)
        gaps = [float(np.linalg.norm(d.position - target)) for d in drones]
        closest = min(
            float(np.linalg.norm(a.position - b.position))
            for i, a in enumerate(drones) for b in drones[i + 1:]
        )
        env.close_logger()

    assert min(gaps) <= config.C_MIDPOINT_STOP_M + 1e-6, (
        "not even the first drone reached the waypoint"
    )
    assert max(gaps) < 150.0, f"someone was left {max(gaps):.0f} m away"
    assert closest >= config.C_SAFE_DISTANCE_M - 1e-9, (
        f"they stacked up to {closest:.1f} m apart"
    )


def test_decisions_come_once_a_second_for_every_c_drone() -> None:
    """One row per C-drone per decision interval, no more and no fewer."""
    with m0_world(c_rule="rl"):
        env = _run(305, MidpointAgent)
        c_drones = _c_drones(env)
        per_decision = config.MAX_STEPS // c_interface.steps_per_decision()
        assert len(env.c_decisions) == per_decision * len(c_drones)
        steps = {row["step"] for row in env.c_decisions}
        assert all(step % c_interface.steps_per_decision() == 0
                   for step in steps)
        env.close_logger()


def test_the_reward_is_the_balance_minus_the_flying() -> None:
    """The formula, checked directly."""
    saved = config.C_FLIGHT_COST
    config.C_FLIGHT_COST = 0.01
    try:
        # 60 delivered, 10 lost, 100 created, flew 30 m at 30 m/s.
        assert c_interface.flight_reward(60, 10, 100, 30.0, 30.0) == pytest.approx(
            0.5 - 0.01, abs=1e-12
        )
        # Nothing created: no delivery term, the flight still costs.
        assert c_interface.flight_reward(0, 0, 0, 15.0, 30.0) == pytest.approx(
            -0.005, abs=1e-12
        )
        # Standing still costs nothing.
        assert c_interface.flight_reward(10, 0, 10, 0.0, 30.0) == 1.0
    finally:
        config.C_FLIGHT_COST = saved


def test_every_decision_but_the_last_gets_a_reward() -> None:
    """Rewards are filled in a second later, so only the last is open."""
    with m0_world(c_rule="rl"):
        env = _run(306, MidpointAgent)
        c_drones = _c_drones(env)
        rewarded = [r for r in env.c_decisions if r["reward"] is not None]
        assert len(rewarded) == len(env.c_decisions) - len(c_drones)
        assert all(np.isfinite(r["reward"]) for r in rewarded)
        env.close_logger()


def test_the_books_still_balance_through_the_interface() -> None:
    """Packets add up and the two views agree with an agent driving."""
    from scripts.metrics_1a import compute_metrics

    with m0_world(c_rule="rl"):
        env = _run(307, MidpointAgent)
    metrics = compute_metrics(env)
    assert not metrics["discrepancies"], metrics["discrepancies"]

    created = len(env.all_packets)
    delivered = sum(1 for p in env.all_packets if p.delivered)
    dropped = sum(1 for p in env.all_packets if p.dropped)
    in_flight = sum(1 for p in env.all_packets
                    if not p.delivered and not p.dropped)
    assert created == delivered + dropped + in_flight
    assert created > 0
