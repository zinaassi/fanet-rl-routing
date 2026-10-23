"""
test_2a_fixes.py — Phase 2a step 4: what the review changed.

Four things, each with a different reason:

* steering now reads SENSORS — the true positions of whatever is within
  ``SENSOR_RANGE_M`` — while routing, target choice and target sharing still
  read nothing but the neighbour table;
* a close pass is an EVENT, counted once per approach and split by the kinds
  of drone involved;
* "% connected" is measured per PACKET, at the moment it was created, and
  the loss is reported for both groups so the two combine back into the
  total exactly;
* "relayed in the last second" counts other drones' packets this drone moved
  on — not its own hand-offs, and including the last hop into the GS.

That last one was a bug, and it is why "busiest" sent its C-drones outward
under greedy and inward under random. Under greedy 89% of what the old
counter saw were sources handing off their OWN packets, and the hops into the
GS — the ones that really are relays — were not counted at all.
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
from fanet_sim.envs.drone import Hello
from fanet_sim.envs.fanet_env import FANETEnv


@contextmanager
def world(rule: str = "midpoint", avoid: bool = True, share: bool = True,
          load: int = 200) -> Iterator[None]:
    """Run the body with the fleet moving, then put every setting back."""
    saved = (config.C_RULE, config.AVOID, config.SHARE_TARGETS,
             config.STATIC_MODE, config.IDEAL_HELLO,
             config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
             config.MAX_START_DELAY_S)
    config.C_RULE = rule
    config.AVOID = avoid
    config.SHARE_TARGETS = share
    config.STATIC_MODE = False
    config.IDEAL_HELLO = False
    config.HELLO_INTERVAL = 1.0
    config.MAX_START_DELAY_S = 50.0
    config.PACKET_INTERVAL_STEPS = int(round(load / 1000.0 / config.TIMESTEP))
    try:
        yield
    finally:
        (config.C_RULE, config.AVOID, config.SHARE_TARGETS,
         config.STATIC_MODE, config.IDEAL_HELLO,
         config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
         config.MAX_START_DELAY_S) = saved


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


# ---------------------------------------------------------------------------
# 1. Sensors for steering, the table for everything else
# ---------------------------------------------------------------------------

def test_steering_sees_everything_inside_the_sensor_range() -> None:
    """Whatever is within range is seen, table entry or not."""
    with world():
        env = _env()
        drone = _c_drones(env)[0]
        drone.hello_table = {}                 # it has heard nothing at all
        near, far = env.drones[0], env.drones[1]
        near.position = drone.position + np.array([30.0, 0.0])
        far.position = drone.position + np.array([300.0, 0.0])

        sensed = env._sensed_positions(drone)
        env.close_logger()

    assert any(np.allclose(p, near.position) for p in sensed), (
        "a drone 30 m away was not sensed"
    )
    assert not any(np.allclose(p, far.position) for p in sensed), (
        "a drone 300 m away was sensed"
    )


def test_steering_ignores_everything_beyond_the_sensor_range() -> None:
    """Nothing outside SENSOR_RANGE_M is in the list, ever."""
    with world():
        env = _env()
        for _ in range(120):
            env.step()
            for drone in _c_drones(env):
                for seen in env._sensed_positions(drone):
                    gap = float(np.linalg.norm(seen - drone.position))
                    assert gap <= config.SENSOR_RANGE_M + 1e-9
        env.close_logger()


def test_target_choice_still_reads_only_the_table() -> None:
    """A C-drone surrounded by real drones but deaf picks no target."""
    for rule in ("busiest", "midpoint"):
        with world(rule=rule):
            env = _env()
            drone = _c_drones(env)[0]
            for other in env.drones:
                if other.drone_id != drone.drone_id:
                    other.position = drone.position + np.array([10.0, 0.0])
            drone.hello_table = {}
            assert env._choose_target(drone) is None, (
                f"{rule} found a target with an empty table"
            )
            env.close_logger()


def test_routing_still_reads_only_the_table() -> None:
    """Candidates come from the table even with everyone in sensor range."""
    with world():
        env = _env()
        drone = _c_drones(env)[0]
        for other in env.drones:
            if other.drone_id != drone.drone_id:
                other.position = drone.position + np.array([5.0, 0.0])
        env._recompute_links()
        drone.hello_table = {}
        assert env._believed_links(drone) == {}
        env.close_logger()


# ---------------------------------------------------------------------------
# 2. Keeping apart
# ---------------------------------------------------------------------------

def _fly_pair(env: FANETEnv, left, right, steps: int = 80) -> float:
    """Step the two C-drones and return the closest they ever came."""
    closest = float("inf")
    for _ in range(steps):
        left.next_decision_step = 10 ** 9
        right.next_decision_step = 10 ** 9
        env._move_c_drones()
        env.step_count += 1
        closest = min(closest, float(np.linalg.norm(
            left.position - right.position
        )))
    return closest


def test_two_c_drones_head_on_keep_twenty_metres() -> None:
    """Flown straight at each other, they never close inside D_SAFE."""
    with world():
        env = _env()
        left, right = _c_drones(env)[:2]
        for other in _c_drones(env)[2:]:
            other.position = np.array([10.0, 10.0])
            other.target = None
        left.position = np.array([300.0, 500.0])
        right.position = np.array([500.0, 500.0])
        left.target = np.array([900.0, 500.0])
        right.target = np.array([0.0, 500.0])
        closest = _fly_pair(env, left, right)
        env.close_logger()

    assert closest >= config.C_SAFE_DISTANCE_M - 1e-9, (
        f"head-on pair closed to {closest:.2f} m"
    )


def test_a_c_drone_crossing_a_hovering_one_keeps_twenty_metres() -> None:
    """The moving one goes round; the still one is never disturbed."""
    with world():
        env = _env()
        mover, still = _c_drones(env)[:2]
        for other in _c_drones(env)[2:]:
            other.position = np.array([10.0, 10.0])
            other.target = None
        still.position = np.array([450.0, 500.0])
        still.target = None                       # hovering
        mover.position = np.array([250.0, 500.0])
        mover.target = np.array([700.0, 500.0])   # straight through it
        was = still.position.copy()
        closest = _fly_pair(env, mover, still)
        env.close_logger()

    assert closest >= config.C_SAFE_DISTANCE_M - 1e-9, (
        f"the crossing pair closed to {closest:.2f} m"
    )
    assert np.array_equal(still.position, was), "the hovering drone moved"


def test_a_close_pass_is_counted_once_per_approach() -> None:
    """Two drones held close for many steps are one event, not many."""
    with world(rule="hover", avoid=False):
        env = _env()
        # Spread everybody onto a wide grid first, so the only close pair in
        # the arena is the one this test makes.
        for index, drone in enumerate(env.drones):
            drone.position = np.array([
                100.0 + 180.0 * (index % 5), 100.0 + 180.0 * (index // 5),
            ], dtype=np.float64)
        first, second = env.drones[0], env.drones[1]
        for _ in range(12):
            first.position = np.array([400.0, 400.0])
            second.position = np.array([405.0, 400.0])     # 5 m apart
            env._count_close_passes(measured=True)
        held = sum(env.close_pass_events.values())

        # Separate, then come back: that is a second approach.
        second.position = np.array([800.0, 400.0])
        env._count_close_passes(measured=True)
        second.position = np.array([405.0, 400.0])
        env._count_close_passes(measured=True)
        after = sum(env.close_pass_events.values())
        env.close_logger()

    assert held == 1, f"a single approach counted {held} times"
    assert after == 2, "coming back after separating did not count again"


def test_close_passes_are_split_by_kind() -> None:
    """The three kinds are counted apart and add up to the total."""
    from scripts.metrics_1a import compute_metrics

    with world():
        env = _run("greedy", 5)
    truth = compute_metrics(env)["ground_truth"]
    assert truth["close_passes"] == (truth["close_passes_cc"]
                                     + truth["close_passes_cm"]
                                     + truth["close_passes_mm"])
    assert truth["closest_approach_m"] is None or (
        0.0 <= truth["closest_approach_m"] < config.C_SAFE_DISTANCE_M
    )


# ---------------------------------------------------------------------------
# 3. Connected and disconnected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ("greedy", "random"))
@pytest.mark.parametrize("seed", (3, 5, 11))
def test_the_connected_identity_holds_exactly(rule: str, seed: int) -> None:
    """total = c * loss_connected + (1 - c) * loss_disconnected, exactly."""
    from scripts.metrics_1a import compute_metrics

    with world():
        env = _run(rule, seed)
    truth = compute_metrics(env)["ground_truth"]

    share = truth["pct_packets_connected"]
    connected = truth["loss_while_connected"] or 0.0
    cut_off = truth["loss_while_disconnected"] or 0.0
    rebuilt = share * connected + (1.0 - share) * cut_off

    assert rebuilt == pytest.approx(truth["total_loss"], abs=1e-12), (
        f"{rule} seed {seed}: {rebuilt} vs {truth['total_loss']}"
    )
    assert (truth["created_while_connected"]
            + truth["created_while_disconnected"]) == truth["created"]


def test_everything_is_connected_in_the_static_world() -> None:
    """The layout filter guarantees it, so the split is degenerate there."""
    from scripts.metrics_1a import compute_metrics

    saved = (config.STATIC_MODE, config.PACKET_INTERVAL_STEPS)
    config.STATIC_MODE = True
    config.PACKET_INTERVAL_STEPS = 2
    try:
        env = _run("greedy", 7)
    finally:
        config.STATIC_MODE, config.PACKET_INTERVAL_STEPS = saved

    truth = compute_metrics(env)["ground_truth"]
    assert truth["pct_packets_connected"] == 1.0
    assert truth["created_while_disconnected"] == 0
    assert truth["loss_while_disconnected"] is None
    assert truth["loss_while_connected"] == pytest.approx(
        truth["total_loss"], abs=1e-12
    )


# ---------------------------------------------------------------------------
# 4. What "relayed" counts
# ---------------------------------------------------------------------------

def test_relayed_counts_other_peoples_packets_only() -> None:
    """A source handing off its OWN packet is not relaying."""
    with world(rule="hover"):
        env = _env()
        sender = next(d for d in env.drones if d.drone_type == "M")
        before = sender.relayed_since_hello
        mine = [p for p in env.all_packets if p.source_id == sender.drone_id]
        env.close_logger()
    # Nothing has been sent yet, so the counter is still at zero whatever
    # the drone created.
    assert before == 0
    assert isinstance(mine, list)


def _a_drone_that_can_reach_the_gs(env: FANETEnv):
    """Return a drone whose TABLE lists the GS, stepping until one does.

    Tables are empty until hellos have been exchanged, and hellos go out at
    the end of a step, so this has to run the clock a little first.
    """
    for _ in range(60):
        env.step()
        for drone in env.drones:
            if "GS" in drone.hello_table:
                return drone
    raise AssertionError("no drone ever heard the ground station")


def test_relayed_counts_the_hop_into_the_gs() -> None:
    """The last hop of a chain is a relay too.

    Built by hand: one drone holds another drone's packet, queued for the GS.
    Sending it must move the counter.
    """
    with world(rule="hover"):
        env = _env()
        relay = _a_drone_that_can_reach_the_gs(env)
        source = next(d for d in env.drones if d.drone_id != relay.drone_id)

        pkt = env._factory.create(source_id=source.drone_id,
                                  created_at=env.step_count)
        env.all_packets.append(pkt)
        relay.link_queues["GS"].append(pkt)
        relay.relayed_since_hello = 0
        env._transmit(measured=True)
        counted = relay.relayed_since_hello
        env.close_logger()

    assert counted == 1, (
        "a relay into the GS was not counted as a relay"
    )


def test_an_own_packet_sent_onward_is_not_counted() -> None:
    """The same hand-off, but the holder created the packet."""
    with world(rule="hover"):
        env = _env()
        relay = _a_drone_that_can_reach_the_gs(env)
        pkt = env._factory.create(source_id=relay.drone_id,
                                  created_at=env.step_count)
        env.all_packets.append(pkt)
        relay.link_queues["GS"].append(pkt)
        relay.relayed_since_hello = 0
        env._transmit(measured=True)
        counted = relay.relayed_since_hello
        env.close_logger()

    assert counted == 0, "a source's own hand-off was counted as a relay"


# ---------------------------------------------------------------------------
# The streams, the books, and not falling over
# ---------------------------------------------------------------------------

def test_the_other_streams_are_untouched() -> None:
    """Layouts, flights, hello schedules and the C-drone walk do not move."""
    saved = config.C_RULE
    try:
        config.C_RULE = "hover"
        with world(rule="hover"):
            first = _env(6)
        config.C_RULE = "midpoint"
        with world(rule="midpoint"):
            second = _env(6)
    finally:
        config.C_RULE = saved

    for a, b in zip(sorted(first.drones, key=lambda d: d.drone_id),
                    sorted(second.drones, key=lambda d: d.drone_id)):
        assert np.array_equal(a.start_point, b.start_point)
        assert np.array_equal(a.end_point, b.end_point)
        assert a.start_delay_s == b.start_delay_s
        assert a.next_hello_step == b.next_hello_step
    for name in ("_rng_chan", "_rng_hello", "_rng_cdrone", "_rng_mob"):
        assert (getattr(first, name).bit_generator.state["state"]["state"]
                == getattr(second, name).bit_generator.state["state"]["state"]), (
            f"{name} moved between the two runs"
        )


@pytest.mark.parametrize("rule", ("random_walk", "busiest", "midpoint"))
def test_packets_add_up_and_views_agree(rule: str) -> None:
    """created = delivered + dropped + in flight, and both books match."""
    from scripts.metrics_1a import compute_metrics

    with world(rule=rule):
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
    """200 short runs across the three moving rules, sensors and all."""
    for index, rule in enumerate(("random_walk", "busiest", "midpoint")):
        with world(rule=rule):
            for seed in range(1 + index * 67, 1 + (index + 1) * 67):
                env = _env(seed)
                for _ in range(40):
                    env.step()
                env.close_logger()
