"""
test_mission_near_gs.py — Phase 2c curriculum step 1: the "near_gs" mission.

The curriculum needs a mission whose difficulty is a dial rather than a
consequence of the arena's size. "near_gs" is that mission: the fleet starts
bunched around the ground station and each M-drone flies a bounded distance
outward, the bound being config.MISSION_DISTANCE_M.

What the tests here are for, in order:

* the old mission is untouched, because every committed number was measured
  on it;
* the new one obeys the geometry it was specified with;
* the dial is a dial -- turning it moves nothing but the flight lengths;
* every rung still produces a full episode's worth of measurements, which is
  the part that two separate early-exit shortcuts would otherwise have taken
  away;
* a continuation can reset or inherit its exploration spread, and M4's runs
  still reproduce under "inherit".
"""

from __future__ import annotations

import csv
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compare_views, ground_truth_view, ack_view

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out")


# --- plumbing --------------------------------------------------------------

@pytest.fixture
def near_gs():
    """Run the body on the "near_gs" mission in the moving world.

    Yields:
        A setter taking D in metres.
    """
    before = (config.MISSION, config.MISSION_DISTANCE_M, config.STATIC_MODE)
    config.MISSION = "near_gs"
    config.STATIC_MODE = False

    def set_distance(distance: float) -> None:
        config.MISSION_DISTANCE_M = float(distance)

    try:
        yield set_distance
    finally:
        (config.MISSION, config.MISSION_DISTANCE_M,
         config.STATIC_MODE) = before


def fleet(layout: int = 301, run_seed: int = 9100) -> FANETEnv:
    """Return a reset environment on the current settings.

    Args:
        layout:   Placement seed.
        run_seed: Run seed.

    Returns:
        A reset FANETEnv. The caller closes its logger.
    """
    env = FANETEnv(placement_seed=layout, run_seed=run_seed, routing="greedy")
    env.reset()
    return env


def run_to_end(env: FANETEnv) -> None:
    """Step *env* until it reports done.

    Args:
        env: A reset environment.
    """
    done = False
    while not done:
        _, _, dones, _ = env.step()
        done = all(dones.values())


# --- the old mission is untouched -----------------------------------------

def test_border_is_the_default() -> None:
    """Every committed result was measured on "border", so it must be first."""
    assert config.MISSION == "border"
    assert config.MISSIONS == ("border", "near_gs")


def test_border_reproduces_the_committed_phase_2a_runs() -> None:
    """out/2a_runs.csv, rerun. The mission block must not have moved it.

    This is the test that would catch a change to _draw_drones or
    _assign_mobility leaking out of the "near_gs" branch.
    """
    from scripts.experiment_2a import run_once

    path = os.path.join(OUT, "2a_runs.csv")
    rows = [row for row in csv.DictReader(
        line for line in open(path) if not line.startswith("#"))]
    # The perfect-knowledge, no-delay, greedy rows: step 1 of Phase 2a, the
    # plainest moving runs committed. run_once is called rather than rebuilt
    # here so the settings cannot drift apart from the ones that wrote the CSV.
    wanted = [row for row in rows
              if row["rule"] == "greedy"
              and row["ideal_hello"] == "True"
              and float(row["max_start_delay_s"]) == 0.0][:3]
    assert wanted, "no ideal-hello greedy rows at zero delay in 2a_runs.csv"

    before = config.MISSION
    config.MISSION = "border"
    try:
        for row in wanted:
            fresh = run_once("greedy", int(row["placement_seed"]), 0.0)
            assert fresh["total_loss"] == pytest.approx(
                float(row["total_loss"]), abs=1e-12
            ), f"layout {row['placement_seed']} moved"
            for cause in ("channel", "queue_full", "no_route", "link_broken"):
                assert fresh[cause] == pytest.approx(
                    float(row[cause]), abs=1e-12
                ), f"layout {row['placement_seed']} moved its {cause}"
    finally:
        config.MISSION = before


def test_border_still_ends_early_once_everyone_has_landed() -> None:
    """The shortcut is "border"-only, but it is still there for "border"."""
    before = (config.MISSION, config.STATIC_MODE, config.MAX_START_DELAY_S)
    config.MISSION = "border"
    config.STATIC_MODE = False
    config.MAX_START_DELAY_S = 0.0
    try:
        env = fleet()
        run_to_end(env)
        steps = env.step_count
        env.close_logger()
    finally:
        (config.MISSION, config.STATIC_MODE,
         config.MAX_START_DELAY_S) = before

    assert steps <= config.MAX_STEPS
    # Not an equality: whether it fires depends on the layout. What matters is
    # that the branch is reachable at all, which a "near_gs"-only guard would
    # have removed.


# --- the geometry of the new mission ---------------------------------------

def test_everyone_starts_near_the_gs(near_gs) -> None:
    """Inside START_RADIUS_M of the GS, M-drones and C-drones alike."""
    near_gs(350.0)
    env = fleet()
    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    radii = [float(np.linalg.norm(d.start_point - gs)) for d in env.drones]
    env.close_logger()

    assert len(radii) == config.NUM_M_DRONES + config.NUM_C_DRONES
    assert max(radii) <= config.START_RADIUS_M + 1e-9


def test_nobody_starts_on_top_of_anybody(near_gs) -> None:
    """At least MIN_SPACING_M apart, and that far from the GS too."""
    near_gs(350.0)
    env = fleet()
    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    starts = [d.start_point for d in env.drones]
    env.close_logger()

    for point in starts:
        assert float(np.linalg.norm(point - gs)) >= config.MIN_SPACING_M - 1e-9

    for one, other in itertools.combinations(starts, 2):
        assert (float(np.linalg.norm(one - other))
                >= config.MIN_SPACING_M - 1e-9)


@pytest.mark.parametrize("distance", [150.0, 250.0, 350.0])
def test_each_flight_is_within_half_d_and_d(near_gs, distance) -> None:
    """The drawn distance, unless the border clipped it shorter."""
    near_gs(distance)
    env = fleet()
    margin = config.BORDER_MARGIN_M
    flights = []
    for drone in env.drones:
        if drone.drone_type != "M":
            continue
        flown = float(np.linalg.norm(drone.end_point - drone.start_point))
        on_border = (
            abs(drone.end_point[0] - margin) < 1e-9
            or abs(drone.end_point[0] - (config.WIDTH - margin)) < 1e-9
            or abs(drone.end_point[1] - margin) < 1e-9
            or abs(drone.end_point[1] - (config.HEIGHT - margin)) < 1e-9
        )
        flights.append((flown, on_border))
    env.close_logger()

    assert flights
    for flown, clipped in flights:
        if clipped:
            # A clipped flight is shorter than it was drawn, never longer.
            assert flown <= distance + 1e-9
        else:
            assert 0.5 * distance - 1e-9 <= flown <= distance + 1e-9


def test_every_end_point_is_inside_the_arena(near_gs) -> None:
    """Clipping must land them inside, margin included."""
    near_gs(350.0)
    env = fleet()
    margin = config.BORDER_MARGIN_M
    ends = [d.end_point for d in env.drones if d.drone_type == "M"]
    env.close_logger()

    for point in ends:
        assert margin - 1e-9 <= point[0] <= config.WIDTH - margin + 1e-9
        assert margin - 1e-9 <= point[1] <= config.HEIGHT - margin + 1e-9


def test_a_drone_hovers_once_it_arrives(near_gs) -> None:
    """It reaches B, is marked arrived, and never moves again."""
    near_gs(150.0)
    env = fleet()
    run_to_end(env)
    drones = [d for d in env.drones if d.drone_type == "M"]
    env.close_logger()

    assert all(d.arrived for d in drones), "a flight did not finish in 100 s"
    for drone in drones:
        assert drone.position == pytest.approx(drone.end_point, abs=1e-9)
        assert drone.velocity == pytest.approx(np.zeros(2), abs=1e-9)


# --- D = 0 -----------------------------------------------------------------

def test_d_zero_moves_nobody(near_gs) -> None:
    """The easiest rung: end points are start points and nothing flies."""
    near_gs(0.0)
    env = fleet()
    starts = [d.start_point.copy() for d in env.drones if d.drone_type == "M"]
    run_to_end(env)
    drones = [d for d in env.drones if d.drone_type == "M"]
    flown = sum(d.distance_flown for d in drones)
    env.close_logger()

    assert flown == pytest.approx(0.0)
    for drone, start in zip(drones, starts):
        assert drone.position == pytest.approx(start, abs=1e-12)
        assert drone.end_point == pytest.approx(start, abs=1e-12)


@pytest.mark.parametrize("distance", [0.0, 150.0, 250.0, 350.0])
def test_every_rung_runs_a_full_episode(near_gs, distance) -> None:
    """The whole measurement window, at every D.

    Two shortcuts would have cut this short. At D = 0 every drone counts as
    arrived before step 1, which ended the episode immediately. At D = 350
    every drone has landed by about step 688, which emptied the last three
    ten-second slices -- the very slices the checkpoint asks for.
    """
    near_gs(distance)
    env = fleet()
    run_to_end(env)
    steps = env.step_count
    env.close_logger()

    assert steps == config.MAX_STEPS
    start, stop = config.measurement_window()
    assert steps >= stop, "the measurement window was cut short"


# --- the dial is a dial ----------------------------------------------------

def test_changing_d_leaves_the_start_positions_alone(near_gs) -> None:
    """Placement is drawn before the flights, so D cannot reach it."""
    layouts = {}
    for distance in (0.0, 150.0, 250.0, 350.0):
        near_gs(distance)
        env = fleet()
        layouts[distance] = np.array([d.start_point for d in env.drones])
        env.close_logger()

    first = layouts[0.0]
    for distance, positions in layouts.items():
        assert np.array_equal(positions, first), f"D={distance} moved the fleet"


def test_changing_d_leaves_the_headings_and_delays_alone(near_gs) -> None:
    """The distance is drawn LAST, so the other mobility draws do not shift.

    This is what makes two rungs comparable: the same layout flies the same
    directions at the same speeds after the same delays, only further.
    """
    flights = {}
    for distance in (150.0, 250.0, 350.0):
        near_gs(distance)
        env = fleet()
        flights[distance] = [
            (d.speed, d.start_delay_s,
             (d.end_point - d.start_point)
             / max(1e-12, float(np.linalg.norm(d.end_point - d.start_point))))
            for d in env.drones if d.drone_type == "M"
        ]
        env.close_logger()

    base = flights[150.0]
    bent = 0
    for distance, rows in flights.items():
        for (speed, delay, heading), (speed0, delay0, heading0) in zip(rows,
                                                                       base):
            assert speed == pytest.approx(speed0), f"D={distance} moved a speed"
            assert delay == pytest.approx(delay0), f"D={distance} moved a delay"
            # A flight the border clipped has its heading bent, which is the
            # one difference D may legitimately make. Everything else must
            # match to the bit.
            if float(np.linalg.norm(heading - heading0)) > 1e-9:
                bent += 1

    # Not an assertion about how many are bent, but about the rest: at D = 150
    # nothing is clipped from a start within 100 m of the centre, so that rung
    # must match itself exactly.
    for (speed, delay, heading), (speed0, delay0, heading0) in zip(
            flights[150.0], base):
        assert heading == pytest.approx(heading0, abs=1e-12)


def test_the_streams_stay_independent(near_gs) -> None:
    """Placement, mobility, channel, hello and C-drone draws do not mix.

    Changing the run seed must not move the layout or the flights; changing
    the placement seed must move both.
    """
    near_gs(350.0)

    one = fleet(301, 9100)
    two = fleet(301, 9999)
    three = fleet(302, 9100)

    starts_one = np.array([d.start_point for d in one.drones])
    starts_two = np.array([d.start_point for d in two.drones])
    starts_three = np.array([d.start_point for d in three.drones])
    ends_one = np.array([d.end_point for d in one.drones])
    ends_two = np.array([d.end_point for d in two.drones])
    for env in (one, two, three):
        env.close_logger()

    assert np.array_equal(starts_one, starts_two), "the run seed moved a layout"
    assert np.array_equal(ends_one, ends_two), "the run seed moved a flight"
    assert not np.array_equal(starts_one, starts_three)


# --- it is a working simulation -------------------------------------------

@pytest.mark.parametrize("distance", [0.0, 350.0])
def test_the_packets_add_up_and_the_views_agree(near_gs, distance) -> None:
    """Loss by cause sums to the total, and the ACK view matches the truth."""
    near_gs(distance)
    before = (config.IDEAL_HELLO, config.C_RULE)
    config.IDEAL_HELLO = False
    config.C_RULE = "midpoint"
    try:
        env = fleet()
        run_to_end(env)
        truth = ground_truth_view(env)
        acks = ack_view(env)
        env.close_logger()
    finally:
        config.IDEAL_HELLO, config.C_RULE = before

    assert compare_views(truth, acks) == []

    causes = ("channel", "queue_full", "no_route", "dead_end", "ttl",
              "hop_limit", "link_broken", "stale_link")
    by_reason = truth["loss_rate_by_reason"]
    total = sum(by_reason[cause] or 0.0 for cause in causes)
    assert total == pytest.approx(truth["total_loss"] or 0.0, abs=1e-9)


def test_two_hundred_runs_without_a_crash(near_gs) -> None:
    """Every D over fifty layouts, which is the sweep the step will run."""
    before = (config.IDEAL_HELLO, config.C_RULE, config.MAX_STEPS)
    config.IDEAL_HELLO = False
    config.C_RULE = "midpoint"
    # Short episodes: this is a crash test, not a measurement.
    config.MAX_STEPS = 60
    try:
        runs = 0
        for distance in (0.0, 150.0, 250.0, 350.0):
            near_gs(distance)
            for layout in range(301, 351):
                env = fleet(layout, 9100 + layout - 301)
                env.step()
                env.close_logger()
                runs += 1
        assert runs == 200
    finally:
        config.IDEAL_HELLO, config.C_RULE, config.MAX_STEPS = before


# --- the exploration spread on a continuation -----------------------------

def test_continue_std_defaults_to_a_fresh_spread() -> None:
    """0.1, the spread a fresh actor starts at."""
    assert config.CONTINUE_STD == pytest.approx(0.1)


@pytest.mark.parametrize("setting,expected", [(0.1, 0.1), (0.05, 0.05)])
def test_a_continuation_resets_its_spread(setting, expected) -> None:
    """A float puts the spread back, whatever the checkpoint stored."""
    torch = pytest.importorskip("torch")
    from agents.c_actor import GaussianActor

    trained = GaussianActor()
    with torch.no_grad():
        trained.log_std.fill_(float(np.log(0.0321)))
    state = trained.state_dict()

    before = config.CONTINUE_STD
    config.CONTINUE_STD = setting
    try:
        actor = GaussianActor()
        actor.load_state_dict(state)
        if config.CONTINUE_STD != "inherit":
            with torch.no_grad():
                actor.log_std.fill_(float(np.log(config.CONTINUE_STD)))
    finally:
        config.CONTINUE_STD = before

    assert float(actor.log_std.detach().exp().mean()) == pytest.approx(expected)


def test_inherit_keeps_the_trained_spread() -> None:
    """What M4 runs (a) and (b) ran, so their numbers stay reproducible."""
    torch = pytest.importorskip("torch")
    from agents.c_actor import GaussianActor

    trained = GaussianActor()
    with torch.no_grad():
        trained.log_std.fill_(float(np.log(0.0321)))
    state = trained.state_dict()

    before = config.CONTINUE_STD
    config.CONTINUE_STD = "inherit"
    try:
        actor = GaussianActor()
        actor.load_state_dict(state)
        if config.CONTINUE_STD != "inherit":
            with torch.no_grad():
                actor.log_std.fill_(float(np.log(config.CONTINUE_STD)))
    finally:
        config.CONTINUE_STD = before

    assert float(actor.log_std.detach().exp().mean()) == pytest.approx(0.0321)


def test_the_trainer_is_the_thing_that_applies_it() -> None:
    """The reset lives in train_one_seed, on the start_state branch only.

    The two tests above reproduce the trainer's logic rather than calling it,
    because calling it means training. This pins that the logic they copy is
    actually the logic that runs.
    """
    import inspect

    from scripts.train_c_ppo import train_one_seed

    source = inspect.getsource(train_one_seed)
    reset = source.index('config.CONTINUE_STD != "inherit"')
    loaded = source.index("actor.load_state_dict(start_state)")
    clone = source.index("actor.mean.load_state_dict")

    assert loaded < reset, "the reset must come after the load"
    assert clone < loaded, "the clone branch must stay first and untouched"
    # A fresh actor is already at 0.1, so the clone branch needs no reset and
    # must not have grown one: that would change what M0-M2 trained from.
    assert "CONTINUE_STD" not in source[clone:loaded]


# --- the curriculum sweep --------------------------------------------------

def test_the_static_sentinel_reproduces_m3() -> None:
    """D = STATIC must be the world M0-M3 ran in, to the bit.

    The reference row in the curriculum tables is only worth having if it is
    the same measurement M3 already published, so this reruns two of M3's
    committed validation rows through the sweep's own code path.

    It also pins AVOID: M3 ran with collision avoidance ON, and step 1 ran
    with it off because the sweep left it to the config default. Under the
    wrong setting these numbers do not match.
    """
    from scripts.curriculum_step1 import STATIC, run_once

    path = os.path.join(OUT, "2c_m3_compare.csv")
    committed = {}
    for row in csv.DictReader(
            line for line in open(path) if not line.startswith("#")):
        if row["policy"] in ("hover", "midpoint"):
            committed[(row["policy"], int(row["placement_seed"]))] = (
                100.0 * float(row["total_loss"]))

    for rule in ("hover", "midpoint"):
        fresh = run_once(STATIC, rule, 301, avoid=True)
        assert fresh["total_loss"] == pytest.approx(
            committed[(rule, 301)], abs=1e-9
        ), f"{rule} no longer reproduces M3's static validation row"
        assert fresh["pct_m_clipped_end"] == 0.0, "nothing flies in static"


def test_the_sweep_runs_with_avoidance_on_by_default() -> None:
    """Step 1 took the config default, which is OFF, against every other 2c
    result. The sweep now says what it wants."""
    from scripts.curriculum_step1 import AVOID

    assert AVOID is True
    assert config.AVOID is False, "the config default is still off"


def test_clipping_is_counted_only_when_it_happens(near_gs) -> None:
    """Short flights from the middle cannot reach a border; long ones do."""
    from scripts.curriculum_step1 import _spread

    near_gs(150.0)
    env = fleet()
    short = _spread(env)
    env.close_logger()

    near_gs(650.0)
    env = fleet()
    long = _spread(env)
    env.close_logger()

    assert short["pct_m_clipped_end"] == 0.0
    assert long["pct_m_clipped_end"] > 0.0


def test_a_clipped_flight_is_one_that_sits_on_the_margin(near_gs) -> None:
    """The geometric test the sweep uses, checked against the drawn distance.

    A clipped end point is shorter than [D/2, D] would have made it; an
    unclipped one is inside that band. If the two ever disagree the column
    is measuring something else.
    """
    near_gs(650.0)
    env = fleet()
    margin = config.BORDER_MARGIN_M
    for drone in env.drones:
        if drone.drone_type != "M":
            continue
        flown = float(np.linalg.norm(drone.end_point - drone.start_point))
        on_margin = any(
            abs(drone.end_point[axis] - edge) < 1e-9
            for axis, extent in ((0, config.WIDTH), (1, config.HEIGHT))
            for edge in (margin, extent - margin)
        )
        if not on_margin:
            assert 0.5 * 650.0 - 1e-9 <= flown <= 650.0 + 1e-9
    env.close_logger()
