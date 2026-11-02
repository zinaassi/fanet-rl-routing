"""
test_curriculum.py — Phase 2c: the staged curriculum on the near_gs mission.

The curriculum adds three things to the shared trainer: a mission distance
that travels per episode, a carry so a later stage continues the critic it
already warmed up, and a warm-up length that a later stage sets to zero. All
three default to what M0-M4 did, and the tests below say so explicitly,
because train_one_seed is the one function every Phase-2c run goes through.

The rest is about the curriculum's own arithmetic: which stages exist, how
much of an update is replay, and the paired confidence interval.
"""

from __future__ import annotations

import inspect
import os
import statistics
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from scripts.train_c_ppo import _restore, _rollout_job, _world, train_one_seed
from scripts.train_c_ppo_curriculum import (
    MIN_IMPROVEMENT_CURRICULUM,
    MIN_UPDATES_PER_STAGE,
    PPO_SEED,
    REPLAY_FRACTION,
    REPLAY_SEED_TAG,
    STAGES,
    START_MODEL,
    replay_schedule,
    validation_run_seed,
)


# --- the stages ------------------------------------------------------------

def test_the_stages_are_the_five_the_checkpoint_set() -> None:
    """250 to 650, easiest first."""
    assert STAGES == (250.0, 350.0, 450.0, 550.0, 650.0)
    assert list(STAGES) == sorted(STAGES), "a curriculum climbs"


def test_the_schedule_numbers() -> None:
    """Replay quarter, 100-update floor, keep every new lowest."""
    assert REPLAY_FRACTION == pytest.approx(0.25)
    assert MIN_UPDATES_PER_STAGE == 100
    # NOT M4's 0.2: the checkpoint asks for the lowest validation loss, and
    # 0.2 threw away two better checkpoints in M4 run (a).
    assert MIN_IMPROVEMENT_CURRICULUM == 0.0


def test_both_runs_start_from_m3_seed_1() -> None:
    """The strongest of the three M3 actors in the moving world."""
    assert START_MODEL == "c_actor_m3_seed1.pt"


def test_validation_run_seeds_match_step_1b() -> None:
    """9100 + offset, so the learned actors sit in step 1b's table."""
    from scripts.curriculum_step1 import run_seed_for

    for layout in (301, 310, 330):
        assert validation_run_seed(layout) == run_seed_for(layout)


# --- replay ----------------------------------------------------------------

def test_the_first_stage_has_nothing_to_replay() -> None:
    """Every episode is the current stage."""
    rng = np.random.default_rng(0)
    schedule = replay_schedule(0, 40, rng)

    distances = schedule(1, 40)
    assert distances == [STAGES[0]] * 40


@pytest.mark.parametrize("stage_index", [1, 2, 3, 4])
def test_a_quarter_of_each_update_is_an_earlier_stage(stage_index) -> None:
    """25% replay, and never a stage that has not been reached."""
    rng = np.random.default_rng(0)
    schedule = replay_schedule(stage_index, 40, rng)
    current = STAGES[stage_index]
    earlier = set(STAGES[:stage_index])

    distances = schedule(1, 40)

    assert len(distances) == 40
    replayed = [d for d in distances if d != current]
    assert len(replayed) == 10, "a quarter of forty"
    assert set(replayed) <= earlier, "replayed a stage not yet reached"
    assert all(d < current for d in replayed), "replay is always easier"


def test_replay_spreads_over_the_earlier_stages() -> None:
    """Uniform, not always the same one.

    Checked over many updates rather than one, since ten draws from four
    stages can legitimately miss one.
    """
    rng = np.random.default_rng(0)
    schedule = replay_schedule(4, 40, rng)
    current = STAGES[4]

    seen = set()
    for update in range(1, 51):
        seen.update(d for d in schedule(update, 40) if d != current)

    assert seen == set(STAGES[:4])


def test_replay_has_its_own_stream() -> None:
    """Its tag is not one of the environment's.

    If replay drew from the trainer's generator it would shift the layouts
    and run seeds, and the staged and direct runs would no longer be the
    same budget on the same episodes.
    """
    assert REPLAY_SEED_TAG not in (config.AGENT_STREAM_TAG,
                                   config.MOBILITY_STREAM_TAG,
                                   config.HELLO_STREAM_TAG,
                                   config.C_DRONE_STREAM_TAG)

    source = inspect.getsource(replay_schedule)
    assert "rng.integers" in source
    # The generator is an argument, never built from the trainer's.
    assert "default_rng" not in source


# --- the mission distance travels per episode ------------------------------

def test_a_six_tuple_job_still_means_no_mission_change() -> None:
    """M0-M4's jobs must keep their meaning exactly."""
    source = inspect.getsource(_rollout_job)

    assert "args[:5]" in source
    assert "len(args) > 6" in source
    assert "else None" in source


def test_world_only_touches_the_mission_when_given_a_distance() -> None:
    """Without a distance, MISSION is whatever it was."""
    before = (config.MISSION, config.MISSION_DISTANCE_M, config.STATIC_MODE)
    try:
        config.MISSION = "border"
        previous = _world(moving=True)
        assert config.MISSION == "border", "the default moved the mission"
        _restore(previous)
        assert config.MISSION == "border"

        previous = _world(moving=True, distance=450.0)
        assert config.MISSION == "near_gs"
        assert config.MISSION_DISTANCE_M == pytest.approx(450.0)
        assert config.STATIC_MODE is False
        _restore(previous)
        assert config.MISSION == "border", "the mission was not put back"
    finally:
        (config.MISSION, config.MISSION_DISTANCE_M,
         config.STATIC_MODE) = before


def test_a_distance_actually_changes_the_episode() -> None:
    """Two distances on one layout must fly different flights."""
    from scripts.train_c_ppo import _run_episode

    short = _run_episode(301, 9100, None, "hover", True, True, 250.0)
    long = _run_episode(301, 9100, None, "hover", True, True, 650.0)

    assert (short["m_gs_distance_end_mean"]
            < long["m_gs_distance_end_mean"]), "D did not reach the mission"
    assert short["pct_m_clipped_end"] <= long["pct_m_clipped_end"]


def test_the_episode_reports_the_mission_spread() -> None:
    """The curriculum's tables need these beside the loss."""
    from scripts.train_c_ppo import _run_episode

    result = _run_episode(301, 9100, None, "hover", True, True, 650.0)

    for key in ("m_gs_distance_end_mean", "pct_m_beyond_reach_end",
                "pct_m_clipped_end", "reach_m"):
        assert key in result


# --- M0-M4 must not move ---------------------------------------------------

def test_the_new_options_default_to_what_m0_to_m4_did() -> None:
    """Every argument the curriculum added is off unless it asks."""
    defaults = inspect.signature(train_one_seed).parameters

    assert defaults["stage_schedule"].default is None
    assert defaults["carry"].default is None

    from scripts.train_c_ppo import CRITIC_ONLY_UPDATES
    assert defaults["warmup"].default == CRITIC_ONLY_UPDATES


def test_a_fresh_run_still_builds_its_own_critic() -> None:
    """The carry branch must not swallow the normal path."""
    source = inspect.getsource(train_one_seed)
    carried = source.index("if carry is not None:")
    fresh = source.index("actor = GaussianActor()")

    assert carried < fresh, "the fresh path is the else branch"
    assert "CCritic(" in source[fresh:], "a fresh run builds a critic"


def test_the_evaluation_checks_the_views_agree() -> None:
    """The reported number must come from runs the ACKs corroborated."""
    source = inspect.getsource(train_one_seed)

    assert 'for r in eval_results if r["views_agree"]' in source
    assert "views disagreed in" in source
    # It must raise, not warn: a disagreeing run means the ACK counters and
    # the simulator no longer describe the same network.
    assert "raise RuntimeError(" in source[source.index("eval_results"):]


# --- the paired interval ---------------------------------------------------

def test_the_confidence_interval_is_paired_and_symmetric() -> None:
    """Mean plus or minus t * sd / sqrt(n), on the differences themselves."""
    from scripts.curriculum_validate import T_95_DF29, confidence_interval

    differences = [-1.0, -2.0, 0.5, -3.0, -0.5, -1.5, -2.5, 1.0, -0.25, -1.75]
    mean, low, high = confidence_interval(differences)

    assert mean == pytest.approx(statistics.fmean(differences))
    assert mean - low == pytest.approx(high - mean)
    half = (T_95_DF29 * statistics.stdev(differences)
            / len(differences) ** 0.5)
    assert high - mean == pytest.approx(half)


def test_one_difference_has_no_interval() -> None:
    """A single layout cannot have a spread; it must not divide by zero."""
    from scripts.curriculum_validate import confidence_interval

    mean, low, high = confidence_interval([-1.25])

    assert mean == low == high == pytest.approx(-1.25)


def test_the_t_value_is_for_thirty_layouts() -> None:
    """29 degrees of freedom, not the normal approximation."""
    from scripts.curriculum_validate import T_95_DF29
    from scripts.curriculum_step1 import VALIDATION_LAYOUTS

    assert len(VALIDATION_LAYOUTS) == 30
    assert T_95_DF29 == pytest.approx(2.045, abs=5e-4)
    assert T_95_DF29 > 1.96, "t must be wider than the normal value"


def test_every_policy_is_evaluated_at_every_stage() -> None:
    """The grid is five by five by thirty."""
    from scripts.curriculum_validate import POLICIES, build_jobs

    actors = {label: {"w": 1} for label, filename in POLICIES
              if filename is not None}
    jobs, labels = build_jobs(actors)

    assert len(jobs) == len(STAGES) * len(POLICIES) * 30
    assert len(set(labels)) == len(STAGES) * len(POLICIES)
    # Evaluation never explores, and always flies the mission.
    assert all(job[4] is True for job in jobs)
    assert all(job[5] is True for job in jobs)
    assert {job[6] for job in jobs} == set(STAGES)


def test_the_validation_grid_never_touches_the_held_back_layouts() -> None:
    """Layouts 1-50 and 401-450 are not this task's to spend."""
    from scripts.curriculum_validate import POLICIES, build_jobs

    actors = {label: {"w": 1} for label, filename in POLICIES
              if filename is not None}
    jobs, _ = build_jobs(actors)
    layouts = {job[0] for job in jobs}

    assert layouts == set(range(301, 331))
    assert layouts.isdisjoint(range(1, 51))
    assert layouts.isdisjoint(range(401, 451))


# --- the S5-from-S4 follow-up ----------------------------------------------

def test_the_s5_test_continues_s4_with_more_patience() -> None:
    """S4's weights, patience 16, D = 650 only, and a warm-up for the critic."""
    from scripts.train_c_ppo_curriculum import (
        S5_FROM_S4_PATIENCE,
        S5_FROM_S4_START,
        train_s5_from_s4,
    )

    assert S5_FROM_S4_START == "c_actor_curriculum_S4.pt"
    assert S5_FROM_S4_PATIENCE == 16

    source = inspect.getsource(train_s5_from_s4)
    assert "patience=S5_FROM_S4_PATIENCE" in source
    assert "[distance] * count" in source, "no replay in this test"
    assert "warmup=CRITIC_ONLY_UPDATES" in source
    # The start_state path is the one that resets the spread.
    assert "start_state=start_weights" in source
    assert "carry=" not in source


def test_the_s5_test_writes_beside_the_committed_tables() -> None:
    """Its own policy and pairs; the default grid stays the five policies."""
    from scripts.curriculum_validate import PAIRS, POLICIES, S5_TEST_POLICY

    assert len(POLICIES) == 5
    assert S5_TEST_POLICY not in POLICIES
    assert ("staged_final", "midpoint") in PAIRS
