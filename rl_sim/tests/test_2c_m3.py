"""
test_2c_m3.py — Phase 2c M3: training on many layouts.

M3 changes three things about M2 and nothing else: which layouts the
episodes run on, which layouts choose the kept version, and how patient the
keep rule is. The tests here are about exactly those, plus the paired
statistics the comparison reports.

The PPO machinery itself is unchanged and is covered by test_2c_ppo.py,
which also pins M2's defaults so this milestone cannot quietly move it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_c_ppo_m3 import (
    EPISODES_PER_UPDATE_M3,
    MAX_UPDATES_M3,
    MIN_IMPROVEMENT_M3,
    PATIENCE_M3,
    TRAINING_LAYOUTS,
    VALIDATION_LAYOUTS,
    by_layout,
    paired_rows,
    sample_layouts,
    validation_jobs,
    validation_run_seed,
)


def test_training_draws_from_the_whole_training_range() -> None:
    """Episodes land across 101-300, not on one corner of it."""
    rng = np.random.default_rng(0)
    drawn = []
    for _ in range(50):
        drawn.extend(sample_layouts(rng, EPISODES_PER_UPDATE_M3))

    low, high = TRAINING_LAYOUTS
    assert min(drawn) >= low
    assert max(drawn) <= high
    # 2000 draws over 200 layouts: seeing nearly all of them is what
    # "uniform over the range" should look like.
    assert len(set(drawn)) > 190


def test_one_update_mixes_layouts() -> None:
    """A single update must not be forty runs of the same world."""
    rng = np.random.default_rng(1)
    one = sample_layouts(rng, EPISODES_PER_UPDATE_M3)

    assert len(one) == EPISODES_PER_UPDATE_M3
    assert len(set(one)) > 1


def test_successive_updates_differ() -> None:
    """The sampler consumes the generator, so updates see new layouts."""
    rng = np.random.default_rng(2)
    first = sample_layouts(rng, EPISODES_PER_UPDATE_M3)
    second = sample_layouts(rng, EPISODES_PER_UPDATE_M3)

    assert first != second


def test_training_never_draws_a_validation_layout() -> None:
    """The set the kept version is chosen on must be unseen."""
    low, high = TRAINING_LAYOUTS
    assert set(range(low, high + 1)).isdisjoint(VALIDATION_LAYOUTS)

    rng = np.random.default_rng(3)
    drawn = set()
    for _ in range(100):
        drawn.update(sample_layouts(rng, EPISODES_PER_UPDATE_M3))
    assert drawn.isdisjoint(VALIDATION_LAYOUTS)


def test_validation_is_thirty_layouts_without_exploration() -> None:
    """Every check runs the same thirty worlds, greedily."""
    jobs = validation_jobs({"w": 1})

    assert len(jobs) == len(VALIDATION_LAYOUTS) == 30
    assert [job[0] for job in jobs] == list(VALIDATION_LAYOUTS)
    assert all(job[3] == "rl" for job in jobs)
    assert all(job[4] is True for job in jobs), "evaluation must not explore"


def test_every_validation_layout_keeps_its_own_run_seed() -> None:
    """One fixed run seed per layout, the same at every check."""
    seeds = [validation_run_seed(layout) for layout in VALIDATION_LAYOUTS]

    assert len(set(seeds)) == len(seeds), "two layouts share a run seed"
    assert seeds == [validation_run_seed(layout)
                     for layout in VALIDATION_LAYOUTS]
    assert [job[1] for job in validation_jobs(None)] == seeds


def test_the_keep_rule_is_looser_and_more_patient_than_m2() -> None:
    """M3's schedule is the one the checkpoint asked for."""
    assert EPISODES_PER_UPDATE_M3 == 40
    assert MAX_UPDATES_M3 == 600
    assert MIN_IMPROVEMENT_M3 == 0.2
    assert PATIENCE_M3 == 8


def _rows(policy: str, losses: dict) -> list:
    """Return comparison rows for one policy.

    Args:
        policy: The label.
        losses: Placement seed -> total loss, as a fraction.

    Returns:
        One row per layout.
    """
    return [{"policy": policy, "placement_seed": layout,
             "total_loss": loss} for layout, loss in losses.items()]


def test_paired_differences_are_per_layout_and_signed() -> None:
    """A difference compares the two policies on the SAME layout."""
    rows = _rows("rl seed 0", {301: 0.10, 302: 0.20, 303: 0.30})
    rows += _rows("midpoint", {301: 0.12, 302: 0.18, 303: 0.40})
    rows += _rows("hover", {301: 0.50, 302: 0.50, 303: 0.50})

    paired = paired_rows(rows, ["rl seed 0"])
    against_midpoint = {row["placement_seed"]: row["difference"]
                        for row in paired if row["against"] == "midpoint"}

    assert against_midpoint[301] == pytest.approx(-2.0)
    assert against_midpoint[302] == pytest.approx(+2.0)
    assert against_midpoint[303] == pytest.approx(-10.0)

    against_hover = {row["placement_seed"]: row["difference"]
                     for row in paired if row["against"] == "hover"}
    assert against_hover[301] == pytest.approx(-40.0)


def test_by_layout_picks_out_one_policy() -> None:
    """The per-layout lookup must not mix policies together."""
    rows = _rows("rl seed 0", {301: 0.10}) + _rows("midpoint", {301: 0.40})

    assert by_layout(rows, "rl seed 0") == {301: 0.10}
    assert by_layout(rows, "midpoint") == {301: 0.40}
    assert by_layout(rows, "hover") == {}


def test_a_policy_that_never_wins_gets_a_significant_sign_test() -> None:
    """The reported p-value must respond to the differences it is given."""
    from scripts.evaluate_1b_large import sign_test

    always_better = [-1.0] * 30
    wins, used, p = sign_test(always_better)
    assert (wins, used) == (30, 30)
    assert p < 1e-6

    even = [-1.0] * 15 + [1.0] * 15
    wins, used, p = sign_test(even)
    assert (wins, used) == (15, 30)
    assert p == pytest.approx(1.0)


def test_compare_only_reuses_the_trained_actors() -> None:
    """--compare-only must read the models, never train another update."""
    import os

    from scripts.train_c_ppo_m3 import already_trained

    models, evaluations = already_trained("out", [0, 1, 2])

    assert sorted(models) == [0, 1, 2]
    for seed, path in models.items():
        assert path.endswith(f"c_actor_m3_seed{seed}.pt")
        assert os.path.exists(path), "the actor must already be on disk"
    assert {row["seed"] for row in evaluations} == {0, 1, 2}
    assert any(row["saved"] for row in evaluations)


def test_compare_only_stops_when_an_actor_is_missing() -> None:
    """A seed with no model is a stop, not a silently smaller comparison."""
    from scripts.train_c_ppo_m3 import already_trained

    with pytest.raises(SystemExit):
        already_trained("out", [0, 99])


def test_the_evaluation_history_round_trips() -> None:
    """Rows read back from the CSV must be typed as training held them."""
    from scripts.train_c_ppo_m3 import read_evaluations

    rows = read_evaluations("out/2c_m3_eval.csv")

    assert rows, "the evaluation history is empty"
    assert all(isinstance(row["seed"], int) for row in rows)
    assert all(isinstance(row["update"], int) for row in rows)
    assert all(isinstance(row["eval_loss"], float) for row in rows)
    assert all(isinstance(row["saved"], bool) for row in rows)
    # plot_learning stars the last saved row, so "saved" must not be the
    # string "False", which is truthy.
    assert not all(row["saved"] for row in rows)
