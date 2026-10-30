"""
test_2c_test_eval.py — Phase 2c M3: the held-back TEST evaluation.

The test set is the only number in Phase 2c that nothing was chosen on, so
the thing worth testing is exactly that: that the fifty layouts never took
part in training or in picking the kept version, that every policy meets the
identical worlds, and that the actor sent to test is the one validation
chose.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_2c_test import (
    TEST_LAYOUTS,
    best_validation_seed,
    check_layouts_are_held_back,
    run_seed_for,
)
from scripts.train_c_ppo_m3 import TRAINING_LAYOUTS, VALIDATION_LAYOUTS


def test_the_test_layouts_are_the_fifty_phase_1b_used() -> None:
    """Phase 2c must be measured on the same held-back set as Phase 1b."""
    assert TEST_LAYOUTS == tuple(range(1, 51))
    assert len(TEST_LAYOUTS) == 50


def test_test_is_disjoint_from_training_and_validation() -> None:
    """Nothing the actor saw, and nothing it was chosen on, is in the test."""
    low, high = TRAINING_LAYOUTS
    assert set(TEST_LAYOUTS).isdisjoint(range(low, high + 1))
    assert set(TEST_LAYOUTS).isdisjoint(VALIDATION_LAYOUTS)

    # The guard in the script agrees, and says so rather than passing quietly.
    check_layouts_are_held_back()


def test_every_test_layout_keeps_its_own_run_seed() -> None:
    """One run seed per layout, and the Phase-1b convention."""
    seeds = [run_seed_for(layout) for layout in TEST_LAYOUTS]

    assert len(set(seeds)) == len(seeds), "two layouts share a run seed"
    assert seeds == list(TEST_LAYOUTS), "Phase 1b used the layout as run seed"


def test_test_run_seeds_are_not_the_validation_run_seeds() -> None:
    """The test traffic must not be the traffic the choice was made on."""
    from scripts.train_c_ppo_m3 import validation_run_seed

    validation = {validation_run_seed(layout)
                  for layout in VALIDATION_LAYOUTS}
    assert validation.isdisjoint(run_seed_for(layout)
                                 for layout in TEST_LAYOUTS)


def test_the_tested_actor_is_the_one_validation_chose() -> None:
    """The lowest validation loss wins, and ties go to the lower seed."""
    assert best_validation_seed({0: 17.33, 1: 15.55, 2: 15.35}) == 2
    assert best_validation_seed({0: 15.35, 1: 15.55, 2: 17.33}) == 0
    assert best_validation_seed({0: 16.0, 1: 16.0}) == 0
    assert best_validation_seed({1: 12.5}) == 1


def test_a_missing_validation_table_stops_instead_of_guessing() -> None:
    """Without the validation table there is no basis for a choice."""
    from scripts.evaluate_2c_test import validation_means

    with pytest.raises(SystemExit):
        validation_means("out/this-file-does-not-exist.csv")
