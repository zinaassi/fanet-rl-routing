"""
test_2c_m4.py — Phase 2c M4: the moving world.

M4 changes four things and nothing else: the mission flies, the runs start
from M3's actors instead of the clone, they may not stop before update 300,
and they save a critic beside the actor. The tests here are about exactly
those, about the held-back layouts, and about the ten-second slices the
checkpoint asks the loss to be cut into.

The PPO machinery is unchanged and is covered by test_2c_ppo.py, which also
pins M2's defaults. The tests below pin M4's additions to the same function
in the same way, so M4 cannot quietly move M2 or M3 either.
"""

from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from scripts.evaluate_2c_m4_test import (
    TEST_LAYOUTS,
    TEST_RUN_SEED_BASE,
    check_layouts_are_held_back,
    run_seed_for,
)
from scripts.metrics_1a import (
    loss_by_time_window,
    measured_packets,
    time_window_labels,
)
from scripts.train_c_ppo import _rollout_job, _world, train_one_seed
from scripts.train_c_ppo_m3 import TRAINING_LAYOUTS, VALIDATION_LAYOUTS
from scripts.train_c_ppo_m4 import (
    MAX_UPDATES_M4,
    MIN_IMPROVEMENT_M4,
    MIN_UPDATES_M4,
    PATIENCE_M4,
    RUNS,
    run_by_name,
    validation_jobs,
)


# --- the four runs ---------------------------------------------------------

def test_the_runs_are_the_four_the_checkpoint_asked_for() -> None:
    """Order, starting point and anchor term of each run."""
    assert [run["name"] for run in RUNS] == ["a", "b", "c", "d"]
    assert [run["start"] for run in RUNS] == [2, 1, None, 0]

    # The three main runs continue an M3 actor, so nothing should pull them
    # back towards midpoint; the comparison run starts from the clone and
    # keeps M3's recipe.
    assert [run["anchor"] for run in RUNS] == [False, False, True, False]
    assert run_by_name("c")["start"] is None


def test_every_run_has_its_own_ppo_seed() -> None:
    """Two runs sharing a seed would share their exploration stream."""
    seeds = [run["seed"] for run in RUNS]
    assert len(set(seeds)) == len(seeds)


def test_an_unknown_run_name_stops() -> None:
    """A typo must not silently train nothing."""
    with pytest.raises(SystemExit):
        run_by_name("z")


def test_the_schedule_is_the_one_the_checkpoint_set() -> None:
    """Cap 600, no stopping before 300, patience 8, keep by 0.2."""
    assert MAX_UPDATES_M4 == 600
    assert MIN_UPDATES_M4 == 300
    assert MIN_IMPROVEMENT_M4 == pytest.approx(0.2)
    assert PATIENCE_M4 == 8


# --- the moving world ------------------------------------------------------

def test_the_world_moves_only_when_asked() -> None:
    """_world(True) flies the mission; the default keeps M0-M3's."""
    before = config.STATIC_MODE
    try:
        previous = _world(moving=True)
        assert config.STATIC_MODE is False
        from scripts.train_c_ppo import _restore
        _restore(previous)

        previous = _world()
        assert config.STATIC_MODE is True, "the default must stay static"
        _restore(previous)
    finally:
        config.STATIC_MODE = before
    assert config.STATIC_MODE == before, "the world was not put back"


def test_validation_runs_in_the_moving_world_without_exploration() -> None:
    """Thirty greedy moving jobs, one per validation layout."""
    jobs = validation_jobs({"w": 1})

    assert len(jobs) == len(VALIDATION_LAYOUTS) == 30
    assert [job[0] for job in jobs] == list(VALIDATION_LAYOUTS)
    assert all(job[4] is True for job in jobs), "evaluation must not explore"
    assert all(job[5] is True for job in jobs), "evaluation must be moving"


def test_a_short_job_still_means_the_static_world() -> None:
    """M0-M3 built 5-tuples, and those must keep their meaning."""
    source = inspect.getsource(_rollout_job)

    assert "args[:5]" in source
    assert "len(args) > 5" in source
    # The default, spelled out: a 5-tuple is not moving.
    assert "else False" in source


# --- M2 and M3 must not move ----------------------------------------------

def test_m4s_options_all_default_to_what_m2_and_m3_did() -> None:
    """Every new argument is off unless M4 turns it on."""
    defaults = inspect.signature(train_one_seed).parameters

    assert defaults["moving"].default is False
    assert defaults["min_updates"].default == 0
    assert defaults["anchor"].default is True
    assert defaults["start_state"].default is None
    assert defaults["critic_name"].default is None


# --- the held-back layouts -------------------------------------------------

def test_m4_tests_on_fresh_layouts() -> None:
    """401-450, used by nothing before."""
    assert TEST_LAYOUTS == tuple(range(401, 451))
    assert len(TEST_LAYOUTS) == 50
    check_layouts_are_held_back()


def test_m4_never_touches_the_phase_1b_test_set() -> None:
    """Layouts 1-50 were already reported on, so M4 must not reuse them."""
    assert set(TEST_LAYOUTS).isdisjoint(range(1, 51))

    low, high = TRAINING_LAYOUTS
    assert set(TEST_LAYOUTS).isdisjoint(range(low, high + 1))
    assert set(TEST_LAYOUTS).isdisjoint(VALIDATION_LAYOUTS)


def test_every_test_layout_keeps_its_own_run_seed() -> None:
    """9400 + offset, one per layout, and not a validation run seed."""
    from scripts.train_c_ppo_m3 import validation_run_seed

    seeds = [run_seed_for(layout) for layout in TEST_LAYOUTS]

    assert seeds[0] == TEST_RUN_SEED_BASE == 9400
    assert len(set(seeds)) == len(seeds)
    validation = {validation_run_seed(layout)
                  for layout in VALIDATION_LAYOUTS}
    assert validation.isdisjoint(seeds)


# --- the ten-second slices -------------------------------------------------

def _fake_env(packets):
    """Return something with just the attribute measured_packets reads.

    Args:
        packets: (created_at, delivered) pairs.

    Returns:
        A stand-in environment.
    """
    return SimpleNamespace(all_packets=[
        SimpleNamespace(created_at=at, delivered=ok) for at, ok in packets
    ])


def test_the_slices_are_ten_seconds_across_the_measured_window() -> None:
    """5-15s through 85-95s, which is the window the report already uses."""
    labels = time_window_labels()

    assert labels[0] == "5-15s"
    assert labels[-1] == "85-95s"
    assert len(labels) == 9


def test_a_slice_holds_exactly_the_packets_created_inside_it() -> None:
    """The boundary belongs to the later slice, as the window does."""
    env = _fake_env([(49, True), (50, True), (149, True), (150, True)])

    assert len(measured_packets(env, (50, 150))) == 2
    assert [p.created_at for p in measured_packets(env, (50, 150))] == [50, 149]


def test_the_slices_partition_the_run() -> None:
    """Every measured packet lands in exactly one slice, and none is lost.

    The weighted mean of the slice losses must come back to the total loss,
    or the split is describing a different set of packets than the headline.
    """
    start, stop = config.measurement_window()
    packets = [(step, step % 3 != 0) for step in range(start, stop, 7)]
    env = _fake_env(packets)

    counts = {}
    for first in range(start, stop, 100):
        counts[f"{first * config.TIMESTEP:.0f}-" \
               f"{min(first + 100, stop) * config.TIMESTEP:.0f}s"] = len(
            measured_packets(env, (first, min(first + 100, stop)))
        )

    assert sum(counts.values()) == len(measured_packets(env))

    by_window = loss_by_time_window(env)
    assert set(by_window) == set(time_window_labels())
    total = sum(counts[label] * by_window[label] for label in by_window)
    measured = measured_packets(env)
    lost = sum(1 for pkt in measured if not pkt.delivered)
    assert total == pytest.approx(lost)


def test_an_empty_slice_does_not_crash() -> None:
    """A run can create nothing in a slice; that is a gap, not an error."""
    start, _ = config.measurement_window()
    by_window = loss_by_time_window(_fake_env([(start, False)]))

    assert by_window["5-15s"] == pytest.approx(1.0)
    assert by_window["85-95s"] in (0.0, None)


def test_the_model_name_lands_in_out_models_2c() -> None:
    """train_one_seed joins out_dir and "models" itself.

    Passing it a name that starts with "models/" once wrote the actor to
    out/models/models/2c and failed the run after ten updates, so the two
    spellings are pinned apart here.
    """
    import os

    from scripts.train_c_ppo_m4 import MODEL_DIR, MODEL_SUBDIR

    assert MODEL_SUBDIR == "2c"
    assert not MODEL_SUBDIR.startswith("models")
    assert MODEL_DIR == os.path.join("models", "2c")

    # What train_one_seed will build from the name M4 hands it.
    built = os.path.join("out", "models",
                         os.path.join(MODEL_SUBDIR, "c_actor_m4_a.pt"))
    assert built == os.path.join("out", "models", "2c", "c_actor_m4_a.pt")
    # And what everything else builds from MODEL_DIR must be the same place.
    assert os.path.join("out", MODEL_DIR, "c_actor_m4_a.pt") == built
