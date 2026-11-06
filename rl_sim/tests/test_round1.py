"""Tests for scripts/train_c_ppo_round1.py."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from fanet_sim import config
from scripts import train_c_ppo_round1 as r1

HAVE_MODELS = (os.path.exists(os.path.join("out", r1.START_ACTOR))
               and os.path.exists(os.path.join("out", r1.START_CRITIC)))


def test_spacing_penalty_on_hand_placed_drones():
    positions = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 100.0],
                          [0.0, 0.0], [300.0, 0.0]])
    c_ids = [1, 2, 3, 4]
    # Drone 3 sits on the M-drone (index 0, not a C) and 50 m from C 1;
    # C 2 is exactly 100 m away (costs nothing); C 4 is far.
    assert r1.spacing_penalty(positions, 3, c_ids, 0.01) == pytest.approx(
        -0.01 * (1 - 50 / 100))
    # On top of each other: the full weight.
    on_top = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert r1.spacing_penalty(on_top, 0, [0, 1], 0.02) == pytest.approx(-0.02)
    assert r1.spacing_penalty(positions, 4, c_ids, 0.01) == 0.0


def test_fairness_share_window():
    window = int(round(r1.FAIRNESS_WINDOW_S / config.TIMESTEP))
    end = 500
    deliveries = [(0, end - 1), (1, end - window), (2, end - window - 1),
                  (3, end), (0, end - 3)]
    # 0 and 1 are inside [end - window, end); 2 is too early, 3 not yet counted.
    assert r1.fairness_share(deliveries, 4, end) == pytest.approx(0.5)


def test_fairness_is_off_in_every_round1_run():
    assert all(v == 0.0 for _, v in r1.RUNS.values())
    assert list(r1.RUNS) == ["control", "spacing_med", "spacing_small",
                             "spacing_large"]
    assert [w for w, _ in r1.RUNS.values()] == [0.0, 0.01, 0.005, 0.02]


@pytest.fixture(scope="module")
def pool():
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as p:
        yield p


@pytest.mark.skipif(not HAVE_MODELS, reason="no direct model")
def test_no_terms_and_fresh_critic_is_train_one_seed(pool, tmp_path):
    """With the terms off and the critic fresh, two updates are bit-for-bit
    train_one_seed's from the same start."""
    import torch

    from scripts.train_c_ppo import train_one_seed
    from scripts.train_c_ppo_m3 import sample_layouts

    start = torch.load(os.path.join("out", r1.START_ACTOR), weights_only=True)
    reference = train_one_seed(
        r1.PPO_SEED, pool, str(tmp_path), 2, 2, layouts=sample_layouts,
        eval_jobs=lambda state: [], min_improvement=0.0, patience=10 ** 9,
        model_name="unused.pt", moving=True, anchor=False, start_state=start,
        stage_schedule=lambda update, count: [r1.DISTANCE_M] * count)
    mine = r1.train(r1.fresh_state("control", "out", critic_from_file=False),
                    pool, w=0.0, v=0.0, until=2, episodes=2, out_dir="out",
                    ckpt_dir=None, model_tag=None)
    for a, b in zip(reference["log"], mine["log"]):
        for key in ("train_loss", "mean_reward", "value_loss", "std"):
            assert a[key] == b[key], key
    for name in ("actor", "critic"):
        sa = reference["carry"][name].state_dict()
        sb = mine[name].state_dict()
        for key in sa:
            assert torch.equal(sa[key], sb[key]), (name, key)


@pytest.mark.skipif(not HAVE_MODELS, reason="no direct model")
def test_checkpoint_round_trip(tmp_path):
    import torch

    s = r1.fresh_state("control", "out")
    s["update"] = 7
    s["best"], s["best_update"] = 12.5, 7
    s["log"] = [{"update": 7}]
    s["normaliser"].update(np.array([0.1, 0.5, 0.9]))
    s["rng"].integers(0, 10, size=5)
    path = r1.save_checkpoint(str(tmp_path), s)
    expected_np = s["rng"].bit_generator.state
    expected_torch = torch.get_rng_state()
    back = r1.load_checkpoint(path)
    assert back["update"] == 7 and back["best"] == 12.5
    assert back["log"] == [{"update": 7}]
    assert back["normaliser"].mean == s["normaliser"].mean
    assert back["normaliser"].var == s["normaliser"].var
    assert back["rng"].bit_generator.state == expected_np
    assert torch.equal(torch.get_rng_state(), expected_torch)
    for name in ("actor", "critic"):
        for key, value in s[name].state_dict().items():
            assert torch.equal(value, back[name].state_dict()[key])


def test_only_the_latest_checkpoints_are_kept(tmp_path):
    for update in (25, 50, 75):
        open(os.path.join(tmp_path, f"ckpt_{update:04d}.pt"), "w").close()
    for old in r1.checkpoint_paths(str(tmp_path))[:-r1.KEEP_CHECKPOINTS]:
        os.remove(old)
    names = [os.path.basename(p) for p in r1.checkpoint_paths(str(tmp_path))]
    assert names == ["ckpt_0050.pt", "ckpt_0075.pt"]
