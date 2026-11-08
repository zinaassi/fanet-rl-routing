"""Tests for scripts/train_c_ppo_round2.py."""

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs import c_interface
from scripts import train_c_ppo_round1 as r1
from scripts import train_c_ppo_round2 as r2

GS = np.asarray(config.GS_POSITION, dtype=np.float64)
HAVE_MODELS = os.path.exists(os.path.join("out", r1.START_ACTOR))


def test_pad_global_full_fleet_is_unchanged_and_small_fleet_is_slotted():
    per = c_interface.GLOBAL_PER_DRONE
    width = c_interface.global_state_width(25)
    full = np.arange(width, dtype=float)
    assert np.array_equal(r2.pad_global(full, 18, 7), full)
    small = np.arange(c_interface.global_state_width(3 + 2), dtype=float) + 1
    padded = r2.pad_global(small, 3, 2)
    assert len(padded) == width
    assert np.array_equal(padded[:3 * per], small[:3 * per])
    assert not padded[3 * per:18 * per].any()
    assert np.array_equal(padded[18 * per:20 * per], small[3 * per:5 * per])
    assert not padded[20 * per:25 * per].any()
    assert np.array_equal(padded[25 * per:], small[5 * per:])


def test_varied_plans_respect_the_ranges():
    rng = np.random.default_rng(0)
    for _ in range(400):
        plan = r2.draw_varied(rng)
        assert 1 <= plan["n_m"] <= 18 and 1 <= plan["n_c"] <= 7
        if plan["kind"] == "a":
            assert 300 <= plan["distance"] <= 650
        if plan["kind"] in ("b", "d"):
            assert 1 <= len(plan["flyers"]) <= min(3, plan["n_m"])
        if plan["kind"] == "c":
            assert len(plan["flyers"]) == math.ceil(plan["n_m"] / 2)
        if plan["kind"] == "d":
            assert len(plan["hover_s"]) == len(plan["flyers"])
            assert all(10 <= h <= 30 for h in plan["hover_s"])
        for point in plan["flyers"]:
            assert 450 <= np.linalg.norm(np.array(point) - GS) <= 622
            assert r2._inside(np.array(point))


def test_validation_plans_cycle_the_types_and_are_fixed():
    plans = r2.varied_validation_plans()
    assert [p[0] for p in plans] == list(range(331, 361))
    assert [p[1] for p in plans] == list(range(9200, 9230))
    assert [p[2]["kind"] for p in plans] == list("abcd" * 8)[:30]
    assert plans == r2.varied_validation_plans()


def test_training_plans_are_half_normal_and_off_without_varied():
    assert all(r2.training_plan(s, False) is r2.NORMAL_PLAN
               for s in range(50))
    kinds = [r2.training_plan(s, True)["kind"] for s in range(2000)]
    share = kinds.count("normal") / len(kinds)
    assert 0.45 < share < 0.55


@pytest.fixture(scope="module")
def pool():
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as p:
        yield p


@pytest.mark.skipif(not HAVE_MODELS, reason="no direct model")
def test_normal_reward_full_fleet_is_round1_control(pool):
    """Normal reward, no varied worlds: two updates bit-for-bit round 1's."""
    import torch

    a = r1.train(r1.fresh_state("control", "out"), pool, w=0.0, v=0.0,
                 until=2, episodes=2, out_dir="out", ckpt_dir=None,
                 model_tag=None)
    b = r2.train(r1.fresh_state("control", "out"), pool, reward="normal",
                 varied=False, until=2, episodes=2, out_dir="out",
                 ckpt_dir=None, model_tag=None)
    for x, y in zip(a["log"], b["log"]):
        for key in ("train_loss", "mean_reward", "value_loss", "std"):
            assert x[key] == y[key], key
    for name in ("actor", "critic"):
        for key, value in a[name].state_dict().items():
            assert torch.equal(value, b[name].state_dict()[key])


@pytest.mark.skipif(not HAVE_MODELS, reason="no direct model")
def test_out_and_back_returns_and_small_fleet_runs(pool):
    import torch

    state = torch.load(os.path.join("out", r1.START_ACTOR), weights_only=True)
    plan = {"kind": "d", "n_m": 3, "n_c": 2, "distance": 650.0,
            "flyers": [[800.0, 800.0]], "hover_s": [10.0]}
    r = r2.run_episode((101, 7, state, False, "fair", plan, "rl"))
    assert r["returned"] is True
    assert r["views_agree"]
    assert r["global"].shape[1] == c_interface.global_state_width(25)
    assert (config.NUM_M_DRONES, config.NUM_C_DRONES) == (18, 7)
