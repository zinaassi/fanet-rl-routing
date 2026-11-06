"""Tests for scripts/edge_cases_2c.py: the scenarios are built as stated."""

import os

import numpy as np
import pytest

from fanet_sim import config
from scripts import edge_cases_2c as edge

GS = np.asarray(config.GS_POSITION, dtype=np.float64)
HAND_BUILT = (1, 2, 3, 4, 5, 6, 8)


@pytest.mark.parametrize("scenario", HAND_BUILT)
def test_starts_are_near_the_gs_and_spaced(scenario):
    plan = edge.scenario_plan(scenario)
    points = plan["m_start"] + plan["c_start"]
    num_m, num_c, _ = edge.SCENARIOS[scenario]
    assert len(plan["m_start"]) == num_m and len(plan["c_start"]) == num_c
    for i, p in enumerate(points):
        gap = np.linalg.norm(p - GS)
        assert config.MIN_SPACING_M <= gap <= config.START_RADIUS_M
        for q in points[i + 1:]:
            assert np.linalg.norm(p - q) >= config.MIN_SPACING_M


@pytest.mark.parametrize("scenario", HAND_BUILT)
def test_end_points_are_inside_the_border(scenario):
    margin = config.BORDER_MARGIN_M
    for end in edge.scenario_plan(scenario)["m_end"]:
        assert margin <= end[0] <= config.WIDTH - margin
        assert margin <= end[1] <= config.HEIGHT - margin


def test_corner_scenarios_end_on_the_corner():
    corner = np.array([config.WIDTH, config.HEIGHT]) - config.BORDER_MARGIN_M
    for scenario in (2, 6):
        end = edge.scenario_plan(scenario)["m_end"][0]
        assert np.allclose(end, corner)
        assert np.linalg.norm(end - GS) == pytest.approx(622.25, abs=0.01)


@pytest.fixture(scope="module")
def s5_run(tmp_path_factory):
    rec_dir = str(tmp_path_factory.mktemp("rec"))
    row = edge.run_one((5, "midpoint", 9901, None, rec_dir))
    rec = np.load(os.path.join(rec_dir, "s5_midpoint_9901.npz"))
    return row, rec


def test_scenario_5_flies_out_and_back(s5_run):
    row, rec = s5_run
    pos = rec["pos"]
    start = edge.scenario_plan(5)["m_start"][0]
    far = np.linalg.norm(pos[:, 0] - GS, axis=-1).max()
    assert far == pytest.approx(600.0, abs=1e-6)
    back = int(round(80.0 / config.TIMESTEP))
    assert np.allclose(pos[back, 0], start)


def test_recorded_track_matches_distance_flown(s5_run):
    row, rec = s5_run
    pos = rec["pos"]
    c = [i for i, t in enumerate(rec["types"]) if t == "C"]
    flown = np.linalg.norm(np.diff(pos[:, c], axis=0), axis=-1).sum(axis=0)
    assert flown.mean() == pytest.approx(row["c_distance_flown_mean"],
                                         rel=1e-6)
    assert row["views_agree"]


def test_fleet_size_is_put_back(s5_run):
    assert (config.NUM_M_DRONES, config.NUM_C_DRONES) == (18, 7)


def test_direct_actor_runs_on_one_m_and_one_c(tmp_path):
    import torch

    path = os.path.join("out", edge.DIRECT_MODEL)
    if not os.path.exists(path):
        pytest.skip("no direct model")
    state = torch.load(path, weights_only=True)
    row = edge.run_one((1, "direct", 9901, state, str(tmp_path)))
    rec = np.load(os.path.join(str(tmp_path), "s1_direct_9901.npz"))
    assert rec["pos"].shape == (config.MAX_STEPS + 1, 2, 2)
    assert row["views_agree"]
