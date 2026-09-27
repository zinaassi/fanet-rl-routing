"""Calibration sweep: recommendation logic, reproducibility, smoke run."""
import os

import pytest

from stage1 import calibrate, config


def _stats(range_m, ring, grid, random, severe):
    means = [ring, grid, random]
    return calibrate.RangeStats(
        range_m=range_m,
        p_sens_dbm=-54.0,
        layout_pdr={"ring": ring, "grid": grid, "random": random},
        overall_pdr=sum(means) / 3.0,
        spread=max(means) - min(means),
        severe_frac=severe,
    )


def test_recommend_picks_smallest_passing_range():
    stats = [
        _stats(150.0, 0.10, 0.12, 0.11, severe=0.40),  # too sparse
        _stats(200.0, 0.40, 0.50, 0.45, severe=0.00),  # passes
        _stats(250.0, 0.60, 0.72, 0.66, severe=0.00),  # also passes, but larger
    ]
    choice, why = calibrate.recommend(stats)
    assert choice is not None and choice.range_m == 200.0
    assert any("smallest candidate" in line for line in why)


def test_recommend_rejects_flat_layout_spread():
    # High PDR but all layouts identical: range does not discriminate.
    stats = [_stats(400.0, 0.95, 0.96, 0.95, severe=0.0)]
    choice, _ = calibrate.recommend(stats)
    assert choice is None


def test_recommend_relaxes_band_when_needed():
    stats = [_stats(200.0, 0.20, 0.30, 0.25, severe=0.01)]  # overall 0.25
    choice, _ = calibrate.recommend(stats)
    assert choice is not None and choice.range_m == 200.0


def test_run_cal_unit_reproducible():
    kwargs = dict(base_seed=5, n_channels=2, n_steps=50, k=config.CAL_K)
    a = calibrate.run_cal_unit((250.0, "ring", 0), **kwargs)
    b = calibrate.run_cal_unit((250.0, "ring", 0), **kwargs)
    assert a == b
    assert 0.0 <= a.connected_frac <= 1.0


def test_calibrate_smoke_run(tmp_path):
    out = str(tmp_path / "out")
    stats, _choice = calibrate.main(
        [
            "--quick",
            "--ranges", "200", "300",
            "--n-topologies", "1",
            "--n-channels", "1",
            "--steps", "50",
            "--out-dir", out,
        ]
    )
    assert len(stats) == 2
    assert os.path.exists(os.path.join(out, "calibration_sweep.csv"))
    for s in stats:
        assert set(s.layout_pdr) == set(config.LAYOUTS)
        assert s.spread == pytest.approx(
            max(s.layout_pdr.values()) - min(s.layout_pdr.values())
        )
