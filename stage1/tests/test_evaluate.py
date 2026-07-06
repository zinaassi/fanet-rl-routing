"""Harness: aggregation math, unit reproducibility, end-to-end smoke run."""
import csv
import os

import numpy as np
import pytest

from stage1 import evaluate, metrics


def test_aggregate_between_vs_within_topology():
    values = np.array([[1.0, 2.0], [3.0, 5.0]])  # 2 topologies x 2 realizations
    s = metrics.aggregate(values)
    assert s.mean == pytest.approx(2.75)
    assert s.between_std == pytest.approx(np.std([1.5, 4.0], ddof=1))
    assert s.within_std == pytest.approx(
        np.mean([np.std([1, 2], ddof=1), np.std([3, 5], ddof=1)])
    )


def test_aggregate_ignores_nan_cells():
    values = np.array([[1.0, np.nan], [3.0, 3.0]])
    s = metrics.aggregate(values)
    assert s.mean == pytest.approx(2.0)


def test_run_unit_is_reproducible():
    kwargs = dict(base_seed=99, routers=("dijkstra",), n_channels=2, n_steps=50)
    a = evaluate.run_unit(("ring", 0.6, 0), **kwargs)
    b = evaluate.run_unit(("ring", 0.6, 0), **kwargs)
    assert a.rows == b.rows
    assert a.prune_disconnected == b.prune_disconnected


def test_end_to_end_quick_run(tmp_path):
    out = str(tmp_path / "out")
    agg = evaluate.main(
        [
            "--quick",
            "--n-topologies", "1",
            "--n-channels", "1",
            "--steps", "50",
            "--out-dir", out,
        ]
    )
    assert len(agg) == 3 * 3 * 3  # layouts x ks x routers
    for name in (
        "results_per_sim.csv",
        "results_summary.csv",
        "drop_locations.csv",
        "calibration.png",
        "pdr.png",
        "delay_decomposition.png",
        "unreachable.png",
    ):
        assert os.path.exists(os.path.join(out, name)), name
    with open(os.path.join(out, "results_per_sim.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 27  # one sim per cell with 1 topology x 1 realization
    assert all(float(r["n_emitted"]) == 36 * 50 for r in rows)
