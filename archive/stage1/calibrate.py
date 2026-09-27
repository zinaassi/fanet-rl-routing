"""RANGE_M / P_sens calibration sweep — run FIRST, before any evaluation.

RANGE_M is our own assumption, not given by the project spec, so it is
chosen deliberately here rather than guessed: too small a range makes the
network too sparse to route on at all (routing quality cannot matter if
there is barely a network); too large makes it fully connected regardless
of topology (router and C-drone-layout comparisons both become meaningless
if every layout already gives ~100% PDR).

For each candidate range (config.CAL_RANGES_M) this sweep derives P_sens
(M = 0 exactly at that range, see ``channel.p_sens_dbm``), runs
CAL_N_TOPOLOGIES x CAL_N_CHANNEL_REALIZATIONS episodes per C-drone layout
with the GLOBAL-INFORMATION DIJKSTRA ROUTER ONLY at k = CAL_K, and prints:

    RANGE_M | mean PDR (ring/grid/random) | layout spread | frac severely-disconnected

It then recommends the SMALLEST range whose overall mean PDR sits in a
middle band (~30-80%), whose layout spread exceeds 5 percentage points
(topology must actually matter), and whose fraction of severely
disconnected topologies (<10% of drones with any path to the GS) is below
5%. The recommendation is NOT auto-applied: a human must confirm it by
setting RANGE_M in stage1/config.py, after which it is FROZEN for the
entire project (Stage 2, Stage 3, Algorithm 2 import it from config.py).

Usage (from the repo root):

    python -m stage1.calibrate --jobs 16
    python -m stage1.calibrate --quick          # tiny smoke sweep
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import channel, config, metrics, sim, world
from .routing import DijkstraRouter, routing_table
from .world import drones_reaching_gs

log = logging.getLogger("stage1.calibrate")

_CAL_CHANNEL_STREAM = 2  # disjoint from evaluate's topology(0)/channel(1) streams
_SEVERE_CONNECTIVITY = 0.10  # a topology is severely disconnected below this

CalUnit = Tuple[float, str, int]  # (range_m, layout, topology index)


@dataclass(frozen=True)
class CalUnitResult:
    range_m: float
    layout: str
    topology: int
    connected_frac: float          # drones with any path to GS / N_DRONES
    pdrs: Tuple[float, ...]        # global PDR per channel realization


@dataclass(frozen=True)
class RangeStats:
    """Aggregates for one candidate range, across the three layouts."""

    range_m: float
    p_sens_dbm: float
    layout_pdr: Dict[str, float]   # layout -> mean global PDR
    overall_pdr: float             # mean of the three layout means
    spread: float                  # max - min of the layout means
    severe_frac: float             # fraction of (layout, topology) graphs
                                   # with <10% of drones connected to GS


def run_cal_unit(
    unit: CalUnit,
    base_seed: int,
    n_channels: int,
    n_steps: int,
    k: float,
    emit_period: int = config.EMIT_PERIOD_STEPS,
    max_tx_per_step: int | None = 1,
) -> CalUnitResult:
    """One (range, layout, topology): dijkstra-only sims over channel seeds."""
    range_m, layout, t = unit
    w = world.build_world(layout, k, (base_seed, 0, t), range_m=range_m)
    connected = len(drones_reaching_gs(w.graph)) / config.N_DRONES
    table = routing_table(DijkstraRouter(), w.graph)
    range_idx = int(range_m)
    layout_idx = config.LAYOUTS.index(layout)
    pdrs: List[float] = []
    for c in range(n_channels):
        seed = np.random.SeedSequence(
            (base_seed, _CAL_CHANNEL_STREAM, range_idx, layout_idx, t, c)
        )
        res = sim.run_sim(
            w.graph, table, np.random.default_rng(seed), n_steps,
            emit_period=emit_period, max_tx_per_step=max_tx_per_step,
        )
        resolved = float(res.resolved.sum())
        pdrs.append(float(res.delivered.sum()) / resolved if resolved else float("nan"))
    return CalUnitResult(
        range_m=range_m,
        layout=layout,
        topology=t,
        connected_frac=connected,
        pdrs=tuple(pdrs),
    )


def aggregate_range(range_m: float, results: List[CalUnitResult]) -> RangeStats:
    """Fold all unit results for one candidate range into RangeStats."""
    layout_pdr: Dict[str, float] = {}
    for layout in config.LAYOUTS:
        vals = [p for r in results if r.layout == layout for p in r.pdrs]
        layout_pdr[layout] = float(np.nanmean(vals)) if vals else float("nan")
    means = list(layout_pdr.values())
    severe = [r.connected_frac < _SEVERE_CONNECTIVITY for r in results]
    return RangeStats(
        range_m=range_m,
        p_sens_dbm=channel.p_sens_dbm(range_m),
        layout_pdr=layout_pdr,
        overall_pdr=float(np.nanmean(means)),
        spread=float(np.nanmax(means) - np.nanmin(means)),
        severe_frac=float(np.mean(severe)),
    )


def recommend(stats: Sequence[RangeStats]) -> Tuple[Optional[RangeStats], List[str]]:
    """Smallest range clearing the calibration criteria, with justification.

    Criteria: overall mean PDR in a middle band (preferred 30-80%, relaxed
    to 20-90% if nothing passes), layout spread > 5 points, severely
    disconnected fraction < 5%.
    """
    ordered = sorted(stats, key=lambda s: s.range_m)
    for lo, hi, band_note in ((0.30, 0.80, "preferred 30-80%"), (0.20, 0.90, "relaxed 20-90%")):
        for s in ordered:
            if s.spread > 0.05 and s.severe_frac < 0.05 and lo <= s.overall_pdr <= hi:
                why = [
                    f"RANGE_M = {s.range_m:.0f} m (P_sens = {s.p_sens_dbm:.1f} dBm) is the "
                    f"smallest candidate clearing all criteria:",
                    f"  - overall mean PDR {s.overall_pdr:.1%} sits in the middle band ({band_note}):"
                    " routable, but far from saturated",
                    f"  - layout spread {s.spread * 100:.1f} points > 5: C-drone placement"
                    " actually matters at this range",
                    f"  - severely disconnected topologies {s.severe_frac:.1%} < 5%:"
                    " healthy networks, not mostly-isolated drones",
                ]
                return s, why
    return None, [
        "No candidate range cleared the criteria (middle-band PDR, spread > 5 pts, "
        "severe disconnection < 5%). Inspect the table: either widen the candidate "
        "set or revisit k / the traffic load before freezing RANGE_M.",
    ]


def print_table(stats: Sequence[RangeStats]) -> None:
    print("\nRange calibration (dijkstra only, k = as configured below)")
    print(
        f"{'RANGE_M':>8} {'P_sens':>8} | {'ring':>6} {'grid':>6} {'random':>6} "
        f"| {'spread':>6} | {'frac severely-disconnected':>27}"
    )
    print("-" * 82)
    for s in sorted(stats, key=lambda x: x.range_m):
        print(
            f"{s.range_m:>8.0f} {s.p_sens_dbm:>7.1f}  |"
            f" {s.layout_pdr['ring']:>6.3f} {s.layout_pdr['grid']:>6.3f}"
            f" {s.layout_pdr['random']:>6.3f} | {s.spread:>6.3f} | {s.severe_frac:>27.3f}"
        )
    print()


def print_ploss_sanity(range_m: float, ks: Sequence[float] = config.K_SWEEP) -> None:
    """p_loss(d) spot table so a degenerate k sweep is visible at a glance."""
    fracs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
    print(f"p_loss(d) at RANGE_M = {range_m:.0f} m (rows: k, cols: d as % of range)")
    print(f"{'k':>6} | " + " ".join(f"{f:>7.0%}" for f in fracs))
    for k in ks:
        vals = [float(channel.p_loss(f * range_m, k, range_m)) for f in fracs]
        print(f"{k:>6g} | " + " ".join(f"{v:>7.3f}" for v in vals))
    print()


def _write_csv(stats: Sequence[RangeStats], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["range_m", "p_sens_dbm", "pdr_ring", "pdr_grid", "pdr_random",
             "overall_pdr", "layout_spread", "severe_disconnected_frac"]
        )
        for s in sorted(stats, key=lambda x: x.range_m):
            writer.writerow(
                [f"{s.range_m:g}", f"{s.p_sens_dbm:.2f}"]
                + [f"{s.layout_pdr[l]:.6g}" for l in config.LAYOUTS]
                + [f"{s.overall_pdr:.6g}", f"{s.spread:.6g}", f"{s.severe_frac:.6g}"]
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m stage1.calibrate",
        description="RANGE_M calibration sweep (run before any evaluation).",
    )
    p.add_argument("--ranges", nargs="+", type=float, default=list(config.CAL_RANGES_M))
    p.add_argument("--k", type=float, default=config.CAL_K,
                   help=f"channel steepness for the sweep (default {config.CAL_K})")
    p.add_argument("--n-topologies", type=int, default=None,
                   help=f"default {config.CAL_N_TOPOLOGIES}")
    p.add_argument("--n-channels", type=int, default=None,
                   help=f"default {config.CAL_N_CHANNEL_REALIZATIONS}")
    p.add_argument("--steps", type=int, default=None,
                   help=f"default {config.N_STEPS}")
    p.add_argument("--emit-period", type=int, default=config.EMIT_PERIOD_STEPS,
                   help="each M-drone emits 1 packet every N steps, staggered "
                        f"by id (default {config.EMIT_PERIOD_STEPS}; 3 = 1/3 load)")
    p.add_argument("--no-queues", action="store_true",
                   help="unlimited transmissions per drone per step: queues never "
                        "build, so the sweep measures pure channel x routing "
                        "(load-independent)")
    p.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    p.add_argument("--out-dir", default=config.OUT_DIR_CALIBRATION)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--quick", action="store_true",
                   help="smoke sweep: 2 topologies x 2 realizations x 200 steps")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Tuple[List[RangeStats], Optional[RangeStats]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    n_topologies = args.n_topologies or (2 if args.quick else config.CAL_N_TOPOLOGIES)
    n_channels = args.n_channels or (2 if args.quick else config.CAL_N_CHANNEL_REALIZATIONS)
    n_steps = args.steps or (config.QUICK_N_STEPS if args.quick else config.N_STEPS)

    units: List[CalUnit] = [
        (r, layout, t)
        for r in args.ranges
        for layout in config.LAYOUTS
        for t in range(n_topologies)
    ]
    log.info(
        "calibration sweep: ranges %s, k=%g, %d topologies x %d realizations x %d steps, "
        "emit period %d (dijkstra only, %d units, jobs=%d)",
        [f"{r:g}" for r in args.ranges], args.k, n_topologies, n_channels, n_steps,
        args.emit_period, len(units), args.jobs,
    )
    worker = partial(
        run_cal_unit,
        base_seed=args.base_seed,
        n_channels=n_channels,
        n_steps=n_steps,
        k=args.k,
        emit_period=args.emit_period,
        max_tx_per_step=None if args.no_queues else 1,
    )
    results: List[CalUnitResult] = []
    started = time.monotonic()
    if args.jobs <= 1:
        for i, unit in enumerate(units, 1):
            results.append(worker(unit))
            _progress(i, len(units), started)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(worker, u) for u in units]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                _progress(i, len(units), started)

    stats = [
        aggregate_range(r, [res for res in results if res.range_m == r])
        for r in args.ranges
    ]
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "calibration_sweep.csv")
    _write_csv(stats, csv_path)
    log.info("wrote %s", csv_path)

    print_table(stats)
    choice, why = recommend(stats)
    for line in why:
        print(line)
    if choice is not None:
        print_ploss_sanity(choice.range_m)
        print(
            "NOT auto-applied: confirm by setting RANGE_M in stage1/config.py. "
            "Once set it is frozen for the entire project (see the comment there)."
        )
    return stats, choice


def _progress(done: int, total: int, started: float) -> None:
    if done == total or done % max(1, total // 10) == 0:
        log.info("units %d/%d done (%.0f s elapsed)", done, total, time.monotonic() - started)


if __name__ == "__main__":
    main()
