"""Evaluation harness: full {layout} x {k} x {router} grid with nested seeding.

Usage (from the repo root):

    python -m stage1.evaluate --quick            # small smoke-test grid
    python -m stage1.evaluate --jobs 8           # full run, 8 worker processes
    python -m stage1.evaluate --layouts ring grid --ks 0.6 --routers dijkstra

Seeding is fully deterministic and independent of --jobs: the topology seed
(positions) is derived from (base_seed, topology index) and the channel seed
(the U(0,1) transmission draws) from (base_seed, topology, realization,
layout, k, router).

Outputs (in --out-dir): results_per_sim.csv, results_summary.csv,
drop_locations.csv, and summary PNG figures.
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

from . import config, metrics, plots, sim, world
from .routing import make_router, routing_table

log = logging.getLogger("stage1.evaluate")

# Stream tags keep topology and channel seed sequences disjoint.
_TOPOLOGY_STREAM = 0
_CHANNEL_STREAM = 1

METRIC_KEYS: Tuple[str, ...] = (
    "n_emitted",
    "n_delivered",
    "n_dropped_channel",
    "n_dropped_no_route",
    "n_in_flight",
    "pdr_global",
    "pdr_routed",
    "unreachable_frac_all",
    "unreachable_frac_m",
    "mean_delay_steps",
    "mean_delay_ms",
    "mean_hops",
    "mean_queue_wait_steps",
    "max_queue_depth",
    "mean_queue_depth",
    "n_unstable_drones",
    "drop_frac_at_m",
    "drop_frac_at_c",
)

# Metrics aggregated into results_summary.csv (mean, between-topology std,
# within-topology std for each).
SUMMARY_KEYS: Tuple[str, ...] = (
    "pdr_global",
    "pdr_routed",
    "unreachable_frac_all",
    "unreachable_frac_m",
    "mean_delay_ms",
    "mean_hops",
    "mean_queue_wait_steps",
    "max_queue_depth",
    "mean_queue_depth",
    "n_unstable_drones",
    "drop_frac_at_m",
    "drop_frac_at_c",
    "prune_disconnected_count",
)

CellKey = Tuple[str, float, str]  # (layout, k, router)


@dataclass(frozen=True)
class UnitResult:
    """Everything produced by one (layout, k, topology) work unit."""

    layout: str
    k: float
    topology: int
    prune_disconnected: Tuple[int, ...]
    rows: List[dict]  # one per (router, channel realization)
    drop_hist: Dict[Tuple[str, str, str], int]  # (router, node kind, reason) -> count


def _drop_class_counts(result: sim.SimResult, graph) -> Dict[Tuple[str, str], int]:
    """Drop counts by (kind of dropping node, drop reason) for one sim."""
    counts: Dict[Tuple[str, str], int] = {}
    reason_of = {sim.DROPPED_CHANNEL: "channel", sim.DROPPED_NO_ROUTE: "no_route"}
    for status, node in zip(result.status, result.end_node):
        reason = reason_of.get(int(status))
        if reason is None:
            continue
        kind = graph.nodes[int(node)]["kind"]
        counts[(kind, reason)] = counts.get((kind, reason), 0) + 1
    return counts


def run_unit(
    unit: Tuple[str, float, int],
    base_seed: int,
    routers: Sequence[str],
    n_channels: int,
    n_steps: int,
    range_m: Optional[float] = None,
    emit_period: int = config.EMIT_PERIOD_STEPS,
    max_tx_per_step: Optional[int] = 1,
) -> UnitResult:
    """Build one topology and run all routers x channel realizations on it.

    ``range_m`` overrides config.RANGE_M (candidate-range validation only);
    ``emit_period`` sets the offered load and ``max_tx_per_step`` the
    service rate (None = no queues; see sim.run_sim).
    """
    layout, k, t = unit
    layout_idx = config.LAYOUTS.index(layout)
    k_idx = config.K_SWEEP.index(k) if k in config.K_SWEEP else int(k * 1000)
    w = world.build_world(layout, k, (base_seed, _TOPOLOGY_STREAM, t), range_m=range_m)

    rows: List[dict] = []
    drop_hist: Dict[Tuple[str, str, str], int] = {}
    for r_idx, router_name in enumerate(routers):
        router = make_router(router_name)
        table = routing_table(router, w.graph)
        for c in range(n_channels):
            seed = np.random.SeedSequence(
                (base_seed, _CHANNEL_STREAM, t, c, layout_idx, k_idx, r_idx)
            )
            result = sim.run_sim(
                w.graph, table, np.random.default_rng(seed), n_steps,
                emit_period=emit_period, max_tx_per_step=max_tx_per_step,
            )
            row = {
                "layout": layout,
                "k": k,
                "router": router_name,
                "topology": t,
                "channel": c,
                "prune_disconnected_count": float(len(w.prune_disconnected)),
            }
            row.update(metrics.sim_metrics(result, w.graph, table))
            rows.append(row)
            for (kind, reason), n in _drop_class_counts(result, w.graph).items():
                key = (router_name, kind, reason)
                drop_hist[key] = drop_hist.get(key, 0) + n
    return UnitResult(
        layout=layout,
        k=k,
        topology=t,
        prune_disconnected=w.prune_disconnected,
        rows=rows,
        drop_hist=drop_hist,
    )


def _run_all_units(
    units: List[Tuple[str, float, int]],
    worker,
    jobs: int,
) -> List[UnitResult]:
    """Run all work units, in-process or via a process pool."""
    results: List[UnitResult] = []
    started = time.monotonic()
    if jobs <= 1:
        for i, unit in enumerate(units, 1):
            results.append(worker(unit))
            _progress(i, len(units), started)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(worker, u) for u in units]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                _progress(i, len(units), started)
    return results


def _progress(done: int, total: int, started: float) -> None:
    if done == total or done % max(1, total // 20) == 0:
        elapsed = time.monotonic() - started
        log.info("units %d/%d done (%.0f s elapsed)", done, total, elapsed)


def _aggregate_cells(
    rows: List[dict],
    n_topologies: int,
    n_channels: int,
) -> Dict[CellKey, Dict[str, metrics.AggStats]]:
    """Reshape per-sim rows into per-cell (topology x realization) aggregates."""
    keys = SUMMARY_KEYS
    raw: Dict[CellKey, Dict[str, np.ndarray]] = {}
    topo_index: Dict[CellKey, Dict[int, int]] = {}
    for row in rows:
        cell = (row["layout"], row["k"], row["router"])
        if cell not in raw:
            raw[cell] = {m: np.full((n_topologies, n_channels), np.nan) for m in keys}
            topo_index[cell] = {}
        tmap = topo_index[cell]
        t = tmap.setdefault(row["topology"], len(tmap))
        for m in keys:
            raw[cell][m][t, row["channel"]] = row[m]
    return {
        cell: {m: metrics.aggregate(arr) for m, arr in per_metric.items()}
        for cell, per_metric in raw.items()
    }


def _write_per_sim_csv(rows: List[dict], path: str) -> None:
    header = (
        ["layout", "k", "router", "topology", "channel", "prune_disconnected_count"]
        + list(METRIC_KEYS)
    )
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in sorted(
            rows, key=lambda r: (r["layout"], r["k"], r["router"], r["topology"], r["channel"])
        ):
            writer.writerow([_fmt(row[h]) for h in header])


def _write_summary_csv(
    agg: Dict[CellKey, Dict[str, metrics.AggStats]], path: str
) -> None:
    header = ["layout", "k", "router"]
    for m in SUMMARY_KEYS:
        header += [f"{m}_mean", f"{m}_between_std", f"{m}_within_std"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for cell in sorted(agg):
            row: List[str] = [str(cell[0]), _fmt(cell[1]), str(cell[2])]
            for m in SUMMARY_KEYS:
                s = agg[cell][m]
                row += [_fmt(s.mean), _fmt(s.between_std), _fmt(s.within_std)]
            writer.writerow(row)


def _write_drop_csv(results: List[UnitResult], path: str) -> None:
    """Aggregated drop-location histogram: counts by node kind and reason."""
    total: Dict[Tuple[str, float, str, str, str], int] = {}
    for res in results:
        for (router, kind, reason), n in res.drop_hist.items():
            key = (res.layout, res.k, router, kind, reason)
            total[key] = total.get(key, 0) + n
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layout", "k", "router", "node_kind", "reason", "n_drops"])
        for key in sorted(total):
            writer.writerow([key[0], _fmt(key[1]), key[2], key[3], key[4], total[key]])


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _print_headline(
    agg: Dict[CellKey, Dict[str, metrics.AggStats]],
    layouts: Sequence[str],
    ks: Sequence[float],
    routers: Sequence[str],
) -> None:
    """Per-layout global-PDR comparison table (expect direct < greedy < dijkstra)."""
    print("\nHeadline: global PDR, mean +/- between-topology std (in-flight excluded)")
    name_w = max(len(r) for r in routers) + 12
    header = f"{'layout':<8} {'k':<5}" + "".join(f"{r:<{name_w}}" for r in routers)
    ordering = " <= ".join(routers)
    print(header + f"ordering ({ordering})")
    print("-" * (len(header) + len(ordering) + 12))
    for layout in layouts:
        for k in ks:
            cells = [agg.get((layout, k, r), {}).get("pdr_global") for r in routers]
            fields = []
            means = []
            for s in cells:
                if s is None or np.isnan(s.mean):
                    fields.append(f"{'-':<{name_w}}")
                    means.append(np.nan)
                else:
                    b = f"+/-{s.between_std:.3f}" if np.isfinite(s.between_std) else ""
                    fields.append(f"{s.mean:.3f} {b:<{name_w - 6}}")
                    means.append(s.mean)
            finite = [m for m in means if np.isfinite(m)]
            ok = all(a <= b + 1e-12 for a, b in zip(finite, finite[1:]))
            verdict = "OK" if len(finite) == len(routers) and ok else (
                "VIOLATED" if len(finite) == len(routers) else "incomplete"
            )
            print(f"{layout:<8} {k:<5g}" + "".join(fields) + verdict)
    print()


def _log_prune_events(results: List[UnitResult]) -> None:
    for res in sorted(results, key=lambda r: (r.layout, r.k, r.topology)):
        if res.prune_disconnected:
            log.info(
                "prune disconnected drones (layout=%s k=%g topology=%d): %s "
                "(raw range alone would connect them)",
                res.layout, res.k, res.topology, list(res.prune_disconnected),
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m stage1.evaluate",
        description="Stage-1 classical routing baseline evaluation.",
    )
    p.add_argument("--layouts", nargs="+", default=list(config.LAYOUTS),
                   choices=config.LAYOUTS, help="C-drone layouts to evaluate")
    p.add_argument("--ks", nargs="+", type=float, default=list(config.K_SWEEP),
                   help="logistic steepness values")
    p.add_argument("--routers", nargs="+", default=list(config.ROUTERS),
                   choices=config.ROUTERS, help="routers to evaluate")
    p.add_argument("--n-topologies", type=int, default=None,
                   help=f"topology seeds per cell (default {config.N_TOPOLOGIES})")
    p.add_argument("--n-channels", type=int, default=None,
                   help=f"channel realizations per topology (default {config.N_CHANNEL_REALIZATIONS})")
    p.add_argument("--steps", type=int, default=None,
                   help=f"episode length in 100 ms steps (default {config.N_STEPS})")
    p.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    p.add_argument("--out-dir", default=config.OUT_DIR_EVALUATION)
    p.add_argument("--emit-period", type=int, default=config.EMIT_PERIOD_STEPS,
                   help="each M-drone emits 1 packet every N steps, staggered "
                        f"by id (default {config.EMIT_PERIOD_STEPS}; 3 = 1/3 load)")
    p.add_argument("--no-queues", action="store_true",
                   help="unlimited transmissions per drone per step: queues never "
                        "build, isolating pure channel x routing effects")
    p.add_argument("--range-m", type=float, default=None,
                   help="override config.RANGE_M (candidate-range validation only; "
                        "the frozen value in config.py is the source of truth)")
    p.add_argument("--jobs", type=int, default=1,
                   help="worker processes (results are identical for any value)")
    p.add_argument("--quick", action="store_true",
                   help=f"smoke test: {config.QUICK_N_TOPOLOGIES} topologies x "
                        f"{config.QUICK_N_CHANNEL_REALIZATIONS} realizations x "
                        f"{config.QUICK_N_STEPS} steps")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[CellKey, Dict[str, metrics.AggStats]]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    n_topologies = args.n_topologies or (
        config.QUICK_N_TOPOLOGIES if args.quick else config.N_TOPOLOGIES
    )
    n_channels = args.n_channels or (
        config.QUICK_N_CHANNEL_REALIZATIONS if args.quick else config.N_CHANNEL_REALIZATIONS
    )
    n_steps = args.steps or (config.QUICK_N_STEPS if args.quick else config.N_STEPS)

    units = [
        (layout, k, t)
        for layout in args.layouts
        for k in args.ks
        for t in range(n_topologies)
    ]
    n_sims = len(units) * len(args.routers) * n_channels
    log.info(
        "grid: %d layouts x %d ks x %d routers, %d topologies x %d realizations "
        "x %d steps -> %d sims (%d units, jobs=%d)",
        len(args.layouts), len(args.ks), len(args.routers),
        n_topologies, n_channels, n_steps, n_sims, len(units), args.jobs,
    )

    if args.range_m is not None:
        log.info("RANGE_M override: %g m (config value %g m untouched)",
                 args.range_m, config.RANGE_M)
    worker = partial(
        run_unit,
        base_seed=args.base_seed,
        routers=tuple(args.routers),
        n_channels=n_channels,
        n_steps=n_steps,
        range_m=args.range_m,
        emit_period=args.emit_period,
        max_tx_per_step=None if args.no_queues else 1,
    )
    results = _run_all_units(units, worker, args.jobs)
    _log_prune_events(results)

    rows = [row for res in results for row in res.rows]
    agg = _aggregate_cells(rows, n_topologies, n_channels)

    os.makedirs(args.out_dir, exist_ok=True)
    per_sim_path = os.path.join(args.out_dir, "results_per_sim.csv")
    summary_path = os.path.join(args.out_dir, "results_summary.csv")
    drops_path = os.path.join(args.out_dir, "drop_locations.csv")
    _write_per_sim_csv(rows, per_sim_path)
    _write_summary_csv(agg, summary_path)
    _write_drop_csv(results, drops_path)
    figure_paths = plots.write_summary_plots(
        agg, args.layouts, sorted(args.ks), args.routers, args.out_dir
    )
    log.info("wrote %s, %s, %s", per_sim_path, summary_path, drops_path)
    log.info("wrote figures: %s", ", ".join(sorted(figure_paths.values())))

    _print_headline(agg, args.layouts, sorted(args.ks), args.routers)
    return agg


if __name__ == "__main__":
    main()
