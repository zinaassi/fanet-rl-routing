"""Greedy vs Dijkstra comparison protocol (restricted_pdr, paired sign test).

Run from the repo root:

    python -m stage1.compare --layout random --k 0.4 --jobs 16
    python -m stage1.compare --layout random          # --k defaults to frozen k

Protocol (finalized in an external planning session):
  * Greedy vs Dijkstra ONLY.
  * NO QUEUES (unlimited transmissions per step) — the comparison isolates
    pure channel x routing; restricted_pdr is then load-independent.
  * The TOPOLOGY is the unit of analysis. For each of COMPARE_N_TOPOLOGIES
    topologies, both routers run on the SAME graph over
    COMPARE_N_CHANNEL_REALIZATIONS channel seeds (common random numbers);
    restricted_pdr is averaged over channels separately per router, then
    delta_i = mean_restricted_pdr_dijkstra_i - mean_restricted_pdr_greedy_i.
  * Sign test over the 1000 deltas + a normal-approximation CI on the mean.

restricted_pdr (metrics.py) is the PRIMARY comparison metric. Per-router
global PDR and %-unreachable are printed as CONTEXT only, not as the
comparison.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

from . import config, metrics, sim, world
from .routing import make_router, routing_table

log = logging.getLogger("stage1.compare")

# Seed streams: reuse evaluate's topology stream (0) so compare and evaluate
# see identical topologies; use a disjoint channel stream (evaluate=1,
# calibrate=2, compare=3). The channel seed is NOT keyed by router, so both
# routers see common random numbers within a (topology, channel) cell.
_TOPOLOGY_STREAM = 0
_COMPARE_CHANNEL_STREAM = 3

_ROUTERS: Tuple[str, str] = ("greedy", "dijkstra")  # this comparison, exactly


@dataclass(frozen=True)
class TopoResult:
    """Per-topology means (over channel seeds), separately per router."""

    topology: int
    restricted_pdr: Dict[str, float]
    global_pdr: Dict[str, float]
    unreachable_m: Dict[str, float]


def run_topology(
    t: int, layout: str, k: float, n_channels: int, n_steps: int, base_seed: int
) -> TopoResult:
    """Both routers on topology ``t``'s graph, averaged over channel seeds."""
    w = world.build_world(layout, k, (base_seed, _TOPOLOGY_STREAM, t))
    graph = w.graph
    tables = {r: routing_table(make_router(r), graph) for r in _ROUTERS}

    rp: Dict[str, List[float]] = {r: [] for r in _ROUTERS}
    gp: Dict[str, List[float]] = {r: [] for r in _ROUTERS}
    um: Dict[str, List[float]] = {r: [] for r in _ROUTERS}
    for c in range(n_channels):
        seed = np.random.SeedSequence((base_seed, _COMPARE_CHANNEL_STREAM, t, c))
        for r in _ROUTERS:
            # Fresh generator from the SAME seed per router -> identical U(0,1)
            # stream (common random numbers), so delta reflects the routing
            # decision rather than channel noise.
            rng = np.random.default_rng(seed)
            res = sim.run_sim(graph, tables[r], rng, n_steps, max_tx_per_step=None)
            rp[r].append(metrics.restricted_pdr(res, graph))
            m = metrics.sim_metrics(res, graph, tables[r])
            gp[r].append(m["pdr_global"])
            um[r].append(m["unreachable_frac_m"])

    def mean(xs: List[float]) -> float:
        return float(np.nanmean(xs)) if len(xs) else float("nan")

    return TopoResult(
        topology=t,
        restricted_pdr={r: mean(rp[r]) for r in _ROUTERS},
        global_pdr={r: mean(gp[r]) for r in _ROUTERS},
        unreachable_m={r: mean(um[r]) for r in _ROUTERS},
    )


@dataclass(frozen=True)
class ComparisonReport:
    layout: str
    k: float
    n_channels: int
    n_topologies_requested: int
    n_valid: int
    n_dropped_nan: int
    wins: int
    ties: int
    losses: int
    sign_test_p: float
    mean_delta: float
    ci_low: float
    ci_high: float
    threshold: float
    # context
    global_pdr: Dict[str, float]
    unreachable_m: Dict[str, float]


def analyze(results: Sequence[TopoResult], layout: str, k: float,
            n_channels: int) -> ComparisonReport:
    """Sign test + normal-approx CI on per-topology restricted_pdr deltas."""
    ordered = sorted(results, key=lambda r: r.topology)
    rd = np.array([r.restricted_pdr["dijkstra"] for r in ordered], dtype=float)
    rg = np.array([r.restricted_pdr["greedy"] for r in ordered], dtype=float)
    delta = rd - rg

    valid = ~np.isnan(delta)
    d = delta[valid]
    n = d.size
    n_dropped = int((~valid).sum())

    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    losses = int((d < 0).sum())

    # Sign test: is Dijkstra's win-rate over topologies > 0.5?
    p = binomtest(wins, n, p=0.5, alternative="greater").pvalue if n > 0 else float("nan")

    mean_d = float(d.mean()) if n else float("nan")
    # Normal approximation for the CI is the deliberate choice here: verified
    # adequate at this N in planning simulations; a bootstrap is a possible
    # later upgrade ONLY if effect sizes ever become small.
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    ci_low, ci_high = mean_d - 1.96 * se, mean_d + 1.96 * se

    def ctx(field: str) -> Dict[str, float]:
        return {
            r: float(np.nanmean([getattr(x, field)[r] for x in ordered]))
            for r in _ROUTERS
        }

    return ComparisonReport(
        layout=layout, k=k, n_channels=n_channels,
        n_topologies_requested=len(ordered), n_valid=n, n_dropped_nan=n_dropped,
        wins=wins, ties=ties, losses=losses, sign_test_p=p,
        mean_delta=mean_d, ci_low=ci_low, ci_high=ci_high,
        threshold=config.PRACTICAL_SIGNIFICANCE_THRESHOLD,
        global_pdr=ctx("global_pdr"), unreachable_m=ctx("unreachable_m"),
    )


def print_report(rep: ComparisonReport) -> None:
    print(f"\n=== Greedy vs Dijkstra — restricted_pdr comparison "
          f"(layout={rep.layout}, k={rep.k:g}, no queues) ===")
    print(f"Unit of analysis: topology.  N = {rep.n_valid} valid deltas"
          + (f" ({rep.n_dropped_nan} dropped: no reachable M-drone)"
             if rep.n_dropped_nan else "")
          + f".  {rep.n_channels} channel seeds/topology, averaged per router.")
    print()
    hdr = f"{'wins/N':>12} {'sign-test p':>13} {'mean delta':>11} {'95% CI':>20} {'vs threshold':>26}"
    print(hdr)
    print("-" * len(hdr))
    thr = rep.threshold
    if rep.ci_low > thr:
        verdict = f"CI>thr ({thr:.2f}): practical"
    elif rep.mean_delta > thr:
        verdict = f"mean>thr ({thr:.2f}); CI spans"
    else:
        verdict = f"below thr ({thr:.2f})"
    print(f"{f'{rep.wins}/{rep.n_valid}':>12} "
          f"{rep.sign_test_p:>13.3g} {rep.mean_delta:>11.4f} "
          f"{f'[{rep.ci_low:.4f}, {rep.ci_high:.4f}]':>20} {verdict:>26}")
    print(f"   (wins={rep.wins}, ties={rep.ties}, losses={rep.losses}; "
          f"dijkstra delta = restricted_pdr_dijkstra - restricted_pdr_greedy)")

    print("\n--- context, NOT part of the comparison "
          "(per-router means over topologies) ---")
    print(f"{'router':>10} {'global PDR':>12} {'% M-unreachable':>18}")
    for r in _ROUTERS:
        print(f"{r:>10} {rep.global_pdr[r]:>12.4f} {100.0 * rep.unreachable_m[r]:>17.2f}%")
    print()


def _write_csv(rep: ComparisonReport, results: Sequence[TopoResult], out_dir: str,
               layout: str, k: float) -> str:
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"compare_{layout}_k{k:g}.csv")
    ordered = sorted(results, key=lambda r: r.topology)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["topology", "restricted_pdr_greedy", "restricted_pdr_dijkstra",
                     "delta", "global_pdr_greedy", "global_pdr_dijkstra",
                     "unreach_m_greedy", "unreach_m_dijkstra"])
        for x in ordered:
            g, dj = x.restricted_pdr["greedy"], x.restricted_pdr["dijkstra"]
            wr.writerow([x.topology, f"{g:.6g}", f"{dj:.6g}", f"{dj - g:.6g}",
                         f"{x.global_pdr['greedy']:.6g}", f"{x.global_pdr['dijkstra']:.6g}",
                         f"{x.unreachable_m['greedy']:.6g}", f"{x.unreachable_m['dijkstra']:.6g}"])
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m stage1.compare",
        description="Greedy vs Dijkstra restricted_pdr comparison (no queues).",
    )
    p.add_argument("--layout", required=True, choices=config.LAYOUTS)
    p.add_argument("--k", type=float, default=config.K_SWEEP[0],
                   help=f"channel steepness (default: frozen config k = {config.K_SWEEP[0]:g})")
    p.add_argument("--n-topologies", type=int, default=config.COMPARE_N_TOPOLOGIES)
    p.add_argument("--n-channels", type=int, default=config.COMPARE_N_CHANNEL_REALIZATIONS)
    p.add_argument("--steps", type=int, default=config.N_STEPS)
    p.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    p.add_argument("--out-dir", default=config.OUT_DIR + "/compare")
    p.add_argument("--jobs", type=int, default=1)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> ComparisonReport:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    log.info(
        "compare greedy vs dijkstra: layout=%s k=%g, %d topologies x %d channels "
        "x %d steps, NO QUEUES (jobs=%d)",
        args.layout, args.k, args.n_topologies, args.n_channels, args.steps, args.jobs,
    )
    worker = partial(
        run_topology, layout=args.layout, k=args.k,
        n_channels=args.n_channels, n_steps=args.steps, base_seed=args.base_seed,
    )
    results: List[TopoResult] = []
    started = time.monotonic()
    topos = range(args.n_topologies)
    if args.jobs <= 1:
        for i, t in enumerate(topos, 1):
            results.append(worker(t))
            _progress(i, args.n_topologies, started)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(worker, t) for t in topos]
            for i, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                _progress(i, args.n_topologies, started)

    rep = analyze(results, args.layout, args.k, args.n_channels)
    path = _write_csv(rep, results, args.out_dir, args.layout, args.k)
    log.info("wrote per-topology deltas to %s", path)
    print_report(rep)
    return rep


def _progress(done: int, total: int, started: float) -> None:
    if done == total or done % max(1, total // 10) == 0:
        log.info("topologies %d/%d done (%.0f s)", done, total, time.monotonic() - started)


if __name__ == "__main__":
    main()
