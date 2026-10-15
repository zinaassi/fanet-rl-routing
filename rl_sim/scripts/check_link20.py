"""
check_link20.py — The non-learned rules at a tighter neighbour cutoff (one-off).

``config.LINK_MAX_LOSS`` decides how bad a link may be and still count as a
link. At the configured 0.5 the edge is 350 m. This runs the Phase-1a grid at
0.2 as well, where the edge is about 283 m, and at 0.5 for comparison on the
same seeds.

Only the NEIGHBOUR CUTOFF changes. The loss curve is untouched: still 10% at
250 m and 50% at 350 m. A link at 300 m still loses 26.8% of what is sent on
it — at the 0.2 cutoff it is simply not a link any more.

The connected-layout filter stays ON and uses the same cutoff, so every
M-drone still has a route to the GS. A tighter cutoff rejects more layouts, so
a seed can draw a different accepted layout than it does at 0.5; the script
counts how many do.

Everything it writes goes under ``out/check_link20/``. No existing CSV or
figure is read or written, and nothing is trained.

Writes:
    out/check_link20/runs.csv       one row per run, both cutoffs
    out/check_link20/geometry.csv   neighbours and GS links per layout
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.evaluate_1b_large import sign_test
from scripts.stamp import csv_comment
from scripts.train_1b import run_once

OUT_SUBDIR = "check_link20"

CUTOFFS = (0.2, 0.5)
SEEDS = tuple(range(1, 21))
LOADS = (1000, 500, 200, 100)
RULES = ("greedy_classic", "rate", "random")

CAUSES = ("channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit")
COMPARISONS = (("rate", "greedy_classic"), ("greedy_classic", "random"))


def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one run at one cutoff.

    Args:
        args: ``(cutoff, rule, load_ms, seed)``.

    Returns:
        That run's row.
    """
    cutoff, rule, load_ms, seed = args
    previous = config.LINK_MAX_LOSS
    config.LINK_MAX_LOSS = cutoff
    try:
        result = run_once(rule, seed, seed, load_ms, training=False)
    finally:
        # Restored so a --workers 1 run does not leave the parent's cutoff
        # wherever the last job happened to set it.
        config.LINK_MAX_LOSS = previous
    return {
        "link_max_loss": cutoff, "rule": rule, "load_ms": load_ms,
        "placement_seed": seed,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        "mean_hops": result["mean_hops"],
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def run_grid(workers: int) -> List[Dict[str, Any]]:
    """Every rule on every layout, at every load and both cutoffs.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    jobs = [(cutoff, rule, load, seed)
            for cutoff in CUTOFFS
            for load in LOADS
            for seed in SEEDS
            for rule in RULES]
    print(f"Running {len(jobs)} runs "
          f"({len(CUTOFFS)} cutoffs x {len(LOADS)} loads x {len(SEEDS)} "
          f"layouts x {len(RULES)} rules) ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [r for r in rows if not r["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — cutoff {row['link_max_loss']} "
                  f"{row['rule']} seed {row['placement_seed']}: "
                  f"{row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _geometry(seed: int, cutoff: float) -> Dict[str, Any]:
    """Measure one accepted layout at one cutoff."""
    from fanet_sim.envs.fanet_env import FANETEnv

    previous = config.LINK_MAX_LOSS
    config.LINK_MAX_LOSS = cutoff
    try:
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=seed, run_seed=seed)
        env.reset()
        positions = np.array([d.position for d in env.drones])
        neighbours = statistics.fmean(
            len(d.neighbors) for d in env.drones
        )
        m_direct = sum(
            1 for d in env.drones
            if d.drone_type == "M"
            and channel.are_connected(d.position, env.gs_position)
        )
        return {
            "link_max_loss": cutoff, "placement_seed": seed,
            "reach_m": channel.max_link_distance(cutoff),
            "mean_neighbours": neighbours,
            "m_direct_to_gs_pct": 100.0 * m_direct / config.NUM_M_DRONES,
            "placement_draws": env.placement_draws,
            "positions": positions,
        }
    finally:
        config.LINK_MAX_LOSS = previous


def geometry_rows(seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Measure every layout at both cutoffs, and flag the ones that moved."""
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        at = {cutoff: _geometry(seed, cutoff) for cutoff in CUTOFFS}
        moved = not np.array_equal(at[0.2]["positions"], at[0.5]["positions"])
        for cutoff in CUTOFFS:
            record = dict(at[cutoff])
            record.pop("positions")
            record["layout_differs_from_50pct"] = moved
            rows.append(record)
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def per_layout(rows: Sequence[Dict[str, Any]], cutoff: float, rule: str,
               load_ms: int, key: str = "total_loss") -> List[float]:
    """One value per layout, in SEEDS order."""
    out: List[float] = []
    for seed in SEEDS:
        vals = [r[key] for r in rows
                if r["link_max_loss"] == cutoff and r["rule"] == rule
                and r["load_ms"] == load_ms and r["placement_seed"] == seed
                and r[key] is not None]
        out.append(statistics.fmean(vals))
    return out


def _fmt_p(p: float) -> str:
    """Format a p-value without pretending to more precision than it has."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def print_totals(rows: Sequence[Dict[str, Any]]) -> None:
    """Table (a): totals and causes, per cutoff and load."""
    width = 116
    print()
    print(f"  (a) {len(SEEDS)} layouts — mean % of packets created")
    print("  " + "-" * width)
    print(f"  {'cutoff':>8}{'reach':>9}{'load':>9}{'rule':>17}{'total':>9}"
          f"{'std':>7}{'channel':>9}{'queue_full':>12}{'no_route':>10}"
          f"{'dead_end':>10}{'ttl':>7}{'hop_lim':>9}")
    print("  " + "-" * width)
    for cutoff in CUTOFFS:
        reach = channel.max_link_distance(cutoff)
        for load in LOADS:
            for rule in RULES:
                tot = per_layout(rows, cutoff, rule, load)
                c = [statistics.fmean(
                    per_layout(rows, cutoff, rule, load, k)
                ) for k in CAUSES]
                print(f"  {cutoff:>8.1f}{reach:>8.0f} m{load:>6} ms"
                      f"{rule:>17}{statistics.fmean(tot)*100:>8.2f}%"
                      f"{statistics.stdev(tot)*100:>7.2f}"
                      f"{c[0]*100:>8.2f}%{c[1]*100:>11.2f}%{c[2]*100:>9.2f}%"
                      f"{c[3]*100:>9.2f}%{c[4]*100:>6.2f}%{c[5]*100:>8.2f}%")
            print("  " + "-" * width)


def print_paired(rows: Sequence[Dict[str, Any]], cutoff: float) -> None:
    """Table (b): paired comparisons at one cutoff."""
    width = 84
    print()
    print(f"  (b) Paired on the same layouts, cutoff {cutoff:.0%} "
          f"(reach {channel.max_link_distance(cutoff):.0f} m)")
    print("      negative = the first rule loses fewer packets")
    print("  " + "-" * width)
    print(f"  {'load':>8}{'comparison':>30}{'mean diff':>12}{'std':>9}"
          f"{'wins':>10}{'p (sign)':>12}")
    print("  " + "-" * width)
    for load in LOADS:
        for first, second in COMPARISONS:
            a = per_layout(rows, cutoff, first, load)
            b = per_layout(rows, cutoff, second, load)
            diffs = [x - y for x, y in zip(a, b)]
            wins, _, p = sign_test(diffs)
            print(f"  {load:>5} ms{f'{first} - {second}':>30}"
                  f"{statistics.fmean(diffs)*100:>11.2f}%"
                  f"{statistics.stdev(diffs)*100:>8.2f}%"
                  f"{f'{wins} / {len(diffs)}':>10}{_fmt_p(p):>12}")
    print("  " + "-" * width)


def print_hops(rows: Sequence[Dict[str, Any]]) -> None:
    """Table (c): hops per delivered packet."""
    width = 72
    print()
    print("  (c) Hops per delivered packet")
    print("  " + "-" * width)
    header = f"  {'cutoff':>8}{'load':>9}"
    for rule in RULES:
        header += f"{rule:>17}"
    print(header)
    print("  " + "-" * width)
    for cutoff in CUTOFFS:
        for load in LOADS:
            line = f"  {cutoff:>8.1f}{load:>6} ms"
            for rule in RULES:
                line += f"{statistics.fmean(per_layout(rows, cutoff, rule, load, 'mean_hops')):>17.2f}"
            print(line)
        print("  " + "-" * width)


def print_geometry(rows: Sequence[Dict[str, Any]]) -> None:
    """Tables (d) and (e): geometry, and how many layouts moved."""
    width = 76
    print()
    print(f"  (d) Geometry of the accepted layouts, {len(SEEDS)} seeds")
    print("  " + "-" * width)
    print(f"  {'cutoff':>8}{'reach':>10}{'mean neighbours':>20}"
          f"{'M direct to GS':>18}{'draws':>10}")
    print("  " + "-" * width)
    for cutoff in CUTOFFS:
        at = [r for r in rows if r["link_max_loss"] == cutoff]
        print(f"  {cutoff:>8.1f}{channel.max_link_distance(cutoff):>8.0f} m"
              f"{statistics.fmean(r['mean_neighbours'] for r in at):>20.2f}"
              f"{statistics.fmean(r['m_direct_to_gs_pct'] for r in at):>17.1f}%"
              f"{statistics.fmean(r['placement_draws'] for r in at):>10.2f}")
    print("  " + "-" * width)

    moved = {r["placement_seed"] for r in rows
             if r["layout_differs_from_50pct"]}
    print()
    print(f"  (e) Seeds drawing a DIFFERENT layout at 20% than at 50%: "
          f"{len(moved)} of {len(SEEDS)}")
    if moved:
        print(f"      {sorted(moved)}")


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              fields: Sequence[str]) -> None:
    """Write *rows* with the settings stamp as a leading comment.

    The stamp reports the CONFIGURED cutoff, which is what the rest of the
    project runs at. This file covers two, so the note says so and the
    link_max_loss column says which each row used. The cutoff is not faked
    in config for the stamp: the stamp derives the reach from it, so a fake
    value would produce a wrong reach rather than an obviously odd one.
    """
    cutoffs = "/".join(f"{c:g}" for c in CUTOFFS)
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(
            loads_ms=list(LOADS), placements=len(SEEDS),
            extra=f"link-cutoff check: rows ran at LINK_MAX_LOSS {cutoffs}, "
                  f"see the link_max_loss column; the stamp's edge is the "
                  f"configured default",
        ))
        writer = csv.DictWriter(handle, fieldnames=list(fields),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def main() -> None:
    """Run both cutoffs and print every table."""
    parser = argparse.ArgumentParser(
        description="Phase-1a grid at a 20% link cutoff, beside the usual 50%."
    )
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the outputs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    args = parser.parse_args()

    target = os.path.join(args.out_dir, OUT_SUBDIR)
    os.makedirs(target, exist_ok=True)

    print(f"Link cutoffs {CUTOFFS}; the loss curve itself is unchanged.")
    for cutoff in CUTOFFS:
        print(f"  cutoff {cutoff:.0%} -> reach "
              f"{channel.max_link_distance(cutoff):.1f} m")

    rows = run_grid(args.workers)
    print_totals(rows)
    print_paired(rows, 0.2)
    print_hops(rows)

    geometry = geometry_rows(SEEDS)
    print_geometry(geometry)

    print()
    write_csv(os.path.join(target, "runs.csv"), rows,
              ["link_max_loss", "rule", "load_ms", "placement_seed",
               "total_loss", *CAUSES, "mean_hops", "views_agree"])
    write_csv(os.path.join(target, "geometry.csv"), geometry,
              ["link_max_loss", "placement_seed", "reach_m",
               "mean_neighbours", "m_direct_to_gs_pct", "placement_draws",
               "layout_differs_from_50pct"])


if __name__ == "__main__":
    main()
