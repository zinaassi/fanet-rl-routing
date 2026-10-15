"""
check_classic_greedy.py — Textbook greedy beside the one we use (a one-off).

``greedy`` minimises link loss among the neighbours that make progress.
``greedy_classic`` is the rule the literature usually means: among the same
neighbours, take the largest geographic step towards the ground station and
ignore link quality. This script measures the second against the first, and
against the other rules, without touching any existing result.

Everything it writes goes under ``out/check_classic_greedy/``. No existing
CSV or figure is read for anything but the trained models, and nothing is
trained.

Writes:
    out/check_classic_greedy/phase1a.csv   the 20 layouts, four loads
    out/check_classic_greedy/test50.csv    the 50 TEST layouts, two loads
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

from fanet_sim import config
from scripts.evaluate_1b_large import TEST50_SEEDS, paired_t_test, sign_test
from scripts.stamp import csv_comment
from scripts.train_1b import run_once

OUT_SUBDIR = "check_classic_greedy"

PHASE1A_SEEDS = tuple(range(1, 21))
PHASE1A_LOADS = (1000, 500, 200, 100)
PHASE1A_RULES = ("greedy_classic", "greedy", "random")

TEST_LOADS = (200, 500)
TEST_RULES = ("greedy_classic", "greedy", "rate", "rl")

CAUSES = ("channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit")
EXTRAS = ("mean_hops", "mean_delay_ms")


def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one run.

    Args:
        args: ``(rule, load_ms, seed, init_seed, state_dict)``.

    Returns:
        That run's row.
    """
    rule, load_ms, seed, init_seed, state = args
    result = run_once(rule, seed, seed, load_ms, state_dict=state,
                      training=False)
    return {
        "rule": rule, "load_ms": load_ms, "placement_seed": seed,
        "init_seed": init_seed if rule == "rl" else -1,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        **{extra: result[extra] for extra in EXTRAS},
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def _run(jobs: Sequence[Tuple], workers: int, what: str) -> List[Dict[str, Any]]:
    """Run *jobs*, stopping if any run's two views disagree."""
    print(f"Running {len(jobs)} {what} runs ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]
    bad = [r for r in rows if not r["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — {row['rule']} "
                  f"seed {row['placement_seed']}: {row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def _states(out_dir: str, init_seeds: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Load the trained rl weights, which this script only reads."""
    import torch

    states: Dict[int, Dict[str, Any]] = {}
    for init_seed in init_seeds:
        path = os.path.join(out_dir, "models", config.LOOP_GUARD,
                            f"rl_seed{init_seed}.pt")
        if not os.path.exists(path):
            print(f"STOP: no trained model at {path}.")
            raise SystemExit(1)
        states[init_seed] = torch.load(path, weights_only=True)
    return states


def per_layout(rows: Sequence[Dict[str, Any]], rule: str, load_ms: int,
               seeds: Sequence[int], key: str = "total_loss") -> List[float]:
    """One value per layout, averaging rl over its init seeds."""
    out: List[float] = []
    for seed in seeds:
        vals = [r[key] for r in rows
                if r["rule"] == rule and r["load_ms"] == load_ms
                and r["placement_seed"] == seed and r[key] is not None]
        out.append(statistics.fmean(vals))
    return out


def _fmt_p(p: float) -> str:
    """Format a p-value without pretending to more precision than it has."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def print_phase1a(rows: Sequence[Dict[str, Any]]) -> None:
    """Table (a): the Phase-1a grid."""
    width = 104
    print()
    print(f"  (a) Phase-1a setup — {len(PHASE1A_SEEDS)} layouts, "
          f"mean % of packets created")
    print("  " + "-" * width)
    print(f"  {'load':>8}{'rule':>17}{'total':>9}{'std':>7}{'channel':>10}"
          f"{'queue_full':>12}{'no_route':>10}{'ttl':>8}{'hop_limit':>11}")
    print("  " + "-" * width)
    for load in PHASE1A_LOADS:
        for rule in PHASE1A_RULES:
            tot = per_layout(rows, rule, load, PHASE1A_SEEDS)
            c = [statistics.fmean(
                per_layout(rows, rule, load, PHASE1A_SEEDS, k)
            ) for k in ("channel", "queue_full", "no_route", "ttl",
                        "hop_limit")]
            print(f"  {load:>5} ms{rule:>17}"
                  f"{statistics.fmean(tot)*100:>8.2f}%"
                  f"{statistics.stdev(tot)*100:>7.2f}"
                  f"{c[0]*100:>9.2f}%{c[1]*100:>11.2f}%{c[2]*100:>9.2f}%"
                  f"{c[3]*100:>7.2f}%{c[4]*100:>10.2f}%")
        print("  " + "-" * width)

    print()
    print("  (a) greedy_classic against random, paired on the same layouts")
    print("  " + "-" * 72)
    print(f"  {'load':>8}{'mean diff':>13}{'std':>9}{'classic lower':>16}"
          f"{'p (sign)':>12}")
    print("  " + "-" * 72)
    for load in PHASE1A_LOADS:
        a = per_layout(rows, "greedy_classic", load, PHASE1A_SEEDS)
        b = per_layout(rows, "random", load, PHASE1A_SEEDS)
        diffs = [x - y for x, y in zip(a, b)]
        wins, _, p = sign_test(diffs)
        print(f"  {load:>5} ms{statistics.fmean(diffs)*100:>12.2f}%"
              f"{statistics.stdev(diffs)*100:>8.2f}%"
              f"{f'{wins} / {len(diffs)}':>16}{_fmt_p(p):>12}")
    print("  " + "-" * 72)


def print_test50(rows: Sequence[Dict[str, Any]]) -> None:
    """Tables (b) and (c): the 50 TEST layouts."""
    width = 92
    print()
    print(f"  (b) {len(TEST50_SEEDS)} TEST layouts — mean % of packets created")
    print("  " + "-" * width)
    print(f"  {'load':>8}{'rule':>17}{'total':>9}{'std':>7}{'channel':>10}"
          f"{'queue_full':>12}{'hops':>8}")
    print("  " + "-" * width)
    for load in TEST_LOADS:
        for rule in TEST_RULES:
            tot = per_layout(rows, rule, load, TEST50_SEEDS)
            print(f"  {load:>5} ms{rule:>17}"
                  f"{statistics.fmean(tot)*100:>8.2f}%"
                  f"{statistics.stdev(tot)*100:>7.2f}"
                  f"{statistics.fmean(per_layout(rows, rule, load, TEST50_SEEDS, 'channel'))*100:>9.2f}%"
                  f"{statistics.fmean(per_layout(rows, rule, load, TEST50_SEEDS, 'queue_full'))*100:>11.2f}%"
                  f"{statistics.fmean(per_layout(rows, rule, load, TEST50_SEEDS, 'mean_hops')):>8.2f}")
        print("  " + "-" * width)

    print()
    print("  (b) Paired rl - greedy_classic on the same 50 layouts")
    print("  " + "-" * 78)
    print(f"  {'load':>8}{'mean diff':>13}{'std':>9}{'rl lower':>12}"
          f"{'p (sign)':>12}{'p (t-test)':>13}")
    print("  " + "-" * 78)
    for load in TEST_LOADS:
        a = per_layout(rows, "rl", load, TEST50_SEEDS)
        b = per_layout(rows, "greedy_classic", load, TEST50_SEEDS)
        diffs = [x - y for x, y in zip(a, b)]
        wins, _, p = sign_test(diffs)
        print(f"  {load:>5} ms{statistics.fmean(diffs)*100:>12.2f}%"
              f"{statistics.stdev(diffs)*100:>8.2f}%"
              f"{f'{wins} / {len(diffs)}':>12}{_fmt_p(p):>12}"
              f"{_fmt_p(paired_t_test(diffs)):>13}")
    print("  " + "-" * 78)

    hops = statistics.fmean(
        per_layout(rows, "greedy_classic", 200, TEST50_SEEDS, "mean_hops")
    )
    print()
    print(f"  (c) greedy_classic hops per delivered packet at 200 ms: "
          f"{hops:.2f}")


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              loads: Sequence[int], placements: int) -> None:
    """Write *rows* with the settings stamp as a leading comment."""
    fields = ["rule", "load_ms", "placement_seed", "init_seed", "total_loss",
              *CAUSES, *EXTRAS, "views_agree"]
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=list(loads), placements=placements,
                                 extra="classic-greedy check"))
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def main() -> None:
    """Run both grids and print the tables."""
    parser = argparse.ArgumentParser(
        description="Measure textbook greedy beside the rule we use."
    )
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Which trained rl models to evaluate.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the models live and the outputs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    args = parser.parse_args()

    target = os.path.join(args.out_dir, OUT_SUBDIR)
    os.makedirs(target, exist_ok=True)

    jobs = [(rule, load, seed, -1, None)
            for load in PHASE1A_LOADS
            for seed in PHASE1A_SEEDS
            for rule in PHASE1A_RULES]
    phase1a = _run(jobs, args.workers, "Phase-1a")
    print_phase1a(phase1a)

    states = _states(args.out_dir, args.init_seeds)
    jobs = []
    for load in TEST_LOADS:
        for seed in TEST50_SEEDS:
            for rule in TEST_RULES:
                if rule == "rl":
                    for init_seed in args.init_seeds:
                        jobs.append((rule, load, seed, init_seed,
                                     states[init_seed]))
                else:
                    jobs.append((rule, load, seed, -1, None))
    test50 = _run(jobs, args.workers, "TEST-layout")
    print_test50(test50)

    print()
    write_csv(os.path.join(target, "phase1a.csv"), phase1a,
              PHASE1A_LOADS, len(PHASE1A_SEEDS))
    write_csv(os.path.join(target, "test50.csv"), test50,
              TEST_LOADS, len(TEST50_SEEDS))


if __name__ == "__main__":
    main()
