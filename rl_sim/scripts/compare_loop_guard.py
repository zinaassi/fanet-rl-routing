"""
compare_loop_guard.py — The loop guard's effect on every routing rule
(CHECKPOINT 12).

The path guard is a property of the network, not of a rule, so it has to be
measured on all of them. This runs the TEST layouts under both
``config.LOOP_GUARD`` settings and tabulates:

  (a) total loss, loss by cause (dead_end included) and mean hops, for
      random / rate / rl under each guard, plus one greedy row — greedy only
      ever moves strictly closer to the GS, so it cannot revisit a drone and
      its numbers are identical under both;
  (b) paired per-layout comparisons under "path": rl-rate, rl-greedy and
      rate-greedy.

Each guard needs its own trained models, read from
``out/models/<guard>/rl_seed*.pt``. Nothing here trains or learns.

Writes:
    out/1b_loop_guard.csv       one row per run
    out/1b_loop_guard_paired.csv  one row per layout per comparison
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from scripts.stamp import csv_comment
from scripts.evaluate_1b_large import (
    TEST50_SEEDS as TEST_SEEDS, paired_t_test, sign_test,
)
from scripts.train_1b import run_once

GUARDS = ("previous", "path")
LOADS_MS = (200, 500)
CAUSES = ("channel", "queue_full", "no_route", "dead_end", "ttl",
          "hop_limit")
EXTRAS = ("revisit_share", "mean_hops")

# greedy is reported once: it cannot revisit, so the guard cannot touch it.
# The run still happens under both guards, and _check_greedy verifies that.
RULES = ("greedy", "random", "rate", "rl")


def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one run under one guard.

    Args:
        args: ``(guard, rule, load_ms, seed, init_seed, state_dict)``.

    Returns:
        That run's row.
    """
    guard, rule, load_ms, seed, init_seed, state = args
    previous = config.LOOP_GUARD
    config.LOOP_GUARD = guard
    try:
        result = run_once(rule, seed, seed, load_ms, state_dict=state,
                          training=False)
    finally:
        # Restored so a --workers 1 run does not leave the parent's guard
        # wherever the last job happened to set it.
        config.LOOP_GUARD = previous
    return {
        "loop_guard": guard, "rule": rule, "load_ms": load_ms,
        "placement_seed": seed,
        "init_seed": init_seed if rule == "rl" else -1,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        **{extra: result[extra] for extra in EXTRAS},
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def _load_states(out_dir: str, guard: str,
                 init_seeds: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Return the trained weights for *guard*, keyed by init seed.

    Raises:
        SystemExit: If a model is missing.
    """
    import torch

    states: Dict[int, Dict[str, Any]] = {}
    for init_seed in init_seeds:
        path = os.path.join(out_dir, "models", guard, f"rl_seed{init_seed}.pt")
        if not os.path.exists(path):
            print(f"STOP: no model at {path}. Train under {guard!r} first.")
            raise SystemExit(1)
        states[init_seed] = torch.load(path, weights_only=True)
    return states


def run_grid(out_dir: str, init_seeds: Sequence[int],
             workers: int) -> List[Dict[str, Any]]:
    """Run every rule on every TEST layout, at both loads, under both guards.

    Returns:
        One row per run.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    jobs: List[Tuple] = []
    for guard in GUARDS:
        states = _load_states(out_dir, guard, init_seeds)
        for load_ms in LOADS_MS:
            for seed in TEST_SEEDS:
                for rule in RULES:
                    if rule == "rl":
                        for init_seed in init_seeds:
                            jobs.append((guard, rule, load_ms, seed,
                                         init_seed, states[init_seed]))
                    else:
                        jobs.append((guard, rule, load_ms, seed, -1, None))

    print(f"Running {len(jobs)} runs "
          f"({len(GUARDS)} guards x {len(LOADS_MS)} loads x "
          f"{len(TEST_SEEDS)} layouts, rl x {len(init_seeds)} init seeds) ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [r for r in rows if not r["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — {row['loop_guard']} "
                  f"{row['rule']} seed {row['placement_seed']}: "
                  f"{row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _select(rows: List[Dict[str, Any]], **where: Any) -> List[Dict[str, Any]]:
    """Return the rows matching every key=value in *where*."""
    return [r for r in rows
            if all(r[key] == value for key, value in where.items())]


def _per_layout(rows: List[Dict[str, Any]], guard: str, rule: str,
                load_ms: int, field: str = "total_loss") -> Dict[int, float]:
    """Return {layout seed: value}, averaging rl over its init seeds."""
    picked = _select(rows, loop_guard=guard, rule=rule, load_ms=load_ms)
    by_seed: Dict[int, List[float]] = {}
    for row in picked:
        value = row[field]
        if value is None:
            continue
        by_seed.setdefault(row["placement_seed"], []).append(float(value))
    return {seed: statistics.fmean(values)
            for seed, values in by_seed.items() if values}


def _check_greedy(rows: List[Dict[str, Any]]) -> None:
    """Fail loudly if greedy is not identical under both guards.

    This is the claim that lets the table carry one greedy row instead of two.
    """
    for load_ms in LOADS_MS:
        for field in ("total_loss", *CAUSES, *EXTRAS):
            previous = _per_layout(rows, "previous", "greedy", load_ms, field)
            path = _per_layout(rows, "path", "greedy", load_ms, field)
            assert previous == path, (
                f"greedy differs between guards at {load_ms} ms on {field}"
            )


def _fmt_p(p: float) -> str:
    """Format a p-value without pretending to more precision than it has."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Print table (a): every rule under every guard, per load."""
    print()
    print(f"  (a) {len(TEST_SEEDS)} TEST layouts (seeds {TEST_SEEDS[0]}-"
          f"{TEST_SEEDS[-1]}), mean % of packets created")
    print("      greedy is listed once: identical under both guards (checked)")
    width = 118
    print("  " + "-" * width)
    print(f"  {'load':>7}{'guard':>10}{'rule':>8}{'total':>9}{'std':>7}"
          f"{'channel':>10}{'queue_full':>12}{'no_route':>10}{'dead_end':>10}"
          f"{'ttl':>7}{'hop_lim':>9}{'revisits':>10}{'hops':>7}")
    print("  " + "-" * width)

    def line(label: str, guard: str, rule: str, load_ms: int) -> None:
        """Print one row."""
        totals = list(_per_layout(rows, guard, rule, load_ms).values())
        causes = [statistics.fmean(
            _per_layout(rows, guard, rule, load_ms, c).values()
        ) for c in CAUSES]
        revisits = statistics.fmean(
            _per_layout(rows, guard, rule, load_ms, "revisit_share").values()
        )
        hops = statistics.fmean(
            _per_layout(rows, guard, rule, load_ms, "mean_hops").values()
        )
        print(f"  {label:>17}{rule:>8}"
              f"{statistics.fmean(totals)*100:>8.2f}%"
              f"{statistics.stdev(totals)*100:>7.2f}"
              f"{causes[0]*100:>9.2f}%{causes[1]*100:>11.2f}%"
              f"{causes[2]*100:>9.2f}%{causes[3]*100:>9.2f}%"
              f"{causes[4]*100:>6.2f}%{causes[5]*100:>8.2f}%"
              f"{revisits*100:>9.2f}%{hops:>7.2f}")

    for load_ms in LOADS_MS:
        line(f"{load_ms:>5} ms      both", "path", "greedy", load_ms)
        for rule in ("random", "rate", "rl"):
            for guard in GUARDS:
                line(f"{guard:>10}", guard, rule, load_ms)
        print("  " + "-" * width)


def guard_paired_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pair "path" against "previous" on the same layout, rule by rule.

    Args:
        rows: Every run's row.

    Returns:
        One record per (load, rule).
    """
    out: List[Dict[str, Any]] = []
    for load_ms in LOADS_MS:
        for rule in RULES:
            path = _per_layout(rows, "path", rule, load_ms)
            previous = _per_layout(rows, "previous", rule, load_ms)
            seeds = sorted(set(path) & set(previous))
            diffs = [path[s] - previous[s] for s in seeds]
            wins, used, p_sign = sign_test(diffs)
            out.append({
                "load_ms": load_ms, "rule": rule, "n": len(diffs),
                "mean_diff": statistics.fmean(diffs),
                "std_diff": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                "wins": wins, "ties": len(diffs) - used,
                "p_sign": p_sign, "p_ttest": paired_t_test(diffs),
            })
    return out


def print_guard_paired(records: Sequence[Dict[str, Any]]) -> None:
    """Print the path-against-previous table."""
    width = 92
    print()
    print(f'  (c) Paired "path" - "previous" on the same {len(TEST_SEEDS)} '
          f'layouts')
    print("      negative = the path guard loses fewer packets. Each rl is")
    print("      evaluated under the guard it was TRAINED with, so that row")
    print("      compares two differently trained agents.")
    print("  " + "-" * width)
    print(f"  {'load':>7}{'rule':>9}{'mean diff':>12}{'std':>9}"
          f"{'path lower':>13}{'ties':>7}{'p (sign)':>12}{'p (t-test)':>13}")
    print("  " + "-" * width)
    for record in records:
        wins = f"{record['wins']} / {record['n']}"
        print(f"  {record['load_ms']:>5} ms{record['rule']:>9}"
              f"{record['mean_diff']*100:>11.2f}%"
              f"{record['std_diff']*100:>8.2f}%"
              f"{wins:>13}{record['ties']:>7}"
              f"{_fmt_p(record['p_sign']):>12}"
              f"{_fmt_p(record['p_ttest']):>13}")
    print("  " + "-" * width)


def print_paired(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Print table (b): paired per-layout comparisons under "path".

    Returns:
        One row per layout per comparison, for the CSV.
    """
    comparisons = (("rl", "rate"), ("rl", "greedy"), ("rate", "greedy"))
    out: List[Dict[str, Any]] = []

    print()
    print('  (b) Paired on identical layouts, LOOP_GUARD = "path"')
    print("      negative = the first rule loses fewer packets")
    print("  " + "-" * 76)
    print(f"  {'load':>7}{'comparison':>18}{'layouts':>9}{'mean diff':>12}"
          f"{'std':>9}{'first lower':>14}")
    print("  " + "-" * 76)

    for load_ms in LOADS_MS:
        for first, second in comparisons:
            a = _per_layout(rows, "path", first, load_ms)
            b = _per_layout(rows, "path", second, load_ms)
            seeds = sorted(set(a) & set(b))
            diffs = [a[s] - b[s] for s in seeds]
            for seed, diff in zip(seeds, diffs):
                out.append({
                    "load_ms": load_ms, "comparison": f"{first} - {second}",
                    "placement_seed": seed, "diff": diff,
                    "first_loss": a[seed], "second_loss": b[seed],
                })
            lower = sum(1 for d in diffs if d < 0)
            print(f"  {load_ms:>5} ms{f'{first} - {second}':>18}{len(seeds):>9}"
                  f"{statistics.fmean(diffs)*100:>11.2f}%"
                  f"{statistics.stdev(diffs)*100:>8.2f}%"
                  f"{f'{lower} / {len(seeds)}':>14}")
        print("  " + "-" * 76)
    return out


def write_csv(path: str, rows: List[Dict[str, Any]],
              fields: Sequence[str], loads: Sequence[int]) -> None:
    """Write *rows* with the settings stamp as a leading comment."""
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=loads, placements=len(TEST_SEEDS),
                                 extra="phase 1b loop-guard comparison"))
        writer = csv.DictWriter(handle, fieldnames=list(fields),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def main() -> None:
    """Run the grid under both guards and print both tables."""
    parser = argparse.ArgumentParser(
        description="Compare the loop-guard modes on the TEST layouts."
    )
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Which trained models to evaluate.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the models live and the CSVs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    args = parser.parse_args()

    # The guard is set per worker, so the parent's own setting must not leak
    # into a table heading or a stamp as if it applied to everything.
    rows = run_grid(args.out_dir, args.init_seeds, args.workers)
    _check_greedy(rows)
    print_table(rows)
    paired = print_paired(rows)
    guard_paired = guard_paired_rows(rows)
    print_guard_paired(guard_paired)

    print()
    # These two files cover BOTH guards, so a stamp naming one of them would
    # be wrong. The guard column says which guard each row ran under.
    previous_guard = config.LOOP_GUARD
    config.LOOP_GUARD = "+".join(GUARDS)
    try:
        write_csv(os.path.join(args.out_dir, "1b_loop_guard.csv"), rows,
                  ["loop_guard", "rule", "load_ms", "placement_seed",
                   "init_seed", "total_loss", *CAUSES, *EXTRAS,
                   "views_agree"], LOADS_MS)
        write_csv(os.path.join(args.out_dir, "1b_loop_guard_paired.csv"),
                  paired, ["load_ms", "comparison", "placement_seed", "diff",
                           "first_loss", "second_loss"], LOADS_MS)
        write_csv(os.path.join(args.out_dir, "1b_loop_guard_by_guard.csv"),
                  guard_paired, ["load_ms", "rule", "n", "mean_diff",
                                 "std_diff", "wins", "ties", "p_sign",
                                 "p_ttest"], LOADS_MS)
    finally:
        config.LOOP_GUARD = previous_guard


if __name__ == "__main__":
    main()
