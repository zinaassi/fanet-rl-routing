"""
evaluate_2c_test.py — Phase 2c M3: the held-back TEST layouts.

M3 trained on layouts 101-300 and chose everything else on the validation
set: when to stop, which update to keep, and which of the three training
seeds to believe. A number chosen on a set is optimistic on that set, so the
validation table cannot be the headline. This runs the chosen actor once on
the fifty TEST layouts, which took no part in any of those decisions, and
reports that.

The run seed of a test layout is the layout number, which is what Phase 1b
did (evaluate_1b_large passes the placement seed as the run seed too). The
settings are otherwise the validation settings exactly: static mission
drones, 1 s hellos, greedy routing, 200 ms traffic, no exploration.

Writes:
    out/2c_test.csv             one row per (policy, test layout)
    out/2c_test_paired.csv      per-layout RL - rule differences
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_1b_large import paired_t_test, sign_test
from scripts.train_c_ppo import (
    CAUSES,
    ROUTING,
    _rollout_job,
    write_csv,
)
from scripts.train_c_ppo_m3 import (
    TRAINING_LAYOUTS,
    VALIDATION_LAYOUTS,
    by_layout,
    paired_rows,
)

#: The held-back layouts, the same fifty Phase 1b tested on.
TEST_LAYOUTS: Tuple[int, ...] = tuple(range(1, 51))


def run_seed_for(layout: int) -> int:
    """Return the channel and traffic seed for a test layout.

    Phase 1b used the placement seed as the run seed as well, so keeping
    that here makes a Phase-2c test row comparable to a Phase-1b one.

    Args:
        layout: A test placement seed.

    Returns:
        Its run seed.
    """
    return layout


def check_layouts_are_held_back() -> None:
    """Stop unless the test layouts took no part in training or choosing.

    Raises:
        SystemExit: If a test layout was trained on or validated on.
    """
    low, high = TRAINING_LAYOUTS
    trained = sorted(set(TEST_LAYOUTS) & set(range(low, high + 1)))
    validated = sorted(set(TEST_LAYOUTS) & set(VALIDATION_LAYOUTS))
    if trained or validated:
        print(f"STOP: test layouts were not held back. trained {trained}, "
              f"validated {validated}")
        raise SystemExit(1)
    print(f"  test {TEST_LAYOUTS[0]}-{TEST_LAYOUTS[-1]} is disjoint from "
          f"training {low}-{high} and validation {VALIDATION_LAYOUTS[0]}-"
          f"{VALIDATION_LAYOUTS[-1]}")


def validation_means(compare_path: str) -> Dict[int, float]:
    """Return each training seed's mean validation loss, in %.

    Args:
        compare_path: out/2c_m3_compare.csv.

    Returns:
        Training seed -> mean total loss over the validation layouts.

    Raises:
        SystemExit: If the validation table is missing or has no actors.
    """
    if not os.path.exists(compare_path):
        print(f"STOP: no validation table at {compare_path}")
        raise SystemExit(1)
    with open(compare_path) as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))

    losses: Dict[int, List[float]] = {}
    for row in rows:
        if row["policy"].startswith("rl seed"):
            seed = int(row["policy"].split()[-1])
            losses.setdefault(seed, []).append(float(row["total_loss"]))
    if not losses:
        print(f"STOP: no rl rows in {compare_path}")
        raise SystemExit(1)
    return {seed: statistics.fmean(values) * 100
            for seed, values in sorted(losses.items())}


def best_validation_seed(means: Dict[int, float]) -> int:
    """Return the training seed with the lowest validation loss.

    Args:
        means: Training seed -> mean validation loss in %.

    Returns:
        The chosen seed.
    """
    best = min(means, key=lambda seed: means[seed])
    print("\n  Choosing the actor on the VALIDATION set:")
    for seed, value in means.items():
        mark = "  <- chosen" if seed == best else ""
        print(f"    seed {seed}: {value:6.2f}%{mark}")
    return best


def build_jobs(out_dir: str, seed: int) -> Tuple[List[Tuple], List[str]]:
    """Return one rollout job per (policy, test layout).

    Args:
        out_dir: Where the models live.
        seed:    The training seed chosen on validation.

    Returns:
        The jobs and their policy labels.

    Raises:
        SystemExit: If a model file is missing.
    """
    import torch

    paths = {
        "clone": os.path.join(out_dir, "models", "c_actor_bc.pt"),
        f"rl seed {seed}": os.path.join(out_dir, "models",
                                        f"c_actor_m3_seed{seed}.pt"),
    }
    states: Dict[str, Any] = {}
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"STOP: no model for {label} at {path}")
            raise SystemExit(1)
        states[label] = torch.load(path, weights_only=True)

    rl_label = f"rl seed {seed}"
    jobs: List[Tuple] = []
    labels: List[str] = []
    for layout in TEST_LAYOUTS:
        run_seed = run_seed_for(layout)
        for label, state, policy in (("hover", None, "hover"),
                                     ("midpoint", None, "midpoint"),
                                     ("clone", states["clone"], "clone"),
                                     (rl_label, states[rl_label], "rl")):
            jobs.append((layout, run_seed, state, policy, True))
            labels.append(label)
    return jobs, labels


def print_table(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Print the test table and return the policy labels in order.

    Args:
        rows: Every test row.

    Returns:
        The policy labels, rules first.
    """
    policies: List[str] = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    width = 128
    print(f"\n  TEST layouts {TEST_LAYOUTS[0]}-{TEST_LAYOUTS[-1]} — "
          f"% of packets created")
    print("  " + "-" * width)
    header = f"{'policy':>14}{'total':>9}{'std':>8}"
    header += "".join(f"{cause[:9]:>11}" for cause in CAUSES)
    header += f"{'flown':>9}{'from GS':>9}"
    print("  " + header)
    print("  " + "-" * width)
    for policy in policies:
        mine = [r for r in rows if r["policy"] == policy]
        totals = [r["total_loss"] * 100 for r in mine]
        line = f"{policy:>14}{statistics.fmean(totals):>8.2f}%"
        line += f"{statistics.stdev(totals) if len(mine) > 1 else 0.0:>8.2f}"
        for cause in CAUSES:
            line += f"{statistics.fmean(r[cause] for r in mine) * 100:>10.2f}%"
        flown = statistics.fmean(r["c_distance_flown_mean"] for r in mine)
        from_gs = statistics.fmean(r["c_gs_distance_mean"] for r in mine)
        line += f"{flown:>9.0f}{from_gs:>9.0f}"
        print("  " + line)
    print("  " + "-" * width)
    return policies


def print_paired(paired: Sequence[Dict[str, Any]],
                 policies: Sequence[str]) -> None:
    """Print the paired differences against each rule.

    Args:
        paired:   Every paired row.
        policies: The RL policy labels.
    """
    print(f"\n  Paired over {len(TEST_LAYOUTS)} TEST layouts — points of "
          f"total loss, negative means the actor loses fewer")
    print("  " + "-" * 92)
    print("  " + f"{'actor':>14}{'against':>10}{'mean diff':>12}"
                 f"{'wins':>8}{'sign p':>12}{'t-test p':>12}")
    print("  " + "-" * 92)
    for rule in ("hover", "midpoint", "clone"):
        for policy in policies:
            diffs = [row["difference"] for row in paired
                     if row["policy"] == policy and row["against"] == rule]
            if not diffs:
                continue
            wins, used, p = sign_test(diffs)
            line = f"{policy:>14}{rule:>10}{statistics.fmean(diffs):>+11.2f} "
            line += f"{wins:>4}/{used:<3}{p:>11.2e}"
            line += f"{paired_t_test(diffs):>12.2e}"
            print("  " + line)
    print("  " + "-" * 92)


def main() -> None:
    """Run the chosen actor and the rules on the held-back test layouts."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M3: the chosen actor on the TEST layouts."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed to test; default is the one with "
                             "the lowest validation loss")
    args = parser.parse_args()

    print("Checking the test layouts were held back ...")
    check_layouts_are_held_back()

    means = validation_means(os.path.join(args.out_dir, "2c_m3_compare.csv"))
    seed = args.seed if args.seed is not None else best_validation_seed(means)
    if args.seed is not None:
        print(f"\n  Testing training seed {seed}, asked for on the command "
              f"line")

    jobs, labels = build_jobs(args.out_dir, seed)
    context = multiprocessing.get_context("spawn")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
            if args.workers > 1 else None)
    print(f"\nRunning {len(jobs)} runs on test {TEST_LAYOUTS[0]}-"
          f"{TEST_LAYOUTS[-1]} ...", flush=True)
    try:
        results = (list(pool.map(_rollout_job, jobs)) if pool is not None
                   else [_rollout_job(job) for job in jobs])
    finally:
        if pool is not None:
            pool.shutdown()

    rows = [{"policy": label, "placement_seed": job[0], "run_seed": job[1],
             **result}
            for job, label, result in zip(jobs, labels, results)]
    disagreed = [row for row in rows if not row["views_agree"]]
    print(f"  views agreed in {len(rows) - len(disagreed)} of {len(rows)} runs")
    if disagreed:
        raise RuntimeError("the ACK views disagreed with the simulator")

    policies = print_table(rows)
    actors = [policy for policy in policies if policy.startswith("rl seed")]
    paired = paired_rows(rows, actors) + _against_clone(rows, actors)
    print_paired(paired, actors)

    note = (f"phase 2c M3, TEST layouts {TEST_LAYOUTS[0]}-{TEST_LAYOUTS[-1]}, "
            f"actor chosen on validation {VALIDATION_LAYOUTS[0]}-"
            f"{VALIDATION_LAYOUTS[-1]}, static mission, hello 1 s, "
            f"{ROUTING} routing")
    write_csv(os.path.join(args.out_dir, "2c_test.csv"), rows,
              ("policy", "placement_seed", "run_seed", "total_loss", *CAUSES,
               "c_distance_flown_mean", "c_gs_distance_mean", "mean_reward",
               "views_agree"), note)
    write_csv(os.path.join(args.out_dir, "2c_test_paired.csv"), paired,
              ("policy", "against", "placement_seed", "rl_loss", "rule_loss",
               "difference"), note)


def _against_clone(rows: Sequence[Dict[str, Any]],
                   policies: Sequence[str]) -> List[Dict[str, Any]]:
    """Return the per-layout differences against the clone.

    paired_rows only compares against the two hand-written rules; the clone
    of midpoint is the actor's own starting point, so the gap from it is what
    the training actually bought.

    Args:
        rows:     Every test row.
        policies: The RL policy labels.

    Returns:
        One row per (actor, layout), differences in points.
    """
    reference = by_layout(rows, "clone")
    out: List[Dict[str, Any]] = []
    for policy in policies:
        mine = by_layout(rows, policy)
        for layout in sorted(reference):
            out.append({
                "policy": policy,
                "against": "clone",
                "placement_seed": layout,
                "rl_loss": mine[layout] * 100,
                "rule_loss": reference[layout] * 100,
                "difference": (mine[layout] - reference[layout]) * 100,
            })
    return out


if __name__ == "__main__":
    main()
