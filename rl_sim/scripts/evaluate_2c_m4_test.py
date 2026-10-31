"""
evaluate_2c_m4_test.py — Phase 2c M4: the held-back test layouts.

M4 chose everything on validation 301-330: when to stop, which update to
keep, and which of the four runs to believe. This runs the chosen actor, and
the two rules, once on layouts 401-450, which took no part in any of it.

These fifty layouts have never been used by anything. Layouts 1-50, the set
Phase 1b and M3 reported on, are deliberately NOT used here and the script
refuses to touch them.

Writes:
    out/2c_m4_test.csv          one row per (policy, test layout)
    out/2c_m4_test_paired.csv   per-layout RL - rule differences
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

from scripts.evaluate_2c_test import TEST_LAYOUTS as PHASE_1B_TEST_LAYOUTS
from scripts.train_c_ppo import CAUSES, ROUTING, write_csv
from scripts.train_c_ppo_m3 import TRAINING_LAYOUTS, VALIDATION_LAYOUTS
from scripts.train_c_ppo_m4 import (
    COMPARE_COLUMNS,
    MODEL_DIR,
    RUNS,
    compare,
    paired_rows,
    print_comparison,
    print_paired,
    print_windows,
)

#: M4's own held-back layouts, used once.
TEST_LAYOUTS: Tuple[int, ...] = tuple(range(401, 451))

#: One run seed per test layout, as the checkpoint set it.
TEST_RUN_SEED_BASE: int = 9400


def run_seed_for(layout: int) -> int:
    """Return the channel and traffic seed for a test layout.

    Args:
        layout: A test placement seed.

    Returns:
        Its run seed.
    """
    return TEST_RUN_SEED_BASE + (layout - TEST_LAYOUTS[0])


def check_layouts_are_held_back() -> None:
    """Stop unless the test layouts took no part in training or choosing.

    Raises:
        SystemExit: If a test layout was trained on, validated on, or is one
            of the Phase-1b fifty that M4 must not reuse.
    """
    low, high = TRAINING_LAYOUTS
    trained = sorted(set(TEST_LAYOUTS) & set(range(low, high + 1)))
    validated = sorted(set(TEST_LAYOUTS) & set(VALIDATION_LAYOUTS))
    reused = sorted(set(TEST_LAYOUTS) & set(PHASE_1B_TEST_LAYOUTS))
    if trained or validated or reused:
        print(f"STOP: test layouts were not held back. trained {trained}, "
              f"validated {validated}, already reported on {reused}")
        raise SystemExit(1)
    print(f"  test {TEST_LAYOUTS[0]}-{TEST_LAYOUTS[-1]} is disjoint from "
          f"training {low}-{high}, validation {VALIDATION_LAYOUTS[0]}-"
          f"{VALIDATION_LAYOUTS[-1]} and the Phase-1b set "
          f"{PHASE_1B_TEST_LAYOUTS[0]}-{PHASE_1B_TEST_LAYOUTS[-1]}")


def validation_means(compare_path: str) -> Dict[str, float]:
    """Return each M4 run's mean validation loss, in %.

    Args:
        compare_path: out/2c_m4_compare.csv.

    Returns:
        Policy label -> mean total loss.

    Raises:
        SystemExit: If the table is missing or holds no M4 run.
    """
    if not os.path.exists(compare_path):
        print(f"STOP: no validation table at {compare_path}")
        raise SystemExit(1)
    with open(compare_path) as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))

    losses: Dict[str, List[float]] = {}
    for row in rows:
        if row["policy"].startswith("m4 run"):
            losses.setdefault(row["policy"], []).append(
                float(row["total_loss"])
            )
    if not losses:
        print(f"STOP: no m4 run rows in {compare_path}")
        raise SystemExit(1)
    return {label: statistics.fmean(values) * 100
            for label, values in sorted(losses.items())}


def best_run(means: Dict[str, float]) -> str:
    """Return the M4 run with the lowest validation loss.

    Args:
        means: Policy label -> mean validation loss in %.

    Returns:
        The chosen run's label.
    """
    best = min(means, key=lambda label: means[label])
    print("\n  Choosing the run on the VALIDATION set:")
    for label, value in means.items():
        mark = "  <- chosen" if label == best else ""
        print(f"    {label}: {value:6.2f}%{mark}")
    return best


def build_jobs(out_dir: str, label: str) -> Tuple[List[Tuple], List[str]]:
    """Return one rollout job per (policy, test layout), moving world.

    Args:
        out_dir: Where the models live.
        label:   The chosen run's label, "m4 run <name>".

    Returns:
        The jobs and their policy labels.

    Raises:
        SystemExit: If the chosen run has no model.
    """
    import torch

    name = label.split()[-1]
    path = os.path.join(out_dir, MODEL_DIR, f"c_actor_m4_{name}.pt")
    if not os.path.exists(path):
        print(f"STOP: no model for {label} at {path}")
        raise SystemExit(1)
    state = torch.load(path, weights_only=True)

    jobs: List[Tuple] = []
    labels: List[str] = []
    for layout in TEST_LAYOUTS:
        run_seed = run_seed_for(layout)
        jobs.append((layout, run_seed, None, "hover", True, True))
        labels.append("hover")
        jobs.append((layout, run_seed, None, "midpoint", True, True))
        labels.append("midpoint")
        jobs.append((layout, run_seed, state, "rl", True, True))
        labels.append(label)
    return jobs, labels


def main() -> None:
    """Run the chosen actor and the rules on the held-back test layouts."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M4: the chosen actor on test layouts 401-450."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--run", type=str, default=None,
                        help="run name to test; default is the one with the "
                             "lowest validation loss")
    args = parser.parse_args()

    print("Checking the test layouts were held back ...")
    check_layouts_are_held_back()

    means = validation_means(os.path.join(args.out_dir, "2c_m4_compare.csv"))
    if args.run is not None:
        label = f"m4 run {args.run}"
        print(f"\n  Testing {label}, asked for on the command line")
    else:
        label = best_run(means)

    jobs, labels = build_jobs(args.out_dir, label)
    context = multiprocessing.get_context("spawn")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
            if args.workers > 1 else None)
    try:
        rows = compare(jobs, labels, pool)
    finally:
        if pool is not None:
            pool.shutdown()

    policies = print_comparison(
        rows, f"TEST layouts {TEST_LAYOUTS[0]}-{TEST_LAYOUTS[-1]}"
    )
    print_windows(rows, policies)
    paired = paired_rows(rows, [label])
    print_paired(paired, [label], len(TEST_LAYOUTS))

    note = (f"phase 2c M4, TEST layouts {TEST_LAYOUTS[0]}-{TEST_LAYOUTS[-1]}, "
            f"run chosen on validation {VALIDATION_LAYOUTS[0]}-"
            f"{VALIDATION_LAYOUTS[-1]}, moving mission, hello 1 s, "
            f"{ROUTING} routing")
    write_csv(os.path.join(args.out_dir, "2c_m4_test.csv"), rows,
              COMPARE_COLUMNS, note)
    write_csv(os.path.join(args.out_dir, "2c_m4_test_paired.csv"), paired,
              ("policy", "against", "placement_seed", "rl_loss", "rule_loss",
               "difference"), note)


if __name__ == "__main__":
    main()
