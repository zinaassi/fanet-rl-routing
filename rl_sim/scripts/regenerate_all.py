"""
regenerate_all.py — Rebuild every figure and table from the current config.

Runs each generating script in order, in a FRESH process so none can leave a
mutated ``config`` behind for the next, and stops at the first failure. Every
artefact therefore comes from one code version and one set of settings, which
is the point: a stale figure next to a fresh table is worse than neither.

Order:
    1. plot_ploss.py                the loss curve
    2. range_check.py               connectivity at several link cutoffs
    3. experiment_1a.py             the grid, its CSVs and its three figures
    4. link_cutoff_sensitivity.py   how much the cutoff moves the results
    5. evaluate_1b.py               Phase 1b on the held-back TEST layouts
    6. plot_1b.py                   the learning curve and validation causes

Phase-1b TRAINING is NOT part of the default run: it takes tens of minutes
and overwrites the saved models. Pass ``--with-training`` to include it,
which inserts train_1b.py before the two 1b steps. Without it, those two
read the models and the log already in out/ — and are skipped if neither is
there yet, so a fresh clone can still regenerate everything else.

A step fails the run if it exits non-zero, or if its output ever says the ACK
view disagreed with ground truth.

Usage:
    python scripts/regenerate_all.py
    python scripts/regenerate_all.py --placements 3      # a quick smoke run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.stamp import settings_stamp

#: Phrases that mean a run must not be trusted, whatever its exit code.
FAILURE_MARKERS = (
    "VIEWS DISAGREE",
    "STOP:",
    "Traceback",
)

#: A run that reports agreement must say so; scripts that check it print this.
AGREEMENT_MARKER = "views agreed in all"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Regenerate every figure and table from the current config."
    )
    parser.add_argument(
        "--placements", type=int, default=None,
        help="Override the placement count for the scripts that take one. "
             "Leave unset to use each script's default.",
    )
    parser.add_argument(
        "--with-training", action="store_true",
        help="Also retrain the Phase-1b agent (tens of minutes, and it "
             "overwrites the saved models).",
    )
    parser.add_argument(
        "--init-seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Phase-1b network initialisation seeds (default 0 1 2).",
    )
    return parser.parse_args()


def steps(
    placements: int | None,
    with_training: bool = False,
    init_seeds: Sequence[int] = (0, 1, 2),
) -> List[Sequence[str]]:
    """Return the commands to run, in order.

    Args:
        placements:    Placement override, or None for each script's default.
        with_training: Whether to retrain the Phase-1b agent first.
        init_seeds:    Phase-1b network initialisation seeds.

    Returns:
        One argv list per step.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(os.path.dirname(scripts_dir), "out")

    def script(name: str, takes_placements: bool = False) -> Sequence[str]:
        """Build one step's argv."""
        argv = [sys.executable, "-u", os.path.join(scripts_dir, name)]
        if takes_placements and placements is not None:
            argv += ["--placements", str(placements)]
        return argv

    seed_args = [str(seed) for seed in init_seeds]
    plan: List[Sequence[str]] = [
        script("plot_ploss.py"),
        script("range_check.py", takes_placements=True),
        script("experiment_1a.py", takes_placements=True),
        script("link_cutoff_sensitivity.py", takes_placements=True),
    ]

    if with_training:
        plan.append(list(script("train_1b.py")) + ["--init-seeds", *seed_args])

    # Without training, the 1b steps need what a previous run left behind.
    models = [os.path.join(out_dir, "models", f"rl_seed{seed}.pt")
              for seed in init_seeds]
    if with_training or all(os.path.exists(path) for path in models):
        plan.append(list(script("evaluate_1b.py")) + ["--init-seeds", *seed_args])
    else:
        print("  (skipping evaluate_1b.py: no trained models in out/models/ — "
              "run with --with-training)")

    if with_training or os.path.exists(os.path.join(out_dir, "1b_training_log.csv")):
        plan.append(script("plot_1b.py"))
    else:
        print("  (skipping plot_1b.py: no out/1b_training_log.csv yet)")

    return plan


def run_step(argv: Sequence[str], repo_root: str) -> str:
    """Run one step and return its output.

    Args:
        argv:      Command to run.
        repo_root: Directory to run it from (rl_sim).

    Returns:
        The step's combined output.

    Raises:
        SystemExit: If the step fails, or its output shows a views_agree
            failure.
    """
    name = os.path.basename(argv[2])
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}", flush=True)

    result = subprocess.run(
        argv, cwd=repo_root, capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    print(output, end="", flush=True)

    if result.returncode != 0:
        print(f"\nFAILED: {name} exited {result.returncode}. Stopping.")
        raise SystemExit(1)

    for marker in FAILURE_MARKERS:
        if marker in output:
            print(f"\nFAILED: {name} reported {marker!r}. Stopping.")
            raise SystemExit(1)

    return output


def main() -> None:
    """Regenerate everything, stopping at the first failure."""
    args = parse_args()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("Regenerating every figure and table from the current config.")
    print(f"  {settings_stamp()}")

    started = time.perf_counter()
    checked = 0
    for argv in steps(args.placements, args.with_training, args.init_seeds):
        output = run_step(argv, repo_root)
        if AGREEMENT_MARKER in output:
            checked += 1

    print(f"\n{'=' * 70}")
    print(f"  all steps finished in {time.perf_counter() - started:.1f}s")
    print(f"  {checked} step(s) confirmed the ACK view matched ground truth")
    print(f"  {settings_stamp()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
