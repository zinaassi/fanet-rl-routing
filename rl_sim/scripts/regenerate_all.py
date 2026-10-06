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
    return parser.parse_args()


def steps(placements: int | None) -> List[Sequence[str]]:
    """Return the commands to run, in order.

    Args:
        placements: Placement override, or None for each script's default.

    Returns:
        One argv list per step.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))

    def script(name: str, takes_placements: bool) -> Sequence[str]:
        """Build one step's argv."""
        argv = [sys.executable, "-u", os.path.join(scripts_dir, name)]
        if takes_placements and placements is not None:
            argv += ["--placements", str(placements)]
        return argv

    return [
        script("plot_ploss.py", takes_placements=False),
        script("range_check.py", takes_placements=True),
        script("experiment_1a.py", takes_placements=True),
        script("link_cutoff_sensitivity.py", takes_placements=True),
    ]


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
    for argv in steps(args.placements):
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
