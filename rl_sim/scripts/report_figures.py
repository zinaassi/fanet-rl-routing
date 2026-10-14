"""
report_figures.py — Every figure the written report uses, in one place.

The report under ``docs/report/`` must be regenerable from one command, so
this script does two things:

  * copies the figures the report shares with the rest of the project out of
    ``rl_sim/out/`` into ``docs/report/figures/``;
  * deletes any figure left in the report folder that it no longer produces,
    so a superseded figure cannot sit there looking current.

The example layout is a hand-made figure supplied for the report. It is
listed under ``KEEP``: this script neither draws nor deletes it.

Nothing is simulated twice: the copied figures are whatever
``regenerate_all.py`` last produced. Figures drawn here carry the same
settings stamp as every other figure, so a report figure can be traced back
to the config that made it.

Usage:
    python scripts/report_figures.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.stamp import stamp_lines

SURFACE = "#faf9f5"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
GOOD_LINK_MAX_LOSS = 0.10   # the "good link" threshold, as in range_check
M_COLOR = "#2a78d6"
C_COLOR = "#1baf7a"
GS_COLOR = "#eb6834"
LINK_COLOR = "#b9c6d4"

# Figures the report shares with the rest of the project.
#: Figures the report shares with the rest of the project. Every one must
#: exist in out/; a missing name is reported rather than passed over, so the
#: report folder can never quietly fall behind.
COPIED = (
    "ploss_curve.png",
    "1a_total_loss_vs_load.png",
    "1a_loss_heatmaps.png",
    "1a_link_matrix_placement1.png",
    "1b_learning_curve.png",
    "1b_learning_curve_seeds.png",
    "1b_validation_causes.png",
    "1b_comparison_total_loss_50.png",
    "1b_loss_heatmaps_50.png",
    "queue_size_total_loss.png",
    "queue_size_load_curve.png",
)

#: Figures this script draws itself. None at the moment: the example layout
#: is a hand-made figure supplied for the report, and is listed under KEEP so
#: a run of this script neither regenerates nor deletes it.
OWN: tuple = ()

#: Figures that live in the report folder but are NOT produced here.
KEEP = ("example_layout.png",)


def _style(ax: plt.Axes) -> None:
    """Apply the shared recessive axis styling."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _stamp(fig: plt.Figure, **kwargs: object) -> None:
    """Put the settings stamp at the bottom-left of *fig*."""
    fig.text(0.01, 0.012, stamp_lines(width=130, **kwargs),  # type: ignore[arg-type]
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def copy_shared(out_dir: str, figures_dir: str) -> List[str]:
    """Copy the figures the report shares with the rest of the project.

    Returns:
        The names that were missing, so a partial run says so plainly.
    """
    missing: List[str] = []
    for name in COPIED:
        source = os.path.join(out_dir, name)
        if not os.path.exists(source):
            missing.append(name)
            continue
        shutil.copy2(source, os.path.join(figures_dir, name))
        print(f"  copied {name}")
    return missing


def prune(figures_dir: str, keep: Sequence[str]) -> List[str]:
    """Delete figures in the report folder that are no longer produced.

    Without this, a figure made with a superseded model would sit in the
    report folder looking current.

    Args:
        figures_dir: The report's figures folder.
        keep:        Every filename this script produces.

    Returns:
        The names deleted.
    """
    removed: List[str] = []
    for name in sorted(os.listdir(figures_dir)):
        if not name.endswith(".png") or name in keep:
            continue
        os.remove(os.path.join(figures_dir, name))
        removed.append(name)
    return removed


def main() -> None:
    """Draw the report-only figures and copy the shared ones."""
    parser = argparse.ArgumentParser(
        description="Build every figure the written report uses."
    )
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the project's figures live.")
    parser.add_argument("--figures-dir", type=str,
                        default=os.path.join("..", "docs", "report",
                                             "figures"),
                        help="Where the report's figures go.")
    args = parser.parse_args()

    figures_dir = os.path.abspath(args.figures_dir)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Report figures -> {figures_dir}")

    missing = copy_shared(args.out_dir, figures_dir)
    if missing:
        print("\n  NOT copied (not generated yet): " + ", ".join(missing))

    removed = prune(figures_dir, OWN + COPIED + KEEP)
    if removed:
        print("  deleted (no longer produced): " + ", ".join(removed))

    print(f"\n  left alone (not produced here): {', '.join(KEEP)}")


if __name__ == "__main__":
    main()
