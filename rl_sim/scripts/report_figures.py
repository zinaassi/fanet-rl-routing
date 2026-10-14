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
    "1a_total_loss_vs_load.png",
    "1a_link_matrix_placement1.png",
    "1b_learning_curve.png",
    "1b_learning_curve_seeds.png",
    "1b_validation_causes.png",
    "1b_comparison_total_loss_50.png",
    "1b_loss_heatmaps_50.png",
    "queue_size_total_loss.png",
    "queue_size_load_curve.png",
)

#: Figures this script draws itself, from results already computed. These
#: are the two the report shows without a settings stamp: the report has its
#: own numbered captions, and the stamp is developer metadata. The stamped
#: versions in rl_sim/out/ are untouched.
OWN = ("ploss_curve.png", "1a_loss_heatmaps.png")

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


def draw_unstamped(out_dir: str, figures_dir: str) -> List[str]:
    """Redraw the report's two stamped figures without their stamp.

    Nothing is re-simulated: the loss curve comes from the channel code, and
    the heatmap is redrawn from the committed summary CSV.

    Args:
        out_dir:     Where the project's CSVs live.
        figures_dir: Where the report's figures go.

    Returns:
        The names that could not be drawn, with the reason.
    """
    from scripts.plot_ploss import plot as plot_curve
    from scripts.stamp import read_stamped_csv

    problems: List[str] = []

    plot_curve(os.path.join(figures_dir, "ploss_curve.png"), stamp=False)
    print("  drew ploss_curve.png (no stamp)")

    summary_path = os.path.join(out_dir, "1a_summary.csv")
    if not os.path.exists(summary_path):
        problems.append(f"1a_loss_heatmaps.png: no {summary_path}")
        return problems

    from scripts.experiment_1a import plot_loss_heatmaps

    # The summary CSV holds exactly the <column>_mean / _std records the
    # heatmap reads, so it can be redrawn without rerunning the grid.
    summary: List[Dict[str, Any]] = []
    for raw in read_stamped_csv(summary_path):
        record: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in ("routing",):
                record[key] = value
            elif value == "":
                record[key] = None
            else:
                record[key] = float(value)
        record["load_ms"] = int(record["load_ms"])
        record["runs"] = int(record["runs"])
        summary.append(record)

    plot_loss_heatmaps(summary,
                       os.path.join(figures_dir, "1a_loss_heatmaps.png"),
                       annotate=False)
    print("  drew 1a_loss_heatmaps.png (no stamp, no caption)")
    return problems


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

    problems = draw_unstamped(args.out_dir, figures_dir)
    if problems:
        print("  NOT drawn: " + "; ".join(problems))

    missing = copy_shared(args.out_dir, figures_dir)
    if missing:
        print("\n  NOT copied (not generated yet): " + ", ".join(missing))

    removed = prune(figures_dir, OWN + COPIED + KEEP)
    if removed:
        print("  deleted (no longer produced): " + ", ".join(removed))

    print(f"\n  left alone (not produced here): {', '.join(KEEP)}")


if __name__ == "__main__":
    main()
