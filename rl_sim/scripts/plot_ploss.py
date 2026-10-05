"""
plot_ploss.py — Plot the channel loss curve, new model against old.

Draws one figure over 0-800 m showing:

  * the CURRENT model (bold): a logistic curve in distance, fitted to measured
    UAV links in Rosati et al., "Dynamic Routing for Flying Ad Hoc Networks"
    (arXiv:1406.4399),
        p_loss(d) = 1 / (1 + exp(-s * (d - D50)))
  * the model it REPLACED (dashed): free-space path loss with a receiver
    sensitivity of -54 dBm, and loss exp(-k * margin_dB) with k = 0.8. That
    curve hits 100% loss at 249.69 m and is undefined past it — the "wall".

The old model is reproduced locally, in :func:`old_model_loss` below, purely so
the two can be compared. Nothing in the simulator uses FSPL any more.

Vertical markers show the distances that matter for a 900 x 900 m area with the
GS at its centre: the edge of "good" links, the 50% point, the link cutoff, and
the farthest a drone can possibly be from the GS (a corner).

Usage:
    python scripts/plot_ploss.py
    python scripts/plot_ploss.py --out somewhere.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel

# Categorical hues, validated as a 2-slot categorical palette.
NEW_COLOR = "#2a78d6"
OLD_COLOR = "#eb6834"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
SURFACE = "#fcfcfb"

# --- the replaced model, kept only to draw it ------------------------------
OLD_PT_DBM = 30.0
OLD_GT_DBI = 2.0
OLD_GR_DBI = 2.0
OLD_F_HZ = 2.4e9
OLD_RX_SENSITIVITY_DBM = -54.0
OLD_K = 0.8
_SPEED_OF_LIGHT = 299_792_458.0
_OLD_FSPL_CONST_DB = (
    20.0 * math.log10(OLD_F_HZ)
    + 20.0 * math.log10(4.0 * math.pi / _SPEED_OF_LIGHT)
)
OLD_MAX_LINK_M = 10.0 ** (
    (OLD_PT_DBM + OLD_GT_DBI + OLD_GR_DBI - OLD_RX_SENSITIVITY_DBM
     - _OLD_FSPL_CONST_DB) / 20.0
)


def old_model_loss(dist_m: float) -> float:
    """Loss under the REPLACED FSPL model, for comparison only.

    ``exp(-0.8 * M)`` where M is the margin in dB above a -54 dBm sensitivity,
    and 1.0 once the margin is gone (past ~249.69 m).

    Args:
        dist_m: Distance between endpoints, in metres.

    Returns:
        Float in [0.0, 1.0].
    """
    if dist_m <= 1e-9:
        return 0.0
    fspl_db = 20.0 * math.log10(dist_m) + _OLD_FSPL_CONST_DB
    margin = (OLD_PT_DBM + OLD_GT_DBI + OLD_GR_DBI - fspl_db
              - OLD_RX_SENSITIVITY_DBM)
    if margin <= 0.0:
        return 1.0
    return min(1.0, math.exp(-OLD_K * margin))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot the channel loss curve, new model against old."
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("out", "ploss_vs_distance.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def markers() -> List[Tuple[float, str]]:
    """Return the vertical reference lines, as (distance, label) pairs.

    Returns:
        The distances that matter for a 900 x 900 m area with a central GS.
    """
    corner = math.hypot(config.WIDTH / 2.0, config.HEIGHT / 2.0)
    return [
        (channel.distance_for_loss(0.10), "10% loss — edge of good links"),
        (config.LOSS_50_DISTANCE_M, "50% loss"),
        (channel.max_link_distance(),
         f"{config.LINK_MAX_LOSS * 100:.0f}% loss — link cutoff"),
        (corner, "farthest point from the GS"),
    ]


def plot(out_path: str) -> str:
    """Draw both curves and write the figure to *out_path*.

    Args:
        out_path: Where to write the PNG. Parent directories are created.

    Returns:
        The path written.
    """
    dists = np.linspace(0.0, 800.0, 3000)

    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    new_losses = [100.0 * channel.p_loss(float(d)) for d in dists]
    old_losses = [100.0 * old_model_loss(float(d)) for d in dists]

    ax.plot(dists, old_losses, color=OLD_COLOR, linewidth=2.0,
            linestyle="--", label="old: exp(-0.8·margin), FSPL", zorder=3)
    ax.plot(dists, new_losses, color=NEW_COLOR, linewidth=2.8,
            label="new: logistic in distance", zorder=4)

    ax.annotate(
        "new", xy=(470.0, 100.0 * channel.p_loss(470.0)),
        xytext=(10, -4), textcoords="offset points",
        color=NEW_COLOR, fontsize=11, fontweight="bold", zorder=6,
    )
    ax.annotate(
        "old", xy=(OLD_MAX_LINK_M * 0.80,
                   100.0 * old_model_loss(OLD_MAX_LINK_M * 0.80)),
        xytext=(-10, 6), textcoords="offset points", ha="right",
        color=OLD_COLOR, fontsize=11, fontweight="bold", zorder=6,
    )

    # The old model's wall: loss is 100% from here on, and no link exists.
    ax.annotate(
        f"old model's wall\nat {OLD_MAX_LINK_M:.0f} m",
        xy=(OLD_MAX_LINK_M, 100.0), xytext=(OLD_MAX_LINK_M - 18, 86),
        textcoords="data", ha="right", va="top",
        color=OLD_COLOR, fontsize=8.5, zorder=6,
    )

    # Reference distances. The labels run ALONG their lines: a rotated label is
    # only a few pixels wide, so it fits in the gaps between the two curves
    # where a horizontal one would lie across them.
    label_heights = [55.0, 80.0, 48.0, 70.0]
    for (dist, label), height in zip(markers(), label_heights):
        ax.axvline(dist, color=TEXT_SECONDARY, linewidth=0.9,
                   linestyle=":", zorder=2)
        ax.text(
            dist - 7.0, height, f"{label}  ·  {dist:.0f} m",
            color=TEXT_SECONDARY, fontsize=8.5, rotation=90,
            ha="center", va="center", zorder=6,
        )

    ax.set_xlim(0, 800)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Distance between sender and receiver (m)",
                  color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("Packet loss (%)", color=TEXT_SECONDARY, fontsize=11)
    ax.set_title(
        "Channel loss vs distance — new logistic model against the old FSPL one",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14, loc="left",
    )

    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

    # Lower right: the only region both curves and all four markers leave clear.
    legend = ax.legend(loc="lower right", frameon=False, fontsize=10,
                       labelcolor=TEXT_SECONDARY)
    legend.set_zorder(7)

    fig.text(
        0.01, 0.015,
        f"new: 1/(1+exp(-{config.LOSS_SLOPE_PER_M}·(d-"
        f"{config.LOSS_50_DISTANCE_M:.0f}))), Rosati et al. arXiv:1406.4399   |   "
        f"link exists while loss < {config.LINK_MAX_LOSS:.2f}   |   "
        f"area {config.WIDTH:.0f}x{config.HEIGHT:.0f} m, GS at centre",
        color=TEXT_SECONDARY, fontsize=8,
    )

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    """Entry point: write the figure and print a table of both curves."""
    args = parse_args()
    path = plot(args.out)

    print(f"wrote {path}")
    print()
    print("  loss at selected distances")
    print("  " + "-" * 44)
    print(f"  {'dist(m)':>8}{'new':>12}{'old':>12}")
    for dist in [50, 100, 150, 200, 250, 270, 300, 356, 400, 450,
                 500, 540, 636, 800]:
        print(f"  {dist:>8}{channel.p_loss(float(dist)):>12.4f}"
              f"{old_model_loss(float(dist)):>12.4f}")
    print("  " + "-" * 44)
    print(f"  link cutoff (loss < {config.LINK_MAX_LOSS:.2f}): "
          f"{channel.max_link_distance():.1f} m"
          f"   (old model: {OLD_MAX_LINK_M:.1f} m)")


if __name__ == "__main__":
    main()
