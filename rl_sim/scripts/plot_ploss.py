"""
plot_ploss.py — The channel loss curve.

Draws ``out/ploss_curve.png``: the chance one transmission is lost against the
distance between the two drones, with the receiver sensitivity and the link
edge marked, and the link margin on a second axis across the top.

The model itself lives in :mod:`fanet_sim.envs.channel`; this script only
draws it and prints its reference table, so the figure can never disagree
with what the simulator runs.

Usage:
    python scripts/plot_ploss.py
    python scripts/plot_ploss.py --out somewhere.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.stamp import stamp_lines

CURVE_COLOR = "#2a78d6"
SENS_COLOR = "#eb6834"
EDGE_COLOR = "#1baf7a"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
SURFACE = "#fcfcfb"

X_MAX_M = 450.0
REFERENCE_DISTANCES = (50.0, 100.0, 150.0, 200.0,
                       channel.SENSITIVITY_DISTANCE_M,
                       250.0, 300.0, 322.0, config.LOSS_50_DISTANCE_M)


def _loss_text(loss_pct: float) -> str:
    """Format a loss for a point label, keeping two significant figures.

    The curve is very flat on its left now: at 100 m the loss is 0.00003%,
    which two decimals would print as "0.00%". The number of decimals follows
    the magnitude so that every label says something.

    Args:
        loss_pct: The loss as a percentage.

    Returns:
        The label text, percent sign included.
    """
    if loss_pct >= 1.0:
        return f"{loss_pct:.1f}%"
    if loss_pct >= 0.01:
        return f"{loss_pct:.2f}%"
    # Below this the digits are noise on a figure; at 100 m the loss is
    # 0.0000018%, which no reader needs to the digit.
    return "<0.01%"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Plot the channel loss curve.")
    parser.add_argument(
        "--out", type=str,
        default=os.path.join(here, "out", "ploss_curve.png"),
        help="Where the figure goes.",
    )
    return parser.parse_args()


def print_reference_table() -> None:
    """Print the model's constants and its loss at the reference distances.

    Every number is computed from :mod:`fanet_sim.envs.channel`, never copied.
    """
    print("  channel model — computed from the configured constants")
    print("  " + "-" * 56)
    print(f"  {'FSPL frequency constant':<28}{channel.FSPL_CONST_DB:>12.6f} dB")
    print(f"  {'margin-0 distance d0':<28}"
          f"{channel.SENSITIVITY_DISTANCE_M:>12.6f} m")
    print(f"  {'M(' + f'{config.LOSS_50_DISTANCE_M:.0f}' + ' m)':<28}"
          f"{channel.margin_db(config.LOSS_50_DISTANCE_M):>12.6f} dB")
    print(f"  {'A':<28}{channel.A:>12.6f}")
    print(f"  {'b':<28}{channel.B_PER_DB:>12.6f} per dB")
    print("  " + "-" * 56)
    print(f"  {'d (m)':>12}{'margin (dB)':>15}{'p_loss':>12}")
    print("  " + "-" * 56)
    for dist in REFERENCE_DISTANCES:
        print(f"  {dist:>12.4f}{channel.margin_db(dist):>15.6f}"
              f"{channel.p_loss(dist):>12.6f}")
    print("  " + "-" * 56)
    print(f"  link edge = {channel.max_link_distance():.1f} m "
          f"(p_loss < {config.LINK_MAX_LOSS})")


def plot(out_path: str, stamp: bool = True) -> str:
    """Draw the loss curve.

    Args:
        out_path: Where the PNG goes.
        stamp:    Whether to print the settings stamp along the bottom. The
                  copy in out/ keeps it, for traceability; the copy used in
                  the written report leaves it off.

    Returns:
        The path written.
    """
    edge = channel.max_link_distance()
    sensitivity = channel.SENSITIVITY_DISTANCE_M
    distances = np.linspace(1.0, X_MAX_M, 2000)
    losses = np.array([channel.p_loss(float(d)) for d in distances]) * 100.0

    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Past the edge there is no link, so that part of the curve is never used.
    ax.axvspan(edge, X_MAX_M, color=GRID_COLOR, alpha=0.8, zorder=1)
    # LOW in the shaded band. The curve climbs through this band to 84% by
    # 450 m, so the top of it is where the curve is; underneath is clear, and
    # the top is now wanted for the link-edge label.
    ax.annotate(f"not a neighbour\n(loss > {config.LINK_MAX_LOSS:.0%})",
                ((edge + X_MAX_M) / 2.0, 22.0), ha="center", va="center",
                color=TEXT_SECONDARY, fontsize=13, fontweight="bold",
                zorder=2)

    ax.plot(distances, losses, color=CURVE_COLOR, linewidth=3.2, zorder=5)

    # Back to back along the top, each on its OWN side of its marker: the two
    # marks are only 50 m apart, so stacking them put the 300 m line straight
    # through the lower label. Nothing crosses the top strip on either side —
    # left of 300 m the curve is still on the floor, and right of 350 m it
    # has not climbed past 84%.
    band_middle = (edge + X_MAX_M) / 2.0
    for dist, label, colour, at, align in (
        (sensitivity, f"sensitivity: {config.SENS_LOSS:.0%} loss",
         SENS_COLOR, (sensitivity - 6.0, 97.0), "right"),
        (edge, f"link edge: {config.LINK_MAX_LOSS:.0%} loss",
         EDGE_COLOR, (band_middle, 48.0), "center"),
    ):
        ax.axvline(dist, color=colour, linestyle="--", linewidth=2.0, zorder=4)
        ax.annotate(f"{label}\n{dist:.1f} m", at, ha=align, va="top",
                    color=colour, fontsize=12, fontweight="bold", zorder=6)

    # Where each label goes, per point: below the axis is not an option for
    # the flat part of the curve, where the label would land on the x-axis
    # tick numbers.
    label_offsets: Dict[float, Tuple[Tuple[int, int], str]] = {
        100.0: ((0, 15), "center"),
        200.0: ((0, 15), "center"),
        250.0: ((0, 18), "center"),
        300.0: ((-14, 6), "right"),
        350.0: ((-14, 6), "right"),
    }
    for dist, (offset, align) in label_offsets.items():
        loss = channel.p_loss(dist) * 100.0
        ax.scatter([dist], [loss], s=70, color=CURVE_COLOR, zorder=7,
                   edgecolors=SURFACE, linewidths=1.4)
        text = _loss_text(loss)
        ax.annotate(text, (dist, loss), textcoords="offset points",
                    xytext=offset, ha=align, color=TEXT_PRIMARY, fontsize=13,
                    fontweight="bold", zorder=8)

    ax.set_xlim(0, X_MAX_M)
    ax.set_ylim(0, 100)
    ax.set_xlabel("distance between drones (m)", color=TEXT_PRIMARY,
                  fontsize=15)
    ax.set_ylabel("loss per transmission (%)", color=TEXT_PRIMARY, fontsize=15)
    ax.grid(True, color=GRID_COLOR, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=12)

    # Second axis on top: the margin that actually drives the curve.
    offset_db = (config.PT_DBM + config.GT_DBI + config.GR_DBI
                 - config.RX_SENSITIVITY_DBM - channel.FSPL_CONST_DB)

    def _to_margin(d: np.ndarray) -> np.ndarray:
        safe = np.clip(np.asarray(d, dtype=float), 1e-6, None)
        return offset_db - 20.0 * np.log10(safe)

    def _to_distance(m: np.ndarray) -> np.ndarray:
        return 10.0 ** ((offset_db - np.asarray(m, dtype=float)) / 20.0)

    secondary = ax.secondary_xaxis("top", functions=(_to_margin, _to_distance))
    secondary.set_xlabel("link margin above sensitivity (dB)",
                         color=TEXT_PRIMARY, fontsize=15, labelpad=8)
    secondary.set_xticks([12, 8, 4, 2, 0, -1, -2, -3])
    secondary.tick_params(colors=TEXT_SECONDARY, labelsize=12)

    fig.suptitle("Packet loss vs distance (FSPL margin, S-curve)",
                 color=TEXT_PRIMARY, fontsize=17, fontweight="bold",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.075 if stamp else 0.02, 1, 0.93))
    if stamp:
        fig.text(0.01, 0.015, stamp_lines(width=150), color=TEXT_SECONDARY,
                 fontsize=8.5, va="bottom")
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main() -> None:
    """Entry point: write the figure and print the reference table."""
    args = parse_args()
    path = plot(args.out)
    print(f"wrote {path}")
    print()
    print_reference_table()


if __name__ == "__main__":
    main()
