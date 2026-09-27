"""
plot_ploss.py — Plot the channel loss curve p_loss vs distance.

Draws one figure with the loss curve for three values of the decay rate k, so
the shape of the model and the effect of k can be read off directly:

    p_loss(d) = exp(-k * M(d)),   M(d) = received power - sensitivity, in dB

The curve is only defined where a link exists (M > 0); at and beyond the
effective range, p_loss is 1 by definition. The vertical marker shows that
range edge.

Usage:
    python scripts/plot_ploss.py                    # writes out/ploss_vs_distance.png
    python scripts/plot_ploss.py --out somewhere.png
    python scripts/plot_ploss.py --k 0.2 0.4 0.8    # override the k values
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

# Allow "python scripts/plot_ploss.py" from the rl_sim directory: put rl_sim
# itself on the path so "fanet_sim" imports, not just "python -m scripts...".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel

# Categorical series colours, in fixed order. Validated as a 3-slot
# categorical palette (lightness band, chroma floor, CVD separation and
# normal-vision separation all pass on the light surface).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
SURFACE = "#fcfcfb"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Plot p_loss vs distance for several k values."
    )
    parser.add_argument(
        "--k",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.8],
        help="Decay rates to draw (default: 0.2 0.4 0.8).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("out", "ploss_vs_distance.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def plot_ploss(k_values: List[float], out_path: str) -> str:
    """Draw the loss curves and write the figure to *out_path*.

    Args:
        k_values: Decay rates to draw, one line each.
        out_path: Where to write the PNG. Parent directories are created.

    Returns:
        The path written.
    """
    max_range = channel.MAX_LINK_DISTANCE_M
    # Sample densely near the range edge, where the curve turns hardest.
    dists = np.linspace(1.0, max_range, 2000)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, k in enumerate(k_values):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        losses = [channel.p_loss(float(d), k=k) for d in dists]
        is_default = abs(k - config.CHANNEL_LOSS_K) < 1e-12
        ax.plot(
            dists,
            losses,
            color=color,
            linewidth=2.4 if is_default else 2.0,
            label=f"k = {k:g}" + ("  (default)" if is_default else ""),
            zorder=3,
        )
        # Direct-label every line at one common distance, chosen where the
        # three curves are far apart vertically — so the labels stack cleanly
        # instead of colliding near the range edge.
        anchor = int(np.argmin(np.abs(dists - max_range * 0.66)))
        ax.annotate(
            f"k = {k:g}",
            xy=(dists[anchor], losses[anchor]),
            xytext=(8, 6),
            textcoords="offset points",
            color=color,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
            zorder=5,
        )

    # The range edge: beyond it there is no link and p_loss is 1 by definition.
    ax.axvline(max_range, color=TEXT_SECONDARY, linewidth=1.0, linestyle="--", zorder=2)
    # Set along the range line itself: the curves only reach this strip very
    # close to p_loss = 1, so nothing collides with it.
    ax.text(
        max_range - 6.0,
        0.60,
        f"no link beyond {max_range:.0f} m",
        color=TEXT_SECONDARY,
        fontsize=9,
        rotation=90,
        ha="center",
        va="center",
        zorder=5,
    )

    ax.set_xlim(0, max_range * 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Distance between endpoints (m)", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("p_loss  (probability a transmission is lost)",
                  color=TEXT_SECONDARY, fontsize=11)
    ax.set_title(
        "Channel loss vs distance:  p_loss = exp(-k · margin_dB)",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14, loc="left",
    )

    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10)

    legend = ax.legend(
        loc="upper left", frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY,
    )
    legend.set_zorder(6)

    fig.text(
        0.01, 0.015,
        f"FSPL channel, Pt={channel.PT_DBM:.0f} dBm, sensitivity="
        f"{channel.RX_SENSITIVITY_DBM:.0f} dBm, f={channel.F_HZ/1e9:.1f} GHz",
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
    """Entry point: parse args and write the figure."""
    args = parse_args()
    path = plot_ploss(args.k, args.out)

    print(f"wrote {path}")
    print()
    print("  p_loss at selected distances")
    print("  " + "-" * 52)
    header = "  dist(m)   margin(dB)" + "".join(f"   k={k:<8g}" for k in args.k)
    print(header)
    for d in [10.0, 50.0, 100.0, 150.0, 200.0, 240.0, 249.0]:
        row = f"  {d:7.0f}   {channel.link_margin_db(d):10.2f}"
        for k in args.k:
            row += f"   {channel.p_loss(d, k=k):<10.3g}"
        print(row)


if __name__ == "__main__":
    main()
