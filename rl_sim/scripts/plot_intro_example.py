"""
plot_intro_example.py — Figure 1 of the report: GS, one C-drone, one M-drone.

The M-drone is 500 m from the GS, beyond the link edge, so it has no direct
link. A C-drone halfway between them turns that into two 250 m hops. Every
number on the figure comes from the channel code (channel.p_loss and
channel.link_exists), not typed in.

Writes:
    out/intro_example.png                      with the settings stamp
    ../docs/report/figures/intro_example.png   without it (--report)

Usage:
    python scripts/plot_intro_example.py [--report]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.report_figures import (C_COLOR, GS_COLOR, M_COLOR, SURFACE,
                                    TEXT_PRIMARY, TEXT_SECONDARY, _stamp)

M_DISTANCE_M: float = 500.0
C_DISTANCE_M: float = M_DISTANCE_M / 2.0


def _loss_label(distance: float) -> str:
    if not channel.link_exists(distance):
        return f"{distance:.0f} m: no link"
    return f"{distance:.0f} m, {100.0 * channel.p_loss(distance):.1f}% loss"


def draw(path: str, stamp: bool) -> None:
    edge = config.LOSS_50_DISTANCE_M
    fig, ax = plt.subplots(figsize=(8.0, 3.6 if stamp else 3.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # The GS's reach.
    ax.add_patch(plt.Circle((0.0, 0.0), edge, fill=False, ls="--",
                            color=TEXT_SECONDARY, lw=1.0))
    ax.text(edge - 8, edge * 0.40, f"GS reach\n{edge:.0f} m",
            color=TEXT_SECONDARY, fontsize=9, ha="right", va="top")

    # Direct link (does not exist) and the two hops through the C-drone.
    ax.plot([0, M_DISTANCE_M], [-70, -70], color=TEXT_SECONDARY, lw=1.0,
            ls=":")
    ax.text(C_DISTANCE_M * 0.7, -95, "direct: " + _loss_label(M_DISTANCE_M),
            color=TEXT_SECONDARY, fontsize=9, ha="center", va="top")
    for a, b in ((0.0, C_DISTANCE_M), (C_DISTANCE_M, M_DISTANCE_M)):
        ax.annotate("", xy=(a + 18, 0), xytext=(b - 18, 0),
                    arrowprops=dict(arrowstyle="-|>", color=TEXT_PRIMARY,
                                    lw=1.4))
        ax.text((a + b) / 2, 22, _loss_label(b - a), color=TEXT_PRIMARY,
                fontsize=9, ha="center", zorder=6,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    ax.scatter([0], [0], marker="s", s=170, color=GS_COLOR, zorder=5)
    ax.scatter([C_DISTANCE_M], [0], marker="^", s=170, color=C_COLOR,
               zorder=5)
    ax.scatter([M_DISTANCE_M], [0], marker="o", s=150, color=M_COLOR,
               zorder=5)
    for x, label in ((0, "GS"), (C_DISTANCE_M, "C-drone"),
                     (M_DISTANCE_M, "M-drone")):
        ax.text(x, -40, label, ha="center", va="top", fontsize=10,
                color=TEXT_PRIMARY)

    ax.set_xlim(-60, M_DISTANCE_M + 60)
    ax.set_ylim(-edge * 0.45, edge * 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    if stamp:
        fig.subplots_adjust(bottom=0.16)
        _stamp(fig)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--report", action="store_true",
                        help="also write the unstamped report copy")
    args = parser.parse_args()
    for d in (M_DISTANCE_M, C_DISTANCE_M):
        print(f"  {_loss_label(d)}  (channel.p_loss / channel.link_exists)")
    draw(os.path.join(args.out_dir, "intro_example.png"), stamp=True)
    if args.report:
        draw(os.path.join("..", "docs", "report", "figures",
                          "intro_example.png"), stamp=False)


if __name__ == "__main__":
    main()
