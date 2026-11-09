"""
plot_report_diagrams.py — Two diagrams for the report.

    rl_loop.png        Figure 2: agent, environment, state / action / reward.
    actor_network.png  The C-drone actor. Every layer size is read from the
                       code (c_interface.SELF_FEATURES / NEIGHBOUR_FEATURES,
                       agents.c_actor hidden sizes, c_interface.ACTION_SCALE_M)
                       and checked against the network torch actually builds.

Writes each figure stamped to out/ and, with --report, unstamped to
../docs/report/figures/.

Usage:
    python scripts/plot_report_diagrams.py [--report]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from agents.c_actor import (JOINT_HIDDEN, NEIGHBOUR_HIDDEN, SELF_HIDDEN,
                            CActor)
from fanet_sim.envs import c_interface
from scripts.report_figures import (C_COLOR, GS_COLOR, M_COLOR, SURFACE,
                                    TEXT_PRIMARY, TEXT_SECONDARY, _stamp)

FIGURES_DIR = os.path.join("..", "docs", "report", "figures")


def _box(ax, xy: Tuple[float, float], w: float, h: float, text: str,
         color: str, size: float = 10) -> None:
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=color + "26", edgecolor=color,
                                lw=1.4))
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color=TEXT_PRIMARY)


def _arrow(ax, a, b, text: str = "", offset=(0.0, 0.0), rad: float = 0.0
           ) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 color=TEXT_PRIMARY, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + offset[0], (a[1] + b[1]) / 2 + offset[1],
                text, ha="center", va="center", fontsize=9.5,
                color=TEXT_SECONDARY)


def _finish(fig, ax, path: str, stamp: bool) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    if stamp:
        fig.subplots_adjust(bottom=0.18)
        _stamp(fig)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def rl_loop(path: str, stamp: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4 if stamp else 3.0))
    fig.patch.set_facecolor(SURFACE)
    _box(ax, (0.0, 0.0), 1.6, 0.7, "Agent", C_COLOR, 12)
    _box(ax, (3.6, 0.0), 1.9, 0.7, "Environment", M_COLOR, 12)
    _arrow(ax, (0.5, 0.42), (3.0, 0.42), "action", (0, 0.68), rad=-0.35)
    _arrow(ax, (3.0, -0.42), (0.5, -0.42), "state", (0, -0.66), rad=-0.35)
    _arrow(ax, (2.75, -0.2), (0.82, -0.2), "reward", (0, 0.14))
    ax.set_xlim(-1.0, 4.7)
    ax.set_ylim(-1.3, 1.3)
    _finish(fig, ax, path, stamp)


def actor_network(path: str, stamp: bool) -> None:
    nf, sf = c_interface.NEIGHBOUR_FEATURES, c_interface.SELF_FEATURES
    nh, sh, jh = NEIGHBOUR_HIDDEN, SELF_HIDDEN, JOINT_HIDDEN
    joined = 2 * nh + sh
    scale = c_interface.ACTION_SCALE_M

    # The diagram must describe the network torch builds.
    net = CActor()
    built = (net.neighbour[0].in_features, net.neighbour[0].out_features,
             net.neighbour[2].out_features, net.own[0].in_features,
             net.own[0].out_features, net.head[0].in_features,
             net.head[0].out_features, net.head[2].out_features)
    drawn = (nf, nh, nh, sf, sh, joined, jh, 2)
    if built != drawn:
        raise SystemExit(f"STOP: drawn sizes {drawn} != built {built}")

    fig, ax = plt.subplots(figsize=(10.0, 3.9 if stamp else 3.5))
    fig.patch.set_facecolor(SURFACE)
    top, bottom = 0.9, -0.9
    _box(ax, (0.0, top), 1.75, 0.75, f"each neighbour\n{nf} numbers",
         M_COLOR)
    _box(ax, (2.3, top), 1.7, 0.75, f"shared layers\n{nf} → {nh} → {nh}",
         TEXT_SECONDARY)
    _box(ax, (4.75, top), 1.95, 0.75, f"average and max\n{nh} + {nh}",
         TEXT_SECONDARY)
    _box(ax, (0.0, bottom), 1.6, 0.75, f"itself\n{sf} numbers", C_COLOR)
    _box(ax, (3.45, bottom), 1.7, 0.75, f"layer\n{sf} → {sh}", TEXT_SECONDARY)
    _box(ax, (6.9, 0.0), 1.7, 0.85, f"together\n{joined} → {jh} → 2",
         TEXT_SECONDARY)
    _box(ax, (9.55, 0.0), 2.25, 0.95,
         f"waypoint\n(x, y) within ±{scale:.0f} m\nof the GS", GS_COLOR, 9.5)
    _arrow(ax, (0.82, top), (1.43, top))
    _arrow(ax, (3.17, top), (3.75, top))
    _arrow(ax, (5.75, top - 0.2), (6.05, 0.3))
    _arrow(ax, (0.82, bottom), (2.58, bottom))
    _arrow(ax, (4.32, bottom), (6.05, -0.3))
    _arrow(ax, (7.77, 0.0), (8.4, 0.0))
    ax.text(2.3, top + 0.55, "the same layers for every neighbour",
            ha="center", fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xlim(-0.9, 10.8)
    ax.set_ylim(-1.55, 1.65)
    _finish(fig, ax, path, stamp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report diagrams.")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--report", action="store_true",
                        help="also write the unstamped report copies")
    args = parser.parse_args()
    for name, draw in (("rl_loop.png", rl_loop),
                       ("actor_network.png", actor_network)):
        draw(os.path.join(args.out_dir, name), stamp=True)
        if args.report:
            draw(os.path.join(FIGURES_DIR, name), stamp=False)


if __name__ == "__main__":
    main()
