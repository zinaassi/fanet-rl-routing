"""
plot_edge_cases_2c.py — Phase 2c edge cases: GIFs and snapshots.

Draws ONLY from the recordings edge_cases_2c.py saved while each run was being
scored: every drone's position per step, the simulator's env.active_links per
step, the drones in range of the GS per step, and each link's loss at that
moment. Nothing is re-simulated and nothing is placed by hand.

Links are coloured by their loss on a green-to-red scale running from 0 to
LINK_MAX_LOSS (50%), the point at which a link stops existing.

Writes, for seed SEED:
    out/2c_edge/s{n}_direct.gif, out/2c_edge/s{n}_midpoint.gif
    out/2c_edge/s{n}_snapshots.png   t = 0, 30, 60, 95 s, midpoint over direct

With --edge-dir and --label, it draws another actor's recordings instead (the
"direct" recordings edge_cases_2c.py --model saved for that actor), naming
the files and titles after the label: s{n}_<label>.gif.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, cm, colors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.edge_cases_2c import SCENARIOS
from scripts.stamp import settings_stamp

SEED: int = 9901
GIF_POLICIES = ("direct", "midpoint")
SNAPSHOT_TIMES_S = (0.0, 30.0, 60.0, 95.0)

#: One frame every this many steps, and the trail length behind a C-drone.
FRAME_EVERY: int = 5
TRAIL_STEPS: int = 50
FPS: int = 20

COLOURS = {"M": "#3a86ff", "C": "#fb8500", "GS": "#06a77d"}
LOSS_MAP = matplotlib.colormaps["RdYlGn_r"]
LOSS_NORM = colors.Normalize(vmin=0.0, vmax=config.LINK_MAX_LOSS)


def load(rec_dir: str, scenario: int, policy: str) -> Dict[str, Any]:
    """Load one recording and index its links by step."""
    rec = dict(np.load(os.path.join(rec_dir,
                                    f"s{scenario}_{policy}_{SEED}.npz")))
    steps = len(rec["pos"])
    links = rec["link"]
    gs_links = rec["gs_link"]
    rec["links_at"] = np.split(links, np.searchsorted(links[:, 0],
                                                      np.arange(1, steps)))
    rec["gs_at"] = np.split(gs_links, np.searchsorted(gs_links[:, 0],
                                                      np.arange(1, steps)))
    return rec


def setup_axis(ax: plt.Axes, rec: Dict[str, Any]) -> None:
    """The arena, the GS and its reach circle."""
    gs = rec["gs"]
    reach = channel.max_link_distance()
    ax.set_xlim(0, config.WIDTH)
    ax.set_ylim(0, config.HEIGHT)
    ax.set_aspect("equal")
    ax.set_xticks([0, 450, 900])
    ax.set_yticks([0, 450, 900])
    ax.tick_params(labelsize=6)
    ax.add_patch(plt.Circle(gs, reach, fill=False, ls="--", lw=0.6,
                            color="#9aa0a6"))
    for end in rec["m_end"]:
        ax.plot(*end, marker="x", color=COLOURS["M"], ms=5, alpha=0.5)
    ax.plot(*gs, marker="s", color=COLOURS["GS"], ms=8, zorder=5)


def draw_step(ax: plt.Axes, rec: Dict[str, Any], step: int,
              artists: Dict[str, Any]) -> None:
    """Draw (or redraw) the network at *step* from the recording."""
    pos = rec["pos"][step]
    types = rec["types"]
    segments, losses = [], []
    for _, a, b, loss in rec["links_at"][step]:
        segments.append([pos[int(a)], pos[int(b)]])
        losses.append(loss)
    for _, d, loss in rec["gs_at"][step]:
        segments.append([pos[int(d)], rec["gs"]])
        losses.append(loss)
    shade = LOSS_MAP(LOSS_NORM(np.asarray(losses))) if losses else []

    if "links" not in artists:
        artists["links"] = LineCollection([], linewidths=1.0, zorder=1)
        ax.add_collection(artists["links"])
        artists["trails"] = [ax.plot([], [], color=COLOURS["C"], lw=1.0,
                                     alpha=0.5, zorder=2)[0]
                             for i, t in enumerate(types) if t == "C"]
        artists["m"] = ax.scatter([], [], s=22, color=COLOURS["M"], zorder=4,
                                  edgecolors="white", linewidths=0.5)
        artists["c"] = ax.scatter([], [], s=34, marker="^",
                                  color=COLOURS["C"], zorder=4,
                                  edgecolors="black", linewidths=0.5)
        artists["time"] = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                                  va="top", fontsize=8)
    artists["links"].set_segments(segments)
    artists["links"].set_color(shade)

    c_index = [i for i, t in enumerate(types) if t == "C"]
    m_index = [i for i, t in enumerate(types) if t == "M"]
    first = max(0, step - TRAIL_STEPS)
    for line, c in zip(artists["trails"], c_index):
        trail = rec["pos"][first:step + 1, c]
        line.set_data(trail[:, 0], trail[:, 1])
    artists["m"].set_offsets(pos[m_index])
    artists["c"].set_offsets(pos[c_index])
    artists["time"].set_text(f"t = {step * config.TIMESTEP:5.1f} s")


def legend_handles() -> Sequence[Line2D]:
    return [
        Line2D([], [], marker="s", ls="", color=COLOURS["GS"], label="GS"),
        Line2D([], [], marker="o", ls="", color=COLOURS["M"], label="M-drone"),
        Line2D([], [], marker="^", ls="", color=COLOURS["C"], label="C-drone"),
        Line2D([], [], marker="x", ls="", color=COLOURS["M"],
               label="M end point"),
        Line2D([], [], ls="--", color="#9aa0a6", label="GS reach 350 m"),
    ]


def colourbar(fig: plt.Figure, ax: Any) -> None:
    bar = fig.colorbar(cm.ScalarMappable(norm=LOSS_NORM, cmap=LOSS_MAP),
                       ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("link loss", fontsize=7)
    bar.ax.tick_params(labelsize=6)


def make_gif(rec_dir: str, edge_dir: str, scenario: int, policy: str,
             name: str = "") -> str:
    """Draw one recording; *name* replaces *policy* in the title and file."""
    name = name or policy
    rec = load(rec_dir, scenario, policy)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    setup_axis(ax, rec)
    num_m, num_c, text = SCENARIOS[scenario]
    ax.set_title(f"Scenario {scenario}: {text} ({num_m} M, {num_c} C)\n"
                 f"{name}, run seed {SEED}", fontsize=8)
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=5.5,
              framealpha=0.8)
    colourbar(fig, ax)
    fig.text(0.5, 0.005, settings_stamp(extra=f"edge case {scenario}"),
             ha="center", fontsize=3.5, color="#666666", wrap=True)
    artists: Dict[str, Any] = {}
    frames = list(range(0, len(rec["pos"]), FRAME_EVERY))

    def update(step: int):
        draw_step(ax, rec, step, artists)
        return []

    anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path = os.path.join(edge_dir, f"s{scenario}_{name}.gif")
    anim.save(path, writer=animation.PillowWriter(fps=FPS), dpi=80)
    plt.close(fig)
    return path


def make_snapshots(rec_dir: str, edge_dir: str, scenario: int,
                   actor: str = "direct actor") -> str:
    fig, axes = plt.subplots(2, len(SNAPSHOT_TIMES_S), figsize=(11, 6.2))
    for row, policy in enumerate(("midpoint", "direct")):
        rec = load(rec_dir, scenario, policy)
        for col, t in enumerate(SNAPSHOT_TIMES_S):
            ax = axes[row, col]
            setup_axis(ax, rec)
            draw_step(ax, rec, int(round(t / config.TIMESTEP)), {})
            if col == 0:
                ax.set_ylabel(policy if policy == "midpoint" else
                              actor.replace(" actor", ""), fontsize=10)
    num_m, num_c, text = SCENARIOS[scenario]
    fig.suptitle(f"Scenario {scenario}: {text} ({num_m} M, {num_c} C), "
                 f"run seed {SEED}; top midpoint, bottom {actor}",
                 fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.03))
    colourbar(fig, axes)
    fig.text(0.5, 0.005, settings_stamp(extra=f"edge case {scenario}"),
             ha="center", fontsize=4.5, color="#666666", wrap=True)
    path = os.path.join(edge_dir, f"s{scenario}_snapshots.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c edge cases: GIFs and snapshots from recordings."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--scenarios", type=int, nargs="+",
                        default=list(SCENARIOS))
    parser.add_argument("--edge-dir", type=str, default=None,
                        help="folder holding rec/ (default <out-dir>/2c_edge)")
    parser.add_argument("--label", type=str, default=None,
                        help="name for the actor's recordings, e.g. "
                             "spacing_large")
    args = parser.parse_args()
    edge_dir = args.edge_dir or os.path.join(args.out_dir, "2c_edge")
    rec_dir = os.path.join(edge_dir, "rec")
    actor = f"{args.label} actor" if args.label else "direct actor"
    for scenario in args.scenarios:
        print(f"  wrote {make_snapshots(rec_dir, edge_dir, scenario, actor)}",
              flush=True)
        for policy in GIF_POLICIES:
            name = args.label if (args.label and policy == "direct") else ""
            print(f"  wrote "
                  f"{make_gif(rec_dir, edge_dir, scenario, policy, name)}",
                  flush=True)


if __name__ == "__main__":
    main()
