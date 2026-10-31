"""
plot_mission_panels.py — Phase 2c curriculum step 1: what the dial does.

One validation layout, four panels, one per setting of
config.MISSION_DISTANCE_M. Each panel shows where the M-drones started and
where they ended up, with a line between, so the dial can be seen rather
than read off a table.

The fleet starts bunched inside config.START_RADIUS_M of the ground station,
which is drawn as a circle, and the reach of a link is drawn as a second
circle: an M-drone outside it cannot talk to the GS directly, whatever lies
in between, so the panels show at a glance how much relaying each rung asks
for.

Writes:
    out/2c_curriculum1_mission.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.curriculum_step1 import DISTANCES, run_seed_for
from scripts.stamp import settings_stamp

#: Which validation layout to draw. One layout, four times.
LAYOUT: int = 301


def flights(layout: int, distance: float):
    """Return the M-drone start and end points at one D.

    Args:
        layout:   Placement seed.
        distance: D in metres.

    Returns:
        (starts, ends), each (N, 2).
    """
    previous = (config.MISSION, config.MISSION_DISTANCE_M, config.STATIC_MODE)
    config.MISSION = "near_gs"
    config.MISSION_DISTANCE_M = float(distance)
    config.STATIC_MODE = False
    try:
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed_for(layout))
        env.reset()
        starts = np.array([d.start_point for d in env.drones
                           if d.drone_type == "M"])
        ends = np.array([d.end_point for d in env.drones
                         if d.drone_type == "M"])
        env.close_logger()
        return starts, ends
    finally:
        (config.MISSION, config.MISSION_DISTANCE_M,
         config.STATIC_MODE) = previous


def draw(layout: int, distances: Sequence[float], path: str) -> None:
    """Draw one panel per distance and save the figure.

    Args:
        layout:    Placement seed.
        distances: The rungs of the dial.
        path:      Where to write the PNG.
    """
    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    reach = channel.max_link_distance()

    figure, axes = plt.subplots(1, len(distances),
                                figsize=(4.0 * len(distances), 5.1))
    for axis, distance in zip(np.atleast_1d(axes), distances):
        starts, ends = flights(layout, distance)

        axis.add_patch(plt.Circle(gs, reach, fill=False, linestyle="--",
                                  linewidth=0.9, edgecolor="#9aa0a6"))
        axis.add_patch(plt.Circle(gs, config.START_RADIUS_M, fill=False,
                                  linewidth=0.9, edgecolor="#4a90d9"))

        for start, end in zip(starts, ends):
            axis.plot([start[0], end[0]], [start[1], end[1]],
                      color="#c9ccd1", linewidth=0.9, zorder=1)
        axis.scatter(starts[:, 0], starts[:, 1], s=16, color="#4a90d9",
                     zorder=3, label="start")
        axis.scatter(ends[:, 0], ends[:, 1], s=16, color="#d1495b",
                     marker="^", zorder=3, label="end")
        axis.scatter([gs[0]], [gs[1]], s=90, marker="*", color="#2b2b2b",
                     zorder=4, label="GS")

        beyond = sum(1 for end in ends
                     if float(np.linalg.norm(end - gs)) > reach)
        axis.set_title(f"D = {distance:.0f} m\n"
                       f"{beyond} of {len(ends)} M-drones beyond "
                       f"{reach:.0f} m", fontsize=10)
        axis.set_xlim(0.0, config.WIDTH)
        axis.set_ylim(0.0, config.HEIGHT)
        axis.set_aspect("equal")
        axis.set_xlabel("x (m)", fontsize=8)
        axis.tick_params(labelsize=7)
    np.atleast_1d(axes)[0].set_ylabel("y (m)", fontsize=8)

    handles = [
        Line2D([], [], marker="o", color="#4a90d9", linestyle="",
               markersize=5, label="M-drone start"),
        Line2D([], [], marker="^", color="#d1495b", linestyle="",
               markersize=5, label="M-drone end"),
        Line2D([], [], marker="*", color="#2b2b2b", linestyle="",
               markersize=9, label="ground station"),
        Line2D([], [], color="#4a90d9", linewidth=0.9,
               label=f"start radius {config.START_RADIUS_M:.0f} m"),
        Line2D([], [], color="#9aa0a6", linewidth=0.9, linestyle="--",
               label=f"link reach {reach:.0f} m"),
    ]
    # The legend sits below the panels, and the panels are pushed up far
    # enough to clear it: at the default rect it lands on top of the x-axis
    # labels.
    figure.legend(handles=handles, loc="lower center", ncol=5,
                  fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.045))
    figure.suptitle(f"The near_gs mission on validation layout {layout}: "
                    f"the dial is how far the mission spreads", fontsize=11)
    figure.text(0.5, 0.008,
                settings_stamp(extra=f"mission=near_gs, layout {layout}"),
                ha="center", fontsize=5.0, color="#666666", wrap=True)
    figure.tight_layout(rect=(0.0, 0.14, 1.0, 0.95))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"  wrote {path}")


def main() -> None:
    """Draw the mission panels."""
    parser = argparse.ArgumentParser(
        description="Phase 2c curriculum step 1: the mission at each D."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--layout", type=int, default=LAYOUT)
    args = parser.parse_args()

    draw(args.layout, DISTANCES,
         os.path.join(args.out_dir, "2c_curriculum1_mission.png"))


if __name__ == "__main__":
    main()
