"""
plot_c_paths.py — Phase 2c M4: where the C-drones go while the mission moves.

In the static world a C-drone's whole story was its start and its end. Here
the M-drones are crossing the arena, so what matters is the path: whether a
C-drone chases the mission, anticipates it, or settles early and stays. This
draws two validation layouts twice, once under the midpoint rule and once
under the chosen actor, with every drone's track coloured by time.

Writes:
    out/2c_m4_paths.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from fanet_sim.envs.c_scripted import MidpointAgent
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.train_c_ppo import ROUTING, _restore, _world
from scripts.train_c_ppo_m3 import VALIDATION_LAYOUTS, validation_run_seed
from scripts.train_c_ppo_m4 import MODEL_DIR

#: Which validation layouts to draw. The first two, so nothing is cherry
#: picked.
LAYOUTS: Tuple[int, ...] = VALIDATION_LAYOUTS[:2]


def run_and_track(layout: int, run_seed: int, policy: str,
                  state: Any = None) -> Dict[str, Any]:
    """Run one episode in the moving world and return every track.

    Args:
        layout:   Placement seed.
        run_seed: Channel and traffic seed.
        policy:   "midpoint" or "rl".
        state:    Actor weights, for "rl".

    Returns:
        The C-drone and M-drone tracks, the ground station and the loss.
    """
    from scripts.metrics_1a import compute_metrics

    previous = _world(moving=True)
    try:
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed)
        env.reset()

        if policy == "midpoint":
            env.c_agent = MidpointAgent(env)
        else:
            import torch

            from agents.c_actor import CActor, WaypointAgent

            actor = CActor()
            actor.load_state_dict(state)
            env.c_agent = WaypointAgent(actor)

        c_track: Dict[int, List[np.ndarray]] = {}
        m_track: Dict[int, List[np.ndarray]] = {}
        for _ in range(config.MAX_STEPS):
            for drone in env.drones:
                where = (c_track if drone.drone_type == "C" else m_track)
                where.setdefault(drone.drone_id, []).append(
                    drone.position.copy()
                )
            env.step()
        env.close_logger()

        truth = compute_metrics(env)["ground_truth"]
        return {
            "c_track": {k: np.array(v) for k, v in c_track.items()},
            "m_track": {k: np.array(v) for k, v in m_track.items()},
            "gs": env.gs_position.copy(),
            "total_loss": (truth["total_loss"] or 0.0) * 100,
        }
    finally:
        _restore(previous)


def draw(panels: Sequence[Dict[str, Any]], path: str) -> None:
    """Draw one panel per (layout, policy).

    Args:
        panels: What run_and_track returned, plus a title for each.
        path:   Output PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    from scripts.stamp import stamp_lines

    surface, grid = "#fcfcfb", "#e4e3dd"
    text, faint = "#2b2a26", "#5c5b54"
    mission = "#c9c7be"

    columns = len(LAYOUTS)
    fig, axes = plt.subplots(2, columns, figsize=(5.6 * columns, 11.0),
                             dpi=150)
    fig.patch.set_facecolor(surface)
    axes = np.atleast_2d(axes)
    colours = plt.get_cmap("viridis")

    for panel in panels:
        ax = axes[panel["row"]][panel["column"]]
        ax.set_facecolor(surface)

        # The mission drones first and in grey: they are the thing being
        # served, not the thing being decided.
        for track in panel["m_track"].values():
            ax.plot(track[:, 0], track[:, 1], color=mission, linewidth=1.0,
                    zorder=2)
            ax.scatter([track[-1, 0]], [track[-1, 1]], s=26, color=mission,
                       zorder=3)

        for track in panel["c_track"].values():
            points = track.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            fading = LineCollection(
                segments, cmap=colours,
                norm=plt.Normalize(0, len(segments)),
                linewidth=2.0, zorder=5,
            )
            fading.set_array(np.arange(len(segments)))
            ax.add_collection(fading)
            ax.scatter([track[0, 0]], [track[0, 1]], s=48, facecolors="none",
                       edgecolors=colours(0.0), linewidths=1.8, zorder=6)
            ax.scatter([track[-1, 0]], [track[-1, 1]], s=70, marker="D",
                       color=colours(1.0), zorder=7)

        ax.scatter([panel["gs"][0]], [panel["gs"][1]], s=240, marker="*",
                   color="#e34948", zorder=8)

        ax.set_title(panel["title"], color=text, fontsize=11,
                     fontweight="bold", loc="left", pad=8)
        ax.set_xlim(0, config.WIDTH)
        ax.set_ylim(0, config.HEIGHT)
        ax.set_aspect("equal")
        ax.grid(True, color=grid, linewidth=0.7, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=faint, labelsize=8)

    # The legend goes under the panels: inside one it would sit on a track.
    handles = [
        Line2D([], [], color=mission, linewidth=1.4, label="M-drone track"),
        Line2D([], [], marker="o", linestyle="none", markersize=8,
               markerfacecolor="none", markeredgecolor=colours(0.0),
               label="C-drone start"),
        Line2D([], [], color=colours(0.55), linewidth=2.4,
               label="C-drone track, dark to light with time"),
        Line2D([], [], marker="D", linestyle="none", markersize=8,
               color=colours(1.0), label="C-drone end"),
        Line2D([], [], marker="*", linestyle="none", markersize=15,
               color="#e34948", label="ground station"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=faint,
               handletextpad=0.5, columnspacing=1.8,
               bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Phase 2c M4 — C-drone paths while the mission flies",
                 color=text, fontsize=13, fontweight="bold", x=0.012,
                 ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.085, 1, 0.975))
    fig.text(0.01, 0.01, stamp_lines(width=112, loads_ms=[200]),
             color=faint, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    """Draw the two layouts under midpoint and under the chosen actor."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M4: C-drone paths, RL against midpoint."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--run", type=str, default=None,
                        help="M4 run to draw; default is the one with the "
                             "lowest validation loss")
    args = parser.parse_args()

    import torch

    from scripts.evaluate_2c_m4_test import best_run, validation_means

    means = validation_means(os.path.join(args.out_dir, "2c_m4_compare.csv"))
    label = f"m4 run {args.run}" if args.run else best_run(means)
    name = label.split()[-1]

    model = os.path.join(args.out_dir, MODEL_DIR, f"c_actor_m4_{name}.pt")
    if not os.path.exists(model):
        print(f"STOP: no model at {model}")
        raise SystemExit(1)
    state = torch.load(model, weights_only=True)
    # The kept file is the Gaussian actor; the deterministic policy that the
    # evaluation runs is its mean network.
    state = {key[len("mean."):]: value for key, value in state.items()
             if key.startswith("mean.")} or state

    panels: List[Dict[str, Any]] = []
    for column, layout in enumerate(LAYOUTS):
        run_seed = validation_run_seed(layout)
        for row, (policy, shown) in enumerate((("midpoint", "midpoint rule"),
                                               ("rl", label))):
            print(f"  layout {layout}, {policy} ...", flush=True)
            panel = run_and_track(layout, run_seed, policy,
                                  state if policy == "rl" else None)
            panel["row"] = row
            panel["column"] = column
            panel["title"] = (f"layout {layout} — {shown}\n"
                              f"total loss {panel['total_loss']:.2f}%")
            panels.append(panel)

    draw(panels, os.path.join(args.out_dir, "2c_m4_paths.png"))


if __name__ == "__main__":
    main()
