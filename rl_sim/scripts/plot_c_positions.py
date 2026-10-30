"""
plot_c_positions.py — Phase 2c M3: where the C-drones end up.

The validation table says the learned policy loses fewer packets than the
midpoint rule. It does not say what the drones do differently. This draws
two validation layouts twice, once under each policy: where every C-drone
started, where it finished, and the mission drones and ground station it
was arranging itself around.

Writes:
    out/2c_m3_positions.png
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

#: Which validation layouts to draw. The first two, so nothing is cherry
#: picked.
LAYOUTS: Tuple[int, ...] = VALIDATION_LAYOUTS[:2]


def run_and_track(layout: int, run_seed: int, policy: str,
                  state: Any = None) -> Dict[str, Any]:
    """Run one episode and return the geometry it ended in.

    Args:
        layout:   Placement seed.
        run_seed: Channel and traffic seed.
        policy:   "midpoint" or "rl".
        state:    Actor weights, for "rl".

    Returns:
        Start and end positions, the mission drones, the ground station and
        the run's total loss.
    """
    from scripts.metrics_1a import compute_metrics

    previous = _world()
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

        starts = {drone.drone_id: drone.position.copy()
                  for drone in env.drones if drone.drone_type == "C"}
        missions = [drone.position.copy() for drone in env.drones
                    if drone.drone_type == "M"]

        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        ends = {drone.drone_id: drone.position.copy()
                for drone in env.drones if drone.drone_type == "C"}
        truth = compute_metrics(env)["ground_truth"]
        return {
            "starts": starts, "ends": ends, "missions": missions,
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

    from scripts.stamp import stamp_lines

    surface, grid = "#fcfcfb", "#e4e3dd"
    text, faint = "#2b2a26", "#5c5b54"
    mission, start, end = "#b9b7ae", "#f0b166", "#2a78d6"

    columns = len(LAYOUTS)
    fig, axes = plt.subplots(2, columns, figsize=(5.4 * columns, 10.6),
                             dpi=150)
    fig.patch.set_facecolor(surface)
    axes = np.atleast_2d(axes)

    for panel in panels:
        ax = axes[panel["row"]][panel["column"]]
        ax.set_facecolor(surface)
        ax.scatter([p[0] for p in panel["missions"]],
                   [p[1] for p in panel["missions"]],
                   s=34, color=mission, marker="o", zorder=2,
                   label="M-drone")
        ax.scatter([panel["gs"][0]], [panel["gs"][1]], s=240, marker="*",
                   color="#e34948", zorder=5, label="ground station")

        for drone_id, first in panel["starts"].items():
            last = panel["ends"][drone_id]
            ax.annotate("", xy=(last[0], last[1]),
                        xytext=(first[0], first[1]),
                        arrowprops=dict(arrowstyle="-|>", color=faint,
                                        linewidth=1.0, alpha=0.55,
                                        shrinkA=3, shrinkB=5),
                        zorder=3)
        ax.scatter([p[0] for p in panel["starts"].values()],
                   [p[1] for p in panel["starts"].values()],
                   s=52, facecolors="none", edgecolors=start, linewidths=1.8,
                   zorder=4, label="C-drone start")
        ax.scatter([p[0] for p in panel["ends"].values()],
                   [p[1] for p in panel["ends"].values()],
                   s=74, color=end, marker="D", zorder=6,
                   label="C-drone end")

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

    # The legend goes under the panels, not inside one: in the corner it
    # covered two of the mission drones it was labelling.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9.5, labelcolor=faint,
               handletextpad=0.5, columnspacing=2.4,
               bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Phase 2c M3 — where the C-drones settle", color=text,
                 fontsize=13, fontweight="bold", x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.085, 1, 0.975))
    fig.text(0.01, 0.01, stamp_lines(width=112, loads_ms=[200]),
             color=faint, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    print(f"  wrote {path}")


def main() -> None:
    """Draw the two layouts under midpoint and under the learned policy."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M3: C-drone end positions, RL against midpoint."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed to draw; default is the kept "
                             "actor with the lowest validation loss")
    args = parser.parse_args()

    import csv

    import torch

    compare_path = os.path.join(args.out_dir, "2c_m3_compare.csv")
    if not os.path.exists(compare_path):
        print(f"STOP: no validation table at {compare_path}")
        raise SystemExit(1)

    seed = args.seed
    if seed is None:
        with open(compare_path) as handle:
            rows = list(csv.DictReader(line for line in handle
                                       if not line.startswith("#")))
        scores: Dict[int, List[float]] = {}
        for row in rows:
            if row["policy"].startswith("rl seed"):
                which = int(row["policy"].split()[-1])
                scores.setdefault(which, []).append(float(row["total_loss"]))
        seed = min(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
    print(f"  drawing training seed {seed}")

    model = os.path.join(args.out_dir, "models", f"c_actor_m3_seed{seed}.pt")
    if not os.path.exists(model):
        print(f"STOP: no model at {model}")
        raise SystemExit(1)
    state = torch.load(model, weights_only=True)
    # The kept file is the Gaussian actor; the deterministic policy is its
    # mean network, which is what the evaluation runs.
    state = {key[len("mean."):]: value for key, value in state.items()
             if key.startswith("mean.")} or state

    panels: List[Dict[str, Any]] = []
    for column, layout in enumerate(LAYOUTS):
        run_seed = validation_run_seed(layout)
        for row, (policy, label) in enumerate((("midpoint", "midpoint rule"),
                                               ("rl", f"RL seed {seed}"))):
            print(f"  layout {layout}, {policy} ...", flush=True)
            panel = run_and_track(layout, run_seed, policy,
                                  state if policy == "rl" else None)
            panel["row"] = row
            panel["column"] = column
            panel["title"] = (f"layout {layout} — {label}\n"
                              f"total loss {panel['total_loss']:.2f}%")
            panels.append(panel)

    draw(panels, os.path.join(args.out_dir, "2c_m3_positions.png"))


if __name__ == "__main__":
    main()
