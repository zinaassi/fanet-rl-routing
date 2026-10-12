"""
report_figures.py — Every figure the written report uses, in one place.

The report under ``docs/report/`` must be regenerable from one command, so
this script does two things:

  * draws the figures that exist only for the report (an example layout, the
    step schematic, a clean channel-loss curve);
  * copies the figures the report shares with the rest of the project out of
    ``rl_sim/out/`` into ``docs/report/figures/``.

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
M_COLOR = "#2a78d6"
C_COLOR = "#1baf7a"
GS_COLOR = "#eb6834"
LINK_COLOR = "#b9c6d4"

# Figures the report shares with the rest of the project.
COPIED = (
    "1a_total_loss_vs_load.png",
    "1a_loss_heatmaps.png",
    "1a_link_matrix_placement1.png",
    "ploss_vs_distance.png",
    "1b_learning_curve_seeds.png",
    "1b_comparison_total_loss_50.png",
    "1b_loss_heatmaps_50.png",
    "queue_size_total_loss.png",
    "queue_size_load_curve.png",
)

EXAMPLE_SEED = 1


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
# Figure: an example layout
# ---------------------------------------------------------------------------

def plot_example_layout(path: str) -> Dict[str, float]:
    """Draw one accepted placement, with every link the simulator sees.

    Args:
        path: Output PNG path.

    Returns:
        The numbers quoted beside the figure in the report, so the text can
        cite a measured value rather than a remembered one.
    """
    from fanet_sim.envs.fanet_env import FANETEnv

    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=EXAMPLE_SEED, run_seed=EXAMPLE_SEED)
    env.reset()

    fig, ax = plt.subplots(figsize=(7.6, 7.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Links first, so the drones sit on top of them.
    for i, a in enumerate(env.drones):
        for b in list(env.drones)[i + 1:]:
            if channel.are_connected(a.position, b.position):
                ax.plot([a.position[0], b.position[0]],
                        [a.position[1], b.position[1]],
                        color=LINK_COLOR, linewidth=0.8, zorder=2)
    in_gs_range = 0
    for drone in env.drones:
        if channel.are_connected(drone.position, env.gs_position):
            in_gs_range += 1
            ax.plot([drone.position[0], env.gs_position[0]],
                    [drone.position[1], env.gs_position[1]],
                    color=GS_COLOR, linewidth=0.8, alpha=0.45, zorder=2)

    for drone in env.drones:
        colour = M_COLOR if drone.drone_type == "M" else C_COLOR
        ax.scatter(*drone.position, s=90, color=colour, zorder=4,
                   edgecolors=SURFACE, linewidths=1.0)
    ax.scatter(*env.gs_position, s=320, marker="*", color=GS_COLOR, zorder=5,
               edgecolors=SURFACE, linewidths=1.2)

    # The link reach, drawn once around the GS for scale.
    reach = channel.max_link_distance()
    ax.add_patch(plt.Circle(tuple(env.gs_position), reach, fill=False,
                            linestyle="--", linewidth=1.2, color=GS_COLOR,
                            alpha=0.55, zorder=3))

    ax.set_xlim(-30, config.WIDTH + 30)
    ax.set_ylim(-30, config.HEIGHT + 30)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("y (m)", color=TEXT_SECONDARY, fontsize=10)
    _style(ax)

    neighbours = [len(d.neighbors) for d in env.drones]
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=M_COLOR,
                   markersize=9, label=f"M-drone ({config.NUM_M_DRONES})"),
        plt.Line2D([], [], marker="o", linestyle="", color=C_COLOR,
                   markersize=9, label=f"C-drone ({config.NUM_C_DRONES})"),
        plt.Line2D([], [], marker="*", linestyle="", color=GS_COLOR,
                   markersize=15, label="ground station"),
        plt.Line2D([], [], color=LINK_COLOR, linewidth=1.5, label="link"),
        plt.Line2D([], [], color=GS_COLOR, linewidth=1.2, linestyle="--",
                   label=f"link reach, {reach:.0f} m"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY, loc="upper left",
              bbox_to_anchor=(0.0, 1.0))

    fig.suptitle(
        f"One accepted placement (seed {EXAMPLE_SEED}) — "
        f"{float(np.mean(neighbours)):.1f} neighbours per drone on average",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    _stamp(fig, placements=1)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")
    return {
        "mean_neighbours": float(np.mean(neighbours)),
        "min_neighbours": float(np.min(neighbours)),
        "max_neighbours": float(np.max(neighbours)),
        "in_gs_range": float(in_gs_range),
        "reach_m": reach,
    }


# ---------------------------------------------------------------------------
# Figure: the channel-loss curve
# ---------------------------------------------------------------------------

def plot_channel_curve(path: str) -> None:
    """Draw the loss curve with the link cutoff marked.

    Args:
        path: Output PNG path.
    """
    reach = channel.max_link_distance()
    distances = np.linspace(0, 700, 1400)
    losses = np.array([channel.p_loss(float(d)) for d in distances]) * 100

    fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(distances, losses, color=M_COLOR, linewidth=2.4, zorder=4)
    # Beyond the cutoff there is no link at all, so that half of the curve is
    # never actually used.
    ax.axvspan(reach, 700, color=GRID_COLOR, alpha=0.65, zorder=1)
    ax.annotate("no link: loss above the cutoff",
                (reach + 12, 92), color=TEXT_SECONDARY, fontsize=9)
    ax.axvline(reach, color=GS_COLOR, linestyle="--", linewidth=1.4, zorder=3)
    ax.axhline(config.LINK_MAX_LOSS * 100, color=GS_COLOR, linestyle="--",
               linewidth=1.4, zorder=3)

    # Offsets chosen per point: at 250 m a label to the right would run into
    # the cutoff line, and at 356 m it would sit on the shaded region's edge.
    offsets = {100: (8, 10), 250: (-10, 14), 356: (12, -16)}
    alignments = {100: "left", 250: "right", 356: "left"}
    for d in (100, 250, 356):
        loss = channel.p_loss(float(d)) * 100
        ax.scatter([d], [loss], s=45, color=M_COLOR, zorder=5,
                   edgecolors=SURFACE, linewidths=1.0)
        ax.annotate(f"{loss:.1f}% at {d} m", (d, loss),
                    textcoords="offset points", xytext=offsets[d],
                    ha=alignments[d], color=TEXT_PRIMARY, fontsize=9,
                    fontweight="bold", zorder=6)

    ax.set_xlim(0, 700)
    ax.set_ylim(0, 100)
    ax.set_xlabel("distance between the two drones (m)",
                  color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("chance one transmission is lost (%)",
                  color=TEXT_SECONDARY, fontsize=10)
    _style(ax)

    fig.suptitle(
        f"Channel loss against distance — the link reach is {reach:.0f} m",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", x=0.01, ha="left",
    )
    fig.text(0.01, 0.905,
             f"A link exists only while the loss stays under "
             f"{config.LINK_MAX_LOSS:.0%}.",
             color=TEXT_SECONDARY, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.09, 1, 0.885))
    _stamp(fig)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Figure: what happens in one step
# ---------------------------------------------------------------------------

def plot_step_schematic(path: str) -> None:
    """Draw the order of events inside one simulation step.

    The order matters and is easy to get wrong in prose: a packet that
    arrives in a step cannot be sent again until the next one, because the
    whole send snapshot is taken before any transmission.

    Args:
        path: Output PNG path.
    """
    steps = [
        ("1. Expire", "packets whose TTL ran out,\nand unacknowledged\nsources give up"),
        ("2. Create", f"each M-drone makes a packet\nevery {config.PACKET_INTERVAL_STEPS} steps,\nand routes it at once"),
        ("3. Send", f"every link queue sends its head\n({config.MAX_TX_PER_STEP} per link per step);\neach send may be lost"),
        ("4. Route arrivals", "each receiver picks the next hop\nand queues the packet,\nin a random order"),
        ("5. Sample", "queue occupancy is recorded\nonce the warm-up is over"),
    ]

    fig, ax = plt.subplots(figsize=(13.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, len(steps) * 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    for index, (title, body) in enumerate(steps):
        x = index * 10 + 0.4
        box = mpatches.FancyBboxPatch(
            (x, 2.6), 8.4, 5.2, boxstyle="round,pad=0.25",
            linewidth=1.3, edgecolor=M_COLOR, facecolor="#ffffff", zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + 4.2, 6.9, title, ha="center", va="center",
                color=TEXT_PRIMARY, fontsize=11, fontweight="bold", zorder=3)
        ax.text(x + 4.2, 4.7, body, ha="center", va="center",
                color=TEXT_SECONDARY, fontsize=9, zorder=3)
        if index < len(steps) - 1:
            ax.annotate("", xy=(x + 9.6, 5.2), xytext=(x + 8.9, 5.2),
                        arrowprops={"arrowstyle": "-|>", "color": TEXT_SECONDARY,
                                    "linewidth": 1.4})

    ax.text(len(steps) * 5, 1.4,
            "A packet that arrives in a step is queued but cannot be sent "
            "until the next one: the whole send list is taken before any "
            "transmission happens.",
            ha="center", va="center", color=TEXT_SECONDARY, fontsize=9.5)

    fig.suptitle(
        f"One simulation step ({config.TIMESTEP * 1000:.0f} ms)",
        color=TEXT_PRIMARY, fontsize=12, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _stamp(fig)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


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

    stats = plot_example_layout(os.path.join(figures_dir,
                                             "example_layout.png"))
    plot_channel_curve(os.path.join(figures_dir, "channel_loss_curve.png"))
    plot_step_schematic(os.path.join(figures_dir, "step_schematic.png"))

    missing = copy_shared(args.out_dir, figures_dir)
    if missing:
        print("\n  NOT copied (not generated yet): " + ", ".join(missing))

    # Printed so the report's text can quote measured values.
    print("\n  example layout (seed "
          f"{EXAMPLE_SEED}): reach {stats['reach_m']:.0f} m, "
          f"{stats['mean_neighbours']:.1f} neighbours on average "
          f"(min {stats['min_neighbours']:.0f}, max {stats['max_neighbours']:.0f}), "
          f"{stats['in_gs_range']:.0f} drones in range of the GS")


if __name__ == "__main__":
    main()
