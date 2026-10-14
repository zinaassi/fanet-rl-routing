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
GOOD_LINK_MAX_LOSS = 0.10   # the "good link" threshold, as in range_check
M_COLOR = "#2a78d6"
C_COLOR = "#1baf7a"
GS_COLOR = "#eb6834"
LINK_COLOR = "#b9c6d4"

# Figures the report shares with the rest of the project.
#: Figures the report shares with the rest of the project. Every one must
#: exist in out/; a missing name is reported rather than passed over, so the
#: report folder can never quietly fall behind.
COPIED = (
    "ploss_curve.png",
    "1a_total_loss_vs_load.png",
    "1a_loss_heatmaps.png",
    "1a_link_matrix_placement1.png",
    "1b_learning_curve.png",
    "1b_learning_curve_seeds.png",
    "1b_validation_causes.png",
    "1b_comparison_total_loss_50.png",
    "1b_loss_heatmaps_50.png",
    "queue_size_total_loss.png",
    "queue_size_load_curve.png",
)

#: Figures this script draws itself.
OWN = ("example_layout.png", "step_schematic.png")

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
# Figure: an example layout, good links and lossy links side by side
# ---------------------------------------------------------------------------

def plot_example_layout(path: str) -> Dict[str, float]:
    """Draw one accepted placement twice: good links, then lossy ones.

    The two panels split the same layout by link quality. A "good" link loses
    less than ``GOOD_LINK_MAX_LOSS``; a lossy one is everything else that
    still counts as a link. Both thresholds are read from the channel code,
    never written down here, so the figure cannot disagree with the model.

    Args:
        path: Output PNG path.

    Returns:
        The numbers quoted in the caption, so the text can cite measured
        values rather than remembered ones.
    """
    import networkx as nx

    from fanet_sim.envs.channel import euclidean_distance
    from fanet_sim.envs.fanet_env import FANETEnv

    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=EXAMPLE_SEED, run_seed=EXAMPLE_SEED)
    env.reset()

    edge = channel.max_link_distance()
    good_edge = channel.distance_for_loss(GOOD_LINK_MAX_LOSS)
    positions = [d.position for d in env.drones]
    n_m = config.NUM_M_DRONES

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    # Reversed, so a LIGHTER line means MORE loss: a faint link reads as a
    # weak one.
    colormap = plt.get_cmap("YlOrRd_r")
    counts = {"good": 0, "lossy": 0}
    lossy_image = None

    for ax, kind in zip(axes, ("good", "lossy")):
        ax.set_facecolor(SURFACE)

        # Drone-to-drone links, then the GS links, then the drones on top.
        for i, a in enumerate(env.drones):
            for b in list(env.drones)[i + 1:]:
                dist = euclidean_distance(a.position, b.position)
                if not channel.link_exists(dist):
                    continue
                loss = channel.p_loss(dist)
                if (loss < GOOD_LINK_MAX_LOSS) != (kind == "good"):
                    continue
                counts[kind] += 1
                if kind == "good":
                    colour, width = LINK_COLOR, 1.0
                else:
                    # Lighter means more loss, over the lossy band only.
                    span = config.LINK_MAX_LOSS - GOOD_LINK_MAX_LOSS
                    colour = colormap(
                        0.10 + 0.75 * (loss - GOOD_LINK_MAX_LOSS) / span
                    )
                    width = 1.4
                ax.plot([a.position[0], b.position[0]],
                        [a.position[1], b.position[1]],
                        color=colour, linewidth=width, zorder=2)

        for drone in env.drones:
            dist = euclidean_distance(drone.position, env.gs_position)
            if not channel.link_exists(dist):
                continue
            loss = channel.p_loss(dist)
            if (loss < GOOD_LINK_MAX_LOSS) != (kind == "good"):
                continue
            counts[kind] += 1
            ax.plot([drone.position[0], env.gs_position[0]],
                    [drone.position[1], env.gs_position[1]],
                    color=GS_COLOR, linewidth=1.4, alpha=0.8, zorder=3)

        for drone in env.drones:
            colour = M_COLOR if drone.drone_type == "M" else C_COLOR
            ax.scatter(*drone.position, s=95, color=colour, zorder=4,
                       edgecolors=SURFACE, linewidths=1.0)

        ax.add_patch(plt.Circle(tuple(env.gs_position), edge, fill=False,
                                linestyle="--", linewidth=1.3,
                                color=TEXT_SECONDARY, alpha=0.6, zorder=3))

        # The GS goes on last and large, with a dark edge, so a drone sitting
        # next to it cannot hide it.
        ax.scatter(*env.gs_position, s=620, marker="*", color=GS_COLOR,
                   zorder=9, edgecolors=TEXT_PRIMARY, linewidths=1.6)

        ax.set_xlim(-30, config.WIDTH + 30)
        ax.set_ylim(-30, config.HEIGHT + 30)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)", color=TEXT_SECONDARY, fontsize=11)
        title = (f"(a) Good links: loss under {GOOD_LINK_MAX_LOSS:.0%} "
                 f"(up to {good_edge:.0f} m)" if kind == "good"
                 else f"(b) Lossy links: {GOOD_LINK_MAX_LOSS:.0%} to "
                      f"{config.LINK_MAX_LOSS:.0%} loss "
                      f"({good_edge:.0f}-{edge:.0f} m)")
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12,
                     fontweight="bold", pad=10)
        _style(ax)

    axes[0].set_ylabel("y (m)", color=TEXT_SECONDARY, fontsize=11)

    # One colour bar for the lossy panel only, since the good panel has no
    # scale to read.
    norm = matplotlib.colors.Normalize(vmin=GOOD_LINK_MAX_LOSS * 100,
                                       vmax=config.LINK_MAX_LOSS * 100)
    lossy_image = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
    bar = fig.colorbar(lossy_image, ax=axes[1], fraction=0.045, pad=0.03)
    bar.set_label("loss per transmission (%)", color=TEXT_SECONDARY,
                  fontsize=10)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    bar.outline.set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=M_COLOR,
                   markersize=9, label=f"M-drone ({config.NUM_M_DRONES})"),
        plt.Line2D([], [], marker="o", linestyle="", color=C_COLOR,
                   markersize=9, label=f"C-drone ({config.NUM_C_DRONES})"),
        plt.Line2D([], [], marker="*", linestyle="", color=GS_COLOR,
                   markeredgecolor=TEXT_PRIMARY, markersize=18,
                   label="ground station"),
        plt.Line2D([], [], color=GS_COLOR, linewidth=1.6,
                   label="link to the ground station"),
        plt.Line2D([], [], color=TEXT_SECONDARY, linewidth=1.3,
                   linestyle="--",
                   label=f"ground-station reach ({edge:.0f} m)"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=10,
               labelcolor=TEXT_SECONDARY, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.085))

    fig.suptitle(
        f"Example layout: {config.NUM_M_DRONES} M-drones, "
        f"{config.NUM_C_DRONES} C-drones, "
        f"{config.WIDTH:.0f} x {config.HEIGHT:.0f} m",
        color=TEXT_PRIMARY, fontsize=14, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.175, 1, 0.95))
    _stamp(fig, placements=1)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")

    # --- the numbers the caption quotes, measured not remembered.
    n_all = len(env.drones)
    good_degree = 0
    all_degree = 0
    good_graph = nx.Graph()
    good_graph.add_nodes_from(range(n_all))
    good_graph.add_node("GS")
    for i, a in enumerate(env.drones):
        for j, b in enumerate(env.drones):
            if i == j:
                continue
            dist = euclidean_distance(a.position, b.position)
            if not channel.link_exists(dist):
                continue
            all_degree += 1
            if channel.p_loss(dist) < GOOD_LINK_MAX_LOSS:
                good_degree += 1
                if i < j:
                    good_graph.add_edge(i, j)
    for i, a in enumerate(env.drones):
        dist = euclidean_distance(a.position, env.gs_position)
        if channel.link_exists(dist) and channel.p_loss(dist) < GOOD_LINK_MAX_LOSS:
            good_graph.add_edge(i, "GS")

    reachable = nx.node_connected_component(good_graph, "GS")
    m_good_reach = sum(1 for i in range(n_m) if i in reachable)

    stats = {
        "good_links": float(counts["good"]),
        "lossy_links": float(counts["lossy"]),
        "good_neighbours": good_degree / n_all,
        "all_neighbours": all_degree / n_all,
        "m_reaching_gs_via_good": float(m_good_reach),
        "edge_m": edge,
        "good_edge_m": good_edge,
    }
    print(f"    links drawn: {counts['good']:.0f} good, "
          f"{counts['lossy']:.0f} lossy")
    print(f"    neighbours per drone: {stats['good_neighbours']:.2f} counting "
          f"good links only, {stats['all_neighbours']:.2f} counting all links")
    print(f"    M-drones that can reach the GS using only good links: "
          f"{m_good_reach} of {n_m}")
    return stats


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


def prune(figures_dir: str, keep: Sequence[str]) -> List[str]:
    """Delete figures in the report folder that are no longer produced.

    Without this, a figure made with a superseded model would sit in the
    report folder looking current.

    Args:
        figures_dir: The report's figures folder.
        keep:        Every filename this script produces.

    Returns:
        The names deleted.
    """
    removed: List[str] = []
    for name in sorted(os.listdir(figures_dir)):
        if not name.endswith(".png") or name in keep:
            continue
        os.remove(os.path.join(figures_dir, name))
        removed.append(name)
    return removed


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
    plot_step_schematic(os.path.join(figures_dir, "step_schematic.png"))

    missing = copy_shared(args.out_dir, figures_dir)
    if missing:
        print("\n  NOT copied (not generated yet): " + ", ".join(missing))

    removed = prune(figures_dir, OWN + COPIED)
    if removed:
        print("  deleted (no longer produced): " + ", ".join(removed))

    # Printed so the report's text can quote measured values.
    print(f"\n  example layout (seed {EXAMPLE_SEED}): link edge "
          f"{stats['edge_m']:.0f} m, good links out to "
          f"{stats['good_edge_m']:.1f} m")
    print(f"    {stats['good_links']:.0f} good links and "
          f"{stats['lossy_links']:.0f} lossy ones")
    print(f"    {stats['good_neighbours']:.2f} neighbours per drone over good "
          f"links, {stats['all_neighbours']:.2f} over all links")
    print(f"    {stats['m_reaching_gs_via_good']:.0f} of "
          f"{config.NUM_M_DRONES} M-drones reach the GS over good links alone")


if __name__ == "__main__":
    main()
