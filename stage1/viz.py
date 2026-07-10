"""Routing visualization: who forwards to whom, for one frozen topology.

    python -m stage1.viz --layout ring --k 8 --topology 0
    python -m stage1.viz --layout random --k 16 --topology 3 --routers greedy dijkstra

One panel per router: all in-range links (light gray), each drone's chosen
next hop (colored arrows), the hard 250 m communication circle around the
GS, and unreachable drones (hollow, red ring). Panel titles carry the PDR
of one simulated episode (seeded exactly like realization 0 in
``stage1.evaluate``), and a per-router summary table is printed to stdout,
so you can check the routers do what they should before a full run.

Output: ``<out-dir>/routes_<layout>_k<k>_t<topology>.png``.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from . import config, metrics, sim, world
from .plots import GRID, INK, INK_2, MUTED, ROUTER_COLOR, SURFACE, _style
from .routing import make_router, routed_drones, routing_table

UNREACHABLE = "#b3372b"  # status color: drone whose chain never reaches the GS

_TOPOLOGY_STREAM = 0  # must match stage1.evaluate
_CHANNEL_STREAM = 1


def _channel_seed(base_seed: int, layout: str, k: float, topology: int, r_idx: int):
    """Same seed as realization 0 of stage1.evaluate for this cell."""
    layout_idx = config.LAYOUTS.index(layout)
    k_idx = config.K_SWEEP.index(k) if k in config.K_SWEEP else int(k * 1000)
    return np.random.SeedSequence(
        (base_seed, _CHANNEL_STREAM, topology, 0, layout_idx, k_idx, r_idx)
    )


def _draw_panel(ax, w: world.World, router_name: str, table, routed, m) -> None:
    g = w.graph
    gs_id = g.graph["gs_id"]
    pos = w.positions
    color = ROUTER_COLOR[router_name]

    # Hard-range circle around the GS and the arena border.
    ax.add_patch(plt.Circle(
        config.GS_POS, config.RANGE_M, fill=False,
        edgecolor=MUTED, linewidth=1, linestyle=(0, (4, 3)),
    ))
    ax.add_patch(plt.Rectangle(
        (0, 0), config.AREA_SIZE_M, config.AREA_SIZE_M,
        fill=False, edgecolor=GRID, linewidth=0.8,
    ))

    # All pruned-graph links, recessive.
    segments = [(pos[u], pos[v]) for u, v in g.edges]
    ax.add_collection(LineCollection(segments, colors=GRID, linewidths=0.6, zorder=1))

    # Chosen next hops: one arrow per routed decision.
    for u, v in table.items():
        if v is None:
            continue
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0), zorder=3,
            arrowprops=dict(
                arrowstyle="-|>", color=color, linewidth=1.4,
                shrinkA=4, shrinkB=5, mutation_scale=9,
            ),
        )

    # Nodes: M = circle, C = square, GS = star; unreachable = hollow + red ring.
    m_ids = range(config.N_M_DRONES)
    c_ids = range(config.N_M_DRONES, config.N_DRONES)
    for ids, marker, size in ((m_ids, "o", 30), (c_ids, "s", 40)):
        ok = [i for i in ids if i in routed]
        bad = [i for i in ids if i not in routed]
        if ok:
            ax.scatter(pos[ok, 0], pos[ok, 1], marker=marker, s=size,
                       c=INK_2, edgecolors=SURFACE, linewidths=0.8, zorder=4)
        if bad:
            ax.scatter(pos[bad, 0], pos[bad, 1], marker=marker, s=size,
                       facecolors=SURFACE, edgecolors=UNREACHABLE,
                       linewidths=1.4, zorder=4)
    ax.scatter(*config.GS_POS, marker="*", s=220, c=INK,
               edgecolors=SURFACE, linewidths=0.8, zorder=5)

    n_m = config.N_M_DRONES
    bad_m = n_m - sum(1 for i in range(n_m) if i in routed)
    ax.set_title(
        f"{router_name} — PDR {m['pdr_global']:.3f}, "
        f"unreachable M {bad_m}/{n_m}",
        fontsize=10,
    )
    pad = 40.0
    ax.set_xlim(-pad, config.AREA_SIZE_M + pad)
    ax.set_ylim(-pad, config.AREA_SIZE_M + pad)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _print_summary(router_name: str, m: dict) -> None:
    delay = (
        f"{m['mean_delay_ms']:.0f} ms" if math.isfinite(m["mean_delay_ms"]) else "n/a"
    )
    print(
        f"  {router_name:<9}"
        f" PDR={m['pdr_global']:.3f}"
        f"  delivered={m['n_delivered']:.0f}/{m['n_emitted']:.0f}"
        f"  drop(channel)={m['n_dropped_channel']:.0f}"
        f"  drop(no_route)={m['n_dropped_no_route']:.0f}"
        f"  unreachable_M={m['unreachable_frac_m']:.0%}"
        f"  mean_delay={delay}"
    )


def render(
    layout: str,
    k: float,
    topology: int,
    routers: Sequence[str],
    n_steps: int,
    base_seed: int,
    out_dir: str,
) -> str:
    """Build the world, run one episode per router, write the figure."""
    _style()
    w = world.build_world(layout, k, (base_seed, _TOPOLOGY_STREAM, topology))
    fig, axes = plt.subplots(
        1, len(routers), figsize=(4.6 * len(routers), 5.0), squeeze=False
    )

    print(f"\nlayout={layout}  k={k:g}  topology={topology}  "
          f"({n_steps} steps, 1 episode per router)")
    if w.prune_disconnected:
        print(f"  prune disconnected drones: {list(w.prune_disconnected)}")
    for r_idx, name in enumerate(routers):
        table = routing_table(make_router(name), w.graph)
        routed = routed_drones(table, w.graph)
        rng = np.random.default_rng(_channel_seed(base_seed, layout, k, topology, r_idx))
        result = sim.run_sim(w.graph, table, rng, n_steps)
        m = metrics.sim_metrics(result, w.graph, table)
        _draw_panel(axes[0, r_idx], w, name, table, routed, m)
        _print_summary(name, m)

    handles = [
        Line2D([], [], marker="o", linestyle="", color=INK_2, label="M-drone (routed)"),
        Line2D([], [], marker="s", linestyle="", color=INK_2, label="C-drone (routed)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
               markeredgecolor=UNREACHABLE, label="unreachable (no route to GS)"),
        Line2D([], [], marker="*", linestyle="", color=INK, markersize=12, label="GS"),
        Line2D([], [], linestyle=(0, (4, 3)), color=MUTED,
               label=f"{config.RANGE_M:.0f} m range around GS"),
        Line2D([], [], color=GRID, label="in-range link (pruned graph)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               labelcolor=INK_2, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(
        f"Routing decisions — layout={layout}, k={k:g}, topology={topology} "
        f"(arrows = chosen next hop)",
        fontsize=11, color=INK,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"routes_{layout}_k{k:g}_t{topology}.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m stage1.viz",
        description="Visualize per-drone routing decisions for one topology.",
    )
    p.add_argument("--layout", default="ring", choices=config.LAYOUTS)
    p.add_argument("--k", type=float, default=config.K_SWEEP[-1])
    p.add_argument("--topology", type=int, default=0)
    p.add_argument("--routers", nargs="+", default=list(config.ROUTERS),
                   choices=config.ROUTERS)
    p.add_argument("--steps", type=int, default=config.N_STEPS)
    p.add_argument("--base-seed", type=int, default=config.BASE_SEED)
    p.add_argument("--out-dir", default=config.OUT_DIR_VIZ)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> str:
    args = parse_args(argv)
    return render(
        layout=args.layout,
        k=args.k,
        topology=args.topology,
        routers=args.routers,
        n_steps=args.steps,
        base_seed=args.base_seed,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
