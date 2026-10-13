"""
edge_scan.py — How the link edge changes the geometry (REPORT ONLY).

The link edge is a design choice, not a physical constant: with 1 W in free
space the radio would reach kilometres. This script measures what each
candidate edge does to the network's geometry, over many random placements,
so the edge can be chosen from geometry alone — before any routing experiment
and without any routing result to tune it against.

It changes NOTHING. config.py is not touched and no routing rule runs. The
connected-layout filter is switched off for the duration, so the layouts are
an unbiased sample rather than the accepted ones.

A link exists while the loss is below ``LINK_MAX_LOSS`` = 0.5, and the loss is
0.5 at exactly the edge, so "a link exists" is the same as "closer than the
edge". The same 3000 placements are therefore scored at every edge, which
makes the rows differ only by the edge and not by the sample.

The channel constants come from :mod:`fanet_sim.envs.channel`, so this scan
can never disagree with the model the simulator runs.

Writes:
    out/edge_scan.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.stamp import settings_stamp

EDGES_M = (300.0, 325.0, 345.0, 350.0, 360.0, 375.0, 400.0, 450.0)
PLACEMENTS = 3000
REPORT_DISTANCES = (200.0, 300.0)


def b_for_edge(edge_m: float) -> float:
    """The S-curve steepness that puts 50% loss at *edge_m*.

    Derived, never hardcoded: the sensitivity point stays at SENS_LOSS and
    the edge moves, so b follows from the edge.
    """
    return math.log(channel.A) / (-channel.margin_db(edge_m))


def p_loss_at(dist_m: float, edge_m: float) -> float:
    """Loss at *dist_m* for the curve whose 50% point is at *edge_m*."""
    return channel.p_loss(dist_m, edge_m=edge_m)


# ---------------------------------------------------------------------------
# Placements
# ---------------------------------------------------------------------------

def _draw_positions(seed: int) -> np.ndarray:
    """Return one placement's positions, using the simulator's own code.

    The connected-layout filter is off, so this is the raw placement the
    simulator would have drawn first.

    Args:
        seed: The placement seed.

    Returns:
        An array of shape (n_drones, 2).
    """
    from fanet_sim.envs.fanet_env import FANETEnv

    previous = config.REQUIRE_CONNECTED_M
    config.REQUIRE_CONNECTED_M = False
    try:
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=seed, run_seed=seed)
        env.reset()
        return np.array([d.position.copy() for d in env.drones])
    finally:
        config.REQUIRE_CONNECTED_M = previous


def _draw_batch(seeds: Sequence[int]) -> List[np.ndarray]:
    """Draw a batch of placements in one worker."""
    return [_draw_positions(seed) for seed in seeds]


def draw_all(count: int, workers: int) -> List[np.ndarray]:
    """Draw *count* placements, seeds 1..count."""
    seeds = list(range(1, count + 1))
    if workers <= 1:
        return _draw_batch(seeds)
    chunk = max(1, len(seeds) // (workers * 4))
    batches = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]
    out: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_draw_batch, batches):
            out.extend(result)
    return out


# ---------------------------------------------------------------------------
# Scoring one edge
# ---------------------------------------------------------------------------

def score(positions: Sequence[np.ndarray], edge_m: float) -> Dict[str, Any]:
    """Measure one edge over every placement.

    Args:
        positions: One (n_drones, 2) array per placement.
        edge_m:    The link edge in metres.

    Returns:
        The row for this edge.
    """
    gs = np.array(config.GS_POSITION, dtype=np.float64)
    n_m = config.NUM_M_DRONES
    n_all = config.NUM_M_DRONES + config.NUM_C_DRONES

    connected_layouts = 0
    direct_m = 0
    total_m = 0
    neighbour_counts: List[float] = []
    hop_means: List[float] = []

    for pos in positions:
        # Pairwise distances once, then everything reads off them.
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(axis=-1))
        linked = (dist < edge_m)
        np.fill_diagonal(linked, False)
        neighbour_counts.append(float(linked.sum()) / n_all)

        gs_dist = np.sqrt(((pos - gs) ** 2).sum(axis=-1))
        gs_linked = gs_dist < edge_m
        direct_m += int(gs_linked[:n_m].sum())
        total_m += n_m

        graph = nx.Graph()
        graph.add_nodes_from(range(n_all))
        graph.add_node("GS")
        for i in range(n_all):
            for j in range(i + 1, n_all):
                if linked[i, j]:
                    graph.add_edge(i, j)
            if gs_linked[i]:
                graph.add_edge(i, "GS")

        reachable = nx.node_connected_component(graph, "GS")
        if all(i in reachable for i in range(n_m)):
            connected_layouts += 1
            lengths = nx.single_source_shortest_path_length(graph, "GS")
            hop_means.append(statistics.fmean(
                lengths[i] for i in range(n_m)
            ))

    return {
        "edge_m": edge_m,
        "b_per_db": b_for_edge(edge_m),
        "connected_pct": 100.0 * connected_layouts / len(positions),
        "direct_to_gs_pct": 100.0 * direct_m / total_m,
        # Other DRONES only, matching the simulator's drone.neighbors; the
        # GS is a separate next-hop key and has its own column.
        "mean_neighbours": statistics.fmean(neighbour_counts),
        "mean_min_hops": statistics.fmean(hop_means) if hop_means else None,
        "p_loss_200m": p_loss_at(200.0, edge_m),
        "p_loss_300m": p_loss_at(300.0, edge_m),
        "placements": len(positions),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    """Print the edge table."""
    width = 104
    print()
    print(f"  Link edge against geometry — {rows[0]['placements']} random "
          f"placements, no connected-layout filter")
    print(f"  {config.NUM_M_DRONES}M + {config.NUM_C_DRONES}C in "
          f"{config.WIDTH:.0f}x{config.HEIGHT:.0f} m, GS at the centre; the "
          f"{config.SENS_LOSS:.0%} point stays at "
          f"{channel.SENSITIVITY_DISTANCE_M:.1f} m and b follows from the edge")
    print("  " + "-" * width)
    print(f"  {'edge (m)':>9}{'b (/dB)':>10}{'connected':>12}"
          f"{'M direct to GS':>17}{'neighbours':>13}{'min hops':>11}"
          f"{'loss 200 m':>13}{'loss 300 m':>13}")
    print("  " + "-" * width)
    for row in rows:
        hops = ("    n/a" if row["mean_min_hops"] is None
                else f"{row['mean_min_hops']:>7.2f}")
        print(f"  {row['edge_m']:>9.0f}{row['b_per_db']:>10.4f}"
              f"{row['connected_pct']:>11.1f}%{row['direct_to_gs_pct']:>16.1f}%"
              f"{row['mean_neighbours']:>13.2f}{hops:>11}"
              f"{row['p_loss_200m']*100:>12.2f}%"
              f"{row['p_loss_300m']*100:>12.2f}%")
    print("  " + "-" * width)


def main() -> None:
    """Measure every candidate edge and write the CSV."""
    parser = argparse.ArgumentParser(
        description="Measure what each candidate link edge does to the "
                    "geometry. Reports only; changes nothing."
    )
    parser.add_argument("--placements", type=int, default=PLACEMENTS,
                        help=f"How many random placements (default "
                             f"{PLACEMENTS}).")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the CSV goes.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers for drawing placements.")
    args = parser.parse_args()

    print(f"Drawing {args.placements} placements with the connected-layout "
          f"filter OFF ...")
    positions = draw_all(args.placements, args.workers)
    print(f"  drew {len(positions)}")

    rows = [score(positions, edge) for edge in EDGES_M]
    print_table(rows)

    path = os.path.join(args.out_dir, "edge_scan.csv")
    with open(path, "w", newline="") as handle:
        handle.write(f"# {settings_stamp(placements=len(positions), extra='edge scan, no connected-layout filter')}\n")
        writer = csv.DictWriter(handle, fieldnames=[
            "edge_m", "b_per_db", "connected_pct", "direct_to_gs_pct",
            "mean_neighbours", "mean_min_hops", "p_loss_200m", "p_loss_300m",
            "placements",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
