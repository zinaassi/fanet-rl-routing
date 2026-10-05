"""
range_check.py — Fleet connectivity under the loss curve (Checkpoint 7, 5a).

Pure geometry, and a REPORT ONLY: this script changes no config. Distance plus
the loss curve decide everything, exactly as the simulator does it.

Three link cutoffs are compared: a link exists while its loss stays under
90%, 95% or 99%. The 99% row is the simulator's own rule
(``config.LINK_MAX_LOSS``).

Reported per cutoff, over 100 random placements:
    * % of M-drones with a path to the GS
    * % of placements with at least one isolated M-drone
    * mean neighbours per drone
    * mean drones linked directly to the GS

And, independent of the cutoff, per layout:
    * "good" neighbours per drone — links with under 10% loss
    * drones with a good (under 10% loss) link straight to the GS
    * hops on the MOST RELIABLE path from each M-drone to the GS, i.e. the
      path maximising the end-to-end delivery probability
      prod(1 - p_loss(edge)). Found by shortest path on -log(1 - p_loss),
      which turns that product into a sum.

Layouts are drawn the way the simulator draws them, but WITHOUT its
connected-layout filter: how often a layout strands an M-drone is exactly what
this script is here to measure, so filtering would answer its own question.

Usage:
    python scripts/range_check.py
    python scripts/range_check.py --placements 20      # quicker
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel

#: Link cutoffs to compare: a link exists while loss is under this.
CUTOFFS = (0.90, 0.95, 0.99)

#: A link counts as "good" while its loss is under this.
GOOD_LINK_MAX_LOSS = 0.10

DEFAULT_PLACEMENTS = 100
GS_NODE = "GS"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Report fleet connectivity under the loss curve."
    )
    parser.add_argument(
        "--placements",
        type=int,
        default=DEFAULT_PLACEMENTS,
        help=f"How many random placements (default {DEFAULT_PLACEMENTS}).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("out", "range_check.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def place_fleet(placement_seed: int) -> Dict[str, Any]:
    """Place one fleet exactly as the simulator draws it, unfiltered.

    Mirrors ``FANETEnv._draw_drones``: M-drones first, then C-drones, each
    drawing a uniform point and then a speed. The speed draws are kept so the
    position sequence matches the simulator's.

    Args:
        placement_seed: Seed for the placement stream.

    Returns:
        A dict with ``positions`` (N x 2 array) and ``types`` (list of "M"/"C").
    """
    rng = np.random.default_rng(placement_seed)
    positions: List[np.ndarray] = []
    types: List[str] = []

    def draw_point() -> np.ndarray:
        """Draw one uniform point inside the arena."""
        return rng.uniform([0.0, 0.0], [config.WIDTH, config.HEIGHT])

    def draw_speed() -> float:
        """Consume one speed draw, to stay in step with the simulator."""
        return float(rng.uniform(config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX))

    for _ in range(config.NUM_M_DRONES):
        start = draw_point()
        if not config.STATIC_MODE:
            draw_point()
        draw_speed()
        positions.append(start)
        types.append("M")

    for _ in range(config.NUM_C_DRONES):
        positions.append(draw_point())
        draw_speed()
        types.append("C")

    return {"positions": np.array(positions), "types": types}


def _distance_matrix(positions: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Return pairwise distances among drones and the GS.

    Args:
        positions: N x 2 array of drone positions.
        gs:        Ground-station position.

    Returns:
        An (N+1) x (N+1) distance matrix; index N is the GS.
    """
    points = np.vstack([positions, gs[None, :]])
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def build_graph(dists: np.ndarray, max_loss: float) -> nx.Graph:
    """Build the link graph at one cutoff, weighted for reliability.

    Each edge carries ``loss`` and ``neglog`` = -ln(1 - loss). A shortest path
    on ``neglog`` is the path of maximum delivery probability, because summing
    -ln(1 - loss) minimises the product of (1 - loss).

    Args:
        dists:    Distance matrix from :func:`_distance_matrix`; last index is
                  the GS.
        max_loss: A link exists while its loss is under this.

    Returns:
        A graph over drone indices plus the ``"GS"`` node.
    """
    size = dists.shape[0]
    gs_index = size - 1

    graph = nx.Graph()
    graph.add_nodes_from(range(gs_index))
    graph.add_node(GS_NODE)

    for i in range(size):
        node_i = GS_NODE if i == gs_index else i
        for j in range(i + 1, size):
            loss = channel.p_loss(float(dists[i, j]))
            if loss >= max_loss:
                continue
            node_j = GS_NODE if j == gs_index else j
            graph.add_edge(
                node_i, node_j,
                loss=loss,
                neglog=-math.log(max(1.0 - loss, 1e-300)),
            )
    return graph


def evaluate(max_loss: float, placements: int) -> Dict[str, Any]:
    """Measure connectivity at one cutoff across many placements.

    Args:
        max_loss:   A link exists while its loss is under this.
        placements: How many random layouts to average over.

    Returns:
        One summary record.
    """
    gs = np.array(config.GS_POSITION, dtype=np.float64)

    m_connected: List[float] = []
    layouts_with_isolated = 0
    neighbours: List[float] = []
    gs_linked: List[float] = []
    good_neighbours: List[float] = []
    good_gs_linked: List[float] = []
    reliable_hops: List[float] = []

    for placement_seed in range(1, placements + 1):
        fleet = place_fleet(placement_seed)
        positions = fleet["positions"]
        types = fleet["types"]
        n_drones = len(positions)
        dists = _distance_matrix(positions, gs)
        graph = build_graph(dists, max_loss)

        m_indices = [i for i, t in enumerate(types) if t == "M"]

        # Hops on the most reliable path, and reachability.
        hop_lengths = nx.single_source_shortest_path_length(graph, GS_NODE)
        best = nx.single_source_dijkstra_path(graph, GS_NODE, weight="neglog")

        connected = [i for i in m_indices if i in hop_lengths]
        m_connected.append(len(connected) / len(m_indices))
        if len(connected) < len(m_indices):
            layouts_with_isolated += 1
        reliable_hops.extend(len(best[i]) - 1 for i in connected)

        neighbours.append(
            statistics.fmean(graph.degree(i) for i in range(n_drones))
        )
        gs_linked.append(graph.degree(GS_NODE))

        # Good links are a property of distance alone, not of the cutoff.
        good = [
            [
                j for j in range(n_drones)
                if j != i and channel.p_loss(float(dists[i, j])) < GOOD_LINK_MAX_LOSS
            ]
            for i in range(n_drones)
        ]
        good_neighbours.append(statistics.fmean(len(g) for g in good))
        good_gs_linked.append(sum(
            1 for i in range(n_drones)
            if channel.p_loss(float(dists[i, -1])) < GOOD_LINK_MAX_LOSS
        ))

    return {
        "max_loss": max_loss,
        "link_reach_m": channel.max_link_distance(max_loss),
        "placements": placements,
        "pct_M_with_path_to_GS": 100.0 * statistics.fmean(m_connected),
        "pct_placements_with_isolated_M": 100.0 * layouts_with_isolated / placements,
        "mean_neighbors_per_drone": statistics.fmean(neighbours),
        "mean_drones_linked_to_GS": statistics.fmean(gs_linked),
        "mean_good_neighbors_per_drone": statistics.fmean(good_neighbours),
        "mean_drones_with_good_GS_link": statistics.fmean(good_gs_linked),
        "mean_hops_most_reliable_path": (
            statistics.fmean(reliable_hops) if reliable_hops else float("nan")
        ),
    }


def print_tables(records: List[Dict[str, Any]]) -> None:
    """Print the range-check results.

    Args:
        records: One record per cutoff.
    """
    first = records[0]
    print()
    print(f"  Range check — {first['placements']} random placements, "
          f"{config.NUM_M_DRONES} M + {config.NUM_C_DRONES} C drones in "
          f"{config.WIDTH:.0f}x{config.HEIGHT:.0f} m, GS at the centre")
    print(f"  Loss curve: 1/(1+exp(-{config.LOSS_SLOPE_PER_M}*(d-"
          f"{config.LOSS_50_DISTANCE_M:.0f})))  (Rosati et al., arXiv:1406.4399)")
    print("  Geometry only. Layouts are UNFILTERED. No config changed.")

    print()
    print("  By link cutoff — a link exists while its loss is under the cutoff")
    print("  " + "-" * 94)
    print(f"  {'cutoff':>8}{'reach':>10}{'% M with':>14}{'% layouts':>16}"
          f"{'mean nbrs':>12}{'mean drones':>14}")
    print(f"  {'':>8}{'(m)':>10}{'path to GS':>14}{'w/ isolated M':>16}"
          f"{'per drone':>12}{'linked to GS':>14}")
    print("  " + "-" * 94)
    for record in records:
        marker = ("  <- simulator's rule"
                  if abs(record["max_loss"] - config.LINK_MAX_LOSS) < 1e-9 else "")
        print(
            f"  {record['max_loss'] * 100:>7.0f}%{record['link_reach_m']:>10.1f}"
            f"{record['pct_M_with_path_to_GS']:>13.1f}%"
            f"{record['pct_placements_with_isolated_M']:>15.1f}%"
            f"{record['mean_neighbors_per_drone']:>12.2f}"
            f"{record['mean_drones_linked_to_GS']:>14.2f}{marker}"
        )

    print()
    print("  Link quality and path length")
    print(f"  A \"good\" link loses under {GOOD_LINK_MAX_LOSS * 100:.0f}%, "
          f"i.e. reaches {channel.distance_for_loss(GOOD_LINK_MAX_LOSS):.0f} m")
    print("  " + "-" * 94)
    print(f"  {'good neighbours per drone':<42}"
          f"{records[-1]['mean_good_neighbors_per_drone']:>8.2f}")
    print(f"  {'drones with a good link straight to the GS':<42}"
          f"{records[-1]['mean_drones_with_good_GS_link']:>8.2f}"
          f"   of {config.NUM_M_DRONES + config.NUM_C_DRONES}")
    for record in records:
        print(f"  {'hops on the most reliable path to the GS':<42}"
              f"{record['mean_hops_most_reliable_path']:>8.2f}"
              f"   at the {record['max_loss'] * 100:.0f}% cutoff")
    print("  " + "-" * 94)


def main() -> None:
    """Run the range check and write the CSV."""
    args = parse_args()
    records = [evaluate(cutoff, args.placements) for cutoff in CUTOFFS]

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print_tables(records)
    print()
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
