"""
range_check.py — Connectivity of the fleet at candidate radio ranges (item L).

Pure geometry, and a REPORT ONLY: this script changes no config. Two drones
count as linked when their distance is <= the range under test, and a drone is
in direct GS range when it is <= the range from the ground station. The FSPL
channel, the transmit power and the receiver sensitivity are not touched.

Drones are placed uniformly at random in the configured arena from the same
placement seed stream the simulator uses.

This script deliberately takes the FIRST layout each seed produces, WITHOUT the
simulator's ``REQUIRE_CONNECTED_M`` filter. The filter exists precisely because
some layouts strand an M-drone; measuring how often that happens is this
script's job, so filtering here would answer its own question.

For each candidate range it reports, over 100 placements:
    * % of M-drones with a path to the GS
    * % of placements with at least one isolated M-drone
    * mean hop count to the GS, over connected M-drones only
    * mean neighbours per drone
    * mean drones in direct GS range
    * the transmit power that would produce this range under the current FSPL
      settings, Pt = 30 + 20*log10(range / 249.69) dBm

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

CANDIDATE_RANGES_M = (200.0, 230.0, 250.0, 280.0, 320.0)
DEFAULT_PLACEMENTS = 100
GS_NODE = "GS"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Report fleet connectivity at candidate radio ranges."
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
    """Place one fleet exactly as the simulator would.

    Mirrors ``FANETEnv._create_drones``: M-drones first, then C-drones, each
    drawing a uniform point and then a speed from the placement stream. The
    speed draws are kept so the position sequence matches the simulator's.

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
        # STATIC_MODE: the end point is the start point, so no extra draw.
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


def build_graph(
    positions: np.ndarray, range_m: float
) -> nx.Graph:
    """Build the link graph at *range_m*, with the GS as an extra node.

    Two endpoints are linked when their distance is <= *range_m*.

    Args:
        positions: N x 2 array of drone positions.
        range_m:   The range under test, in metres.

    Returns:
        A graph over drone indices plus the ``"GS"`` node.
    """
    gs = np.array(config.GS_POSITION, dtype=np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(positions)))
    graph.add_node(GS_NODE)

    for i, pos_i in enumerate(positions):
        if float(np.linalg.norm(pos_i - gs)) <= range_m:
            graph.add_edge(i, GS_NODE)
        for j in range(i + 1, len(positions)):
            if float(np.linalg.norm(pos_i - positions[j])) <= range_m:
                graph.add_edge(i, j)
    return graph


def required_tx_power_dbm(range_m: float) -> float:
    """Return the transmit power that yields *range_m* under the current FSPL.

    Free-space loss grows as 20*log10(distance), so moving the range from the
    current 249.69 m costs (or saves) 20*log10(ratio) dB of transmit power.

    Args:
        range_m: The range under test, in metres.

    Returns:
        Transmit power in dBm.
    """
    return channel.PT_DBM + 20.0 * math.log10(
        range_m / channel.MAX_LINK_DISTANCE_M
    )


def evaluate_range(range_m: float, placements: int) -> Dict[str, Any]:
    """Measure connectivity at one candidate range across many placements.

    Args:
        range_m:    The range under test, in metres.
        placements: How many random layouts to average over.

    Returns:
        One summary record for this range.
    """
    m_connected_fractions: List[float] = []
    layouts_with_isolated = 0
    hop_counts: List[float] = []
    neighbours_per_drone: List[float] = []
    in_gs_range: List[float] = []

    gs = np.array(config.GS_POSITION, dtype=np.float64)

    for placement_seed in range(1, placements + 1):
        fleet = place_fleet(placement_seed)
        positions = fleet["positions"]
        types = fleet["types"]
        graph = build_graph(positions, range_m)

        m_indices = [i for i, t in enumerate(types) if t == "M"]
        hops = nx.single_source_shortest_path_length(graph, GS_NODE)

        connected = [i for i in m_indices if i in hops]
        m_connected_fractions.append(len(connected) / len(m_indices))
        if len(connected) < len(m_indices):
            layouts_with_isolated += 1
        hop_counts.extend(float(hops[i]) for i in connected)

        degrees = [graph.degree(i) for i in range(len(positions))]
        neighbours_per_drone.append(statistics.fmean(degrees))

        in_gs_range.append(
            sum(
                1 for pos in positions
                if float(np.linalg.norm(pos - gs)) <= range_m
            )
        )

    return {
        "range_m": range_m,
        "placements": placements,
        "pct_M_with_path_to_GS": 100.0 * statistics.fmean(m_connected_fractions),
        "pct_placements_with_isolated_M": 100.0 * layouts_with_isolated / placements,
        "mean_hops_to_GS": statistics.fmean(hop_counts) if hop_counts else float("nan"),
        "mean_neighbors_per_drone": statistics.fmean(neighbours_per_drone),
        "mean_drones_in_GS_range": statistics.fmean(in_gs_range),
        "required_tx_power_dbm": required_tx_power_dbm(range_m),
    }


def print_table(records: List[Dict[str, Any]]) -> None:
    """Print the range-check results as a table.

    Args:
        records: One record per candidate range.
    """
    print()
    print(f"  Range check — {records[0]['placements']} random placements, "
          f"{config.NUM_M_DRONES} M + {config.NUM_C_DRONES} C drones in "
          f"{config.WIDTH:.0f}x{config.HEIGHT:.0f} m")
    print("  Pure geometry: linked when distance <= range. No config changed.")
    print("  " + "-" * 104)
    print(f"  {'range':>7}{'% M with path':>16}{'% layouts with':>17}"
          f"{'mean hops':>12}{'mean nbrs':>12}{'mean drones':>14}{'Pt needed':>12}")
    print(f"  {'(m)':>7}{'to GS':>16}{'an isolated M':>17}"
          f"{'to GS':>12}{'per drone':>12}{'in GS range':>14}{'(dBm)':>12}")
    print("  " + "-" * 104)
    for record in records:
        marker = "  <- current" if abs(
            record["range_m"] - channel.MAX_LINK_DISTANCE_M
        ) < 1.0 else ""
        print(
            f"  {record['range_m']:>7.0f}"
            f"{record['pct_M_with_path_to_GS']:>15.1f}%"
            f"{record['pct_placements_with_isolated_M']:>16.1f}%"
            f"{record['mean_hops_to_GS']:>12.2f}"
            f"{record['mean_neighbors_per_drone']:>12.2f}"
            f"{record['mean_drones_in_GS_range']:>14.2f}"
            f"{record['required_tx_power_dbm']:>12.2f}{marker}"
        )
    print("  " + "-" * 104)
    print(f"  The simulator's current effective range is "
          f"{channel.MAX_LINK_DISTANCE_M:.2f} m at Pt = {channel.PT_DBM:.0f} dBm.")


def main() -> None:
    """Run the range check and write the CSV."""
    args = parse_args()

    records = [
        evaluate_range(range_m, args.placements)
        for range_m in CANDIDATE_RANGES_M
    ]

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print_table(records)
    print()
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
