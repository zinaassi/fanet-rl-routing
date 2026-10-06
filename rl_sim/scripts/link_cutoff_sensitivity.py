"""
link_cutoff_sensitivity.py — How much the link cutoff decides the results.

``config.LINK_MAX_LOSS`` says how bad a link may be and still count as a link.
It is PROVISIONAL, and it sets how far a link reaches, how many neighbours a
drone has, and — for the RANDOM rule, which picks uniformly — how bad the
average chosen link is. This script measures that dependence directly.

Everything else is held fixed: the working load (200 ms), the same 20
placement seeds, the same loss curve, the same routing rules. Only the cutoff
moves.

Reported per cutoff and rule:
    total loss and its five causes, mean hops of delivered packets, mean
    neighbours per drone, drones linked directly to the GS, and how many
    layouts had to be drawn before one connected every M-drone.

Output:
    out/link_cutoff_sensitivity.csv, plus the same table printed.

Usage:
    python scripts/link_cutoff_sensitivity.py
    python scripts/link_cutoff_sensitivity.py --placements 3    # quicker
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs import channel
from scripts.experiment_1a import LOADS_MS, ROUTING_RULES, run_one
from scripts.stamp import csv_comment, settings_stamp

#: Cutoffs to compare. 0.5 is the configured value.
CUTOFFS = (0.99, 0.5, 0.2)

#: The working load, in milliseconds.
LOAD_MS = 200

DEFAULT_PLACEMENTS = 20

CSV_COLUMNS = [
    "link_max_loss", "link_reach_m", "routing", "load_ms", "placements",
    "total_loss", "loss_channel", "loss_queue_full", "loss_no_route",
    "loss_ttl", "loss_hop_limit", "mean_hops",
    "mean_neighbors_per_drone", "drones_linked_to_GS", "placement_draws",
    "views_agree",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Measure how the link cutoff moves the Phase-1a results."
    )
    parser.add_argument(
        "--placements", type=int, default=DEFAULT_PLACEMENTS,
        help=f"How many placement seeds, 1..N (default {DEFAULT_PLACEMENTS}).",
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join("out", "link_cutoff_sensitivity.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def measure(cutoff: float, routing: str, placements: int) -> Dict[str, Any]:
    """Average one (cutoff, rule) cell over the placement seeds.

    Args:
        cutoff:     Value to give ``config.LINK_MAX_LOSS`` for these runs.
        routing:    "greedy" or "random".
        placements: How many placement seeds to average over.

    Returns:
        One record for the CSV.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    previous = config.LINK_MAX_LOSS
    try:
        config.LINK_MAX_LOSS = cutoff
        rows = []
        for placement_seed in range(1, placements + 1):
            row, _ = run_one(
                routing=routing, load_ms=LOAD_MS,
                placement_seed=placement_seed, run_seed=placement_seed,
            )
            if not row["views_agree"]:
                print("\nSTOP: ACK view disagrees with ground truth.")
                print(f"  cutoff={cutoff} routing={routing} "
                      f"placement_seed={placement_seed}")
                for problem in row["_discrepancies"]:
                    print(f"    ! {problem}")
                raise SystemExit(1)
            rows.append(row)

        def mean(column: str) -> float:
            """Mean of *column* over this cell's runs."""
            return statistics.fmean(float(r[column]) for r in rows)

        return {
            "link_max_loss": cutoff,
            "link_reach_m": round(channel.max_link_distance(cutoff), 1),
            "routing": routing,
            "load_ms": LOAD_MS,
            "placements": len(rows),
            "total_loss": mean("total_loss"),
            "loss_channel": mean("loss_channel"),
            "loss_queue_full": mean("loss_queue_full"),
            "loss_no_route": mean("loss_no_route"),
            "loss_ttl": mean("loss_ttl"),
            "loss_hop_limit": mean("loss_hop_limit"),
            "mean_hops": mean("mean_hops"),
            "mean_neighbors_per_drone": mean("mean_neighbors_per_drone"),
            "drones_linked_to_GS": mean("drones_in_GS_range"),
            "placement_draws": mean("placement_draws"),
            "views_agree": True,
        }
    finally:
        config.LINK_MAX_LOSS = previous


def print_table(records: List[Dict[str, Any]]) -> None:
    """Print the sensitivity table.

    Args:
        records: One record per (cutoff, rule).
    """
    total = config.NUM_M_DRONES + config.NUM_C_DRONES
    print()
    print(f"  Link-cutoff sensitivity — {records[0]['placements']} placements, "
          f"{LOAD_MS} ms load, everything else fixed")
    print("  A link exists while its loss is under the cutoff. "
          "All figures are % of packets created,")
    print("  except hops, neighbours, GS links and draws.")
    print("  " + "-" * 118)
    print(f"  {'cutoff':>7}{'reach':>8}{'rule':>9}{'total':>9}{'channel':>9}"
          f"{'queue':>8}{'no':>7}{'ttl':>7}{'hop':>7}{'hops':>8}"
          f"{'nbrs':>8}{'GS':>8}{'draws':>8}")
    print(f"  {'':>7}{'(m)':>8}{'':>9}{'loss':>9}{'':>9}"
          f"{'full':>8}{'route':>7}{'':>7}{'limit':>7}{'(dlvd)':>8}"
          f"{'/drone':>8}{f'/{total}':>8}{'':>8}")
    print("  " + "-" * 118)
    for record in records:
        marker = ("  <- configured"
                  if abs(record["link_max_loss"] - config.LINK_MAX_LOSS) < 1e-9
                  else "")
        print(
            f"  {record['link_max_loss'] * 100:>6.0f}%{record['link_reach_m']:>8.1f}"
            f"{record['routing']:>9}"
            f"{record['total_loss'] * 100:>9.1f}"
            f"{record['loss_channel'] * 100:>9.1f}"
            f"{record['loss_queue_full'] * 100:>8.1f}"
            f"{record['loss_no_route'] * 100:>7.1f}"
            f"{record['loss_ttl'] * 100:>7.1f}"
            f"{record['loss_hop_limit'] * 100:>7.1f}"
            f"{record['mean_hops']:>8.2f}"
            f"{record['mean_neighbors_per_drone']:>8.2f}"
            f"{record['drones_linked_to_GS']:>8.2f}"
            f"{record['placement_draws']:>8.2f}{marker}"
        )
    print("  " + "-" * 118)


def main() -> None:
    """Run every (cutoff, rule) cell, write the CSV and print the table."""
    args = parse_args()

    records: List[Dict[str, Any]] = []
    total_cells = len(CUTOFFS) * len(ROUTING_RULES)
    started = time.perf_counter()
    print(f"Running {total_cells * args.placements} runs "
          f"({len(CUTOFFS)} cutoffs x {len(ROUTING_RULES)} rules x "
          f"{args.placements} placements) at {LOAD_MS} ms ...")

    for cutoff in CUTOFFS:
        for routing in ROUTING_RULES:
            records.append(measure(cutoff, routing, args.placements))
            print(f"  cutoff {cutoff:>4}  {routing:<7}  done   "
                  f"{time.perf_counter() - started:6.1f}s")

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=[LOAD_MS], placements=args.placements,
                                 extra="cutoff varies by row"))
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    print_table(records)
    print()
    print(f"  {settings_stamp(loads_ms=[LOAD_MS], placements=args.placements)}")
    print()
    print(f"  views agreed in all {total_cells * args.placements} runs")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
