"""
reference_2a.py — Freeze today's static-world results, before Phase 2a moves.

Phase 2a makes the M-drones fly. Everything it adds — a mobility stream,
broken-link handling, new metrics — must leave the STATIC world exactly as it
is, or the comparison against Phase 1 means nothing. This script writes the
reference that proves it, and tests/test_2a_reference.py checks the simulator
still reproduces it.

Two things are recorded, both at the 200 ms working load on the 20 Phase-1a
layouts:

    out/2a_reference.csv   per layout and rule: total loss and the loss split
                           by cause, for greedy and random.
    out/2a_layout_hash.csv per layout: a SHA-256 over EVERY drone's start
                           position, M-drones and C-drones alike, in drone-id
                           order.

The layout hash is the sharper of the two. Positions come from the placement
stream, so if Phase 2a ever draws its movement randomness from that stream
instead of its own, the hash moves and the test fails — which is the whole
point of giving mobility a separate stream.

Nothing here is Phase-2 code: it runs the simulator as it stands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.stamp import csv_comment

#: The Phase-1a layouts, at the Phase-1a working load.
PLACEMENTS: Tuple[int, ...] = tuple(range(1, 21))
RULES: Tuple[str, ...] = ("greedy", "random")
LOAD_MS: int = 200

CAUSES: Tuple[str, ...] = (
    "channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit",
)

RUN_COLUMNS: Tuple[str, ...] = (
    "rule", "load_ms", "placement_seed", "total_loss", *CAUSES,
    "mean_hops", "views_agree",
)
HASH_COLUMNS: Tuple[str, ...] = (
    "placement_seed", "positions_sha256", "num_drones", "placement_draws",
)


def layout_hash(seed: int) -> Tuple[str, int, int]:
    """Return a hash of every drone's start position for one layout.

    The positions are taken in drone-id order and hashed as raw float64
    bytes, so the digest changes if any coordinate moves by one bit.

    Args:
        seed: The placement seed.

    Returns:
        ``(hex digest, number of drones, placement draws)``.
    """
    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    drones = sorted(env.drones, key=lambda d: d.drone_id)
    positions = np.array([d.position for d in drones], dtype=np.float64)
    env.close_logger()
    return (hashlib.sha256(positions.tobytes()).hexdigest(),
            len(drones), env.placement_draws)


def run_once(rule: str, seed: int) -> Dict[str, Any]:
    """Run one layout under one rule and return its row.

    Args:
        rule: "greedy" or "random".
        seed: Placement seed, used as the run seed as well.

    Returns:
        The row for ``out/2a_reference.csv``.
    """
    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(
        round(LOAD_MS / 1000.0 / config.TIMESTEP)
    )
    try:
        env = FANETEnv(routing=rule, log_path=os.devnull,
                       placement_seed=seed, run_seed=seed)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        by_reason = truth["loss_rate_by_reason"]
        return {
            "rule": rule,
            "load_ms": LOAD_MS,
            "placement_seed": seed,
            "total_loss": truth["total_loss"] or 0.0,
            **{cause: by_reason[cause] for cause in CAUSES},
            "mean_hops": truth["mean_hops"],
            "views_agree": not metrics["discrepancies"],
            "discrepancies": metrics["discrepancies"],
        }
    finally:
        config.PACKET_INTERVAL_STEPS = previous


def _job(args: Tuple[str, int]) -> Dict[str, Any]:
    """Worker entry point for one (rule, seed) pair."""
    return run_once(*args)


def run_grid(workers: int) -> List[Dict[str, Any]]:
    """Run every rule on every layout.

    Args:
        workers: Parallel workers; 1 runs sequentially.

    Returns:
        One row per run.

    Raises:
        SystemExit: If any run's two views of the traffic disagree.
    """
    jobs = [(rule, seed) for rule in RULES for seed in PLACEMENTS]
    print(f"Running {len(jobs)} runs ({len(RULES)} rules x "
          f"{len(PLACEMENTS)} layouts at {LOAD_MS} ms) ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [row for row in rows if not row["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — {row['rule']} "
                  f"seed {row['placement_seed']}: {row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    """Print the reference, one line per run."""
    width = 104
    print()
    print("  Static-world reference — 20 layouts at 200 ms, % of packets "
          "created")
    print("  " + "-" * width)
    print(f"  {'rule':>8}{'seed':>6}{'total':>9}"
          + "".join(f"{c[:9]:>11}" for c in CAUSES) + f"{'hops':>7}")
    print("  " + "-" * width)
    for rule in RULES:
        for row in (r for r in rows if r["rule"] == rule):
            print(f"  {row['rule']:>8}{row['placement_seed']:>6}"
                  f"{row['total_loss'] * 100:8.3f}%"
                  + "".join(f"{row[c] * 100:10.3f}%" for c in CAUSES)
                  + f"{row['mean_hops'] or 0.0:7.2f}")
        print("  " + "-" * width)


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str], note: str) -> None:
    """Write *rows* to *path* under a settings stamp.

    Args:
        path:    Destination file.
        rows:    Rows to write.
        columns: Column order.
        note:    Appended to the stamp, saying what the file is.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=[LOAD_MS],
                                 placements=len(PLACEMENTS), extra=note))
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    print(f"  wrote {path}")


def main() -> None:
    """Write the static-world reference."""
    parser = argparse.ArgumentParser(
        description="Freeze the static-world results before Phase 2a moves."
    )
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the two CSVs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    args = parser.parse_args()

    if not config.STATIC_MODE:
        print("STOP: config.STATIC_MODE is False. The reference is the "
              "STATIC world; record it with movement off.")
        raise SystemExit(1)

    rows = run_grid(args.workers)
    print_table(rows)

    hashes = [
        dict(zip(("positions_sha256", "num_drones", "placement_draws"),
                 layout_hash(seed)),
             placement_seed=seed)
        for seed in PLACEMENTS
    ]
    print("\n  Layout hashes (every drone's start position, drone-id order)")
    print("  " + "-" * 78)
    for row in hashes:
        print(f"  seed {row['placement_seed']:>3}  "
              f"{row['positions_sha256'][:32]}...  "
              f"{row['num_drones']} drones, {row['placement_draws']} draw(s)")
    print("  " + "-" * 78)

    note = "phase-2a static reference"
    write_csv(os.path.join(args.out_dir, "2a_reference.csv"), rows,
              RUN_COLUMNS, note)
    write_csv(os.path.join(args.out_dir, "2a_layout_hash.csv"), hashes,
              HASH_COLUMNS, note)


if __name__ == "__main__":
    main()
