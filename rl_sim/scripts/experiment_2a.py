"""
experiment_2a.py — Phase 2a step 1: greedy and random with the M-drones flying.

The same 20 layouts and the same 200 ms load as Phase 1a, run four ways:
the two baseline rules, each with the start delays spread over 50 s and with
every drone setting off at once. C-drones hover throughout.

What is new beside the Phase-1 numbers:

    link_broken          packets whose link went away while they waited and
                         which the re-route could not place anywhere
    reroutes             how often a queued packet was handed back to the
                         routing rule because its link had gone
    pct_time_connected   mean over M-drones of the share of measured steps on
                         which that drone still had SOME path to the GS
    loss_while_connected loss counting only packets created at a moment when
                         their source could still reach the GS

The static reference for the same layouts and rules is out/2a_reference.csv,
written before any of this existed.

Writes:
    out/2a_runs.csv      one row per run
    out/2a_summary.csv   one row per (setting, rule)
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.stamp import csv_comment

PLACEMENTS: Tuple[int, ...] = tuple(range(1, 21))
RULES: Tuple[str, ...] = ("greedy", "random")
LOAD_MS: int = 200

#: The two start-delay settings asked for: spread out, and all at once.
DELAYS_S: Tuple[float, ...] = (50.0, 0.0)

CAUSES: Tuple[str, ...] = (
    "channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit",
    "link_broken",
)

EXTRAS: Tuple[str, ...] = (
    "pct_time_connected", "loss_while_connected", "created_while_connected",
    "reroutes", "link_broken_drops", "mean_hops",
)

RUN_COLUMNS: Tuple[str, ...] = (
    "max_start_delay_s", "rule", "load_ms", "placement_seed",
    "total_loss", *CAUSES, *EXTRAS, "arrived_m_drones", "views_agree",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "max_start_delay_s", "rule", "runs", "total_loss_mean", "total_loss_std",
    *(f"{cause}_mean" for cause in CAUSES),
    "pct_time_connected_mean", "loss_while_connected_mean",
    "loss_while_connected_std", "reroutes_mean", "link_broken_drops_mean",
    "mean_hops_mean",
)


def run_once(rule: str, seed: int, max_start_delay_s: float) -> Dict[str, Any]:
    """Run one layout under one rule at one start-delay setting.

    Args:
        rule:              "greedy" or "random".
        seed:              Placement seed, used as the run seed too.
        max_start_delay_s: Upper end of the per-drone start delay.

    Returns:
        The row for ``out/2a_runs.csv``.
    """
    previous = (config.PACKET_INTERVAL_STEPS, config.STATIC_MODE,
                config.MAX_START_DELAY_S)
    config.PACKET_INTERVAL_STEPS = int(
        round(LOAD_MS / 1000.0 / config.TIMESTEP)
    )
    config.STATIC_MODE = False
    config.MAX_START_DELAY_S = max_start_delay_s
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
        arrived = sum(1 for d in env.drones
                      if d.drone_type == "M" and d.arrived)
        return {
            "max_start_delay_s": max_start_delay_s,
            "rule": rule,
            "load_ms": LOAD_MS,
            "placement_seed": seed,
            "total_loss": truth["total_loss"] or 0.0,
            **{cause: by_reason[cause] for cause in CAUSES},
            **{extra: truth[extra] for extra in EXTRAS},
            "arrived_m_drones": arrived,
            "views_agree": not metrics["discrepancies"],
            "discrepancies": metrics["discrepancies"],
        }
    finally:
        (config.PACKET_INTERVAL_STEPS, config.STATIC_MODE,
         config.MAX_START_DELAY_S) = previous


def _job(args: Tuple[str, int, float]) -> Dict[str, Any]:
    """Worker entry point for one run."""
    return run_once(*args)


def run_grid(workers: int) -> List[Dict[str, Any]]:
    """Run every (delay, rule, layout) combination.

    Args:
        workers: Parallel workers; 1 runs sequentially.

    Returns:
        One row per run.

    Raises:
        SystemExit: If any run's two views of the traffic disagree.
    """
    jobs = [(rule, seed, delay)
            for delay in DELAYS_S
            for rule in RULES
            for seed in PLACEMENTS]
    print(f"Running {len(jobs)} runs ({len(DELAYS_S)} delay settings x "
          f"{len(RULES)} rules x {len(PLACEMENTS)} layouts at {LOAD_MS} ms) ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [row for row in rows if not row["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — delay {row['max_start_delay_s']} "
                  f"{row['rule']} seed {row['placement_seed']}: "
                  f"{row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def _mean(values: Sequence[Optional[float]]) -> float:
    """Mean over the values that are not None, or 0.0 if there are none."""
    kept = [v for v in values if v is not None]
    return statistics.fmean(kept) if kept else 0.0


def _std(values: Sequence[Optional[float]]) -> float:
    """Sample standard deviation, or 0.0 below two values."""
    kept = [v for v in values if v is not None]
    return statistics.stdev(kept) if len(kept) > 1 else 0.0


def summarise(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one summary row per (delay setting, rule)."""
    out: List[Dict[str, Any]] = []
    for delay in DELAYS_S:
        for rule in RULES:
            chosen = [r for r in rows
                      if r["max_start_delay_s"] == delay and r["rule"] == rule]
            totals = [r["total_loss"] for r in chosen]
            out.append({
                "max_start_delay_s": delay,
                "rule": rule,
                "runs": len(chosen),
                "total_loss_mean": _mean(totals),
                "total_loss_std": _std(totals),
                **{f"{cause}_mean": _mean([r[cause] for r in chosen])
                   for cause in CAUSES},
                "pct_time_connected_mean": _mean(
                    [r["pct_time_connected"] for r in chosen]
                ),
                "loss_while_connected_mean": _mean(
                    [r["loss_while_connected"] for r in chosen]
                ),
                "loss_while_connected_std": _std(
                    [r["loss_while_connected"] for r in chosen]
                ),
                "reroutes_mean": _mean([r["reroutes"] for r in chosen]),
                "link_broken_drops_mean": _mean(
                    [r["link_broken_drops"] for r in chosen]
                ),
                "mean_hops_mean": _mean([r["mean_hops"] for r in chosen]),
            })
    return out


def print_summary(summary: Sequence[Dict[str, Any]]) -> None:
    """Print the loss and its causes, per setting and rule."""
    width = 118
    print(f"\n  {len(PLACEMENTS)} layouts at {LOAD_MS} ms — mean % of packets "
          f"created (C-drones hover)")
    print("  " + "-" * width)
    print(f"  {'delay':>7}{'rule':>8}{'total':>9}{'std':>7}"
          + "".join(f"{c[:9]:>11}" for c in CAUSES))
    print("  " + "-" * width)
    for row in summary:
        print(f"  {row['max_start_delay_s']:>6.0f}s{row['rule']:>8}"
              f"{row['total_loss_mean'] * 100:8.2f}%"
              f"{row['total_loss_std'] * 100:7.2f}"
              + "".join(f"{row[f'{c}_mean'] * 100:10.2f}%" for c in CAUSES))
    print("  " + "-" * width)

    print(f"\n  {'delay':>7}{'rule':>8}{'connected':>12}"
          f"{'loss|conn':>12}{'std':>8}{'reroutes':>11}{'broken':>9}{'hops':>7}")
    print("  " + "-" * 76)
    for row in summary:
        print(f"  {row['max_start_delay_s']:>6.0f}s{row['rule']:>8}"
              f"{row['pct_time_connected_mean'] * 100:11.2f}%"
              f"{row['loss_while_connected_mean'] * 100:11.2f}%"
              f"{row['loss_while_connected_std'] * 100:8.2f}"
              f"{row['reroutes_mean']:11.1f}{row['link_broken_drops_mean']:9.1f}"
              f"{row['mean_hops_mean']:7.2f}")
    print("  " + "-" * 76)


def print_against_static(summary: Sequence[Dict[str, Any]],
                         out_dir: str) -> None:
    """Print the moving numbers beside the frozen static ones.

    Args:
        summary: The summary rows.
        out_dir: Where out/2a_reference.csv lives.
    """
    from scripts.stamp import read_stamped_csv

    path = os.path.join(out_dir, "2a_reference.csv")
    if not os.path.exists(path):
        print("\n  (no out/2a_reference.csv — skipping the static comparison)")
        return
    reference = read_stamped_csv(path)

    print("\n  Against the static world (same layouts, same load)")
    print("  " + "-" * 62)
    print(f"  {'delay':>7}{'rule':>8}{'static':>10}{'moving':>10}{'change':>10}")
    print("  " + "-" * 62)
    for row in summary:
        still = statistics.fmean(
            float(r["total_loss"]) for r in reference if r["rule"] == row["rule"]
        )
        moving = row["total_loss_mean"]
        print(f"  {row['max_start_delay_s']:>6.0f}s{row['rule']:>8}"
              f"{still * 100:9.2f}%{moving * 100:9.2f}%"
              f"{(moving - still) * 100:+9.2f}")
    print("  " + "-" * 62)


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str], note: str) -> None:
    """Write *rows* to *path* under a settings stamp."""
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
    """Run the Phase-2a grid and print every table."""
    parser = argparse.ArgumentParser(
        description="Phase 2a step 1: baselines with the M-drones flying."
    )
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the CSVs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    args = parser.parse_args()

    rows = run_grid(args.workers)
    summary = summarise(rows)
    print_summary(summary)
    print_against_static(summary, args.out_dir)

    note = (f"phase 2a step 1, M-drones flying, C-drones hover, "
            f"start delay {'/'.join(f'{d:.0f}' for d in DELAYS_S)} s")
    write_csv(os.path.join(args.out_dir, "2a_runs.csv"), rows,
              RUN_COLUMNS, note)
    write_csv(os.path.join(args.out_dir, "2a_summary.csv"), summary,
              SUMMARY_COLUMNS, note)


if __name__ == "__main__":
    main()
