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

#: The two start-delay settings step 1 asked for: spread out, and all at
#: once. Step 2 fixes the delay at 50 s and sweeps the hello interval
#: instead, so --step 1 and --step 2 are two different grids.
DELAYS_S: Tuple[float, ...] = (50.0, 0.0)

#: Step 2: hello intervals in seconds, plus the ideal-knowledge control.
#: ``None`` means IDEAL_HELLO, which is step 1's perfect knowledge.
HELLO_SETTINGS: Tuple[Optional[float], ...] = (0.2, 0.5, 1.0, 2.0, None)

#: The start delay step 2 runs at.
STEP2_DELAY_S: float = 50.0

CAUSES: Tuple[str, ...] = (
    "channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit",
    "link_broken", "stale_link",
)

EXTRAS: Tuple[str, ...] = (
    "pct_time_connected", "loss_while_connected", "created_while_connected",
    "reroutes", "link_broken_drops", "mean_hops",
    "table_precision", "table_recall", "stale_sends", "stale_link_drops",
    "hellos_sent_per_drone_s", "hellos_received_per_drone_s",
)

RUN_COLUMNS: Tuple[str, ...] = (
    "hello_interval_s", "ideal_hello", "max_start_delay_s", "rule",
    "load_ms", "placement_seed",
    "total_loss", *CAUSES, *EXTRAS, "arrived_m_drones", "views_agree",
)

SUMMARY_COLUMNS: Tuple[str, ...] = (
    "setting", "hello_interval_s", "ideal_hello", "max_start_delay_s",
    "rule", "runs", "total_loss_mean", "total_loss_std",
    *(f"{cause}_mean" for cause in CAUSES),
    *(f"{extra}_mean" for extra in EXTRAS),
    "loss_while_connected_std",
)


def run_once(rule: str, seed: int, max_start_delay_s: float,
             hello_interval_s: Optional[float] = None) -> Dict[str, Any]:
    """Run one layout under one rule, at one delay and one hello setting.

    Args:
        rule:              "greedy" or "random".
        seed:              Placement seed, used as the run seed too.
        max_start_delay_s: Upper end of the per-drone start delay.
        hello_interval_s:  Seconds between hellos, or None for IDEAL_HELLO —
                           the perfect neighbour knowledge of step 1.

    Returns:
        The row for the runs CSV.
    """
    previous = (config.PACKET_INTERVAL_STEPS, config.STATIC_MODE,
                config.MAX_START_DELAY_S, config.IDEAL_HELLO,
                config.HELLO_INTERVAL)
    config.PACKET_INTERVAL_STEPS = int(
        round(LOAD_MS / 1000.0 / config.TIMESTEP)
    )
    config.STATIC_MODE = False
    config.MAX_START_DELAY_S = max_start_delay_s
    config.IDEAL_HELLO = hello_interval_s is None
    if hello_interval_s is not None:
        config.HELLO_INTERVAL = hello_interval_s
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
            "hello_interval_s": (0.0 if hello_interval_s is None
                                 else hello_interval_s),
            "ideal_hello": hello_interval_s is None,
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
         config.MAX_START_DELAY_S, config.IDEAL_HELLO,
         config.HELLO_INTERVAL) = previous


def _job(args: Tuple[str, int, float, Optional[float]]) -> Dict[str, Any]:
    """Worker entry point for one run."""
    return run_once(*args)


def build_jobs(step: int) -> List[Tuple[str, int, float, Optional[float]]]:
    """Return the grid for one step of Phase 2a.

    Step 1 sweeps the start delay with perfect neighbour knowledge; step 2
    fixes the delay and sweeps the hello interval, with IDEAL_HELLO as its
    control.

    Args:
        step: 1 or 2.

    Returns:
        ``(rule, seed, delay, hello interval or None)`` per run.
    """
    if step == 1:
        return [(rule, seed, delay, None)
                for delay in DELAYS_S
                for rule in RULES
                for seed in PLACEMENTS]
    return [(rule, seed, STEP2_DELAY_S, hello)
            for hello in HELLO_SETTINGS
            for rule in RULES
            for seed in PLACEMENTS]


def run_grid(workers: int, step: int) -> List[Dict[str, Any]]:
    """Run every combination for *step*.

    Args:
        workers: Parallel workers; 1 runs sequentially.
        step:    1 or 2.

    Returns:
        One row per run.

    Raises:
        SystemExit: If any run's two views of the traffic disagree.
    """
    jobs = build_jobs(step)
    print(f"Running {len(jobs)} runs (step {step}, {len(RULES)} rules x "
          f"{len(PLACEMENTS)} layouts at {LOAD_MS} ms) ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [row for row in rows if not row["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — hello {row['hello_interval_s']} "
                  f"delay {row['max_start_delay_s']} {row['rule']} "
                  f"seed {row['placement_seed']}: {row['discrepancies']}")
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


def _setting_label(row: Dict[str, Any]) -> str:
    """Return a short name for the row's setting, for the printed tables."""
    if row.get("ideal_hello"):
        return "ideal"
    return f"{row['hello_interval_s']:.1f}s"


def summarise(rows: Sequence[Dict[str, Any]], step: int) -> List[Dict[str, Any]]:
    """Return one summary row per setting and rule.

    Args:
        rows: Every run.
        step: 1 groups by start delay, 2 by hello setting.

    Returns:
        The summary rows, in the order the grid was run.
    """
    if step == 1:
        groups = [({"max_start_delay_s": d}, f"{d:.0f}s") for d in DELAYS_S]
    else:
        groups = [
            ({"ideal_hello": h is None,
              **({} if h is None else {"hello_interval_s": h})},
             "ideal" if h is None else f"{h:.1f}s")
            for h in HELLO_SETTINGS
        ]

    out: List[Dict[str, Any]] = []
    for where, label in groups:
        for rule in RULES:
            chosen = [
                r for r in rows
                if r["rule"] == rule
                and all(r[k] == v for k, v in where.items())
            ]
            if not chosen:
                continue
            totals = [r["total_loss"] for r in chosen]
            out.append({
                "setting": label,
                "hello_interval_s": chosen[0]["hello_interval_s"],
                "ideal_hello": chosen[0]["ideal_hello"],
                "max_start_delay_s": chosen[0]["max_start_delay_s"],
                "rule": rule,
                "runs": len(chosen),
                "total_loss_mean": _mean(totals),
                "total_loss_std": _std(totals),
                **{f"{cause}_mean": _mean([r[cause] for r in chosen])
                   for cause in CAUSES},
                **{f"{extra}_mean": _mean([r[extra] for r in chosen])
                   for extra in EXTRAS},
                "loss_while_connected_std": _std(
                    [r["loss_while_connected"] for r in chosen]
                ),
            })
    return out


def print_summary(summary: Sequence[Dict[str, Any]]) -> None:
    """Print the loss and its causes, per setting and rule."""
    width = 126
    print(f"\n  {len(PLACEMENTS)} layouts at {LOAD_MS} ms — mean % of packets "
          f"created (C-drones hover)")
    print("  " + "-" * width)
    print(f"  {'setting':>8}{'rule':>8}{'total':>9}{'std':>7}"
          + "".join(f"{c[:9]:>11}" for c in CAUSES))
    print("  " + "-" * width)
    for row in summary:
        print(f"  {row['setting']:>8}{row['rule']:>8}"
              f"{row['total_loss_mean'] * 100:8.2f}%"
              f"{row['total_loss_std'] * 100:7.2f}"
              + "".join(f"{row[f'{c}_mean'] * 100:10.2f}%" for c in CAUSES))
    print("  " + "-" * width)

    print(f"\n  {'setting':>8}{'rule':>8}{'conn':>8}{'loss|conn':>11}"
          f"{'std':>7}{'reroute':>9}{'broken':>8}"
          f"{'believed ok':>13}{'found':>8}{'stale snd':>11}{'hello/s':>9}")
    print("  " + "-" * 102)
    for row in summary:
        print(f"  {row['setting']:>8}{row['rule']:>8}"
              f"{row['pct_time_connected_mean'] * 100:7.1f}%"
              f"{row['loss_while_connected_mean'] * 100:10.2f}%"
              f"{row['loss_while_connected_std'] * 100:7.2f}"
              f"{row['reroutes_mean']:9.1f}"
              f"{row['link_broken_drops_mean']:8.1f}"
              f"{row['table_precision_mean'] * 100:12.1f}%"
              f"{row['table_recall_mean'] * 100:7.1f}%"
              f"{row['stale_sends_mean']:11.1f}"
              f"{row['hellos_sent_per_drone_s_mean']:9.2f}")
    print("  " + "-" * 102)


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
    print(f"  {'setting':>8}{'rule':>8}{'static':>10}{'moving':>10}{'change':>10}")
    print("  " + "-" * 62)
    for row in summary:
        still = statistics.fmean(
            float(r["total_loss"]) for r in reference if r["rule"] == row["rule"]
        )
        moving = row["total_loss_mean"]
        print(f"  {row['setting']:>8}{row['rule']:>8}"
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
    parser.add_argument("--step", type=int, default=2, choices=(1, 2),
                        help="1 sweeps the start delay with perfect "
                             "neighbour knowledge; 2 fixes the delay at "
                             f"{STEP2_DELAY_S:.0f} s and sweeps the hello "
                             "interval. Default 2.")
    args = parser.parse_args()

    rows = run_grid(args.workers, args.step)
    summary = summarise(rows, args.step)
    print_summary(summary)
    print_against_static(summary, args.out_dir)

    if args.step == 1:
        note = (f"phase 2a step 1, M-drones flying, C-drones hover, "
                f"ideal neighbour knowledge, start delay "
                f"{'/'.join(f'{d:.0f}' for d in DELAYS_S)} s")
        stem = "2a"
    else:
        note = (f"phase 2a step 2, hellos every "
                f"{'/'.join(f'{h:.1f}' for h in HELLO_SETTINGS if h)} s "
                f"plus ideal, start delay {STEP2_DELAY_S:.0f} s")
        stem = "2a_hello"
    write_csv(os.path.join(args.out_dir, f"{stem}_runs.csv"), rows,
              RUN_COLUMNS, note)
    write_csv(os.path.join(args.out_dir, f"{stem}_summary.csv"), summary,
              SUMMARY_COLUMNS, note)


if __name__ == "__main__":
    main()
