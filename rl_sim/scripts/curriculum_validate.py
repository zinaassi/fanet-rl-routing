"""
curriculum_validate.py — Phase 2c: the curriculum's actors on validation.

No training. Runs five policies over the five curriculum stages on the
thirty validation layouts, with the run seeds step 1b used, so the learned
actors land in the same table as the hand-written rules:

    hover            the C-drones stay where they started
    midpoint         the hand-written rule
    m3_seed1         what both runs started from, before any curriculum
    staged_final     S5's best weights after climbing D = 250 -> 650
    direct_final     the same budget spent entirely at D = 650

Every policy is evaluated at every stage, not just the one it trained on,
which is what shows whether the curriculum bought generality or only moved
the actor to the hardest rung.

Paired differences are taken layout by layout on identical seeds, so the
confidence interval is over thirty matched pairs rather than two means.

Writes:
    out/2c_curriculum_validation.csv          one row per (stage, policy, layout)
    out/2c_curriculum_validation_summary.csv  one row per (stage, policy)
    out/2c_curriculum_validation_paired.csv   the paired differences
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from scripts.curriculum_step1 import (
    CAUSES,
    VALIDATION_LAYOUTS,
    _mean,
    _std,
    run_seed_for,
    write_csv,
)
from scripts.metrics_1a import time_window_labels
from scripts.train_c_ppo import _rollout_job
from scripts.train_c_ppo_curriculum import MODEL_DIR, STAGES

#: Policy label -> the model file it needs, or None for a scripted rule.
POLICIES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("hover", None),
    ("midpoint", None),
    ("m3_seed1", "c_actor_m3_seed1.pt"),
    ("staged_final", "c_actor_curriculum_staged.pt"),
    ("direct_final", "c_actor_curriculum_direct.pt"),
)

#: The paired comparisons the checkpoint asks for.
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("staged_final", "midpoint"),
    ("staged_final", "hover"),
    ("staged_final", "direct_final"),
)

#: Student's t at 95% for 29 degrees of freedom. Thirty layouts is small
#: enough that the normal approximation is noticeably too tight.
T_95_DF29: float = 2.045

SUMMARY_MEANS: Tuple[str, ...] = (
    ("total_loss",) + CAUSES
    + ("pct_packets_connected", "loss_while_connected")
    + tuple(f"loss_{label}" for label in time_window_labels())
    + ("m_gs_distance_end_mean", "pct_m_beyond_reach_end",
       "pct_m_clipped_end", "c_distance_flown_mean")
)


#: The follow-up test adds one policy, and its own comparisons, under its own
#: file names, so the committed curriculum tables are left as they are.
S5_TEST_POLICY: Tuple[str, str] = ("s5_from_s4",
                                   "c_actor_curriculum_s5_from_s4.pt")
S5_TEST_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("s5_from_s4", "direct_final"),
    ("s5_from_s4", "staged_final"),
    ("s5_from_s4", "midpoint"),
    ("s5_from_s4", "hover"),
)


def load_actors(out_dir: str) -> Dict[str, Any]:
    """Load the weights of every learned policy.

    Args:
        out_dir: Where the models live.

    Returns:
        Label -> state dict.

    Raises:
        SystemExit: If a model is missing, naming which.
    """
    import torch

    actors: Dict[str, Any] = {}
    missing: List[str] = []
    for label, filename in POLICIES:
        if filename is None:
            continue
        path = os.path.join(out_dir, MODEL_DIR, filename)
        if not os.path.exists(path):
            missing.append(path)
            continue
        actors[label] = torch.load(path, weights_only=True)
    if missing:
        raise SystemExit("missing models:\n  " + "\n  ".join(missing))
    return actors


def build_jobs(actors: Dict[str, Any]
               ) -> Tuple[List[Tuple], List[Tuple[float, str]]]:
    """Return every rollout job, and the (stage, policy) each belongs to.

    Args:
        actors: Label -> weights.

    Returns:
        The jobs and their labels, in the same order.
    """
    jobs: List[Tuple] = []
    labels: List[Tuple[float, str]] = []
    for distance in STAGES:
        for label, filename in POLICIES:
            policy = "rl" if filename is not None else label
            state = actors.get(label)
            for layout in VALIDATION_LAYOUTS:
                # greedy=True: no exploration at evaluation time.
                jobs.append((layout, run_seed_for(layout), state, policy,
                             True, True, distance))
                labels.append((distance, label))
    return jobs, labels


def run_grid(workers: int, actors: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run every job and return its rows.

    Args:
        workers: Pool size.
        actors:  Label -> weights.

    Returns:
        One row per (stage, policy, layout).

    Raises:
        RuntimeError: If any run's ACK view disagreed with the simulator.
    """
    jobs, labels = build_jobs(actors)
    print(f"Running {len(jobs)} validation runs: {len(STAGES)} stages x "
          f"{len(POLICIES)} policies x {len(VALIDATION_LAYOUTS)} layouts, "
          f"no training ...", flush=True)

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        results = list(pool.map(_rollout_job, jobs))

    rows = [{
        "mission_distance_m": distance,
        "stage": f"S{STAGES.index(distance) + 1}",
        "policy": label,
        "placement_seed": job[0],
        "run_seed": job[1],
        "total_loss": 100.0 * result["total_loss"],
        **{cause: 100.0 * result[cause] for cause in CAUSES},
        "pct_packets_connected": 100.0 * result["pct_packets_connected"],
        "loss_while_connected": 100.0 * result["loss_while_connected"],
        **{f"loss_{name}": 100.0 * (result[f"loss_{name}"] or 0.0)
           for name in time_window_labels()},
        "m_gs_distance_end_mean": result["m_gs_distance_end_mean"],
        "pct_m_beyond_reach_end": result["pct_m_beyond_reach_end"],
        "pct_m_clipped_end": result["pct_m_clipped_end"],
        "c_distance_flown_mean": result["c_distance_flown_mean"],
        "views_agree": result["views_agree"],
    } for job, (distance, label), result in zip(jobs, labels, results)]

    agreed = sum(1 for row in rows if row["views_agree"])
    print(f"  views agreed in {agreed} of {len(rows)} runs")
    if agreed != len(rows):
        raise RuntimeError("the ACK views disagreed with the simulator")
    return rows


def summarise(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse to one row per (stage, policy).

    Args:
        rows: Every run's row.

    Returns:
        One row per setting, stages outermost.
    """
    out: List[Dict[str, Any]] = []
    for distance in STAGES:
        for label, _ in POLICIES:
            group = [row for row in rows
                     if row["mission_distance_m"] == distance
                     and row["policy"] == label]
            summary: Dict[str, Any] = {
                "mission_distance_m": distance,
                "stage": f"S{STAGES.index(distance) + 1}",
                "policy": label,
                "layouts": len(group),
                "total_loss_std": _std([row["total_loss"] for row in group]),
            }
            for column in SUMMARY_MEANS:
                summary[column] = _mean([row[column] for row in group])
            out.append(summary)
    return out


def confidence_interval(differences: Sequence[float]
                        ) -> Tuple[float, float, float]:
    """Return the mean difference and its 95% interval.

    A paired interval: the layouts and seeds are identical on both sides, so
    the quantity with a distribution is the per-layout difference, not either
    mean on its own.

    Args:
        differences: One per layout.

    Returns:
        (mean, low, high).
    """
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean, mean, mean
    half = T_95_DF29 * statistics.stdev(differences) / math.sqrt(
        len(differences))
    return mean, mean - half, mean + half


def paired_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the paired differences for every stage and pair.

    Args:
        rows: Every run's row.

    Returns:
        One row per (stage, pair).
    """
    out: List[Dict[str, Any]] = []
    for distance in STAGES:
        at_stage = [row for row in rows
                    if row["mission_distance_m"] == distance]
        by_policy = {
            label: {row["placement_seed"]: row["total_loss"]
                    for row in at_stage if row["policy"] == label}
            for label, _ in POLICIES
        }
        for actor, other in PAIRS:
            differences = [by_policy[actor][layout] - by_policy[other][layout]
                           for layout in VALIDATION_LAYOUTS]
            mean, low, high = confidence_interval(differences)
            out.append({
                "mission_distance_m": distance,
                "stage": f"S{STAGES.index(distance) + 1}",
                "actor": actor,
                "against": other,
                "layouts": len(differences),
                "mean_diff": mean,
                "ci95_low": low,
                "ci95_high": high,
                "wins": sum(1 for d in differences if d < 0),
            })
    return out


def print_totals(summary: Sequence[Dict[str, Any]]) -> None:
    """Print total loss and loss by cause, stage by stage.

    Args:
        summary: The per-setting rows.
    """
    print("\n  Validation 301-330 — % of packets created, by stage")
    header = f"  {'stage':>5} {'D':>5} {'policy':>13} {'total':>8} {'std':>6}"
    header += "".join(f"{cause[:9]:>11}" for cause in CAUSES)
    header += f"{'conn%':>9}{'loss|conn':>11}"
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    previous = None
    for row in summary:
        if previous is not None and row["stage"] != previous:
            print()
        previous = row["stage"]
        line = (f"  {row['stage']:>5} {row['mission_distance_m']:>5.0f} "
                f"{row['policy']:>13} {row['total_loss']:>7.2f}% "
                f"{row['total_loss_std']:>6.2f}")
        line += "".join(f"{row[cause]:>10.2f}%" for cause in CAUSES)
        line += (f"{row['pct_packets_connected']:>8.2f}%"
                 f"{row['loss_while_connected']:>10.2f}%")
        print(line)
    print("  " + "-" * (len(header) + 2))


def print_windows(summary: Sequence[Dict[str, Any]]) -> None:
    """Print loss per ten-second slice.

    Args:
        summary: The per-setting rows.
    """
    labels = time_window_labels()
    print("\n  Loss per ten seconds — % of packets created in that slice")
    header = f"  {'stage':>5} {'policy':>13}"
    header += "".join(f"{label:>9}" for label in labels)
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    previous = None
    for row in summary:
        if previous is not None and row["stage"] != previous:
            print()
        previous = row["stage"]
        line = f"  {row['stage']:>5} {row['policy']:>13}"
        line += "".join(f"{row[f'loss_{label}']:>8.2f}%" for label in labels)
        print(line)
    print("  " + "-" * (len(header) + 2))


def print_spread(summary: Sequence[Dict[str, Any]]) -> None:
    """Print the mission spread and what the C-drones flew.

    Args:
        summary: The per-setting rows.
    """
    print("\n  The mission and the relays, at the last step")
    header = (f"  {'stage':>5} {'D':>5} {'policy':>13} {'mean M from GS':>16} "
              f"{'% M beyond':>12} {'% M clipped':>13} {'C flown':>10}")
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    previous = None
    for row in summary:
        if previous is not None and row["stage"] != previous:
            print()
        previous = row["stage"]
        print(f"  {row['stage']:>5} {row['mission_distance_m']:>5.0f} "
              f"{row['policy']:>13} "
              f"{row['m_gs_distance_end_mean']:>15.1f}m "
              f"{row['pct_m_beyond_reach_end']:>11.1f}% "
              f"{row['pct_m_clipped_end']:>12.1f}% "
              f"{row['c_distance_flown_mean']:>9.0f}m")
    print("  " + "-" * (len(header) + 2))


def print_paired(paired: Sequence[Dict[str, Any]]) -> None:
    """Print the paired differences with their intervals.

    Args:
        paired: The paired rows.
    """
    print("\n  Paired over the 30 validation layouts — points of total loss,"
          " negative means the actor loses fewer")
    header = (f"  {'stage':>5} {'D':>5} {'actor':>13} {'against':>13} "
              f"{'mean diff':>11} {'95% CI':>22} {'wins':>7}")
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    previous = None
    for row in paired:
        if previous is not None and row["stage"] != previous:
            print()
        previous = row["stage"]
        interval = f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]"
        crosses = "" if row["ci95_low"] * row["ci95_high"] > 0 else "  n.s."
        print(f"  {row['stage']:>5} {row['mission_distance_m']:>5.0f} "
              f"{row['actor']:>13} {row['against']:>13} "
              f"{row['mean_diff']:>+10.2f} {interval:>22} "
              f"{row['wins']:>4}/{row['layouts']}{crosses}")
    print("  " + "-" * (len(header) + 2))
    print("  n.s. = the 95% interval includes zero")


def main() -> None:
    """Run the validation grid and print the tables."""
    parser = argparse.ArgumentParser(
        description="Phase 2c: the curriculum's actors on validation."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--s5-test", action="store_true",
                        help="add the S5-from-S4 actor; writes "
                             "2c_curriculum_s5test_*.csv instead")
    args = parser.parse_args()

    global POLICIES, PAIRS
    prefix = "2c_curriculum_validation"
    if args.s5_test:
        POLICIES = POLICIES + (S5_TEST_POLICY,)
        PAIRS = S5_TEST_PAIRS
        prefix = "2c_curriculum_s5test_validation"

    actors = load_actors(args.out_dir)
    rows = run_grid(args.workers, actors)
    summary = summarise(rows)
    paired = paired_rows(rows)

    print_totals(summary)
    print_windows(summary)
    print_spread(summary)
    print_paired(paired)

    windows = [f"loss_{label}" for label in time_window_labels()]
    note = ("phase 2c curriculum, validation 301-330, near_gs, "
            "stages " + "/".join(f"{d:.0f}" for d in STAGES)
            + " m, avoid=True, greedy routing, NO training")

    write_csv(os.path.join(args.out_dir, f"{prefix}.csv"),
              rows,
              ["mission_distance_m", "stage", "policy", "placement_seed",
               "run_seed", "total_loss"] + list(CAUSES)
              + ["pct_packets_connected", "loss_while_connected"] + windows
              + ["m_gs_distance_end_mean", "pct_m_beyond_reach_end",
                 "pct_m_clipped_end", "c_distance_flown_mean", "views_agree"],
              note)
    write_csv(
        os.path.join(args.out_dir, f"{prefix}_summary.csv"),
        summary,
        ["mission_distance_m", "stage", "policy", "layouts", "total_loss",
         "total_loss_std"] + list(CAUSES)
        + ["pct_packets_connected", "loss_while_connected"] + windows
        + ["m_gs_distance_end_mean", "pct_m_beyond_reach_end",
           "pct_m_clipped_end", "c_distance_flown_mean"],
        note)
    write_csv(
        os.path.join(args.out_dir, f"{prefix}_paired.csv"),
        paired,
        ["mission_distance_m", "stage", "actor", "against", "layouts",
         "mean_diff", "ci95_low", "ci95_high", "wins"],
        note)


if __name__ == "__main__":
    main()
