"""
recheck_2c_m2.py — Phase 2c M3 item 0: re-check M2 on seeds it never saw.

M2 picked its kept version with run seeds 9001-9003 and then REPORTED on
those same three seeds, so the headline numbers were selected on the set
they were measured on. This re-runs the three kept actors, and the three
rules, on ten run seeds that took no part in M2: neither the evaluation
seeds nor any of the seeds its rollouts actually drew.

That last part is checked rather than assumed. Training draws its run seeds
from ``default_rng(training seed)``, which is reproducible now that the
exploration is seeded, so the exact list M2 used can be regenerated and the
new seeds tested against it.

Writes:
    out/2c_m2_recheck.csv       one row per (policy, run seed)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.train_c_ppo import (
    CAUSES,
    EPISODES_PER_UPDATE,
    EVAL_RUN_SEEDS,
    LOAD_MS,
    ROUTING,
    TRAIN_LAYOUT,
    _rollout_job,
    write_csv,
)

#: Ten run seeds that M2 never touched. Checked, not trusted.
RECHECK_RUN_SEEDS: Tuple[int, ...] = tuple(range(7001, 7011))

#: The training seeds M2 ran.
TRAINING_SEEDS: Tuple[int, ...] = (0, 1, 2)


def updates_per_seed(log_path: str) -> Dict[int, int]:
    """Return how many updates each training seed ran, from M2's log.

    Args:
        log_path: out/2c_ppo_log.csv.

    Returns:
        Training seed -> number of updates.
    """
    counts: Dict[int, int] = {}
    with open(log_path) as handle:
        rows = csv.DictReader(line for line in handle
                              if not line.startswith("#"))
        for row in rows:
            seed = int(row["seed"])
            counts[seed] = counts.get(seed, 0) + 1
    return counts


def seeds_used_in_training(counts: Dict[int, int]) -> set:
    """Replay the run seeds M2's rollouts actually drew.

    train_one_seed seeds ``default_rng(training seed)`` once and draws
    ``episodes`` run seeds per update, so replaying the same draws for the
    same number of updates gives back the exact list.

    Args:
        counts: Training seed -> number of updates it ran.

    Returns:
        Every run seed any rollout used.
    """
    used: set = set()
    for seed, updates in sorted(counts.items()):
        rng = np.random.default_rng(seed)
        for _ in range(updates):
            used.update(
                int(s) for s in
                rng.integers(1, 10 ** 8, size=EPISODES_PER_UPDATE).tolist()
            )
    return used


def check_seeds_are_new(counts: Dict[int, int]) -> None:
    """Stop unless the re-check seeds took no part in M2.

    Args:
        counts: Training seed -> number of updates it ran.

    Raises:
        SystemExit: If any re-check seed was used for training or evaluation.
    """
    used = seeds_used_in_training(counts)
    clash_train = sorted(set(RECHECK_RUN_SEEDS) & used)
    clash_eval = sorted(set(RECHECK_RUN_SEEDS) & set(EVAL_RUN_SEEDS))
    print(f"  M2 drew {len(used)} distinct run seeds across "
          f"{sum(counts.values())} updates")
    if clash_train or clash_eval:
        print(f"STOP: re-check seeds were not new. training {clash_train}, "
              f"evaluation {clash_eval}")
        raise SystemExit(1)
    print(f"  none of {RECHECK_RUN_SEEDS[0]}-{RECHECK_RUN_SEEDS[-1]} appears "
          f"in training or in {EVAL_RUN_SEEDS}")


def build_jobs(out_dir: str) -> Tuple[List[Tuple], List[str]]:
    """Return one rollout job per (policy, re-check run seed).

    Args:
        out_dir: Where the models live.

    Returns:
        The jobs and their policy labels.
    """
    import torch

    clone = torch.load(os.path.join(out_dir, "models", "c_actor_bc.pt"),
                       weights_only=True)
    actors: Dict[int, Any] = {}
    for seed in TRAINING_SEEDS:
        path = os.path.join(out_dir, "models", f"c_actor_ppo_seed{seed}.pt")
        if not os.path.exists(path):
            print(f"STOP: no M2 model at {path}")
            raise SystemExit(1)
        actors[seed] = torch.load(path, weights_only=True)

    jobs: List[Tuple] = []
    labels: List[str] = []
    for run_seed in RECHECK_RUN_SEEDS:
        jobs.append((TRAIN_LAYOUT, run_seed, None, "hover", True))
        labels.append("hover")
        jobs.append((TRAIN_LAYOUT, run_seed, None, "midpoint", True))
        labels.append("midpoint")
        jobs.append((TRAIN_LAYOUT, run_seed, clone, "clone", True))
        labels.append("clone")
        for seed in TRAINING_SEEDS:
            jobs.append((TRAIN_LAYOUT, run_seed, actors[seed], "rl", True))
            labels.append(f"rl seed {seed}")
    return jobs, labels


def print_table(rows: Sequence[Dict[str, Any]], old: Dict[str, float]) -> None:
    """Print the re-check beside what M2 reported.

    Args:
        rows: One row per (policy, run seed).
        old:  Policy -> the mean M2 reported on 9001-9003, in %.
    """
    policies = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    width = 118
    print(f"\n  Layout {TRAIN_LAYOUT}, {len(RECHECK_RUN_SEEDS)} run seeds M2 "
          f"never saw — % of packets created")
    print("  " + "-" * width)
    header = f"{'policy':>14}{'total':>9}{'std':>8}{'M2 said':>10}{'move':>8}"
    header += "".join(f"{cause[:9]:>11}" for cause in CAUSES[:4])
    print("  " + header)
    print("  " + "-" * width)
    for policy in policies:
        mine = [r for r in rows if r["policy"] == policy]
        total = statistics.fmean(r["total_loss"] for r in mine) * 100
        spread = (statistics.stdev([r["total_loss"] for r in mine]) * 100
                  if len(mine) > 1 else 0.0)
        before = old.get(policy)
        line = f"{policy:>14}{total:>8.2f}%{spread:>8.2f}"
        line += (f"{before:>9.2f}%{total - before:>+8.2f}" if before is not None
                 else f"{'-':>10}{'-':>8}")
        for cause in CAUSES[:4]:
            line += f"{statistics.fmean(r[cause] for r in mine) * 100:>10.2f}%"
        print("  " + line)
    print("  " + "-" * width)


def m2_reported(path: str) -> Dict[str, float]:
    """Return the means M2 reported on 9001-9003, for the side-by-side.

    Args:
        path: out/2c_ppo_compare.csv.

    Returns:
        Policy -> mean total loss in %, empty if the file is missing.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))
    policies = {row["policy"] for row in rows}
    return {
        policy: statistics.fmean(
            float(row["total_loss"]) for row in rows
            if row["policy"] == policy
        ) * 100
        for policy in policies
    }


def main() -> None:
    """Re-run M2's kept actors and the rules on unseen run seeds."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M3 item 0: re-check M2 on unseen run seeds."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    args = parser.parse_args()

    log_path = os.path.join(args.out_dir, "2c_ppo_log.csv")
    if not os.path.exists(log_path):
        print(f"STOP: no M2 log at {log_path}")
        raise SystemExit(1)

    counts = updates_per_seed(log_path)
    print("Checking the re-check seeds are new ...")
    check_seeds_are_new(counts)

    jobs, labels = build_jobs(args.out_dir)
    context = multiprocessing.get_context("spawn")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
            if args.workers > 1 else None)
    print(f"\nRunning {len(jobs)} runs on layout {TRAIN_LAYOUT} ...")
    try:
        results = (list(pool.map(_rollout_job, jobs)) if pool is not None
                   else [_rollout_job(job) for job in jobs])
    finally:
        if pool is not None:
            pool.shutdown()

    rows = [{"policy": label, "run_seed": job[1], **result}
            for job, label, result in zip(jobs, labels, results)]
    disagreed = [row for row in rows if not row["views_agree"]]
    print(f"  views agreed in {len(rows) - len(disagreed)} of {len(rows)} runs")
    if disagreed:
        raise RuntimeError("the ACK views disagreed with the simulator")

    print_table(rows, m2_reported(os.path.join(args.out_dir,
                                               "2c_ppo_compare.csv")))

    note = (f"phase 2c M3 item 0, M2 re-checked on run seeds "
            f"{RECHECK_RUN_SEEDS[0]}-{RECHECK_RUN_SEEDS[-1]}, layout "
            f"{TRAIN_LAYOUT}, static mission, hello 1 s, {ROUTING} routing")
    write_csv(os.path.join(args.out_dir, "2c_m2_recheck.csv"), rows,
              ("policy", "run_seed", "total_loss", *CAUSES,
               "c_distance_flown_mean", "c_gs_distance_mean", "mean_reward",
               "views_agree"), note)


if __name__ == "__main__":
    main()
