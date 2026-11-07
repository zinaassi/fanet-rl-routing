"""
evaluate_r1_test.py — Phase 2c round 1: spacing_large on the test layouts.

Runs five policies once each on two test sets, full fleet (18 M + 7 C), the
near_gs mission at D = 650 m, the same rollout path the direct actor's test
used (train_c_ppo._rollout_job with a 7-tuple job):

    spacing_large   round-1 actor, spacing weight 0.02
    control         round-1 actor, same fine-tuning without the penalty
    direct          the direct-650 actor round 1 started from
    midpoint, hover the two fixed rules

Sets:
    401-450, run seed 9400 + (layout - 401): ALREADY SEEN. direct, midpoint
        and hover were run there once and the pictures of 413 and 432 were
        drawn from it. Those three must reproduce the saved file
        out/2c_viz_test/test_401_450_d650_source.csv exactly, or this stops.
    501-550, run seed 9500 + (layout - 501): FRESH. Nothing has used it; the
        script refuses to run if it overlaps any range already used.

Writes:
    out/2c_r1_test.csv          one row per (set, policy, layout)
    out/2c_r1_test_paired.csv   spacing_large minus each other policy, per set
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

from scripts.train_c_ppo import CAUSES, _rollout_job, write_csv

DISTANCE_M: float = 650.0

#: (name, first layout, run-seed base, number of layouts)
SETS: Tuple[Tuple[str, int, int, int], ...] = (
    ("seen_401_450", 401, 9400, 50),
    ("fresh_501_550", 501, 9500, 50),
)

#: Ranges used before this script, which the fresh set must avoid.
USED_BEFORE: Dict[str, range] = {
    "phase 1b / 2c test": range(1, 51),
    "training": range(101, 301),
    "validation": range(301, 331),
    "test 401-450": range(401, 451),
    "edge cases": range(900, 901),
}

MODELS: Dict[str, str] = {
    "spacing_large": "models/2c/c_actor_r1_spacing_large.pt",
    "control": "models/2c/c_actor_r1_control.pt",
    "direct": "models/2c/c_actor_curriculum_direct.pt",
}
POLICIES: Tuple[str, ...] = ("spacing_large", "control", "direct",
                             "midpoint", "hover")
SOURCE_CSV = "2c_viz_test/test_401_450_d650_source.csv"
SOURCE_LABEL = {"direct": "direct_final", "midpoint": "midpoint",
                "hover": "hover"}
MATCH_TOLERANCE: float = 1e-9

#: Two-sided 95% t value, 49 degrees of freedom (scipy.stats.t.ppf(0.975, 49)).
T_95_DF49: float = 2.009575

PERCENT = ("total_loss",) + tuple(CAUSES)
COLUMNS = ("set", "policy", "placement_seed", "run_seed", "total_loss",
           *CAUSES, "c_distance_flown_mean", "c_gs_distance_mean",
           "views_agree")


def check_fresh_set() -> None:
    """Stop unless the fresh set overlaps nothing used before."""
    name, first, _, count = SETS[1]
    fresh = set(range(first, first + count))
    for use, used in USED_BEFORE.items():
        overlap = sorted(fresh & set(used))
        if overlap:
            raise SystemExit(f"STOP: {name} overlaps {use}: {overlap}")
    print(f"  {name} overlaps none of: " + ", ".join(
        f"{u} {r.start}-{r.stop - 1}" for u, r in USED_BEFORE.items()))


def build_jobs(out_dir: str) -> Tuple[List[Tuple], List[Tuple[str, str]]]:
    import torch

    states = {}
    for name, path in MODELS.items():
        full = os.path.join(out_dir, path)
        if not os.path.exists(full):
            raise SystemExit(f"STOP: no model {full}")
        states[name] = torch.load(full, weights_only=True)
        print(f"  {name:>13}: {full}")
    jobs, keys = [], []
    for set_name, first, base, count in SETS:
        for layout in range(first, first + count):
            run_seed = base + (layout - first)
            for policy in POLICIES:
                if policy in states:
                    job = (layout, run_seed, states[policy], "rl", True, True,
                           DISTANCE_M)
                else:
                    job = (layout, run_seed, None, policy, True, True,
                           DISTANCE_M)
                jobs.append(job)
                keys.append((set_name, policy))
    return jobs, keys


def check_source(rows: Sequence[Dict[str, Any]], out_dir: str) -> None:
    """Stop unless direct, midpoint and hover on 401-450 match the old test."""
    with open(os.path.join(out_dir, SOURCE_CSV)) as handle:
        source = list(csv.DictReader(line for line in handle
                                     if not line.startswith("#")))
    old = {(r["policy"], int(r["placement_seed"])): r for r in source}
    worst, checked = 0.0, 0
    for row in rows:
        if row["set"] != SETS[0][0] or row["policy"] not in SOURCE_LABEL:
            continue
        ref = old[(SOURCE_LABEL[row["policy"]], row["placement_seed"])]
        if int(ref["run_seed"]) != row["run_seed"]:
            raise SystemExit(f"STOP: run seed differs for {row['policy']} "
                             f"{row['placement_seed']}")
        for column in PERCENT + ("c_distance_flown_mean",):
            gap = abs(float(ref[column]) - row[column])
            worst = max(worst, gap)
            if gap > MATCH_TOLERANCE:
                raise SystemExit(
                    f"STOP: {row['policy']} layout {row['placement_seed']} "
                    f"{column} {row[column]!r} != saved {ref[column]}")
        checked += 1
    print(f"  401-450 rerun matches the saved test: {checked} runs "
          f"(direct, midpoint, hover), largest gap {worst:.1e}")


def paired(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for set_name, *_ in SETS:
        loss = {(r["policy"], r["placement_seed"]): r["total_loss"]
                for r in rows if r["set"] == set_name}
        layouts = sorted({k[1] for k in loss})
        for other in POLICIES[1:]:
            diffs = [loss[("spacing_large", l)] - loss[(other, l)]
                     for l in layouts]
            mean = statistics.fmean(diffs)
            half = T_95_DF49 * statistics.stdev(diffs) / len(diffs) ** 0.5
            out.append({
                "set": set_name, "policy": "spacing_large", "against": other,
                "layouts": len(diffs), "mean_diff": mean,
                "ci95_low": mean - half, "ci95_high": mean + half,
                "wins": sum(d < 0 for d in diffs),
                "ties": sum(d == 0 for d in diffs),
            })
    return out


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as handle:
        return list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))


def print_tables(out_dir: str) -> None:
    rows = read_csv(os.path.join(out_dir, "2c_r1_test.csv"))
    for set_name, *_ in SETS:
        print(f"\n  {set_name}: mean % of packets created (from the saved file)")
        print(f"  {'policy':>13} {'total':>6} {'ci95':>13} "
              + " ".join(f"{c[:9]:>9}" for c in CAUSES) + "   C flown")
        for policy in POLICIES:
            mine = [r for r in rows if r["set"] == set_name
                    and r["policy"] == policy]
            total = [float(r["total_loss"]) for r in mine]
            mean = statistics.fmean(total)
            half = T_95_DF49 * statistics.stdev(total) / len(total) ** 0.5
            causes = " ".join(
                f"{statistics.fmean(float(r[c]) for r in mine):9.2f}"
                for c in CAUSES)
            flown = statistics.fmean(float(r["c_distance_flown_mean"])
                                     for r in mine)
            print(f"  {policy:>13} {mean:6.2f} [{mean - half:5.2f},"
                  f"{mean + half:6.2f}] {causes}   {flown:5.0f} m")
    print("\n  spacing_large minus each policy, paired (from the saved file)")
    for r in read_csv(os.path.join(out_dir, "2c_r1_test_paired.csv")):
        print(f"  {r['set']:>14} vs {r['against']:>8}: "
              f"{float(r['mean_diff']):+6.2f} [{float(r['ci95_low']):+6.2f}, "
              f"{float(r['ci95_high']):+6.2f}]  lower on {r['wins']}/"
              f"{r['layouts']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c round 1: spacing_large on 401-450 and 501-550."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    args = parser.parse_args()

    print("Checking the fresh set ...")
    check_fresh_set()
    print("Models:")
    jobs, keys = build_jobs(args.out_dir)
    print(f"Running {len(jobs)} episodes on {args.workers} workers ...",
          flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        results = list(pool.map(_rollout_job, jobs, chunksize=1))

    rows = []
    for job, (set_name, policy), result in zip(jobs, keys, results):
        row = {"set": set_name, "policy": policy, "placement_seed": job[0],
               "run_seed": job[1], **result}
        for column in PERCENT:
            row[column] = 100.0 * row[column]
        rows.append(row)
    disagreed = [r for r in rows if not r["views_agree"]]
    print(f"  views agreed in {len(rows) - len(disagreed)} of {len(rows)} runs")
    if disagreed:
        raise SystemExit("STOP: the ACK views disagreed")
    check_source(rows, args.out_dir)

    note = (f"phase 2c round 1 test, near_gs D = {DISTANCE_M:.0f}, 18M+7C; "
            f"401-450 already seen, 501-550 fresh")
    write_csv(os.path.join(args.out_dir, "2c_r1_test.csv"), rows, COLUMNS,
              note)
    write_csv(os.path.join(args.out_dir, "2c_r1_test_paired.csv"),
              paired(rows),
              ("set", "policy", "against", "layouts", "mean_diff",
               "ci95_low", "ci95_high", "wins", "ties"), note)
    print_tables(args.out_dir)


if __name__ == "__main__":
    main()
