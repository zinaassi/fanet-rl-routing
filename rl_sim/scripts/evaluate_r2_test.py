"""
evaluate_r2_test.py — Phase 2c round 2: the test sets.

Seven policies, once each, on two FRESH test sets, through the same episode
code the round-2 validation used (train_c_ppo_round2.run_episode, greedy,
normal reward):

    A, B, C          round-2 best actors
    control          round-1 control actor
    direct           the direct-650 actor round 2 started from
    midpoint, hover  the two fixed rules

Sets:
    full    601-650, run seed 9600 + (layout - 601): 18 M + 7 C, near_gs,
            D = 650 m, the full-fleet validation world.
    varied  651-750, run seed 9700 + (layout - 651): drawn like the varied
            validation set (types a, b, c, d in turn, 25 of each) from
            SeedSequence([layout, VARIED_TEST_TAG]), a tag of their own.

The TWO MAIN COMPARISONS, named before running: B minus control and B minus
midpoint. Every policy is also paired against control and against midpoint,
per set and per mission type.

Before the test, every policy is rerun on both validation sets and must
reproduce the saved rows (out/2c_r2_validation.csv for A, B, C and
out/2c_r2_references.csv for the rest) to 1e-9, or this stops.

Writes:
    out/2c_r2_test.csv          one row per (policy, set, layout)
    out/2c_r2_test_paired.csv   each policy minus control and minus midpoint
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_c_ppo import CAUSES, write_csv
from scripts.train_c_ppo_round1 import read_csv
from scripts.train_c_ppo_round2 import (
    CONTROL_ACTOR,
    DIRECT_ACTOR,
    NORMAL_PLAN,
    ROW_COLUMNS,
    TYPES,
    VARIED_TEST_TAG,
    draw_varied,
    full_jobs,
    run_episode,
    varied_jobs,
)

FULL_LAYOUTS = range(601, 651)
FULL_SEED_BASE = 9600
VARIED_LAYOUTS = range(651, 751)
VARIED_SEED_BASE = 9700

#: Layout ranges and run seeds used before, which the test sets must avoid.
USED_LAYOUTS: Dict[str, range] = {
    "phase 1b / 2c test": range(1, 51),
    "training": range(101, 301),
    "full validation": range(301, 331),
    "varied validation": range(331, 361),
    "test 401-450": range(401, 451),
    "round-1 test": range(501, 551),
    "edge cases": range(900, 901),
}
USED_SEEDS: Dict[str, range] = {
    "full validation": range(9100, 9130),
    "varied validation": range(9200, 9230),
    "test 401-450": range(9400, 9450),
    "round-1 test": range(9500, 9550),
    "edge cases": range(9900, 9905),
    "term measurement": range(12345, 12346),
}

MODELS: Dict[str, str] = {
    "A": os.path.join("models", "2c", "c_actor_r2_A.pt"),
    "B": os.path.join("models", "2c", "c_actor_r2_B.pt"),
    "C": os.path.join("models", "2c", "c_actor_r2_C.pt"),
    "control": CONTROL_ACTOR,
    "direct": DIRECT_ACTOR,
}
POLICIES: Tuple[str, ...] = ("control", "A", "B", "C", "direct", "midpoint",
                             "hover")
BASELINES: Tuple[str, ...] = ("control", "midpoint")
MAIN: Tuple[Tuple[str, str], ...] = (("B", "control"), ("B", "midpoint"))
MATCH_TOLERANCE = 1e-9

#: Two-sided 95% t values, scipy.stats.t.ppf(0.975, df).
T_95: Dict[int, float] = {24: 2.063899, 49: 2.009575, 99: 1.984217}

PERCENT = ("total_loss",) + tuple(CAUSES)


def check_fresh() -> None:
    """Stop unless both test sets overlap nothing used before."""
    layouts = set(FULL_LAYOUTS) | set(VARIED_LAYOUTS)
    seeds = (set(range(FULL_SEED_BASE, FULL_SEED_BASE + len(FULL_LAYOUTS)))
             | set(range(VARIED_SEED_BASE,
                         VARIED_SEED_BASE + len(VARIED_LAYOUTS))))
    for what, used, mine in (("layouts", USED_LAYOUTS, layouts),
                             ("run seeds", USED_SEEDS, seeds)):
        for use, r in used.items():
            overlap = sorted(mine & set(r))
            if overlap:
                raise SystemExit(f"STOP: test {what} overlap {use}: "
                                 f"{overlap[:5]}")
    print("  layouts 601-650, 651-750 and run seeds 9600-9649, 9700-9799 "
          "overlap nothing used before")


def test_plans() -> List[Tuple[int, int, Dict[str, Any]]]:
    """(layout, run seed, plan) for every test world."""
    out = [(layout, FULL_SEED_BASE + i, NORMAL_PLAN)
           for i, layout in enumerate(FULL_LAYOUTS)]
    for i, layout in enumerate(VARIED_LAYOUTS):
        rng = np.random.default_rng(
            np.random.SeedSequence([layout, VARIED_TEST_TAG]))
        out.append((layout, VARIED_SEED_BASE + i,
                    draw_varied(rng, TYPES[i % len(TYPES)])))
    return out


def load_states(out_dir: str) -> Dict[str, Any]:
    import torch

    states: Dict[str, Any] = {}
    for name in POLICIES:
        if name in MODELS:
            path = os.path.join(out_dir, MODELS[name])
            if not os.path.exists(path):
                raise SystemExit(f"STOP: no model {path}")
            states[name] = torch.load(path, weights_only=True)
            print(f"  {name:>8}: {path}")
        else:
            states[name] = None
    return states


def policy_of(name: str) -> str:
    return name if name in ("midpoint", "hover") else "rl"


def to_row(name: str, job: Tuple, r: Dict[str, Any]) -> Dict[str, Any]:
    plan = job[5]
    return {
        "policy": name,
        "set": "full" if plan["kind"] == "normal" else "varied",
        "placement_seed": job[0], "run_seed": job[1], "type": plan["kind"],
        "n_m": plan["n_m"], "n_c": plan["n_c"],
        "total_loss": 100.0 * r["total_loss"],
        **{c: 100.0 * r[c] for c in CAUSES},
        "c_distance_flown_mean": r["c_distance_flown_mean"],
        "min_c_to_c_m": r["min_c_to_c_m"],
        "worst_m_loss": r["worst_m_loss"],
        "pct_m_over_half": r["pct_m_over_half"],
        "views_agree": r["views_agree"],
    }


def run_all(pool: Any, named_jobs: List[Tuple[str, Tuple]]
            ) -> List[Dict[str, Any]]:
    results = list(pool.map(run_episode, [j for _, j in named_jobs],
                            chunksize=1))
    rows = [to_row(name, job, r) for (name, job), r in
            zip(named_jobs, results)]
    bad = [r for r in rows if not r["views_agree"]]
    if bad:
        raise SystemExit(f"STOP: the ACK views disagreed in {len(bad)} runs")
    return rows


def check_validation(pool: Any, states: Dict[str, Any], out_dir: str) -> None:
    """Every policy must reproduce its saved validation rows."""
    jobs = []
    for name in POLICIES:
        for job in (full_jobs(states[name], policy_of(name))
                    + varied_jobs(states[name], policy_of(name))):
            jobs.append((name, job))
    rows = run_all(pool, jobs)
    saved = {}
    for f in ("2c_r2_validation.csv", "2c_r2_references.csv"):
        for r in read_csv(os.path.join(out_dir, f)):
            saved[(r["policy"], r["set"], int(r["placement_seed"]))] = r
    worst = 0.0
    for row in rows:
        ref = saved[(row["policy"], row["set"], row["placement_seed"])]
        if int(ref["run_seed"]) != row["run_seed"]:
            raise SystemExit(f"STOP: run seed differs for {row['policy']} "
                             f"{row['placement_seed']}")
        for column in PERCENT + ("c_distance_flown_mean",):
            gap = abs(float(ref[column]) - row[column])
            worst = max(worst, gap)
            if gap > MATCH_TOLERANCE:
                raise SystemExit(
                    f"STOP: {row['policy']} {row['set']} "
                    f"{row['placement_seed']} {column} {row[column]!r} != "
                    f"saved {ref[column]}")
    print(f"  validation rerun matches the saved rows: {len(rows)} runs, "
          f"largest gap {worst:.1e}")


def paired(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    loss = {(r["policy"], r["set"], r["placement_seed"]): r["total_loss"]
            for r in rows}
    kinds = {(r["set"], r["placement_seed"]): r["type"] for r in rows}
    groups = [("full", "all"), ("varied", "all")] + [("varied", k)
                                                     for k in TYPES]
    out = []
    for base in BASELINES:
        for name in POLICIES:
            if name == base:
                continue
            for which, kind in groups:
                layouts = sorted(l for (s, l), k in kinds.items()
                                 if s == which and (kind == "all" or k == kind))
                diffs = [loss[(name, which, l)] - loss[(base, which, l)]
                         for l in layouts]
                m = statistics.fmean(diffs)
                half = (T_95[len(diffs) - 1] * statistics.stdev(diffs)
                        / math.sqrt(len(diffs)))
                out.append({
                    "policy": name, "against": base, "set": which,
                    "type": kind, "layouts": len(diffs), "mean_diff": m,
                    "ci95_low": m - half, "ci95_high": m + half,
                    "wins": sum(d < 0 for d in diffs),
                    "ties": sum(d == 0 for d in diffs),
                    "main": (name, base) in MAIN,
                })
    return out


def print_tables(out_dir: str) -> None:
    rows = read_csv(os.path.join(out_dir, "2c_r2_test.csv"))

    def mean(g: Sequence[Dict], key: str) -> float:
        return statistics.fmean(float(r[key]) for r in g)

    print("\n  Full-fleet test 601-650, mean % of packets created "
          "(from out/2c_r2_test.csv)")
    print(f"  {'policy':>9} {'total':>6} " + " ".join(f"{c[:8]:>8}"
                                                   for c in CAUSES)
          + f" {'C flown':>8}")
    for name in POLICIES:
        g = [r for r in rows if r["policy"] == name and r["set"] == "full"]
        print(f"  {name:>9} {mean(g, 'total_loss'):6.2f} "
              + " ".join(f"{mean(g, c):8.2f}" for c in CAUSES)
              + f" {mean(g, 'c_distance_flown_mean'):7.0f}m")
    print("\n  Varied test 651-750 (25 worlds per type)")
    print(f"  {'policy':>9} {'total':>6} {'worst M':>8} {'M >50%':>7}   "
          + "  ".join(f"type {k}" for k in TYPES))
    for name in POLICIES:
        g = [r for r in rows if r["policy"] == name and r["set"] == "varied"]
        types = "  ".join(f"{mean([r for r in g if r['type'] == k], 'total_loss'):6.2f}"
                          for k in TYPES)
        print(f"  {name:>9} {mean(g, 'total_loss'):6.2f} "
              f"{mean(g, 'worst_m_loss'):8.2f} "
              f"{mean(g, 'pct_m_over_half'):6.1f}%   {types}")
    p = read_csv(os.path.join(out_dir, "2c_r2_test_paired.csv"))
    for base in BASELINES:
        print(f"\n  Paired against {base} (points of total loss; "
              f"* = main comparison)")
        for r in p:
            if r["against"] != base:
                continue
            star = "*" if r["main"] == "True" else " "
            label = r["set"] if r["type"] == "all" else f"type {r['type']}"
            print(f" {star}{r['policy']:>8} {label:>7}  "
                  f"{float(r['mean_diff']):+6.2f} [{float(r['ci95_low']):+.2f}"
                  f", {float(r['ci95_high']):+.2f}]  lower on {r['wins']}/"
                  f"{r['layouts']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c round 2: test sets 601-650 and 651-750.")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    if args.print_only:
        print_tables(args.out_dir)
        return

    print("Main comparisons, named before running: "
          + ", ".join(f"{a} minus {b}" for a, b in MAIN))
    print("Checking the test sets ...")
    check_fresh()
    plans = test_plans()
    counts = {k: sum(p["kind"] == k for _, _, p in plans)
              for k in ("normal",) + TYPES}
    print(f"  worlds: {counts}")
    print("Models:")
    states = load_states(args.out_dir)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        print("Reproducing validation ...", flush=True)
        check_validation(pool, states, args.out_dir)
        jobs = [(name, (layout, seed, states[name], True, "normal", plan,
                        policy_of(name)))
                for layout, seed, plan in plans for name in POLICIES]
        print(f"Running {len(jobs)} test episodes on {args.workers} "
              f"workers ...", flush=True)
        rows = run_all(pool, jobs)

    note = ("phase 2c round 2 TEST: full 601-650 (run seed 9600+), varied "
            "651-750 (run seed 9700+, tag 0x2C2C); main comparisons B minus "
            "control and B minus midpoint")
    write_csv(os.path.join(args.out_dir, "2c_r2_test.csv"), rows,
              ROW_COLUMNS, note)
    pr = paired(rows)
    write_csv(os.path.join(args.out_dir, "2c_r2_test_paired.csv"), pr,
              list(pr[0]), note)
    print_tables(args.out_dir)


if __name__ == "__main__":
    main()
