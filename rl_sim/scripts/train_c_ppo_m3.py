"""
train_c_ppo_m3.py — Phase 2c M3: train the C-drones on all training layouts.

M2 trained and chose on ONE layout, so it showed only that the actor can
learn that layout. M3 keeps the method exactly and changes what it is shown:
every episode draws a layout from the training range, and the kept version is
chosen on thirty validation layouts the training never sees.

Everything else is M2's: the static world, 1 s hellos, greedy routing, 200 ms
traffic, seven C-drones sharing one actor, a centralised critic used only
while training, the same reward with its 0.01 movement cost, the same PPO
settings, the same start from the clone of midpoint, the same ten
critic-only updates and the same "stay near midpoint" term fading over
fifty. The PPO loop itself is train_c_ppo.train_one_seed, called with a
different layout sampler and a different evaluation.

Writes:
    out/2c_m3_log.csv           one row per update, written as it goes
    out/2c_m3_eval.csv          one row per evaluation, written as it goes
    out/2c_m3_compare.csv       one row per (policy, validation layout)
    out/2c_m3_paired.csv        per-layout RL - rule differences
    out/2c_m3_learning.png      validation loss against update
    out/models/c_actor_m3_seed<N>.pt
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.evaluate_1b_large import paired_t_test, sign_test
from scripts.train_c_ppo import (
    CAUSES,
    LOAD_MS,
    ROUTING,
    _rollout_job,
    plot_learning,
    train_one_seed,
    write_csv,
)

# --- what M3 changes ------------------------------------------------------
#: Every episode draws one of these, independently, within an update.
TRAINING_LAYOUTS: Tuple[int, int] = (101, 300)

#: Thirty layouts the training never draws.
VALIDATION_LAYOUTS: Tuple[int, ...] = tuple(range(301, 331))

#: One run seed per validation layout, the same at every check and the same
#: for the rules, so every policy meets the identical thirty worlds.
VALIDATION_RUN_SEED_BASE: int = 9100

EPISODES_PER_UPDATE_M3: int = 40
MAX_UPDATES_M3: int = 600
MIN_IMPROVEMENT_M3: float = 0.2     # percentage points of total loss
PATIENCE_M3: int = 8

TRAINING_SEEDS: Tuple[int, ...] = (0, 1, 2)

NOTE = (f"phase 2c M3, MAPPO on layouts {TRAINING_LAYOUTS[0]}-"
        f"{TRAINING_LAYOUTS[1]}, validation {VALIDATION_LAYOUTS[0]}-"
        f"{VALIDATION_LAYOUTS[-1]}, static mission, hello 1 s, "
        f"{ROUTING} routing")

LOG_COLUMNS = ("seed", "update", "train_loss", "mean_reward", "policy_loss",
               "value_loss", "anchor_loss", "anchor_weight", "std",
               "decisions", "seconds")
EVAL_COLUMNS = ("seed", "update", "eval_loss", "saved")
COMPARE_COLUMNS = ("policy", "placement_seed", "run_seed", "total_loss",
                   *CAUSES, "c_distance_flown_mean", "c_gs_distance_mean",
                   "mean_reward", "views_agree")


def validation_run_seed(layout: int) -> int:
    """Return the fixed run seed for a validation layout.

    Args:
        layout: A validation placement seed.

    Returns:
        Its run seed, the same at every evaluation and for every policy.
    """
    return VALIDATION_RUN_SEED_BASE + (layout - VALIDATION_LAYOUTS[0])


def sample_layouts(rng: Any, episodes: int) -> List[int]:
    """Draw one training layout per episode, independently.

    Args:
        rng:      The seed's generator, drawn from AFTER the run seeds.
        episodes: How many episodes this update runs.

    Returns:
        One placement seed per episode.
    """
    low, high = TRAINING_LAYOUTS
    return rng.integers(low, high + 1, size=episodes).tolist()


def validation_jobs(state: Dict[str, Any]) -> List[Tuple]:
    """Return one greedy rollout per validation layout.

    Args:
        state: The frozen actor weights.

    Returns:
        Thirty jobs, one per validation layout.
    """
    return [(layout, validation_run_seed(layout), state, "rl", True)
            for layout in VALIDATION_LAYOUTS]


def saver(out_dir: str) -> Any:
    """Return a callback that keeps the CSVs current while training runs.

    M2 wrote its CSVs only after all three seeds finished, so an interrupted
    run lost every per-update row it had earned. M3 is far longer, so the
    files are rewritten after each update instead.

    Args:
        out_dir: Where the CSVs go.

    Returns:
        A function taking (log rows, evaluation rows).
    """
    log_path = os.path.join(out_dir, "2c_m3_log.csv")
    eval_path = os.path.join(out_dir, "2c_m3_eval.csv")
    history: Dict[str, List[Dict[str, Any]]] = {"log": [], "eval": []}

    def keep(log: Sequence[Dict[str, Any]],
             evaluations: Sequence[Dict[str, Any]]) -> None:
        """Rewrite both CSVs with everything finished so far.

        Args:
            log:         This seed's update rows.
            evaluations: This seed's evaluation rows.
        """
        write_csv(log_path, [*history["log"], *log], LOG_COLUMNS, NOTE)
        write_csv(eval_path, [*history["eval"], *evaluations],
                  EVAL_COLUMNS, NOTE)

    def finish(log: Sequence[Dict[str, Any]],
               evaluations: Sequence[Dict[str, Any]]) -> None:
        """Move one finished seed's rows into the history.

        Args:
            log:         That seed's update rows.
            evaluations: That seed's evaluation rows.
        """
        history["log"].extend(log)
        history["eval"].extend(evaluations)

    keep.finish = finish                                    # type: ignore
    return keep


def read_evaluations(path: str) -> List[Dict[str, Any]]:
    """Read an evaluation history back from the CSV training wrote.

    Args:
        path: out/2c_m3_eval.csv.

    Returns:
        One row per evaluation, typed as training held them.

    Raises:
        SystemExit: If the file is missing.
    """
    if not os.path.exists(path):
        print(f"STOP: no evaluation history at {path}")
        raise SystemExit(1)
    with open(path) as handle:
        rows = csv.DictReader(line for line in handle
                              if not line.startswith("#"))
        return [{"seed": int(row["seed"]), "update": int(row["update"]),
                 "eval_loss": float(row["eval_loss"]),
                 "saved": row["saved"] == "True"} for row in rows]


def already_trained(out_dir: str,
                    seeds: Sequence[int]) -> Tuple[Dict[int, str],
                                                   List[Dict[str, Any]]]:
    """Return the models and evaluation history a finished run left behind.

    This is what --compare-only stands on: the actors are read from disk and
    never retrained, so the comparison can be re-run without spending the
    training again.

    Args:
        out_dir: Where the models and the evaluation CSV live.
        seeds:   The training seeds to compare.

    Returns:
        Training seed -> model path, and the evaluation rows.

    Raises:
        SystemExit: If a model is missing.
    """
    models: Dict[int, str] = {}
    for seed in seeds:
        path = os.path.join(out_dir, "models", f"c_actor_m3_seed{seed}.pt")
        if not os.path.exists(path):
            print(f"STOP: no trained model at {path}")
            raise SystemExit(1)
        models[seed] = path
    evaluations = read_evaluations(os.path.join(out_dir, "2c_m3_eval.csv"))
    kept = [row for row in evaluations if row["saved"]]
    print(f"  re-using {len(models)} trained actors and "
          f"{len(evaluations)} evaluations ({len(kept)} saved); "
          f"nothing is retrained")
    return models, [row for row in evaluations if row["seed"] in models]


def compare_on_validation(models: Dict[int, str], pool: Any,
                          out_dir: str) -> List[Dict[str, Any]]:
    """Run every policy on all thirty validation layouts.

    Args:
        models:  Training seed -> path of the kept actor.
        pool:    A process pool, or None.
        out_dir: Where the clone lives.

    Returns:
        One row per (policy, validation layout).
    """
    import torch

    clone = torch.load(os.path.join(out_dir, "models", "c_actor_bc.pt"),
                       weights_only=True)
    actors = {seed: torch.load(path, weights_only=True)
              for seed, path in sorted(models.items())
              if os.path.exists(path)}

    jobs: List[Tuple] = []
    labels: List[str] = []
    for layout in VALIDATION_LAYOUTS:
        run_seed = validation_run_seed(layout)
        jobs.append((layout, run_seed, None, "hover", True))
        labels.append("hover")
        jobs.append((layout, run_seed, None, "midpoint", True))
        labels.append("midpoint")
        jobs.append((layout, run_seed, clone, "clone", True))
        labels.append("clone")
        for seed, state in actors.items():
            jobs.append((layout, run_seed, state, "rl", True))
            labels.append(f"rl seed {seed}")

    print(f"\nComparing {len(jobs)} runs on validation "
          f"{VALIDATION_LAYOUTS[0]}-{VALIDATION_LAYOUTS[-1]} ...", flush=True)
    results = (list(pool.map(_rollout_job, jobs)) if pool is not None
               else [_rollout_job(job) for job in jobs])

    rows = []
    for (layout, run_seed, _, _, _), label, result in zip(jobs, labels,
                                                          results):
        rows.append({"policy": label, "placement_seed": layout,
                     "run_seed": run_seed, **result})
    disagreed = [row for row in rows if not row["views_agree"]]
    print(f"  views agreed in {len(rows) - len(disagreed)} of {len(rows)} runs")
    if disagreed:
        raise RuntimeError("the ACK views disagreed with the simulator")
    return rows


def by_layout(rows: Sequence[Dict[str, Any]],
              policy: str) -> Dict[int, float]:
    """Return one policy's total loss for each validation layout.

    Args:
        rows:   Every comparison row.
        policy: Which policy to pull out.

    Returns:
        Placement seed -> total loss, as a fraction.
    """
    return {int(row["placement_seed"]): float(row["total_loss"])
            for row in rows if row["policy"] == policy}


def paired_rows(rows: Sequence[Dict[str, Any]],
                policies: Sequence[str]) -> List[Dict[str, Any]]:
    """Return the per-layout differences of each actor against each rule.

    Args:
        rows:     Every comparison row.
        policies: The RL policy labels.

    Returns:
        One row per (actor, rule, layout), differences in points.
    """
    out: List[Dict[str, Any]] = []
    for rule in ("midpoint", "hover"):
        reference = by_layout(rows, rule)
        for policy in policies:
            mine = by_layout(rows, policy)
            for layout in sorted(reference):
                out.append({
                    "policy": policy,
                    "against": rule,
                    "placement_seed": layout,
                    "rl_loss": mine[layout] * 100,
                    "rule_loss": reference[layout] * 100,
                    "difference": (mine[layout] - reference[layout]) * 100,
                })
    return out


def print_comparison(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Print the validation table and return the policy labels in order.

    Args:
        rows: Every comparison row.

    Returns:
        The policy labels, rules first.
    """
    policies: List[str] = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    width = 128
    print(f"\n  Validation layouts {VALIDATION_LAYOUTS[0]}-"
          f"{VALIDATION_LAYOUTS[-1]} — % of packets created")
    print("  " + "-" * width)
    header = f"{'policy':>14}{'total':>9}{'std':>8}"
    header += "".join(f"{cause[:9]:>11}" for cause in CAUSES)
    header += f"{'flown':>9}{'from GS':>9}"
    print("  " + header)
    print("  " + "-" * width)
    for policy in policies:
        mine = [r for r in rows if r["policy"] == policy]
        totals = [r["total_loss"] * 100 for r in mine]
        line = f"{policy:>14}{statistics.fmean(totals):>8.2f}%"
        line += f"{statistics.stdev(totals) if len(mine) > 1 else 0.0:>8.2f}"
        for cause in CAUSES:
            line += f"{statistics.fmean(r[cause] for r in mine) * 100:>10.2f}%"
        line += f"{statistics.fmean(r['c_distance_flown_mean'] for r in mine):>9.0f}"
        line += f"{statistics.fmean(r['c_gs_distance_mean'] for r in mine):>9.0f}"
        print("  " + line)
    print("  " + "-" * width)
    return policies


def print_paired(paired: Sequence[Dict[str, Any]],
                 policies: Sequence[str]) -> None:
    """Print each actor's paired difference against each rule.

    Args:
        paired:   Every paired row.
        policies: The RL policy labels.
    """
    print(f"\n  Paired over {len(VALIDATION_LAYOUTS)} validation layouts — "
          f"points of total loss, negative means the actor loses fewer")
    print("  " + "-" * 92)
    print("  " + f"{'actor':>14}{'against':>10}{'mean diff':>12}"
                 f"{'wins':>8}{'sign p':>12}{'t-test p':>12}")
    print("  " + "-" * 92)
    for rule in ("hover", "midpoint"):
        for policy in policies:
            diffs = [row["difference"] for row in paired
                     if row["policy"] == policy and row["against"] == rule]
            if not diffs:
                continue
            wins, used, p = sign_test(diffs)
            line = f"{policy:>14}{rule:>10}{statistics.fmean(diffs):>+11.2f} "
            line += f"{wins:>4}/{used:<3}{p:>11.2e}"
            line += f"{paired_t_test(diffs):>12.2e}"
            print("  " + line)
    print("  " + "-" * 92)


def main() -> None:
    """Train on all training layouts and compare on the validation set."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M3: MAPPO over the training layouts."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(TRAINING_SEEDS))
    parser.add_argument("--updates", type=int, default=MAX_UPDATES_M3)
    parser.add_argument("--episodes", type=int,
                        default=EPISODES_PER_UPDATE_M3)
    parser.add_argument("--compare-only", action="store_true",
                        help="skip training and re-run the validation "
                             "comparison from the models already saved")
    args = parser.parse_args()

    clone = os.path.join(args.out_dir, "models", "c_actor_bc.pt")
    if not os.path.exists(clone):
        print(f"STOP: no clone at {clone}; run scripts/clone_midpoint.py")
        raise SystemExit(1)

    # Spawn, not fork: the parent holds torch's thread pool, and forked
    # workers that touch torch deadlock on it.
    context = multiprocessing.get_context("spawn")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
            if args.workers > 1 else None)
    keep = saver(args.out_dir)
    started = time.perf_counter()
    try:
        evaluations: List[Dict[str, Any]] = []
        models: Dict[int, str] = {}
        if args.compare_only:
            models, evaluations = already_trained(args.out_dir, args.seeds)
        for seed in (() if args.compare_only else args.seeds):
            print(f"\n=== training seed {seed} ===", flush=True)
            result = train_one_seed(
                seed, pool, args.out_dir, args.updates, args.episodes,
                layouts=sample_layouts,
                eval_jobs=validation_jobs,
                min_improvement=MIN_IMPROVEMENT_M3,
                patience=PATIENCE_M3,
                model_name=f"c_actor_m3_seed{seed}.pt",
                progress=keep,
            )
            keep.finish(result["log"], result["evaluations"])
            evaluations.extend(result["evaluations"])
            models[seed] = result["model"]
            print(f"  seed {seed}: kept {result['kept']:.2f}% at update "
                  f"{result['best_update']}  (lowest seen "
                  f"{result['best']:.2f}%)", flush=True)
            print(f"  elapsed {(time.perf_counter() - started) / 3600:.2f} h",
                  flush=True)

        rows = compare_on_validation(models, pool, args.out_dir)
    finally:
        if pool is not None:
            pool.shutdown()

    policies = print_comparison(rows)
    actors = [p for p in policies if p.startswith("rl seed")]
    paired = paired_rows(rows, actors)
    print_paired(paired, actors)

    print("\n  kept versions")
    for seed in sorted(models):
        kept = [r for r in evaluations if r["seed"] == seed and r["saved"]]
        where = kept[-1]["update"] if kept else "never saved"
        print(f"    training seed {seed}: update {where}")

    baselines = {
        label: statistics.fmean(
            [r["total_loss"] for r in rows if r["policy"] == label]
        ) * 100
        for label in ("hover", "midpoint", "clone")
    }
    plot_learning(evaluations, baselines,
                  os.path.join(args.out_dir, "2c_m3_learning.png"),
                  title=(f"Phase 2c M3 — trained on layouts "
                         f"{TRAINING_LAYOUTS[0]}-{TRAINING_LAYOUTS[1]}, "
                         f"scored on {VALIDATION_LAYOUTS[0]}-"
                         f"{VALIDATION_LAYOUTS[-1]}"))

    write_csv(os.path.join(args.out_dir, "2c_m3_compare.csv"), rows,
              COMPARE_COLUMNS, NOTE)
    write_csv(os.path.join(args.out_dir, "2c_m3_paired.csv"), paired,
              ("policy", "against", "placement_seed", "rl_loss", "rule_loss",
               "difference"), NOTE)
    print(f"\n  total {(time.perf_counter() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
