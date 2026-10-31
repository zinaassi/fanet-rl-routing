"""
train_c_ppo_m4.py — Phase 2c M4: the C-drones in the moving world.

M3 learned a static mission: the M-drones sat where Phase 1 put them, and a
C-drone that found a good place could stay there. Here the mission flies.
Each M-drone waits out a random delay, crosses the arena in a straight line
and hovers at the border, so the network the C-drones are serving keeps
changing shape for most of the run.

Four runs, in the order the checkpoint asked for:

    a, b, d   start from an M3 actor with a FRESH critic and no
              stay-near-midpoint term: that actor already knows the task, and
              pulling it back towards midpoint would only undo what M3 bought.
    c         the comparison run, starting from the clone of midpoint with
              M3's recipe. It is the only one that answers "did training on
              the static world first help at all?".

All four freeze the actor for the first ten updates while the fresh critic
catches up, and none may stop before update 300: the moving world is harder,
and an actor that looks flat at update 100 may still be climbing out of it.

Writes:
    out/2c_m4_log.csv           one row per update, written as it goes
    out/2c_m4_eval.csv          one row per evaluation, written as it goes
    out/2c_m4_compare.csv       one row per (policy, validation layout)
    out/2c_m4_paired.csv        per-layout RL - rule differences
    out/2c_m4_learning.png      validation loss against update
    out/models/2c/c_actor_m4_<run>.pt   and c_critic_m4_<run>.pt
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.evaluate_1b_large import paired_t_test, sign_test
from scripts.metrics_1a import time_window_labels
from scripts.train_c_ppo import (
    CAUSES,
    ROUTING,
    _rollout_job,
    plot_learning,
    train_one_seed,
    write_csv,
)
from scripts.train_c_ppo_m3 import (
    TRAINING_LAYOUTS,
    VALIDATION_LAYOUTS,
    by_layout,
    sample_layouts,
    validation_run_seed,
)

#: Where M4's own models go, under out/. Whitelisted in .gitignore, unlike
#: the rest of out/models/.
MODEL_DIR: str = os.path.join("models", "2c")

#: The same folder as train_one_seed names it. That function already joins
#: out_dir and "models", so the name it is given must start BELOW those.
MODEL_SUBDIR: str = "2c"

#: 40 episodes an update, as M3 ran.
EPISODES_PER_UPDATE_M4: int = 40

#: The moving world is harder, so the cap is higher than M3's and the run is
#: not allowed to give up early.
MAX_UPDATES_M4: int = 600
MIN_UPDATES_M4: int = 300
MIN_IMPROVEMENT_M4: float = 0.2
PATIENCE_M4: int = 8

#: The four runs, in the order the checkpoint set. ``start`` names the M3
#: training seed to continue from, or None for the clone of midpoint.
RUNS: Tuple[Dict[str, Any], ...] = (
    {"name": "a", "start": 2, "anchor": False, "seed": 2},
    {"name": "b", "start": 1, "anchor": False, "seed": 1},
    {"name": "c", "start": None, "anchor": True, "seed": 3},
    {"name": "d", "start": 0, "anchor": False, "seed": 0},
)

#: The M3 actors the main runs continue from.
M3_MODEL = "c_actor_m3_seed{seed}.pt"

NOTE = (
    f"phase 2c M4, MAPPO in the MOVING world, training "
    f"{TRAINING_LAYOUTS[0]}-{TRAINING_LAYOUTS[1]}, validation "
    f"{VALIDATION_LAYOUTS[0]}-{VALIDATION_LAYOUTS[-1]}, moving mission, "
    f"hello 1 s, {ROUTING} routing"
)

#: The per-slice loss columns, one per ten seconds of the run.
WINDOW_COLUMNS: Tuple[str, ...] = tuple(f"loss_{label}"
                                        for label in time_window_labels())

LOG_COLUMNS = ("run", "seed", "update", "train_loss", "mean_reward",
               "policy_loss", "value_loss", "anchor_loss", "anchor_weight",
               "std", "seconds")
EVAL_COLUMNS = ("run", "seed", "update", "eval_loss", "saved")
COMPARE_COLUMNS = ("policy", "placement_seed", "run_seed", "total_loss",
                   *CAUSES, "pct_packets_connected", "loss_while_connected",
                   *WINDOW_COLUMNS, "c_distance_flown_mean",
                   "c_gs_distance_mean", "mean_reward", "views_agree")


def run_by_name(name: str) -> Dict[str, Any]:
    """Return one run's settings.

    Args:
        name: "a", "b", "c" or "d".

    Returns:
        That run's entry in RUNS.

    Raises:
        SystemExit: If the name is not one of the four.
    """
    for run in RUNS:
        if run["name"] == name:
            return run
    print(f"STOP: no run named {name!r}; the runs are "
          f"{[r['name'] for r in RUNS]}")
    raise SystemExit(1)


def validation_jobs(state: Dict[str, Any]) -> List[Tuple]:
    """Return one greedy rollout per validation layout, in the moving world.

    Args:
        state: The frozen actor weights.

    Returns:
        Thirty jobs, each asking for the moving world.
    """
    return [(layout, validation_run_seed(layout), state, "rl", True, True)
            for layout in VALIDATION_LAYOUTS]


def start_state(run: Dict[str, Any], out_dir: str) -> Optional[Dict[str, Any]]:
    """Return the weights a run starts from.

    Args:
        run:     One entry of RUNS.
        out_dir: Where the models live.

    Returns:
        The M3 actor to continue from, or None to start from the clone.

    Raises:
        SystemExit: If the M3 actor is missing.
    """
    if run["start"] is None:
        return None

    import torch

    path = os.path.join(out_dir, "models",
                        M3_MODEL.format(seed=run["start"]))
    if not os.path.exists(path):
        print(f"STOP: run {run['name']} continues M3 seed {run['start']}, "
              f"but there is no actor at {path}")
        raise SystemExit(1)
    print(f"  run {run['name']} starts from M3 seed {run['start']} "
          f"({path}), with a fresh critic and no anchor term")
    return torch.load(path, weights_only=True)


def saver(out_dir: str) -> Any:
    """Return a callback that keeps the CSVs current while training runs.

    M4 is the longest run in the project, so the per-update rows are written
    as they are earned rather than at the end.

    Args:
        out_dir: Where the CSVs go.

    Returns:
        A function taking (log rows, evaluation rows), with a .finish and a
        .run attribute.
    """
    log_path = os.path.join(out_dir, "2c_m4_log.csv")
    eval_path = os.path.join(out_dir, "2c_m4_eval.csv")
    history: Dict[str, List[Dict[str, Any]]] = {"log": [], "eval": []}
    current = {"run": ""}

    def tagged(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return *rows* with the run they belong to."""
        return [{**row, "run": current["run"]} for row in rows]

    def keep(log: Sequence[Dict[str, Any]],
             evaluations: Sequence[Dict[str, Any]]) -> None:
        """Rewrite both CSVs with everything finished so far."""
        write_csv(log_path, [*history["log"], *tagged(log)],
                  LOG_COLUMNS, NOTE)
        write_csv(eval_path, [*history["eval"], *tagged(evaluations)],
                  EVAL_COLUMNS, NOTE)

    def finish(log: Sequence[Dict[str, Any]],
               evaluations: Sequence[Dict[str, Any]]) -> None:
        """Move one finished run's rows into the history."""
        history["log"].extend(tagged(log))
        history["eval"].extend(tagged(evaluations))

    def begin(name: str, out_dir_: str) -> None:
        """Start a run, re-reading what earlier runs already wrote."""
        current["run"] = name
        history["log"] = [row for row in read_rows(log_path)
                          if row["run"] != name]
        history["eval"] = [row for row in read_rows(eval_path)
                           if row["run"] != name]

    keep.finish = finish                                    # type: ignore
    keep.begin = begin                                      # type: ignore
    return keep


def read_rows(path: str) -> List[Dict[str, Any]]:
    """Read a CSV back, typed as training held it.

    Each run is launched separately, so a later run has to pick up the rows
    the earlier ones left in the file rather than start it again.

    Args:
        path: One of the M4 CSVs.

    Returns:
        Its rows, or an empty list if it does not exist yet.
    """
    if not os.path.exists(path):
        return []
    numbers = ("train_loss", "mean_reward", "policy_loss", "value_loss",
               "anchor_loss", "anchor_weight", "std", "seconds", "eval_loss")
    out: List[Dict[str, Any]] = []
    with open(path) as handle:
        for row in csv.DictReader(line for line in handle
                                  if not line.startswith("#")):
            typed: Dict[str, Any] = dict(row)
            typed["seed"] = int(row["seed"])
            typed["update"] = int(row["update"])
            if "saved" in row:
                typed["saved"] = row["saved"] == "True"
            for key in numbers:
                if key in row and row[key] != "":
                    typed[key] = float(row[key])
            out.append(typed)
    return out


def train_one_run(run: Dict[str, Any], pool: Any, out_dir: str,
                  max_updates: int, episodes: int,
                  keep: Any) -> Dict[str, Any]:
    """Train one of the four runs.

    Args:
        run:         One entry of RUNS.
        pool:        The process pool.
        out_dir:     Where models and CSVs go.
        max_updates: Cap on updates.
        episodes:    Episodes per update.
        keep:        The progress callback from saver().

    Returns:
        What train_one_seed returned.
    """
    os.makedirs(os.path.join(out_dir, MODEL_DIR), exist_ok=True)
    keep.begin(run["name"], out_dir)
    return train_one_seed(
        run["seed"], pool, out_dir, max_updates, episodes,
        layouts=sample_layouts,
        eval_jobs=validation_jobs,
        min_improvement=MIN_IMPROVEMENT_M4,
        patience=PATIENCE_M4,
        model_name=os.path.join(MODEL_SUBDIR, f"c_actor_m4_{run['name']}.pt"),
        critic_name=os.path.join(MODEL_SUBDIR,
                                 f"c_critic_m4_{run['name']}.pt"),
        progress=keep,
        moving=True,
        min_updates=MIN_UPDATES_M4,
        anchor=run["anchor"],
        start_state=start_state(run, out_dir),
    )


def comparison_jobs(out_dir: str,
                    actors: Dict[str, Any]) -> Tuple[List[Tuple], List[str]]:
    """Return one rollout per (policy, validation layout), moving world.

    Beside the rules and the clone, the three M3 actors run WITHOUT any
    further training: that is what says whether the static training carries
    over to the moving world on its own.

    Args:
        out_dir: Where the models live.
        actors:  Label -> weights for every M4 run that has a model.

    Returns:
        The jobs and their policy labels.
    """
    import torch

    states: Dict[str, Any] = {}
    clone = os.path.join(out_dir, "models", "c_actor_bc.pt")
    if os.path.exists(clone):
        states["clone"] = ("clone", torch.load(clone, weights_only=True))
    for seed in (0, 1, 2):
        path = os.path.join(out_dir, "models", M3_MODEL.format(seed=seed))
        if os.path.exists(path):
            states[f"m3 seed {seed}"] = ("rl", torch.load(path,
                                                          weights_only=True))
    for label, state in actors.items():
        states[label] = ("rl", state)

    jobs: List[Tuple] = []
    labels: List[str] = []
    for layout in VALIDATION_LAYOUTS:
        run_seed = validation_run_seed(layout)
        jobs.append((layout, run_seed, None, "hover", True, True))
        labels.append("hover")
        jobs.append((layout, run_seed, None, "midpoint", True, True))
        labels.append("midpoint")
        for label, (policy, state) in states.items():
            jobs.append((layout, run_seed, state, policy, True, True))
            labels.append(label)
    return jobs, labels


def m4_actors(out_dir: str, names: Sequence[str]) -> Dict[str, Any]:
    """Return the trained actor of each named run that has one.

    Args:
        out_dir: Where the models live.
        names:   Run names to look for.

    Returns:
        Label -> weights.
    """
    import torch

    out: Dict[str, Any] = {}
    for name in names:
        path = os.path.join(out_dir, MODEL_DIR, f"c_actor_m4_{name}.pt")
        if os.path.exists(path):
            out[f"m4 run {name}"] = torch.load(path, weights_only=True)
    return out


def compare(jobs: Sequence[Tuple], labels: Sequence[str],
            pool: Any) -> List[Dict[str, Any]]:
    """Run every comparison job and return its rows.

    Args:
        jobs:   The rollout jobs.
        labels: Their policy labels.
        pool:   The process pool.

    Returns:
        One row per (policy, layout).

    Raises:
        RuntimeError: If any run's ACK view disagreed with the simulator.
    """
    print(f"\nComparing {len(jobs)} runs on validation "
          f"{VALIDATION_LAYOUTS[0]}-{VALIDATION_LAYOUTS[-1]} "
          f"in the MOVING world ...", flush=True)
    results = (list(pool.map(_rollout_job, jobs)) if pool is not None
               else [_rollout_job(job) for job in jobs])
    rows = [{"policy": label, "placement_seed": job[0], "run_seed": job[1],
             **result}
            for job, label, result in zip(jobs, labels, results)]
    disagreed = [row for row in rows if not row["views_agree"]]
    print(f"  views agreed in {len(rows) - len(disagreed)} of {len(rows)} runs")
    if disagreed:
        raise RuntimeError("the ACK views disagreed with the simulator")
    return rows


def paired_rows(rows: Sequence[Dict[str, Any]],
                policies: Sequence[str]) -> List[Dict[str, Any]]:
    """Return the per-layout differences of each actor against each rule.

    Args:
        rows:     Every comparison row.
        policies: The actor labels.

    Returns:
        One row per (actor, rule, layout), differences in points.
    """
    out: List[Dict[str, Any]] = []
    for rule in ("midpoint", "hover"):
        reference = by_layout(rows, rule)
        if not reference:
            continue
        for policy in policies:
            mine = by_layout(rows, policy)
            for layout in sorted(reference):
                if layout not in mine:
                    continue
                out.append({
                    "policy": policy,
                    "against": rule,
                    "placement_seed": layout,
                    "rl_loss": mine[layout] * 100,
                    "rule_loss": reference[layout] * 100,
                    "difference": (mine[layout] - reference[layout]) * 100,
                })
    return out


def print_comparison(rows: Sequence[Dict[str, Any]],
                     where: str) -> List[str]:
    """Print the main table and return the policy labels in order.

    Args:
        rows:  Every comparison row.
        where: What to call the layout set in the heading.

    Returns:
        The policy labels, rules first.
    """
    policies: List[str] = []
    for row in rows:
        if row["policy"] not in policies:
            policies.append(row["policy"])

    width = 150
    print(f"\n  {where} — % of packets created, MOVING world")
    print("  " + "-" * width)
    header = f"{'policy':>14}{'total':>9}{'std':>8}"
    header += "".join(f"{cause[:9]:>11}" for cause in CAUSES)
    header += f"{'conn%':>9}{'loss|conn':>11}{'flown':>9}{'from GS':>9}"
    print("  " + header)
    print("  " + "-" * width)
    for policy in policies:
        mine = [r for r in rows if r["policy"] == policy]
        totals = [r["total_loss"] * 100 for r in mine]
        line = f"{policy:>14}{statistics.fmean(totals):>8.2f}%"
        line += f"{statistics.stdev(totals) if len(mine) > 1 else 0.0:>8.2f}"
        for cause in CAUSES:
            line += f"{statistics.fmean(r[cause] for r in mine) * 100:>10.2f}%"
        connected = statistics.fmean(r["pct_packets_connected"] for r in mine)
        while_conn = statistics.fmean(r["loss_while_connected"] for r in mine)
        flown = statistics.fmean(r["c_distance_flown_mean"] for r in mine)
        from_gs = statistics.fmean(r["c_gs_distance_mean"] for r in mine)
        line += f"{connected * 100:>8.2f}%{while_conn * 100:>10.2f}%"
        line += f"{flown:>9.0f}{from_gs:>9.0f}"
        print("  " + line)
    print("  " + "-" * width)
    return policies


def print_windows(rows: Sequence[Dict[str, Any]],
                  policies: Sequence[str]) -> None:
    """Print the loss in each ten-second slice of the run.

    Args:
        rows:     Every comparison row.
        policies: The policy labels, in order.
    """
    labels = time_window_labels()
    width = 14 + 9 * len(labels)
    print("\n  Loss per ten seconds, by the slice a packet was created in "
          "— % of packets created in that slice")
    print("  " + "-" * width)
    print("  " + f"{'policy':>14}" + "".join(f"{label:>9}"
                                             for label in labels))
    print("  " + "-" * width)
    for policy in policies:
        mine = [r for r in rows if r["policy"] == policy]
        line = f"{policy:>14}"
        for label in labels:
            values = [r[f"loss_{label}"] for r in mine
                      if r.get(f"loss_{label}") is not None]
            line += (f"{statistics.fmean(values) * 100:>8.2f}%"
                     if values else f"{'-':>9}")
        print("  " + line)
    print("  " + "-" * width)


def print_paired(paired: Sequence[Dict[str, Any]],
                 policies: Sequence[str], count: int) -> None:
    """Print each actor's paired difference against each rule.

    Args:
        paired:   Every paired row.
        policies: The actor labels.
        count:    How many layouts were paired.
    """
    print(f"\n  Paired over {count} layouts — points of total loss, "
          f"negative means the actor loses fewer")
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


def print_kept(out_dir: str) -> None:
    """Print the update each run's kept version was saved at.

    Args:
        out_dir: Where the evaluation CSV lives.
    """
    rows = read_rows(os.path.join(out_dir, "2c_m4_eval.csv"))
    if not rows:
        return
    print("\n  kept versions")
    for name in sorted({row["run"] for row in rows}):
        saved = [row for row in rows if row["run"] == name and row["saved"]]
        if saved:
            print(f"    run {name}: update {saved[-1]['update']}  "
                  f"({saved[-1]['eval_loss']:.2f}%)")
        else:
            print(f"    run {name}: never saved")


def draw_learning(out_dir: str, rows: Sequence[Dict[str, Any]]) -> None:
    """Draw validation loss against update, one line per run.

    plot_learning draws one line per "seed", so each run's rows are relabelled
    by its name before they go in.

    Args:
        out_dir: Where the figure goes.
        rows:    The comparison rows, for the baseline lines.
    """
    evaluations = read_rows(os.path.join(out_dir, "2c_m4_eval.csv"))
    if not evaluations:
        print("  no evaluations yet; skipping the learning curve")
        return

    names = sorted({row["run"] for row in evaluations})
    renumbered = [{**row, "seed": names.index(row["run"])}
                  for row in evaluations]
    baselines = {
        label: statistics.fmean(
            [r["total_loss"] for r in rows if r["policy"] == label]
        ) * 100
        for label in ("hover", "midpoint")
        if any(r["policy"] == label for r in rows)
    }
    plot_learning(renumbered, baselines,
                  os.path.join(out_dir, "2c_m4_learning.png"),
                  title=(f"Phase 2c M4 — moving world, trained on "
                         f"{TRAINING_LAYOUTS[0]}-{TRAINING_LAYOUTS[1]}, "
                         f"scored on {VALIDATION_LAYOUTS[0]}-"
                         f"{VALIDATION_LAYOUTS[-1]}"),
                  names=[f"run {name}" for name in names])


def main() -> None:
    """Train the requested runs, then compare on validation."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M4: MAPPO in the moving world."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--runs", type=str, nargs="+",
                        default=[run["name"] for run in RUNS],
                        help="which runs to train, e.g. --runs a")
    parser.add_argument("--updates", type=int, default=MAX_UPDATES_M4)
    parser.add_argument("--episodes", type=int,
                        default=EPISODES_PER_UPDATE_M4)
    parser.add_argument("--compare-only", action="store_true",
                        help="skip training and compare what is on disk")
    args = parser.parse_args()

    chosen = [run_by_name(name) for name in args.runs]

    context = multiprocessing.get_context("spawn")
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=context)
            if args.workers > 1 else None)
    keep = saver(args.out_dir)
    started = time.perf_counter()
    try:
        for run in (() if args.compare_only else chosen):
            print(f"\n=== run {run['name']} ===", flush=True)
            result = train_one_run(run, pool, args.out_dir, args.updates,
                                   args.episodes, keep)
            keep.finish(result["log"], result["evaluations"])
            print(f"  run {run['name']}: kept {result['kept']:.2f}% at "
                  f"update {result['best_update']}  (lowest seen "
                  f"{result['best']:.2f}%)", flush=True)
            print(f"  elapsed {(time.perf_counter() - started) / 3600:.2f} h",
                  flush=True)

        actors = m4_actors(args.out_dir, [run["name"] for run in RUNS])
        jobs, labels = comparison_jobs(args.out_dir, actors)
        rows = compare(jobs, labels, pool)
    finally:
        if pool is not None:
            pool.shutdown()

    policies = print_comparison(
        rows, f"Validation {VALIDATION_LAYOUTS[0]}-{VALIDATION_LAYOUTS[-1]}"
    )
    print_windows(rows, policies)
    actors_seen = [p for p in policies
                   if p.startswith(("m4 run", "m3 seed", "clone"))]
    paired = paired_rows(rows, actors_seen)
    print_paired(paired, actors_seen, len(VALIDATION_LAYOUTS))
    print_kept(args.out_dir)
    draw_learning(args.out_dir, rows)

    write_csv(os.path.join(args.out_dir, "2c_m4_compare.csv"), rows,
              COMPARE_COLUMNS, NOTE)
    write_csv(os.path.join(args.out_dir, "2c_m4_paired.csv"), paired,
              ("policy", "against", "placement_seed", "rl_loss", "rule_loss",
               "difference"), NOTE)
    print(f"\n  total {(time.perf_counter() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
