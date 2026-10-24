"""
clone_midpoint.py — Phase 2c M1: teach the actor to copy the midpoint rule.

Three steps, in one run:

1. COLLECT. Run the Phase-2a midpoint rule through the Phase-2c interface on
   the training layouts, recording every C-drone decision: the observation it
   saw and the waypoint it chose. The interface is the same one the network
   will use, so the two see exactly the same thing.
2. CLONE. Fit the actor to those waypoints by least squares, keeping a tenth
   of the decisions back to measure the copy.
3. CHECK. Run hover, midpoint and the trained network on the validation
   layouts and report what each of them loses.

The world here is the one M0-M3 use: the M-drones hold their Phase-1
positions while the C-drones fly, hellos are on at 1 s, and routing is
greedy. That isolates what moving the relays is worth from what the mission's
own movement costs, which comes back in M4.

Writes:
    out/2c_clone_log.csv      one row per training epoch
    out/2c_validation.csv     one row per validation run
    out/models/c_actor_bc.pt  the trained network
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

from fanet_sim import config
from fanet_sim.envs import c_interface
from fanet_sim.envs.c_scripted import HoverAgent, MidpointAgent
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.stamp import csv_comment

TRAIN_SEEDS: Tuple[int, ...] = tuple(range(101, 301))
VALIDATION_SEEDS: Tuple[int, ...] = tuple(range(301, 331))
LOAD_MS: int = 200
ROUTING: str = "greedy"

CAUSES: Tuple[str, ...] = (
    "channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit",
    "link_broken", "stale_link",
)

VALIDATION_COLUMNS: Tuple[str, ...] = (
    "policy", "placement_seed", "total_loss", *CAUSES,
    "mean_reward_per_second", "c_distance_flown_mean", "views_agree",
)


def _world(c_rule: str):
    """Set the M0-M3 world and return the settings to put back.

    Args:
        c_rule: What the C-drones do.

    Returns:
        The previous values, for restoring.
    """
    previous = (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
                config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
                config.SHARE_TARGETS, config.AVOID)
    config.C_RULE = c_rule
    config.STATIC_MODE = True          # the MISSION is static; C-drones fly
    config.IDEAL_HELLO = False
    config.HELLO_INTERVAL = 1.0
    config.SHARE_TARGETS = True
    config.AVOID = True
    config.PACKET_INTERVAL_STEPS = int(
        round(LOAD_MS / 1000.0 / config.TIMESTEP)
    )
    return previous


def _restore(previous) -> None:
    """Put the settings back.

    Args:
        previous: What :func:`_world` returned.
    """
    (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
     config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
     config.SHARE_TARGETS, config.AVOID) = previous


def run_episode(seed: int, policy: str,
                state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run one layout under one policy and return what happened.

    Args:
        seed:   Placement seed, used as the run seed too.
        policy: "hover", "midpoint" or "network". All three run through the
                Phase-2c interface.
        state:  For "network", the actor's weights.

    Returns:
        The metrics, the decision log and the mean reward per second.
    """
    # Every policy goes through the interface, hover included. Native
    # hovering would be the same run — tests/test_2c_interface.py checks
    # that — but it logs no decisions, so it would report no reward, and a
    # blank is easily misread as a zero. Running it through the interface
    # gives it the reward it actually earns for sitting still.
    previous = _world("rl")
    try:
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=seed, run_seed=seed)
        env.reset()

        if policy == "hover":
            env.c_agent = HoverAgent(env)
        elif policy == "midpoint":
            env.c_agent = MidpointAgent(env)
        elif policy == "network":
            from agents.c_actor import CActor, WaypointAgent
            import torch

            # One thread per worker: twenty workers each grabbing every core
            # for a 256-wide forward pass is slower than one, not faster.
            torch.set_num_threads(1)
            actor = CActor()
            actor.load_state_dict(state)
            env.c_agent = WaypointAgent(actor)

        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        rewards = [row["reward"] for row in env.c_decisions
                   if row["reward"] is not None]
        return {
            "metrics": metrics,
            "decisions": env.c_decisions,
            "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        }
    finally:
        _restore(previous)


def _collect_job(seed: int) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """Worker: run midpoint on one layout and return its decisions."""
    result = run_episode(seed, "midpoint")
    observations = [row["observation"] for row in result["decisions"]]
    actions = np.array([row["action"] for row in result["decisions"]],
                       dtype=np.float64)
    return observations, actions


def collect(seeds: Sequence[int], workers: int
            ) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """Run midpoint over *seeds* and gather every decision.

    Args:
        seeds:   Training layouts.
        workers: Parallel workers; 1 runs sequentially.

    Returns:
        The observations and the waypoints chosen, aligned.
    """
    print(f"Collecting midpoint decisions on {len(seeds)} training layouts "
          f"at {LOAD_MS} ms ...")
    started = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_collect_job, seeds))
    else:
        parts = [_collect_job(seed) for seed in seeds]

    observations: List[Dict[str, np.ndarray]] = []
    actions: List[np.ndarray] = []
    for seen, chosen in parts:
        observations.extend(seen)
        if len(chosen):
            actions.append(chosen)
    stacked = np.concatenate(actions) if actions else np.zeros((0, 2))
    print(f"  {len(observations)} decisions in "
          f"{time.perf_counter() - started:.1f}s")
    return observations, stacked


def _validate_job(args: Tuple[int, str, Optional[Dict[str, Any]]]
                  ) -> Dict[str, Any]:
    """Worker: one validation run."""
    seed, policy, state = args
    result = run_episode(seed, policy, state)
    truth = result["metrics"]["ground_truth"]
    by_reason = truth["loss_rate_by_reason"]
    return {
        "policy": policy,
        "placement_seed": seed,
        "total_loss": truth["total_loss"] or 0.0,
        **{cause: by_reason[cause] for cause in CAUSES},
        "mean_reward_per_second": result["mean_reward"],
        "c_distance_flown_mean": truth["c_distance_flown_mean"],
        "views_agree": not result["metrics"]["discrepancies"],
        "discrepancies": result["metrics"]["discrepancies"],
    }


def validate(seeds: Sequence[int], state: Dict[str, Any],
             workers: int) -> List[Dict[str, Any]]:
    """Run hover, midpoint and the network over the validation layouts.

    Args:
        seeds:   Validation layouts.
        state:   The trained actor's weights.
        workers: Parallel workers.

    Returns:
        One row per run.

    Raises:
        SystemExit: If any run's two views of the traffic disagree.
    """
    jobs: List[Tuple[int, str, Optional[Dict[str, Any]]]] = []
    for policy in ("hover", "midpoint", "network"):
        for seed in seeds:
            jobs.append((seed, policy, state if policy == "network" else None))

    print(f"\nValidating on {len(seeds)} layouts x 3 policies "
          f"({len(jobs)} runs) ...")
    if workers > 1:
        # SPAWN, not fork. By this point the parent has trained the actor,
        # so torch has started its own thread pool; forking a process that
        # then touches torch deadlocks on it — the workers sit at a few
        # percent CPU and never finish. Spawned workers start clean.
        # Collection above is still forked: no torch has been imported yet.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=context) as pool:
            rows = list(pool.map(_validate_job, jobs))
    else:
        rows = [_validate_job(job) for job in jobs]

    bad = [row for row in rows if not row["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — {row['policy']} "
                  f"seed {row['placement_seed']}: {row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def print_validation(rows: Sequence[Dict[str, Any]]) -> None:
    """Print the validation table, one line per policy."""
    width = 116
    print(f"\n  Validation layouts {VALIDATION_SEEDS[0]}-"
          f"{VALIDATION_SEEDS[-1]}, static mission, {LOAD_MS} ms, "
          f"{ROUTING} routing — mean % of packets created")
    print("  " + "-" * width)
    print(f"  {'policy':>10}{'total':>9}{'std':>7}"
          + "".join(f"{c[:9]:>11}" for c in CAUSES)
          + f"{'reward/s':>10}{'flown':>8}")
    print("  " + "-" * width)
    for policy in ("hover", "midpoint", "network"):
        chosen = [r for r in rows if r["policy"] == policy]
        totals = [r["total_loss"] * 100 for r in chosen]
        print(f"  {policy:>10}{statistics.fmean(totals):8.2f}%"
              f"{statistics.stdev(totals):7.2f}"
              + "".join(
                  f"{statistics.fmean([r[c] for r in chosen]) * 100:10.2f}%"
                  for c in CAUSES
              )
              + f"{statistics.fmean([r['mean_reward_per_second'] for r in chosen]):10.4f}"
              f"{statistics.fmean([r['c_distance_flown_mean'] for r in chosen]):8.0f}")
    print("  " + "-" * width)


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str], note: str) -> None:
    """Write *rows* to *path* under a settings stamp."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=[LOAD_MS], extra=note))
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    print(f"  wrote {path}")


def main() -> None:
    """Collect midpoint's decisions, clone them, and check the result."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M1: clone the midpoint rule into the actor."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the network and the held-out split.")
    parser.add_argument("--train-layouts", type=int,
                        default=len(TRAIN_SEEDS),
                        help="How many of the training layouts to use.")
    args = parser.parse_args()

    import torch
    from agents.c_actor import train_clone

    seeds = TRAIN_SEEDS[:args.train_layouts]
    observations, actions = collect(seeds, args.workers)
    if not observations:
        print("STOP: midpoint made no decisions at all.")
        raise SystemExit(1)

    print(f"\nCloning: {len(observations)} decisions, "
          f"{args.epochs} epochs ...")
    actor, record = train_clone(observations, actions, epochs=args.epochs,
                                seed=args.seed, log=print)
    final = record["final"]
    print(f"\n  held out {record['holdout_decisions']} decisions of "
          f"{record['decisions']}")
    print(f"  copying error on the held-out tenth: "
          f"{final['holdout_metres']:.2f} m "
          f"(MSE {final['holdout_mse']:.6f} on the normalised waypoint)")

    models = os.path.join(args.out_dir, "models")
    os.makedirs(models, exist_ok=True)
    path = os.path.join(models, "c_actor_bc.pt")
    torch.save(actor.state_dict(), path)
    print(f"  wrote {path}")

    rows = validate(VALIDATION_SEEDS, actor.state_dict(), args.workers)
    print_validation(rows)

    note = (f"phase 2c M1, clone of midpoint, static mission, hello 1 s, "
            f"{ROUTING} routing")
    write_csv(os.path.join(args.out_dir, "2c_clone_log.csv"),
              record["history"],
              ("epoch", "train_mse", "holdout_mse", "holdout_metres"), note)
    write_csv(os.path.join(args.out_dir, "2c_validation.csv"), rows,
              VALIDATION_COLUMNS, note)


if __name__ == "__main__":
    main()
