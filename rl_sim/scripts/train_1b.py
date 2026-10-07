"""
train_1b.py — Train the Phase-1b "rl" routing agent.

Protocol, all at the 200 ms working load:

    TRAINING    placement seeds 101-300, cycled in order (run_seed = seed)
    VALIDATION  seeds 301-310, never trained on
    TRAIN-CHECK seeds 101-110, for the overfitting check only
    TEST        seeds 1-20, held back for evaluate_1b.py

A training run is one layout for 1000 steps with learning on and exploration
at ``config.RL_EPSILON_TRAIN``. Every ``EVALUATE_EVERY`` runs the network is
frozen — no exploration, no gradient steps — and measured on the validation
and train-check layouts. The per-link delivery rates still update during
evaluation: those are observations, not learning.

The best validation model is kept under ``out/models/`` (gitignored). Training
stops after ``MAX_TRAINING_RUNS``, or early when validation has not improved
by ``MIN_IMPROVEMENT`` points over the last ``PATIENCE`` evaluations.

Evaluations run in parallel: the network is frozen and the layouts are
independent, so a worker pool gives the same numbers as a sequential pass —
each run is fully determined by its own seeds. ``--workers 1`` forces the
sequential path for checking that.

Outputs:
    out/1b_training_log.csv   one row per training run and per evaluation
    out/models/rl_seed<N>.pt  the best-validation network for each init seed

Usage:
    python scripts/train_1b.py --init-seeds 0
    python scripts/train_1b.py --init-seeds 0 1 2
    python scripts/train_1b.py --init-seeds 0 --runs 10     # smoke
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.stamp import csv_comment, settings_stamp

# --- the data split -------------------------------------------------------
TRAIN_SEEDS = tuple(range(101, 301))
VALIDATION_SEEDS = tuple(range(301, 311))
TRAIN_CHECK_SEEDS = tuple(range(101, 111))
TEST_SEEDS = tuple(range(1, 21))

LOAD_MS = 200
EVALUATE_EVERY = 10
MAX_TRAINING_RUNS = 300
PATIENCE = 5            # evaluations without improvement before stopping
MIN_IMPROVEMENT = 0.5   # percentage points of total loss

REFERENCE_RULES = ("greedy", "random", "rate")

LOG_COLUMNS = [
    "init_seed", "run_index", "kind", "routing",
    "train_total_loss", "train_bce",
    "validation_total_loss", "train_check_total_loss",
    "validation_channel", "validation_queue_full",
    "validation_no_route", "validation_ttl_hop",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train the Phase-1b RL router.")
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0],
                        help="Network initialisation seeds to train (default 0).")
    parser.add_argument("--runs", type=int, default=MAX_TRAINING_RUNS,
                        help=f"Training runs per init seed (default {MAX_TRAINING_RUNS}).")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1),
                        help="Parallel evaluation workers; 1 runs them sequentially.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the log and models go.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_once(
    routing: str,
    placement_seed: int,
    run_seed: int,
    load_ms: int = LOAD_MS,
    state_dict: Optional[Dict[str, Any]] = None,
    training: bool = False,
    init_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one layout and return its metrics.

    Args:
        routing:        "greedy", "random", "rate" or "rl".
        placement_seed: Layout seed.
        run_seed:       Channel/traffic seed.
        load_ms:        Traffic load in milliseconds.
        state_dict:     For "rl", network weights to use.
        training:       For "rl", whether to explore and learn.
        init_seed:      For "rl", seeds a fresh network when no weights given.

    Returns:
        ``total_loss``, the loss split by cause, ``views_agree``, and for a
        training run the mean BCE and the resulting weights.
    """
    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(round(load_ms / 1000.0 / config.TIMESTEP))
    try:
        router = None
        if routing in ("rate", "rl"):
            from agents import make_router

            router = make_router(routing, run_seed=run_seed,
                                 training=training, init_seed=init_seed)
            if state_dict is not None:
                router.net.load_state_dict(state_dict)

        env = FANETEnv(routing=routing, log_path=os.devnull,
                       placement_seed=placement_seed, run_seed=run_seed,
                       router=router)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        by_reason = truth["loss_rate_by_reason"]
        result: Dict[str, Any] = {
            "total_loss": truth["total_loss"] or 0.0,
            "channel": by_reason["channel"],
            "queue_full": by_reason["queue_full"],
            "no_route": by_reason["no_route"],
            "ttl_hop": by_reason["ttl"] + by_reason["hop_limit"],
            "views_agree": not metrics["discrepancies"],
            "discrepancies": metrics["discrepancies"],
        }
        if routing == "rl" and training:
            result["bce"] = router.mean_batch_loss()
            result["state_dict"] = {
                k: v.clone() for k, v in router.net.state_dict().items()
            }
        return result
    finally:
        config.PACKET_INTERVAL_STEPS = previous


def _eval_task(job: Tuple[str, int, int, int, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Worker entry point: run one frozen evaluation.

    Args:
        job: ``(routing, placement_seed, run_seed, load_ms, state_dict)``.

    Returns:
        The run's metrics.
    """
    routing, placement_seed, run_seed, load_ms, state_dict = job
    return run_once(routing, placement_seed, run_seed, load_ms,
                    state_dict=state_dict, training=False)


def evaluate(
    routing: str,
    seeds: Sequence[int],
    state_dict: Optional[Dict[str, Any]] = None,
    workers: int = 1,
    load_ms: int = LOAD_MS,
) -> Dict[str, float]:
    """Average one rule over a set of layouts, with the network frozen.

    Args:
        routing:    The rule to measure.
        seeds:      Layout seeds; run_seed equals placement_seed.
        state_dict: For "rl", the weights to evaluate.
        workers:    Parallel workers. 1 runs sequentially.
        load_ms:    Traffic load.

    Returns:
        Mean total loss and mean loss by cause, as fractions.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    jobs = [(routing, seed, seed, load_ms, state_dict) for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_eval_task, jobs))
    else:
        results = [_eval_task(job) for job in jobs]

    for seed, result in zip(seeds, results):
        if not result["views_agree"]:
            print(f"\nSTOP: views disagree — {routing}, seed {seed}")
            for problem in result["discrepancies"]:
                print(f"    ! {problem}")
            raise SystemExit(1)

    return {
        key: statistics.fmean(r[key] for r in results)
        for key in ("total_loss", "channel", "queue_full", "no_route", "ttl_hop")
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_seed(
    init_seed: int, runs: int, workers: int, out_dir: str
) -> List[Dict[str, Any]]:
    """Train one network from scratch and log its progress.

    Args:
        init_seed: Seeds the network's initialisation.
        runs:      Maximum training runs.
        workers:   Parallel evaluation workers.
        out_dir:   Where the best model is written.

    Returns:
        The log rows for this init seed.
    """
    from agents import make_router

    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    router = make_router("rl", run_seed=TRAIN_SEEDS[0],
                         training=True, init_seed=init_seed)
    state = {k: v.clone() for k, v in router.net.state_dict().items()}

    rows: List[Dict[str, Any]] = []
    best_loss: Optional[float] = None
    best_eval_index = 0
    evaluations = 0
    started = time.perf_counter()

    for run_index in range(1, runs + 1):
        seed = TRAIN_SEEDS[(run_index - 1) % len(TRAIN_SEEDS)]
        result = run_once("rl", seed, seed, LOAD_MS, state_dict=state,
                          training=True, init_seed=init_seed)
        state = result["state_dict"]

        rows.append({
            "init_seed": init_seed, "run_index": run_index, "kind": "train",
            "routing": "rl",
            "train_total_loss": result["total_loss"],
            "train_bce": result.get("bce"),
            "validation_total_loss": None, "train_check_total_loss": None,
            "validation_channel": None, "validation_queue_full": None,
            "validation_no_route": None, "validation_ttl_hop": None,
        })

        if run_index % EVALUATE_EVERY:
            continue

        validation = evaluate("rl", VALIDATION_SEEDS, state, workers)
        train_check = evaluate("rl", TRAIN_CHECK_SEEDS, state, workers)
        evaluations += 1

        rows.append({
            "init_seed": init_seed, "run_index": run_index, "kind": "evaluation",
            "routing": "rl",
            "train_total_loss": None, "train_bce": None,
            "validation_total_loss": validation["total_loss"],
            "train_check_total_loss": train_check["total_loss"],
            "validation_channel": validation["channel"],
            "validation_queue_full": validation["queue_full"],
            "validation_no_route": validation["no_route"],
            "validation_ttl_hop": validation["ttl_hop"],
        })

        marker = ""
        if best_loss is None or validation["total_loss"] < best_loss - MIN_IMPROVEMENT / 100.0:
            best_loss = validation["total_loss"]
            best_eval_index = evaluations
            import torch

            torch.save(state, os.path.join(models_dir, f"rl_seed{init_seed}.pt"))
            marker = "  <- best, saved"
        elif best_loss is None or validation["total_loss"] < best_loss:
            best_loss = validation["total_loss"]

        print(f"  run {run_index:>3}  validation {validation['total_loss']*100:6.2f}%  "
              f"train-check {train_check['total_loss']*100:6.2f}%  "
              f"bce {_fmt(rows[-2]['train_bce'])}  "
              f"{time.perf_counter()-started:6.1f}s{marker}", flush=True)

        if evaluations - best_eval_index >= PATIENCE:
            print(f"  early stop: validation has not improved by "
                  f"{MIN_IMPROVEMENT} points in {PATIENCE} evaluations")
            break

    return rows


def _fmt(value: Optional[float]) -> str:
    """Format an optional float for the progress line."""
    return "   n/a" if value is None else f"{value:6.4f}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Train every requested init seed and write the log."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Phase-1b training at {LOAD_MS} ms")
    print(f"  {settings_stamp(loads_ms=[LOAD_MS])}")
    print(f"  train {len(TRAIN_SEEDS)} layouts | validation {len(VALIDATION_SEEDS)}"
          f" | train-check {len(TRAIN_CHECK_SEEDS)} | workers {args.workers}")

    print("\n  reference rules on the validation layouts:")
    references: Dict[str, Dict[str, float]] = {}
    for rule in REFERENCE_RULES:
        references[rule] = evaluate(rule, VALIDATION_SEEDS, workers=args.workers)
        print(f"    {rule:<7} total loss {references[rule]['total_loss']*100:6.2f}%")

    rows: List[Dict[str, Any]] = []
    for init_seed in args.init_seeds:
        print(f"\n  init seed {init_seed}")
        rows.extend(train_one_seed(init_seed, args.runs, args.workers, args.out_dir))

    for rule, values in references.items():
        rows.append({
            "init_seed": -1, "run_index": 0, "kind": "reference", "routing": rule,
            "train_total_loss": None, "train_bce": None,
            "validation_total_loss": values["total_loss"],
            "train_check_total_loss": None,
            "validation_channel": values["channel"],
            "validation_queue_full": values["queue_full"],
            "validation_no_route": values["no_route"],
            "validation_ttl_hop": values["ttl_hop"],
        })

    path = os.path.join(args.out_dir, "1b_training_log.csv")
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=[LOAD_MS],
                                 placements=len(VALIDATION_SEEDS),
                                 extra="phase 1b training"))
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
