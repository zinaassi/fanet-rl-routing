"""
train_c_ppo_curriculum.py — Phase 2c: a staged curriculum on the near_gs
mission, against training straight at the hardest stage.

The question: does walking the C-drone actor up through easier missions beat
putting it in the hardest one from the start, for the same number of updates?

The staged run climbs D = 250, 350, 450, 550, 650. From the second stage on,
a quarter of each update's episodes are drawn from a random EARLIER stage, so
the actor is not free to forget what it already learned. Each stage keeps the
weights with the lowest validation loss, stops once it has run at least
MIN_UPDATES_PER_STAGE updates and then gone PATIENCE evaluations without a
new lowest, and reloads its best weights before the next stage starts.

The direct run starts from the same weights and trains at D = 650 only, for
exactly as many updates as the staged run used in total. That is the whole
point of the comparison: equal budget, different order.

What is carried across stages and what is not:

* the actor, the critic, the value normaliser and both optimisers continue,
  so the critic is warmed up once at the start and never again;
* the best-weight bookkeeping and the patience counter restart per stage,
  because "best at D = 250" and "best at D = 650" are different questions.

Every PPO number is imported from train_c_ppo and train_c_ppo_m4 rather than
retyped, and the run prints them before it starts.

Writes:
    out/2c_curriculum_staged_curve.csv   validation loss vs update, per stage
    out/2c_curriculum_direct_curve.csv   the same for the direct run
    out/2c_curriculum_curves.png         both, drawn
    out/2c_curriculum_config.json        every setting and seed used
    out/models/2c/c_actor_curriculum_staged.pt   (+ critic)
    out/models/2c/c_actor_curriculum_direct.pt   (+ critic)
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from scripts.stamp import csv_comment, git_commit
from scripts.train_c_ppo import (
    CLIP,
    CRITIC_ONLY_UPDATES,
    EPOCHS_PER_UPDATE,
    EVALUATE_EVERY,
    GAE_LAMBDA,
    GAMMA,
    GRAD_CLIP,
    LEARNING_RATE,
    LOAD_MS,
    MINIBATCH,
    ROUTING,
    train_one_seed,
    write_csv,
)
from scripts.train_c_ppo_m3 import (
    TRAINING_LAYOUTS,
    VALIDATION_LAYOUTS,
    sample_layouts,
)
from scripts.train_c_ppo_m4 import (
    EPISODES_PER_UPDATE_M4,
    MAX_UPDATES_M4,
    MODEL_DIR,
    MODEL_SUBDIR,
    PATIENCE_M4,
)

#: The rungs, in metres, easiest first. Step 1b measured these: D = 250 is
#: almost free, D = 650 puts 91% of M-drones out of direct reach of the GS.
STAGES: Tuple[float, ...] = (250.0, 350.0, 450.0, 550.0, 650.0)

#: Share of each update's episodes drawn from an earlier stage, from S2 on.
REPLAY_FRACTION: float = 0.25

#: A stage runs at least this many updates before patience may end it.
MIN_UPDATES_PER_STAGE: int = 100

#: Keep every new lowest validation loss. M4 used 0.2, which threw away two
#: checkpoints in run (a) that were genuinely better than the one it kept;
#: the curriculum is specified to keep the lowest, so the threshold is 0.
MIN_IMPROVEMENT_CURRICULUM: float = 0.0

#: The actor both runs start from: M3 seed 1, the best of the three M3 actors
#: in the moving world before any curriculum.
START_MODEL: str = "c_actor_m3_seed1.pt"

#: PPO seed. One value, used by both runs, so they draw the same layouts and
#: run seeds update for update and differ only in the mission they fly.
PPO_SEED: int = 1

#: Its own stream, so adding or removing replay cannot shift the layout and
#: run-seed draws that the staged and direct runs share.
REPLAY_SEED_TAG: int = 0xC0FFEE

STAGED: str = "staged"
DIRECT: str = "direct"

CURVE_COLUMNS = ("run", "stage", "mission_distance_m", "update",
                 "stage_update", "eval_loss", "saved", "views_agree")


def print_settings() -> None:
    """Print every number this run uses, and where it came from."""
    print("Phase 2c curriculum — settings, and the file each came from")
    print("  " + "-" * 72)
    rows = [
        ("mission", config.MISSION_DISTANCE_M and "near_gs" or "near_gs",
         "spec"),
        ("START_RADIUS_M", config.START_RADIUS_M, "config.py"),
        ("MIN_SPACING_M", config.MIN_SPACING_M, "config.py"),
        ("AVOID", True, "train_c_ppo._world"),
        ("CONTINUE_STD", config.CONTINUE_STD, "config.py"),
        ("stages (MISSION_DISTANCE_M)", "/".join(f"{d:.0f}" for d in STAGES),
         "this file"),
        ("replay fraction", REPLAY_FRACTION, "this file"),
        ("training layouts", f"{TRAINING_LAYOUTS[0]}-{TRAINING_LAYOUTS[1]}",
         "train_c_ppo_m3.py"),
        ("validation layouts",
         f"{VALIDATION_LAYOUTS[0]}-{VALIDATION_LAYOUTS[-1]}",
         "train_c_ppo_m3.py"),
        ("episodes per update", EPISODES_PER_UPDATE_M4, "train_c_ppo_m4.py"),
        ("critic-only warm-up", CRITIC_ONLY_UPDATES, "train_c_ppo.py"),
        ("evaluate every", EVALUATE_EVERY, "train_c_ppo.py"),
        ("patience", PATIENCE_M4, "train_c_ppo_m4.py"),
        ("max updates per stage", MAX_UPDATES_M4, "train_c_ppo_m4.py"),
        ("min updates per stage", MIN_UPDATES_PER_STAGE, "spec"),
        ("keep threshold", MIN_IMPROVEMENT_CURRICULUM,
         "spec — M4 used 0.2, see the note in this file"),
        ("anchor term", "off", "train_c_ppo_m4.py RUNS[0]"),
        ("gamma", GAMMA, "train_c_ppo.py"),
        ("GAE lambda", GAE_LAMBDA, "train_c_ppo.py"),
        ("clip", CLIP, "train_c_ppo.py"),
        ("epochs per update", EPOCHS_PER_UPDATE, "train_c_ppo.py"),
        ("learning rate", LEARNING_RATE, "train_c_ppo.py"),
        ("grad clip", GRAD_CLIP, "train_c_ppo.py"),
        ("minibatch", MINIBATCH, "train_c_ppo.py"),
        ("load", f"{LOAD_MS} ms", "train_c_ppo.py"),
        ("routing", ROUTING, "train_c_ppo.py"),
        ("start weights", START_MODEL, "spec"),
        ("PPO seed", PPO_SEED, "this file"),
    ]
    for name, value, source in rows:
        print(f"  {name:<28} {str(value):<22} {source}")
    print("  " + "-" * 72, flush=True)


def validation_run_seed(layout: int) -> int:
    """Return the run seed of a validation layout.

    The same seeds step 1b used, so the curriculum's validation numbers sit
    in the same table as the hand-written rules'.

    Args:
        layout: A placement seed in VALIDATION_LAYOUTS.

    Returns:
        9100 + (layout - 301).
    """
    return 9100 + layout - VALIDATION_LAYOUTS[0]


def validation_jobs(distance: float) -> Any:
    """Return an eval_jobs callback that evaluates at one mission distance.

    Args:
        distance: D in metres.

    Returns:
        state -> the thirty rollout jobs of one evaluation.
    """
    def jobs(state: Dict[str, Any]) -> List[Tuple]:
        return [(layout, validation_run_seed(layout), state, "rl", True,
                 True, distance)
                for layout in VALIDATION_LAYOUTS]
    return jobs


def replay_schedule(stage_index: int, episodes: int,
                    rng: np.random.Generator) -> Any:
    """Return a stage_schedule that mixes earlier stages into each update.

    From the second stage on, REPLAY_FRACTION of an update's episodes fly a
    uniformly chosen earlier stage and the rest fly the current one. The
    first stage has nothing to replay, so every episode is the current stage.

    The draws come from *rng*, which is NOT the generator the trainer uses
    for layouts and run seeds: if replay consumed from that one, the staged
    and direct runs would see different layouts and the comparison would be
    about more than the order of the stages.

    Args:
        stage_index: 0-based index into STAGES.
        episodes:    Episodes per update.
        rng:         The replay generator.

    Returns:
        (update, episodes) -> one mission distance per episode.
    """
    current = STAGES[stage_index]
    earlier = STAGES[:stage_index]
    replayed = int(round(REPLAY_FRACTION * episodes)) if earlier else 0

    def schedule(update: int, count: int) -> List[float]:
        distances = [current] * count
        if replayed:
            # The first `replayed` episodes of the update are the replayed
            # ones. Which episode carries which layout is already random, so
            # there is nothing to gain from shuffling the positions too.
            for index in range(min(replayed, count)):
                distances[index] = float(
                    earlier[rng.integers(len(earlier))]
                )
        return distances

    return schedule


def curve_rows(run: str, stage: str, distance: float, offset: int,
               evaluations: Sequence[Dict[str, Any]]
               ) -> List[Dict[str, Any]]:
    """Turn one stage's evaluations into curve rows.

    Args:
        run:         "staged" or "direct".
        stage:       The stage label.
        distance:    D in metres.
        offset:      Updates used before this stage began.
        evaluations: What train_one_seed returned.

    Returns:
        One row per evaluation.
    """
    return [{
        "run": run,
        "stage": stage,
        "mission_distance_m": distance,
        "update": offset + row["update"],
        "stage_update": row["update"],
        "eval_loss": row["eval_loss"],
        "saved": row["saved"],
        "views_agree": row.get("views_agree", ""),
    } for row in evaluations]


def start_weights(out_dir: str) -> Dict[str, Any]:
    """Load the actor both runs start from.

    Args:
        out_dir: Where the models live.

    Returns:
        The M3 seed 1 state dict.

    Raises:
        SystemExit: If the file is not there.
    """
    import torch

    path = os.path.join(out_dir, MODEL_DIR, START_MODEL)
    if not os.path.exists(path):
        path = os.path.join(out_dir, "models", START_MODEL)
    if not os.path.exists(path):
        raise SystemExit(f"the start model {START_MODEL} is not in {out_dir}")
    print(f"  starting both runs from {path}", flush=True)
    return torch.load(path, weights_only=True)


def train_staged(pool: Any, out_dir: str, episodes: int,
                 max_updates: int) -> Dict[str, Any]:
    """Run the staged curriculum and return its curve and totals.

    Args:
        pool:        The process pool.
        out_dir:     Where models and CSVs go.
        episodes:    Episodes per update.
        max_updates: Cap per stage.

    Returns:
        The curve rows, the per-stage update counts and the total.
    """
    import torch

    os.makedirs(os.path.join(out_dir, MODEL_DIR), exist_ok=True)
    replay_rng = np.random.default_rng(
        np.random.SeedSequence([PPO_SEED, REPLAY_SEED_TAG])
    )

    carry: Optional[Dict[str, Any]] = None
    state = start_weights(out_dir)
    rows: List[Dict[str, Any]] = []
    used: List[Dict[str, Any]] = []
    total = 0

    for index, distance in enumerate(STAGES):
        label = f"S{index + 1}"
        earlier = "/".join(f"{d:.0f}" for d in STAGES[:index]) or "none"
        print(f"\n=== staged {label}: D = {distance:.0f} m, "
              f"replaying {earlier} ===", flush=True)

        actor_name = os.path.join(MODEL_SUBDIR,
                                  f"c_actor_curriculum_{label}.pt")
        critic_name = os.path.join(MODEL_SUBDIR,
                                   f"c_critic_curriculum_{label}.pt")
        result = train_one_seed(
            PPO_SEED, pool, out_dir, max_updates, episodes,
            layouts=sample_layouts,
            eval_jobs=validation_jobs(distance),
            min_improvement=MIN_IMPROVEMENT_CURRICULUM,
            patience=PATIENCE_M4,
            model_name=actor_name,
            critic_name=critic_name,
            moving=True,
            min_updates=MIN_UPDATES_PER_STAGE,
            anchor=False,
            start_state=None if carry is not None else state,
            carry=carry,
            # The critic is cold only once. A later stage that froze its
            # actor again would spend its first ten updates not learning.
            warmup=CRITIC_ONLY_UPDATES if carry is None else 0,
            stage_schedule=replay_schedule(index, episodes, replay_rng),
        )

        rows.extend(curve_rows(STAGED, label, distance, total,
                               result["evaluations"]))
        spent = result["updates_used"]
        total += spent
        hit_cap = spent >= max_updates
        used.append({"stage": label, "mission_distance_m": distance,
                     "updates": spent, "best": result["best"],
                     "kept": result["kept"],
                     "best_update": result["best_update"],
                     "hit_cap": hit_cap})
        print(f"  staged {label}: {spent} updates, best "
              f"{result['best']:.2f}% at update {result['best_update']}"
              + ("  (HIT THE CAP)" if hit_cap else ""), flush=True)

        carry = result["carry"]
        # Condition (c): the stage's best weights go on to the next stage,
        # not whatever the last update happened to leave behind.
        kept_path = result["model"]
        if os.path.exists(kept_path):
            carry["actor"].load_state_dict(
                torch.load(kept_path, weights_only=True))
            print(f"  loaded {label}'s best weights back before advancing",
                  flush=True)
        else:
            print(f"  {label} never saved a version; carrying on as is",
                  flush=True)

    final_actor = os.path.join(out_dir, MODEL_DIR,
                               "c_actor_curriculum_staged.pt")
    final_critic = os.path.join(out_dir, MODEL_DIR,
                                "c_critic_curriculum_staged.pt")
    torch.save(carry["actor"].state_dict(), final_actor)
    torch.save(carry["critic"].state_dict(), final_critic)
    print(f"\n  staged_final = S5's best weights -> {final_actor}",
          flush=True)

    return {"rows": rows, "stages": used, "total": total}


def train_direct(pool: Any, out_dir: str, episodes: int,
                 budget: int) -> Dict[str, Any]:
    """Train at the hardest stage only, for the staged run's total budget.

    Args:
        pool:     The process pool.
        out_dir:  Where models and CSVs go.
        episodes: Episodes per update.
        budget:   Exactly this many updates, no early stop.

    Returns:
        The curve rows and the result.
    """
    import torch

    distance = STAGES[-1]
    print(f"\n=== direct: D = {distance:.0f} m only, {budget} updates "
          f"(the staged run's total) ===", flush=True)

    result = train_one_seed(
        PPO_SEED, pool, out_dir, budget, episodes,
        layouts=sample_layouts,
        eval_jobs=validation_jobs(distance),
        min_improvement=MIN_IMPROVEMENT_CURRICULUM,
        # Equal budget means equal budget: the run uses every update the
        # staged run used, so patience must not cut it short.
        patience=10 ** 9,
        model_name=os.path.join(MODEL_SUBDIR,
                                "c_actor_curriculum_direct.pt"),
        critic_name=os.path.join(MODEL_SUBDIR,
                                 "c_critic_curriculum_direct.pt"),
        moving=True,
        min_updates=budget,
        anchor=False,
        start_state=start_weights(out_dir),
        warmup=CRITIC_ONLY_UPDATES,
        stage_schedule=lambda update, count: [distance] * count,
    )

    print(f"  direct: {result['updates_used']} updates, best "
          f"{result['best']:.2f}% at update {result['best_update']}",
          flush=True)
    return {"rows": curve_rows(DIRECT, "S5", distance, 0,
                               result["evaluations"]),
            "result": result}


def write_config(path: str, staged: Dict[str, Any], budget: int) -> None:
    """Record every setting and seed, so the run can be rebuilt.

    Args:
        path:   Where to write the JSON.
        staged: What train_staged returned.
        budget: The shared update budget.
    """
    payload = {
        "commit": git_commit(),
        "mission": {
            "name": "near_gs",
            "start_radius_m": config.START_RADIUS_M,
            "min_spacing_m": config.MIN_SPACING_M,
            "stages_m": list(STAGES),
            "replay_fraction": REPLAY_FRACTION,
        },
        "world": {
            "avoid": True,
            "ideal_hello": False,
            "hello_interval_s": 1.0,
            "share_targets": True,
            "load_ms": LOAD_MS,
            "routing": ROUTING,
            "queue_capacity": config.QUEUE_CAPACITY,
            "num_m_drones": config.NUM_M_DRONES,
            "num_c_drones": config.NUM_C_DRONES,
        },
        "ppo": {
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip": CLIP,
            "epochs_per_update": EPOCHS_PER_UPDATE,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "minibatch": MINIBATCH,
            "episodes_per_update": EPISODES_PER_UPDATE_M4,
            "critic_only_updates": CRITIC_ONLY_UPDATES,
            "evaluate_every": EVALUATE_EVERY,
            "patience": PATIENCE_M4,
            "max_updates_per_stage": MAX_UPDATES_M4,
            "min_updates_per_stage": MIN_UPDATES_PER_STAGE,
            "min_improvement": MIN_IMPROVEMENT_CURRICULUM,
            "anchor": False,
            "continue_std": config.CONTINUE_STD,
            "start_model": START_MODEL,
        },
        "seeds": {
            "ppo_seed": PPO_SEED,
            "replay_seed_sequence": [PPO_SEED, REPLAY_SEED_TAG],
            "agent_stream_tag": config.AGENT_STREAM_TAG,
            "mobility_stream_tag": config.MOBILITY_STREAM_TAG,
            "hello_stream_tag": config.HELLO_STREAM_TAG,
            "c_drone_stream_tag": config.C_DRONE_STREAM_TAG,
            "training_layouts": list(TRAINING_LAYOUTS),
            "validation_layouts": list(VALIDATION_LAYOUTS),
            "validation_run_seeds": {
                str(layout): validation_run_seed(layout)
                for layout in VALIDATION_LAYOUTS
            },
        },
        "budget": {
            "staged_total_updates": budget,
            "direct_total_updates": budget,
            "per_stage": staged["stages"],
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"  wrote {path}")


def draw_curves(out_dir: str, staged: Sequence[Dict[str, Any]],
                direct: Sequence[Dict[str, Any]]) -> None:
    """Draw both validation curves, with the stage boundaries marked.

    Args:
        out_dir: Where the figure goes.
        staged:  The staged run's curve rows.
        direct:  The direct run's curve rows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.stamp import settings_stamp

    figure, axis = plt.subplots(figsize=(10.0, 5.4))

    colours = ["#4a90d9", "#d1495b", "#4c9a6a", "#d98b1f", "#8e6bb5"]
    for index, distance in enumerate(STAGES):
        label = f"S{index + 1}"
        rows = [row for row in staged if row["stage"] == label]
        if not rows:
            continue
        axis.plot([row["update"] for row in rows],
                  [row["eval_loss"] for row in rows],
                  color=colours[index % len(colours)], linewidth=1.6,
                  marker="o", markersize=3,
                  label=f"staged {label} (D = {distance:.0f} m)")
        axis.axvline(rows[0]["update"] - 1, color="#cccccc",
                     linewidth=0.8, zorder=0)

    if direct:
        axis.plot([row["update"] for row in direct],
                  [row["eval_loss"] for row in direct],
                  color="#2b2b2b", linewidth=1.6, linestyle="--",
                  marker="s", markersize=3,
                  label=f"direct (D = {STAGES[-1]:.0f} m throughout)")

    axis.set_xlabel("update")
    axis.set_ylabel("validation total loss (% of packets created)")
    axis.set_title("Phase 2c curriculum: validation loss per stage, "
                   "against training at the hardest stage only")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, frameon=False, ncol=2)
    figure.text(0.5, 0.005, settings_stamp(
        extra="phase 2c curriculum, validation 301-330, near_gs"),
        ha="center", fontsize=5.0, color="#666666", wrap=True)
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 1.0))

    path = os.path.join(out_dir, "2c_curriculum_curves.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    print(f"  wrote {path}")


def main() -> None:
    """Train the staged curriculum and the direct run, and save both."""
    parser = argparse.ArgumentParser(
        description="Phase 2c: a staged curriculum on the near_gs mission."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--episodes", type=int,
                        default=EPISODES_PER_UPDATE_M4)
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES_M4)
    parser.add_argument("--runs", type=str, nargs="+",
                        default=[STAGED, DIRECT],
                        choices=[STAGED, DIRECT])
    parser.add_argument("--budget", type=int, default=None,
                        help="updates for the direct run; the default is "
                             "whatever the staged run used")
    args = parser.parse_args()

    print_settings()
    started = time.perf_counter()

    context = multiprocessing.get_context("spawn")
    rows: List[Dict[str, Any]] = []
    staged: Dict[str, Any] = {"rows": [], "stages": [], "total": 0}
    direct_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        if STAGED in args.runs:
            staged = train_staged(pool, args.out_dir, args.episodes,
                                  args.max_updates)
            rows.extend(staged["rows"])
            write_csv(
                os.path.join(args.out_dir, "2c_curriculum_staged_curve.csv"),
                staged["rows"], CURVE_COLUMNS,
                "phase 2c curriculum, STAGED run, validation 301-330",
            )
            print("\n  updates per stage")
            for stage in staged["stages"]:
                print(f"    {stage['stage']}  D = "
                      f"{stage['mission_distance_m']:.0f} m  "
                      f"{stage['updates']:>4} updates  best "
                      f"{stage['best']:.2f}%"
                      + ("  (cap)" if stage["hit_cap"] else ""))
            print(f"    total {staged['total']} updates")

        budget = args.budget if args.budget is not None else staged["total"]
        if DIRECT in args.runs:
            if not budget:
                raise SystemExit(
                    "the direct run needs a budget: run the staged run "
                    "first, or pass --budget"
                )
            direct = train_direct(pool, args.out_dir, args.episodes, budget)
            direct_rows = direct["rows"]
            rows.extend(direct_rows)
            write_csv(
                os.path.join(args.out_dir, "2c_curriculum_direct_curve.csv"),
                direct_rows, CURVE_COLUMNS,
                "phase 2c curriculum, DIRECT run, validation 301-330",
            )

    if staged["rows"] or direct_rows:
        draw_curves(args.out_dir, staged["rows"], direct_rows)
    if staged["stages"]:
        write_config(
            os.path.join(args.out_dir, "2c_curriculum_config.json"),
            staged, budget,
        )

    print(f"\n  total {(time.perf_counter() - started) / 3600.0:.2f} h")


if __name__ == "__main__":
    main()
