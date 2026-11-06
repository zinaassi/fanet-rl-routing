"""
train_c_ppo_round1.py — Phase 2c round 1: a reward sweep from the direct model.

Fine-tunes the direct actor (c_actor_curriculum_direct.pt) four times, one
reward change at a time, against simply training more:

    control        the reward unchanged
    spacing_med    spacing penalty, w = 0.01
    spacing_small  spacing penalty, w = 0.005
    spacing_large  spacing penalty, w = 0.02

THE REWARD. Per C-drone, per second, as c_interface.flight_reward gives it,
plus (training only) two terms measured at the END of the second the reward
covers, i.e. at the step the simulator closes that reward:

    spacing    - w * sum over the OTHER C-drones closer than 100 m of
                 (1 - d / 100), on true positions
    fairness   + v * share of M-drones with a packet delivered in the last
                 5 s. Kept in the code for mixed-fleet training later; every
                 round-1 run has v = 0.

The score that chooses anything is validation TOTAL PACKET LOSS, never the
reward, which differs between runs.

THE RUNS. Every run starts from the same place: the direct actor (spread put
back to config.CONTINUE_STD), the direct critic's weights, a fresh value
normaliser and fresh optimisers, ten critic-only warm-up updates. Same PPO
seed for all four, so they see the same layouts and run seeds in the same
order. 400 updates, no early stop; validation 301-330 every 10 updates; the
best weights are kept. The PPO settings are train_c_ppo's, imported, not
retyped.

CHECKPOINTS. Every CHECKPOINT_EVERY updates the whole training state goes to
out/checkpoints/2c_round1/<run>/, the latest KEEP_CHECKPOINTS kept, and
--resume continues a run from its latest one. --resume-test checks that a
resumed run is bit-for-bit the run that never stopped.

Writes:
    out/2c_r1_curve_<run>.csv        validation curve
    out/2c_r1_log_<run>.csv          per-update training log
    out/2c_r1_curves.png             every finished run's curve
    out/models/2c/c_{actor,critic}_r1_<run>.pt
    out/2c_r1_validation.csv         best actors on validation, per layout
    out/2c_r1_summary.csv            per run: loss, causes, flown, C-C gap
    out/2c_r1_paired.csv             each run - control, per layout, 95% CI
    out/2c_r1_edge.csv               edge-case totals for each best actor
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import multiprocessing
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from fanet_sim.envs import c_interface
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.train_c_ppo import (
    CAUSES,
    CLIP,
    CRITIC_ONLY_UPDATES,
    EPOCHS_PER_UPDATE,
    EVALUATE_EVERY,
    GAE_LAMBDA,
    GAMMA,
    GRAD_CLIP,
    LEARNING_RATE,
    MINIBATCH,
    ROUTING,
    SamplingAgent,
    _restore,
    _transitions,
    _world,
    advantages,
    write_csv,
)
from scripts.train_c_ppo_curriculum import (
    MIN_IMPROVEMENT_CURRICULUM,
    validation_run_seed,
)
from scripts.train_c_ppo_m3 import VALIDATION_LAYOUTS, sample_layouts
from scripts.train_c_ppo_m4 import EPISODES_PER_UPDATE_M4

#: Run -> (spacing weight w, fairness weight v), in the order they run.
RUNS: Dict[str, Tuple[float, float]] = {
    "control": (0.0, 0.0),
    "spacing_med": (0.01, 0.0),
    "spacing_small": (0.005, 0.0),
    "spacing_large": (0.02, 0.0),
}

#: The fairness weights proposed for later, mixed-fleet, rounds. Unused here.
FAIRNESS_PROPOSED: Dict[str, float] = {"small": 0.05, "medium": 0.10}

SPACING_RADIUS_M: float = 100.0
FAIRNESS_WINDOW_S: float = 5.0

PPO_SEED: int = 2
UPDATES: int = 400
DISTANCE_M: float = 650.0
CHECKPOINT_EVERY: int = 25
KEEP_CHECKPOINTS: int = 2

START_ACTOR: str = os.path.join("models", "2c", "c_actor_curriculum_direct.pt")
START_CRITIC: str = os.path.join("models", "2c",
                                 "c_critic_curriculum_direct.pt")
CHECKPOINT_DIR: str = os.path.join("checkpoints", "2c_round1")

#: Student's t at 95% for 29 degrees of freedom, as curriculum_validate uses.
T_95_DF29: float = 2.045


# ---------------------------------------------------------------------------
# The two extra reward terms
# ---------------------------------------------------------------------------

def spacing_penalty(positions: np.ndarray, me: int, c_ids: Sequence[int],
                    w: float) -> float:
    """Return -w * sum over other C-drones within the radius of (1 - d/R).

    Args:
        positions: Every drone's true position at that moment.
        me:        This C-drone's index.
        c_ids:     Every C-drone's index.
        w:         The weight; two drones on top of each other cost w.

    Returns:
        The penalty, zero or negative.
    """
    total = 0.0
    for other in c_ids:
        if other == me:
            continue
        d = float(np.linalg.norm(positions[me] - positions[other]))
        if d < SPACING_RADIUS_M:
            total += 1.0 - d / SPACING_RADIUS_M
    return -w * total


def fairness_share(deliveries: Sequence[Tuple[int, int]], m_count: int,
                   end_step: int) -> float:
    """Share of M-drones with a packet delivered in the window before *end_step*.

    The window is the FAIRNESS_WINDOW_S seconds whose deliveries the
    simulator has counted by the time it closes the reward at *end_step*:
    delivered_at in [end_step - window, end_step).

    Args:
        deliveries: (source id, delivered_at step) per delivered packet.
        m_count:    How many M-drones there are.
        end_step:   The step at which the reward is closed.

    Returns:
        A share in [0, 1].
    """
    window = int(round(FAIRNESS_WINDOW_S / config.TIMESTEP))
    sources = {source for source, at in deliveries
               if end_step - window <= at < end_step}
    return len(sources) / m_count if m_count else 0.0


def shape_rewards(env: Any, positions: np.ndarray, w: float, v: float
                  ) -> Dict[str, float]:
    """Add the extra terms to every closed reward in env.c_decisions.

    A decision taken at step s has its reward closed at s + one decision
    interval, before anything moves in that step, so the positions are the
    ones recorded after step s + interval - 1 finished, positions[s +
    interval].

    Args:
        env:       The finished environment.
        positions: positions[k] = every drone's position after k steps.
        w:         Spacing weight.
        v:         Fairness weight.

    Returns:
        The mean of each added term, per decision.
    """
    interval = c_interface.steps_per_decision()
    c_ids = [d.drone_id for d in env.drones if d.drone_type == "C"]
    m_count = sum(1 for d in env.drones if d.drone_type == "M")
    deliveries = [(p.source_id, p.delivered_at) for p in env.delivered]
    added_spacing: List[float] = []
    added_fair: List[float] = []
    for row in env.c_decisions:
        if row["reward"] is None:
            continue
        end = row["step"] + interval
        spacing = (spacing_penalty(positions[end], row["drone_id"], c_ids, w)
                   if w else 0.0)
        fair = v * fairness_share(deliveries, m_count, end) if v else 0.0
        if w or v:
            row["reward"] = row["reward"] + spacing + fair
        added_spacing.append(spacing)
        added_fair.append(fair)
    return {
        "spacing_term": statistics.fmean(added_spacing or [0.0]),
        "fairness_term": statistics.fmean(added_fair or [0.0]),
    }


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------

def run_episode(job: Tuple) -> Dict[str, Any]:
    """Run one near_gs episode with the actor; train or evaluate.

    The same world, agent and seeding as train_c_ppo._run_episode, plus a
    record of every drone's position per step, which the spacing term and
    the C-to-C distance need.

    Args:
        job: (layout, run_seed, actor state, greedy, w, v).

    Returns:
        The metrics, and the transitions when not greedy.
    """
    import torch

    from agents.c_actor import GaussianActor

    layout, run_seed, state, greedy, w, v = job
    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [run_seed, config.AGENT_STREAM_TAG]).generate_state(1)[0]))
    previous = _world(True, DISTANCE_M)
    try:
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed)
        env.reset()
        actor = GaussianActor()
        actor.load_state_dict(state)
        actor.eval()
        agent = SamplingAgent(env, actor, greedy=greedy)
        env.c_agent = agent

        positions = [np.array([d.position for d in env.drones])]
        for _ in range(config.MAX_STEPS):
            env.step()
            positions.append(np.array([d.position for d in env.drones]))
        env.close_logger()
        positions = np.array(positions)

        terms = (shape_rewards(env, positions, w, v) if not greedy
                 else {"spacing_term": 0.0, "fairness_term": 0.0})

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        c_ids = [d.drone_id for d in env.drones if d.drone_type == "C"]
        c_pos = positions[:, c_ids, :]
        gaps = np.linalg.norm(c_pos[:, :, None, :] - c_pos[:, None, :, :],
                              axis=-1)
        upper = np.triu_indices(len(c_ids), k=1)
        out: Dict[str, Any] = {
            "total_loss": truth["total_loss"] or 0.0,
            **{cause: truth["loss_rate_by_reason"][cause] for cause in CAUSES},
            "c_distance_flown_mean": truth["c_distance_flown_mean"],
            "min_c_to_c_m": float(gaps[:, upper[0], upper[1]].min()),
            "views_agree": not metrics["discrepancies"],
            "mean_reward": statistics.fmean(
                [r["reward"] for r in env.c_decisions
                 if r["reward"] is not None] or [0.0]),
            **terms,
        }
        if not greedy:
            out.update(_transitions(env, agent))
        return out
    finally:
        _restore(previous)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_paths(ckpt_dir: str) -> List[str]:
    """Every checkpoint in *ckpt_dir*, oldest first."""
    return sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))


def save_checkpoint(ckpt_dir: str, s: Dict[str, Any]) -> str:
    """Write the whole training state, then keep only the latest few."""
    import torch

    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{s['update']:04d}.pt")
    payload = {
        "run": s["run"],
        "update": s["update"],
        "actor": s["actor"].state_dict(),
        "critic": s["critic"].state_dict(),
        "actor_opt": s["actor_opt"].state_dict(),
        "critic_opt": s["critic_opt"].state_dict(),
        "normaliser": {"mean": s["normaliser"].mean,
                       "var": s["normaliser"].var,
                       "count": s["normaliser"].count},
        "numpy_rng": s["rng"].bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "best": s["best"],
        "best_update": s["best_update"],
        "log": s["log"],
        "evaluations": s["evaluations"],
    }
    # Written under a temporary name and renamed, so a crash mid-write
    # never leaves a half checkpoint looking like the latest one.
    partial = path + ".partial"
    torch.save(payload, partial)
    os.replace(partial, path)
    for old in checkpoint_paths(ckpt_dir)[:-KEEP_CHECKPOINTS]:
        os.remove(old)
    return path


def fresh_state(run: str, out_dir: str, critic_from_file: bool = True
                ) -> Dict[str, Any]:
    """Build a run's starting state from the direct model.

    The construction order matches train_one_seed's, so with
    critic_from_file=False and no extra terms this IS train_one_seed's start.

    Args:
        run:              The run's name.
        out_dir:          Where the models live.
        critic_from_file: Load the direct critic's weights (round 1). False
                          keeps the freshly initialised critic, for the test
                          that compares against train_one_seed.

    Returns:
        The training state.
    """
    import torch

    from agents.c_actor import CCritic, GaussianActor, RunningNorm

    torch.manual_seed(PPO_SEED)
    rng = np.random.default_rng(PPO_SEED)
    actor = GaussianActor()
    actor.load_state_dict(torch.load(os.path.join(out_dir, START_ACTOR),
                                     weights_only=True))
    with torch.no_grad():
        actor.log_std.fill_(float(np.log(config.CONTINUE_STD)))
    critic = CCritic(c_interface.global_state_width(
        config.NUM_M_DRONES + config.NUM_C_DRONES))
    if critic_from_file:
        critic.load_state_dict(torch.load(os.path.join(out_dir, START_CRITIC),
                                          weights_only=True))
    return {
        "run": run, "update": 0, "rng": rng,
        "actor": actor, "critic": critic, "normaliser": RunningNorm(),
        "actor_opt": torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE),
        "critic_opt": torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE),
        "best": float("inf"), "best_update": 0, "log": [], "evaluations": [],
    }


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Rebuild the training state from a checkpoint."""
    import torch

    from agents.c_actor import CCritic, GaussianActor, RunningNorm

    payload = torch.load(path, weights_only=False)
    actor = GaussianActor()
    actor.load_state_dict(payload["actor"])
    critic = CCritic(c_interface.global_state_width(
        config.NUM_M_DRONES + config.NUM_C_DRONES))
    critic.load_state_dict(payload["critic"])
    actor_opt = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    actor_opt.load_state_dict(payload["actor_opt"])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)
    critic_opt.load_state_dict(payload["critic_opt"])
    normaliser = RunningNorm()
    normaliser.mean = payload["normaliser"]["mean"]
    normaliser.var = payload["normaliser"]["var"]
    normaliser.count = payload["normaliser"]["count"]
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["numpy_rng"]
    # Last, after every constructor above has drawn from the torch RNG.
    torch.set_rng_state(payload["torch_rng"])
    return {
        "run": payload["run"], "update": payload["update"], "rng": rng,
        "actor": actor, "critic": critic, "normaliser": normaliser,
        "actor_opt": actor_opt, "critic_opt": critic_opt,
        "best": payload["best"], "best_update": payload["best_update"],
        "log": payload["log"], "evaluations": payload["evaluations"],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def validation_jobs(state: Dict[str, Any]) -> List[Tuple]:
    """Thirty greedy validation rollouts, with the curriculum's run seeds."""
    return [(layout, validation_run_seed(layout), state, True, 0.0, 0.0)
            for layout in VALIDATION_LAYOUTS]


def train(s: Dict[str, Any], pool: Any, *, w: float, v: float,
          until: int, episodes: int, out_dir: str,
          ckpt_dir: Optional[str], model_tag: Optional[str],
          on_progress: Any = None) -> Dict[str, Any]:
    """Continue the training state *s* up to update *until*.

    The update itself is train_one_seed's, with the anchor off (as the
    direct run had it), MIN_IMPROVEMENT_CURRICULUM as the keep threshold and
    no early stop.

    Args:
        s:           The training state, fresh or from a checkpoint.
        pool:        The process pool.
        w, v:        The spacing and fairness weights.
        until:       Stop after this update.
        episodes:    Episodes per update.
        out_dir:     Where models go.
        ckpt_dir:    Where checkpoints go, or None for none.
        model_tag:   Models are saved as c_{actor,critic}_r1_<tag>.pt, or not
                     at all when None.
        on_progress: Called with the state after every evaluation.

    Returns:
        The state, advanced.
    """
    import torch

    actor, critic = s["actor"], s["critic"]
    normaliser, rng = s["normaliser"], s["rng"]
    actor_opt, critic_opt = s["actor_opt"], s["critic_opt"]
    model_dir = os.path.join(out_dir, "models", "2c")

    for update in range(s["update"] + 1, until + 1):
        started = time.perf_counter()
        state = {k: t.clone() for k, t in actor.state_dict().items()}
        run_seeds = rng.integers(1, 10 ** 8, size=episodes).tolist()
        episode_layouts = sample_layouts(rng, episodes)
        jobs = [(int(layout), int(seed), state, False, w, v)
                for layout, seed in zip(episode_layouts, run_seeds)]
        results = list(pool.map(run_episode, jobs))
        if not all(r["views_agree"] for r in results):
            raise RuntimeError(f"views disagreed during update {update}")

        batch = {key: np.concatenate([r[key] for r in results])
                 for key in ("self", "neighbours", "mask", "global", "action",
                             "log_prob", "reward", "drone", "order")}
        offsets = np.concatenate([np.full(len(r["reward"]), index * 1000)
                                  for index, r in enumerate(results)])
        trajectories = batch["drone"] + offsets

        own = torch.as_tensor(batch["self"], dtype=torch.float32)
        neighbours = torch.as_tensor(batch["neighbours"], dtype=torch.float32)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32)
        global_state = torch.as_tensor(batch["global"], dtype=torch.float32)
        actions = torch.as_tensor(batch["action"], dtype=torch.float32)
        old_log_prob = torch.as_tensor(batch["log_prob"], dtype=torch.float32)

        with torch.no_grad():
            values = normaliser.denormalise(
                critic(global_state).numpy().astype(np.float64))
        advantage, returns = advantages(batch["reward"], values,
                                        trajectories, batch["order"])
        normaliser.update(returns)
        target = torch.as_tensor(normaliser.normalise(returns),
                                 dtype=torch.float32)
        advantage_t = torch.as_tensor(
            (advantage - advantage.mean()) / (advantage.std() + 1e-8),
            dtype=torch.float32)

        train_actor = update > CRITIC_ONLY_UPDATES
        policy_loss = value_loss = 0.0
        for _ in range(EPOCHS_PER_UPDATE):
            order = torch.randperm(len(target))
            for start in range(0, len(order), MINIBATCH):
                picked = order[start:start + MINIBATCH]
                predicted = critic(global_state[picked])
                v_loss = ((predicted - target[picked]) ** 2).mean()
                critic_opt.zero_grad()
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                critic_opt.step()
                value_loss = float(v_loss.detach())
                if not train_actor:
                    continue
                distribution = actor.distribution(
                    own[picked], neighbours[picked], mask[picked])
                log_prob = distribution.log_prob(actions[picked]).sum(-1)
                ratio = (log_prob - old_log_prob[picked]).exp()
                clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
                p_loss = -torch.min(ratio * advantage_t[picked],
                                    clipped * advantage_t[picked]).mean()
                actor_opt.zero_grad()
                p_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                actor_opt.step()
                policy_loss = float(p_loss.detach())

        row = {
            "run": s["run"], "update": update,
            "train_loss": statistics.fmean([r["total_loss"]
                                            for r in results]) * 100,
            "mean_reward": statistics.fmean([r["mean_reward"]
                                             for r in results]),
            "spacing_term": statistics.fmean([r["spacing_term"]
                                              for r in results]),
            "fairness_term": statistics.fmean([r["fairness_term"]
                                               for r in results]),
            "policy_loss": policy_loss, "value_loss": value_loss,
            "std": float(actor.log_std.exp().mean().detach()),
            "decisions": len(target),
            "seconds": time.perf_counter() - started,
        }
        s["log"].append(row)
        s["update"] = update
        print(f"  {s['run']} update {update:>3}  train {row['train_loss']:6.2f}%"
              f"  reward {row['mean_reward']:+.4f}  spacing "
              f"{row['spacing_term']:+.5f}  std {row['std']:.3f}  "
              f"{row['seconds']:5.1f}s", flush=True)

        if update % EVALUATE_EVERY == 0:
            eval_state = {k: t.clone() for k, t in actor.state_dict().items()}
            evals = list(pool.map(run_episode, validation_jobs(eval_state)))
            agreed = sum(1 for r in evals if r["views_agree"])
            if agreed != len(evals):
                raise RuntimeError(f"views disagreed in validation at "
                                   f"update {update}")
            score = statistics.fmean([r["total_loss"] for r in evals]) * 100
            saved = score < s["best"] - MIN_IMPROVEMENT_CURRICULUM
            if saved:
                s["best"], s["best_update"] = score, update
                if model_tag is not None:
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(actor.state_dict(), os.path.join(
                        model_dir, f"c_actor_r1_{model_tag}.pt"))
                    torch.save(critic.state_dict(), os.path.join(
                        model_dir, f"c_critic_r1_{model_tag}.pt"))
            s["evaluations"].append({
                "run": s["run"], "update": update, "eval_loss": score,
                "saved": saved, "views_agree": f"{agreed}/{len(evals)}"})
            print(f"  {s['run']} update {update:>3}  EVALUATION {score:6.2f}%"
                  + ("  <- best, saved" if saved else ""), flush=True)
            if on_progress is not None:
                on_progress(s)

        if ckpt_dir is not None and update % CHECKPOINT_EVERY == 0:
            path = save_checkpoint(ckpt_dir, s)
            print(f"  checkpoint {path}", flush=True)
    return s


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def write_curves(s: Dict[str, Any], out_dir: str) -> None:
    """Rewrite one run's curve and training log."""
    w, v = RUNS[s["run"]]
    note = (f"phase 2c round 1, run {s['run']}, spacing w={w}, fairness "
            f"v={v}, PPO seed {PPO_SEED}, from the direct model, D=650, "
            f"validation 301-330, score = total loss")
    write_csv(os.path.join(out_dir, f"2c_r1_curve_{s['run']}.csv"),
              s["evaluations"],
              ["run", "update", "eval_loss", "saved", "views_agree"], note)
    write_csv(os.path.join(out_dir, f"2c_r1_log_{s['run']}.csv"), s["log"],
              ["run", "update", "train_loss", "mean_reward", "spacing_term",
               "fairness_term", "policy_loss", "value_loss", "std",
               "decisions", "seconds"], note)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path) as handle:
        return list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))


def plot_curves(out_dir: str) -> None:
    """One plot with every run whose curve exists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.stamp import settings_stamp

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run in RUNS:
        path = os.path.join(out_dir, f"2c_r1_curve_{run}.csv")
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        ax.plot([int(r["update"]) for r in rows],
                [float(r["eval_loss"]) for r in rows], lw=1.2,
                label=f"{run} (w={RUNS[run][0]})")
    ax.set_xlabel("update")
    ax.set_ylabel("validation total loss (%)")
    ax.set_title("Round 1: fine-tuning the direct actor, validation 301-330, "
                 "D = 650 m", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.005, settings_stamp(extra="phase 2c round 1"),
             ha="center", fontsize=4.5, color="#666666", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out_dir, "2c_r1_curves.png"), dpi=130)
    plt.close(fig)


def validate_best(run: str, pool: Any, out_dir: str) -> List[Dict[str, Any]]:
    """Run a run's best actor on the thirty validation layouts."""
    import torch

    state = torch.load(os.path.join(out_dir, "models", "2c",
                                    f"c_actor_r1_{run}.pt"), weights_only=True)
    jobs = validation_jobs(state)
    results = list(pool.map(run_episode, jobs))
    if not all(r["views_agree"] for r in results):
        raise RuntimeError(f"views disagreed validating {run}")
    return [{
        "run": run, "placement_seed": job[0], "run_seed": job[1],
        "total_loss": 100.0 * r["total_loss"],
        **{cause: 100.0 * r[cause] for cause in CAUSES},
        "c_distance_flown_mean": r["c_distance_flown_mean"],
        "min_c_to_c_m": r["min_c_to_c_m"],
        "views_agree": r["views_agree"],
    } for job, r in zip(jobs, results)]


def run_edge_cases(run: str, out_dir: str, workers: int) -> None:
    """The edge-case script on this run's best actor, in its own folder."""
    model = os.path.join(out_dir, "models", "2c", f"c_actor_r1_{run}.pt")
    command = [sys.executable, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "edge_cases_2c.py"),
        "--out-dir", out_dir, "--workers", str(workers),
        "--model", model, "--out", os.path.join(out_dir, f"2c_edge_r1_{run}")]
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def report(out_dir: str) -> None:
    """Build the summary, paired and edge tables from the saved files."""
    note = ("phase 2c round 1, best actor per run on validation 301-330, "
            "D = 650, from the saved per-layout file")
    vpath = os.path.join(out_dir, "2c_r1_validation.csv")
    if not os.path.exists(vpath):
        return
    rows = read_csv(vpath)
    runs = [run for run in RUNS if any(r["run"] == run for r in rows)]
    numeric = ["total_loss"] + list(CAUSES) + ["c_distance_flown_mean",
                                               "min_c_to_c_m"]

    summary = []
    for run in runs:
        group = [r for r in rows if r["run"] == run]
        curve = read_csv(os.path.join(out_dir, f"2c_r1_curve_{run}.csv"))
        best = min(curve, key=lambda r: float(r["eval_loss"]))
        summary.append({
            "run": run, "spacing_w": RUNS[run][0],
            "best_update": int(best["update"]),
            "curve_best": float(best["eval_loss"]),
            **{c: statistics.fmean(float(r[c]) for r in group)
               for c in numeric},
            "min_c_to_c_m_worst": min(float(r["min_c_to_c_m"])
                                      for r in group),
        })
    write_csv(os.path.join(out_dir, "2c_r1_summary.csv"), summary,
              ["run", "spacing_w", "best_update", "curve_best"] + numeric
              + ["min_c_to_c_m_worst"], note)

    paired = []
    if "control" in runs:
        base = {int(r["placement_seed"]): float(r["total_loss"])
                for r in rows if r["run"] == "control"}
        for run in runs:
            if run == "control":
                continue
            mine = {int(r["placement_seed"]): float(r["total_loss"])
                    for r in rows if r["run"] == run}
            diffs = [mine[k] - base[k] for k in VALIDATION_LAYOUTS]
            mean = statistics.fmean(diffs)
            half = T_95_DF29 * statistics.stdev(diffs) / math.sqrt(len(diffs))
            paired.append({"run": run, "against": "control",
                           "layouts": len(diffs), "mean_diff": mean,
                           "ci95_low": mean - half, "ci95_high": mean + half,
                           "wins": sum(1 for d in diffs if d < 0)})
    write_csv(os.path.join(out_dir, "2c_r1_paired.csv"), paired,
              ["run", "against", "layouts", "mean_diff", "ci95_low",
               "ci95_high", "wins"], note)

    edge = []
    for run in runs:
        path = os.path.join(out_dir, f"2c_edge_r1_{run}", "edge_summary.csv")
        if not os.path.exists(path):
            continue
        for r in read_csv(path):
            if r["policy"] == "direct":
                edge.append({"run": run, "scenario": int(float(r["scenario"])),
                             "total_loss": float(r["total_loss"]),
                             "c_distance_flown_mean":
                                 float(r["c_distance_flown_mean"])})
    write_csv(os.path.join(out_dir, "2c_r1_edge.csv"), edge,
              ["run", "scenario", "total_loss", "c_distance_flown_mean"],
              "phase 2c round 1, edge cases (edge_cases_2c.py) on each run's "
              "best actor, mean over run seeds 9900-9904")

    print("\n  Round 1 summary (from out/2c_r1_summary.csv)")
    print(f"  {'run':>14} {'w':>6} {'best@':>5} {'total':>6} "
          + " ".join(f"{c[:8]:>8}" for c in CAUSES)
          + f" {'C flown':>8} {'minCC':>6} {'worstCC':>7}")
    for r in read_csv(os.path.join(out_dir, "2c_r1_summary.csv")):
        print(f"  {r['run']:>14} {float(r['spacing_w']):6.3f} "
              f"{r['best_update']:>5} {float(r['total_loss']):6.2f} "
              + " ".join(f"{float(r[c]):8.2f}" for c in CAUSES)
              + f" {float(r['c_distance_flown_mean']):7.0f}m"
              f" {float(r['min_c_to_c_m']):6.1f}"
              f" {float(r['min_c_to_c_m_worst']):7.1f}")
    paired_rows = read_csv(os.path.join(out_dir, "2c_r1_paired.csv"))
    if paired_rows:
        print("\n  Paired against control, 30 layouts, points of total loss")
        for r in paired_rows:
            print(f"  {r['run']:>14}  {float(r['mean_diff']):+6.2f}  "
                  f"[{float(r['ci95_low']):+.2f}, {float(r['ci95_high']):+.2f}]"
                  f"  lower on {r['wins']}/{r['layouts']}")
    edge_rows = read_csv(os.path.join(out_dir, "2c_r1_edge.csv"))
    if edge_rows:
        print("\n  Edge cases, best actor, total loss % (mean of 5 seeds)")
        print(f"  {'run':>14} " + " ".join(f"{'s' + str(k):>6}"
                                           for k in range(1, 9)))
        for run in runs:
            cells = {int(r["scenario"]): float(r["total_loss"])
                     for r in edge_rows if r["run"] == run}
            if cells:
                print(f"  {run:>14} " + " ".join(f"{cells[k]:6.2f}"
                                                 for k in range(1, 9)))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def print_settings(episodes: int) -> None:
    import scripts.train_c_ppo as base

    print("Round 1 settings:")
    rows = [
        ("reward", "(delivered - lost)/created - C_FLIGHT_COST*(metres/speed)",
         "c_interface.flight_reward"),
        ("C_FLIGHT_COST", config.C_FLIGHT_COST, "config.py"),
        ("C_DRONE_SPEED", config.C_DRONE_SPEED, "config.py"),
        ("largest flight cost in 1 s",
         config.C_FLIGHT_COST * config.C_DRONE_SPEED / config.C_DRONE_SPEED,
         "C_FLIGHT_COST * (speed * 1 s) / speed"),
        ("runs (w, v)", RUNS, "this file"),
        ("spacing radius m", SPACING_RADIUS_M, "spec"),
        ("fairness window s", FAIRNESS_WINDOW_S, "spec (off here)"),
        ("start actor", START_ACTOR, "spec"),
        ("start critic", START_CRITIC, "spec; fresh normaliser + optimisers"),
        ("CONTINUE_STD", config.CONTINUE_STD, "config.py"),
        ("critic-only warm-up", CRITIC_ONLY_UPDATES, "train_c_ppo.py"),
        ("PPO seed", PPO_SEED, "approved"),
        ("updates per run", UPDATES, "spec, no early stop"),
        ("episodes per update", episodes, "train_c_ppo_m4.py"),
        ("clip / epochs / lr", (base.CLIP, base.EPOCHS_PER_UPDATE,
                                base.LEARNING_RATE), "train_c_ppo.py"),
        ("grad clip / minibatch", (base.GRAD_CLIP, base.MINIBATCH),
         "train_c_ppo.py"),
        ("gamma / lambda", (base.GAMMA, base.GAE_LAMBDA), "train_c_ppo.py"),
        ("evaluate every", EVALUATE_EVERY, "train_c_ppo.py"),
        ("keep threshold", MIN_IMPROVEMENT_CURRICULUM,
         "train_c_ppo_curriculum.py"),
        ("mission", f"near_gs D={DISTANCE_M}, AVOID, hello 1 s, 200 ms, "
                    f"{ROUTING}", "train_c_ppo._world"),
        ("fleet", f"{config.NUM_M_DRONES} M + {config.NUM_C_DRONES} C",
         "config.py"),
        ("validation", "301-330, run seed 9100 + (layout - 301)",
         "train_c_ppo_curriculum.validation_run_seed"),
        ("checkpoint every / keep", (CHECKPOINT_EVERY, KEEP_CHECKPOINTS),
         "spec"),
    ]
    for name, value, source in rows:
        print(f"  {name:28s} {str(value):60s} [{source}]")


def resume_test(pool: Any, out_dir: str, episodes: int) -> None:
    """30 updates straight vs 25 + stop + resume to 30, on the control run.

    Raises:
        SystemExit: If anything differs.
    """
    import torch

    w, v = RUNS["control"]
    scratch = tempfile.mkdtemp(prefix="r1_resume_")
    print(f"\n=== resume test: control, 30 straight vs 25 + resume, "
          f"scratch {scratch} ===", flush=True)
    straight = train(fresh_state("control", out_dir), pool, w=w, v=v,
                     until=30, episodes=episodes, out_dir=out_dir,
                     ckpt_dir=os.path.join(scratch, "a"), model_tag=None)
    first = train(fresh_state("control", out_dir), pool, w=w, v=v,
                  until=25, episodes=episodes, out_dir=out_dir,
                  ckpt_dir=os.path.join(scratch, "b"), model_tag=None)
    del first
    resumed = load_checkpoint(checkpoint_paths(os.path.join(scratch, "b"))[-1])
    print(f"  resumed from update {resumed['update']}", flush=True)
    resumed = train(resumed, pool, w=w, v=v, until=30, episodes=episodes,
                    out_dir=out_dir, ckpt_dir=os.path.join(scratch, "b"),
                    model_tag=None)

    problems = []
    for a, b in zip(straight["log"], resumed["log"]):
        for key in ("train_loss", "mean_reward", "policy_loss", "value_loss",
                    "std"):
            if a[key] != b[key]:
                problems.append(f"update {a['update']} {key}: {a[key]} vs "
                                f"{b[key]}")
    if [e["eval_loss"] for e in straight["evaluations"]] != \
            [e["eval_loss"] for e in resumed["evaluations"]]:
        problems.append("validation losses differ")
    for name in ("actor", "critic"):
        sa, sb = straight[name].state_dict(), resumed[name].state_dict()
        for key in sa:
            if not torch.equal(sa[key], sb[key]):
                problems.append(f"{name} weight {key} differs")
    if len(straight["log"]) != 30 or len(resumed["log"]) != 30:
        problems.append("a run did not reach 30 updates")

    print(f"  validation losses straight {[round(e['eval_loss'], 6) for e in straight['evaluations']]}")
    print(f"  validation losses resumed  {[round(e['eval_loss'], 6) for e in resumed['evaluations']]}")
    print(f"  update 30 train loss {straight['log'][-1]['train_loss']:.6f} vs "
          f"{resumed['log'][-1]['train_loss']:.6f}")
    shutil.rmtree(scratch)
    if problems:
        print("STOP: the resumed run differs:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("  RESUME TEST PASSED: every update's losses, the validation "
          "losses and every actor and critic weight match exactly")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c round 1: reward sweep from the direct model.")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_UPDATE_M4)
    parser.add_argument("--runs", type=str, nargs="+", default=list(RUNS),
                        choices=list(RUNS))
    parser.add_argument("--resume", action="store_true",
                        help="continue each run from its latest checkpoint")
    parser.add_argument("--resume-test", action="store_true",
                        help="only run the resume test")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        report(args.out_dir)
        return

    print_settings(args.episodes)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        if args.resume_test:
            resume_test(pool, args.out_dir, args.episodes)
            return
        for run in args.runs:
            w, v = RUNS[run]
            ckpt_dir = os.path.join(args.out_dir, CHECKPOINT_DIR, run)
            existing = checkpoint_paths(ckpt_dir)
            if args.resume and existing:
                s = load_checkpoint(existing[-1])
                print(f"\n=== {run}: resumed from {existing[-1]} "
                      f"(update {s['update']}) ===", flush=True)
            else:
                if existing:
                    raise SystemExit(f"{ckpt_dir} already has checkpoints; "
                                     f"pass --resume or move them away")
                s = fresh_state(run, args.out_dir)
                print(f"\n=== {run}: w = {w}, v = {v}, fresh from the direct "
                      f"model ===", flush=True)
            if s["update"] < UPDATES:
                started = time.time()
                s = train(s, pool, w=w, v=v, until=UPDATES,
                          episodes=args.episodes, out_dir=args.out_dir,
                          ckpt_dir=ckpt_dir, model_tag=run,
                          on_progress=lambda st: write_curves(st,
                                                              args.out_dir))
                print(f"  {run}: {UPDATES} updates, best {s['best']:.2f}% at "
                      f"update {s['best_update']}, "
                      f"{(time.time() - started) / 3600:.2f} h", flush=True)
            write_curves(s, args.out_dir)
            plot_curves(args.out_dir)

            vpath = os.path.join(args.out_dir, "2c_r1_validation.csv")
            old = ([r for r in read_csv(vpath) if r["run"] != run]
                   if os.path.exists(vpath) else [])
            fresh = validate_best(run, pool, args.out_dir)
            mean = statistics.fmean(r["total_loss"] for r in fresh)
            print(f"  {run}: best actor re-validated at {mean:.4f}% "
                  f"(curve best {s['best']:.4f}%)", flush=True)
            if abs(mean - s["best"]) > 1e-9:
                raise SystemExit("STOP: the saved best actor does not "
                                 "reproduce its validation score")
            columns = (["run", "placement_seed", "run_seed", "total_loss"]
                       + list(CAUSES) + ["c_distance_flown_mean",
                                         "min_c_to_c_m", "views_agree"])
            ordered = [r for name in RUNS
                       for r in (fresh if name == run else
                                 [o for o in old if o["run"] == name])]
            write_csv(vpath, ordered, columns,
                      "phase 2c round 1, best actor per run, validation "
                      "301-330, D = 650")
            run_edge_cases(run, args.out_dir, args.workers)
            report(args.out_dir)


if __name__ == "__main__":
    main()
