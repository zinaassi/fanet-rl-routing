"""
train_c_ppo.py — Phase 2c M2: first RL training of the C-drones.

MAPPO on one layout. Seven C-drones share one actor and learn together; a
centralised critic reads the whole network's state and is used ONLY while
training, so the drones themselves still decide from their own neighbour
tables alone.

The run is shaped to keep the first attempt from falling over:

* the actor STARTS as the M1 clone of midpoint, so it begins somewhere
  sensible rather than at random;
* the first ``CRITIC_ONLY_UPDATES`` updates train the critic alone, so the
  actor is not pushed around by a critic that knows nothing yet;
* for the next ``ANCHOR_UPDATES`` the actor carries a "stay near midpoint"
  term, faded linearly to nothing, so it lets go of the rule gradually
  instead of all at once.

Keep-best works like Phase 1b: freeze every ``EVALUATE_EVERY`` updates, run
without exploration on fixed seeds, and keep the version that beats the best
so far by at least ``MIN_IMPROVEMENT`` points.

Writes:
    out/2c_ppo_log.csv          one row per update
    out/2c_ppo_eval.csv         one row per evaluation
    out/models/c_actor_ppo_seed<N>.pt
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

# --- the problem ----------------------------------------------------------
TRAIN_LAYOUT: int = 101
EVAL_RUN_SEEDS: Tuple[int, ...] = (9001, 9002, 9003)
LOAD_MS: int = 200
ROUTING: str = "greedy"

#: Every table holds at most the other 24 drones plus the GS.
MAX_NEIGHBOURS: int = 25

# --- PPO ------------------------------------------------------------------
GAMMA: float = 0.97
GAE_LAMBDA: float = 0.95
CLIP: float = 0.1
EPOCHS_PER_UPDATE: int = 5
LEARNING_RATE: float = 3e-4
GRAD_CLIP: float = 0.5
EPISODES_PER_UPDATE: int = 20
MINIBATCH: int = 1024

CRITIC_ONLY_UPDATES: int = 10
ANCHOR_UPDATES: int = 50
ANCHOR_WEIGHT: float = 1.0

MAX_UPDATES: int = 300
EVALUATE_EVERY: int = 10
PATIENCE: int = 5
MIN_IMPROVEMENT: float = 0.5        # percentage points of total loss

CAUSES: Tuple[str, ...] = (
    "channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit",
    "link_broken", "stale_link",
)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

def _world():
    """Set the M0-M3 world and return the settings to put back."""
    previous = (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
                config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
                config.SHARE_TARGETS, config.AVOID)
    config.C_RULE = "rl"
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


# ---------------------------------------------------------------------------
# The agent that collects a rollout
# ---------------------------------------------------------------------------

class SamplingAgent:
    """Samples waypoints from the actor, and records what midpoint wanted.

    The midpoint target is not used to decide anything — it is recorded so
    the update can pull the actor toward it while the anchor term is still
    alive.

    Args:
        env:    The environment.
        actor:  The Gaussian actor.
        greedy: True to take the mean instead of sampling.
    """

    def __init__(self, env: Any, actor: Any, greedy: bool = False) -> None:
        self.env = env
        self.actor = actor
        self.greedy = greedy
        self.log_probs: List[float] = []
        self.anchors: List[np.ndarray] = []

    def decide(self, observations):
        """Return one action per observation, recording the extras.

        Args:
            observations: One per C-drone, in drone-id order.

        Returns:
            One (2,) action per observation.
        """
        import torch
        from agents.c_actor import to_tensors

        if not observations:
            return []
        batch = c_interface.pad_observations(list(observations),
                                             width=MAX_NEIGHBOURS)
        own, neighbours, mask = to_tensors(batch)
        with torch.no_grad():
            distribution = self.actor.distribution(own, neighbours, mask)
            actions = (distribution.mean if self.greedy
                       else distribution.sample())
            log_prob = distribution.log_prob(actions).sum(-1)

        self.log_probs.extend(log_prob.numpy().tolist())
        self._record_anchors()
        return [row.astype(np.float64) for row in actions.numpy()]

    def _record_anchors(self) -> None:
        """Record what the midpoint rule would have chosen, as an action."""
        previous = config.C_RULE
        config.C_RULE = "midpoint"
        try:
            for drone in self.env.drones:
                if drone.drone_type != "C":
                    continue
                target = self.env._choose_target(drone)
                if target is None:
                    target = drone.position
                self.anchors.append(
                    c_interface.action_from_waypoint(
                        target, self.env.gs_position
                    )
                )
        finally:
            config.C_RULE = previous


# ---------------------------------------------------------------------------
# Running one episode
# ---------------------------------------------------------------------------

def _run_episode(layout: int, run_seed: int, state: Optional[Dict[str, Any]],
                 policy: str, greedy: bool) -> Dict[str, Any]:
    """Run one episode and return its transitions and its metrics.

    Args:
        layout:   Placement seed.
        run_seed: Channel and traffic seed, and the seed of this
                  episode's exploration stream.
        state:    Actor weights, for the "rl" policy.
        policy:   "rl", "midpoint", "hover" or "clone".
        greedy:   For "rl", take the mean instead of sampling.

    Returns:
        The transitions (empty unless "rl") and the run's metrics.
    """
    import torch

    torch.set_num_threads(1)
    # The actor draws its exploration noise from torch's global RNG, and this
    # runs in a SPAWNED worker, which would otherwise seed that RNG at random
    # and make the whole training run irreproducible. run_seed is already
    # unique per (training seed, update, episode); the agent tag keeps this
    # stream apart from the environment's channel and traffic streams, the
    # same convention the Phase-1b agent follows.
    torch.manual_seed(int(np.random.SeedSequence(
        [run_seed, config.AGENT_STREAM_TAG]
    ).generate_state(1)[0]))
    previous = _world()
    try:
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed)
        env.reset()

        agent: Any
        if policy == "hover":
            agent = HoverAgent(env)
        elif policy == "midpoint":
            agent = MidpointAgent(env)
        else:
            from agents.c_actor import CActor, GaussianActor, WaypointAgent

            if policy == "clone":
                actor = CActor()
                actor.load_state_dict(state)
                agent = WaypointAgent(actor)
            else:
                actor = GaussianActor()
                actor.load_state_dict(state)
                actor.eval()
                agent = SamplingAgent(env, actor, greedy=greedy)
        env.c_agent = agent

        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        out: Dict[str, Any] = {
            "total_loss": truth["total_loss"] or 0.0,
            **{cause: truth["loss_rate_by_reason"][cause] for cause in CAUSES},
            "c_distance_flown_mean": truth["c_distance_flown_mean"],
            "c_gs_distance_mean": (
                statistics.fmean(truth["c_distance_from_gs"])
                if truth["c_distance_from_gs"] else 0.0
            ),
            "views_agree": not metrics["discrepancies"],
            "mean_reward": statistics.fmean(
                [r["reward"] for r in env.c_decisions
                 if r["reward"] is not None] or [0.0]
            ),
        }

        if policy == "rl" and not greedy:
            out.update(_transitions(env, agent))
        return out
    finally:
        _restore(previous)


def _transitions(env: Any, agent: SamplingAgent) -> Dict[str, Any]:
    """Pack one episode's decisions into arrays, with returns and advantages.

    Each C-drone is its own trajectory: the reward that follows a decision is
    the one that decision earned. The last decision of each drone has no
    reward yet — the run ended before its second did — so it is dropped.

    Args:
        env:   The finished environment.
        agent: The agent that drove it, for the log-probs and anchors.

    Returns:
        Arrays ready for the update.
    """
    rows = env.c_decisions
    by_drone: Dict[int, List[int]] = {}
    for index, row in enumerate(rows):
        by_drone.setdefault(row["drone_id"], []).append(index)

    keep: List[int] = []
    for indices in by_drone.values():
        keep.extend(i for i in indices if rows[i]["reward"] is not None)
    keep.sort()

    observations = [rows[i]["observation"] for i in keep]
    batch = c_interface.pad_observations(observations, width=MAX_NEIGHBOURS)
    return {
        "self": batch["self"],
        "neighbours": batch["neighbours"],
        "mask": batch["mask"],
        "global": np.stack([rows[i]["global_state"] for i in keep]),
        "action": np.stack([rows[i]["action"] for i in keep]),
        "log_prob": np.array([agent.log_probs[i] for i in keep]),
        "anchor": np.stack([agent.anchors[i] for i in keep]),
        "reward": np.array([rows[i]["reward"] for i in keep]),
        "drone": np.array([rows[i]["drone_id"] for i in keep]),
        "order": np.array(keep),
    }


def _rollout_job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one rollout."""
    layout, run_seed, state, policy, greedy = args
    return _run_episode(layout, run_seed, state, policy, greedy)


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def advantages(rewards: np.ndarray, values: np.ndarray,
               drones: np.ndarray, order: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Return GAE advantages and returns, one trajectory per drone.

    Args:
        rewards: Per decision.
        values:  The critic's estimate per decision.
        drones:  Which drone each decision belongs to.
        order:   The decision's position in the episode, for sorting.

    Returns:
        ``(advantages, returns)``, in the order given.
    """
    advantage = np.zeros_like(rewards)
    for drone_id in np.unique(drones):
        picked = np.where(drones == drone_id)[0]
        picked = picked[np.argsort(order[picked])]
        running = 0.0
        for position in reversed(range(len(picked))):
            index = picked[position]
            nxt = (values[picked[position + 1]]
                   if position + 1 < len(picked) else 0.0)
            delta = rewards[index] + GAMMA * nxt - values[index]
            running = delta + GAMMA * GAE_LAMBDA * running
            advantage[index] = running
    return advantage, advantage + values


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def anchor_weight(update: int) -> float:
    """Return the weight of the stay-near-midpoint term at this update.

    Full weight while the critic is still warming up, then faded linearly to
    nothing over ``ANCHOR_UPDATES``.

    Args:
        update: 1-based update number.

    Returns:
        A weight in [0, ANCHOR_WEIGHT].
    """
    if update <= CRITIC_ONLY_UPDATES:
        return ANCHOR_WEIGHT
    done = update - CRITIC_ONLY_UPDATES
    if done >= ANCHOR_UPDATES:
        return 0.0
    return ANCHOR_WEIGHT * (1.0 - done / ANCHOR_UPDATES)


def train_one_seed(seed: int, pool: Any, out_dir: str,
                   max_updates: int, episodes: int) -> Dict[str, Any]:
    """Train one actor from the clone, and keep the best version.

    Args:
        seed:        Seeds the PPO part — the network's extra weights, the
                     exploration and the run seeds.
        pool:        A process pool for the rollouts, or None to run here.
        out_dir:     Where the model goes.
        max_updates: Cap on updates.
        episodes:    Episodes per update.

    Returns:
        The log rows, the evaluation rows and where the best model went.
    """
    import torch
    from agents.c_actor import CCritic, GaussianActor, RunningNorm, to_tensors

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    actor = GaussianActor()
    clone_path = os.path.join(out_dir, "models", "c_actor_bc.pt")
    actor.mean.load_state_dict(torch.load(clone_path, weights_only=True))
    critic = CCritic(c_interface.global_state_width(
        config.NUM_M_DRONES + config.NUM_C_DRONES
    ))
    normaliser = RunningNorm()

    actor_opt = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    log: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    best = float("inf")
    best_update = 0
    last_improved = 0
    model_path = os.path.join(out_dir, "models", f"c_actor_ppo_seed{seed}.pt")

    for update in range(1, max_updates + 1):
        started = time.perf_counter()
        state = {k: v.clone() for k, v in actor.state_dict().items()}
        run_seeds = rng.integers(1, 10 ** 8, size=episodes).tolist()
        jobs = [(TRAIN_LAYOUT, int(s), state, "rl", False) for s in run_seeds]
        results = (list(pool.map(_rollout_job, jobs)) if pool is not None
                   else [_rollout_job(job) for job in jobs])

        if not all(r["views_agree"] for r in results):
            raise RuntimeError(f"views disagreed during update {update}")

        batch = {
            key: np.concatenate([r[key] for r in results])
            for key in ("self", "neighbours", "mask", "global", "action",
                        "log_prob", "anchor", "reward", "drone", "order")
        }
        # Decisions from different episodes must not share a trajectory, so
        # the drone id is made unique per episode before the GAE pass.
        offsets = np.concatenate([
            np.full(len(r["reward"]), index * 1000)
            for index, r in enumerate(results)
        ])
        trajectories = batch["drone"] + offsets

        own = torch.as_tensor(batch["self"], dtype=torch.float32)
        neighbours = torch.as_tensor(batch["neighbours"], dtype=torch.float32)
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32)
        global_state = torch.as_tensor(batch["global"], dtype=torch.float32)
        actions = torch.as_tensor(batch["action"], dtype=torch.float32)
        old_log_prob = torch.as_tensor(batch["log_prob"], dtype=torch.float32)
        anchors = torch.as_tensor(batch["anchor"], dtype=torch.float32)

        with torch.no_grad():
            values = normaliser.denormalise(
                critic(global_state).numpy().astype(np.float64)
            )
        advantage, returns = advantages(batch["reward"], values,
                                        trajectories, batch["order"])
        normaliser.update(returns)
        target = torch.as_tensor(normaliser.normalise(returns),
                                 dtype=torch.float32)
        advantage_t = torch.as_tensor(
            (advantage - advantage.mean()) / (advantage.std() + 1e-8),
            dtype=torch.float32,
        )

        weight = anchor_weight(update)
        train_actor = update > CRITIC_ONLY_UPDATES
        policy_loss = value_loss = anchor_loss = 0.0

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
                    own[picked], neighbours[picked], mask[picked]
                )
                log_prob = distribution.log_prob(actions[picked]).sum(-1)
                ratio = (log_prob - old_log_prob[picked]).exp()
                clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)
                p_loss = -torch.min(ratio * advantage_t[picked],
                                    clipped * advantage_t[picked]).mean()
                a_loss = ((distribution.mean - anchors[picked]) ** 2).sum(-1).mean()
                total = p_loss + weight * a_loss

                actor_opt.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                actor_opt.step()
                policy_loss = float(p_loss.detach())
                anchor_loss = float(a_loss.detach())

        row = {
            "seed": seed,
            "update": update,
            "train_loss": statistics.fmean(
                [r["total_loss"] for r in results]
            ) * 100,
            "mean_reward": statistics.fmean([r["mean_reward"] for r in results]),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "anchor_loss": anchor_loss,
            "anchor_weight": weight,
            "std": float(actor.log_std.exp().mean().detach()),
            "decisions": len(target),
            "seconds": time.perf_counter() - started,
        }
        log.append(row)
        print(f"  seed {seed} update {update:>3}  "
              f"train {row['train_loss']:6.2f}%  reward {row['mean_reward']:+.4f}  "
              f"anchor {weight:.2f}  std {row['std']:.3f}  "
              f"{row['seconds']:5.1f}s", flush=True)

        if update % EVALUATE_EVERY:
            continue

        marker = ""
        eval_state = {k: v.clone() for k, v in actor.state_dict().items()}
        eval_jobs = [(TRAIN_LAYOUT, int(s), eval_state, "rl", True)
                     for s in EVAL_RUN_SEEDS]
        eval_results = (list(pool.map(_rollout_job, eval_jobs))
                        if pool is not None
                        else [_rollout_job(job) for job in eval_jobs])
        score = statistics.fmean(
            [r["total_loss"] for r in eval_results]
        ) * 100
        evaluations.append({
            "seed": seed, "update": update, "eval_loss": score,
            "saved": False,
        })
        if score < best - MIN_IMPROVEMENT:
            best, best_update = score, update
            last_improved = len(evaluations)
            torch.save(actor.state_dict(), model_path)
            evaluations[-1]["saved"] = True
            marker = "  <- best, saved"
        elif score < best:
            best = score
        print(f"  seed {seed} update {update:>3}  EVALUATION "
              f"{score:6.2f}%{marker}", flush=True)

        if len(evaluations) - last_improved >= PATIENCE:
            print(f"  seed {seed}: early stop — {PATIENCE} checks without "
                  f"a {MIN_IMPROVEMENT} point gain", flush=True)
            break

    return {"log": log, "evaluations": evaluations, "model": model_path,
            "best_update": best_update, "best": best}


# ---------------------------------------------------------------------------
# Comparison, plot and entry point
# ---------------------------------------------------------------------------

def compare(models: Dict[int, str], pool: Any) -> List[Dict[str, Any]]:
    """Run every policy on the training layout with the fixed seeds.

    Args:
        models: Training seed -> path of the kept actor.
        pool:   A process pool, or None.

    Returns:
        One row per (policy, run seed).
    """
    import torch

    clone = torch.load(os.path.join("out", "models", "c_actor_bc.pt"),
                       weights_only=True)
    jobs: List[Tuple] = []
    labels: List[str] = []
    for run_seed in EVAL_RUN_SEEDS:
        jobs.append((TRAIN_LAYOUT, run_seed, None, "hover", True))
        labels.append("hover")
        jobs.append((TRAIN_LAYOUT, run_seed, None, "midpoint", True))
        labels.append("midpoint")
        jobs.append((TRAIN_LAYOUT, run_seed, clone, "clone", True))
        labels.append("clone")
        for seed, path in sorted(models.items()):
            if not os.path.exists(path):
                continue
            jobs.append((TRAIN_LAYOUT, run_seed,
                         torch.load(path, weights_only=True), "rl", True))
            labels.append(f"rl seed {seed}")

    print(f"\nComparing {len(jobs)} runs on layout {TRAIN_LAYOUT} ...")
    results = (list(pool.map(_rollout_job, jobs)) if pool is not None
               else [_rollout_job(job) for job in jobs])
    rows = []
    for (layout, run_seed, _, _, _), label, result in zip(jobs, labels, results):
        rows.append({"policy": label, "run_seed": run_seed, **result})
    if not all(r["views_agree"] for r in rows):
        print("STOP: views disagreed in the comparison")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def print_comparison(rows: Sequence[Dict[str, Any]]) -> None:
    """Print the comparison table, one line per policy."""
    order: List[str] = []
    for row in rows:
        if row["policy"] not in order:
            order.append(row["policy"])
    width = 118
    print(f"\n  Layout {TRAIN_LAYOUT}, mean of run seeds "
          f"{', '.join(str(s) for s in EVAL_RUN_SEEDS)} — % of packets created")
    print("  " + "-" * width)
    print(f"  {'policy':>12}{'total':>8}" + "".join(f"{c[:9]:>11}" for c in CAUSES)
          + f"{'flown':>9}{'from GS':>9}")
    print("  " + "-" * width)
    for policy in order:
        chosen = [r for r in rows if r["policy"] == policy]
        print(f"  {policy:>12}{statistics.fmean([r['total_loss'] for r in chosen]) * 100:7.2f}%"
              + "".join(
                  f"{statistics.fmean([r[c] for r in chosen]) * 100:10.2f}%"
                  for c in CAUSES
              )
              + f"{statistics.fmean([r['c_distance_flown_mean'] for r in chosen]):9.0f}"
              f"{statistics.fmean([r['c_gs_distance_mean'] for r in chosen]):9.0f}")
    print("  " + "-" * width)


def plot_learning(evaluations: Sequence[Dict[str, Any]],
                  baselines: Dict[str, float], path: str) -> None:
    """Draw evaluation loss against update, one line per training seed.

    Args:
        evaluations: Every evaluation row.
        baselines:   Reference losses to draw as horizontal lines.
        path:        Output PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.stamp import stamp_lines

    surface, grid = "#fcfcfb", "#e4e3dd"
    text, faint = "#2b2a26", "#5c5b54"
    seed_colours = {0: "#2a78d6", 1: "#1baf7a", 2: "#b5509c"}
    reference = {"hover": "#eb6834", "midpoint": "#4a3aa7",
                 "clone": "#e34948"}

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for label, value in baselines.items():
        ax.axhline(value, color=reference.get(label, faint), linestyle="--",
                   linewidth=2.0, zorder=3)
        ax.annotate(f"{label}  {value:.1f}%",
                    (0.995, value), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=11, fontweight="bold",
                    color=reference.get(label, faint), zorder=6)

    for seed in sorted({row["seed"] for row in evaluations}):
        rows = [r for r in evaluations if r["seed"] == seed]
        updates = [r["update"] for r in rows]
        losses = [r["eval_loss"] for r in rows]
        colour = seed_colours.get(seed, faint)
        ax.plot(updates, losses, color=colour, linewidth=2.2, marker="o",
                markersize=5, label=f"training seed {seed}", zorder=4)
        kept = [r for r in rows if r["saved"]]
        if kept:
            final = kept[-1]
            ax.scatter([final["update"]], [final["eval_loss"]], marker="*",
                       s=420, color=colour, edgecolors=surface,
                       linewidths=1.0, zorder=7)

    ax.set_xlabel("PPO updates", color=faint, fontsize=11)
    ax.set_ylabel("Total packet loss (%)\nlower is better", color=faint,
                  fontsize=11)
    ax.set_title(f"Phase 2c M2 — the actor on layout {TRAIN_LAYOUT}, "
                 f"without exploration", color=text, fontsize=13,
                 fontweight="bold", pad=12, loc="left")
    ax.grid(True, color=grid, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=faint, labelsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=10,
              labelcolor=faint)
    # Room on the right for the baseline labels.
    ax.set_xlim(left=0)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.012, stamp_lines(width=112, loads_ms=[LOAD_MS]),
             color=faint, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    print(f"  wrote {path}")


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str], note: str) -> None:
    """Write *rows* to *path* under a settings stamp."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=[LOAD_MS], extra=note))
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})
    print(f"  wrote {path}")


def main() -> None:
    """Train the C-drone actor with PPO and compare it to the rules."""
    parser = argparse.ArgumentParser(
        description="Phase 2c M2: MAPPO on one layout."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_UPDATE)
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
    try:
        log: List[Dict[str, Any]] = []
        evaluations: List[Dict[str, Any]] = []
        models: Dict[int, str] = {}
        for seed in args.seeds:
            print(f"\n=== training seed {seed} ===", flush=True)
            result = train_one_seed(seed, pool, args.out_dir,
                                    args.updates, args.episodes)
            log.extend(result["log"])
            evaluations.extend(result["evaluations"])
            models[seed] = result["model"]
            print(f"  seed {seed}: best {result['best']:.2f}% at update "
                  f"{result['best_update']}", flush=True)

        rows = compare(models, pool)
    finally:
        if pool is not None:
            pool.shutdown()

    print_comparison(rows)
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
                  os.path.join(args.out_dir, "2c_ppo_learning.png"))

    note = (f"phase 2c M2, MAPPO on layout {TRAIN_LAYOUT}, static mission, "
            f"hello 1 s, {ROUTING} routing")
    write_csv(os.path.join(args.out_dir, "2c_ppo_log.csv"), log,
              ("seed", "update", "train_loss", "mean_reward", "policy_loss",
               "value_loss", "anchor_loss", "anchor_weight", "std",
               "decisions", "seconds"), note)
    write_csv(os.path.join(args.out_dir, "2c_ppo_eval.csv"), evaluations,
              ("seed", "update", "eval_loss", "saved"), note)
    write_csv(os.path.join(args.out_dir, "2c_ppo_compare.csv"), rows,
              ("policy", "run_seed", "total_loss", *CAUSES,
               "c_distance_flown_mean", "c_gs_distance_mean", "mean_reward",
               "views_agree"), note)


if __name__ == "__main__":
    main()
