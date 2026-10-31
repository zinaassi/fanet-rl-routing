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
from scripts.metrics_1a import (compute_metrics,
                                loss_by_time_window, mission_spread)
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

def _world(moving: bool = False, distance: Optional[float] = None):
    """Set the Phase-2c world and return the settings to put back.

    Args:
        moving:   True for M4's moving world, where the M-drones fly as they
                  do in Phase 2a. False, the default, keeps the static
                  mission M0-M3 ran on, so every one of those results stays
                  reproducible.
        distance: A mission distance in metres asks for the "near_gs"
                  curriculum mission at that D, and implies a moving world.
                  None, the default, leaves MISSION alone, so nothing before
                  the curriculum can see this argument at all.

    Returns:
        The settings _restore puts back.
    """
    previous = (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
                config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
                config.SHARE_TARGETS, config.AVOID, config.MISSION,
                config.MISSION_DISTANCE_M)
    config.C_RULE = "rl"
    config.STATIC_MODE = not moving    # the MISSION; C-drones always fly
    if distance is not None:
        config.MISSION = "near_gs"
        config.MISSION_DISTANCE_M = float(distance)
        config.STATIC_MODE = False
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
     config.SHARE_TARGETS, config.AVOID, config.MISSION,
     config.MISSION_DISTANCE_M) = previous


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
                 policy: str, greedy: bool,
                 moving: bool = False,
                 distance: Optional[float] = None) -> Dict[str, Any]:
    """Run one episode and return its transitions and its metrics.

    Args:
        layout:   Placement seed.
        run_seed: Channel and traffic seed, and the seed of this
                  episode's exploration stream.
        state:    Actor weights, for the "rl" policy.
        policy:   "rl", "midpoint", "hover" or "clone".
        greedy:   For "rl", take the mean instead of sampling.
        moving:   True for M4's moving world; False keeps M0-M3's static
                  mission.
        distance: A mission distance in metres runs the "near_gs"
                  curriculum mission at that D instead. None leaves the
                  mission alone.

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
    previous = _world(moving, distance)
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
            # Only M4 reports these, but they cost a pass over the packet
            # list beside a thousand simulated steps, so they are always
            # here rather than behind another flag.
            "pct_packets_connected": truth["pct_packets_connected"] or 0.0,
            "loss_while_connected": truth["loss_while_connected"] or 0.0,
            **{f"loss_{label}": value for label, value
               in loss_by_time_window(env).items()},
            # How far the mission itself got, which the curriculum reports
            # beside the loss: the same numbers step 1b printed.
            **mission_spread(env),
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
    """Worker entry point for one rollout.

    The job is M0-M3's 5-tuple, a 6-tuple whose last field asks for the
    moving world, or a 7-tuple whose last field is a mission distance in
    metres, which asks for the "near_gs" curriculum mission at that D.
    Accepting the short forms means no earlier caller had to change, so
    their jobs are still the jobs they were.

    The distance travels per JOB, not as a setting, because the curriculum's
    replay puts several stages inside one update.
    """
    layout, run_seed, state, policy, greedy = args[:5]
    moving = bool(args[5]) if len(args) > 5 else False
    distance = args[6] if len(args) > 6 else None
    return _run_episode(layout, run_seed, state, policy, greedy, moving,
                        distance)


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


def one_layout(rng: Any, episodes: int) -> List[int]:
    """Return the M2 layout list: the training layout, every episode.

    It takes *rng* and does not touch it, so the default draws nothing and
    M2's seed stream is exactly what it was.

    Args:
        rng:      The seed's generator. Unused.
        episodes: How many episodes this update runs.

    Returns:
        One layout per episode.
    """
    return [TRAIN_LAYOUT] * episodes


def layout_eval_jobs(state: Dict[str, Any]) -> List[Tuple]:
    """Return M2's evaluation: the training layout on the fixed run seeds.

    Args:
        state: The frozen actor weights.

    Returns:
        One greedy rollout job per evaluation seed.
    """
    return [(TRAIN_LAYOUT, int(s), state, "rl", True) for s in EVAL_RUN_SEEDS]


def train_one_seed(seed: int, pool: Any, out_dir: str,
                   max_updates: int, episodes: int, *,
                   layouts: Any = one_layout,
                   eval_jobs: Any = layout_eval_jobs,
                   min_improvement: float = MIN_IMPROVEMENT,
                   patience: int = PATIENCE,
                   model_name: Optional[str] = None,
                   progress: Any = None,
                   moving: bool = False,
                   min_updates: int = 0,
                   anchor: bool = True,
                   start_state: Optional[Dict[str, Any]] = None,
                   critic_name: Optional[str] = None,
                   stage_schedule: Any = None,
                   carry: Optional[Dict[str, Any]] = None,
                   warmup: int = CRITIC_ONLY_UPDATES) -> Dict[str, Any]:
    """Train one actor from the clone, and keep the best version.

    Args:
        seed:        Seeds the PPO part — the network's extra weights, the
                     exploration and the run seeds.
        pool:        A process pool for the rollouts, or None to run here.
        out_dir:     Where the model goes.
        max_updates: Cap on updates.
        episodes:    Episodes per update.
        layouts:     (rng, episodes) -> one placement seed per episode. The
                     default is M2's: the training layout every time, drawing
                     nothing from the generator.
        eval_jobs:   state -> the rollout jobs one evaluation runs. The
                     default is M2's: the training layout on EVAL_RUN_SEEDS.
        min_improvement: Points of total loss a version must beat the best by
                     to be kept.
        patience:    Evaluations without a keep before stopping.
        model_name:  File name under out/models/, or None for M2's.
        progress:    Called with (log rows, evaluation rows) after every
                     update, so an interrupted run keeps what it has.
        moving:      Run the episodes in M4's moving world.
        min_updates: Never stop early before this update. M4 asks for this
                     because the moving world is harder and an actor that
                     has plateaued at update 100 may still be climbing out
                     of it at 300.
        anchor:      Keep the fading stay-near-midpoint term. M4's main runs
                     start from an actor that already knows the task, so
                     pulling it back towards midpoint would only undo that.
        start_state: Weights to start the actor from, the whole Gaussian
                     actor. None, the default, is M2's: the clone loaded
                     into the mean and a fresh log_std.
        critic_name: File name under out/models/ for the critic, saved
                     whenever the actor is. None keeps the critic unsaved,
                     which is what M1-M3 did.
        stage_schedule: (update, episodes) -> one mission distance per
                     episode, for the curriculum, whose replay mixes stages
                     inside a single update. None, the default, hands every
                     job None and leaves the mission exactly as M0-M4 had it.
        carry:       The actor, critic, normaliser and optimisers of the
                     previous stage, so a curriculum continues instead of
                     starting over. None, the default, builds them fresh.
        warmup:      Updates during which only the critic learns. A later
                     curriculum stage passes 0, because its critic is
                     already warm and freezing the actor again would waste
                     the stage's first ten updates.

    Returns:
        The log rows, the evaluation rows and where the best model went.
    """
    import torch
    from agents.c_actor import CCritic, GaussianActor, RunningNorm, to_tensors

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if carry is not None:
        # A later stage of a curriculum: the actor, the critic, the value
        # normaliser and both optimisers continue from where the previous
        # stage left them. A fresh critic here would throw away everything it
        # learned and the actor would train on noise for ten updates again.
        actor = carry["actor"]
        critic = carry["critic"]
        normaliser = carry["normaliser"]
        actor_opt = carry["actor_opt"]
        critic_opt = carry["critic_opt"]
    else:
        actor = GaussianActor()
        if start_state is None:
            clone_path = os.path.join(out_dir, "models", "c_actor_bc.pt")
            actor.mean.load_state_dict(
                torch.load(clone_path, weights_only=True))
        else:
            actor.load_state_dict(start_state)
            # A continuation inherits the spread the earlier stage annealed
            # down to, which is confidence earned in a world this run is not
            # flying. config.CONTINUE_STD = a float puts it back; "inherit"
            # keeps it, the behaviour M4 runs (a) and (b) ran under.
            if config.CONTINUE_STD != "inherit":
                with torch.no_grad():
                    actor.log_std.fill_(float(np.log(config.CONTINUE_STD)))
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
    model_path = os.path.join(
        out_dir, "models", model_name or f"c_actor_ppo_seed{seed}.pt"
    )
    critic_path = (os.path.join(out_dir, "models", critic_name)
                   if critic_name else None)

    for update in range(1, max_updates + 1):
        started = time.perf_counter()
        state = {k: v.clone() for k, v in actor.state_dict().items()}
        run_seeds = rng.integers(1, 10 ** 8, size=episodes).tolist()
        # Run seeds FIRST, then the layouts: the default sampler draws
        # nothing, so M2's stream is bit-for-bit what it was.
        episode_layouts = layouts(rng, episodes)
        # The curriculum hands back one mission distance per episode, because
        # its replay puts several stages inside a single update. Without a
        # schedule every job is None, which is M0-M4's world untouched.
        distances = (stage_schedule(update, episodes)
                     if stage_schedule is not None else [None] * episodes)
        jobs = [(int(layout), int(s), state, "rl", False, moving, d)
                for layout, s, d in zip(episode_layouts, run_seeds, distances)]
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

        weight = anchor_weight(update) if anchor else 0.0
        train_actor = update > warmup
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

        if progress is not None:
            progress(log, evaluations)

        if update % EVALUATE_EVERY:
            continue

        marker = ""
        eval_state = {k: v.clone() for k, v in actor.state_dict().items()}
        jobs_now = eval_jobs(eval_state)
        eval_results = (list(pool.map(_rollout_job, jobs_now))
                        if pool is not None
                        else [_rollout_job(job) for job in jobs_now])
        # The training rollouts above already refuse to learn from a run
        # whose ACK view disagreed with the simulator. The evaluation is the
        # number that gets reported, so it is checked too.
        agreed = sum(1 for r in eval_results if r["views_agree"])
        if agreed != len(eval_results):
            raise RuntimeError(
                f"views disagreed in {len(eval_results) - agreed} of "
                f"{len(eval_results)} evaluation runs at update {update}"
            )
        score = statistics.fmean(
            [r["total_loss"] for r in eval_results]
        ) * 100
        evaluations.append({
            "seed": seed, "update": update, "eval_loss": score,
            "saved": False, "views_agree": f"{agreed}/{len(eval_results)}",
        })
        if score < best - min_improvement:
            best, best_update = score, update
            last_improved = len(evaluations)
            torch.save(actor.state_dict(), model_path)
            if critic_path is not None:
                # Saved with the actor, never on its own: a critic that does
                # not belong to the kept actor would be worse than none.
                torch.save(critic.state_dict(), critic_path)
            evaluations[-1]["saved"] = True
            marker = "  <- best, saved"
        elif score < best:
            # Better, but not by enough to keep: best_update must stay on the
            # version that was actually SAVED, or the two no longer describe
            # the same model.
            best = score
        print(f"  seed {seed} update {update:>3}  EVALUATION "
              f"{score:6.2f}%{marker}", flush=True)

        if progress is not None:
            progress(log, evaluations)

        if update < min_updates:
            continue
        if len(evaluations) - last_improved >= patience:
            print(f"  seed {seed}: early stop — {patience} checks without "
                  f"a {min_improvement} point gain", flush=True)
            break

    kept = [row for row in evaluations if row["saved"]]
    return {"log": log, "evaluations": evaluations, "model": model_path,
            "best_update": best_update, "best": best,
            "updates_used": len(log),
            # The live objects, so a curriculum's next stage can go on from
            # here instead of starting its critic cold.
            "carry": {"actor": actor, "critic": critic,
                      "normaliser": normaliser, "actor_opt": actor_opt,
                      "critic_opt": critic_opt},
            "kept": kept[-1]["eval_loss"] if kept else float("nan")}


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
                  baselines: Dict[str, float], path: str,
                  title: Optional[str] = None,
                  names: Optional[Sequence[str]] = None) -> None:
    """Draw evaluation loss against update, one line per training seed.

    Args:
        evaluations: Every evaluation row.
        baselines:   Reference losses to draw as horizontal lines.
        path:        Output PNG.
        title:       Heading, or None for M2's.
        names:       One label per line, in seed order, for callers whose
                     lines are not training seeds. M4's are its four runs.
                     None, the default, labels them "training seed N".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scripts.stamp import stamp_lines

    surface, grid = "#fcfcfb", "#e4e3dd"
    text, faint = "#2b2a26", "#5c5b54"
    seed_colours = {0: "#2a78d6", 1: "#1baf7a", 2: "#b5509c", 3: "#d98b1f"}
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

    for index, seed in enumerate(sorted({row["seed"] for row in evaluations})):
        rows = [r for r in evaluations if r["seed"] == seed]
        updates = [r["update"] for r in rows]
        losses = [r["eval_loss"] for r in rows]
        colour = seed_colours.get(seed, faint)
        label = (names[index] if names is not None and index < len(names)
                 else f"training seed {seed}")
        ax.plot(updates, losses, color=colour, linewidth=2.2, marker="o",
                markersize=5, label=label, zorder=4)
        kept = [r for r in rows if r["saved"]]
        if kept:
            final = kept[-1]
            ax.scatter([final["update"]], [final["eval_loss"]], marker="*",
                       s=420, color=colour, edgecolors=surface,
                       linewidths=1.0, zorder=7)

    ax.set_xlabel("PPO updates", color=faint, fontsize=11)
    ax.set_ylabel("Total packet loss (%)\nlower is better", color=faint,
                  fontsize=11)
    ax.set_title(title or (f"Phase 2c M2 — the actor on layout "
                           f"{TRAIN_LAYOUT}, without exploration"),
                 color=text, fontsize=13,
                 fontweight="bold", pad=12, loc="left")
    ax.grid(True, color=grid, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=faint, labelsize=9)
    # Lower left: the curves all start high and fall to the right, and the
    # baseline labels own the upper right.
    ax.legend(loc="lower left", frameon=False, fontsize=10,
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
            # The KEPT version is the one on disk and the one compared
            # below; the lowest score seen may belong to a later version that
            # did not improve by enough to be saved.
            print(f"  seed {seed}: kept {result['kept']:.2f}% at update "
                  f"{result['best_update']}  (lowest seen "
                  f"{result['best']:.2f}%)", flush=True)

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
