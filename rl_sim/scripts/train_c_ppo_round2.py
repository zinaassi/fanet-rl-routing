"""
train_c_ppo_round2.py — Phase 2c round 2: fair reward and varied missions.

Three runs from the direct model, each against round 1's control (the direct
model plus 400 updates with nothing changed):

    A  fair reward,   full fleet only
    B  normal reward, varied missions
    C  fair reward,   varied missions

Same start, budget and PPO settings as round 1 (its functions are imported,
not retyped): the direct actor with its spread put back to CONTINUE_STD, the
direct critic, a fresh value normaliser and fresh optimisers, ten critic-only
warm-up updates, PPO seed 2, 400 updates, no early stop, a checkpoint every 25
updates and --resume.

THE FAIR REWARD (A, C). The delivery term of c_interface.flight_reward,
(delivered - lost) / created over the second, is replaced by

    k * mean over M-drones i of log(r_i + FAIR_FLOOR),
    r_i = min(1, delivered_i / created_i) over the last FAIR_WINDOW_S,

counted the way the simulator counts its delivery term: events in the window
[end - 100, end), where end is the step the reward is closed at (decision step
+ 10); delivered_i by delivered_at, created_i by created_at. The flight cost
is kept exactly: C_FLIGHT_COST * metres / speed, metres from distance_flown
between the decision and the close. Old reward + that flight cost must give
one and the same delivery term for every C-drone in a second, or the run
stops. FAIR_K is set so the fair term's std per second equals the old
delivery term's, measured by --measure on one full-fleet episode.

THE VARIED MISSIONS (B, C). Each training episode draws its world from its
own stream, SeedSequence([run_seed, VARIED_TAG]): half the time the normal
mission (near_gs, D = 650, 18 M + 7 C), half a varied world: M uniform 1-18,
C uniform 1-7, and a mission type, equal chance:
    a  spread        near_gs, D uniform 300-650
    b  lone flyers   min(M, 1-3) M-drones fly to points 450-622 m from the GS
    c  group         ceil(M/2) fly within +-20 deg of one heading, 450-622 m
    d  out and back  as b, then hover 10-30 s and return to their start
In b, c and d the other M-drones hover where they start; flyers keep the
mission's own speed (10-30 m/s) and start delay (0-50 s). Every far point is
redrawn until it lies inside the border margin, so its distance really is
450-622 m. Start points are the near_gs draw of the training layout.

THE CRITIC. The global state is always 25 rows: 18 M slots, then 7 C slots, a
row of zeros for an absent drone (its M/C flags both 0 mark it absent). Full
fleet: bit-identical to the unpadded vector, so the direct critic loads.

VALIDATION, every 10 updates, two sets:
    full    301-330, D = 650, run seed 9100 + (layout - 301)   (round 1's)
    varied  331-360, run seed 9200 + (layout - 331), types a, b, c, d in turn
Keep-best: the lowest AVERAGE of the two sets' mean total loss.

Writes:
    out/2c_r2_curve_<run>.csv, out/2c_r2_log_<run>.csv, out/2c_r2_curves.png
    out/models/2c/c_{actor,critic}_r2_<run>.pt
    out/2c_r2_varied_plans.csv        the 30 varied validation worlds
    out/2c_r2_references.csv          control, direct, midpoint, hover, both sets
    out/2c_r2_validation.csv          best actors, both sets, per layout
    out/2c_r2_summary.csv, out/2c_r2_by_type.csv, out/2c_r2_paired.csv
    out/2c_r2_edge.csv                13 edge cases
"""

from __future__ import annotations

import argparse
import csv
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
from fanet_sim.envs.c_scripted import HoverAgent, MidpointAgent
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics, measured_packets
from scripts.train_c_ppo import (
    CAUSES,
    CLIP,
    CRITIC_ONLY_UPDATES,
    EPOCHS_PER_UPDATE,
    EVALUATE_EVERY,
    GRAD_CLIP,
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
from scripts.train_c_ppo_round1 import (
    CHECKPOINT_EVERY,
    PPO_SEED,
    UPDATES,
    checkpoint_paths,
    fresh_state,
    load_checkpoint,
    read_csv,
    save_checkpoint,
)

#: Run -> (reward kind, varied missions), in the order they run.
RUNS: Dict[str, Tuple[str, bool]] = {
    "A": ("fair", False),
    "B": ("normal", True),
    "C": ("fair", True),
}

FAIR_FLOOR: float = 0.05
FAIR_WINDOW_S: float = 10.0
#: Set from --measure (layout 101, run seed 12345, direct actor, std 0.1):
#: std of the old delivery term / std of the unscaled fair term.
FAIR_K: Optional[float] = 1.4099498888140802   # 0.239344 / 0.169754
K_LAYOUT, K_RUN_SEED = 101, 12345

FULL_M, FULL_C = 18, 7
DISTANCE_M: float = 650.0
VARIED_SHARE: float = 0.5
VARIED_TAG: int = 0x2C2A
VARIED_VAL_TAG: int = 0x2C2B
TYPES: Tuple[str, ...] = ("a", "b", "c", "d")
SPREAD_D_M: Tuple[float, float] = (300.0, 650.0)
FAR_M: Tuple[float, float] = (450.0, 622.0)
GROUP_SPREAD_RAD: float = math.radians(20.0)
HOVER_S: Tuple[float, float] = (10.0, 30.0)
MAX_REDRAWS: int = 10000

VARIED_LAYOUTS: Tuple[int, ...] = tuple(range(331, 361))
VARIED_SEED_BASE: int = 9200

NORMAL_PLAN: Dict[str, Any] = {"kind": "normal", "n_m": FULL_M,
                               "n_c": FULL_C, "distance": DISTANCE_M}

CHECKPOINT_DIR: str = os.path.join("checkpoints", "2c_round2")
CONTROL_ACTOR = os.path.join("models", "2c", "c_actor_r1_control.pt")
DIRECT_ACTOR = os.path.join("models", "2c", "c_actor_curriculum_direct.pt")

T_95_DF29: float = 2.045
EDGE_GIF_SCENARIOS: Tuple[int, ...] = (1, 6, 9, 10, 13)


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

def _inside(point: np.ndarray) -> bool:
    m = config.BORDER_MARGIN_M
    return bool(m <= point[0] <= config.WIDTH - m
                and m <= point[1] <= config.HEIGHT - m)


def _far_point(rng: np.random.Generator, centre: Optional[float] = None
               ) -> List[float]:
    """A point FAR_M from the GS, inside the border; around *centre* (rad)
    within +-GROUP_SPREAD_RAD when given, anywhere otherwise."""
    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    for _ in range(MAX_REDRAWS):
        angle = (float(rng.uniform(0.0, 2.0 * math.pi)) if centre is None
                 else centre + float(rng.uniform(-GROUP_SPREAD_RAD,
                                                 GROUP_SPREAD_RAD)))
        distance = float(rng.uniform(*FAR_M))
        point = gs + distance * np.array([math.cos(angle), math.sin(angle)])
        if _inside(point):
            return [float(point[0]), float(point[1])]
    raise RuntimeError("no far point inside the border")


def draw_varied(rng: np.random.Generator, kind: Optional[str] = None
                ) -> Dict[str, Any]:
    """One varied world: fleet size, mission type and its flights."""
    n_m = int(rng.integers(1, FULL_M + 1))
    n_c = int(rng.integers(1, FULL_C + 1))
    if kind is None:
        kind = TYPES[int(rng.integers(0, len(TYPES)))]
    plan: Dict[str, Any] = {"kind": kind, "n_m": n_m, "n_c": n_c,
                            "distance": DISTANCE_M, "flyers": [],
                            "hover_s": []}
    if kind == "a":
        plan["distance"] = float(rng.uniform(*SPREAD_D_M))
    elif kind in ("b", "d"):
        count = min(n_m, int(rng.integers(1, 4)))
        plan["flyers"] = [_far_point(rng) for _ in range(count)]
        if kind == "d":
            plan["hover_s"] = [float(rng.uniform(*HOVER_S))
                               for _ in range(count)]
    elif kind == "c":
        heading = float(rng.uniform(0.0, 2.0 * math.pi))
        plan["flyers"] = [_far_point(rng, heading)
                          for _ in range(math.ceil(n_m / 2))]
        plan["heading_deg"] = math.degrees(heading)
    return plan


def training_plan(run_seed: int, varied: bool) -> Dict[str, Any]:
    """The world of one training episode."""
    if not varied:
        return NORMAL_PLAN
    rng = np.random.default_rng(np.random.SeedSequence([run_seed, VARIED_TAG]))
    if rng.random() < VARIED_SHARE:
        return NORMAL_PLAN
    return draw_varied(rng)


def varied_validation_plans() -> List[Tuple[int, int, Dict[str, Any]]]:
    """(layout, run seed, plan) for the thirty varied validation worlds."""
    out = []
    for index, layout in enumerate(VARIED_LAYOUTS):
        rng = np.random.default_rng(
            np.random.SeedSequence([layout, VARIED_VAL_TAG]))
        out.append((layout, VARIED_SEED_BASE + index,
                    draw_varied(rng, TYPES[index % len(TYPES)])))
    return out


# ---------------------------------------------------------------------------
# The critic's padded view, and the fair reward
# ---------------------------------------------------------------------------

def pad_global(state: np.ndarray, n_m: int, n_c: int) -> np.ndarray:
    """Pad a global state to 18 M slots then 7 C slots, zero rows absent."""
    per = c_interface.GLOBAL_PER_DRONE
    m_part = state[:n_m * per]
    c_part = state[n_m * per:(n_m + n_c) * per]
    tail = state[(n_m + n_c) * per:]
    return np.concatenate([m_part, np.zeros((FULL_M - n_m) * per),
                           c_part, np.zeros((FULL_C - n_c) * per), tail])


def fair_rewards(env: FANETEnv, flown: np.ndarray, k: float,
                 replace: bool) -> Dict[str, np.ndarray]:
    """Per closed decision: the old delivery term, the unscaled fair term and
    the flight cost; with *replace*, the reward becomes k * fair - flight.

    Args:
        env:     The finished environment.
        flown:   flown[s, d] = drone d's distance_flown after s steps.
        k:       The fair term's scale.
        replace: Write the new reward into env.c_decisions.

    Raises:
        RuntimeError: If old reward + flight cost is not one value per second.
    """
    interval = c_interface.steps_per_decision()
    window = int(round(FAIR_WINDOW_S / config.TIMESTEP))
    m_ids = [d.drone_id for d in env.drones if d.drone_type == "M"]
    created = {i: [] for i in m_ids}
    for p in env.all_packets:
        created[p.source_id].append(p.created_at)
    delivered = {i: [] for i in m_ids}
    for p in env.delivered:
        delivered[p.source_id].append(p.delivered_at)
    created = {i: np.sort(np.array(v)) for i, v in created.items()}
    delivered = {i: np.sort(np.array(v)) for i, v in delivered.items()}

    def count(times: np.ndarray, lo: int, hi: int) -> int:
        return int(np.searchsorted(times, hi, "left")
                   - np.searchsorted(times, lo, "left"))

    fair_at: Dict[int, float] = {}
    balance_at: Dict[int, float] = {}
    old, fair, flight = [], [], []
    for row in env.c_decisions:
        if row["reward"] is None:
            continue
        start, end = row["step"], row["step"] + interval
        drone = env.drones[row["drone_id"]]
        cost = (config.C_FLIGHT_COST
                * (flown[end, row["drone_id"]] - flown[start, row["drone_id"]])
                / drone.speed) if drone.speed > 0.0 else 0.0
        balance = row["reward"] + cost
        if end in balance_at:
            if abs(balance - balance_at[end]) > 1e-12:
                raise RuntimeError(f"delivery term differs between C-drones "
                                   f"at step {end}: {balance} vs "
                                   f"{balance_at[end]}")
        else:
            balance_at[end] = balance
        if end not in fair_at:
            lo = max(0, end - window)
            logs = []
            for i in m_ids:
                made = count(created[i], lo, end)
                share = (min(1.0, count(delivered[i], lo, end) / made)
                         if made else 1.0)
                logs.append(math.log(share + FAIR_FLOOR))
            fair_at[end] = statistics.fmean(logs)
        old.append(balance)
        fair.append(fair_at[end])
        flight.append(cost)
        if replace:
            row["reward"] = k * fair_at[end] - cost
    return {"old": np.array(old), "fair": np.array(fair),
            "flight": np.array(flight)}


# ---------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------

def run_episode(job: Tuple) -> Dict[str, Any]:
    """One episode in the world *plan*, under the actor or a rule.

    Args:
        job: (layout, run_seed, actor state or None, greedy, reward kind,
              plan, policy), policy "rl", "midpoint" or "hover".
    """
    import torch

    layout, run_seed, state, greedy, reward_kind, plan, policy = job
    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [run_seed, config.AGENT_STREAM_TAG]).generate_state(1)[0]))
    previous = _world(True, plan["distance"])
    counts = (config.NUM_M_DRONES, config.NUM_C_DRONES)
    config.NUM_M_DRONES, config.NUM_C_DRONES = plan["n_m"], plan["n_c"]
    try:
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed)
        env.reset()
        m_drones = [d for d in env.drones if d.drone_type == "M"]
        flyers = (plan.get("flyers", []) if plan["kind"] in ("b", "c", "d")
                  else [])
        if plan["kind"] in ("b", "c", "d"):
            for index, drone in enumerate(m_drones):
                if index < len(flyers):
                    drone.end_point = np.asarray(flyers[index],
                                                 dtype=np.float64).copy()
                    drone.arrived = False
                else:
                    drone.end_point = drone.position.copy()
                    drone.arrived = True
        starts = [d.position.copy() for d in m_drones[:len(flyers)]]
        hover_steps = [int(round(h / config.TIMESTEP))
                       for h in plan.get("hover_s", [])]
        arrived_at: List[Optional[int]] = [None] * len(hover_steps)
        turned = [False] * len(hover_steps)

        agent: Any
        if policy == "hover":
            agent = HoverAgent(env)
        elif policy == "midpoint":
            agent = MidpointAgent(env)
        else:
            from agents.c_actor import GaussianActor

            actor = GaussianActor()
            actor.load_state_dict(state)
            actor.eval()
            agent = SamplingAgent(env, actor, greedy=greedy)
        env.c_agent = agent

        positions = [np.array([d.position for d in env.drones])]
        flown = [np.array([d.distance_flown for d in env.drones])]
        for step in range(1, config.MAX_STEPS + 1):
            env.step()
            for j, wait in enumerate(hover_steps):
                drone = m_drones[j]
                if arrived_at[j] is None and drone.arrived:
                    arrived_at[j] = step
                if (not turned[j] and arrived_at[j] is not None
                        and step - arrived_at[j] == wait):
                    drone.end_point = starts[j].copy()
                    drone.arrived = False
                    turned[j] = True
            positions.append(np.array([d.position for d in env.drones]))
            flown.append(np.array([d.distance_flown for d in env.drones]))
        env.close_logger()
        positions = np.array(positions)
        flown = np.array(flown)

        terms = {"fair_term": 0.0}
        if reward_kind == "fair" and policy == "rl" and not greedy:
            sizes = fair_rewards(env, flown, FAIR_K, replace=True)
            terms["fair_term"] = float(FAIR_K * sizes["fair"].mean())

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        arrived = {id(p) for p in env.delivered}
        per_m = []
        for drone in m_drones:
            mine = [p for p in measured_packets(env)
                    if p.source_id == drone.drone_id]
            per_m.append(100.0 * sum(id(p) not in arrived for p in mine)
                         / len(mine) if mine else float("nan"))
        c_ids = [d.drone_id for d in env.drones if d.drone_type == "C"]
        min_cc = float("nan")
        if len(c_ids) > 1:
            c_pos = positions[:, c_ids, :]
            gaps = np.linalg.norm(c_pos[:, :, None, :] - c_pos[:, None, :, :],
                                  axis=-1)
            upper = np.triu_indices(len(c_ids), k=1)
            min_cc = float(gaps[:, upper[0], upper[1]].min())
        out: Dict[str, Any] = {
            "total_loss": truth["total_loss"] or 0.0,
            **{cause: truth["loss_rate_by_reason"][cause] for cause in CAUSES},
            "c_distance_flown_mean": truth["c_distance_flown_mean"],
            "min_c_to_c_m": min_cc,
            "worst_m_loss": max(per_m),
            "pct_m_over_half": 100.0 * sum(v > 50.0 for v in per_m)
            / len(per_m),
            "views_agree": not metrics["discrepancies"],
            "mean_reward": statistics.fmean(
                [r["reward"] for r in env.c_decisions
                 if r["reward"] is not None] or [0.0]),
            "returned": all(turned) if turned else None,
            **terms,
        }
        if policy == "rl" and not greedy:
            out.update(_transitions(env, agent))
            out["global"] = np.stack([pad_global(g, plan["n_m"], plan["n_c"])
                                      for g in out["global"]])
        return out
    finally:
        config.NUM_M_DRONES, config.NUM_C_DRONES = counts
        _restore(previous)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def full_jobs(state: Any, policy: str = "rl") -> List[Tuple]:
    return [(layout, validation_run_seed(layout), state, True, "normal",
             NORMAL_PLAN, policy) for layout in VALIDATION_LAYOUTS]


def varied_jobs(state: Any, policy: str = "rl") -> List[Tuple]:
    return [(layout, seed, state, True, "normal", plan, policy)
            for layout, seed, plan in varied_validation_plans()]


def evaluate(pool: Any, state: Any, policy: str = "rl"
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Both validation sets; returns the per-layout rows of each."""
    jobs = full_jobs(state, policy) + varied_jobs(state, policy)
    results = list(pool.map(run_episode, jobs))
    if not all(r["views_agree"] for r in results):
        raise RuntimeError("views disagreed in validation")
    rows = []
    for job, r in zip(jobs, results):
        plan = job[5]
        rows.append({
            "set": "full" if plan["kind"] == "normal" else "varied",
            "placement_seed": job[0], "run_seed": job[1],
            "type": plan["kind"], "n_m": plan["n_m"], "n_c": plan["n_c"],
            "total_loss": 100.0 * r["total_loss"],
            **{c: 100.0 * r[c] for c in CAUSES},
            "c_distance_flown_mean": r["c_distance_flown_mean"],
            "min_c_to_c_m": r["min_c_to_c_m"],
            "worst_m_loss": r["worst_m_loss"],
            "pct_m_over_half": r["pct_m_over_half"],
            "views_agree": r["views_agree"],
        })
    full = [r for r in rows if r["set"] == "full"]
    varied = [r for r in rows if r["set"] == "varied"]
    return full, varied


def score(full: Sequence[Dict], varied: Sequence[Dict]) -> Tuple[float, ...]:
    f = statistics.fmean(r["total_loss"] for r in full)
    v = statistics.fmean(r["total_loss"] for r in varied)
    return f, v, 0.5 * (f + v)


# ---------------------------------------------------------------------------
# Training (round 1's update, imported settings, new jobs and keep rule)
# ---------------------------------------------------------------------------

def train(s: Dict[str, Any], pool: Any, *, reward: str, varied: bool,
          until: int, episodes: int, out_dir: str,
          ckpt_dir: Optional[str], model_tag: Optional[str],
          on_progress: Any = None, evaluate_fn: Any = None) -> Dict[str, Any]:
    """Continue *s* to update *until*. The PPO update is round 1's line for
    line; the episodes' worlds and rewards and the keep rule are round 2's."""
    import torch

    actor, critic = s["actor"], s["critic"]
    normaliser, rng = s["normaliser"], s["rng"]
    actor_opt, critic_opt = s["actor_opt"], s["critic_opt"]
    model_dir = os.path.join(out_dir, "models", "2c")
    evaluate_fn = evaluate_fn or evaluate

    for update in range(s["update"] + 1, until + 1):
        started = time.perf_counter()
        state = {k: t.clone() for k, t in actor.state_dict().items()}
        run_seeds = rng.integers(1, 10 ** 8, size=episodes).tolist()
        episode_layouts = sample_layouts(rng, episodes)
        plans = [training_plan(int(seed), varied) for seed in run_seeds]
        jobs = [(int(layout), int(seed), state, False, reward, plan, "rl")
                for layout, seed, plan in zip(episode_layouts, run_seeds,
                                              plans)]
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
            "fair_term": statistics.fmean([r["fair_term"] for r in results]),
            "varied_episodes": sum(p["kind"] != "normal" for p in plans),
            "policy_loss": policy_loss, "value_loss": value_loss,
            "std": float(actor.log_std.exp().mean().detach()),
            "decisions": len(target),
            "seconds": time.perf_counter() - started,
        }
        s["log"].append(row)
        s["update"] = update
        print(f"  {s['run']} update {update:>3}  train {row['train_loss']:6.2f}%"
              f"  reward {row['mean_reward']:+.4f}  fair "
              f"{row['fair_term']:+.4f}  varied {row['varied_episodes']:>2}"
              f"  std {row['std']:.3f}  {row['seconds']:5.1f}s", flush=True)

        if update % EVALUATE_EVERY == 0:
            eval_state = {k: t.clone() for k, t in actor.state_dict().items()}
            full, var = evaluate_fn(pool, eval_state)
            f, v, avg = score(full, var)
            saved = avg < s["best"] - MIN_IMPROVEMENT_CURRICULUM
            if saved:
                s["best"], s["best_update"] = avg, update
                if model_tag is not None:
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(actor.state_dict(), os.path.join(
                        model_dir, f"c_actor_r2_{model_tag}.pt"))
                    torch.save(critic.state_dict(), os.path.join(
                        model_dir, f"c_critic_r2_{model_tag}.pt"))
            s["evaluations"].append({
                "run": s["run"], "update": update, "full_loss": f,
                "varied_loss": v, "eval_loss": avg, "saved": saved})
            print(f"  {s['run']} update {update:>3}  EVALUATION full {f:6.2f}%"
                  f"  varied {v:6.2f}%  average {avg:6.2f}%"
                  + ("  <- best, saved" if saved else ""), flush=True)
            if on_progress is not None:
                on_progress(s)

        if ckpt_dir is not None and update % CHECKPOINT_EVERY == 0:
            path = save_checkpoint(ckpt_dir, s)
            print(f"  checkpoint {path}", flush=True)
    return s


# ---------------------------------------------------------------------------
# Before the runs
# ---------------------------------------------------------------------------

def measure_terms(out_dir: str) -> float:
    """Old delivery term, unscaled fair term and flight cost on one full-fleet
    episode with the direct actor (spread CONTINUE_STD, sampled as in
    training). Returns k = std(old) / std(fair)."""
    import torch

    state = torch.load(os.path.join(out_dir, DIRECT_ACTOR), weights_only=True)
    state["log_std"] = torch.full_like(state["log_std"],
                                       float(np.log(config.CONTINUE_STD)))
    sizes = _measure_episode(state)
    k = float(sizes["old"].std() / sizes["fair"].std())
    print(f"\nTerm sizes, layout {K_LAYOUT}, run seed {K_RUN_SEED}, direct "
          f"actor, std {config.CONTINUE_STD}, {len(sizes['old'])} decisions:")
    for name, values in (("old delivery term", sizes["old"]),
                         ("fair term, unscaled", sizes["fair"]),
                         ("flight cost", sizes["flight"])):
        print(f"  {name:22s} mean {values.mean():+.6f}  std {values.std():.6f}"
              f"  min {values.min():+.6f}  max {values.max():+.6f}")
    print(f"  k = std(old) / std(fair) = {sizes['old'].std():.6f} / "
          f"{sizes['fair'].std():.6f} = {k:.6f}")
    print(f"  fair term with k: mean {k * sizes['fair'].mean():+.6f}  std "
          f"{k * sizes['fair'].std():.6f}")
    return k


def _measure_episode(state: Any) -> Dict[str, np.ndarray]:
    """run_episode's world, keeping the term sizes (no reward replaced)."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [K_RUN_SEED, config.AGENT_STREAM_TAG]).generate_state(1)[0]))
    previous = _world(True, DISTANCE_M)
    try:
        from agents.c_actor import GaussianActor

        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=K_LAYOUT, run_seed=K_RUN_SEED)
        env.reset()
        actor = GaussianActor()
        actor.load_state_dict(state)
        actor.eval()
        env.c_agent = SamplingAgent(env, actor, greedy=False)
        flown = [np.array([d.distance_flown for d in env.drones])]
        for _ in range(config.MAX_STEPS):
            env.step()
            flown.append(np.array([d.distance_flown for d in env.drones]))
        env.close_logger()
        return fair_rewards(env, np.array(flown), 1.0, replace=False)
    finally:
        _restore(previous)


def smoke_types(pool: Any, out_dir: str) -> None:
    """One training-style episode of each varied type, with the direct actor."""
    import torch

    state = torch.load(os.path.join(out_dir, DIRECT_ACTOR), weights_only=True)
    jobs, plans = [], []
    for kind, layout in zip(TYPES, (101, 102, 103, 104)):
        rng = np.random.default_rng(np.random.SeedSequence([layout, 77, 0]))
        plan = draw_varied(rng, kind)
        plans.append(plan)
        jobs.append((layout, 4242, state, False, "fair", plan, "rl"))
    results = list(pool.map(run_episode, jobs))
    print("\nOne episode per varied type (direct actor, sampled, fair reward):")
    for job, plan, r in zip(jobs, plans, results):
        flyers = len(plan["flyers"])
        print(f"  type {plan['kind']}  layout {job[0]}  {plan['n_m']:>2} M + "
              f"{plan['n_c']} C  D {plan['distance']:.0f}  flyers {flyers}  "
              f"hover {[round(h, 1) for h in plan['hover_s']]}  "
              f"returned {r['returned']}  total loss "
              f"{100 * r['total_loss']:.2f}%  worst M {r['worst_m_loss']:.1f}%"
              f"  decisions {len(r['reward'])}  global width "
              f"{r['global'].shape[1]}  views {r['views_agree']}")


def write_plans(out_dir: str) -> None:
    rows = []
    for layout, seed, plan in varied_validation_plans():
        rows.append({"placement_seed": layout, "run_seed": seed,
                     "type": plan["kind"], "n_m": plan["n_m"],
                     "n_c": plan["n_c"], "distance": plan["distance"],
                     "flyers": len(plan["flyers"]),
                     "flyer_points": " ".join(f"{x:.0f},{y:.0f}"
                                              for x, y in plan["flyers"]),
                     "hover_s": " ".join(f"{h:.1f}" for h in plan["hover_s"])})
    write_csv(os.path.join(out_dir, "2c_r2_varied_plans.csv"), rows,
              list(rows[0]), "phase 2c round 2, varied validation worlds, "
              "layouts 331-360, run seed 9200 + (layout - 331)")
    print("\nVaried validation worlds (out/2c_r2_varied_plans.csv):")
    for r in read_csv(os.path.join(out_dir, "2c_r2_varied_plans.csv")):
        print(f"  {r['placement_seed']} seed {r['run_seed']} type {r['type']}"
              f"  {r['n_m']:>2} M + {r['n_c']} C  D {float(r['distance']):.0f}"
              f"  flyers {r['flyers']}  hover {r['hover_s']}")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

ROW_COLUMNS = (["policy", "set", "placement_seed", "run_seed", "type", "n_m",
                "n_c", "total_loss"] + list(CAUSES)
               + ["c_distance_flown_mean", "min_c_to_c_m", "worst_m_loss",
                  "pct_m_over_half", "views_agree"])


def references(pool: Any, out_dir: str) -> None:
    """control, direct, midpoint and hover once on both sets. control's full
    set must reproduce round 1's saved validation exactly."""
    import torch

    rows = []
    for name, policy, path in (("control", "rl", CONTROL_ACTOR),
                               ("direct", "rl", DIRECT_ACTOR),
                               ("midpoint", "midpoint", None),
                               ("hover", "hover", None)):
        state = (torch.load(os.path.join(out_dir, path), weights_only=True)
                 if path else None)
        full, var = evaluate(pool, state, policy)
        rows += [{"policy": name, **r} for r in full + var]
    old = {int(r["placement_seed"]): float(r["total_loss"])
           for r in read_csv(os.path.join(out_dir, "2c_r1_validation.csv"))
           if r["run"] == "control"}
    worst = max(abs(r["total_loss"] - old[r["placement_seed"]])
                for r in rows if r["policy"] == "control" and r["set"] == "full")
    print(f"  control full set vs round 1's saved validation: largest gap "
          f"{worst:.1e}")
    if worst > 1e-9:
        raise SystemExit("STOP: control does not reproduce round 1")
    write_csv(os.path.join(out_dir, "2c_r2_references.csv"), rows,
              ROW_COLUMNS, "phase 2c round 2, references on both validation "
              "sets (full 301-330 D=650; varied 331-360)")


def write_curves(s: Dict[str, Any], out_dir: str) -> None:
    reward, varied = RUNS[s["run"]]
    note = (f"phase 2c round 2, run {s['run']}, reward {reward}, varied "
            f"{varied}, k {FAIR_K}, PPO seed {PPO_SEED}, from the direct "
            f"model; score = average of full and varied validation loss")
    write_csv(os.path.join(out_dir, f"2c_r2_curve_{s['run']}.csv"),
              s["evaluations"], ["run", "update", "full_loss", "varied_loss",
                                 "eval_loss", "saved"], note)
    write_csv(os.path.join(out_dir, f"2c_r2_log_{s['run']}.csv"), s["log"],
              ["run", "update", "train_loss", "mean_reward", "fair_term",
               "varied_episodes", "policy_loss", "value_loss", "std",
               "decisions", "seconds"], note)


def plot_curves(out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.stamp import settings_stamp

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharex=True)
    for run in RUNS:
        path = os.path.join(out_dir, f"2c_r2_curve_{run}.csv")
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        x = [int(r["update"]) for r in rows]
        for ax, key in zip(axes, ("full_loss", "varied_loss", "eval_loss")):
            ax.plot(x, [float(r[key]) for r in rows], lw=1.2,
                    label=f"{run} ({RUNS[run][0]}, "
                          f"{'varied' if RUNS[run][1] else 'full fleet'})")
    for ax, title in zip(axes, ("full-fleet validation 301-330",
                                "varied validation 331-360",
                                "average (the keep score)")):
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("update")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("total loss (%)")
    axes[-1].legend(fontsize=7)
    fig.text(0.5, 0.005, settings_stamp(extra="phase 2c round 2"),
             ha="center", fontsize=4.5, color="#666666", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out_dir, "2c_r2_curves.png"), dpi=130)
    plt.close(fig)


def run_edge_cases(name: str, model: str, out_dir: str, workers: int) -> None:
    """Edge cases 1-8 and 9-13 on one actor, each in its own folder."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "edge_cases_2c.py")
    for command in (
            [sys.executable, script, "--out-dir", out_dir, "--workers",
             str(workers), "--model", model, "--out",
             os.path.join(out_dir, f"2c_edge_r2_{name}")],
            [sys.executable, script, "--out-dir", out_dir, "--workers",
             str(workers), "--bigfleet", "--actor", f"{name}={model}",
             "--out", os.path.join(out_dir, f"2c_edge_bigfleet_r2_{name}")]):
        print("  " + " ".join(command), flush=True)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def report(out_dir: str) -> None:
    """Summary, per-type, paired and edge tables, from the saved files."""
    note = "phase 2c round 2, validation only, from the saved per-layout files"
    rows = []
    rpath = os.path.join(out_dir, "2c_r2_references.csv")
    vpath = os.path.join(out_dir, "2c_r2_validation.csv")
    for path in (rpath, vpath):
        if os.path.exists(path):
            rows += read_csv(path)
    names = [n for n in ("control", "A", "B", "C", "direct", "midpoint",
                         "hover") if any(r["policy"] == n for r in rows)]

    def group(name: str, which: str, kind: Optional[str] = None) -> List[Dict]:
        return [r for r in rows if r["policy"] == name and r["set"] == which
                and (kind is None or r["type"] == kind)]

    def mean(g: Sequence[Dict], key: str) -> float:
        values = [float(r[key]) for r in g if r[key] not in ("", "nan")]
        return statistics.fmean(values) if values else float("nan")

    summary = []
    for name in names:
        full, var = group(name, "full"), group(name, "varied")
        summary.append({
            "policy": name,
            "full_total": mean(full, "total_loss"),
            **{f"full_{c}": mean(full, c) for c in CAUSES},
            "full_c_flown": mean(full, "c_distance_flown_mean"),
            "varied_total": mean(var, "total_loss"),
            "varied_worst_m": mean(var, "worst_m_loss"),
            "varied_pct_m_over_half": mean(var, "pct_m_over_half"),
            "average": 0.5 * (mean(full, "total_loss")
                              + mean(var, "total_loss")),
        })
    write_csv(os.path.join(out_dir, "2c_r2_summary.csv"), summary,
              list(summary[0]), note)

    by_type = []
    for name in names:
        for kind in TYPES:
            g = group(name, "varied", kind)
            by_type.append({"policy": name, "type": kind, "layouts": len(g),
                            "total_loss": mean(g, "total_loss"),
                            "worst_m_loss": mean(g, "worst_m_loss"),
                            "pct_m_over_half": mean(g, "pct_m_over_half")})
    write_csv(os.path.join(out_dir, "2c_r2_by_type.csv"), by_type,
              list(by_type[0]), note)

    paired = []
    for name in names:
        if name == "control":
            continue
        for which in ("full", "varied"):
            base = {r["placement_seed"]: float(r["total_loss"])
                    for r in group("control", which)}
            mine = {r["placement_seed"]: float(r["total_loss"])
                    for r in group(name, which)}
            diffs = [mine[k] - base[k] for k in sorted(base)]
            m = statistics.fmean(diffs)
            half = T_95_DF29 * statistics.stdev(diffs) / math.sqrt(len(diffs))
            paired.append({"policy": name, "set": which, "layouts": len(diffs),
                           "mean_diff": m, "ci95_low": m - half,
                           "ci95_high": m + half,
                           "wins": sum(d < 0 for d in diffs)})
    write_csv(os.path.join(out_dir, "2c_r2_paired.csv"), paired,
              list(paired[0]), note + "; each policy minus control")

    edge = []
    sources = {"control": ("2c_edge_r1_control", "direct",
                           "2c_edge_bigfleet_r2_control", "control"),
               "midpoint": ("2c_edge", "midpoint", "2c_edge_bigfleet",
                            "midpoint")}
    for run in RUNS:
        sources[run] = (f"2c_edge_r2_{run}", "direct",
                        f"2c_edge_bigfleet_r2_{run}", run)
    for name in ("control", "A", "B", "C", "midpoint"):
        small, small_policy, big, big_policy = sources[name]
        p1 = os.path.join(out_dir, small, "edge_summary.csv")
        p2 = os.path.join(out_dir, big, "edge_summary.csv")
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        for r in read_csv(p1):
            if r["policy"] == small_policy:
                # Scenarios 1-8: every M-drone flies, so flying = total.
                edge.append({"policy": name,
                             "scenario": int(float(r["scenario"])),
                             "total_loss": float(r["total_loss"]),
                             "flying_m_loss": float(r["total_loss"])})
        for r in read_csv(p2):
            if r["policy"] == big_policy:
                edge.append({"policy": name,
                             "scenario": int(float(r["scenario"])),
                             "total_loss": float(r["total_loss"]),
                             "flying_m_loss": float(r["loss_flying"])})
    if edge:
        write_csv(os.path.join(out_dir, "2c_r2_edge.csv"), edge,
                  ["policy", "scenario", "total_loss", "flying_m_loss"],
                  "phase 2c round 2, edge cases 1-13, mean of run seeds "
                  "9900-9904; in 1-8 every M-drone flies")
    print_report(out_dir)


def print_report(out_dir: str) -> None:
    s = read_csv(os.path.join(out_dir, "2c_r2_summary.csv"))
    print("\n  Full-fleet validation 301-330 (from out/2c_r2_summary.csv)")
    print(f"  {'policy':>9} {'total':>6} " + " ".join(f"{c[:8]:>8}"
                                                   for c in CAUSES)
          + f" {'C flown':>8}")
    for r in s:
        print(f"  {r['policy']:>9} {float(r['full_total']):6.2f} "
              + " ".join(f"{float(r['full_' + c]):8.2f}" for c in CAUSES)
              + f" {float(r['full_c_flown']):7.0f}m")
    print("\n  Varied validation 331-360")
    print(f"  {'policy':>9} {'total':>6} {'worst M':>8} {'M >50%':>7} "
          f"{'average':>8}   " + "  ".join(f"type {k}" for k in TYPES))
    bt = read_csv(os.path.join(out_dir, "2c_r2_by_type.csv"))
    for r in s:
        types = "  ".join(f"{float(t['total_loss']):6.2f}" for t in bt
                          if t["policy"] == r["policy"])
        print(f"  {r['policy']:>9} {float(r['varied_total']):6.2f} "
              f"{float(r['varied_worst_m']):8.2f} "
              f"{float(r['varied_pct_m_over_half']):6.1f}% "
              f"{float(r['average']):8.2f}   {types}")
    p = os.path.join(out_dir, "2c_r2_paired.csv")
    if os.path.exists(p):
        print("\n  Paired against control (points of total loss)")
        for r in read_csv(p):
            print(f"  {r['policy']:>9} {r['set']:>6}  "
                  f"{float(r['mean_diff']):+6.2f} [{float(r['ci95_low']):+.2f}"
                  f", {float(r['ci95_high']):+.2f}]  lower on {r['wins']}/"
                  f"{r['layouts']}")
    e = os.path.join(out_dir, "2c_r2_edge.csv")
    if os.path.exists(e):
        rows = read_csv(e)
        print("\n  Edge cases: total loss / flying-M loss, % (mean of 5 seeds)")
        print(f"  {'policy':>9} " + " ".join(f"{'s' + str(k):>11}"
                                             for k in range(1, 14)))
        for name in ("control", "A", "B", "C", "midpoint"):
            cells = {int(r["scenario"]): r for r in rows
                     if r["policy"] == name}
            if len(cells) == 13:
                print(f"  {name:>9} " + " ".join(
                    f"{float(cells[k]['total_loss']):5.1f}/"
                    f"{float(cells[k]['flying_m_loss']):5.1f}"
                    for k in range(1, 14)))


def finish_run(run: str, s: Dict[str, Any], pool: Any, out_dir: str,
               workers: int) -> None:
    """Re-validate the best actor, save its rows, run the edge cases."""
    import torch

    write_curves(s, out_dir)
    plot_curves(out_dir)
    state = torch.load(os.path.join(out_dir, "models", "2c",
                                    f"c_actor_r2_{run}.pt"), weights_only=True)
    full, var = evaluate(pool, state)
    f, v, avg = score(full, var)
    print(f"  {run}: best actor re-validated at full {f:.4f}%, varied "
          f"{v:.4f}%, average {avg:.4f}% (curve best {s['best']:.4f}%)",
          flush=True)
    if abs(avg - s["best"]) > 1e-9:
        raise SystemExit("STOP: the saved best actor does not reproduce its "
                         "validation score")
    vpath = os.path.join(out_dir, "2c_r2_validation.csv")
    old = ([r for r in read_csv(vpath) if r["policy"] != run]
           if os.path.exists(vpath) else [])
    fresh = [{"policy": run, **r} for r in full + var]
    ordered = [r for name in RUNS
               for r in (fresh if name == run else
                         [o for o in old if o["policy"] == name])]
    write_csv(vpath, ordered, ROW_COLUMNS, "phase 2c round 2, best actor per "
              "run, both validation sets")
    run_edge_cases(run, os.path.join(out_dir, "models", "2c",
                                     f"c_actor_r2_{run}.pt"), out_dir, workers)
    report(out_dir)


def draw_gifs(out_dir: str) -> None:
    """GIFs for the best of A/B/C (lowest keep score) on 1, 6, 9, 10, 13."""
    summary = read_csv(os.path.join(out_dir, "2c_r2_summary.csv"))
    done = [r for r in summary if r["policy"] in RUNS]
    if len(done) != len(RUNS):
        raise SystemExit("STOP: not every run is finished")
    best = min(done, key=lambda r: float(r["average"]))["policy"]
    print(f"  best of A/B/C by the keep score: {best}")
    plot = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "plot_edge_cases_2c.py")
    subprocess.run([sys.executable, plot, "--out-dir", out_dir, "--edge-dir",
                    os.path.join(out_dir, f"2c_edge_r2_{best}"), "--label",
                    best, "--scenarios", "1", "6"], check=True)
    subprocess.run([sys.executable, plot, "--out-dir", out_dir, "--bigfleet",
                    "--edge-dir", os.path.join(out_dir,
                                               f"2c_edge_bigfleet_r2_{best}"),
                    "--policies", "midpoint", best, "--scenarios", "9", "10",
                    "13"], check=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def print_settings(episodes: int) -> None:
    import scripts.train_c_ppo as base

    print("Round 2 settings:")
    rows = [
        ("runs (reward, varied)", RUNS, "spec"),
        ("old reward", "(delivered - lost)/created - C_FLIGHT_COST*m/speed",
         "c_interface.flight_reward"),
        ("C_FLIGHT_COST", config.C_FLIGHT_COST, "config.py"),
        ("fair term", f"k*mean log(min(1,d_i/c_i)+{FAIR_FLOOR}), "
                      f"{FAIR_WINDOW_S} s window", "spec, approved"),
        ("k", FAIR_K, "--measure"),
        ("varied share", VARIED_SHARE, "spec"),
        ("varied ranges", f"M 1-{FULL_M}, C 1-{FULL_C}, D {SPREAD_D_M}, far "
                          f"{FAR_M}, +-20 deg, hover {HOVER_S}", "spec"),
        ("CONTINUE_STD", config.CONTINUE_STD, "config.py"),
        ("critic-only warm-up", CRITIC_ONLY_UPDATES, "train_c_ppo.py"),
        ("PPO seed / updates", (PPO_SEED, UPDATES), "train_c_ppo_round1.py"),
        ("episodes per update", episodes, "train_c_ppo_m4.py"),
        ("clip / epochs / lr", (base.CLIP, base.EPOCHS_PER_UPDATE,
                                base.LEARNING_RATE), "train_c_ppo.py"),
        ("grad clip / minibatch", (base.GRAD_CLIP, base.MINIBATCH),
         "train_c_ppo.py"),
        ("gamma / lambda", (base.GAMMA, base.GAE_LAMBDA), "train_c_ppo.py"),
        ("evaluate every", EVALUATE_EVERY, "train_c_ppo.py"),
        ("checkpoint every", CHECKPOINT_EVERY, "train_c_ppo_round1.py"),
        ("validation", "full 301-330 seed 9100+; varied 331-360 seed 9200+",
         "spec"),
        ("keep rule", "lowest average of the two validation losses", "spec"),
    ]
    for name, value, source in rows:
        print(f"  {name:24s} {str(value):64s} [{source}]")


def resume_test(pool: Any, out_dir: str, episodes: int) -> None:
    """Run C: 30 updates straight vs 25 + stop + resume, compared exactly."""
    import torch

    reward, varied = RUNS["C"]
    scratch = tempfile.mkdtemp(prefix="r2_resume_")
    print(f"\n=== resume test: run C, 30 straight vs 25 + resume, scratch "
          f"{scratch} ===", flush=True)
    kw = dict(reward=reward, varied=varied, episodes=episodes,
              out_dir=out_dir, model_tag=None)
    straight = train(fresh_state("C", out_dir), pool, until=30,
                     ckpt_dir=os.path.join(scratch, "a"), **kw)
    first = train(fresh_state("C", out_dir), pool, until=25,
                  ckpt_dir=os.path.join(scratch, "b"), **kw)
    del first
    resumed = load_checkpoint(checkpoint_paths(os.path.join(scratch, "b"))[-1])
    print(f"  resumed from update {resumed['update']}", flush=True)
    resumed = train(resumed, pool, until=30,
                    ckpt_dir=os.path.join(scratch, "b"), **kw)
    problems = []
    for a, b in zip(straight["log"], resumed["log"]):
        for key in ("train_loss", "mean_reward", "fair_term", "policy_loss",
                    "value_loss", "std", "varied_episodes"):
            if a[key] != b[key]:
                problems.append(f"update {a['update']} {key}")
    for key in ("full_loss", "varied_loss"):
        if [e[key] for e in straight["evaluations"]] != \
                [e[key] for e in resumed["evaluations"]]:
            problems.append(f"{key} differs")
    for name in ("actor", "critic"):
        sa, sb = straight[name].state_dict(), resumed[name].state_dict()
        problems += [f"{name} {k}" for k in sa if not torch.equal(sa[k], sb[k])]
    if len(straight["log"]) != 30 or len(resumed["log"]) != 30:
        problems.append("a run did not reach 30 updates")
    for label, run in (("straight", straight), ("resumed ", resumed)):
        print(f"  validation {label} "
              f"{[(round(e['full_loss'], 6), round(e['varied_loss'], 6)) for e in run['evaluations']]}")
    shutil.rmtree(scratch)
    if problems:
        print("STOP: the resumed run differs:\n  " + "\n  ".join(problems))
        raise SystemExit(1)
    print("  RESUME TEST PASSED: every update's losses, both validation sets "
          "and every actor and critic weight match exactly")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2c round 2: fair reward and varied missions.")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_UPDATE_M4)
    parser.add_argument("--runs", type=str, nargs="+", default=list(RUNS),
                        choices=list(RUNS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--measure", action="store_true",
                        help="print the term sizes and k, then stop")
    parser.add_argument("--checks", action="store_true",
                        help="varied plans, one episode per type, references")
    parser.add_argument("--resume-test", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--gifs", action="store_true")
    args = parser.parse_args()

    if args.measure:
        measure_terms(args.out_dir)
        return
    if args.report_only:
        report(args.out_dir)
        return
    if args.gifs:
        draw_gifs(args.out_dir)
        return
    if FAIR_K is None:
        raise SystemExit("STOP: FAIR_K is not set; run --measure first")

    print_settings(args.episodes)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        if args.checks:
            write_plans(args.out_dir)
            smoke_types(pool, args.out_dir)
            references(pool, args.out_dir)
            # control's edge cases 1-8 are round 1's; 9-13 run here.
            script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "edge_cases_2c.py")
            command = [sys.executable, script, "--out-dir", args.out_dir,
                       "--workers", str(args.workers), "--bigfleet",
                       "--actor", "control=" + os.path.join(
                           args.out_dir, CONTROL_ACTOR),
                       "--out", os.path.join(args.out_dir,
                                             "2c_edge_bigfleet_r2_control")]
            print("  " + " ".join(command), flush=True)
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            report(args.out_dir)
            return
        if args.resume_test:
            resume_test(pool, args.out_dir, args.episodes)
            return
        for run in args.runs:
            reward, varied = RUNS[run]
            ckpt_dir = os.path.join(args.out_dir, CHECKPOINT_DIR, run)
            existing = checkpoint_paths(ckpt_dir)
            if args.resume and existing:
                s = load_checkpoint(existing[-1])
                print(f"\n=== {run}: resumed from {existing[-1]} (update "
                      f"{s['update']}) ===", flush=True)
            else:
                if existing:
                    raise SystemExit(f"{ckpt_dir} already has checkpoints; "
                                     f"pass --resume or move them away")
                s = fresh_state(run, args.out_dir)
                print(f"\n=== {run}: reward {reward}, varied {varied}, fresh "
                      f"from the direct model ===", flush=True)
            if s["update"] < UPDATES:
                started = time.time()
                s = train(s, pool, reward=reward, varied=varied,
                          until=UPDATES, episodes=args.episodes,
                          out_dir=args.out_dir, ckpt_dir=ckpt_dir,
                          model_tag=run,
                          on_progress=lambda st: write_curves(st,
                                                              args.out_dir))
                print(f"  {run}: {UPDATES} updates, best {s['best']:.2f}% at "
                      f"update {s['best_update']}, "
                      f"{(time.time() - started) / 3600:.2f} h", flush=True)
            finish_run(run, s, pool, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
