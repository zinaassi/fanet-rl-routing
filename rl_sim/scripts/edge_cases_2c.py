"""
edge_cases_2c.py — Phase 2c: hand-built scenarios for the frozen direct actor.

No training. Eight small worlds, each built to ask one question of the actor
out/models/2c/c_actor_curriculum_direct.pt, run beside hover and midpoint on
the same seeds:

    1  1 C, 1 M   M to 600 m and hovers       C between GS and M
    2  3 C, 1 M   M to the corner (~622 m)    the C-drones form a line
    3  2 C, 2 M   M-drones 600 m, opposite    one C each way
    4  3 C, 1 M   M to 250 m only             the C-drones barely move
    5  2 C, 1 M   M to 600 m, 20 s, and back  the C-drones follow out and back
    6  1 C, 1 M   M to the corner (~622 m)    C holds near the middle
    7  7 C, 18 M  M 400-650 m, 45 +- 20 deg   the C-drones stretch that way
    8  7 C, 1 M   M to 600 m                  the spare C-drones do not crowd

The arena is 900 x 900 m with the GS in the middle, so a straight flight of
more than 440 m only fits along a diagonal. Every flight here goes along the
45-degree diagonal (scenario 3 uses both ends of it), and anything longer than
the diagonal is clipped at the border exactly as the mission clips it.

The world is the direct run's evaluation world, set by train_c_ppo._world:
greedy routing, 200 ms, hello 1 s, AVOID on, 1000 steps. Only the fleet size,
the start positions and the M-drone flights are the scenario's. Those are set
on the environment after reset; nothing in fanet_sim is changed. The one path
the simulator cannot fly by itself, scenario 5's return leg, is given by
resetting the M-drone's end point from here once it has hovered 20 s.

Every run is recorded step by step from the environment it was scored on:
positions, env.active_links, which drones are in range of the GS, and the
path of every delivered packet. plot_edge_cases_2c.py draws only from those
recordings.

Big-fleet scenarios (--bigfleet), 18 M + 7 C, every drone starting at the
near_gs mission's own draw for this placement seed; staying M-drones hover
where they start, the flying ones (the lowest-numbered) go at 20 m/s with no
delay:

    9   1 M to the NE corner (~622 m)
    10  3 M to the NE, NW and SW corners
    11  1 M to 300 m only (still in reach of the GS)
    12  1 M to the NE corner, hovers 20 s, back to its own start point
    13  9 M toward NE, +-20 deg, 450-622 m

They run hover, midpoint, direct and spacing_large into out/2c_edge_bigfleet/.
Scenarios 1-8 and their outputs are untouched by them.

Writes:
    out/2c_edge/edge_runs.csv          one row per (scenario, policy, seed)
    out/2c_edge/edge_summary.csv       means over the five seeds
    out/2c_edge/edge_geometry.csv      line distances and hop lengths, 1/2/5/6
    out/2c_edge/edge_s6_midpoint.csv   scenario 6: C to the exact midpoint
    out/2c_edge/rec/*.npz              the recordings (not committed)
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.c_scripted import HoverAgent, MidpointAgent
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics, mission_spread
from scripts.train_c_ppo import (CAUSES, ROUTING, SamplingAgent, _restore,
                                 _world, write_csv)

#: Placement seed for every scenario. In no training, validation or test set.
PLACEMENT_SEED: int = 900

#: The five channel and traffic seeds, the same for every policy.
RUN_SEEDS: Tuple[int, ...] = (9900, 9901, 9902, 9903, 9904)

POLICIES: Tuple[str, ...] = ("hover", "midpoint", "direct")
DIRECT_MODEL: str = os.path.join("models", "2c", "c_actor_curriculum_direct.pt")

#: Scenarios 1-6 and 8 fly at this speed with no start delay.
M_SPEED: float = 20.0

#: The diagonal the flights follow, and the opposite one for scenario 3.
NE: float = np.pi / 4.0
SW: float = NE + np.pi

#: Where the M-drones and the C-drones start, as distances from the GS.
M_START_M: float = 40.0
C_RING_M: float = 60.0

#: Scenario 5 hovers this long at the far end before it turns back.
HOVER_BEFORE_RETURN_S: float = 20.0

#: Scenario 7's spread about the diagonal and its flight lengths.
S7_SPREAD_RAD: float = np.radians(20.0)
S7_DISTANCE_M: Tuple[float, float] = (400.0, 650.0)

#: A flight long enough that clipping puts it on the corner.
TO_CORNER_M: float = 2000.0

#: The times the geometry table samples, in seconds.
GEOMETRY_TIMES_S: Tuple[float, ...] = (30.0, 60.0, 95.0)
GEOMETRY_SCENARIOS: Tuple[int, ...] = (1, 2, 5, 6)

#: A route at time t is the delivered packet created closest to t, within this.
ROUTE_WINDOW_S: float = 2.0

#: Scenario -> (M-drones, C-drones, one-line description).
SCENARIOS: Dict[int, Tuple[int, int, str]] = {
    1: (1, 1, "M to 600 m and hovers"),
    2: (1, 3, "M to the corner (~622 m)"),
    3: (2, 2, "two M 600 m in opposite directions"),
    4: (1, 3, "M to 250 m only"),
    5: (1, 2, "M to 600 m, hovers 20 s, flies back"),
    6: (1, 1, "M to the corner (~622 m)"),
    7: (18, 7, "18 M 400-650 m within 45 +- 20 deg"),
    8: (1, 7, "M to 600 m, 7 C"),
}

#: Big-fleet scenarios, run only with --bigfleet. Same tuple as SCENARIOS.
BIGFLEET: Dict[int, Tuple[int, int, str]] = {
    9: (18, 7, "1 M to the NE corner, 17 M stay"),
    10: (18, 7, "3 M to the NE, NW, SW corners, 15 M stay"),
    11: (18, 7, "1 M to 300 m, 17 M stay"),
    12: (18, 7, "1 M to the NE corner, 20 s, back; 17 M stay"),
    13: (18, 7, "9 M toward NE +-20 deg 450-622 m, 9 M stay"),
}
ALL_SCENARIOS: Dict[int, Tuple[int, int, str]] = {**SCENARIOS, **BIGFLEET}
BIGFLEET_POLICIES: Tuple[str, ...] = ("hover", "midpoint", "direct",
                                      "spacing_large")
BIGFLEET_MODELS: Dict[str, str] = {
    "direct": DIRECT_MODEL,
    "spacing_large": os.path.join("models", "2c",
                                  "c_actor_r1_spacing_large.pt"),
}
NW: float = NE + np.pi / 2.0
S13_SPREAD_RAD: float = np.radians(20.0)
S13_DISTANCE_M: Tuple[float, float] = (450.0, 622.0)
#: The scenarios whose flying M-drone returns to its start after hovering.
RETURNING: Tuple[int, ...] = (5, 12)

#: Scenario -> (direction, distance from the GS) per M-drone; 7 is drawn.
FLIGHTS: Dict[int, List[Tuple[float, float]]] = {
    1: [(NE, 600.0)],
    2: [(NE, TO_CORNER_M)],
    3: [(NE, 600.0), (SW, 600.0)],
    4: [(NE, 250.0)],
    5: [(NE, 600.0)],
    6: [(NE, TO_CORNER_M)],
    8: [(NE, 600.0)],
    9: [(NE, TO_CORNER_M)],
    10: [(NE, TO_CORNER_M), (NW, TO_CORNER_M), (SW, TO_CORNER_M)],
    11: [(NE, 300.0)],
    12: [(NE, TO_CORNER_M)],
}


def _unit(angle: float) -> np.ndarray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)


def clip_inside(point: np.ndarray) -> np.ndarray:
    """Clip a point inside the border margin, the way the mission does."""
    return FANETEnv._clip_inside_border(point)


def c_starts(count: int) -> List[np.ndarray]:
    """Return the C-drones' start points: evenly round a ring, off the NE axis.

    The ring starts at 135 degrees, a right angle away from the flights, so no
    C-drone begins on top of an M-drone.

    Args:
        count: How many C-drones.

    Returns:
        One start point per C-drone.
    """
    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    return [gs + C_RING_M * _unit(NE + np.pi / 2.0 + 2.0 * np.pi * k / count)
            for k in range(count)]


def scenario_plan(scenario: int) -> Dict[str, Any]:
    """Return every start point and flight a scenario needs.

    Args:
        scenario: 1-8.

    Returns:
        Lists "m_start", "m_end", "m_speed", "m_delay" and "c_start".
    """
    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    num_m, num_c, _ = ALL_SCENARIOS[scenario]
    plan: Dict[str, Any] = {"m_start": [], "m_end": [], "m_speed": [],
                            "m_delay": [], "c_start": None}

    if scenario in BIGFLEET:
        # Start points are the near_gs mission's own draw (the caller keeps
        # them). m_end holds a point for each FLYING drone, measured from the
        # GS; every other M-drone stays where it starts.
        if scenario == 13:
            rng = np.random.default_rng(
                np.random.SeedSequence([PLACEMENT_SEED, scenario]))
            flights = [(NE + float(rng.uniform(-S13_SPREAD_RAD,
                                               S13_SPREAD_RAD)),
                        float(rng.uniform(*S13_DISTANCE_M)))
                       for _ in range(9)]
        else:
            flights = FLIGHTS[scenario]
        for angle, distance in flights:
            plan["m_end"].append(clip_inside(gs + distance * _unit(angle)))
            plan["m_speed"].append(M_SPEED)
            plan["m_delay"].append(0.0)
        plan["flying"] = len(flights)
        return plan

    if scenario == 7:
        # Start points are the near_gs mission's own draw for this placement
        # seed (filled in by the caller); only the flights are drawn here,
        # on the scenario's own stream, from the mission's ranges.
        rng = np.random.default_rng(
            np.random.SeedSequence([PLACEMENT_SEED, scenario]))
        for _ in range(num_m):
            angle = NE + float(rng.uniform(-S7_SPREAD_RAD, S7_SPREAD_RAD))
            plan["m_end"].append((angle, float(rng.uniform(*S7_DISTANCE_M))))
            plan["m_speed"].append(float(rng.uniform(
                config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX)))
            plan["m_delay"].append(float(rng.uniform(
                0.0, config.MAX_START_DELAY_S)))
        return plan

    for angle, distance in FLIGHTS[scenario]:
        plan["m_start"].append(gs + M_START_M * _unit(angle))
        plan["m_end"].append(clip_inside(gs + distance * _unit(angle)))
        plan["m_speed"].append(M_SPEED)
        plan["m_delay"].append(0.0)
    plan["c_start"] = c_starts(num_c)
    assert len(plan["m_start"]) == num_m
    return plan


def build_env(scenario: int, run_seed: int) -> FANETEnv:
    """Reset an environment and put the scenario's fleet into it.

    The caller must already have set the world and the fleet size.

    Args:
        scenario: 1-8.
        run_seed: Channel and traffic seed.

    Returns:
        The reset environment, ready to step.
    """
    env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                   placement_seed=PLACEMENT_SEED, run_seed=run_seed)
    env.reset()
    plan = scenario_plan(scenario)
    m_drones = [d for d in env.drones if d.drone_type == "M"]
    c_drones = [d for d in env.drones if d.drone_type == "C"]

    if scenario in BIGFLEET:
        for index, drone in enumerate(m_drones):
            if index < plan["flying"]:
                drone.end_point = np.asarray(plan["m_end"][index],
                                             dtype=np.float64).copy()
                drone.speed = plan["m_speed"][index]
            else:
                drone.end_point = drone.position.copy()
            drone.start_delay_s = 0.0
            drone.arrived = bool(np.array_equal(drone.end_point,
                                                drone.position))
        env._recompute_links()
        env._update_active_links()
        return env

    for index, drone in enumerate(m_drones):
        if scenario == 7:
            angle, distance = plan["m_end"][index]
            end = clip_inside(drone.position + distance * _unit(angle))
        else:
            drone.position = plan["m_start"][index].copy()
            end = plan["m_end"][index]
        drone.end_point = np.asarray(end, dtype=np.float64).copy()
        drone.speed = plan["m_speed"][index]
        drone.start_delay_s = plan["m_delay"][index]
        drone.arrived = bool(np.array_equal(drone.end_point, drone.position))
    if plan["c_start"] is not None:
        for drone, start in zip(c_drones, plan["c_start"]):
            drone.position = start.copy()

    env._recompute_links()
    env._update_active_links()
    return env


def _record_step(env: FANETEnv, step: int, store: Dict[str, list]) -> None:
    """Append what the simulator holds right now to the recording."""
    store["pos"].append(np.array([d.position for d in env.drones]))
    for a, b in sorted(env.active_links):
        distance = float(np.linalg.norm(env.drones[a].position
                                        - env.drones[b].position))
        store["link"].append((step, a, b, channel.p_loss(distance)))
    for drone in env.drones:
        if channel.are_connected(drone.position, env.gs_position):
            distance = float(np.linalg.norm(drone.position - env.gs_position))
            store["gs"].append((step, drone.drone_id, channel.p_loss(distance)))


def _agent(policy: str, env: FANETEnv, state: Optional[Dict[str, Any]]) -> Any:
    if policy == "hover":
        return HoverAgent(env)
    if policy == "midpoint":
        return MidpointAgent(env)
    from agents.c_actor import GaussianActor

    actor = GaussianActor()
    actor.load_state_dict(state)
    actor.eval()
    return SamplingAgent(env, actor, greedy=True)


def run_one(job: Tuple[int, str, int, Any, str]) -> Dict[str, Any]:
    """Run one scenario under one policy and seed, record it, and score it.

    Args:
        job: (scenario, policy, run_seed, actor weights or None, rec dir).

    Returns:
        The metrics row.
    """
    import torch

    scenario, policy, run_seed, state, rec_dir = job
    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [run_seed, config.AGENT_STREAM_TAG]).generate_state(1)[0]))

    num_m, num_c, _ = ALL_SCENARIOS[scenario]
    previous = _world(moving=True, distance=650.0)
    counts = (config.NUM_M_DRONES, config.NUM_C_DRONES)
    config.NUM_M_DRONES, config.NUM_C_DRONES = num_m, num_c
    try:
        env = build_env(scenario, run_seed)
        env.c_agent = _agent(policy, env, state)

        store: Dict[str, list] = {"pos": [], "link": [], "gs": []}
        _record_step(env, 0, store)
        start = env.drones[0].position.copy()
        hover_from: Optional[int] = None
        for step in range(1, config.MAX_STEPS + 1):
            env.step()
            if scenario in RETURNING:
                drone = env.drones[0]
                if hover_from is None and drone.arrived:
                    hover_from = step
                if (hover_from is not None and step - hover_from
                        == int(round(HOVER_BEFORE_RETURN_S / config.TIMESTEP))):
                    drone.end_point = start.copy()
                    drone.arrived = False
            _record_step(env, step, store)
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        spread = mission_spread(env)
        positions = np.array(store["pos"])
        types = np.array([d.drone_type for d in env.drones])
        delivered = [p for p in env.delivered if p.source_id in range(num_m)]
        flying = (scenario_plan(scenario)["flying"] if scenario in BIGFLEET
                  else num_m)
        flies = [i < flying for i in range(num_m)]

        path = os.path.join(rec_dir, f"s{scenario}_{policy}_{run_seed}.npz")
        np.savez_compressed(
            path,
            pos=positions,
            types=types,
            gs=np.asarray(env.gs_position, dtype=np.float64),
            link=np.array(store["link"], dtype=np.float64).reshape(-1, 4),
            gs_link=np.array(store["gs"], dtype=np.float64).reshape(-1, 3),
            pkt_source=np.array([p.source_id for p in delivered], dtype=int),
            pkt_created=np.array([p.created_at for p in delivered], dtype=int),
            pkt_path=np.array(["-".join(str(h) for h in p.path)
                               for p in delivered], dtype=str),
            m_end=np.array([d.end_point for d in env.drones
                            if d.drone_type == "M"]),
            m_flies=np.array(flies, dtype=bool),
        )

        c_index = [i for i, t in enumerate(types) if t == "C"]
        min_cc = float("nan")
        if len(c_index) > 1:
            c_pos = positions[:, c_index, :]
            gaps = np.linalg.norm(c_pos[:, :, None, :] - c_pos[:, None, :, :],
                                  axis=-1)
            upper = np.triu_indices(len(c_index), k=1)
            min_cc = float(gaps[:, upper[0], upper[1]].min())

        extra: Dict[str, Any] = {}
        if scenario in BIGFLEET:
            extra = bigfleet_columns(env, positions, types, flies, truth)
        return {
            **extra,
            "scenario": scenario,
            "policy": policy,
            "run_seed": run_seed,
            "num_m": num_m,
            "num_c": num_c,
            "total_loss": 100.0 * (truth["total_loss"] or 0.0),
            **{cause: 100.0 * truth["loss_rate_by_reason"][cause]
               for cause in CAUSES},
            "c_distance_flown_mean": truth["c_distance_flown_mean"],
            "min_c_to_c_m": min_cc,
            "pct_m_clipped_end": spread["pct_m_clipped_end"],
            "views_agree": not metrics["discrepancies"],
        }
    finally:
        config.NUM_M_DRONES, config.NUM_C_DRONES = counts
        _restore(previous)


def group_loss(env: FANETEnv, sources: Sequence[int]) -> float:
    """% of the measured packets created by *sources* that never arrived."""
    from scripts.metrics_1a import measured_packets

    wanted = set(sources)
    arrived = {id(p) for p in env.delivered}
    mine = [p for p in measured_packets(env) if p.source_id in wanted]
    if not mine:
        return float("nan")
    return 100.0 * sum(id(p) not in arrived for p in mine) / len(mine)


def bigfleet_columns(env: FANETEnv, positions: np.ndarray, types: np.ndarray,
                     flies: Sequence[bool], truth: Dict[str, Any]
                     ) -> Dict[str, Any]:
    """Loss of the flying and the staying M-drones, and the farthest C.

    Stops if the two groups do not add back up to the total loss."""
    flying = [i for i, f in enumerate(flies) if f]
    staying = [i for i, f in enumerate(flies) if not f]
    out = {"loss_flying": group_loss(env, flying),
           "loss_staying": group_loss(env, staying)}
    both = group_loss(env, flying + staying)
    if abs(both - 100.0 * (truth["total_loss"] or 0.0)) > 1e-9:
        raise RuntimeError(f"group losses {both} do not recombine to the "
                           f"total {100.0 * truth['total_loss']}")
    gs = np.asarray(env.gs_position, dtype=np.float64)
    c = [i for i, t in enumerate(types) if t == "C"]
    for t in GEOMETRY_TIMES_S:
        step = int(round(t / config.TIMESTEP))
        out[f"far_c_{t:.0f}"] = float(
            np.linalg.norm(positions[step, c] - gs, axis=-1).max())
    return out


BIGFLEET_COLUMNS: Tuple[str, ...] = (
    "loss_flying", "loss_staying", "far_c_30", "far_c_60", "far_c_95")


def point_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Return the distance from *point* to the segment a-b."""
    ab = b - a
    length = float(ab @ ab)
    t = 0.0 if length == 0.0 else min(1.0, max(0.0, float((point - a) @ ab)
                                               / length))
    return float(np.linalg.norm(point - (a + t * ab)))


def geometry_rows(rec_dir: str) -> List[Dict[str, Any]]:
    """Line distances and hop lengths for scenarios 1, 2, 5 and 6.

    Read from the recordings only. The route at time t is the delivered
    packet from the M-drone created closest to t, within ROUTE_WINDOW_S; its
    hop lengths are measured on the positions at the step it was created.

    Args:
        rec_dir: Where the recordings are.

    Returns:
        One row per (scenario, policy, seed, time, C-drone).
    """
    rows: List[Dict[str, Any]] = []
    window = int(round(ROUTE_WINDOW_S / config.TIMESTEP))
    for scenario in GEOMETRY_SCENARIOS:
        for policy in POLICIES:
            for run_seed in RUN_SEEDS:
                rec = np.load(os.path.join(
                    rec_dir, f"s{scenario}_{policy}_{run_seed}.npz"))
                pos, gs, types = rec["pos"], rec["gs"], rec["types"]
                for t in GEOMETRY_TIMES_S:
                    step = int(round(t / config.TIMESTEP))
                    m = pos[step, 0]
                    created = rec["pkt_created"]
                    near = np.abs(created - step)
                    route, hops = "no delivered packet", ""
                    if len(near) and near.min() <= window:
                        k = int(np.argmin(near))
                        route = str(rec["pkt_path"][k])
                        at = pos[min(int(created[k]), len(pos) - 1)]
                        points = [gs if h == "GS" else at[int(h)]
                                  for h in route.split("-")]
                        hops = " ".join(
                            f"{np.linalg.norm(points[i + 1] - points[i]):.0f}"
                            for i in range(len(points) - 1))
                    for c in [i for i, kind in enumerate(types) if kind == "C"]:
                        rows.append({
                            "scenario": scenario, "policy": policy,
                            "run_seed": run_seed, "time_s": t, "c_id": c,
                            "c_to_line_m": point_to_segment(pos[step, c], gs, m),
                            "c_gs_distance_m": float(
                                np.linalg.norm(pos[step, c] - gs)),
                            "m_gs_distance_m": float(np.linalg.norm(m - gs)),
                            "route": route, "hop_lengths_m": hops,
                        })
    return rows


def s6_midpoint_rows(rec_dir: str) -> List[Dict[str, Any]]:
    """Scenario 6, once a second: the C-drone's distance from the midpoint."""
    rows: List[Dict[str, Any]] = []
    per_second = int(round(1.0 / config.TIMESTEP))
    for policy in POLICIES:
        for run_seed in RUN_SEEDS:
            rec = np.load(os.path.join(rec_dir, f"s6_{policy}_{run_seed}.npz"))
            pos, gs = rec["pos"], rec["gs"]
            for step in range(0, len(pos), per_second):
                middle = 0.5 * (gs + pos[step, 0])
                rows.append({
                    "policy": policy, "run_seed": run_seed,
                    "time_s": step * config.TIMESTEP,
                    "c_to_midpoint_m": float(np.linalg.norm(pos[step, 1]
                                                            - middle)),
                    "c_to_line_m": point_to_segment(pos[step, 1], gs,
                                                    pos[step, 0]),
                })
    return rows


SUMMARY_COLUMNS: Tuple[str, ...] = (
    ("total_loss",) + CAUSES
    + ("c_distance_flown_mean", "min_c_to_c_m", "pct_m_clipped_end")
)


def summarise(rows: Sequence[Dict[str, Any]],
              scenarios: Sequence[int] = tuple(SCENARIOS),
              policies: Sequence[str] = POLICIES,
              columns: Sequence[str] = SUMMARY_COLUMNS
              ) -> List[Dict[str, Any]]:
    """Means over the five seeds, per (scenario, policy)."""
    out = []
    for scenario in scenarios:
        for policy in policies:
            group = [r for r in rows
                     if r["scenario"] == scenario and r["policy"] == policy]
            row: Dict[str, Any] = {"scenario": scenario, "policy": policy,
                                   "seeds": len(group)}
            for column in columns:
                values = [r[column] for r in group
                          if not np.isnan(r[column])]
                row[column] = statistics.fmean(values) if values else ""
            row["min_c_to_c_m_worst"] = (
                min(r["min_c_to_c_m"] for r in group)
                if not np.isnan(group[0]["min_c_to_c_m"]) else "")
            out.append(row)
    return out


def print_tables(summary: Sequence[Dict[str, Any]],
                 geometry: Sequence[Dict[str, Any]],
                 s6: Sequence[Dict[str, Any]]) -> None:
    """Print the tables, from the rows just written."""
    print(f"\n  Mean over run seeds {RUN_SEEDS}, % of packets created")
    head = (f"  {'sc':>2} {'policy':>8} {'total':>6} "
            + " ".join(f"{c[:8]:>8}" for c in CAUSES)
            + f" {'C flown':>8} {'minCC':>6} {'worstCC':>7} {'clip%':>5}")
    print(head)
    for row in summary:
        def f(v, w, p):
            return f"{v:{w}.{p}f}" if v != "" else f"{'-':>{w}}"
        print(f"  {row['scenario']:>2.0f} {row['policy']:>8} "
              f"{row['total_loss']:6.2f} "
              + " ".join(f"{row[c]:8.2f}" for c in CAUSES)
              + f" {row['c_distance_flown_mean']:7.0f}m"
              f" {f(row['min_c_to_c_m'], 6, 1)}"
              f" {f(row['min_c_to_c_m_worst'], 7, 1)}"
              f" {row['pct_m_clipped_end']:5.0f}")
        if row["policy"] == POLICIES[-1]:
            print()

    print("  Geometry, mean over seeds: C distance from the GS-M segment (m)")
    for scenario in GEOMETRY_SCENARIOS:
        for policy in POLICIES:
            cells = []
            for t in GEOMETRY_TIMES_S:
                group = [r for r in geometry if r["scenario"] == scenario
                         and r["policy"] == policy and r["time_s"] == t]
                ids = sorted({r["c_id"] for r in group})
                per_c = "/".join(
                    f"{statistics.fmean(r['c_to_line_m'] for r in group if r['c_id'] == c):.0f}"
                    for c in ids)
                cells.append(f"t={t:.0f}: {per_c}")
            print(f"  s{scenario} {policy:>8}  " + "   ".join(cells))
    print("\n  Routes on seed 9901 (path; hop lengths in m)")
    for scenario in GEOMETRY_SCENARIOS:
        for policy in POLICIES:
            cells = []
            for t in GEOMETRY_TIMES_S:
                row = next(r for r in geometry if r["scenario"] == scenario
                           and r["policy"] == policy and r["time_s"] == t
                           and r["run_seed"] == 9901)
                cells.append(f"t={t:.0f}: {row['route']} [{row['hop_lengths_m']}]")
            print(f"  s{scenario} {policy:>8}  " + "   ".join(cells))

    print("\n  Scenario 6: C distance from the exact GS-M midpoint (m), "
          "mean over seeds")
    for policy in POLICIES:
        cells = []
        for t in (0, 10, 20, 30, 40, 60, 80, 99):
            values = [r["c_to_midpoint_m"] for r in s6
                      if r["policy"] == policy and r["time_s"] == t]
            cells.append(f"{t:>2}s {statistics.fmean(values):5.0f}")
        late = [r["c_to_midpoint_m"] for r in s6
                if r["policy"] == policy and r["time_s"] >= 40]
        print(f"  {policy:>8}  " + "  ".join(cells)
              + f"   | 40-99 s mean {statistics.fmean(late):.0f}, "
              f"max {max(late):.0f}")


def read_csv(path: str) -> List[Dict[str, Any]]:
    """Read a stamped CSV back, numbers as floats, blanks as "".

    The tables are printed from these, not from the rows in memory, so what
    is printed is what was saved.
    """
    import csv

    with open(path) as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith("#")))
    out = []
    for row in rows:
        parsed: Dict[str, Any] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if value not in ("", "nan") else ""
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def main() -> None:
    """Run every scenario, record it, and write the tables."""
    import torch

    parser = argparse.ArgumentParser(
        description="Phase 2c: the frozen direct actor on hand-built scenarios."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--model", type=str, default=None,
                        help="actor to run as 'direct'; default is the "
                             "direct model under --out-dir")
    parser.add_argument("--out", type=str, default=None,
                        help="output folder; default <out-dir>/2c_edge")
    parser.add_argument("--bigfleet", action="store_true",
                        help="run scenarios 9-13 into "
                             "<out-dir>/2c_edge_bigfleet instead")
    parser.add_argument("--actor", type=str, action="append", default=None,
                        help="with --bigfleet: NAME=PATH, repeatable; "
                             "replaces direct and spacing_large")
    args = parser.parse_args()
    if args.bigfleet:
        run_bigfleet(args)
        return

    edge_dir = args.out or os.path.join(args.out_dir, "2c_edge")
    rec_dir = os.path.join(edge_dir, "rec")
    os.makedirs(rec_dir, exist_ok=True)
    model = args.model or os.path.join(args.out_dir, DIRECT_MODEL)
    print(f"actor: {model}")
    state = torch.load(model, weights_only=True)

    print(f"placement seed {PLACEMENT_SEED}, run seeds {RUN_SEEDS}")
    jobs = [(scenario, policy, seed, state if policy == "direct" else None,
             rec_dir)
            for scenario in SCENARIOS for policy in POLICIES
            for seed in RUN_SEEDS]
    print(f"Running {len(jobs)} runs: {len(SCENARIOS)} scenarios x "
          f"{len(POLICIES)} policies x {len(RUN_SEEDS)} seeds ...", flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        rows = list(pool.map(run_one, jobs))

    agreed = sum(1 for r in rows if r["views_agree"])
    print(f"  views agreed in {agreed} of {len(rows)} runs")
    if agreed != len(rows):
        raise RuntimeError("the ACK views disagreed with the simulator")

    actor_text = ("frozen direct actor" if args.model is None else
                  f"frozen actor {os.path.basename(model)} (as 'direct')")
    note = (f"phase 2c edge cases, {actor_text}, placement seed "
            f"{PLACEMENT_SEED}, fleet set per scenario (not 18M+7C), flights "
            f"along the 45-degree diagonal, greedy routing, AVOID on, "
            f"NO training")
    run_columns = (["scenario", "policy", "run_seed", "num_m", "num_c"]
                   + list(SUMMARY_COLUMNS) + ["views_agree"])
    write_csv(os.path.join(edge_dir, "edge_runs.csv"), rows, run_columns,
              note)
    summary = summarise(rows)
    write_csv(os.path.join(edge_dir, "edge_summary.csv"), summary,
              ["scenario", "policy", "seeds"] + list(SUMMARY_COLUMNS)
              + ["min_c_to_c_m_worst"], note)
    geometry = geometry_rows(rec_dir)
    write_csv(os.path.join(edge_dir, "edge_geometry.csv"), geometry,
              ["scenario", "policy", "run_seed", "time_s", "c_id",
               "c_to_line_m", "c_gs_distance_m", "m_gs_distance_m", "route",
               "hop_lengths_m"], note)
    s6 = s6_midpoint_rows(rec_dir)
    write_csv(os.path.join(edge_dir, "edge_s6_midpoint.csv"), s6,
              ["policy", "run_seed", "time_s", "c_to_midpoint_m",
               "c_to_line_m"], note)

    print_tables(read_csv(os.path.join(edge_dir, "edge_summary.csv")),
                 read_csv(os.path.join(edge_dir, "edge_geometry.csv")),
                 read_csv(os.path.join(edge_dir, "edge_s6_midpoint.csv")))


def print_bigfleet(summary: Sequence[Dict[str, Any]],
                   policies: Sequence[str] = BIGFLEET_POLICIES) -> None:
    """Print the big-fleet tables, from the rows read back from disk."""
    print(f"\n  Mean over run seeds {RUN_SEEDS}, % of packets created")
    print(f"  {'sc':>2} {'policy':>13} {'total':>6} {'flying':>6} "
          f"{'staying':>7} " + " ".join(f"{c[:8]:>8}" for c in CAUSES))
    for row in summary:
        print(f"  {row['scenario']:>2.0f} {row['policy']:>13} "
              f"{row['total_loss']:6.2f} {row['loss_flying']:6.2f} "
              f"{row['loss_staying']:7.2f} "
              + " ".join(f"{row[c]:8.2f}" for c in CAUSES))
        if row["policy"] == policies[-1]:
            print()
    print("  Farthest C-drone from the GS (m), C metres flown, closest C pair")
    print(f"  {'sc':>2} {'policy':>13} {'t=30':>6} {'t=60':>6} {'t=95':>6} "
          f"{'C flown':>8} {'minCC':>6} {'worstCC':>7}")
    for row in summary:
        print(f"  {row['scenario']:>2.0f} {row['policy']:>13} "
              f"{row['far_c_30']:6.0f} {row['far_c_60']:6.0f} "
              f"{row['far_c_95']:6.0f} {row['c_distance_flown_mean']:7.0f}m "
              f"{row['min_c_to_c_m']:6.1f} {row['min_c_to_c_m_worst']:7.1f}")
        if row["policy"] == policies[-1]:
            print()


def run_bigfleet(args: Any) -> None:
    """Scenarios 9-13, four policies, five seeds, into 2c_edge_bigfleet."""
    import torch

    edge_dir = args.out or os.path.join(args.out_dir, "2c_edge_bigfleet")
    rec_dir = os.path.join(edge_dir, "rec")
    os.makedirs(rec_dir, exist_ok=True)
    states = {}
    if args.actor:
        # NAME=PATH pairs replace the default actors; paths as given.
        models = dict(pair.split("=", 1) for pair in args.actor)
    else:
        models = {name: os.path.join(args.out_dir, path)
                  for name, path in BIGFLEET_MODELS.items()}
    policies = ("hover", "midpoint") + tuple(models)
    for policy, full in models.items():
        print(f"  {policy:>13}: {full}")
        states[policy] = torch.load(full, weights_only=True)
    print(f"placement seed {PLACEMENT_SEED}, run seeds {RUN_SEEDS}")
    jobs = [(scenario, policy, seed, states.get(policy), rec_dir)
            for scenario in BIGFLEET for policy in policies
            for seed in RUN_SEEDS]
    print(f"Running {len(jobs)} runs: {len(BIGFLEET)} scenarios x "
          f"{len(policies)} policies x {len(RUN_SEEDS)} seeds ...",
          flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=context) as pool:
        rows = list(pool.map(run_one, jobs))
    agreed = sum(1 for r in rows if r["views_agree"])
    print(f"  views agreed in {agreed} of {len(rows)} runs")
    if agreed != len(rows):
        raise RuntimeError("the ACK views disagreed with the simulator")

    note = (f"phase 2c big-fleet edge cases 9-13, 18M+7C from the near_gs "
            f"start draw of placement seed {PLACEMENT_SEED}, staying M hover, "
            f"flying M 20 m/s no delay, greedy routing, AVOID on, actors "
            f"{', '.join(models)} greedy, NO training")
    columns = SUMMARY_COLUMNS + BIGFLEET_COLUMNS
    write_csv(os.path.join(edge_dir, "edge_runs.csv"), rows,
              ["scenario", "policy", "run_seed", "num_m", "num_c"]
              + list(columns) + ["views_agree"], note)
    summary = summarise(rows, tuple(BIGFLEET), policies, columns)
    write_csv(os.path.join(edge_dir, "edge_summary.csv"), summary,
              ["scenario", "policy", "seeds"] + list(columns)
              + ["min_c_to_c_m_worst"], note)
    print_bigfleet(read_csv(os.path.join(edge_dir, "edge_summary.csv")),
                   policies)


if __name__ == "__main__":
    main()
