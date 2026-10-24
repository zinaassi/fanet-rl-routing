"""
c_interface.py — What a C-drone agent sees, does, and is paid.

Phase 2c M0. The C-drone rules of Phase 2a were written in Python: look at
the table, pick a point, fly there. This module is the same job expressed as
something a network can be plugged into — an observation, an action and a
reward — so that M1 onward can replace the Python with a policy and compare
like with like.

The three pieces:

ACTION      a waypoint, every ``config.C_DECISION_INTERVAL`` seconds, given
            relative to the GS and normalised by ``ACTION_SCALE_M``. The
            drone flies straight to it at its own speed and hovers within
            ``config.C_MIDPOINT_STOP_M`` of it until its next decision. The
            sensor-based steering of Phase 2a stays underneath as a safety
            layer: it can turn the drone aside or hold it, but it never
            changes the waypoint.

OBSERVATION ``SELF_FEATURES`` numbers about the drone itself, plus
            ``NEIGHBOUR_FEATURES`` per entry in its neighbour table, with a
            mask. EVERYTHING comes from the table and the drone's own
            counters — never from another drone's true position. That is the
            point of the exercise, and tests/test_2c_interface.py checks it
            by moving the other drones and watching the observation not
            change.

REWARD      the network's delivery balance over the last second, shared by
            every C-drone, minus a small charge for flying:

                r = (delivered - lost) / created  -  c * (metres / speed)

            with ``c = config.C_FLIGHT_COST``. Logged from M0 and used for
            training from M2.

This module is torch-free, like the rest of ``fanet_sim``. The policy lives
in ``agents/``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fanet_sim import config
from fanet_sim.envs.channel import are_connected, euclidean_distance

#: Waypoints are given relative to the GS and divided by this, so the whole
#: arena fits inside [-1, 1] whichever corner the drone is heading for.
ACTION_SCALE_M: float = 450.0

#: How many numbers describe the drone itself, and each table entry.
SELF_FEATURES: int = 16
NEIGHBOUR_FEATURES: int = 16

#: The order of the type one-hot inside a neighbour's features.
TYPES: Tuple[str, ...] = ("M", "C", "GS")


def steps_per_decision() -> int:
    """Return how many steps pass between one waypoint and the next."""
    return max(1, int(round(config.C_DECISION_INTERVAL / config.TIMESTEP)))


def steps_per_second() -> int:
    """Return how many simulation steps make one second."""
    return max(1, int(round(1.0 / config.TIMESTEP)))


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

def waypoint_from_action(action: np.ndarray,
                         gs_position: np.ndarray) -> np.ndarray:
    """Turn a normalised action into a point in the arena.

    Args:
        action:      Two numbers, nominally in [-1, 1]. Anything outside is
                     allowed through and clipped by the arena below, so a
                     policy that saturates is not silently rescaled.
        gs_position: Where the ground station is.

    Returns:
        A waypoint inside the arena.
    """
    point = gs_position + ACTION_SCALE_M * np.asarray(action, dtype=np.float64)
    point[0] = min(max(point[0], 0.0), config.WIDTH)
    point[1] = min(max(point[1], 0.0), config.HEIGHT)
    return point


def action_from_waypoint(waypoint: np.ndarray,
                         gs_position: np.ndarray) -> np.ndarray:
    """The inverse of :func:`waypoint_from_action`, for scripted agents.

    Round-tripping a waypoint through this pair is NOT bit-exact: dividing
    by 450 and multiplying back changes the last bit of about a quarter of
    all points, by some 6e-14 m. That is far below anything the simulator can
    notice — a step is 3 m — but it is why the midpoint-through-the-interface
    check compares to a tolerance rather than bit-for-bit.

    Args:
        waypoint:    A point in the arena.
        gs_position: Where the ground station is.

    Returns:
        The normalised action that produces it.
    """
    return (np.asarray(waypoint, dtype=np.float64) - gs_position) / ACTION_SCALE_M


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

def _link_ack_ratio(drone: Any, key: Any) -> Tuple[float, float]:
    """Return this drone's measured success rate on one link, and a flag.

    Built from the drone's own hop-ACK book, so it is local knowledge. The
    flag says whether anything has been sent on that link at all; without it
    a link never used and a link that always fails both read as zero.

    Args:
        drone: The drone that owns the link.
        key:   The next hop.

    Returns:
        ``(rate, 1.0 if measured else 0.0)``.
    """
    row = drone.link_stats.get(key)
    if not row or not row["sent"]:
        return 0.0, 0.0
    return row["acked"] / row["sent"], 1.0


def _self_features(env: Any, drone: Any) -> np.ndarray:
    """Return the drone's own part of the observation.

    Args:
        env:   The environment, for the GS position and the clock.
        drone: The C-drone observing.

    Returns:
        ``SELF_FEATURES`` numbers.
    """
    gs = env.gs_position
    to_gs = gs - drone.position
    distance = float(np.linalg.norm(to_gs))
    heading = to_gs / distance if distance > 0.0 else np.zeros(2)

    queues = list(drone.link_queues.values())
    mean_fill = (float(np.mean([len(q) for q in queues])) if queues else 0.0)
    gs_queue = len(drone.link_queues["GS"]) if "GS" in drone.link_queues else 0

    relayed, sources = drone.relays_in_last_second(
        env.step_count, steps_per_second()
    )

    sent = sum(row["sent"] for row in drone.link_stats.values())
    acked = sum(row["acked"] for row in drone.link_stats.values())
    hop_ratio = acked / sent if sent else 0.0
    created = drone.packets_created
    # A C-drone creates nothing, so this is 0 for every C-drone. It is kept
    # because the observation is defined for any drone, and because an agent
    # that one day runs on an M-drone would want it.
    e2e_ratio = (drone.packets_gs_acked / created) if created else 0.0

    return np.array([
        drone.position[0] / ACTION_SCALE_M,
        drone.position[1] / ACTION_SCALE_M,
        distance / config.LOSS_50_DISTANCE_M,
        heading[0],
        heading[1],
        1.0 if "GS" in drone.hello_table else 0.0,
        gs_queue / config.QUEUE_CAPACITY,
        mean_fill / config.QUEUE_CAPACITY,
        relayed / 10.0,
        sources / 6.0,
        hop_ratio,
        e2e_ratio,
        len(drone.hello_table) / 25.0,
        env.step_count * config.TIMESTEP / 100.0,
        drone.velocity[0] / config.C_DRONE_SPEED,
        drone.velocity[1] / config.C_DRONE_SPEED,
    ], dtype=np.float64)


def _neighbour_features(env: Any, drone: Any, hello: Any) -> np.ndarray:
    """Return one table entry's part of the observation.

    The position is the one the neighbour advertised, carried forward by its
    advertised velocity for as long as the hello has been sitting there — the
    drone's best guess, not the truth.

    Args:
        env:   The environment, for the GS position and the clock.
        drone: The C-drone observing.
        hello: One entry of its neighbour table.

    Returns:
        ``NEIGHBOUR_FEATURES`` numbers.
    """
    gs = env.gs_position
    age = (env.step_count - hello.sent_at) * config.TIMESTEP
    guessed = hello.position + hello.velocity * age
    offset = guessed - drone.position

    mine = float(np.linalg.norm(gs - drone.position))
    theirs = float(np.linalg.norm(gs - guessed))
    rate, measured = _link_ack_ratio(drone, hello.sender_id)

    end_point = hello.end_point
    end_offset = (np.zeros(2) if end_point is None
                  else (end_point - drone.position) / ACTION_SCALE_M)

    return np.array([
        offset[0] / config.LOSS_50_DISTANCE_M,
        offset[1] / config.LOSS_50_DISTANCE_M,
        hello.velocity[0] / config.C_DRONE_SPEED,
        hello.velocity[1] / config.C_DRONE_SPEED,
        1.0 if hello.sender_type == "M" else 0.0,
        1.0 if hello.sender_type == "C" else 0.0,
        1.0 if hello.sender_type == "GS" else 0.0,
        hello.queue_fill / config.QUEUE_CAPACITY,
        hello.relayed / 10.0,
        (theirs - mine) / config.LOSS_50_DISTANCE_M,
        1.0 if are_connected(guessed, gs) else 0.0,
        rate,
        measured,
        age,
        end_offset[0],
        end_offset[1],
    ], dtype=np.float64)


def build_observation(env: Any, drone: Any) -> Dict[str, np.ndarray]:
    """Return everything one C-drone knows, as arrays.

    Args:
        env:   The environment.
        drone: The C-drone observing.

    Returns:
        ``self`` of shape (SELF_FEATURES,), ``neighbours`` of shape
        (n, NEIGHBOUR_FEATURES) and ``mask`` of shape (n,). The table is
        taken in sorted key order so the same situation always produces the
        same array.
    """
    keys = sorted(drone.hello_table, key=str)
    rows = [_neighbour_features(env, drone, drone.hello_table[key])
            for key in keys]
    neighbours = (np.stack(rows) if rows
                  else np.zeros((0, NEIGHBOUR_FEATURES), dtype=np.float64))
    return {
        "self": _self_features(env, drone),
        "neighbours": neighbours,
        "mask": np.ones(len(rows), dtype=np.float64),
    }


def pad_observations(
    observations: List[Dict[str, np.ndarray]],
    width: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Stack a batch of observations to a common neighbour count.

    Args:
        observations: One observation per decision.
        width:        Neighbour slots to pad to. Defaults to the widest
                      observation in the batch, or 1 when none has any.

    Returns:
        ``self`` (B, SELF_FEATURES), ``neighbours`` (B, width,
        NEIGHBOUR_FEATURES) and ``mask`` (B, width).
    """
    if width is None:
        width = max((len(o["mask"]) for o in observations), default=1) or 1
    batch = len(observations)
    selves = np.zeros((batch, SELF_FEATURES), dtype=np.float64)
    neighbours = np.zeros((batch, width, NEIGHBOUR_FEATURES), dtype=np.float64)
    mask = np.zeros((batch, width), dtype=np.float64)
    for index, observation in enumerate(observations):
        selves[index] = observation["self"]
        count = min(len(observation["mask"]), width)
        if count:
            neighbours[index, :count] = observation["neighbours"][:count]
            mask[index, :count] = 1.0
    return {"self": selves, "neighbours": neighbours, "mask": mask}


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def flight_reward(delivered: int, lost: int, created: int,
                  metres: float, speed: float) -> float:
    """Return one C-drone's reward for the second just ended.

    The delivery term is the whole network's, so every C-drone gets the same
    number: they are one team and there is no way to say which of them a
    delivered packet is owed to. The flight term is the drone's own, charged
    as a fraction of how far it could have flown, so the cost does not change
    meaning if the speed does.

    Args:
        delivered: Packets delivered network-wide in that second.
        lost:      Packets dropped network-wide in that second.
        created:   Packets created network-wide in that second.
        metres:    How far this drone flew in that second.
        speed:     Its top speed, in m/s.

    Returns:
        The reward. Zero delivery term when nothing was created.
    """
    balance = (delivered - lost) / created if created else 0.0
    if speed <= 0.0:
        return balance
    return balance - config.C_FLIGHT_COST * (metres / speed)
