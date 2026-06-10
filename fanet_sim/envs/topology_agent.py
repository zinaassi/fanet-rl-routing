"""
topology_agent.py — Topology controller for C-drones.

C-drones have no mission; their only job is to improve network connectivity.
In phase 1 they followed scripted waypoints. They are now steered by an
untrained PyTorch MLP (one per C-drone, living on the drone as
``Drone.topo_policy``). This module is the thin controller that turns a
C-drone's local state into a movement command and computes its local reward;
the network weights themselves are owned by the drone. The interface
(``act`` + ``reward``) is unchanged so a trained policy can drop in later.

The policy uses local information only, taken from ``Drone.get_state()``
(pos_x, pos_y, vel_x, vel_y, num_neighbors, mean_link_quality,
mean_distance_to_neighbours, queue_length) and outputs a movement vector
``[dx, dy]`` clamped to ``config.TOPOLOGY_MAX_STEP_M`` metres per axis.

Reward — RELAY COVERAGE objective (local):
    + coverage : number of MISSION (M) drones within radio range (settling value)
    + progress : metres of distance REDUCED toward the nearest M-drone this step,
                 scaled by the max step size (potential-based shaping — the dense
                 per-step signal that actually drives movement toward relays)
    - motion   : small penalty on motion energy consumed this step

The C-drone's job is to relay mission traffic, so it homes toward mission drones
and sits where it covers them. ``coverage`` alone is sparse (an integer that only
changes when an M-drone crosses the range boundary); the arena is ~2475 m across
but a C-drone moves ≤3 m/step, so a static distance term changes only ~0.002/step
— below the noise floor. ``progress`` fixes that: a full-speed step straight at
the nearest M-drone yields ≈+1 reward, a strong learnable signal. Only M-drones
count, so C-drones do not chase each other. Both quantities are supplied by the
environment (which has the global drone list); ``prev_state``/``new_state`` are
kept only for interface compatibility.

Design history: earlier rewards used the *change* in own-neighbour count (made
the borders a zero-signal trap), then an absolute own-neighbour count (froze
drones once they had any neighbours), then a static M-proximity term (too weak
per step to overcome the arena/step-size scale). This progress-shaped relay
reward is the version that actually produces movement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from fanet_sim import config

if TYPE_CHECKING:
    from fanet_sim.envs.drone import Drone


def _mean(values) -> float:
    """Return the mean of *values*, or 0.0 if empty."""
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TopologyAgent:
    """Controller that drives each C-drone's own topology MLP.

    Stateless: a single shared instance can drive every C-drone, because the
    learnable weights live per-drone in ``Drone.topo_policy``. The RL topology
    agent (phase 4) trains those weights while keeping these signatures.
    """

    def act(self, drone: "Drone", state: dict) -> np.ndarray:
        """Return a movement vector ``[dx, dy]`` for *drone* from its state.

        Delegates to the drone's own topology MLP (``drone.policy_move``),
        which builds the 8 local features, runs the network, and clamps the
        output to the max step size.

        Args:
            drone: The C-drone being steered (owns the topology MLP).
            state: A ``Drone.get_state()`` dict for that drone.

        Returns:
            A length-2 NumPy array — the desired displacement this step.
        """
        return drone.policy_move(state)

    def reward(
        self,
        prev_state: dict,
        new_state: dict,
        distance_moved: float,
        coverage: int = 0,
        progress: float = 0.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute the relay-coverage reward for one C-drone move.

        Args:
            prev_state:     State before the move (interface compat; unused).
            new_state:      State after the move (interface compat; unused).
            distance_moved: Metres travelled this step (drives motion energy).
            coverage:       Number of M-drones currently within radio range.
            progress:       Metres of distance reduced toward the nearest M-drone
                            this step (negative if the C-drone moved away).
            weights:        Optional override for
                            ``config.TOPOLOGY_REWARD_WEIGHTS``.

        Returns:
            The scalar local reward (float).
        """
        w = config.TOPOLOGY_REWARD_WEIGHTS if weights is None else weights

        # Scale progress by the max step so a full-speed approach ≈ +1.
        progress_scaled = progress / config.TOPOLOGY_MAX_STEP_M
        motion_energy = config.ENERGY_PER_MOVE * distance_moved

        return (
            w["coverage"] * coverage
            + w["progress"] * progress_scaled
            - w["motion_energy"] * motion_energy
        )
