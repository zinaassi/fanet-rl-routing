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

Reward (local only — no global metrics):
    + increase in num_neighbors
    + improvement in mean local link quality
    - motion energy consumed this step
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
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute the local reward for one C-drone move.

        Args:
            prev_state:     The C-drone's state *before* it moved this step.
            new_state:      Its state *after* moving and re-selecting links.
            distance_moved: Metres travelled this step (drives motion energy).
            weights:        Optional override for
                            ``config.TOPOLOGY_REWARD_WEIGHTS``.

        Returns:
            The scalar local reward (float).
        """
        w = config.TOPOLOGY_REWARD_WEIGHTS if weights is None else weights

        d_neighbors = new_state["num_neighbors"] - prev_state["num_neighbors"]
        d_quality = (
            _mean(new_state["link_quality"].values())
            - _mean(prev_state["link_quality"].values())
        )
        motion_energy = config.ENERGY_PER_MOVE * distance_moved

        return (
            w["num_neighbors"] * d_neighbors
            + w["link_quality"] * d_quality
            - w["motion_energy"] * motion_energy
        )
