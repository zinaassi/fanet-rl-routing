"""
qrouting.py — ARCHIVED Q-routing baseline (Phase 1A: out of scope).

Cut verbatim out of ``rl_sim/fanet_sim/envs/fanet_env.py`` when the Phase-1A
scope was fixed to the simulator plus the two non-learned routing rules
(greedy, random). Kept only for reference; nothing in the live simulator
imports it, and it will not run as-is (its ``Drone`` import path assumed the
pre-archive layout).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fanet_sim.envs.drone import Drone


class QRouter:
    """Simple Q-table router.

    State  = drone ID (current holder).
    Action = next-hop neighbour ID.
    Reward = negative delay (−1 per hop).

    The Q-table is a 2-D array indexed by [drone_id, neighbour_id].
    Unknown (drone_id, neighbour_id) entries default to 0.
    """

    def __init__(self, num_drones: int, alpha: float = 0.1, gamma: float = 0.9) -> None:
        """Initialise the Q-table.

        Args:
            num_drones: Total number of drones (determines table size).
            alpha:      Learning rate.
            gamma:      Discount factor.
        """
        self.q: np.ndarray = np.zeros((num_drones, num_drones), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, drone: Drone) -> Optional[int]:
        """Choose the next-hop neighbour ID with the highest Q-value.

        Falls back to greedy geographic routing if no neighbours exist.

        Args:
            drone: The drone currently holding the packet.

        Returns:
            Neighbour drone ID, or None if no neighbours.
        """
        if not drone.neighbors:
            return None
        nids = list(drone.neighbors.keys())
        # Pick the neighbour with the highest Q-value for this drone.
        best_nid = max(nids, key=lambda nid: self.q[drone.drone_id, nid])
        return best_nid

    def update(
        self,
        from_id: int,
        to_id: int,
        reward: float,
        next_drone: Optional[Drone],
    ) -> None:
        """Perform a Q-learning update.

        Args:
            from_id:    ID of the drone that forwarded the packet.
            to_id:      ID of the next-hop drone.
            reward:     Immediate reward (negative delay).
            next_drone: The next-hop Drone object (used for max-Q bootstrap).
        """
        current_q = self.q[from_id, to_id]
        if next_drone and next_drone.neighbors:
            max_next = max(
                self.q[to_id, nid] for nid in next_drone.neighbors
            )
        else:
            max_next = 0.0
        self.q[from_id, to_id] = current_q + self.alpha * (
            reward + self.gamma * max_next - current_q
        )

