"""
drone.py — Drone agent class for the FANET simulator.

Each drone object tracks its own position, velocity, energy, packet queue,
and neighbour state. Both M-drones (mission) and C-drones (communication)
are represented by the same class; the *drone_type* attribute distinguishes
them.

Mobility models:
    M-drones — straight line from a fixed start point to a fixed end point.
               When the end point is reached the drone stops moving.
    C-drones — controlled externally by the topology agent, which commands a
               movement vector each step via :meth:`Drone.apply_velocity`.

Link selection:
    There is no link cap. Each step a drone gathers every in-range *candidate*
    (:meth:`update_candidates`) and keeps all of them as its active
    ``neighbors`` (:meth:`update_neighbors`). Link existence is decided purely
    by the FSPL received-power test in :mod:`fanet_sim.envs.channel`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from fanet_sim import config
from fanet_sim.envs.channel import (
    are_connected,
    euclidean_distance,
    link_quality,
)
from fanet_sim.envs.packet import Packet


class Drone:
    """A FANET drone that flies a mobility model and relays packets.

    Attributes:
        drone_id:    Unique integer identifier.
        drone_type:  'M' for mission drone, 'C' for communication drone.
        position:    Current (x, y) position as a NumPy float64 array.
        velocity:    Current (vx, vy) velocity vector as a NumPy float64 array.
        speed:       Scalar max speed in m/s (constant throughout episode).
        start_point: (x, y) start point (M-drones fly away from here).
        end_point:   (x, y) end point (M-drones stop here).
        arrived:     True once an M-drone has reached its end point.
        energy:      Remaining energy in joules.
        queue:       List of Packet objects this drone is holding.
        candidates:  Dict mapping drone_id → Drone for every drone currently
                     in radio range.
        neighbors:   Dict mapping drone_id → Drone for every active link
                     (identical to *candidates* — there is no link cap).
                     Populated by the env each step.
        gs_position: Ground-station position as a NumPy array.
    """

    def __init__(
        self,
        drone_id: int,
        drone_type: str,
        initial_position: np.ndarray,
        speed: float,
        gs_position: np.ndarray,
        end_point: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise a drone.

        Args:
            drone_id:         Unique integer identifier (0-indexed).
            drone_type:       'M' or 'C'.
            initial_position: Starting (x, y) position (also the start point).
            speed:            Max flight speed in m/s.
            gs_position:      (x, y) position of the ground station.
            end_point:        (x, y) destination for M-drones. C-drones ignore
                              this (they are steered by the topology agent); if
                              omitted it defaults to the start point.
        """
        self.drone_id: int = drone_id
        self.drone_type: str = drone_type
        self.position: np.ndarray = initial_position.astype(np.float64)
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float64)
        self.speed: float = float(speed)
        self.start_point: np.ndarray = self.position.copy()
        self.end_point: np.ndarray = (
            self.position.copy() if end_point is None else end_point.astype(np.float64)
        )
        self.arrived: bool = False
        # Energy is tracked as two independent budgets so the topology agent's
        # motion cost (Phase 4) does not get hidden inside the radio cost.
        self.energy_radio: float = config.INITIAL_ENERGY
        self.energy_motion: float = config.INITIAL_ENERGY
        self.queue: List[Packet] = []
        self.candidates: Dict[int, "Drone"] = {}
        self.neighbors: Dict[int, "Drone"] = {}
        self.gs_position: np.ndarray = gs_position.astype(np.float64)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def step_move(self, dt: float = config.TIMESTEP) -> float:
        """Advance an M-drone one timestep in a straight line toward its end.

        The drone flies from its start point toward :attr:`end_point` at
        constant speed and stops once it arrives (within
        ``config.WAYPOINT_ARRIVAL_THRESHOLD`` metres). C-drones do not use
        this method — they are steered by the topology agent through
        :meth:`apply_velocity` — so calling it on a C-drone is a no-op.

        Args:
            dt: Duration of this timestep in seconds.

        Returns:
            Distance moved in metres.
        """
        if self.drone_type != "M":
            # C-drones move via apply_velocity(); nothing to do here.
            self.velocity = np.zeros(2, dtype=np.float64)
            return 0.0

        direction = self.end_point - self.position
        dist_to_end = float(np.linalg.norm(direction))

        if dist_to_end < config.WAYPOINT_ARRIVAL_THRESHOLD:
            # Arrived at the end point — stop and stay put for good.
            self.velocity = np.zeros(2, dtype=np.float64)
            self.arrived = True
            return 0.0

        unit = direction / dist_to_end
        max_move = self.speed * dt
        actual_move = min(max_move, dist_to_end)

        self.velocity = unit * self.speed
        self.position = self.position + unit * actual_move
        self.energy_motion -= config.ENERGY_PER_MOVE * actual_move
        self.energy_motion = max(0.0, self.energy_motion)

        return actual_move

    def apply_velocity(self, delta: np.ndarray, dt: float = config.TIMESTEP) -> float:
        """Move the drone by a commanded vector (used by the topology agent).

        *delta* is a desired displacement vector ``[dx, dy]`` in metres for
        this step. Its magnitude is capped to ``speed * dt`` so the command
        cannot exceed the drone's physical speed. Updates *position*,
        *velocity*, and the motion-energy budget, and keeps the drone inside
        the simulation area.

        Args:
            delta: Desired movement vector ``[dx, dy]`` in metres.
            dt:    Duration of this timestep in seconds.

        Returns:
            Distance moved in metres.
        """
        delta = np.asarray(delta, dtype=np.float64)
        mag = float(np.linalg.norm(delta))
        if mag < 1e-9:
            self.velocity = np.zeros(2, dtype=np.float64)
            return 0.0

        max_move = self.speed * dt
        actual_move = min(mag, max_move)
        unit = delta / mag

        self.velocity = unit * (actual_move / dt)
        new_pos = self.position + unit * actual_move
        # Keep the drone inside the [0, WIDTH] x [0, HEIGHT] arena.
        new_pos[0] = min(max(new_pos[0], 0.0), config.WIDTH)
        new_pos[1] = min(max(new_pos[1], 0.0), config.HEIGHT)
        # Charge motion energy on the distance actually travelled after clamping.
        moved = float(np.linalg.norm(new_pos - self.position))
        self.position = new_pos
        self.energy_motion -= config.ENERGY_PER_MOVE * moved
        self.energy_motion = max(0.0, self.energy_motion)

        return moved

    # ------------------------------------------------------------------
    # Neighbour state (populated by the environment each step)
    # ------------------------------------------------------------------

    def update_candidates(self, all_drones: List["Drone"]) -> None:
        """Recompute the pool of in-range candidate links.

        A drone is a candidate if the FSPL received-signal test in
        :func:`fanet_sim.envs.channel.are_connected` passes — i.e. the
        received power at the receiver clears RX_SENSITIVITY_DBM. This is the
        passive "who can I hear" set; the active top-K links are then chosen
        from it by :meth:`update_neighbors`.

        Args:
            all_drones: Every Drone object in the simulation.
        """
        self.candidates = {}
        for other in all_drones:
            if other.drone_id == self.drone_id:
                continue
            if are_connected(self.position, other.position):
                self.candidates[other.drone_id] = other

    def update_neighbors(self) -> None:
        """Promote every in-range candidate to an active neighbour link.

        Phase 1A has no link cap: a drone keeps a link to every drone it can
        hear, so the active neighbour set is exactly the candidate pool.
        Requires :meth:`update_candidates` to have been called first.
        """
        self.neighbors = dict(self.candidates)

    # ------------------------------------------------------------------
    # Packet handling
    # ------------------------------------------------------------------

    def enqueue(self, pkt: Packet) -> None:
        """Add a packet to this drone's transmit queue.

        Args:
            pkt: The Packet to buffer.
        """
        self.queue.append(pkt)

    def dequeue_all(self) -> List[Packet]:
        """Remove and return all packets in the queue.

        Returns:
            List of Packet objects (may be empty).
        """
        pkts, self.queue = self.queue, []
        return pkts

    def consume_tx_energy(self) -> None:
        """Deduct one packet-transmission energy unit from the radio battery."""
        self.energy_radio -= config.ENERGY_PER_TX
        self.energy_radio = max(0.0, self.energy_radio)

    def consume_rx_energy(self, n_packets: int = 1) -> None:
        """Deduct receive energy for *n_packets* received this step.

        Args:
            n_packets: Number of packets received during the current step.
        """
        self.energy_radio -= config.ENERGY_PER_RX * n_packets
        self.energy_radio = max(0.0, self.energy_radio)

    def consume_idle_energy(self) -> None:
        """Deduct one timestep of idle/listen energy from the radio battery."""
        self.energy_radio -= config.ENERGY_PER_IDLE
        self.energy_radio = max(0.0, self.energy_radio)

    # ------------------------------------------------------------------
    # State vector
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Return the full feature vector for this drone at the current step.

        This dict is the input to the GNN / RL agent in later phases.

        Returns:
            A dict with the following keys:

            - ``position``            (x, y) tuple of current coordinates.
            - ``velocity``            (vx, vy) current velocity vector.
            - ``distance_to_gs``      Euclidean distance to the ground station.
            - ``residual_energy``     Remaining energy in joules.
            - ``queue_length``        Number of packets currently buffered.
            - ``num_neighbors``       Number of drones within comm range.
            - ``neighbor_ids``        List of neighbour drone IDs.
            - ``neighbor_positions``  List of (x, y) for each neighbour.
            - ``neighbor_distances``  List of distances to each neighbour.
            - ``link_quality``        Dict {neighbour_id: float 0-1}.
            - ``is_connected_to_gs``  Bool — can this drone reach GS in one hop?
        """
        neighbor_ids: List[int] = []
        neighbor_positions: List[Tuple[float, float]] = []
        neighbor_distances: List[float] = []
        lq_map: Dict[int, float] = {}

        for nid, nbr in self.neighbors.items():
            dist = euclidean_distance(self.position, nbr.position)
            lq = link_quality(dist)
            neighbor_ids.append(nid)
            neighbor_positions.append(tuple(nbr.position))
            neighbor_distances.append(dist)
            lq_map[nid] = lq

        dist_to_gs = euclidean_distance(self.position, self.gs_position)
        is_connected_to_gs = are_connected(self.position, self.gs_position)

        return {
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "distance_to_gs": dist_to_gs,
            "residual_energy": self.energy_radio + self.energy_motion,
            "energy_radio": self.energy_radio,
            "energy_motion": self.energy_motion,
            "queue_length": len(self.queue),
            "num_neighbors": len(self.neighbors),
            "neighbor_ids": neighbor_ids,
            "neighbor_positions": neighbor_positions,
            "neighbor_distances": neighbor_distances,
            "link_quality": lq_map,
            "is_connected_to_gs": is_connected_to_gs,
        }

    def __repr__(self) -> str:
        pos = tuple(self.position.round(1))
        return (
            f"Drone(id={self.drone_id}, type={self.drone_type}, "
            f"pos={pos}, e_radio={self.energy_radio:.0f}J, "
            f"e_motion={self.energy_motion:.0f}J, queue={len(self.queue)})"
        )
