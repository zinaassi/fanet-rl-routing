"""
drone.py — Drone agent class for the FANET simulator.

Each drone object tracks its own position, velocity, energy, packet queue,
and neighbour state. Both M-drones (mission) and C-drones (communication)
are represented by the same class; the *drone_type* attribute distinguishes
them.

Mobility model (phase 1):
    Waypoint model — drone flies toward the current waypoint at constant
    speed.  When it arrives it advances to the next waypoint cyclically.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from fanet_sim import config
from fanet_sim.envs.channel import link_quality, euclidean_distance
from fanet_sim.envs.packet import Packet


class Drone:
    """A FANET drone that moves on a waypoint path and relays packets.

    Attributes:
        drone_id:    Unique integer identifier.
        drone_type:  'M' for mission drone, 'C' for communication drone.
        position:    Current (x, y) position as a NumPy float64 array.
        velocity:    Current (vx, vy) velocity vector as a NumPy float64 array.
        speed:       Scalar speed in m/s (constant throughout episode).
        waypoints:   Ordered list of (x, y) waypoints (NumPy arrays).
        wp_index:    Index into *waypoints* of the current target waypoint.
        energy:      Remaining energy in joules.
        queue:       List of Packet objects this drone is holding.
        neighbors:   Dict mapping neighbour drone_id → Drone for drones
                     currently within comm_range.  Populated by the env.
        gs_position: Ground-station position as a NumPy array.
    """

    def __init__(
        self,
        drone_id: int,
        drone_type: str,
        initial_position: np.ndarray,
        waypoints: List[np.ndarray],
        speed: float,
        gs_position: np.ndarray,
    ) -> None:
        """Initialise a drone.

        Args:
            drone_id:         Unique integer identifier (0-indexed).
            drone_type:       'M' or 'C'.
            initial_position: Starting (x, y) position.
            waypoints:        Cyclic list of (x, y) waypoints.
            speed:            Constant flight speed in m/s.
            gs_position:      (x, y) position of the ground station.
        """
        self.drone_id: int = drone_id
        self.drone_type: str = drone_type
        self.position: np.ndarray = initial_position.astype(np.float64)
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float64)
        self.speed: float = float(speed)
        self.waypoints: List[np.ndarray] = [w.astype(np.float64) for w in waypoints]
        self.wp_index: int = 0
        self.energy: float = config.INITIAL_ENERGY
        self.queue: List[Packet] = []
        self.neighbors: Dict[int, "Drone"] = {}
        self.gs_position: np.ndarray = gs_position.astype(np.float64)

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def step_move(self, dt: float = config.TIMESTEP) -> float:
        """Advance position by one timestep toward the current waypoint.

        Updates *position*, *velocity*, and *energy*.

        Args:
            dt: Duration of this timestep in seconds.

        Returns:
            Distance moved in metres.
        """
        target = self.waypoints[self.wp_index]
        direction = target - self.position
        dist_to_wp = float(np.linalg.norm(direction))

        if dist_to_wp < config.WAYPOINT_ARRIVAL_THRESHOLD:
            # Arrived — advance to next waypoint
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)
            target = self.waypoints[self.wp_index]
            direction = target - self.position
            dist_to_wp = float(np.linalg.norm(direction))

        if dist_to_wp < 1e-9:
            # Already at the waypoint (degenerate case)
            self.velocity = np.zeros(2, dtype=np.float64)
            return 0.0

        unit = direction / dist_to_wp
        max_move = self.speed * dt
        actual_move = min(max_move, dist_to_wp)

        self.velocity = unit * self.speed
        self.position = self.position + unit * actual_move
        self.energy -= config.ENERGY_PER_MOVE * actual_move
        self.energy = max(0.0, self.energy)

        return actual_move

    # ------------------------------------------------------------------
    # Neighbour state (populated by the environment each step)
    # ------------------------------------------------------------------

    def update_neighbors(self, all_drones: List["Drone"]) -> None:
        """Recompute the neighbour set from the full drone list.

        A drone is a neighbour if it is within COMM_RANGE and is not *self*.

        Args:
            all_drones: Every Drone object in the simulation.
        """
        self.neighbors = {}
        for other in all_drones:
            if other.drone_id == self.drone_id:
                continue
            dist = euclidean_distance(self.position, other.position)
            if dist < config.COMM_RANGE:
                self.neighbors[other.drone_id] = other

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
        """Deduct one packet-transmission energy unit from the battery."""
        self.energy -= config.ENERGY_PER_TX
        self.energy = max(0.0, self.energy)

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
        is_connected_to_gs = dist_to_gs < config.COMM_RANGE

        return {
            "position": tuple(self.position),
            "velocity": tuple(self.velocity),
            "distance_to_gs": dist_to_gs,
            "residual_energy": self.energy,
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
            f"pos={pos}, energy={self.energy:.0f}J, queue={len(self.queue)})"
        )
