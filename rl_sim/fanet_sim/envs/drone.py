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

Queues:
    ONE FIFO QUEUE PER OUTGOING LINK — one for each current neighbour, plus one
    for the GS link whenever the GS is in range. Each holds at most
    ``config.QUEUE_CAPACITY`` packets and releases at most
    ``config.MAX_TX_PER_STEP`` per step, oldest first. All of a drone's link
    queues send in the same step: there is no collision model, so there is no
    per-drone send limit. A packet put into a full link queue is refused
    (:meth:`enqueue` returns False) and the environment drops it at this drone
    with reason "queue_full".

    The next hop — and therefore which queue a packet joins — is decided when
    the packet is CREATED here or ARRIVES here, not when it is sent.

    Receiving is unlimited: a drone accepts every packet that reaches it and
    then routes it into one of its own link queues.

ACK bookkeeping:
    Every counter on a drone is built ONLY from information the drone has
    locally — a hop-by-hop ACK, an end-to-end GS ACK, or its own queues — never
    from the simulator's global view. Per outgoing link (keyed by next-hop
    drone id, or the string "GS"):

        offered            packets this drone tried to put in that queue
        dropped_queue_full those refused because the queue was full
        sent               transmissions attempted
        acked              transmissions the next hop confirmed receiving
        lost_channel       transmissions lost in the channel
        expired_in_queue   packets that died of TTL or hop limit while waiting

    A queue-full drop needs no ACK: it happens at this drone, in this drone's
    own queue. Per M-drone there is also the end-to-end view: packets created,
    packets the GS acknowledged, and packets whose TTL ran out unacknowledged.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from fanet_sim import config
from fanet_sim.envs.channel import (
    are_connected,
    euclidean_distance,
    link_quality,
)
from fanet_sim.envs.packet import Packet

#: A next-hop key: either a drone id, or the string "GS" for the ground station.
NextHop = Union[int, str]

#: What each drone tracks per outgoing link. ``offered`` is the denominator of
#: that link's loss; ``dropped_queue_full``, ``lost_channel`` and
#: ``expired_in_queue`` are the three ways an offered packet fails to get
#: across.
LINK_FIELDS = (
    "offered",
    "dropped_queue_full",
    "sent",
    "acked",
    "lost_channel",
    "expired_in_queue",
)


def _link_row() -> Dict[str, int]:
    """Return a zeroed per-link counter row."""
    return {field: 0 for field in LINK_FIELDS}


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
        link_queues: One FIFO packet list per outgoing link.
        candidates:  Dict mapping drone_id → Drone for every drone currently
                     in radio range.
        neighbors:   Dict mapping drone_id → Drone for every active link
                     (identical to *candidates* — there is no link cap).
                     Populated by the env each step.
        gs_position: Ground-station position as a NumPy array.
        link_queues: Dict mapping next-hop key → that link's FIFO packet list.
        link_stats:  Per outgoing link, a row of :data:`LINK_FIELDS` counts
                     over the whole episode.
        link_stats_measured: The same rows, counting only post-warm-up steps.
        packets_*_measured: The same source counters, restricted to packets
                     created inside the measurement window.
        packets_created:  Packets this drone originated (M-drones only).
        packets_gs_acked: Own packets the GS acknowledged (M-drones only).
        packets_lost_no_ack: Own packets whose TTL ran out with no GS ACK.
        outstanding: Own packet_id → creation step, for packets still awaiting
                     a GS ACK (M-drones only).
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
        # One FIFO queue per outgoing link, created on first use. Keys are
        # neighbour drone ids, plus "GS" for the ground-station link.
        self.link_queues: Dict[NextHop, List[Packet]] = defaultdict(list)
        self.candidates: Dict[int, "Drone"] = {}
        self.neighbors: Dict[int, "Drone"] = {}
        self.gs_position: np.ndarray = gs_position.astype(np.float64)

        # --- ACK-derived counters (the drone's own view of the network) ---
        # Keyed by next-hop: an int drone id, or the string "GS". Each entry
        # holds sent / acked / lost_channel / lost_queue_full.
        # ``link_stats`` is the drone's full book; ``link_stats_measured``
        # counts only what happened after the warm-up, and is what the reports
        # use so the ACK view covers the same window as the ground truth.
        self.link_stats: Dict[NextHop, Dict[str, int]] = defaultdict(_link_row)
        self.link_stats_measured: Dict[NextHop, Dict[str, int]] = defaultdict(_link_row)
        # End-to-end view, meaningful for M-drones (packet sources). The
        # ``_measured`` variants count only packets created inside the
        # measurement window.
        self.packets_created: int = 0
        self.packets_gs_acked: int = 0
        self.packets_lost_no_ack: int = 0
        self.packets_created_measured: int = 0
        self.packets_gs_acked_measured: int = 0
        self.packets_lost_no_ack_measured: int = 0
        self.outstanding: Dict[int, int] = {}

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

    def queue_is_full(self, next_hop: NextHop) -> bool:
        """True if this drone's queue for *next_hop* cannot accept another packet.

        Exact local knowledge: the queue belongs to this drone, so it needs no
        ACK or estimate to answer.

        Args:
            next_hop: Drone id, or "GS".
        """
        return len(self.link_queues[next_hop]) >= config.QUEUE_CAPACITY

    def queue_len(self, next_hop: NextHop) -> int:
        """Return how many packets are waiting on the link to *next_hop*."""
        return len(self.link_queues[next_hop])

    def total_queued(self) -> int:
        """Return how many packets this drone is holding across all link queues."""
        return sum(len(q) for q in self.link_queues.values())

    def queued_packets(self) -> List[Packet]:
        """Return every packet this drone is holding, across all link queues."""
        return [pkt for queue in self.link_queues.values() for pkt in queue]

    def enqueue(self, next_hop: NextHop, pkt: Packet) -> bool:
        """Try to add a packet to the tail of the queue for *next_hop*.

        Args:
            next_hop: The link this packet is committed to. Chosen when the
                      packet is created here or arrives here, and fixed from
                      then on.
            pkt:      The Packet to buffer.

        Returns:
            True if the packet was accepted, False if that link's queue was
            already at ``config.QUEUE_CAPACITY`` (the caller must then drop the
            packet with reason "queue_full").
        """
        if self.queue_is_full(next_hop):
            return False
        self.link_queues[next_hop].append(pkt)
        return True

    def dequeue_head(self, next_hop: NextHop, n: Optional[int] = None) -> List[Packet]:
        """Remove and return this step's send batch for ONE link.

        Args:
            next_hop: The link to release from.
            n:        How many packets to release. Defaults to
                      ``config.MAX_TX_PER_STEP``.

        Returns:
            Up to *n* Packet objects, oldest first (may be empty).
        """
        if n is None:
            n = config.MAX_TX_PER_STEP
        n = max(0, int(n))
        queue = self.link_queues[next_hop]
        batch, self.link_queues[next_hop] = queue[:n], queue[n:]
        return batch

    # ------------------------------------------------------------------
    # ACK bookkeeping (built only from what an ACK tells this drone)
    # ------------------------------------------------------------------

    def _bump_link(self, next_hop: NextHop, field: str, measured: bool) -> None:
        """Add one to *field* of this drone's counter row for *next_hop*.

        Args:
            next_hop: Drone id, or "GS".
            field:    One of :data:`LINK_FIELDS`.
            measured: Whether the warm-up is over, so the measured copy of the
                      row should move too.
        """
        self.link_stats[next_hop][field] += 1
        if measured:
            self.link_stats_measured[next_hop][field] += 1

    def note_offered(self, next_hop: NextHop, measured: bool = True) -> None:
        """Record that this drone tried to put a packet on the link to *next_hop*.

        This is the denominator of that link's loss.

        Args:
            next_hop: Drone id, or "GS".
            measured: True once the warm-up is over.
        """
        self._bump_link(next_hop, "offered", measured)

    def note_queue_full(self, next_hop: NextHop, measured: bool = True) -> None:
        """Record that the queue for *next_hop* refused a packet.

        No ACK is involved: the full queue is this drone's own.

        Args:
            next_hop: Drone id, or "GS".
            measured: True once the warm-up is over.
        """
        self._bump_link(next_hop, "dropped_queue_full", measured)

    def note_expired_in_queue(
        self, next_hop: NextHop, measured: bool = True
    ) -> None:
        """Record that a packet died of TTL or hop limit while waiting to be sent.

        Args:
            next_hop: The link whose queue the packet was waiting on.
            measured: True once the warm-up is over.
        """
        self._bump_link(next_hop, "expired_in_queue", measured)

    def note_sent(self, next_hop: NextHop, measured: bool = True) -> None:
        """Record one transmission attempt toward *next_hop*.

        Args:
            next_hop: Drone id, or "GS".
            measured: True once the warm-up is over.
        """
        self._bump_link(next_hop, "sent", measured)

    def note_hop_ack(
        self, next_hop: NextHop, outcome: str, measured: bool = True
    ) -> None:
        """Record what the hop-by-hop ACK reported for the last send.

        A send either arrives or is lost in the channel — the receiver never
        refuses it, since receiving is unlimited. Queue-full drops happen at
        the sender and are recorded by :meth:`note_queue_full` instead.

        Args:
            next_hop: The drone id (or "GS") the packet was sent to.
            outcome:  ``"ok"`` or ``"channel"``.
            measured: True once the warm-up is over.

        Raises:
            ValueError: If *outcome* is neither of those two values.
        """
        field = {"ok": "acked", "channel": "lost_channel"}.get(outcome)
        if field is None:
            raise ValueError(f"unknown hop-ACK outcome: {outcome!r}")
        self._bump_link(next_hop, field, measured)

    def note_created(self, pkt: Packet) -> None:
        """Record that this drone originated *pkt* and now awaits its GS ACK."""
        self.packets_created += 1
        if config.in_measurement_window(pkt.created_at):
            self.packets_created_measured += 1
        self.outstanding[pkt.packet_id] = pkt.created_at

    def note_gs_ack(self, packet_id: int) -> None:
        """Record the end-to-end GS ACK for one of this drone's own packets.

        Args:
            packet_id: The acknowledged packet. Ignored if this drone is not
                       waiting on it.
        """
        created_at = self.outstanding.pop(packet_id, None)
        if created_at is not None:
            self.packets_gs_acked += 1
            if config.in_measurement_window(created_at):
                self.packets_gs_acked_measured += 1

    def expire_unacked(self, current_step: int, ttl: Optional[int] = None) -> int:
        """Give up on own packets whose TTL ran out with no GS ACK.

        This is the source-side timer: the drone knows when it created each
        packet and knows the TTL, so it can conclude on its own that a packet
        is lost. No global information is used.

        Args:
            current_step: The current simulation timestep.
            ttl:          Packet lifetime in steps. Defaults to
                          ``config.PACKET_TTL``.

        Returns:
            How many packets were given up on during this call.
        """
        if ttl is None:
            ttl = config.PACKET_TTL
        timed_out = [
            pid for pid, created in self.outstanding.items()
            if current_step - created >= ttl
        ]
        for pid in timed_out:
            created_at = self.outstanding.pop(pid)
            if config.in_measurement_window(created_at):
                self.packets_lost_no_ack_measured += 1
        self.packets_lost_no_ack += len(timed_out)
        return len(timed_out)

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
            - ``queue_length``        Packets buffered across all link queues.
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
            "queue_length": self.total_queued(),
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
            f"e_motion={self.energy_motion:.0f}J, queued={self.total_queued()})"
        )
