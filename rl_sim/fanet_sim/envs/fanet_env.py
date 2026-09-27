"""
fanet_env.py — Main FANET simulation environment.

Provides a PettingZoo-style interface (reset / step) so that a MARL
framework can be plugged in during a later phase without rewriting the
core.  In phase 1A the *actions* argument to step() is ignored and the
selected non-learned routing rule runs internally.

Routing rules implemented here:
    - Greedy geographic routing  (default)
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.channel import are_connected, euclidean_distance
from fanet_sim.envs.drone import Drone
from fanet_sim.envs.packet import DropReason, Packet, PacketFactory
from fanet_sim.utils.event_log import EventLogger
from fanet_sim.utils.metrics import connectivity_sample


# ---------------------------------------------------------------------------
# Greedy geographic routing
# ---------------------------------------------------------------------------

def greedy_next_hop(
    drone: Drone,
    neighbors: Dict[int, Drone],
    gs_position: np.ndarray,
) -> Optional[Drone]:
    """Select the best next hop using greedy geographic forwarding.

    Forwards the packet to whichever neighbour is closest to the ground
    station.  If no neighbour is closer to the GS than the current drone,
    returns None (routing void).

    Args:
        drone:       The drone currently holding the packet.
        neighbors:   Dict of candidate next-hop drones keyed by ID.
        gs_position: Ground-station position as a NumPy array.

    Returns:
        The Drone object to forward to, or None if no improvement is found.
    """
    if not neighbors:
        return None

    best: Optional[Drone] = min(
        neighbors.values(),
        key=lambda n: euclidean_distance(n.position, gs_position),
    )
    current_dist = euclidean_distance(drone.position, gs_position)
    best_dist = euclidean_distance(best.position, gs_position)

    if best_dist < current_dist:
        return best
    return None  # routing void


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class FANETEnv:
    """FANET simulation environment.

    Exposes reset() and step() following a PettingZoo-like MARL interface.
    In phase 1, the *actions* argument is ignored; greedy routing runs
    internally.

    Attributes:
        drones:           List of all Drone objects (M then C).
        gs_position:      Ground-station position as a NumPy array.
        step_count:       Current timestep index.
        all_packets:      Every Packet ever created in this episode.
        delivered:        Packets that reached the GS.
        dropped:          Packets that were dropped.
        active_links:     Set of (id_a, id_b) pairs that were active this step
                          (used by the visualiser).
        tx_events:        List of (from_id, to_id) transmissions this step.
        routing:          'greedy'.
        rng:              NumPy random generator.
    """

    def __init__(
        self,
        routing: str = "greedy",
        log_path: Optional[str] = None,
        episode_id: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        """Create the environment (does NOT run reset automatically).

        Args:
            routing:     Routing rule to use: 'greedy'.
            log_path:    Path to write the Stage-1 JSONL event log. If None,
                         a default of ``{config.LOG_DIR}/episode_{id}.jsonl``
                         is used.
            episode_id:  Integer episode identifier, embedded in every log
                         record so multiple episodes can be concatenated.
            seed:        RNG seed for this run. Defaults to config.RANDOM_SEED.
                         Recorded in the episode-meta log record so the run
                         is reproducible.
        """
        self.routing = routing
        self.episode_id = episode_id
        self.seed = config.RANDOM_SEED if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)

        self.gs_position: np.ndarray = np.array(config.GS_POSITION, dtype=np.float64)
        self._factory = PacketFactory(
            ttl=config.PACKET_TTL,
            max_hops=config.MAX_HOPS,
            size_bytes=config.PACKET_SIZE,
        )

        # Populated by reset()
        self.drones: List[Drone] = []
        self.step_count: int = 0
        self.all_packets: List[Packet] = []
        self.delivered: List[Packet] = []
        self.dropped: List[Packet] = []
        self.active_links: set = set()
        self.tx_events: List[Tuple[int, int]] = []

        # Stage-1 event logger
        if log_path is None:
            log_path = os.path.join(config.LOG_DIR, f"episode_{episode_id}.jsonl")
        self.log_path = log_path
        self._logger: Optional[EventLogger] = None
        # Per-step receive counts, used to charge rx energy on the recipient.
        self._rx_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def reset(self) -> Dict[int, dict]:
        """Reset the environment and start a new episode.

        Opens a fresh JSONL event log at ``self.log_path`` (truncating any
        prior file at that path) and writes the per-episode metadata record.

        Returns:
            observations: Dict mapping drone_id → state dict (from get_state()).
        """
        # Re-seed so reset() is reproducible regardless of how many episodes
        # have already been run with this env instance.
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)

        self._factory.reset()
        self.step_count = 0
        self.all_packets = []
        self.delivered = []
        self.dropped = []
        self.active_links = set()
        self.tx_events = []
        self._rx_counts = defaultdict(int)

        # Open a new logger for this episode (close any prior one).
        if self._logger is not None:
            self._logger.close()
        self._logger = EventLogger(self.log_path, episode_id=self.episode_id)

        self.drones = self._create_drones()

        # Compute initial candidate pools and active links.
        self._recompute_links()

        # Episode metadata — the seed MUST be recorded (spec §D).
        self._logger.log_episode_meta(
            seed=self.seed,
            num_drones=len(self.drones),
            num_M=config.NUM_M_DRONES,
            num_C=config.NUM_C_DRONES,
            episode_length=config.MAX_STEPS,
            mobility_params={
                "speed_min": config.DRONE_SPEED_MIN,
                "speed_max": config.DRONE_SPEED_MAX,
                "m_drone_mobility": config.M_DRONE_MOBILITY,
                "area_width": config.WIDTH,
                "area_height": config.HEIGHT,
            },
            traffic_load={
                "packet_rate_per_M_per_step": config.PACKET_RATE,
                "packet_size_bytes": config.PACKET_SIZE,
                "ttl": config.PACKET_TTL,
                "max_hops": config.MAX_HOPS,
            },
            connectivity_model_params={
                "model": "FSPL",
                "pt_dbm": channel.PT_DBM,
                "gt_dbi": channel.GT_DBI,
                "gr_dbi": channel.GR_DBI,
                "f_hz": channel.F_HZ,
                "rx_sensitivity_dbm": channel.RX_SENSITIVITY_DBM,
                "link_budget_db": channel.LINK_BUDGET_DB,
                "max_link_distance_m": channel.MAX_LINK_DISTANCE_M,
                "timestep_s": config.TIMESTEP,
                "routing": self.routing,
            },
            anchor={
                "paper": "IQMR",
                "citation": "Sharvari et al., 2024",
                "arxiv": "2408.09109",
                "section": "V.A",
                "notes": (
                    "IQMR-faithful: speed range (10-30 m/s), transmit power "
                    "(1 W = 30 dBm), energy budget (11.1 V x 5200 mAh = "
                    "207792 J), radio range (250 m), AND deployment area "
                    "(1750x1750 m ~= 3.06 km^2 ~= IQMR's 1000 m-radius disk "
                    "of 3.14 km^2) all match IQMR. Receiver sensitivity "
                    "(-54 dBm) is derived from the FSPL model to reproduce "
                    "IQMR's 250 m range at IQMR's 1 W transmit power, not "
                    "picked arbitrarily. "
                    "Deviations: 2D instead of 3D; drone count split 36 M + "
                    "14 C is our own (IQMR has no M/C distinction)."
                ),
            },
        )

        # Initial step-state and drone-state snapshots at t=0.
        self._log_step_and_drone_state()

        return {d.drone_id: d.get_state() for d in self.drones}

    def _create_drones(self) -> List[Drone]:
        """Create and return all drones.

        M-drones get a random start point and a random end point and fly the
        straight line between them. C-drones get a random start point only;
        they are steered each step by the topology agent.

        Returns:
            List of Drone objects (M-drones first, then C-drones).
        """
        drones: List[Drone] = []
        drone_id = 0

        for _ in range(config.NUM_M_DRONES):
            start = self._random_point()
            end = self._random_point()
            drones.append(Drone(
                drone_id=drone_id,
                drone_type="M",
                initial_position=start,
                speed=self._random_speed(),
                gs_position=self.gs_position,
                end_point=end,
            ))
            drone_id += 1

        for _ in range(config.NUM_C_DRONES):
            drones.append(Drone(
                drone_id=drone_id,
                drone_type="C",
                initial_position=self._random_point(),
                speed=self._random_speed(),
                gs_position=self.gs_position,
            ))
            drone_id += 1

        return drones

    def _random_point(self) -> np.ndarray:
        """Return a uniformly random (x, y) point inside the arena."""
        return self.rng.uniform(
            [0.0, 0.0], [config.WIDTH, config.HEIGHT]
        ).astype(np.float64)

    def _random_speed(self) -> float:
        """Return a uniformly random speed in [DRONE_SPEED_MIN, DRONE_SPEED_MAX]."""
        return float(self.rng.uniform(config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX))

    def _recompute_links(self) -> None:
        """Refresh every drone's candidate pool, then its active link set.

        Two passes are kept so every candidate pool exists before any drone
        reads its neighbours.
        """
        for drone in self.drones:
            drone.update_candidates(self.drones)
        for drone in self.drones:
            drone.update_neighbors()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: Optional[Dict[int, object]] = None,
    ) -> Tuple[Dict[int, dict], Dict[int, float], Dict[int, bool], Dict[int, dict]]:
        """Advance the simulation by one timestep.

        In phase 1A *actions* is ignored; the selected routing rule runs
        internally.

        Args:
            actions: Optional dict of drone_id → action (ignored in phase 1).

        Returns:
            observations: Dict[drone_id, state_dict]
            rewards:      Dict[drone_id, float]. Always 0.0 in phase 1A — no
                          policy is being trained.
            dones:        Dict[drone_id, bool]
            infos:        Dict[drone_id, dict]   (empty for now)
        """
        self.tx_events = []
        self._rx_counts = defaultdict(int)

        # 1. Move M-drones along their straight start -> end line.
        for drone in self.drones:
            if drone.drone_type == "M":
                drone.step_move(config.TIMESTEP)

        # 2. Recompute candidate pools and re-select active links.
        self._recompute_links()

        # 3. Update active links for visualiser
        self._update_active_links()

        # 4. Generate new packets from M-drones
        self._generate_packets()

        # 5. Route packets
        self._route_packets()

        # 6. Expire stale packets still in queues
        self._expire_queued_packets()

        # 7. Radio idle/listen energy and accumulated rx energy for the step.
        for drone in self.drones:
            drone.consume_idle_energy()
            rx_n = self._rx_counts.get(drone.drone_id, 0)
            if rx_n:
                drone.consume_rx_energy(rx_n)

        self.step_count += 1

        # 8. Log per-step network state and per-drone state.
        self._log_step_and_drone_state()

        # The episode ends when the step budget is exhausted OR every M-drone
        # has reached its destination (there is no mission traffic left to
        # generate or deliver once they have all arrived).
        m_drones = [d for d in self.drones if d.drone_type == "M"]
        all_m_arrived = bool(m_drones) and all(d.arrived for d in m_drones)
        done = self.step_count >= config.MAX_STEPS or all_m_arrived
        if done:
            self.close_logger()

        observations = {d.drone_id: d.get_state() for d in self.drones}
        # Phase 1A has no learned policy, so every reward is 0.0. The key is
        # kept so the PettingZoo-style step() signature stays intact.
        rewards = {d.drone_id: 0.0 for d in self.drones}
        dones = {d.drone_id: done for d in self.drones}
        infos: Dict[int, dict] = {d.drone_id: {} for d in self.drones}

        return observations, rewards, dones, infos

    def close_logger(self) -> None:
        """Flush and close the event log. Safe to call multiple times."""
        if self._logger is not None:
            self._logger.close()
            self._logger = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_active_links(self) -> None:
        """Recompute the set of active wireless links for this step."""
        self.active_links = set()
        for drone in self.drones:
            for nid in drone.neighbors:
                link = tuple(sorted((drone.drone_id, nid)))
                self.active_links.add(link)

    def _generate_packets(self) -> None:
        """Have each M-drone generate PACKET_RATE new packets."""
        t = self._sim_time()
        for drone in self.drones:
            if drone.drone_type != "M":
                continue
            for _ in range(config.PACKET_RATE):
                pkt = self._factory.create(
                    source_id=drone.drone_id,
                    created_at=self.step_count,
                )
                drone.enqueue(pkt)
                self.all_packets.append(pkt)
                if self._logger is not None:
                    self._logger.log_packet_event(
                        event="generated",
                        time=t,
                        packet_id=pkt.packet_id,
                        src_drone=pkt.source_id,
                        current_drone=drone.drone_id,
                        hop_index=0,
                        is_control=pkt.is_control,
                    )

    def _route_packets(self) -> None:
        """Forward every queued packet one hop using the selected routing baseline."""
        t = self._sim_time()

        # Collect packets from all drones before forwarding (snapshot approach)
        # to avoid a drone receiving and re-forwarding in the same step.
        pending: List[Tuple[Drone, Packet]] = []
        for drone in self.drones:
            for pkt in drone.dequeue_all():
                pending.append((drone, pkt))

        for drone, pkt in pending:
            if not pkt.is_alive(self.step_count):
                self._expire_packet(pkt, drone)
                continue

            # Check if the drone itself can reach the GS directly
            if are_connected(drone.position, self.gs_position):
                pkt.relay_to("GS")
                pkt.mark_delivered(self.step_count)
                self.delivered.append(pkt)
                drone.consume_tx_energy()
                self.tx_events.append((drone.drone_id, "GS"))
                if self._logger is not None:
                    self._logger.log_packet_event(
                        event="delivered",
                        time=t,
                        packet_id=pkt.packet_id,
                        src_drone=pkt.source_id,
                        current_drone=drone.drone_id,
                        next_hop="GS",
                        hop_index=pkt.hop_count,
                        is_control=pkt.is_control,
                    )
                continue

            # Select next hop
            next_hop = self._select_next_hop(drone, pkt)
            if next_hop is None:
                pkt.mark_dropped(DropReason.NO_NEXT_HOP)
                self.dropped.append(pkt)
                if self._logger is not None:
                    self._logger.log_packet_event(
                        event="dropped",
                        time=t,
                        packet_id=pkt.packet_id,
                        src_drone=pkt.source_id,
                        current_drone=drone.drone_id,
                        hop_index=pkt.hop_count,
                        drop_reason="no_route",
                        is_control=pkt.is_control,
                    )
                continue

            # Forward
            pkt.relay_to(next_hop.drone_id)
            next_hop.enqueue(pkt)
            drone.consume_tx_energy()
            self.tx_events.append((drone.drone_id, next_hop.drone_id))
            self._rx_counts[next_hop.drone_id] += 1
            if self._logger is not None:
                self._logger.log_packet_event(
                    event="forwarded",
                    time=t,
                    packet_id=pkt.packet_id,
                    src_drone=pkt.source_id,
                    current_drone=drone.drone_id,
                    next_hop=next_hop.drone_id,
                    hop_index=pkt.hop_count,
                    is_control=pkt.is_control,
                )

    def _select_next_hop(self, drone: Drone, pkt: Packet) -> Optional[Drone]:
        """Select the next-hop drone for *pkt* according to the routing policy.

        Args:
            drone: Current holder.
            pkt:   The packet to forward.

        Returns:
            A Drone object or None if no valid hop exists.
        """
        return greedy_next_hop(drone, drone.neighbors, self.gs_position)

    def _expire_packet(self, pkt: Packet, holder: Optional[Drone] = None) -> None:
        """Drop *pkt* with the appropriate reason based on its state.

        Args:
            pkt:    The packet to expire.
            holder: The drone currently holding the packet (used for the log
                    record). May be None if the holder is unknown.
        """
        if pkt.hop_count > config.MAX_HOPS:
            pkt.mark_dropped(DropReason.MAX_HOPS)
            reason = "max_hops"
        else:
            pkt.mark_dropped(DropReason.TTL_EXPIRED)
            reason = "ttl_expired"
        self.dropped.append(pkt)
        if self._logger is not None:
            self._logger.log_packet_event(
                event="dropped",
                time=self._sim_time(),
                packet_id=pkt.packet_id,
                src_drone=pkt.source_id,
                current_drone=holder.drone_id if holder is not None else pkt.current_holder,
                hop_index=pkt.hop_count,
                drop_reason=reason,
                is_control=pkt.is_control,
            )

    def _expire_queued_packets(self) -> None:
        """Scan every queue and drop packets that have expired this step."""
        for drone in self.drones:
            still_alive: List[Packet] = []
            for pkt in drone.queue:
                if pkt.is_alive(self.step_count):
                    still_alive.append(pkt)
                else:
                    self._expire_packet(pkt, holder=drone)
            drone.queue = still_alive

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _sim_time(self) -> float:
        """Return the current simulation time in seconds."""
        return self.step_count * config.TIMESTEP

    def _log_step_and_drone_state(self) -> None:
        """Emit the per-step network state plus one per-drone state record."""
        if self._logger is None:
            return
        t = self._sim_time()
        frac, num_components = connectivity_sample(self.drones, self.gs_position)
        self._logger.log_step_state(
            time=t,
            frac_connected_to_gs=frac,
            num_components=num_components,
        )
        if config.LOG_DRONE_STATE_EVERY_STEP or self.step_count >= config.MAX_STEPS:
            for d in self.drones:
                self._logger.log_drone_state(
                    time=t,
                    drone_id=d.drone_id,
                    drone_type=d.drone_type,
                    position=d.position,
                    energy_radio=d.energy_radio,
                    energy_motion=d.energy_motion,
                )

    # ------------------------------------------------------------------
    # Accessors used by metrics / visualiser
    # ------------------------------------------------------------------

    @property
    def num_drones(self) -> int:
        """Total number of drones in this episode."""
        return len(self.drones)

    def get_drone_by_id(self, drone_id: int) -> Drone:
        """Return the Drone with the given ID.

        Args:
            drone_id: The drone's integer ID.

        Returns:
            The matching Drone object.

        Raises:
            ValueError: If the ID is not found.
        """
        for d in self.drones:
            if d.drone_id == drone_id:
                return d
        raise ValueError(f"No drone with id {drone_id}")
