"""
fanet_env.py — Main FANET simulation environment.

Provides a PettingZoo-style interface (reset / step) so that a MARL
framework can be plugged in during a later phase without rewriting the
core.  In phase 1A the *actions* argument to step() is ignored and the
selected non-learned routing rule runs internally.

Routing rules implemented here:
    - Greedy geographic routing  (default)

Each step, in order:
    1. Move drones (skipped entirely while config.STATIC_MODE is True).
    2. Recompute every drone's in-range neighbour set.
    3. Retire packets whose TTL or hop limit ran out, and let each source
       give up on its own packets that the GS never acknowledged.
    4. Create new packets at the M-drones.
    5. Forward packets: each drone sends at most config.MAX_TX_PER_STEP,
       oldest first, choosing the next hop at send time.
    6. Charge radio energy.

Expiry runs BEFORE forwarding so a dead packet at the head of a queue can
never consume that drone's send budget.

Randomness comes from two independent seeds so one layout can be replayed
under different randomness:
    placement_seed — drone positions, speeds, end points.
    run_seed       — channel-loss draws and traffic timing. The RANDOM routing
                     rule draws from a third stream spawned off run_seed, so
                     its choices never shift the channel or traffic draws.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.channel import are_connected, euclidean_distance
from fanet_sim.envs.drone import Drone, NextHop
from fanet_sim.envs.packet import DropReason, Packet, PacketFactory
from fanet_sim.utils.event_log import EventLogger
from fanet_sim.utils.metrics import connectivity_sample


# ---------------------------------------------------------------------------
# Greedy geographic routing
# ---------------------------------------------------------------------------

def greedy_next_hop(
    holder_position: np.ndarray,
    candidates: Dict[NextHop, np.ndarray],
    gs_position: np.ndarray,
) -> Optional[NextHop]:
    """Pick the next hop closest to the ground station (greedy geographic).

    The ground station counts as a candidate whenever it is in range, keyed by
    the string ``"GS"``; being at distance 0 from itself it always wins.

    Returns None — a routing void, which the caller drops as "no_route" — when
    there are no candidates at all, or when none of them is STRICTLY closer to
    the GS than the current holder.

    Args:
        holder_position: Position of the drone currently holding the packet.
        candidates:      Reachable next hops, mapping key → position. Keys are
                         drone ids, plus ``"GS"`` when the GS is in range.
        gs_position:     Ground-station position as a NumPy array.

    Returns:
        The winning candidate's key, or None.
    """
    if not candidates:
        return None

    best_key = min(
        candidates,
        key=lambda key: euclidean_distance(candidates[key], gs_position),
    )
    best_dist = euclidean_distance(candidates[best_key], gs_position)
    current_dist = euclidean_distance(holder_position, gs_position)

    if best_dist < current_dist:
        return best_key
    return None  # routing void: no progress available


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
        placement_seed:   Seed for drone placement (positions, speeds).
        run_seed:         Seed for channel-loss and traffic randomness.
    """

    def __init__(
        self,
        routing: str = "greedy",
        log_path: Optional[str] = None,
        episode_id: int = 0,
        placement_seed: Optional[int] = None,
        run_seed: Optional[int] = None,
    ) -> None:
        """Create the environment (does NOT run reset automatically).

        Args:
            routing:     Routing rule to use: 'greedy'.
            log_path:    Path to write the Stage-1 JSONL event log. If None,
                         a default of ``{config.LOG_DIR}/episode_{id}.jsonl``
                         is used.
            episode_id:  Integer episode identifier, embedded in every log
                         record so multiple episodes can be concatenated.
            placement_seed: Seed for drone placement — positions, speeds and
                         end points. Defaults to config.PLACEMENT_SEED. Hold it
                         fixed to re-run the same layout.
            run_seed:    Seed for channel-loss draws and traffic timing.
                         Defaults to config.RUN_SEED. Vary it to re-run one
                         layout under different randomness. The RANDOM routing
                         rule draws from its own stream spawned off this seed,
                         so switching routing rules does not shift the channel
                         or traffic draws.

        Both seeds are recorded in the episode-meta log record.
        """
        self.routing = routing
        self.episode_id = episode_id
        self.placement_seed = (
            config.PLACEMENT_SEED if placement_seed is None else placement_seed
        )
        self.run_seed = config.RUN_SEED if run_seed is None else run_seed
        # Built in reset(); see _seed_streams().
        self._rng_place: np.random.Generator
        self._rng_chan: np.random.Generator
        self._rng_route: np.random.Generator
        self._seed_streams()

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
        self._seed_streams()

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
            seed={"placement": self.placement_seed, "run": self.run_seed},
            num_drones=len(self.drones),
            num_M=config.NUM_M_DRONES,
            num_C=config.NUM_C_DRONES,
            episode_length=config.MAX_STEPS,
            mobility_params={
                "speed_min": config.DRONE_SPEED_MIN,
                "speed_max": config.DRONE_SPEED_MAX,
                "m_drone_mobility": config.M_DRONE_MOBILITY,
                "static_mode": config.STATIC_MODE,
                "area_width": config.WIDTH,
                "area_height": config.HEIGHT,
            },
            traffic_load={
                "packet_rate_per_M_per_step": config.PACKET_RATE,
                "packet_size_bytes": config.PACKET_SIZE,
                "ttl": config.PACKET_TTL,
                "max_hops": config.MAX_HOPS,
                "queue_capacity": config.QUEUE_CAPACITY,
                "max_tx_per_step": config.MAX_TX_PER_STEP,
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
                "channel_loss_k": config.CHANNEL_LOSS_K,
                "timestep_s": config.TIMESTEP,
                "routing": self.routing,
            },
            anchor={
                "paper": "IQMR",
                "citation": "Sharvari et al., 2024",
                "arxiv": "2408.09109",
                "section": "V.A",
                "notes": (
                    "From IQMR: speed range (10-30 m/s), transmit power "
                    "(1 W = 30 dBm), 2.4 GHz carrier, energy budget "
                    "(11.1 V x 5200 mAh = 207792 J) and radio range (250 m). "
                    "Receiver sensitivity (-54 dBm) is derived from the FSPL "
                    "model to reproduce that 250 m range at 1 W rather than "
                    "picked arbitrarily. "
                    "NOT from IQMR: the 900x900 m arena, the 18 M + 7 C fleet "
                    "split, and 2D instead of 3D — those are Phase-1A choices "
                    "for this project."
                ),
            },
        )

        # Initial step-state and drone-state snapshots at t=0.
        self._log_step_and_drone_state()

        return {d.drone_id: d.get_state() for d in self.drones}

    def _create_drones(self) -> List[Drone]:
        """Create and return all drones.

        Every drone is placed uniformly at random inside the arena. While
        ``config.STATIC_MODE`` is True that position is also its end point and
        nothing ever moves. Otherwise (Phase 1B) each M-drone additionally gets
        a random end point and flies the straight line to it.

        Returns:
            List of Drone objects (M-drones first, then C-drones).
        """
        drones: List[Drone] = []
        drone_id = 0

        for _ in range(config.NUM_M_DRONES):
            start = self._random_point()
            end = start.copy() if config.STATIC_MODE else self._random_point()
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

    def _seed_streams(self) -> None:
        """(Re)build the three independent random streams from the two seeds.

        ``_rng_place`` is driven by ``placement_seed``. ``_rng_chan`` (channel
        loss and traffic timing) and ``_rng_route`` (the RANDOM routing rule)
        are two independent streams spawned off ``run_seed``, so consuming
        routing draws can never shift the channel or traffic draws.
        """
        self._rng_place = np.random.default_rng(self.placement_seed)
        chan_seed, route_seed = np.random.SeedSequence(self.run_seed).spawn(2)
        self._rng_chan = np.random.default_rng(chan_seed)
        self._rng_route = np.random.default_rng(route_seed)

    def _random_point(self) -> np.ndarray:
        """Return a uniformly random (x, y) point inside the arena."""
        return self._rng_place.uniform(
            [0.0, 0.0], [config.WIDTH, config.HEIGHT]
        ).astype(np.float64)

    def _random_speed(self) -> float:
        """Return a uniformly random speed in [DRONE_SPEED_MIN, DRONE_SPEED_MAX]."""
        return float(
            self._rng_place.uniform(config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX)
        )

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

        # 1. Move M-drones along their straight start -> end line. Phase 1A is
        #    static: nothing moves, so this is skipped entirely.
        if not config.STATIC_MODE:
            for drone in self.drones:
                if drone.drone_type == "M":
                    drone.step_move(config.TIMESTEP)

        # 2. Recompute candidate pools and re-select active links.
        self._recompute_links()

        # 3. Update active links for visualiser
        self._update_active_links()

        # 4. Retire packets whose TTL or hop limit ran out, and let each source
        #    give up on its own packets the GS never acknowledged. This runs
        #    BEFORE forwarding so a dead packet at the head of a queue cannot
        #    consume that drone's send budget for the step.
        self._expire_queued_packets()
        self._expire_unacked_at_sources()

        # 5. Generate new packets from M-drones
        self._generate_packets()

        # 6. Route packets: at most MAX_TX_PER_STEP per drone, oldest first.
        self._route_packets()

        # 7. Radio idle/listen energy and accumulated rx energy for the step.
        for drone in self.drones:
            drone.consume_idle_energy()
            rx_n = self._rx_counts.get(drone.drone_id, 0)
            if rx_n:
                drone.consume_rx_energy(rx_n)

        self.step_count += 1

        # 8. Log per-step network state and per-drone state.
        self._log_step_and_drone_state()

        # Fixed episode length. In static mode that is the ONLY end condition:
        # an M-drone that never moves is trivially "arrived", so the
        # all-arrived shortcut below would end the episode on step 1.
        done = self.step_count >= config.MAX_STEPS
        if not config.STATIC_MODE:
            # Phase 1B: also stop once every M-drone has reached its end point,
            # since no mission traffic is left to generate or deliver.
            m_drones = [d for d in self.drones if d.drone_type == "M"]
            done = done or (bool(m_drones) and all(d.arrived for d in m_drones))
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
        """Have each M-drone create its new packets for this step.

        A new packet goes into the same queue as the relayed traffic, so it is
        dropped with reason "queue_full" if that queue is already at capacity.
        The source counts every packet it creates and starts waiting for the
        GS ACK.
        """
        t = self._sim_time()
        for drone in self.drones:
            if drone.drone_type != "M":
                continue
            for _ in range(config.PACKET_RATE):
                pkt = self._factory.create(
                    source_id=drone.drone_id,
                    created_at=self.step_count,
                )
                self.all_packets.append(pkt)
                drone.note_created(pkt)
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
                if not drone.enqueue(pkt):
                    # Own queue full: the packet dies where it was born. There
                    # is no link involved, so no per-link counter moves.
                    drone.note_own_queue_full()
                    self._drop(pkt, DropReason.QUEUE_FULL, drone)

    def _next_hop_candidates(self, drone: Drone) -> Dict[NextHop, np.ndarray]:
        """Return every next hop *drone* can currently reach, key → position.

        Keys are neighbour drone ids, plus the string ``"GS"`` when the ground
        station is in range. The GS is a normal member of this set: both
        routing rules treat it as just another neighbour.

        Args:
            drone: The drone about to send.

        Returns:
            Dict mapping next-hop key to that endpoint's position.
        """
        candidates: Dict[NextHop, np.ndarray] = {
            nid: nbr.position for nid, nbr in drone.neighbors.items()
        }
        if are_connected(drone.position, self.gs_position):
            candidates["GS"] = self.gs_position
        return candidates

    def _select_next_hop(
        self,
        drone: Drone,
        candidates: Dict[NextHop, np.ndarray],
    ) -> Optional[NextHop]:
        """Choose the next hop for one packet according to the routing rule.

        Called once per packet at SEND time, not when the packet arrives.

        Args:
            drone:      Current holder.
            candidates: Reachable next hops from :meth:`_next_hop_candidates`.

        Returns:
            The chosen next-hop key, or None if the rule finds no usable hop
            (the caller then drops the packet with reason "no_route").
        """
        return greedy_next_hop(drone.position, candidates, self.gs_position)

    def _route_packets(self) -> None:
        """Forward this step's send batch from every drone.

        Each drone releases at most ``config.MAX_TX_PER_STEP`` packets, oldest
        first. All batches are taken as one snapshot before any forwarding, so
        a packet cannot be received and re-forwarded in the same step.

        Every transmission over an existing link can still fail: it is lost
        with probability ``channel.p_loss(distance)``, which also covers the
        last hop into the GS. A surviving packet is refused if the next hop's
        queue is full. Either way the sender learns which of the two happened
        from the (ideal) hop-by-hop ACK.
        """
        t = self._sim_time()

        # Snapshot every drone's send batch before forwarding anything.
        pending: List[Tuple[Drone, Packet]] = []
        for drone in self.drones:
            for pkt in drone.dequeue_for_send():
                pending.append((drone, pkt))

        for drone, pkt in pending:
            candidates = self._next_hop_candidates(drone)
            next_key = self._select_next_hop(drone, candidates)

            if next_key is None:
                # Routing void: no candidate at all, or none closer to the GS.
                self._drop(pkt, DropReason.NO_ROUTE, drone)
                continue

            # --- transmit ---
            drone.note_sent(next_key)
            drone.consume_tx_energy()
            dist = euclidean_distance(drone.position, candidates[next_key])

            if self._rng_chan.random() < channel.p_loss(dist):
                drone.note_hop_ack(next_key, "channel")
                self._drop(pkt, DropReason.CHANNEL, drone, next_hop=next_key)
                continue

            if next_key == "GS":
                # The GS has no queue and no per-step limit: it always accepts.
                pkt.relay_to("GS")
                pkt.mark_delivered(self.step_count)
                self.delivered.append(pkt)
                drone.note_hop_ack("GS", "ok")
                # End-to-end ACK: tell the source its packet arrived.
                self.get_drone_by_id(pkt.source_id).note_gs_ack(pkt.packet_id)
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

            next_hop = drone.neighbors[next_key]
            if not next_hop.enqueue(pkt):
                # Offered but refused — the hop is not taken, so hop_count
                # stays where it was.
                drone.note_hop_ack(next_key, "queue_full")
                self._drop(pkt, DropReason.QUEUE_FULL, drone, next_hop=next_key)
                continue

            pkt.relay_to(next_key)
            drone.note_hop_ack(next_key, "ok")
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

    # ------------------------------------------------------------------
    # Drops and expiry
    # ------------------------------------------------------------------

    def _drop(
        self,
        pkt: Packet,
        reason: DropReason,
        holder: Optional[Drone] = None,
        next_hop: Optional[NextHop] = None,
    ) -> None:
        """Retire *pkt* with exactly one drop *reason* and log it.

        Args:
            pkt:      The packet being dropped.
            reason:   The single reason it died.
            holder:   The drone holding it (for the log record).
            next_hop: For a send that failed, the hop it was aimed at.
        """
        pkt.mark_dropped(reason)
        self.dropped.append(pkt)
        if self._logger is not None:
            self._logger.log_packet_event(
                event="dropped",
                time=self._sim_time(),
                packet_id=pkt.packet_id,
                src_drone=pkt.source_id,
                current_drone=(
                    holder.drone_id if holder is not None else pkt.current_holder
                ),
                next_hop=next_hop,
                hop_index=pkt.hop_count,
                drop_reason=reason.label,
                is_control=pkt.is_control,
            )

    def _expire_packet(self, pkt: Packet, holder: Optional[Drone] = None) -> None:
        """Drop a packet that ran out of TTL or of hops.

        Args:
            pkt:    The packet to retire.
            holder: The drone currently holding it.
        """
        if pkt.hop_count >= config.MAX_HOPS:
            self._drop(pkt, DropReason.HOP_LIMIT, holder)
        else:
            self._drop(pkt, DropReason.TTL, holder)

    def _expire_queued_packets(self) -> None:
        """Scan every queue and retire packets that can no longer be sent."""
        for drone in self.drones:
            still_alive: List[Packet] = []
            for pkt in drone.queue:
                if pkt.is_alive(self.step_count):
                    still_alive.append(pkt)
                else:
                    self._expire_packet(pkt, holder=drone)
            drone.queue = still_alive

    def _expire_unacked_at_sources(self) -> None:
        """Let each M-drone give up on its own packets the GS never acked.

        This is the source-side timer from the ACK model: a drone knows when it
        created each packet and knows the TTL, so it concludes on its own that
        an unacknowledged packet is lost. It reads nothing global.
        """
        for drone in self.drones:
            if drone.drone_type == "M":
                drone.expire_unacked(self.step_count)

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
