"""
fanet_env.py — Main FANET simulation environment.

Provides a PettingZoo-style interface (reset / step) so that a MARL
framework can be plugged in during a later phase without rewriting the
core.  In phase 1A the *actions* argument to step() is ignored and the
selected non-learned routing rule runs internally.

Routing rules implemented here, both using local information only:
    - "greedy": among the neighbours strictly closer to the GS, take the one
      with the lowest combined link loss (channel loss, plus certain loss if
      this drone's own queue for that link is already full).
    - "random": uniform over all current neighbours.
The GS counts as a neighbour whenever it is in range.

Both rules choose the next hop when a packet is CREATED at a drone or ARRIVES
at one, and the packet then waits in that link's queue. The choice is not
revisited at send time.

Each step, in order:
    1. Move drones (skipped entirely while config.STATIC_MODE is True) and
       recompute every drone's in-range neighbour set.
    2. Retire packets whose TTL or hop limit ran out while waiting in a link
       queue, and let each source give up on its own packets that the GS never
       acknowledged.
    3. Create new packets at the M-drones and route each into a link queue.
    4. Transmit: snapshot the head of every link queue, then run each one's
       channel draw. A lost transmission is dropped "channel"; one aimed at the
       GS is delivered and acknowledged end-to-end; one aimed at a drone
       arrives there.
    5. Route the packets that arrived this step into the receivers' link
       queues. Arrivals are processed in a RANDOM order drawn from the
       channel/traffic stream — ordering by drone id would hand low-id drones
       the last free slot in a nearly full queue every time.
    6. Charge radio energy.

Expiry runs before anything else so a dead packet can never hold a queue slot
or a send slot. A packet that arrives in step 5 is routed immediately but
cannot be sent before the next step; a packet created in step 3 can be sent in
the same step.

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

import networkx as nx
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.channel import are_connected, euclidean_distance
from fanet_sim.envs.drone import Drone, NextHop, _link_row
from fanet_sim.envs.packet import DropReason, Packet, PacketFactory
from fanet_sim.utils.event_log import EventLogger
from fanet_sim.utils.metrics import build_graph_with_gs
from fanet_sim.utils.metrics import connectivity_sample


# ---------------------------------------------------------------------------
# Greedy geographic routing
# ---------------------------------------------------------------------------

def greedy_next_hop(
    holder_position: np.ndarray,
    candidates: Dict[NextHop, np.ndarray],
    gs_position: np.ndarray,
    link_full: Optional[Dict[NextHop, bool]] = None,
) -> Optional[NextHop]:
    """Pick the next hop least likely to lose the packet, among those that progress.

    Two stages:

    1. Keep only the candidates STRICTLY closer to the ground station than the
       current holder. This is the progress condition; it is what stops a
       packet circling among peers.
    2. Among those, minimise the combined link loss

           link_loss    = 1 - (1 - channel_loss) * (1 - queue_full)
           channel_loss = channel.p_loss(link distance)
           queue_full   = 1 if the holder's OWN queue for that link is at
                          capacity right now, else 0

       Ties break on the lower channel_loss, then on the candidate closer to
       the GS.

    Both inputs are things the drone itself knows: the channel figure comes
    from the link distance — the signal margin a radio could measure — and the
    queue figure is exact, because the queue belongs to this drone. Neither
    reads an ACK counter.

    A full queue makes that link's loss 1, so greedy routes around its own
    congestion whenever some other qualifying link still has room. If every
    qualifying link is full they all score 1, the tie-break picks the cleanest
    channel, and the packet is dropped "queue_full" on enqueue.

    The ground station counts as a candidate whenever it is in range, keyed by
    the string ``"GS"``. It always satisfies the progress condition (it is at
    distance 0 from itself), but it does not automatically win: a nearer
    neighbour with a cleaner link can beat a marginal direct shot at the GS.

    Returns None — a routing void, which the caller drops as "no_route" — when
    there are no candidates at all, or when none is strictly closer to the GS.

    Args:
        holder_position: Position of the drone currently holding the packet.
        candidates:      Reachable next hops, mapping key → position. Keys are
                         drone ids, plus ``"GS"`` when the GS is in range.
        gs_position:     Ground-station position as a NumPy array.
        link_full:       Per candidate, whether the holder's queue for that
                         link is at capacity. Missing keys count as not full.

    Returns:
        The winning candidate's key, or None.
    """
    holder_dist = euclidean_distance(holder_position, gs_position)
    if link_full is None:
        link_full = {}

    # Stage 1: only candidates that actually make progress.
    closer = {
        key: pos
        for key, pos in candidates.items()
        if euclidean_distance(pos, gs_position) < holder_dist
    }
    if not closer:
        return None  # routing void: no progress available

    # Stage 2: lowest combined loss; ties on channel, then on distance to GS.
    def rank(key: NextHop) -> Tuple[float, float, float]:
        channel_loss = channel.p_loss(
            euclidean_distance(holder_position, closer[key])
        )
        queue_loss = 1.0 if link_full.get(key, False) else 0.0
        link_loss = 1.0 - (1.0 - channel_loss) * (1.0 - queue_loss)
        return (
            link_loss,
            channel_loss,
            euclidean_distance(closer[key], gs_position),
        )

    return min(closer, key=rank)


def random_next_hop(
    candidates: Dict[NextHop, np.ndarray],
    rng: np.random.Generator,
) -> Optional[NextHop]:
    """Pick uniformly at random among every current neighbour.

    The ground station is one of the options whenever it is in range. There is
    no progress condition and no regard for queue state: this rule is the naive
    baseline greedy is measured against. Like greedy, it chooses at enqueue
    time.

    Args:
        candidates: Reachable next hops, mapping key → position.
        rng:        The routing rule's own random stream, so its draws never
                    shift the channel or traffic randomness.

    Returns:
        The chosen candidate's key, or None if there are no candidates at all
        (which the caller drops as "no_route").
    """
    if not candidates:
        return None

    # Fix the order before drawing, so the choice is reproducible regardless of
    # dict insertion order. Drone ids ascending, then the GS.
    keys: List[NextHop] = sorted(k for k in candidates if k != "GS")
    if "GS" in candidates:
        keys.append("GS")

    return keys[int(rng.integers(len(keys)))]


#: The routing rules this environment can run.
ROUTING_RULES = ("greedy", "random")


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
        placement_draws:  Layouts drawn before one passed the connectivity
                          filter (1 = the first one did).
        link_stats:       Ground-truth per-link counters, keyed (from, to).
        link_stats_measured: The same, counting only post-warm-up steps.
        queue_samples:    Per-step link-queue occupancy, sampled after warm-up.
        expired_on_arrival: Packets that arrived already out of TTL or hops.
        routing:          'greedy' or 'random'.
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
            routing:     Routing rule to use: 'greedy' or 'random'.
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
        if routing not in ROUTING_RULES:
            raise ValueError(
                f"unknown routing rule {routing!r}; expected one of {ROUTING_RULES}"
            )
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
        # Ground-truth per-link bookkeeping, kept independently of the ACK
        # counters on the drones so the two views can be compared.
        self.link_stats: Dict[Tuple[int, NextHop], Dict[str, int]] = defaultdict(
            _link_row
        )
        self.link_stats_measured: Dict[Tuple[int, NextHop], Dict[str, int]] = (
            defaultdict(_link_row)
        )
        # Queue occupancy, one sample per link queue per post-warm-up step.
        self.queue_samples: List[int] = []
        # Packets that reached a drone already out of TTL or hops. They never
        # waited on a link, so no link is charged for them.
        self.expired_on_arrival: int = 0
        self.expired_on_arrival_measured: int = 0
        # Per-M-drone traffic phase, drawn in reset().
        self._traffic_offset: Dict[int, int] = {}
        # How many layouts had to be drawn before one was accepted by the
        # connectivity filter. 1 means the first draw passed.
        self.placement_draws: int = 0

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
        self.link_stats = defaultdict(_link_row)
        self.link_stats_measured = defaultdict(_link_row)
        self.queue_samples = []
        self.expired_on_arrival = 0
        self.expired_on_arrival_measured = 0
        self._rx_counts = defaultdict(int)

        # Open a new logger for this episode (close any prior one).
        if self._logger is not None:
            self._logger.close()
        self._logger = EventLogger(self.log_path, episode_id=self.episode_id)

        self.drones = self._create_drones()
        self._assign_traffic_offsets()

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
                "require_connected_m": config.REQUIRE_CONNECTED_M,
                "placement_draws": self.placement_draws,
                "area_width": config.WIDTH,
                "area_height": config.HEIGHT,
            },
            traffic_load={
                "packet_interval_steps": config.PACKET_INTERVAL_STEPS,
                "packet_interval_ms": config.PACKET_INTERVAL_STEPS * config.TIMESTEP * 1000,
                "random_traffic_offsets": config.RANDOM_TRAFFIC_OFFSETS,
                "packet_size_bytes": config.PACKET_SIZE,
                "ttl": config.PACKET_TTL,
                "max_hops": config.MAX_HOPS,
                "queue_capacity": config.QUEUE_CAPACITY,
                "max_tx_per_step": config.MAX_TX_PER_STEP,
                "warmup_steps": config.WARMUP_STEPS,
                "drain_steps": config.DRAIN_STEPS,
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
        """Draw layouts until one is acceptable, and return its drones.

        With ``config.REQUIRE_CONNECTED_M`` set, a layout is accepted only if
        every M-drone has a path to the GS. A rejected layout is discarded and
        another is drawn from the SAME placement stream, so a given
        placement_seed still always produces the same accepted layout. The
        number of draws is recorded in :attr:`placement_draws`.

        Returns:
            List of Drone objects (M-drones first, then C-drones).

        Raises:
            RuntimeError: If no acceptable layout turns up within
                ``config.MAX_PLACEMENT_DRAWS`` draws.
        """
        for draw in range(1, config.MAX_PLACEMENT_DRAWS + 1):
            drones = self._draw_drones()
            if not config.REQUIRE_CONNECTED_M or self._all_m_reach_gs(drones):
                self.placement_draws = draw
                return drones

        raise RuntimeError(
            f"no layout with every M-drone connected to the GS after "
            f"{config.MAX_PLACEMENT_DRAWS} draws (placement_seed="
            f"{self.placement_seed})"
        )

    def _all_m_reach_gs(self, drones: List[Drone]) -> bool:
        """True if every M-drone has some path to the GS.

        Uses the simulator's own link rule — an edge wherever
        :func:`are_connected` passes — and allows paths through any drone,
        C-drones included. C-drones themselves need no path.

        Args:
            drones: A candidate fleet.

        Returns:
            Whether the layout is acceptable.
        """
        # The link sets are what build_graph_with_gs reads, so populate them on
        # the candidate fleet before asking.
        for drone in drones:
            drone.update_candidates(drones)
        for drone in drones:
            drone.update_neighbors()

        graph, gs_label = build_graph_with_gs(drones, self.gs_position)
        reachable = nx.node_connected_component(graph, gs_label)
        return all(
            d.drone_id in reachable for d in drones if d.drone_type == "M"
        )

    def _draw_drones(self) -> List[Drone]:
        """Draw one candidate fleet from the placement stream.

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

    def _assign_traffic_offsets(self) -> None:
        """Give every M-drone the step within its interval on which it creates.

        Without this, every M-drone would create its packet on the same steps
        and the network would see one synchronised burst per interval. Offsets
        are drawn from the traffic/channel stream, so they follow run_seed.
        """
        interval = max(1, config.PACKET_INTERVAL_STEPS)
        self._traffic_offset = {}
        for drone in self.drones:
            if drone.drone_type != "M":
                continue
            if config.RANDOM_TRAFFIC_OFFSETS and interval > 1:
                self._traffic_offset[drone.drone_id] = int(
                    self._rng_chan.integers(interval)
                )
            else:
                self._traffic_offset[drone.drone_id] = 0

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
        measured = config.is_warm(self.step_count)

        # 1. Move M-drones along their straight start -> end line, then refresh
        #    the neighbour sets. Phase 1A is static, so no drone moves.
        if not config.STATIC_MODE:
            for drone in self.drones:
                if drone.drone_type == "M":
                    drone.step_move(config.TIMESTEP)
        self._recompute_links()
        self._update_active_links()

        # 2. Retire packets that died waiting in a link queue, and let each
        #    source give up on its own packets the GS never acknowledged. This
        #    runs first so a dead packet holds neither a queue slot nor a send
        #    slot.
        self._expire_queued_packets()
        self._expire_unacked_at_sources()

        # 3. Create this step's packets and route each into a link queue.
        self._generate_packets()

        # 4. Transmit the head of every link queue, and collect what arrives.
        arrivals = self._transmit(measured)

        # 5. Route the arrivals into the receivers' own link queues, in a
        #    random order so no drone is systematically last to a free slot.
        self._route_arrivals(arrivals, measured)

        # 6. Sample per-link queue occupancy once the warm-up is over.
        if measured:
            for drone in self.drones:
                self.queue_samples.extend(
                    len(queue) for queue in drone.link_queues.values()
                )

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
        """Create this step's packets at the M-drones and route each one.

        An M-drone creates one packet every ``config.PACKET_INTERVAL_STEPS``
        steps, on the step matching its own traffic offset. The packet is
        routed immediately into one of that drone's link queues, exactly like
        a packet arriving from elsewhere.
        """
        t = self._sim_time()
        interval = max(1, config.PACKET_INTERVAL_STEPS)
        measured = config.is_warm(self.step_count)

        for drone in self.drones:
            if drone.drone_type != "M":
                continue
            offset = self._traffic_offset.get(drone.drone_id, 0)
            if self.step_count % interval != offset:
                continue

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
            self._route_into_queue(drone, pkt, measured)

    # ------------------------------------------------------------------
    # Routing (at creation and on arrival)
    # ------------------------------------------------------------------

    def _next_hop_candidates(self, drone: Drone) -> Dict[NextHop, np.ndarray]:
        """Return every next hop *drone* can currently reach, key → position.

        Keys are neighbour drone ids, plus the string ``"GS"`` when the ground
        station is in range. The GS is a normal member of this set: both
        routing rules treat it as just another neighbour.

        Args:
            drone: The drone about to route a packet.

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
        """Choose the link a packet should join, according to the routing rule.

        Called when a packet is created at *drone* or arrives there — never at
        send time.

        Args:
            drone:      The drone routing the packet.
            candidates: Reachable next hops from :meth:`_next_hop_candidates`.

        Returns:
            The chosen next-hop key, or None if the rule finds no usable hop
            (the caller then drops the packet with reason "no_route").
        """
        if self.routing == "random":
            return random_next_hop(candidates, self._rng_route)

        link_full = {key: drone.queue_is_full(key) for key in candidates}
        return greedy_next_hop(
            drone.position, candidates, self.gs_position, link_full
        )

    def _route_into_queue(
        self, drone: Drone, pkt: Packet, measured: bool
    ) -> None:
        """Commit *pkt* to one of *drone*'s link queues, or drop it.

        Three outcomes: no qualifying next hop drops the packet "no_route";
        a chosen link whose queue is full drops it "queue_full" at this drone;
        otherwise it waits on that link until its turn to be sent.

        A packet that has already used up its TTL or its hop budget is retired
        here rather than taking a queue slot it can never leave. It never
        waited on a link, so no link is charged for it; it is counted in
        :attr:`expired_on_arrival` instead, which keeps the identity

            ttl + hop_limit = expired_in_queue + expired_on_arrival

        true across the run.

        Args:
            drone:    The drone holding the packet right now.
            pkt:      The packet to route.
            measured: True once the warm-up is over.
        """
        if not pkt.is_alive(self.step_count):
            self.expired_on_arrival += 1
            if measured:
                self.expired_on_arrival_measured += 1
            self._expire_packet(pkt, holder=drone)
            return

        candidates = self._next_hop_candidates(drone)
        next_key = self._select_next_hop(drone, candidates)

        if next_key is None:
            # Routing void: no candidate at all, or none closer to the GS.
            self._drop(pkt, DropReason.NO_ROUTE, drone)
            return

        drone.note_offered(next_key, measured)
        self._note_link(drone.drone_id, next_key, "offered", measured)

        if not drone.enqueue(next_key, pkt):
            drone.note_queue_full(next_key, measured)
            self._note_link(
                drone.drone_id, next_key, "dropped_queue_full", measured
            )
            self._drop(pkt, DropReason.QUEUE_FULL, drone, next_hop=next_key)

    # ------------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------------

    def _transmit(self, measured: bool) -> List[Tuple[Drone, Packet]]:
        """Send the head of every link queue and return what arrived where.

        The whole snapshot is taken before any transmission, so a packet can
        never be received and re-sent in the same step. Each link queue sends
        at most ``config.MAX_TX_PER_STEP``; all of a drone's link queues send
        in the same step, since there is no collision model.

        Every send draws against ``channel.p_loss`` for that link's distance,
        including the last hop into the GS. A lost send is dropped "channel".
        A send that gets through is never refused: receiving is unlimited.

        Args:
            measured: True once the warm-up is over.

        Returns:
            A list of ``(receiving_drone, packet)`` pairs to be routed next.
        """
        t = self._sim_time()

        # Snapshot: (sender, next-hop key, packet) for every link queue head.
        pending: List[Tuple[Drone, NextHop, Packet]] = []
        for drone in self.drones:
            for next_key in list(drone.link_queues):
                for pkt in drone.dequeue_head(next_key):
                    pending.append((drone, next_key, pkt))

        arrivals: List[Tuple[Drone, Packet]] = []

        for drone, next_key, pkt in pending:
            target_pos = (
                self.gs_position
                if next_key == "GS"
                else drone.neighbors[next_key].position
            )

            drone.note_sent(next_key, measured)
            self._note_link(drone.drone_id, next_key, "sent", measured)
            drone.consume_tx_energy()

            dist = euclidean_distance(drone.position, target_pos)
            if self._rng_chan.random() < channel.p_loss(dist):
                drone.note_hop_ack(next_key, "channel", measured)
                self._note_link(drone.drone_id, next_key, "lost_channel", measured)
                self._drop(pkt, DropReason.CHANNEL, drone, next_hop=next_key)
                continue

            drone.note_hop_ack(next_key, "ok", measured)
            self._note_link(drone.drone_id, next_key, "acked", measured)
            self.tx_events.append((drone.drone_id, next_key))

            if next_key == "GS":
                pkt.relay_to("GS")
                pkt.mark_delivered(self.step_count)
                self.delivered.append(pkt)
                # End-to-end ACK: tell the source its packet arrived.
                self.get_drone_by_id(pkt.source_id).note_gs_ack(pkt.packet_id)
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
            pkt.relay_to(next_key)
            self._rx_counts[next_hop.drone_id] += 1
            arrivals.append((next_hop, pkt))
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

        return arrivals

    def _route_arrivals(
        self, arrivals: List[Tuple[Drone, Packet]], measured: bool
    ) -> None:
        """Route everything that arrived this step, in a random order.

        Order matters: each routing decision reads the receiver's queues as
        they stand, so whoever is handled first gets the last free slot in a
        nearly full queue. Going by drone id would hand that advantage to the
        same drones every step, so the order is drawn from the run's
        channel/traffic stream instead — reproducible for a given run_seed, and
        identical for greedy and random.

        A packet routed here cannot be sent until the next step: this step's
        transmission snapshot was taken before it arrived.

        Args:
            arrivals: ``(receiving_drone, packet)`` pairs from :meth:`_transmit`.
            measured: True once the warm-up is over.
        """
        order = self._rng_chan.permutation(len(arrivals))
        for index in order:
            receiver, pkt = arrivals[int(index)]
            self._route_into_queue(receiver, pkt, measured)


    def _note_link(
        self, from_id: int, to_key: NextHop, field: str, measured: bool
    ) -> None:
        """Record one ground-truth per-link event.

        This is the simulator's own book, kept separately from the ACK-derived
        counters on the drones so the two views can be compared.

        Args:
            from_id:  Sending drone id.
            to_key:   Receiving drone id, or "GS".
            field:    One of drone.LINK_FIELDS.
            measured: True once the warm-up is over.
        """
        self.link_stats[(from_id, to_key)][field] += 1
        if measured:
            self.link_stats_measured[(from_id, to_key)][field] += 1

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

    def _expire_packet(
        self,
        pkt: Packet,
        holder: Optional[Drone] = None,
        next_hop: Optional[NextHop] = None,
    ) -> None:
        """Drop a packet that ran out of TTL or of hops.

        Args:
            pkt:      The packet to retire.
            holder:   The drone currently holding it.
            next_hop: The link it was waiting on, if it was in a queue.
        """
        if pkt.hop_count >= config.MAX_HOPS:
            self._drop(pkt, DropReason.HOP_LIMIT, holder, next_hop=next_hop)
        else:
            self._drop(pkt, DropReason.TTL, holder, next_hop=next_hop)

    def _expire_queued_packets(self) -> None:
        """Retire packets that died waiting in a link queue.

        A packet that has run out of TTL or of hops is removed from whichever
        link queue it was waiting on, and that link is charged an
        ``expired_in_queue``: it was offered the packet and never got it
        across.
        """
        measured = config.is_warm(self.step_count)
        for drone in self.drones:
            for next_key, queue in drone.link_queues.items():
                still_alive: List[Packet] = []
                for pkt in queue:
                    if pkt.is_alive(self.step_count):
                        still_alive.append(pkt)
                        continue
                    drone.note_expired_in_queue(next_key, measured)
                    self._note_link(
                        drone.drone_id, next_key, "expired_in_queue", measured
                    )
                    self._expire_packet(pkt, holder=drone, next_hop=next_key)
                drone.link_queues[next_key] = still_alive

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
