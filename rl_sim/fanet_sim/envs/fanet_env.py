"""
fanet_env.py — Main FANET simulation environment.

Provides a PettingZoo-style interface (reset / step) so that a MARL
framework can be plugged in during a later phase without rewriting the
core.  In phase 1A the *actions* argument to step() is ignored and the
selected non-learned routing rule runs internally.

Routing rules implemented here, both using local information only:
    - "greedy": classic greedy geographic forwarding. Among the neighbours
      strictly closer to the GS, skip those whose queue for that link is
      full and take the one CLOSEST to the GS; ties break on the lower
      channel loss. If every closer neighbour is full the closest is taken
      anyway and the packet drops "queue_full" on enqueue, so congestion
      costs the packet but never bends its direction.
    - "random": uniform over all current neighbours.
The GS counts as a neighbour whenever it is in range. Being at distance 0
from itself it always wins greedy's comparison when present and not full.

The two Phase-1b rules, "rate" and "rl", are NOT implemented here: they live
in agents/ and are injected as a router object, which this module calls
through five duck-typed hooks.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs import c_interface
from fanet_sim.envs.channel import are_connected, euclidean_distance
from fanet_sim.envs.drone import Drone, Hello, NextHop, _link_row
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
    """Pick the candidate CLOSEST to the ground station, among those closer.

    Greedy geographic forwarding, as the literature usually means it: take
    the largest geographic step you can towards the destination.

    Two stages:

    1. Keep only the candidates STRICTLY closer to the ground station than
       the current holder. This is the progress condition; it is what stops
       a packet circling among peers, and it is why the rule can never
       revisit a drone.
    2. Among those whose queue still has room, pick the one closest to the
       GS. Ties break on the lower channel loss.

    If every closer candidate has a full queue, the closest is picked anyway
    and the caller drops the packet "queue_full" on enqueue. A full queue
    therefore costs the packet, but never steers it in a worse direction.

    Both inputs are things the drone itself knows: the distances from its own
    position and its neighbours', and the queue figure exactly, because the
    queue belongs to this drone. Neither reads an ACK counter.

    The ground station counts as a candidate whenever it is in range, keyed
    by the string ``"GS"``. Being at distance 0 from itself it always wins
    stage 2 when it is present and not full.

    Args:
        holder_position: Position of the drone currently holding the packet.
        candidates:      Reachable next hops, mapping key → position. Keys are
                         drone ids, plus ``"GS"`` when the GS is in range.
        gs_position:     Ground-station position as a NumPy array.
        link_full:       Per candidate, whether the holder's queue for that
                         link is at capacity. Missing keys count as not full.

    Returns:
        The winning candidate's key, or None when there are no candidates at
        all or none is strictly closer to the GS — a routing void, which the
        caller drops as "no_route".
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

    # Stage 2: closest to the GS; ties on the cleaner link.
    def rank(key: NextHop) -> Tuple[float, float]:
        return (
            euclidean_distance(closer[key], gs_position),
            channel.p_loss(
                euclidean_distance(holder_position, closer[key])
            ),
        )

    with_room = {key: pos for key, pos in closer.items()
                 if not link_full.get(key, False)}
    # Everything full: still take the closest, and let the caller drop it.
    return min(with_room or closer, key=rank)


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


#: The routing rules this environment can run. "rate" and "rl" live in
#: rl_sim/agents/ and must be supplied as a *router* object: the simulator
#: stays torch-free and never imports that package.
ROUTING_RULES = ("greedy", "random", "rate", "rl")

#: Rules that need an injected router.
ROUTER_RULES = ("rate", "rl")


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
        router: Optional[object] = None,
        c_agent: Optional[object] = None,
    ) -> None:
        """Create the environment (does NOT run reset automatically).

        Args:
            routing:     Routing rule: 'greedy', 'random', 'rate' or 'rl'.
            router:      For 'rate' and 'rl', the object that chooses hops and
                         learns. It lives in rl_sim/agents/ and is injected so
                         the simulator never imports that package. It must
                         provide five hooks:

                             select_next_hop(drone, candidates, pkt, came_from)
                             on_sent(drone_id, next_hop, packet_id)
                             on_delivered(pkt)
                             on_step_end(step)
                             on_reset()
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
        if config.LOOP_GUARD not in config.LOOP_GUARD_MODES:
            raise ValueError(
                f"unknown LOOP_GUARD {config.LOOP_GUARD!r}; expected one of "
                f"{config.LOOP_GUARD_MODES}"
            )
        if routing not in ROUTING_RULES:
            raise ValueError(
                f"unknown routing rule {routing!r}; expected one of {ROUTING_RULES}"
            )
        if routing in ROUTER_RULES and router is None:
            raise ValueError(
                f"routing {routing!r} needs a router object; build one with "
                f"agents.make_router({routing!r}, run_seed=...) and pass it as "
                f"router="
            )
        self.routing = routing
        self.router = router
        # Phase 2c. With config.C_RULE == "rl" the C-drones ask this object
        # for a waypoint each decision. Injected like the router, so
        # fanet_sim never imports agents/ and stays torch-free. It needs one
        # method: decide(observations) -> one action per observation.
        self.c_agent = c_agent
        self.episode_id = episode_id
        self.placement_seed = (
            config.PLACEMENT_SEED if placement_seed is None else placement_seed
        )
        self.run_seed = config.RUN_SEED if run_seed is None else run_seed
        # Built in reset(); see _seed_streams().
        self._rng_place: np.random.Generator
        self._rng_chan: np.random.Generator
        self._rng_route: np.random.Generator
        self._rng_mob: np.random.Generator
        self._rng_hello: np.random.Generator
        self._rng_cdrone: np.random.Generator
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
        # Phase 2a. Packets handed back to the routing rule because the link
        # they were queued on went away, and the ones that found nowhere to
        # go. Both counted over the measurement window only.
        self.reroutes: int = 0
        self.link_broken_drops: int = 0
        # Per M-drone, how many measured steps it had SOME path to the GS,
        # and how many measured steps there were in total.
        self.connected_steps: Dict[int, int] = {}
        self.measured_steps: int = 0
        self._reachable_from_gs: Optional[set] = None
        # Phase 2a step 2. Sends aimed at a neighbour the table still listed
        # but that had really moved out of range, and how many of those the
        # channel then lost.
        self.stale_sends: int = 0
        self.stale_link_drops: int = 0
        # The ground station broadcasts too, so drones can learn it is in
        # range; it has no table of its own, since it never routes.
        self._gs_next_hello_step: int = 0
        self._gs_hellos_sent: int = 0
        # Neighbour-table accuracy, summed over drones and measured steps:
        # how much of what a drone believed was real, and how much of what
        # was real it knew about.
        self._accuracy_precision: float = 0.0
        self._accuracy_recall: float = 0.0
        self._accuracy_samples: int = 0
        # Phase 2a step 3. Pairs of drones really within C_SAFE_DISTANCE_M of
        # each other, counted per step on TRUE positions, and the share of
        # those involving an M-drone, which does not avoid anybody.
        # Close passes, counted ONCE per approach rather than per step, and
        # split by the pair's kinds: "CC", "CM" or "MM".
        # Phase 2c. The C-drone agent's log: one row per decision, with the
        # observation it saw, the action it gave and the reward for the
        # second that followed.
        self.c_decisions: List[Dict[str, Any]] = []
        self._second_start: Dict[str, int] = {}
        self._flown_at_decision: Dict[int, float] = {}
        self.close_pass_events: Dict[str, int] = {}
        self.closest_approach_m: float = float("inf")
        self._close_now: set = set()
        # The C-drones' mean distance from the GS, one value per second.
        self.c_distance_series: List[float] = []
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
        self.reroutes = 0
        self.link_broken_drops = 0
        self.connected_steps = {}
        self.measured_steps = 0
        self.stale_sends = 0
        self.stale_link_drops = 0
        self._gs_next_hello_step = 0
        self._gs_hellos_sent = 0
        for drone in self.drones if self.drones else []:
            drone.hello_table.clear()
        self._accuracy_precision = 0.0
        self._accuracy_recall = 0.0
        self._accuracy_samples = 0
        self.c_decisions = []
        self._second_start = {}
        self._flown_at_decision = {}
        self.close_pass_events = {}
        self.closest_approach_m = float("inf")
        self._close_now = set()
        self.c_distance_series = []
        self._rx_counts = defaultdict(int)

        # Open a new logger for this episode (close any prior one).
        if self._logger is not None:
            self._logger.close()
        self._logger = EventLogger(self.log_path, episode_id=self.episode_id)

        if self.router is not None:
            self.router.on_reset()  # type: ignore[union-attr]

        self.drones = self._create_drones()
        self._assign_mobility()
        self._assign_hello_schedule()
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
                "mission": config.MISSION,
                "mission_distance_m": config.MISSION_DISTANCE_M,
                "start_radius_m": config.START_RADIUS_M,
                "min_spacing_m": config.MIN_SPACING_M,
                "require_connected_m": config.REQUIRE_CONNECTED_M,
                "loop_guard": config.LOOP_GUARD,
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
                "model": "fspl_sigmoid",
                "sensitivity_definition": "802.11 OFDM, 10% PER",
                "pt_dbm": config.PT_DBM,
                "gt_dbi": config.GT_DBI,
                "gr_dbi": config.GR_DBI,
                "f_hz": config.F_HZ,
                "rx_sensitivity_dbm": config.RX_SENSITIVITY_DBM,
                "sens_loss": config.SENS_LOSS,
                "loss_50_distance_m": config.LOSS_50_DISTANCE_M,
                "b_per_db": channel.B_PER_DB,
                "link_max_loss": config.LINK_MAX_LOSS,
                "max_link_distance_m": channel.max_link_distance(),
                "timestep_s": config.TIMESTEP,
                "routing": self.routing,
            },
            anchor={
                "notes": (
                    "Every parameter is a choice made for this project. The "
                    "radio figures (1 W, 2.4 GHz, and the receiver "
                    "sensitivity) are calibration values that put the loss "
                    "curve through 10% at 300 m and 50% at 350 m; those two "
                    "distances are the actual choice."
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
        :func:`are_connected` passes, i.e. loss below ``config.LINK_MAX_LOSS``
        — and allows paths through any drone,
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

        Under ``config.MISSION = "border"`` every drone is placed uniformly at
        random inside the arena. While ``config.STATIC_MODE`` is True that
        position is also its end point and nothing ever moves. Otherwise
        (Phase 1B) each M-drone additionally gets a random end point and flies
        the straight line to it.

        Under ``"near_gs"`` the whole fleet starts bunched around the GS
        instead; see :meth:`_near_gs_points`. Where a drone starts is the only
        difference — everything after this point is shared.

        Returns:
            List of Drone objects (M-drones first, then C-drones).
        """
        drones: List[Drone] = []
        drone_id = 0

        # Under "near_gs" every start position is drawn up front, because each
        # one has to be checked against the ones already placed. The order is
        # the same as the per-drone draws below: M-drones first, then C.
        near_gs = (self._near_gs_points()
                   if config.MISSION == "near_gs" else None)

        for index in range(config.NUM_M_DRONES):
            start = (near_gs[index] if near_gs is not None
                     else self._random_point())
            # The end point is NOT drawn here. It used to be, from this same
            # placement stream, whenever STATIC_MODE was off — which moved
            # every position after it and so handed a moving run a different
            # layout from a still one. B now comes from the mobility stream,
            # in _assign_mobility, after the layout has been accepted.
            end = start.copy()
            drones.append(Drone(
                drone_id=drone_id,
                drone_type="M",
                initial_position=start,
                speed=self._random_speed(),
                gs_position=self.gs_position,
                end_point=end,
            ))
            drone_id += 1

        for index in range(config.NUM_C_DRONES):
            drones.append(Drone(
                drone_id=drone_id,
                drone_type="C",
                initial_position=(
                    near_gs[config.NUM_M_DRONES + index]
                    if near_gs is not None else self._random_point()
                ),
                speed=self._random_speed(),
                gs_position=self.gs_position,
            ))
            drone_id += 1

        return drones

    def _near_gs_points(self) -> List[np.ndarray]:
        """Draw every drone's start position for the "near_gs" mission.

        One point per drone, M-drones first, uniform over the disc of
        ``config.START_RADIUS_M`` about the GS and at least
        ``config.MIN_SPACING_M`` from the GS and from every point already
        drawn. Rejected points are simply redrawn from the placement stream,
        so a placement seed still names one layout.

        Uniform over the DISC, not over radius and angle independently: the
        latter piles drones up at the centre, which is the one thing the
        minimum spacing exists to prevent. Hence the sqrt.

        Returns:
            ``NUM_M_DRONES + NUM_C_DRONES`` points inside the arena.

        Raises:
            RuntimeError: If a point cannot be placed within
                ``config.MAX_PLACEMENT_DRAWS`` attempts, which means the
                radius is too small to hold the fleet at this spacing.
        """
        centre = np.asarray(self.gs_position, dtype=np.float64)
        points: List[np.ndarray] = []

        for _ in range(config.NUM_M_DRONES + config.NUM_C_DRONES):
            for _attempt in range(config.MAX_PLACEMENT_DRAWS):
                angle = float(self._rng_place.uniform(0.0, 2.0 * np.pi))
                radius = config.START_RADIUS_M * float(
                    np.sqrt(self._rng_place.uniform(0.0, 1.0))
                )
                point = centre + radius * np.array(
                    [np.cos(angle), np.sin(angle)], dtype=np.float64
                )
                if radius < config.MIN_SPACING_M:
                    continue
                if all(float(np.linalg.norm(point - other))
                       >= config.MIN_SPACING_M for other in points):
                    points.append(point)
                    break
            else:
                raise RuntimeError(
                    f"could not place drone {len(points)} of "
                    f"{config.NUM_M_DRONES + config.NUM_C_DRONES} within "
                    f"{config.START_RADIUS_M} m of the GS at "
                    f"{config.MIN_SPACING_M} m spacing"
                )

        return points

    def _assign_mobility(self) -> None:
        """Give every M-drone its flight: direction, speed and start delay.

        Called once, AFTER a layout has been accepted, so the number of
        rejected draws cannot shift the mobility stream.

        Every draw here comes from ``_rng_mob`` and nothing else. The speed
        the placement stream drew in :meth:`_draw_drones` is overwritten: that
        draw stays where it is, still consuming exactly what it consumed in
        Phase 1, because removing it would shift every position after it and
        move all twenty layouts. It is ballast now, and this is the only place
        that says so.

        The flight is A -> B at constant speed:

        * A is the drone's Phase-1 position, where it already is;
        * the direction is uniform in [0, 360);
        * B is where that ray leaves the arena, pulled back
          ``config.BORDER_MARGIN_M`` so it lands inside. A drone already
          within the margin of the border in its own direction gets B = A and
          simply hovers, which is the honest answer rather than a nudge
          inwards that would bend its direction.

        With ``config.STATIC_MODE`` set this does nothing at all, so a static
        run never touches the mobility stream.
        """
        # C-drones fly at a fixed speed, and apply_velocity caps each command
        # at drone.speed — so this has to be set, or they would crawl at
        # whatever the placement stream happened to draw for them. It is set
        # even in the static world, where the C-drones may still fly. No draw
        # is made here.
        for drone in self.drones:
            if drone.drone_type == "C":
                drone.speed = config.C_DRONE_SPEED

        if config.STATIC_MODE:
            return

        for drone in self.drones:
            if drone.drone_type != "M":
                continue
            angle = float(self._rng_mob.uniform(0.0, 2.0 * np.pi))
            direction = np.array([np.cos(angle), np.sin(angle)],
                                 dtype=np.float64)
            drone.speed = float(self._rng_mob.uniform(
                config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX
            ))
            drone.start_delay_s = float(self._rng_mob.uniform(
                0.0, config.MAX_START_DELAY_S
            ))
            if config.MISSION == "near_gs":
                # The flight is a bounded distance outward, drawn AFTER the
                # three draws above so that turning the dial cannot shift
                # them: a layout's directions, speeds and delays are the same
                # at every D, and only the distance changes.
                distance = float(self._rng_mob.uniform(
                    0.5 * config.MISSION_DISTANCE_M,
                    config.MISSION_DISTANCE_M
                ))
                drone.end_point = self._clip_inside_border(
                    drone.position + distance * direction
                )
            else:
                drone.end_point = self._border_point(drone.position, direction)
            drone.arrived = bool(
                np.array_equal(drone.end_point, drone.position)
            )

    @staticmethod
    def _border_point(start: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Return where *direction* leaves the arena, minus the margin.

        Solves for the smallest positive t with ``start + t * direction`` on
        the arena boundary, then steps back ``config.BORDER_MARGIN_M`` along
        the ray. Clamped at the start point, so B is never behind A.

        Args:
            start:     The drone's position, inside the arena.
            direction: A unit vector.

        Returns:
            B, always inside the arena.
        """
        distances = []
        for axis, extent in ((0, config.WIDTH), (1, config.HEIGHT)):
            component = direction[axis]
            if component > 0.0:
                distances.append((extent - start[axis]) / component)
            elif component < 0.0:
                distances.append((0.0 - start[axis]) / component)
        # A unit vector always has a non-zero component, so this is never
        # empty; min() is the first boundary the ray meets.
        to_border = min(d for d in distances if d >= 0.0)
        travel = max(0.0, to_border - config.BORDER_MARGIN_M)
        return start + travel * direction

    @staticmethod
    def _clip_inside_border(point: np.ndarray) -> np.ndarray:
        """Clamp *point* to ``config.BORDER_MARGIN_M`` inside the arena.

        Used by the "near_gs" mission, whose flight distance is drawn without
        regard to where the border is, so a long flight can land outside. The
        prompt's "10 m inside the border" is ``BORDER_MARGIN_M``, read from
        config rather than typed again here.

        This clamps per axis, so a drone flying diagonally out of a corner
        keeps the component that still fits. That shortens its flight and
        bends its final heading, which is the honest consequence of a wall:
        the alternative, scaling the whole vector back, would stop it short
        of a border it had room to reach.

        Args:
            point: Where the flight would end, inside or outside the arena.

        Returns:
            A point inside the arena.
        """
        margin = config.BORDER_MARGIN_M
        return np.array([
            float(np.clip(point[0], margin, config.WIDTH - margin)),
            float(np.clip(point[1], margin, config.HEIGHT - margin)),
        ], dtype=np.float64)

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
        """(Re)build the four independent random streams from the two seeds.

        ``_rng_place`` is driven by ``placement_seed``. ``_rng_chan`` (channel
        loss and traffic timing) and ``_rng_route`` (the RANDOM routing rule)
        are two independent streams spawned off ``run_seed``, so consuming
        routing draws can never shift the channel or traffic draws.

        ``_rng_mob`` (Phase 2a: direction, speed and start delay) has its OWN
        SeedSequence keyed by the placement seed and a fixed tag, in the style
        of the agent's exploration stream. It is deliberately not a third
        spawn off either of the others: an independent stream cannot be
        shifted by a change in how many draws the others make, whichever way
        round that would have gone. Keying it on the placement seed makes a
        layout's trajectories a property of the layout, so greedy and random
        on the same placement fly identically.
        """
        self._rng_place = np.random.default_rng(self.placement_seed)
        chan_seed, route_seed = np.random.SeedSequence(self.run_seed).spawn(2)
        self._rng_chan = np.random.default_rng(chan_seed)
        self._rng_route = np.random.default_rng(route_seed)
        self._rng_mob = np.random.default_rng(
            np.random.SeedSequence([self.placement_seed,
                                    config.MOBILITY_STREAM_TAG])
        )
        # Phase 2a step 2. Hello schedules and hello reception draws, on their
        # own stream again. Keyed on the RUN seed, not the placement seed:
        # whether a hello is heard is a channel event, and it belongs with the
        # other things a run seed decides. Turning hellos on therefore cannot
        # shift a layout, a trajectory or a data-channel draw.
        self._rng_hello = np.random.default_rng(
            np.random.SeedSequence([self.run_seed, config.HELLO_STREAM_TAG])
        )
        # Phase 2a step 3. The C-drones' own randomness — only random_walk
        # needs any — on its own stream again, keyed on the PLACEMENT seed so
        # a layout's C-drone behaviour is the same under every routing rule,
        # the way the M-drone trajectories already are.
        self._rng_cdrone = np.random.default_rng(
            np.random.SeedSequence([self.placement_seed,
                                    config.C_DRONE_STREAM_TAG])
        )

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
        # Who can reach the GS this step, computed at most once and thrown
        # away when the step ends, since the drones move between steps.
        self._reachable_from_gs = None
        measured = config.is_warm(self.step_count)

        # 1. Move M-drones along their straight A -> B line, then refresh the
        #    neighbour sets. Phase 1 is static, so no drone moves there.
        if not config.STATIC_MODE:
            now_s = self._sim_time()
            for drone in self.drones:
                if drone.drone_type == "M":
                    drone.step_move(config.TIMESTEP, now_s=now_s)
        # The C-drones move whatever STATIC_MODE says. It is the MISSION that
        # is static in Phase 2c M0-M3: the M-drones hold their Phase-1
        # positions while the C-drones fly, which isolates what moving the
        # relays is worth from what the mission's movement costs. C_RULE
        # defaults to "hover", so a Phase-1 run still has nobody moving.
        # They decide from their tables, which hold what they heard BEFORE
        # this step, so they move on last step's picture.
        self._move_c_drones()
        self._recompute_links()
        self._update_active_links()

        # 1b. Neighbours that have gone quiet for too long drop out of the
        #     table, which is what makes their queues look broken below.
        self._expire_hello_entries(measured)

        # 1c. Links that went away this step: hand their queued packets back
        #     to the routing rule before anything else looks at a queue.
        self._rescue_broken_links(measured)

        # 1d. Who can still reach the GS at all, measured on the TRUE network,
        #     and how well each drone's table matches it.
        in_window = config.in_measurement_window(self.step_count)
        self._sample_connectivity(in_window)
        self._sample_table_accuracy(in_window)
        self._count_close_passes(in_window)
        self._sample_c_distance()

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

        # 7. Hellos go out LAST, so one heard now is first usable next step.
        self._exchange_hellos(measured)

        # 8. Radio idle/listen energy and accumulated rx energy for the step.
        for drone in self.drones:
            drone.consume_idle_energy()
            rx_n = self._rx_counts.get(drone.drone_id, 0)
            if rx_n:
                drone.consume_rx_energy(rx_n)

        # The routing rule settles packets whose TTL ran out this step, and
        # learns from them. Called before step_count advances, so a deadline
        # of created_at + TTL resolves on the same step the simulator expires
        # the packet.
        if self.router is not None:
            self.router.on_step_end(self.step_count)  # type: ignore[union-attr]

        self.step_count += 1

        # 8. Log per-step network state and per-drone state.
        self._log_step_and_drone_state()

        # Fixed episode length. In static mode that is the ONLY end condition:
        # an M-drone that never moves is trivially "arrived", so the
        # all-arrived shortcut below would end the episode on step 1.
        done = self.step_count >= config.MAX_STEPS
        if not config.STATIC_MODE and config.MISSION == "border":
            # Phase 1B: also stop once every M-drone has reached its end point,
            # since no mission traffic is left to generate or deliver.
            #
            # "border" only, and that is not a detail. Under "border" a flight
            # runs to the edge of the arena and takes most of the episode, so
            # this shortcut almost never fires before MAX_STEPS and the
            # measurement window is whole. Under "near_gs" a flight is a few
            # hundred metres and the whole fleet is parked early -- at
            # MISSION_DISTANCE_M = 350 every drone has arrived by step 688,
            # and at 0 before step 1 -- so the shortcut would cut the window
            # short and leave the late ten-second slices empty at exactly the
            # settings they were asked for. A curriculum rung therefore always
            # runs its full length, whether or not anyone is still flying.
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
            # Was there any path to the GS at this moment? "Loss while
            # connected" counts only packets for which there was, so a drone
            # that has flown out of the network is not charged for packets it
            # never had a way to deliver.
            pkt.source_connected = self._has_path_to_gs(drone.drone_id)
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
        return self._believed_links(drone)

    def _allowed_candidates(
        self, drone: Drone, pkt: Packet
    ) -> Dict[NextHop, np.ndarray]:
        """Return the next hops the loop guard permits for this packet.

        Under ``config.LOOP_GUARD == "path"`` every drone already in the
        packet's path is removed — the header carries that list, so the
        decision stays local. The GS is never removed: it is the destination,
        and a packet reaching it stops.

        Under ``"previous"`` nothing is removed here; "rate" and "rl" drop the
        immediately preceding hop themselves and greedy and random drop
        nothing, which is the behaviour Phase 1b was measured under.

        Args:
            drone: The drone routing the packet.
            pkt:   The packet being routed.

        Returns:
            The permitted next hops, key → position.
        """
        candidates = self._next_hop_candidates(drone)
        if config.LOOP_GUARD != "path":
            return candidates

        visited = set(pkt.path)
        return {
            key: position for key, position in candidates.items()
            if key == "GS" or key not in visited
        }

    def _select_next_hop(
        self,
        drone: Drone,
        candidates: Dict[NextHop, np.ndarray],
        pkt: Optional[Packet] = None,
    ) -> Optional[NextHop]:
        """Choose the link a packet should join, according to the routing rule.

        Called when a packet is created at *drone* or arrives there — never at
        send time.

        Args:
            drone:      The drone routing the packet.
            candidates: Reachable next hops from :meth:`_next_hop_candidates`.
            pkt:        The packet being routed. Only the injected rules need
                        it — for the loop guard and to key their decision.

        Returns:
            The chosen next-hop key, or None if the rule finds no usable hop
            (the caller then drops the packet with reason "no_route").
        """
        if self.routing in ROUTER_RULES:
            # The drone this packet just arrived from, so the rule can avoid
            # bouncing it straight back. path is [source, hop1, ...], and the
            # last entry is this drone.
            came_from = (
                pkt.path[-2] if pkt is not None and len(pkt.path) >= 2 else None
            )
            return self.router.select_next_hop(  # type: ignore[union-attr]
                drone, candidates, pkt, came_from
            )

        if self.routing == "random":
            return random_next_hop(candidates, self._rng_route)

        link_full = {key: drone.queue_is_full(key) for key in candidates}
        return greedy_next_hop(
            drone.position, candidates, self.gs_position, link_full
        )

    def _route_into_queue(
        self, drone: Drone, pkt: Packet, measured: bool,
        fail_reason: Optional[DropReason] = None,
        broken_link: Optional[NextHop] = None,
    ) -> None:
        """Commit *pkt* to one of *drone*'s link queues, or drop it.

        Three outcomes: no qualifying next hop drops the packet "no_route";
        a chosen link whose queue is full drops it "queue_full" at this drone;
        otherwise it waits on that link until its turn to be sent.

        *fail_reason* overrides all three failure labels. The broken-link
        rescue passes ``LINK_BROKEN``, because a packet that had a perfectly
        good next hop until its link vanished was not lost to a routing void
        or to congestion; it was lost to the link going away. Expiry is NOT
        overridden: a packet already out of TTL or hops is retired as such,
        which is what "respect TTL and the hop limit" means.

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

        def _fail(default: DropReason) -> None:
            """Drop *pkt*, under the rescue's label when there is one."""
            reason = default if fail_reason is None else fail_reason
            if fail_reason is not None:
                self.link_broken_drops += 1 if measured else 0
                if broken_link is not None:
                    drone.note_link_broken(broken_link, measured)
                    self._note_link(drone.drone_id, broken_link,
                                    "dropped_link_broken", measured)
            self._drop(pkt, reason, drone)

        reachable = self._next_hop_candidates(drone)
        if not reachable:
            # Nowhere to send it at all.
            _fail(DropReason.NO_ROUTE)
            return

        candidates = self._allowed_candidates(drone, pkt)
        if not candidates:
            # There were options, and the loop guard took the last one.
            _fail(DropReason.DEAD_END)
            return

        next_key = self._select_next_hop(drone, candidates, pkt)
        if next_key is None:
            # The rule found nothing usable — for greedy, nothing closer.
            _fail(DropReason.NO_ROUTE)
            return

        drone.note_offered(next_key, measured)
        self._note_link(drone.drone_id, next_key, "offered", measured)

        if not drone.enqueue(next_key, pkt):
            if fail_reason is not None:
                # The rescue found a hop, but its queue was full. The per-link
                # book charges the link that REFUSED it — the one just
                # offered — so every "offered" keeps exactly one outcome on
                # its own link. Only the packet's cause says link_broken,
                # which is the honest summary of why it died.
                self.link_broken_drops += 1 if measured else 0
                drone.note_queue_full(next_key, measured)
                self._note_link(drone.drone_id, next_key,
                                "dropped_queue_full", measured)
                self._drop(pkt, fail_reason, drone, next_hop=next_key)
                return
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
            # The link is checked again HERE, not only when the packet was
            # enqueued. The rescue at the top of the step should already have
            # emptied every dead queue, so this is the backstop that makes
            # "sent to the GS after flying out of range" impossible rather
            # than merely unlikely. A packet caught here goes back through the
            # routing rule like any other stranded one.
            if not self._link_is_live(drone, next_key):
                self.reroutes += 1 if measured else 0
                self._route_into_queue(
                    drone, pkt, measured,
                    fail_reason=DropReason.LINK_BROKEN,
                    broken_link=next_key,
                )
                continue

            # The channel draw uses where the receiver REALLY is, not where
            # it last said it was: the radio does not care what the neighbour
            # table believes. A send aimed at a neighbour that has since moved
            # out of range is "stale" — it still goes out, and still draws
            # against the same curve, but a loss is reported as stale_link so
            # the cost of an out-of-date table can be told apart from the
            # ordinary cost of the radio.
            if next_key == "GS":
                target_pos = self.gs_position
            else:
                target_pos = self.get_drone_by_id(next_key).position
            stale = not are_connected(drone.position, target_pos)
            if stale:
                self.stale_sends += 1 if measured else 0

            drone.note_sent(next_key, measured)
            self._note_link(drone.drone_id, next_key, "sent", measured)
            if self.router is not None:
                self.router.on_sent(  # type: ignore[union-attr]
                    drone.drone_id, next_key, pkt.packet_id
                )
            drone.consume_tx_energy()

            dist = euclidean_distance(drone.position, target_pos)
            if self._rng_chan.random() < channel.p_loss(dist):
                outcome = "stale_link" if stale else "channel"
                field = "lost_stale_link" if stale else "lost_channel"
                reason = (DropReason.STALE_LINK if stale
                          else DropReason.CHANNEL)
                if stale:
                    self.stale_link_drops += 1 if measured else 0
                drone.note_hop_ack(next_key, outcome, measured)
                self._note_link(drone.drone_id, next_key, field, measured)
                self._drop(pkt, reason, drone, next_hop=next_key)
                continue

            drone.note_hop_ack(next_key, "ok", measured)
            self._note_link(drone.drone_id, next_key, "acked", measured)
            self.tx_events.append((drone.drone_id, next_key))

            # A RELAY is another drone's packet moved on by this one. A
            # source handing off its own packet is not relaying, and a hop
            # into the GS is — it is the last relay of the chain. Counting
            # own hand-offs and skipping GS hops made "busiest" point at
            # whoever CREATED the most, which under greedy is whoever is
            # farthest out.
            if pkt.source_id != drone.drone_id:
                drone.relayed_since_hello += 1
                drone.note_relay(self.step_count, pkt.source_id)

            if next_key == "GS":
                pkt.relay_to("GS")
                pkt.mark_delivered(self.step_count)
                self.delivered.append(pkt)
                # End-to-end ACK: tell the source its packet arrived, and
                # walk it back along the path for the routing rule.
                self.get_drone_by_id(pkt.source_id).note_gs_ack(pkt.packet_id)
                if self.router is not None:
                    self.router.on_delivered(pkt)  # type: ignore[union-attr]
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

            next_hop = self.get_drone_by_id(next_key)
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


    # ------------------------------------------------------------------
    # Hello messages (Phase 2a step 2)
    # ------------------------------------------------------------------

    def _hello_period_steps(self) -> int:
        """Return the hello interval in whole simulation steps, at least one."""
        return max(1, int(round(config.HELLO_INTERVAL / config.TIMESTEP)))

    def _assign_hello_schedule(self) -> None:
        """Give every sender its first hello step.

        Each starts at a uniformly random point inside one interval, so the
        fleet does not broadcast in lockstep from the first step. Nothing is
        drawn while ``config.IDEAL_HELLO`` is set, which keeps the hello
        stream untouched in every run that does not use it.
        """
        self._gs_next_hello_step = 0
        if config.IDEAL_HELLO:
            return

        period = self._hello_period_steps()
        for drone in self.drones:
            drone.next_hello_step = int(self._rng_hello.integers(period))
            drone.relayed_since_hello = 0
        self._gs_next_hello_step = int(self._rng_hello.integers(period))

    def _next_hello_gap(self) -> int:
        """Return the gap to the next hello, in steps, jitter included.

        The interval is SHORTENED by a uniform fraction of itself, up to
        ``config.HELLO_JITTER_FRACTION``, the way OLSR jitters its control
        traffic: without it every drone that once lined up stays lined up.

        Returns:
            A gap of at least one step.
        """
        period = self._hello_period_steps()
        jitter = float(self._rng_hello.uniform(
            0.0, config.HELLO_JITTER_FRACTION
        ))
        return max(1, int(round(period * (1.0 - jitter))))

    def _exchange_hellos(self, measured: bool) -> None:
        """Broadcast this step's hellos and deliver the ones that are heard.

        Run at the END of the step, so a hello heard now is first usable on
        the next one — a drone cannot act on information it has only just
        received.

        Every other drone hears a hello with probability
        ``1 - p_loss(distance)`` over the TRUE distance, on the same curve the
        data uses, drawn from the hello stream. An entry is only taken if the
        ADVERTISED position is in range of the receiver: a drone will not
        believe in a neighbour that says it is 400 m away.

        Hellos are free here — no queue slot, no link capacity. They are
        counted, not charged.

        Args:
            measured: True once the warm-up is over, for the per-second rates.
        """
        if config.IDEAL_HELLO:
            return

        senders: List[Tuple[NextHop, str, np.ndarray, np.ndarray, int, int,
                            Optional[np.ndarray], np.ndarray]] = []
        for drone in self.drones:
            if self.step_count >= drone.next_hello_step:
                senders.append((
                    drone.drone_id, drone.drone_type, drone.position.copy(),
                    drone.velocity.copy(), drone.total_queued(),
                    drone.relayed_since_hello,
                    None if drone.target is None else drone.target.copy(),
                    drone.end_point.copy(),
                ))
                drone.next_hello_step = self.step_count + self._next_hello_gap()
                drone.relayed_since_hello = 0
                drone.hellos_sent += 1
        if self.step_count >= self._gs_next_hello_step:
            senders.append(("GS", "GS", self.gs_position.copy(),
                            np.zeros(2, dtype=np.float64), 0, 0, None,
                            self.gs_position.copy()))
            self._gs_next_hello_step = (
                self.step_count + self._next_hello_gap()
            )
            self._gs_hellos_sent += 1

        if not senders:
            return

        for (sender_id, sender_type, position, velocity, fill, relayed,
             target, end_point) in senders:
            for receiver in self.drones:
                if receiver.drone_id == sender_id:
                    continue
                distance = euclidean_distance(receiver.position, position)
                if self._rng_hello.random() < channel.p_loss(distance):
                    continue        # the hello itself was lost
                receiver.hellos_received += 1
                # Believe it only if what it SAYS puts it in range.
                if not are_connected(receiver.position, position):
                    continue
                receiver.hello_table[sender_id] = Hello(
                    sender_id=sender_id,
                    sender_type=sender_type,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    sent_at=self.step_count,
                    queue_fill=fill,
                    relayed=relayed,
                    target=None if target is None else target.copy(),
                    end_point=end_point.copy(),
                )

    # ------------------------------------------------------------------
    # C-drone movement (Phase 2a step 3)
    # ------------------------------------------------------------------

    def _move_c_drones(self) -> None:
        """Let every C-drone take one step under the configured rule.

        A C-drone decides from its OWN neighbour table and nothing else: the
        same out-of-date picture the router works from. It never reads a true
        position, which is why the rules can send one toward a drone that has
        already left.

        Order: pick a heading under the rule, then let the steering look at
        it. Steering can only turn the heading aside or refuse the step; it
        never changes the target.
        """
        if config.C_RULE == "hover":
            return

        if config.C_RULE == "rl" and self.step_count % \
                c_interface.steps_per_decision() == 0:
            self._agent_waypoints()

        for drone in self.drones:
            if drone.drone_type != "C":
                continue
            heading = self._c_heading(drone)
            if heading is None:
                drone.velocity = np.zeros(2, dtype=np.float64)
                continue
            if config.AVOID:
                heading = self._steer_around(drone, heading)
                if heading is None:
                    drone.avoid_waits += 1
                    drone.velocity = np.zeros(2, dtype=np.float64)
                    continue
            drone.apply_velocity(heading * config.C_DRONE_SPEED
                                 * config.TIMESTEP)

    def _agent_waypoints(self) -> None:
        """Ask the injected agent for one waypoint per C-drone.

        Runs on the decision steps only. Each drone's observation is built
        from its own table and counters alone, the agent answers with a
        normalised action, and that becomes the drone's target for the next
        second. The reward for the second just ended is attached to the
        PREVIOUS decision, which is the one that earned it.

        Raises:
            RuntimeError: If C_RULE is "rl" with no agent injected.
        """
        if self.c_agent is None:
            raise RuntimeError(
                'config.C_RULE is "rl" but no c_agent was injected into '
                'FANETEnv; pass c_agent=... the way the router is passed'
            )

        c_drones = [d for d in self.drones if d.drone_type == "C"]
        self._close_last_second(c_drones)

        observations = [c_interface.build_observation(self, drone)
                        for drone in c_drones]
        actions = self.c_agent.decide(observations)  # type: ignore[union-attr]
        if len(actions) != len(c_drones):
            raise RuntimeError(
                f"the agent returned {len(actions)} actions for "
                f"{len(c_drones)} C-drones"
            )

        for drone, observation, action in zip(c_drones, observations, actions):
            if action is None:
                # "Nowhere to go this second." A drone with no waypoint flies
                # nowhere AND announces nothing, which is what the Phase-2a
                # rules do when they find no target — and other C-drones'
                # target sharing reads those announcements, so the difference
                # is visible to the fleet, not just to this drone. The logged
                # action is the drone's own position, since that is the
                # waypoint a policy would have to output to stay put.
                drone.target = None
                waypoint = drone.position.copy()
                action = c_interface.action_from_waypoint(
                    waypoint, self.gs_position
                )
            else:
                action = np.asarray(action, dtype=np.float64)
                waypoint = c_interface.waypoint_from_action(
                    action, self.gs_position
                )
                drone.target = waypoint
            self._flown_at_decision[drone.drone_id] = drone.distance_flown
            self.c_decisions.append({
                "step": self.step_count,
                "drone_id": drone.drone_id,
                "observation": observation,
                # The critic's view of the same moment. Recorded here rather
                # than rebuilt later because the world moves on: by the time
                # an update runs, these positions are gone.
                "global_state": c_interface.build_global_state(self, drone),
                "action": np.asarray(action, dtype=np.float64).copy(),
                "waypoint": waypoint.copy(),
                "reward": None,
            })

        self._second_start = {
            "delivered": len(self.delivered),
            "dropped": len(self.dropped),
            "created": len(self.all_packets),
        }

    def _close_last_second(self, c_drones: List[Drone]) -> None:
        """Fill in the reward for the decisions taken a second ago.

        Args:
            c_drones: The C-drones, in the order they decided.
        """
        if not self._second_start:
            return
        delivered = len(self.delivered) - self._second_start["delivered"]
        dropped = len(self.dropped) - self._second_start["dropped"]
        created = len(self.all_packets) - self._second_start["created"]

        pending = self.c_decisions[-len(c_drones):]
        for drone, row in zip(c_drones, pending):
            if row["reward"] is not None:
                continue
            flown = (drone.distance_flown
                     - self._flown_at_decision.get(drone.drone_id, 0.0))
            row["reward"] = c_interface.flight_reward(
                delivered, dropped, created, flown, drone.speed
            )

    def _c_heading(self, drone: Drone) -> Optional[np.ndarray]:
        """Return the unit heading *drone* wants this step, or None to hover.

        Args:
            drone: A C-drone.

        Returns:
            A unit vector, or None when the rule says to stay put.
        """
        if config.C_RULE == "random_walk":
            return self._random_walk_heading(drone)
        if config.C_RULE == "rl":
            # The agent has already set the target; just fly to it. The stop
            # distance is the midpoint one, which is what "hovers within
            # 10 m of it" means.
            return self._heading_to_target(drone, config.C_MIDPOINT_STOP_M)
        return self._target_heading(drone)

    def _random_walk_heading(self, drone: Drone) -> Optional[np.ndarray]:
        """A straight line, a new direction every C_TURN_INTERVAL, bouncing.

        The bounce is a reflection off whichever wall it reached, so a drone
        that runs into the border keeps its speed and turns, rather than
        sliding along the wall or stopping.

        Args:
            drone: A C-drone.

        Returns:
            Its unit heading for this step.
        """
        turn_every = max(1, int(round(config.C_TURN_INTERVAL / config.TIMESTEP)))
        if drone.heading is None or self.step_count >= drone.next_decision_step:
            angle = float(self._rng_cdrone.uniform(0.0, 2.0 * np.pi))
            drone.heading = np.array([np.cos(angle), np.sin(angle)],
                                     dtype=np.float64)
            drone.next_decision_step = self.step_count + turn_every

        # Reflect off a wall it would otherwise walk through.
        step = drone.heading * config.C_DRONE_SPEED * config.TIMESTEP
        ahead = drone.position + step
        bounced = drone.heading.copy()
        if ahead[0] < 0.0 or ahead[0] > config.WIDTH:
            bounced[0] = -bounced[0]
        if ahead[1] < 0.0 or ahead[1] > config.HEIGHT:
            bounced[1] = -bounced[1]
        drone.heading = bounced
        return bounced

    def _target_heading(self, drone: Drone) -> Optional[np.ndarray]:
        """Pick or keep a target under "busiest"/"midpoint", and head for it.

        The target is re-chosen every ``config.C_DECISION_INTERVAL`` seconds.
        Between decisions the drone keeps flying to the point it picked, even
        if the neighbour it picked it from has since gone quiet.

        Args:
            drone: A C-drone.

        Returns:
            A unit heading, or None when there is no target or it is already
            close enough to one.
        """
        every = max(1, int(round(config.C_DECISION_INTERVAL / config.TIMESTEP)))
        if self.step_count >= drone.next_decision_step:
            drone.target = self._choose_target(drone)
            drone.next_decision_step = self.step_count + every

        stop = (config.C_BUSIEST_STOP_M if config.C_RULE == "busiest"
                else config.C_MIDPOINT_STOP_M)
        return self._heading_to_target(drone, stop)

    @staticmethod
    def _heading_to_target(drone: Drone,
                           stop_m: float) -> Optional[np.ndarray]:
        """Return the unit heading to the drone's target, or None to hover.

        Args:
            drone:  The C-drone.
            stop_m: How close counts as arrived.

        Returns:
            A unit heading, or None when there is no target or it is already
            close enough.
        """
        if drone.target is None:
            return None
        gap = drone.target - drone.position
        distance = float(np.linalg.norm(gap))
        if distance <= stop_m:
            return None
        return gap / distance

    def _choose_target(self, drone: Drone) -> Optional[np.ndarray]:
        """Return where *drone* should head, from its table alone.

        "busiest" takes the neighbour that relayed the most in the last
        second, ties going to the fuller queue, and heads for the position it
        last advertised. "midpoint" takes the M-drone it can hear that is
        farthest from the GS, and heads for the middle of the line between
        that drone and the GS — the gap the network is most likely to need
        bridged.

        With ``config.SHARE_TARGETS`` the drone skips any candidate within
        ``config.C_TARGET_SPREAD_M`` of a target another C-drone has
        announced, and takes the best one left. Without it they are free to
        pile onto the same point.

        Args:
            drone: A C-drone.

        Returns:
            A target point, or None when nothing qualifies.
        """
        taken = self._announced_targets(drone)

        if config.C_RULE == "busiest":
            ranked = sorted(
                (h for h in drone.hello_table.values() if h.relayed > 0),
                key=lambda h: (-h.relayed, -h.queue_fill, str(h.sender_id)),
            )
            for hello in ranked:
                if self._far_enough(hello.position, taken):
                    return hello.position.copy()
            return None

        # midpoint
        ranked = sorted(
            (h for h in drone.hello_table.values() if h.sender_type == "M"),
            key=lambda h: (-float(np.linalg.norm(h.position - self.gs_position)),
                           str(h.sender_id)),
        )
        for hello in ranked:
            point = (hello.position + self.gs_position) / 2.0
            if self._far_enough(point, taken):
                return point
        return None

    def _announced_targets(self, drone: Drone) -> List[np.ndarray]:
        """Return the targets other C-drones have told *drone* about."""
        if not config.SHARE_TARGETS:
            return []
        return [h.target for h in drone.hello_table.values()
                if h.sender_type == "C" and h.target is not None]

    @staticmethod
    def _far_enough(point: np.ndarray,
                    taken: Sequence[np.ndarray]) -> bool:
        """True if *point* is clear of every already-announced target."""
        return all(
            float(np.linalg.norm(point - other)) >= config.C_TARGET_SPREAD_M
            for other in taken
        )

    def _sensed_positions(self, drone: Drone) -> List[np.ndarray]:
        """Where everything within sensing range really is, right now.

        Steering is the one thing that does not go through the neighbour
        table. Keeping clear of another aircraft is a sensing job: a drone
        sees what is near it, and a hello that may be three seconds old is no
        basis for deciding whether a 3 m step is safe. Everything beyond
        ``config.SENSOR_RANGE_M`` is ignored — it cannot be reached in a step
        anyway, since that is more than sixteen steps of flying.

        Routing, target choice and target sharing still read the table alone.

        Args:
            drone: The C-drone looking around.

        Returns:
            True positions of the other drones within sensing range.
        """
        return [
            other.position for other in self.drones
            if other.drone_id != drone.drone_id
            and float(np.linalg.norm(other.position - drone.position))
            <= config.SENSOR_RANGE_M
        ]

    def _steer_around(self, drone: Drone,
                      heading: np.ndarray) -> Optional[np.ndarray]:
        """Return a heading that keeps clear, or None to stay put this step.

        Tries the planned heading first, then turns of
        ``config.C_AVOID_TURN_DEG`` either way, widening to
        ``config.C_AVOID_MAX_TURN_DEG``. The first one that keeps the drone
        at least ``config.C_SAFE_DISTANCE_M`` from every predicted position
        wins.

        Args:
            drone:   The C-drone.
            heading: The unit heading it wanted.

        Returns:
            A unit heading, or None when nothing is clear.
        """
        others = self._sensed_positions(drone)
        if not others:
            return heading

        step = config.C_DRONE_SPEED * config.TIMESTEP
        # Clear by D_SAFE plus one step. Even with true positions, every
        # C-drone decides at the same moment and they all move together, so
        # the other one can close up to a step's worth of the gap after this
        # drone has committed. Leaving that room makes the D_SAFE guarantee
        # hold whatever the other drone does.
        clearance = config.C_SAFE_DISTANCE_M + step
        angles = [0.0]
        turn = config.C_AVOID_TURN_DEG
        while turn <= config.C_AVOID_MAX_TURN_DEG + 1e-9:
            angles.extend((turn, -turn))
            turn += config.C_AVOID_TURN_DEG

        for index, degrees in enumerate(angles):
            radians = np.radians(degrees)
            rotation = np.array([
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ])
            candidate = rotation @ heading
            landing = drone.position + candidate * step
            if all(float(np.linalg.norm(landing - other)) >= clearance
                   for other in others):
                if index:
                    drone.avoid_turns += 1
                return candidate
        return None

    def _count_close_passes(self, measured: bool) -> None:
        """Count each approach inside C_SAFE_DISTANCE_M once, not per step.

        A pair that stays close for ten steps is ONE close pass, not ten:
        what matters is how often two aircraft came together, not how long
        they lingered. A pair counts again only after it has separated.

        Split by what was involved, because the three mean different things.
        C-C is the only kind the steering can do anything about; M-drones fly
        their mission paths and avoid nobody, so C-M and M-M are a property
        of the layout.

        Args:
            measured: True on the steps inside the creation window.
        """
        if not measured:
            return
        drones = self.drones
        for i, first in enumerate(drones):
            for second in drones[i + 1:]:
                pair = (first.drone_id, second.drone_id)
                gap = float(np.linalg.norm(first.position - second.position))
                if gap < config.C_SAFE_DISTANCE_M:
                    kinds = "".join(sorted((first.drone_type,
                                            second.drone_type)))
                    if pair not in self._close_now:
                        self._close_now.add(pair)
                        self.close_pass_events[kinds] = (
                            self.close_pass_events.get(kinds, 0) + 1
                        )
                    if gap < self.closest_approach_m:
                        self.closest_approach_m = gap
                else:
                    self._close_now.discard(pair)

    def _sample_c_distance(self) -> None:
        """Record the C-drones' mean distance from the GS, once per second."""
        every = max(1, int(round(1.0 / config.TIMESTEP)))
        if self.step_count % every:
            return
        distances = [
            float(np.linalg.norm(d.position - self.gs_position))
            for d in self.drones if d.drone_type == "C"
        ]
        if distances:
            self.c_distance_series.append(float(np.mean(distances)))

    def _expire_hello_entries(self, measured: bool) -> None:
        """Forget neighbours that have not said anything for too long.

        A neighbour is dropped after ``config.HELLO_TIMEOUT_INTERVALS``
        intervals of silence. Packets queued for it are then re-routed by the
        same rescue that handles a link going away, because from the drone's
        side that is exactly what has happened.

        Args:
            measured: True once the warm-up is over.
        """
        if config.IDEAL_HELLO:
            return

        timeout = int(round(
            config.HELLO_TIMEOUT_INTERVALS * self._hello_period_steps()
        ))
        for drone in self.drones:
            dead = [
                key for key, hello in drone.hello_table.items()
                if self.step_count - hello.sent_at > timeout
            ]
            for key in dead:
                drone.hello_table.pop(key)

    def _believed_links(self, drone: Drone) -> Dict[NextHop, np.ndarray]:
        """Return what *drone* currently believes it can reach, key → position.

        With ``config.IDEAL_HELLO`` this is the true neighbour set and the
        true positions, which is what every Phase-1 result was measured with.
        Otherwise it is the neighbour table, and the positions are the ones
        the neighbours last ADVERTISED — out of date by however long ago the
        hello was sent.

        Args:
            drone: The drone about to decide.

        Returns:
            Dict mapping next-hop key to the position the router should use.
        """
        if config.IDEAL_HELLO:
            candidates: Dict[NextHop, np.ndarray] = {
                nid: nbr.position for nid, nbr in drone.neighbors.items()
            }
            if are_connected(drone.position, self.gs_position):
                candidates["GS"] = self.gs_position
            return candidates
        return {key: hello.position
                for key, hello in drone.hello_table.items()}

    def _sample_table_accuracy(self, measured: bool) -> None:
        """Measure how well each drone's table matches the real network.

        Two numbers per drone: of the neighbours it believes it has, the share
        that are real, and of its real neighbours, the share it knows about.
        Both are 1.0 under IDEAL_HELLO by construction.

        A drone with nothing to compare — no believed neighbours, or no real
        ones — is left out of that side's average rather than counted as a
        perfect score.

        Args:
            measured: True on the steps inside the creation window.
        """
        if not measured:
            return
        for drone in self.drones:
            believed = set(self._believed_links(drone))
            real = set(drone.neighbors)
            if are_connected(drone.position, self.gs_position):
                real.add("GS")
            if not believed and not real:
                continue
            hit = believed & real
            self._accuracy_precision += (
                len(hit) / len(believed) if believed else 1.0
            )
            self._accuracy_recall += len(hit) / len(real) if real else 1.0
            self._accuracy_samples += 1

    def _link_is_live(self, drone: Drone, key: NextHop) -> bool:
        """True if *drone* can still use the link to *key* right now.

        The one test used both by the broken-link rescue and by the send-time
        check, so the two can never disagree about what "still a link" means.

        Args:
            drone: The drone that owns the queue.
            key:   Neighbour id, or "GS".

        Returns:
            Whether that next hop is still reachable.
        """
        if config.IDEAL_HELLO:
            if key == "GS":
                return bool(are_connected(drone.position, self.gs_position))
            return key in drone.neighbors
        # With real hellos the drone acts on what it was told. A queue is
        # kept while the table still lists that neighbour, and emptied when
        # the entry times out — whether or not the neighbour is really there.
        return key in drone.hello_table

    def _rescue_broken_links(self, measured: bool) -> None:
        """Re-route everything queued on a link that no longer exists.

        Called at the top of the step, right after the drones have moved and
        the neighbour sets have been rebuilt. For every queue whose next hop
        is gone, each packet goes back through the drone's own routing rule
        exactly as if it had just arrived — same visited list, same TTL, same
        hop budget — and the empty queue is then deleted.

        A packet the rule cannot place, for any reason, is dropped
        "link_broken": there is no sense in calling it "no_route" or
        "queue_full" when the thing that killed it was the link going away
        while it waited.

        Args:
            measured: True once the warm-up is over.
        """
        if config.STATIC_MODE and config.IDEAL_HELLO:
            # Nothing moves and knowledge is perfect, so no link can break.
            # Skipping keeps the static world bit-for-bit what Phase 1
            # produced. With real hellos a table entry can still time out
            # even while nobody moves, so the rescue has to run.
            return

        for drone in self.drones:
            broken = [
                key for key in list(drone.link_queues)
                if drone.link_queues[key] and not self._link_is_live(drone, key)
            ]
            for key in broken:
                stranded = drone.link_queues.pop(key)
                for pkt in stranded:
                    self.reroutes += 1 if measured else 0
                    self._route_into_queue(
                        drone, pkt, measured,
                        fail_reason=DropReason.LINK_BROKEN,
                        broken_link=key,
                    )
            # Queues that went away empty leave nothing to re-route, but the
            # key should still go: it is no longer a link.
            for key in [k for k in list(drone.link_queues)
                        if not drone.link_queues[k]
                        and not self._link_is_live(drone, k)]:
                drone.link_queues.pop(key)

    def _has_path_to_gs(self, drone_id: int) -> bool:
        """True if *drone_id* has some path to the GS right now.

        Reads the set computed once per step by :meth:`_sample_connectivity`,
        so asking for eighteen drones costs one graph, not eighteen.

        Args:
            drone_id: The drone to ask about.

        Returns:
            Whether a path exists on the TRUE network.
        """
        if self._reachable_from_gs is None:
            graph, gs_label = build_graph_with_gs(self.drones, self.gs_position)
            self._reachable_from_gs = nx.node_connected_component(
                graph, gs_label
            )
        return drone_id in self._reachable_from_gs

    def _sample_connectivity(self, measured: bool) -> None:
        """Note which M-drones can still reach the GS, on the true network.

        One graph per step over the real positions — not what any drone
        believes — so "connected" means there is genuinely some path, however
        lossy, from that M-drone to the ground station.

        Args:
            measured: True on the steps inside the CREATION window
                      [WARMUP_STEPS, MAX_STEPS - DRAIN_STEPS). The same window
                      the loss numbers use, so "x% of the time connected" and
                      "y% of packets lost" describe the same stretch of the
                      run.
        """
        if not measured:
            return
        self.measured_steps += 1

        if self._reachable_from_gs is None:
            graph, gs_label = build_graph_with_gs(self.drones, self.gs_position)
            self._reachable_from_gs = nx.node_connected_component(
                graph, gs_label
            )
        reachable = self._reachable_from_gs
        for drone in self.drones:
            if drone.drone_type != "M":
                continue
            if drone.drone_id in reachable:
                self.connected_steps[drone.drone_id] = (
                    self.connected_steps.get(drone.drone_id, 0) + 1
                )
            else:
                self.connected_steps.setdefault(drone.drone_id, 0)

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
