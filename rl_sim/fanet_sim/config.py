"""
config.py — All simulation parameters in one place.

Every constant used by the simulator lives here. Change values here and
re-run; no other file needs to be touched.

Each value is tagged with its status:
    [DECIDED]      fixed for Phase 1A.
    [PROVISIONAL]  a temporary default, marked "PROVISIONAL - to confirm"
                   so it is easy to find and change later.

----------------------------------------------------------------------------
Relationship to IQMR (Sharvari et al., 2024, arXiv:2408.09109) §V.A
----------------------------------------------------------------------------
Earlier versions of this file sized the world to match IQMR's scenario
(1750 x 1750 m, 36 M + 14 C drones). Phase 1A does NOT: the world is a
900 x 900 m square with 18 M + 7 C drones, chosen for this project
(Phase 1A). Those are our numbers, not IQMR's.

What still comes from IQMR is the speed range (10-30 m/s) and the energy
budget (11.1 V x 5200 mAh = 207792 J). The radio no longer does: as of
Checkpoint 7 the channel is a logistic loss curve in distance fitted to
measured UAV links by Rosati et al. (arXiv:1406.4399), and how far a link
reaches falls out of that curve rather than out of a power budget.

The simulation is 2-D. There is no MAC layer and no interference model.
"""

# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------
AREA_SIZE: float = 1000.0       # radius in meters (3-D sphere), used in 3-D mode
USE_3D: bool = False             # start False; extend to 3-D later
WIDTH: float = 900.0             # metres — 900x900 m square arena          [DECIDED]
HEIGHT: float = 900.0            # metres                                    [DECIDED]

# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------
NUM_M_DRONES: int = 18           # mission drones: create AND relay packets  [DECIDED]
NUM_C_DRONES: int = 7            # communication drones: relay only          [DECIDED]
DRONE_SPEED_MIN: float = 10.0    # m/s — from IQMR
DRONE_SPEED_MAX: float = 30.0    # m/s — from IQMR

# There is no COMM_RANGE constant: how far a link reaches now falls out of the
# loss curve below. Ask channel.max_link_distance() for the single number.

# ---------------------------------------------------------------------------
# Static world (Phase 1A)
# ---------------------------------------------------------------------------
# In Phase 1A nothing moves: drones are placed uniformly at random once per run
# and hold that position for the whole episode. The mobility code in drone.py
# (Drone.step_move / Drone.apply_velocity) is kept for Phase 1B and is simply
# not called while this flag is True.
STATIC_MODE: bool = True         # no movement during a run                  [DECIDED]

# ---------------------------------------------------------------------------
# Layout filter
# ---------------------------------------------------------------------------
# Only accept placements in which EVERY M-drone has a path to the GS, using the
# same link rule as the simulator (an edge wherever the loss is below
# LINK_MAX_LOSS) and allowing paths through any drone, C-drones included.
# C-drones themselves need no path.
#
# A rejected layout is discarded and another is drawn from the SAME placement
# stream, so a given placement_seed still always yields the same accepted
# layout. FANETEnv.placement_draws records how many draws that took.
#
# This removes stranded M-drones, whose packets are lost as "no_route" no
# matter what the routing rule does. Greedy can still reach a dead end -- a
# drone that has a path to the GS but no neighbour closer to it -- and those
# still drop as "no_route".
REQUIRE_CONNECTED_M: bool = True  # every M-drone must reach the GS        [DECIDED]
MAX_PLACEMENT_DRAWS: int = 10000  # give up rather than loop forever

# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
INITIAL_ENERGY: float = 207792.0  # joules  (11.1 V × 5200 mAh, from IQMR)
ENERGY_PER_MOVE: float = 0.1      # joules per meter flown (approximate)
ENERGY_PER_TX: float = 0.01       # joules per packet transmitted
ENERGY_PER_RX: float = 0.005      # joules per packet received
ENERGY_PER_IDLE: float = 0.001    # joules per timestep listening (idle radio)

# ---------------------------------------------------------------------------
# Packets
# ---------------------------------------------------------------------------
# Traffic load: each M-drone creates ONE packet every PACKET_INTERVAL_STEPS
# steps. At TIMESTEP = 0.1 s the four loads in the Phase-1A grid are
#     10 steps = 1000 ms    5 steps = 500 ms    2 steps = 200 ms    1 step = 100 ms
# Each M-drone also gets a random start offset in [0, interval), so they do not
# all create their packets on the same step.
# 200 ms is the working load for the next phases; the experiment still sweeps
# all four.
PACKET_INTERVAL_STEPS: int = 2    # 200 ms — the working load             [DECIDED]
TRAFFIC_LOADS_MS: tuple = (1000, 500, 200, 100)   # the grid in item K
RANDOM_TRAFFIC_OFFSETS: bool = True   # PROVISIONAL - to confirm

PACKET_SIZE: int = 512            # bytes
MAX_HOPS: int = 10                # a packet may take at most this many hops  [DECIDED]
PACKET_TTL: int = 50              # timesteps before a packet expires         [DECIDED]

# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
# ONE FIFO QUEUE PER OUTGOING LINK: each drone keeps one queue per current
# neighbour, plus one for the GS link when the GS is in range. The next hop --
# and so the queue a packet joins -- is decided when the packet is created at a
# drone or arrives at it, not when it is sent.
#
# All of a drone's link queues send in the same step. There is no collision
# model, so there is no per-drone send limit: a drone with five links may send
# five packets in a step, one per link. Receiving is unlimited.
#
# A packet put into a full link queue is dropped at that drone with reason
# "queue_full".
QUEUE_CAPACITY: int = 10          # max packets per LINK queue                [DECIDED]
MAX_TX_PER_STEP: int = 1          # packets each LINK queue may send per step,
                                  # oldest first                             [DECIDED]

# ---------------------------------------------------------------------------
# Channel loss and link existence
# ---------------------------------------------------------------------------
# A logistic curve in DISTANCE, fitted to measured UAV links in
#   Rosati et al., "Dynamic Routing for Flying Ad Hoc Networks",
#   arXiv:1406.4399
#
#     p_loss(d) = 1 / (1 + exp(-LOSS_SLOPE_PER_M * (d - LOSS_50_DISTANCE_M)))
#
# which is the paper's 1 / (1 + exp(-(0.025*d - 8.9))). Loss is ~0.2% at 100 m,
# ~6.6% at 250 m, 50% at 356 m and ~99% at 540 m. The same curve applies to the
# last hop into the GS. A lost transmission is dropped with reason "channel".
#
# REPLACES the earlier model, which took link existence from a free-space path
# loss budget (Pt 30 dBm, 2.4 GHz, receiver sensitivity -54 dBm -> a hard
# 249.69 m range) and loss from exp(-k * M) on the margin M above sensitivity,
# with k = 0.8. That made loss ~1 right at 250 m, which measured UAV links do
# not show. The FSPL code is gone; scripts/plot_ploss.py keeps a local copy of
# the old curve purely to draw it alongside the new one.
LOSS_50_DISTANCE_M: float = 356.0  # distance where p_loss = 0.5  PROVISIONAL - to confirm
LOSS_SLOPE_PER_M: float = 0.025    # measured steepness           PROVISIONAL - to confirm

# A link exists while its loss is below this. One rule everywhere: neighbour
# sets, the GS link, the connected-layout filter and the range check. At the
# values above this puts the edge at ~539.8 m.
LINK_MAX_LOSS: float = 0.99        # PROVISIONAL - to confirm

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
TIMESTEP: float = 0.1             # seconds per simulation step
MAX_STEPS: int = 1000             # steps per episode                 PROVISIONAL - to confirm
NUM_EPISODES: int = 1             # number of episodes (for testing)

# ---------------------------------------------------------------------------
# Metric window (warm-up and drain cuts)
# ---------------------------------------------------------------------------
# Metrics are computed only over packets CREATED inside the window
#     WARMUP_STEPS <= created_at < MAX_STEPS - DRAIN_STEPS
# The warm-up cut lets queues fill before anything is measured. The drain cut
# is one full TTL wide, so every measured packet has had time to finish (be
# delivered, or dropped, or time out) before the episode ends.
WARMUP_STEPS: int = 50            # PROVISIONAL - to confirm
DRAIN_STEPS: int = PACKET_TTL     # PROVISIONAL - to confirm


def measurement_window() -> tuple:
    """Return the (first, last_exclusive) creation steps that count as measured.

    Returns:
        A ``(start, stop)`` pair of step indices. A packet counts toward the
        reported metrics when ``start <= created_at < stop``.
    """
    return (WARMUP_STEPS, MAX_STEPS - DRAIN_STEPS)


def in_measurement_window(step: int) -> bool:
    """True if a packet created at *step* counts toward the reported metrics.

    Args:
        step: The step a packet was created at.

    Returns:
        True when the step falls inside :func:`measurement_window`.
    """
    start, stop = measurement_window()
    return start <= step < stop


def is_warm(step: int) -> bool:
    """True once the warm-up is over at simulation *step*.

    Instantaneous per-step quantities — per-link transmissions and queue
    occupancy — are measured from here on. They need no drain cut, since a
    transmission does not need time to finish the way a packet does.

    Args:
        step: The current simulation step.

    Returns:
        True when *step* is at or past WARMUP_STEPS.
    """
    return step >= WARMUP_STEPS

# ---------------------------------------------------------------------------
# Ground station
# ---------------------------------------------------------------------------
# The GS receives every packet that arrives: no per-step limit and no queue.
GS_POSITION: tuple = (450.0, 450.0)   # centre of the 900 x 900 m arena      [DECIDED]

# ---------------------------------------------------------------------------
# M-drone mobility (Phase 1B; unused while STATIC_MODE is True)
# ---------------------------------------------------------------------------
# Each M-drone is given a random start point (its initial position) and a
# random end point at episode start; it flies the straight line between them
# and then stops.
M_DRONE_MOBILITY: str = "straight_line"
WAYPOINT_ARRIVAL_THRESHOLD: float = 2.0  # metres — drone is "at" its end point when this close

# ---------------------------------------------------------------------------
# Random seeds
# ---------------------------------------------------------------------------
# Two independent seeds, so one layout can be re-run with different randomness:
#   PLACEMENT_SEED — drone positions (and speeds / end points).
#   RUN_SEED       — channel loss draws and traffic start offsets.
# The RANDOM routing rule draws from a third stream spawned off RUN_SEED, so
# its choices never shift the channel or traffic draws. Greedy and random
# therefore run on identical placement and identical channel/traffic randomness.
PLACEMENT_SEED: int = 42
RUN_SEED: int = 42

# ---------------------------------------------------------------------------
# Event log (Stage 1 output — consumed by scripts/analyze.py in Stage 2)
# ---------------------------------------------------------------------------
LOG_DIR: str = "logs"             # directory for per-episode JSONL files
LOG_DRONE_STATE_EVERY_STEP: bool = True   # if False, only one snapshot at episode end
