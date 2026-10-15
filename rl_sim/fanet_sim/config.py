"""
config.py — All simulation parameters in one place.

Every constant used by the simulator lives here. Change values here and
re-run; no other file needs to be touched.

Each value is tagged with its status:
    [DECIDED]      fixed for Phase 1A.
    [PROVISIONAL]  a temporary default, marked "PROVISIONAL - to confirm"
                   so it is easy to find and change later.

----------------------------------------------------------------------------
Where the numbers come from
----------------------------------------------------------------------------
They are OURS. The 900 x 900 m square, the 18 M + 7 C fleet, the queue sizes,
the loads and the 350 m link edge are choices made for this project, not
values taken from a paper.

The radio figures (1 W at 2.4 GHz into a -54 dBm receiver) are calibration
values: they are what makes the loss curve pass through 10% at 250 m and 50%
at 350 m. Those two distances are the real choice; the radio figures follow
from them and no result depends on them separately.

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
DRONE_SPEED_MIN: float = 10.0    # m/s — unused while static
DRONE_SPEED_MAX: float = 30.0    # m/s — unused while static

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
INITIAL_ENERGY: float = 207792.0  # joules  (11.1 V × 5200 mAh)
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
# Channel loss and link existence                      [DECIDED 2026-10-13]
# ---------------------------------------------------------------------------
# A free-space path loss budget gives each link a MARGIN above the receiver's
# sensitivity, and that margin drives an S-curve for the chance one
# transmission is lost:
#
#     FSPL_dB(d) = 20*log10(d) + 20*log10(F_HZ) + 20*log10(4*pi/c)
#     M(d)       = PT_DBM + GT_DBI + GR_DBI - FSPL_dB(d) - RX_SENSITIVITY_DBM
#     p_loss(M)  = 1 / (1 + A * exp(b * M)),   A = (1 - SENS_LOSS) / SENS_LOSS
#
# The sensitivity is NOT a wall. It is the point where 10% of transmissions
# are lost, following 802.11 OFDM, which defines receiver sensitivity as the
# power giving 10% packet error rate on 1000-octet frames. That puts the 10%
# point at 249.69 m. Links beyond it have a NEGATIVE margin and still exist;
# they simply lose more. The same curve covers the last hop into the GS, and a
# lost transmission is dropped with reason "channel".
#
# b is DERIVED, never hardcoded: it is whatever value puts 50% loss at
# LOSS_50_DISTANCE_M. See channel.py.
#
# History: an earlier FSPL model took loss as exp(-0.8 * M), which reaches
# 100% loss right at the sensitivity and so put a wall at 250 m; it is
# replaced by this S-curve, where the sensitivity means 10% loss.
PT_DBM: float = 30.0               # transmit power, 1 W                [DECIDED]
GT_DBI: float = 2.0                # transmit antenna gain              [DECIDED]
GR_DBI: float = 2.0                # receive antenna gain               [DECIDED]
F_HZ: float = 2.4e9                # carrier frequency                  [DECIDED]
RX_SENSITIVITY_DBM: float = -54.0  # receiver sensitivity               [DECIDED]
SENS_LOSS: float = 0.10            # loss AT the sensitivity point      [DECIDED]

# The distance where half of all transmissions are lost, which is also the
# link edge. A DESIGN CHOICE, not a physical constant: with 1 W in free space
# the radio would reach kilometres. Chosen from geometry alone, before any
# routing experiment, and FIXED. See CLAUDE.md and out/edge_scan.csv.
LOSS_50_DISTANCE_M: float = 350.0  #                                    [DECIDED]

# A link exists while its loss is below this. One rule everywhere: neighbour
# sets, the GS link, the connected-layout filter and the range check. At 0.5
# the edge sits at exactly LOSS_50_DISTANCE_M.
LINK_MAX_LOSS: float = 0.5         #                                    [DECIDED]

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
# Loop guard
# ---------------------------------------------------------------------------
# A DESIGN ASSUMPTION ABOUT THE NETWORK, not about any one routing rule: the
# packet header carries the list of drones it has already visited, so every
# drone can refuse to send it somewhere it has been. With 25 drones that list
# costs about 4 bytes. The idea is BGP's AS_PATH loop prevention.
#
#   "path"     options exclude EVERY drone already in the packet's path. The
#              GS is never excluded. Applies to greedy, random, rate and rl
#              alike, in training (including exploration picks) and at test.
#   "previous" the earlier behaviour: rate and rl exclude only the drone the
#              packet just came from, greedy and random exclude nothing. Kept
#              as the baseline the Phase-1b results were measured under.
#
# Under "path", a packet whose every neighbour has already been visited (and
# which cannot reach the GS) is dropped with reason "dead_end" -- distinct
# from "no_route", which means there was nowhere to send it in the first
# place. Under "previous" those cases stay "no_route".
LOOP_GUARD: str = "path"           # PROVISIONAL - to confirm
LOOP_GUARD_MODES: tuple = ("path", "previous")

# ---------------------------------------------------------------------------
# Phase 1b routing agents (rl_sim/agents/ only)
# ---------------------------------------------------------------------------
# Two learned-ish rules sit beside greedy and random. Both pick among the
# current neighbours plus the GS when in range, EXCLUDING the drone the packet
# just came from. Both are driven by a per-link delivery rate each drone keeps
# from end-to-end ACKs: the mean of the last DELIVERY_RATE_WINDOW resolved
# outcomes on that link, seeded at 1 - channel_loss before any outcome.
#
# The simulator itself knows nothing about these: fanet_sim/ stays torch-free
# and the router is injected into FANETEnv.
DELIVERY_RATE_WINDOW: int = 50     # resolved outcomes kept per link  PROVISIONAL - to confirm

# "rl": a 3 -> RL_HIDDEN -> RL_HIDDEN -> 1 network with ReLU and a sigmoid
# output, shared by every drone, scoring each option from three local inputs
# (delivery_rate, own queue fill, channel loss).
RL_HIDDEN: int = 32                # PROVISIONAL - to confirm
RL_EPSILON_TRAIN: float = 0.1      # exploration while training    PROVISIONAL - to confirm
RL_EPSILON_EVAL: float = 0.0       # none while evaluating                   [DECIDED]
RL_LEARNING_RATE: float = 1e-3     # Adam                          PROVISIONAL - to confirm
RL_BATCH_SIZE: int = 32            # examples before one gradient step  PROVISIONAL - to confirm

# The agent's exploration stream is derived from run_seed through its OWN
# SeedSequence, NOT by spawning a third stream off the existing two — spawning
# a third would change _rng_chan and silently move every Phase 1a result.
AGENT_STREAM_TAG: int = 0xA6E17    # arbitrary, fixed                        [DECIDED]

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
