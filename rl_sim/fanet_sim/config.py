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

What still comes from IQMR is the radio only: speed range (10-30 m/s),
transmit power (1 W = 30 dBm), the 2.4 GHz carrier and the energy budget
(11.1 V x 5200 mAh = 207792 J), all in this file or channel.py. The receiver
sensitivity (-54 dBm, channel.py) is derived rather than picked, so the FSPL
link test reproduces IQMR's reported 250 m range at IQMR's 1 W transmit
power. The effective range that falls out is 249.7 m.

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
TX_POWER: float = 1.0            # watts — from IQMR (1 W = 30 dBm); the FSPL
                                 # channel uses PT_DBM in channel.py as source of truth.

# COMM_RANGE is not a tunable. The FSPL channel model in channel.py decides
# link existence from received signal power. This alias is the single-number
# effective range for code that wants one (e.g. the visualiser).
# To change the radio range, edit the parameters at the top of channel.py.
from fanet_sim.envs.channel import MAX_LINK_DISTANCE_M as _MAX_LINK_DISTANCE_M
COMM_RANGE: float = _MAX_LINK_DISTANCE_M

# ---------------------------------------------------------------------------
# Static world (Phase 1A)
# ---------------------------------------------------------------------------
# In Phase 1A nothing moves: drones are placed uniformly at random once per run
# and hold that position for the whole episode. The mobility code in drone.py
# (Drone.step_move / Drone.apply_velocity) is kept for Phase 1B and is simply
# not called while this flag is True.
STATIC_MODE: bool = True         # no movement during a run                  [DECIDED]

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
# Traffic load. Item G of the Phase-1A spec replaces this per-step rate with a
# per-M-drone creation INTERVAL (every 10 / 5 / 2 / 1 steps) plus a random start
# offset; that lands at CHECKPOINT 4. Until then each M-drone creates
# PACKET_RATE packets per step.
PACKET_RATE: int = 1              # packets created per M-drone per step

PACKET_SIZE: int = 512            # bytes
MAX_HOPS: int = 10                # a packet may take at most this many hops  [DECIDED]
PACKET_TTL: int = 50              # timesteps before a packet expires         [DECIDED]

# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
# One FIFO queue per drone. An M-drone's own new packets and the packets it
# relays share the same queue and the same per-step send budget. A packet that
# arrives at a full queue is dropped with reason "queue_full".
QUEUE_CAPACITY: int = 10          # max packets held per drone                [DECIDED]
MAX_TX_PER_STEP: int = 1          # packets a drone may send per step,
                                  # oldest first                             [DECIDED]

# ---------------------------------------------------------------------------
# Channel loss
# ---------------------------------------------------------------------------
# For every transmission over an existing link:
#     p_loss = exp(-CHANNEL_LOSS_K * M),  M = received power - sensitivity, in dB
# p_loss is 1 when there is no link (M <= 0). A lost transmission is dropped
# with reason "channel". The same model applies to the last hop into the GS.
CHANNEL_LOSS_K: float = 0.4       # PROVISIONAL - to confirm

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
