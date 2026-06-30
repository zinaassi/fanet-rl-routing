"""
config.py — All simulation parameters in one place.

Every constant used by the simulator lives here. Change values here and
re-run; no other file needs to be touched.

----------------------------------------------------------------------------
Setup anchored to IQMR (Sharvari et al., 2024, arXiv:2408.09109) §V.A
----------------------------------------------------------------------------
IQMR uses a 3-D cylinder of radius 1000 m with 100–300 m altitude variation;
UAV speeds 10–30 m/s; transmit power 1 W (= 30 dBm); radio range 250 m at
their threshold. We anchor to that as closely as possible for literature
comparability.

Documented deviations from IQMR:
  (a) 2-D instead of 3-D. We simplify the 1000 m-radius cylinder
      (horizontal area pi*1000^2 ~= 3.14 km^2) to a 1750 x 1750 m
      square (3.0625 km^2), which matches IQMR's disk area closely while
      keeping cartesian coordinates simple.
  (b) 50 drones split into 36 M + 14 C. IQMR has no M/C distinction; the
      split here is our own (~72%/28%) and is not from the paper. Fleet
      size sits comfortably inside IQMR's reported scenario range.

Anchored to IQMR: speed range (10-30 m/s), transmit power (1 W = 30 dBm),
2.4 GHz carrier (channel.py), the energy budget (11.1 V x 5200 mAh =
207792 J), AND the radio range (250 m). The receiver sensitivity
(-54 dBm in channel.py) is derived — not picked — so that the FSPL test
reproduces IQMR's reported 250 m range at IQMR's 1 W transmit power.
That is, both the source of power and the resulting range are now
IQMR-faithful, not just the power alone.
"""

# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------
AREA_SIZE: float = 1000.0       # radius in meters (3-D sphere), used in 3-D mode
USE_3D: bool = False             # start False; extend to 3-D later
WIDTH: float = 1750.0            # meters (2-D mode) — 1750x1750 m square (~3.06 km^2)
HEIGHT: float = 1750.0           # closely matches IQMR's 1000 m-radius disk area (~3.14 km^2)

# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------
NUM_M_DRONES: int = 36           # mission drones   (fleet total = 50, our M/C split)
NUM_C_DRONES: int = 14           # communication drones
DRONE_SPEED_MIN: float = 10.0    # m/s — matches IQMR
DRONE_SPEED_MAX: float = 30.0    # m/s — matches IQMR
TX_POWER: float = 1.0            # watts — matches IQMR (1 W = 30 dBm); the FSPL
                                 # channel uses PT_DBM in channel.py as source of truth.

# COMM_RANGE is no longer a tunable. The FSPL channel model in channel.py
# decides link existence from received signal power. This alias is kept for
# any code that wants a single-number effective range (e.g. the visualiser).
# To change the radio range, edit the parameters at the top of channel.py.
from fanet_sim.envs.channel import MAX_LINK_DISTANCE_M as _MAX_LINK_DISTANCE_M
COMM_RANGE: float = _MAX_LINK_DISTANCE_M

# ---------------------------------------------------------------------------
# K-link selection policy
# ---------------------------------------------------------------------------
# Instead of passively keeping every reachable drone as a neighbour, each drone
# actively keeps only its top-K candidate links. Candidates are ranked by a
# per-drone PyTorch MLP (see fanet_sim/envs/policies.py:LinkScorePolicy):
#   Input(5) -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> Linear(1)
# The 5 input features per candidate are link quality, distance to the
# candidate, the candidate's distance to GS, the candidate's queue length, and
# the candidate's number of current links. Weights are randomly initialised —
# there is no training yet. K_LINKS is the default value; a learned policy may
# later pick a per-drone value anywhere in [K_LINKS_MIN, K_LINKS_MAX].
K_LINKS: int = 3                 # default active links kept per drone
K_LINKS_MIN: int = 2             # policy lower bound on K
K_LINKS_MAX: int = 5             # policy upper bound on K

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
PACKET_RATE: int = 1              # packets generated per M-drone per timestep
PACKET_SIZE: int = 512            # bytes
MAX_HOPS: int = 10                # drop packet if hop count exceeds this
PACKET_TTL: int = 50              # timesteps before packet expires

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
TIMESTEP: float = 0.1             # seconds per simulation step
MAX_STEPS: int = 1000             # steps per episode
NUM_EPISODES: int = 1             # number of episodes (for testing)

# ---------------------------------------------------------------------------
# Ground station
# ---------------------------------------------------------------------------
GS_POSITION: tuple = (875.0, 875.0)   # centre of the 1750 x 1750 m 2-D area

# ---------------------------------------------------------------------------
# M-drone mobility (straight line: start -> end, then stop)
# ---------------------------------------------------------------------------
# M-drones no longer roam on random cyclic waypoints. Each M-drone is given a
# random start point (its initial position) and a random end point at episode
# start; it flies in a straight line from start to end and then stops.
M_DRONE_MOBILITY: str = "straight_line"
WAYPOINT_ARRIVAL_THRESHOLD: float = 2.0  # metres — drone is "at" its end point when this close

# ---------------------------------------------------------------------------
# C-drone topology policy (untrained PyTorch MLP; RL trains it later)
# ---------------------------------------------------------------------------
#  Each C-drone owns a PyTorch MLP
# (see fanet_sim/envs/policies.py:TopologyPolicy) that maps its 8-feature local
# state to a movement vector [dx, dy]:
#   Input(8) -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(2)
# The 8 input features are pos_x, pos_y, vel_x, vel_y, num_neighbors,
# mean_link_quality, mean_distance_to_neighbours, and queue_length. Weights are
# randomly initialised — there is no training yet. The output is clamped per
# axis to TOPOLOGY_MAX_STEP_M metres before it is applied.
TOPOLOGY_MAX_STEP_M: float = DRONE_SPEED_MAX * TIMESTEP   # max metres per axis per step

# Local reward for the topology policy — RELAY COVERAGE objective. C-drones home
# toward MISSION (M) drones so they sit where they can relay mission traffic,
# instead of freezing in place or clumping with each other:
#   coverage      — + reward per M-drone currently within radio range (integer)
#   progress      — + metres of distance REDUCED toward the nearest M-drone this
#                   step, scaled by the max step size (≈+1 for a full-speed
#                   approach). This is the dense per-step gradient that overcomes
#                   the arena-size vs 3 m/step scale problem and drives movement.
#   motion_energy — - small penalty per joule of motion energy spent moving
# (Only M-drones count, so C-drones don't reward each other for clustering.)
TOPOLOGY_REWARD_WEIGHTS: dict = {
    "coverage": 1.0,
    "progress": 3.0,
    "motion_energy": 0.2,
}

# ---------------------------------------------------------------------------
# Random seed (None = non-deterministic)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Event log (Stage 1 output — consumed by scripts/analyze.py in Stage 2)
# ---------------------------------------------------------------------------
LOG_DIR: str = "logs"             # directory for per-episode JSONL files
LOG_DRONE_STATE_EVERY_STEP: bool = True   # if False, only one snapshot at episode end

# ---------------------------------------------------------------------------
# PPO training (used by train.py only — the phase-1 simulator never trains)
# ---------------------------------------------------------------------------
# Two policies are trained, each drone/​C-drone owning its own weights:
#   * the K-link selection MLP (LinkScorePolicy) on every drone, and
#   * the topology movement MLP (TopologyPolicy) on every C-drone.
# Both are optimised with PPO (clipped surrogate + GAE). These constants are the
# shared hyper-parameters; train.py exposes --episodes / --steps overrides.
TRAIN_EPISODES: int = 60          # number of training episodes
TRAIN_MAX_STEPS: int = 300        # steps per training episode (overrides MAX_STEPS)

PPO_LR: float = 3e-4              # Adam learning rate (per policy)
PPO_GAMMA: float = 0.99          # reward discount factor
PPO_LAMBDA: float = 0.95         # GAE smoothing parameter
PPO_CLIP: float = 0.2            # PPO clipped-surrogate epsilon
PPO_EPOCHS: int = 4              # optimisation passes per policy per episode
PPO_VALUE_COEF: float = 0.5      # weight of the critic (value) loss
PPO_ENTROPY_COEF: float = 0.01   # weight of the entropy bonus (exploration)

# K-link reward: +1 when a packet that ORIGINATED at this drone is delivered to
# the GS, -1 when one of its packets is dropped. (Only M-drones generate
# packets, so only they receive a non-zero link reward.)
LINK_REWARD_DELIVERED: float = 1.0
LINK_REWARD_DROPPED: float = -1.0

# Where train.py writes the trained weights (one checkpoint for all policies).
POLICY_SAVE_PATH: str = "trained_policies.pt"
