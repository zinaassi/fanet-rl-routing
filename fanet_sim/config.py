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
# Waypoints
# ---------------------------------------------------------------------------
NUM_WAYPOINTS: int = 5            # waypoints generated per drone at episode start
WAYPOINT_ARRIVAL_THRESHOLD: float = 2.0  # metres — drone is "at" a waypoint when this close

# ---------------------------------------------------------------------------
# Random seed (None = non-deterministic)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Event log (Stage 1 output — consumed by scripts/analyze.py in Stage 2)
# ---------------------------------------------------------------------------
LOG_DIR: str = "logs"             # directory for per-episode JSONL files
LOG_DRONE_STATE_EVERY_STEP: bool = True   # if False, only one snapshot at episode end
