"""
config.py — All simulation parameters in one place.

Every constant used by the simulator lives here. Change values here and
re-run; no other file needs to be touched.
"""

# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------
AREA_SIZE: float = 1000.0       # radius in meters (3-D sphere), used in 3-D mode
USE_3D: bool = False             # start False; extend to 3-D later
WIDTH: float = 1000.0            # meters (2-D mode)
HEIGHT: float = 1000.0           # meters (2-D mode)

# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------
NUM_M_DRONES: int = 6            # mission drones
NUM_C_DRONES: int = 3            # communication drones
DRONE_SPEED_MIN: float = 10.0    # m/s
DRONE_SPEED_MAX: float = 30.0    # m/s
COMM_RANGE: float = 250.0        # meters — two drones can communicate if closer than this
TX_POWER: float = 1.0            # watts (uniform across all drones)

# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------
INITIAL_ENERGY: float = 207792.0  # joules  (11.1 V × 5200 mAh, from IQMR)
ENERGY_PER_MOVE: float = 0.1      # joules per meter flown (approximate)
ENERGY_PER_TX: float = 0.01       # joules per packet transmitted

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
GS_POSITION: tuple = (500.0, 500.0)   # centre of the area in 2-D mode

# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------
NUM_WAYPOINTS: int = 5            # waypoints generated per drone at episode start
WAYPOINT_ARRIVAL_THRESHOLD: float = 2.0  # metres — drone is "at" a waypoint when this close

# ---------------------------------------------------------------------------
# Random seed (None = non-deterministic)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
