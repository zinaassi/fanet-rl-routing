"""All Stage-1 constants in one place.

Values marked "pending advisor confirmation" are deliberate one-line knobs:
change them here, nothing else in the codebase hardcodes them.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Arena / fleet
# --------------------------------------------------------------------------
AREA_SIZE_M: float = 1750.0
GS_POS: tuple[float, float] = (AREA_SIZE_M / 2.0, AREA_SIZE_M / 2.0)  # (875, 875)

N_M_DRONES: int = 36            # mission drones (packet sources, also relay)
N_C_DRONES: int = 14            # communication drones (relay only)
N_DRONES: int = N_M_DRONES + N_C_DRONES
GS_ID: int = N_DRONES           # ground station node id (= 50); pure sink

RING_RADIUS_M: float = 450.0    # 'ring' layout: C-drone ring radius around GS
GRID_ROW_SIZES: tuple[int, ...] = (4, 3, 4, 3)  # 'grid' layout: staggered lattice, 14 points

# --------------------------------------------------------------------------
# Channel (pending advisor confirmation: P_SENS_DBM at 2.4 GHz)
# --------------------------------------------------------------------------
P_TX_DBM: float = 30.0
G_TX_DBI: float = 2.0
G_RX_DBI: float = 2.0
FREQ_HZ: float = 2.4e9
P_SENS_DBM: float = -54.0       # calibrated so margin M(d)=0 at ~250 m
K_SWEEP: tuple[float, ...] = (0.3, 0.6, 1.0)   # logistic steepness sweep

# Link-existence cutoff (pending advisor confirmation)
P_LOSS_CUTOFF: float = 0.95     # edge i->j exists iff p_loss(d_ij) <= P_LOSS_CUTOFF

# --------------------------------------------------------------------------
# Graph pruning (pending advisor confirmation: cap=5, GS-bound edges uncapped)
# --------------------------------------------------------------------------
MAX_OUT_EDGES: int = 5          # per-drone cap on drone->drone outgoing edges
                                # (the drone->GS edge is exempt and always kept,
                                #  so GS in-degree is unbounded)

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
STEP_MS: float = 100.0          # one discrete step = 100 ms
N_STEPS: int = 1000             # episode length in steps

INSTABILITY_WINDOW: int = 100   # look at the last N steps of queue depth
INSTABILITY_SLOPE_MIN: float = 0.05  # packets/step; queue is "unstable" above this

# --------------------------------------------------------------------------
# Evaluation grid
# --------------------------------------------------------------------------
N_TOPOLOGIES: int = 50
N_CHANNEL_REALIZATIONS: int = 20
LAYOUTS: tuple[str, ...] = ("ring", "grid", "random")
ROUTERS: tuple[str, ...] = ("direct", "greedy", "dijkstra")

BASE_SEED: int = 20260706       # root of all seeding; override with --base-seed
OUT_DIR: str = "stage1/out"

# Quick smoke-test overrides (used by `python -m stage1.evaluate --quick`)
QUICK_N_TOPOLOGIES: int = 3
QUICK_N_CHANNEL_REALIZATIONS: int = 2
QUICK_N_STEPS: int = 200
