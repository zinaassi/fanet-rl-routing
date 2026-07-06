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

RING_RADIUS_M: float = 250.0    # 'ring' layout: C-drone ring radius around GS
                                # NOTE: equal to COMM_RANGE_M, so ring C-drones sit
                                # exactly at the GS's range edge where p_loss = 1 —
                                # they have NO usable direct link to the GS and can
                                # only relay inward. Set below 250 (e.g. 125 m,
                                # where p_loss = 0.5) to give them a usable last hop.
GRID_ROW_SIZES: tuple[int, ...] = (4, 3, 4, 3)  # 'grid' layout: staggered lattice, 14 points

# --------------------------------------------------------------------------
# Channel: hard communication range + distance-normalised logistic loss
# --------------------------------------------------------------------------
COMM_RANGE_M: float = 250.0     # hard range: no link exists beyond this distance
K_SWEEP: tuple[float, ...] = (4.0, 8.0, 16.0)  # logistic steepness sweep
                                # (dimensionless, on d / COMM_RANGE_M; small k ~ linear
                                #  ramp 0 -> 1, large k ~ step at COMM_RANGE_M / 2)

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
