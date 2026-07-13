"""All Stage-1 constants in one place.

Values marked "pending confirmation" are deliberate one-line knobs:
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

RING_RADIUS_M: float = 320.0    # 'ring' layout: C-drone ring radius around GS
                                # NOTE: kept equal to RANGE_M (re-synced 2026-07-07
                                # after RANGE_M was frozen at 320), so ring C-drones
                                # sit exactly at the GS's range edge — no direct link
                                # to the GS exists there (edges require d < RANGE_M)
                                # and they can only relay inward. Set below RANGE_M
                                # to give them a usable last hop.
GRID_ROW_SIZES: tuple[int, ...] = (4, 3, 4, 3)  # 'grid' layout: staggered lattice, 14 points

# --------------------------------------------------------------------------
# Channel: FSPL margin + exponential packet loss on a hard range
# --------------------------------------------------------------------------
P_TX_DBM: float = 30.0
G_TX_DBI: float = 2.0
G_RX_DBI: float = 2.0
FREQ_HZ: float = 2.4e9

# Hard communication range: an edge i->j exists iff d(i,j) < RANGE_M, and
# p_loss(d) = exp(-k * M(d)) is only defined on that domain (beyond it the
# margin is negative and exp(-k*M) > 1 is not a probability). P_sens is
# DERIVED from RANGE_M (see channel.p_sens_dbm): M = 0 lands exactly at
# RANGE_M, so p_loss -> 1 at the range edge.
#
# RANGE_M is fixed for the entire project after calibration in
# stage1/calibrate.py — do not re-tune per stage, or cross-stage
# comparisons become invalid. FROZEN 2026-07-07 at 320 m.
#
# Justification: RANGE_M=320 was initially selected via a PDR-gate
# calibration sweep and RETAINED after an independent, algorithm-free check:
# it sits at ~1.16x the Gupta-Kumar critical connectivity radius
# (r_c ≈ 276 m for 51 nodes in 1750x1750 m), giving ~79% probability of a
# fully connected random topology. The primary comparison metric
# (restricted_pdr, see metrics.py) conditions on graph-reachability, so
# residual disconnection cannot bias the router comparison.
RANGE_M: float = 320.0

# K_SWEEP is FROZEN to the single confirmed operating point (was a 3-value
# sweep during calibration). k=0.4 was originally chosen by inspecting
# Dijkstra-only layout-spread plots — a circular provenance (the metric that
# picked k was itself Dijkstra-based). That provenance is not re-derived;
# instead robustness evidence at a second k value is maintained via
# stage1/compare.py runs (see stage1/out/compare/). Kept as a tuple so
# evaluate.py/plots.py, which iterate over it, need no changes.
K_SWEEP: tuple[float, ...] = (0.4,)

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

EMIT_PERIOD_STEPS: int = 1      # each M-drone emits 1 packet every N steps,
                                # staggered by drone id. 1 = the spec load
                                # (36 pkt/step), which SATURATES the network
                                # (~half the drones develop unstable queues and
                                # in-flight censoring distorts PDR comparisons);
                                # 3 = 1/3 load, congestion-free routing
                                # comparisons. Pending advisor decision.

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

# Output layout: one subfolder per tool so results don't overwrite each other.
OUT_DIR: str = "stage1/out"
OUT_DIR_EVALUATION: str = OUT_DIR + "/evaluation"    # stage1.evaluate
OUT_DIR_CALIBRATION: str = OUT_DIR + "/calibration"  # stage1.calibrate + channel studies
OUT_DIR_VALIDATION: str = OUT_DIR + "/validation"    # full runs at candidate (k, RANGE_M)
OUT_DIR_VIZ: str = OUT_DIR + "/viz"                  # stage1.viz route maps

# Quick smoke-test overrides (used by `python -m stage1.evaluate --quick`)
QUICK_N_TOPOLOGIES: int = 3
QUICK_N_CHANNEL_REALIZATIONS: int = 2
QUICK_N_STEPS: int = 200

# --------------------------------------------------------------------------
# Range calibration sweep (stage1/calibrate.py — run BEFORE any evaluation;
# its recommendation is applied by a human setting RANGE_M above)
# --------------------------------------------------------------------------
CAL_RANGES_M: tuple[float, ...] = (150.0, 200.0, 250.0, 300.0, 350.0, 400.0)
CAL_N_TOPOLOGIES: int = 30
CAL_N_CHANNEL_REALIZATIONS: int = 10
CAL_K: float = 0.1              # default k for calibrate.py's OWN exploratory
                                # sweep (independent of the frozen K_SWEEP
                                # above); override per run with --k

# --------------------------------------------------------------------------
# Router comparison protocol (stage1/compare.py — greedy vs dijkstra)
# --------------------------------------------------------------------------
COMPARE_N_TOPOLOGIES: int = 1000
COMPARE_N_CHANNEL_REALIZATIONS: int = 20
PRACTICAL_SIGNIFICANCE_THRESHOLD: float = 0.10
                                # default carried from planning; NOT yet
                                # confirmed by the team — confirm before citing
                                # in any writeup.
