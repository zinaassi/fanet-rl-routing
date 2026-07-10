# Stage 1 — Classical Routing Baseline

A completely standalone mini-codebase (no imports from the rest of this
repository, no RL): **static drones, full global information, one-shot
routing computed at t=0, then a queued packet simulation to measure
delivery.** Dependencies: `numpy`, `networkx`, `matplotlib` (+ `pytest` for
the tests). Every random process is seedable and reproducible.

## How to run

From the **repo root**:

```bash
# test suite
python -m pytest stage1/tests -q

# RANGE/P_sens calibration sweep — run FIRST; prints a recommendation that a
# human applies by setting RANGE_M in stage1/config.py (never auto-applied)
python -m stage1.calibrate --jobs 16

# small smoke test (3 topologies x 2 channel realizations x 200 steps, full grid)
python -m stage1.evaluate --quick

# full run (50 topologies x 20 realizations x 1000 steps = 27,000 sims);
# results are bit-identical for any --jobs value
python -m stage1.evaluate --jobs 16

# slices of the grid
python -m stage1.evaluate --layouts ring grid --ks 0.1 --routers dijkstra greedy
python -m stage1.evaluate --n-topologies 10 --n-channels 5 --steps 500 --base-seed 7

# visualize the routing decisions for one topology (one panel per router:
# links, next-hop arrows, unreachable drones, PDR of one simulated episode)
python -m stage1.viz --layout random --k 0.1 --topology 1
python -m stage1.viz --layout ring --k 0.2 --topology 0 --routers greedy dijkstra
```

Outputs land in `stage1/out/`, one subfolder per tool (override with
`--out-dir`):

| Path | Contents |
|---|---|
| `evaluation/results_per_sim.csv` | one row per (layout, k, router, topology, realization) |
| `evaluation/results_summary.csv` | per-cell mean, between-topology std, within-topology std for each metric |
| `evaluation/drop_locations.csv` | drop histogram per cell, by node kind (M/C) and reason (channel / no_route) |
| `evaluation/*.png` | pdr, delay-decomposition, unreachable and channel-calibration figures for the evaluated grid |
| `calibration/calibration_sweep.csv` | per-candidate-range stats from `stage1.calibrate` |
| `calibration/sens_grid.csv`, `calibration/sensitivity_grid.png` | full (k, RANGE_M) sensitivity grid (validation study) |
| `calibration/ploss_curves_R450.png` | loss curves at the recommended operating point |
| `validation/k<k>_R<range>/` | full three-router runs at candidate (k, RANGE_M) points via `--range-m` (see its README) |
| `viz/routes_<layout>_k<k>_t<topology>.png` | per-topology routing map from `stage1.viz` |

A headline table (global PDR per layout/k, checking `direct <= greedy <=
dijkstra`) is printed at the end of every run. Prune-disconnection events
(see below) are logged at INFO level.

## World model

- Area 1750 x 1750 m; ground station (GS, node 50) fixed at the centre
  (875, 875). The GS is a **pure sink**: it never transmits and emits nothing.
- 36 mission drones (M-drones, nodes 0–35): positions uniform at random.
- 14 communication drones (C-drones, nodes 36–49), placed per `--layout`:
  - `ring` — evenly spaced on a circle of radius 250 m around the GS
    (deterministic; starts at angle 0). **Note:** the radius currently
    equals the hard communication range, so ring C-drones sit exactly at
    the GS's range edge — no link exists there (edges require
    `d < RANGE_M`), so they have *no direct link to the GS* and can only
    relay inward (`RING_RADIUS_M` in `config.py`; set it below `RANGE_M`
    to give them a usable last hop).
  - `grid` — a deterministic staggered lattice covering the area: rows of
    4/3/4/3 points at the row/cell centres (`config.GRID_ROW_SIZES`).
  - `random` — uniform at random, resampled per topology seed.
- Everything is **frozen** for the whole episode; all 50 drones relay
  (M-drones forward too).

Seeding: the topology seed drives two independent RNG streams (M positions,
C positions), so **M-drone positions are identical across layouts** for the
same topology index — layout comparisons see the same traffic geometry.

## Channel model (`stage1/channel.py`)

FSPL margin with an exponential loss curve on a hard range:

```
P_rx(d)   = P_tx + G_tx + G_rx - 20*log10(d) - 20*log10(f) + 147.55   [dBm]
P_sens    = P_rx(RANGE_M)        (derived: margin is 0 exactly at RANGE_M)
M(d)      = P_rx(d) - P_sens  =  20*log10(RANGE_M / d)                [dB]
p_loss(d) = exp(-k * M(d))       valid ONLY for 0 < d < RANGE_M
```

with `P_tx = 30 dBm`, `G_tx = G_rx = 2 dBi`, `f = 2.4 GHz`. Properties
(unit-tested): `p_loss -> 1` as `d -> RANGE_M` from below (M -> 0),
`p_loss -> 0` as `d -> 0` (M -> +inf), and p_loss is strictly increasing in
d over `(0, RANGE_M)`. Equivalently `p_loss = (d/RANGE_M)^(20k/ln 10)` — a
power law. Decay sweep `k ∈ {0.05, 0.1, 0.2}` (per dB of margin).

**Hard range, required by the formula (not redundant with it):** for
`d >= RANGE_M` the margin is `<= 0` and `exp(-k*M) >= 1`, which is not a
valid probability — `p_loss` is *undefined* there and is never evaluated
(it raises if asked). Edge existence is therefore defined purely by the
hard range:

```
edge (i -> j) exists  iff  d(i,j) < RANGE_M
```

This rule is what keeps the loss formula inside its valid domain. There is
no separate `P_LOSS_CUTOFF` constant any more. (One guard remains: a link
within floating-point dust of the boundary, delivery probability below
1e-12, is not created — it could never carry a packet.)

`P_sens` is not stored in config: it is derived from `RANGE_M` via
`channel.p_sens_dbm()` (for the historical 250 m assumption this
reproduces the original −54 dBm calibration).

## Choosing RANGE_M (`stage1/calibrate.py` — run FIRST)

`RANGE_M` is our own assumption, not given by the project spec: too small
and the network is too sparse to route on at all; too large and every
layout is fully connected and ~saturated, so neither router nor C-drone
layout comparisons can discriminate. `python -m stage1.calibrate` sweeps
candidate ranges (150–400 m), derives `P_sens` for each, runs the
global-information dijkstra router only (30 topologies × 10 channel seeds
per layout at `k = CAL_K`), and prints per range: mean PDR per layout, the
layout spread (max−min), and the fraction of severely disconnected
topologies (<10% of drones with any path to the GS). It recommends the
smallest range with mid-band PDR (~30–80%), spread > 5 points, and
severe-disconnection < 5% — with justification, but **never auto-applies
it**: a human confirms by setting `RANGE_M` in `stage1/config.py`.

**Once set, `RANGE_M` is frozen for the entire project** — Stage 2,
Stage 3, and Algorithm 2 must import it from `stage1/config.py` rather
than redefining it; re-tuning it per stage invalidates cross-stage
comparisons.

## Graph construction (`stage1/world.py`)

- Directed graph over 51 nodes; candidate edge `i -> j` iff
  `d_ij < RANGE_M` (the hard-range rule; GS never a source).
- **Prune**: each drone keeps only its `MAX_OUT_EDGES = 5` lowest-p_loss
  outgoing **drone-to-drone** edges (ties broken by neighbour id).
- **GS exemption (interpretation)**: the drone→GS edge is *exempt from the
  cap and always kept*, so GS in-degree is unbounded and the `direct`
  router is unaffected by pruning. This is our reading of "GS in-degree is
  unbounded and exempt" — flagged for advisor confirmation.
- Edge weight `w(i,j) = -log(1 - p_loss(i,j))`, so shortest paths maximise
  end-to-end delivery probability.
- Whenever the prune disconnects a drone that raw range alone would
  connect (a path to the GS exists in the range-only graph but not in the
  pruned graph), the event is logged and counted
  (`prune_disconnected_count` in the CSVs).

## Routers (`stage1/routing.py`)

All share the interface `next_hop(drone_id, graph) -> node | None`; routing
is computed once at t=0 on the pruned graph.

1. `dijkstra` — first hop of the minimum-weight path to the GS. Implemented
   as ONE Dijkstra from the GS on the reversed graph, memoised per graph
   object behind the per-drone interface. An assertion verifies the
   next-hop chains are loop-free (guaranteed by the shortest-path tree).
2. `greedy` — the out-neighbour geometrically closest to the GS; `None`
   unless some neighbour is *strictly* closer than the drone itself
   (drop-on-no-progress). The GS counts as distance 0, ties break by id.
3. `direct` — the GS if a direct edge exists, else `None`.

A drone is **unreachable** for a router if its next-hop *chain* does not
terminate at the GS (covers `None` hops, downstream dead ends, and cycles).
`unreachable_frac_m` is over the 36 M-drones (traffic sources);
`unreachable_frac_all` over all 50 drones.

## Simulator (`stage1/sim.py`)

- Discrete 100 ms steps; episode = 1000 steps (`--steps`).
- Offered load: each M-drone emits 1 packet every `EMIT_PERIOD_STEPS` steps
  (staggered by drone id; `--emit-period` on both CLIs). The spec default
  is 1 (36 pkt/step), which **saturates** the network — roughly half the
  drones develop unstable queues and the in-flight exclusion then biases
  router comparisons; period 3 (1/3 load) gives congestion-free
  comparisons. Pending advisor decision.
- Each step, in order:
  1. every M-drone due to emit appends 1 new packet to its own unbounded
     FIFO queue;
  2. every drone with a non-empty queue attempts to transmit exactly its
     head-of-queue packet: draw `r ~ U(0,1)`; if `r < p_loss(link)` the
     packet is **dropped permanently** (no retransmission), otherwise it is
     delivered (next hop = GS) or handed to the next hop's queue. A drone
     whose next hop is `None` instead drops its head packet
     (reason `no_route`) — one per step.
  3. hand-offs join the receiving queue at the **end of the step**, so a
     packet moves at most one hop per step and the result is independent of
     iteration order; same-step arrivals enqueue in sender-id order.
- Determinism: drones transmit in ascending id order and one uniform draw
  is consumed per queue-active drone per step, so the channel seed fully
  reproduces the episode.
- Delay accounting: a packet emitted in step `e` and resolved in step `x`
  spent `x - e + 1` steps in the network; for delivered packets this is
  exactly `hops + queue_wait` (each step is either the packet's one
  successful transmission on some hop, or a step waiting in a queue).
- Packets still in flight at episode end are **excluded from all PDR
  denominators**.

## Metrics (`stage1/metrics.py`)

- `pdr_global` — delivered / resolved, over all packets.
- `pdr_routed` — same, restricted to packets emitted by routed sources.
- Delay (delivered packets only), decomposed into transmission hops and
  queue wait; reported in steps and ms.
- Queue depth: max and mean over drones and steps.
- **Queue instability**: a drone is flagged if the least-squares slope of
  its queue depth over the last `INSTABILITY_WINDOW = 100` steps exceeds
  `INSTABILITY_SLOPE_MIN = 0.05` packets/step.
- Drop-location histogram by node kind (M/C) and reason
  (`channel` vs `no_route`), aggregated per cell in `drop_locations.csv`.
- Aggregation separates **between-topology** std (std over per-topology
  means) from **within-topology** std (mean over topologies of the
  per-topology std across channel realizations).

## Nested seeding

- Topology seed = `SeedSequence((base_seed, 0, topology_index))` — controls
  positions only.
- Channel seed = `SeedSequence((base_seed, 1, topology, realization,
  layout_idx, k_idx, router_idx))` — controls the U(0,1) draws only.
- `--base-seed` moves the whole experiment; results are independent of
  `--jobs`.

## Open questions (advisor to confirm; each is a one-line change in `stage1/config.py`)

| Knob | Current default |
|---|---|
| `RANGE_M` (hard link range; fixes `P_sens`) | 250 m — pending confirmation of the `stage1/calibrate.py` recommendation; frozen project-wide once set |
| `RING_RADIUS_M` | 250 m (= `RANGE_M`, so ring C-drones have no direct GS link — intended?) |
| `K_SWEEP` (loss-decay per dB of margin) | (0.05, 0.1, 0.2) |
| `MAX_OUT_EDGES` (per-drone out-edge cap) | 5 |
| GS-bound edges exempt from the cap | yes (see interpretation above) |
