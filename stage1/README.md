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

# small smoke test (3 topologies x 2 channel realizations x 200 steps, full grid)
python -m stage1.evaluate --quick

# full run (50 topologies x 20 realizations x 1000 steps = 27,000 sims);
# results are bit-identical for any --jobs value
python -m stage1.evaluate --jobs 16

# slices of the grid
python -m stage1.evaluate --layouts ring grid --ks 8 --routers dijkstra greedy
python -m stage1.evaluate --n-topologies 10 --n-channels 5 --steps 500 --base-seed 7

# visualize the routing decisions for one topology (one panel per router:
# links, next-hop arrows, unreachable drones, PDR of one simulated episode)
python -m stage1.viz --layout random --k 8 --topology 1
python -m stage1.viz --layout ring --k 16 --topology 0 --routers greedy dijkstra
```

Outputs land in `stage1/out/` (override with `--out-dir`):

| File | Contents |
|---|---|
| `results_per_sim.csv` | one row per (layout, k, router, topology, realization) |
| `results_summary.csv` | per-cell mean, between-topology std, within-topology std for each metric |
| `drop_locations.csv` | drop histogram per cell, by node kind (M/C) and reason (channel / no_route) |
| `calibration.png` | p_loss vs distance for each k, with the hard 250 m range |
| `pdr.png`, `delay_decomposition.png`, `unreachable.png` | summary figures over the grid |
| `routes_<layout>_k<k>_t<topology>.png` | per-topology routing map from `stage1.viz` |

A headline table (global PDR per layout/k, checking `direct <= greedy <=
dijkstra`) is printed at the end of every run. Prune-disconnection events
(see below) are logged at INFO level.

## World model

- Area 1750 x 1750 m; ground station (GS, node 50) fixed at the centre
  (875, 875). The GS is a **pure sink**: it never transmits and emits nothing.
- 36 mission drones (M-drones, nodes 0–35): positions uniform at random.
- 14 communication drones (C-drones, nodes 36–49), placed per `--layout`:
  - `ring` — evenly spaced on a circle of radius 250 m around the GS
    (deterministic; starts at angle 0). **Note:** the radius equals the
    hard communication range, so ring C-drones sit exactly at the GS's
    range edge where `p_loss = 1` — they have *no usable direct link to
    the GS* and can only relay inward (`RING_RADIUS_M` in `config.py`;
    set it below 250 m, e.g. 125 m where `p_loss = 0.5`, to give them a
    usable last hop).
  - `grid` — a deterministic staggered lattice covering the area: rows of
    4/3/4/3 points at the row/cell centres (`config.GRID_ROW_SIZES`).
  - `random` — uniform at random, resampled per topology seed.
- Everything is **frozen** for the whole episode; all 50 drones relay
  (M-drones forward too).

Seeding: the topology seed drives two independent RNG streams (M positions,
C positions), so **M-drone positions are identical across layouts** for the
same topology index — layout comparisons see the same traffic geometry.

## Channel model (`stage1/channel.py`)

Hard communication range `COMM_RANGE_M = 250 m` plus a distance-normalised
logistic loss curve:

```
u(d)      = clip(d / COMM_RANGE_M, 0, 1)
sigma(x)  = 1 / (1 + exp(-x))
p_loss(d) = (sigma(k*(u - 1/2)) - sigma(-k/2)) / (sigma(k/2) - sigma(-k/2))
```

The rescaling pins the endpoints for **every** steepness k:
`p_loss(0) = 0`, `p_loss(125 m) = 0.5`, `p_loss(250 m) = 1`, and the curve
saturates at 1 beyond the range. k is a pure shape knob — it never changes
the communication range: small k approaches a linear ramp 0 → 1, large k
approaches a step at 125 m. Steepness sweep `k ∈ {4, 8, 16}`.

Link existence: an edge exists iff `d <= COMM_RANGE_M` **and** the link can
actually deliver (`p_loss < 1`): drones farther apart than 250 m have no
link at all, and packets can never be routed through them.

## Graph construction (`stage1/world.py`)

- Directed graph over 51 nodes; candidate edge `i -> j` iff
  `d_ij <= COMM_RANGE_M` and `p_loss < 1` (GS never a source).
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
- Each step, in order:
  1. every M-drone appends 1 new packet to its own unbounded FIFO queue;
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
| `COMM_RANGE_M` (hard link range) | 250 m |
| `RING_RADIUS_M` | 250 m (= `COMM_RANGE_M`, so ring C-drones have no direct GS link — intended?) |
| `K_SWEEP` (loss-curve steepness) | (4, 8, 16) |
| `MAX_OUT_EDGES` (per-drone out-edge cap) | 5 |
| GS-bound edges exempt from the cap | yes (see interpretation above) |
