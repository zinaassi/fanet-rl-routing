# Stage 1 — Change log

## 2026-07-06 — Hard 250 m communication range + rescaled loss curve

Goal: links only exist between nodes within 250 m of each other; packets can
never be routed through drones outside that radius. The logistic loss curve
now maps distance 0 → 250 m onto p_loss 0 → 1, replacing the old
Friis/margin model and its `P_LOSS_CUTOFF` link rule.

### `config.py`
- **Added** `COMM_RANGE_M = 250.0` — the hard link range.
- **Removed** the RF constants `P_TX_DBM`, `G_TX_DBI`, `G_RX_DBI`, `FREQ_HZ`,
  `P_SENS_DBM`, and the link rule `P_LOSS_CUTOFF = 0.95` (the hard range
  replaces it; see git history to recover the old model).
- **Changed** `RING_RADIUS_M`: 450 m → **250 m**. ⚠️ This equals
  `COMM_RANGE_M`, so ring C-drones sit exactly at the GS's range edge where
  `p_loss = 1`: they get **no direct GS link** and can only relay inward. In
  practice the ring layout delivers ~nothing unless M-drones happen to fall
  inside the ring. Set `RING_RADIUS_M` below 250 (e.g. 125 m, where
  `p_loss = 0.5`) if the C-ring is supposed to be a usable last hop.
- **Changed** `K_SWEEP`: (0.3, 0.6, 1.0) per-dB → **(4, 8, 16)**
  dimensionless (steepness on normalised distance `d / COMM_RANGE_M`).

### `channel.py` (rewritten)
- `p_loss(d, k)` is now a logistic in normalised distance, **rescaled so the
  endpoints are pinned for every k**: `p_loss(0) = 0`,
  `p_loss(125 m) = 0.5`, `p_loss(250 m) = 1`, saturating at 1 beyond range.
  k only shapes the curve inside the range (k → 0 ≈ linear ramp, large k ≈
  step at 125 m) and never changes the communication range.
- Removed `received_power_dbm`, `margin_db`, `max_link_range_m` (the range
  is now the constant `COMM_RANGE_M`, identical for all k).

### `world.py`
- Link-existence rule in `candidate_graph`: was `p_loss <= P_LOSS_CUTOFF`
  (range 350–773 m depending on k), now **`d <= COMM_RANGE_M` and the link
  can actually deliver (`1 - p_loss >= 1e-12`)**. The delivery-probability
  floor drops boundary links at d ≈ 250 m (which could never carry a packet)
  and makes edge existence immune to floating-point dust — relevant because
  the ring C-drones sit exactly on the boundary.
- Everything else (per-drone out-edge cap of 5, GS exemption, edge weight
  `-log(1 - p_loss)`, prune-disconnect logging) is unchanged.

### `plots.py`
- Calibration figure redrawn for the new model: p_loss over 0–300 m, hard
  range line at 250 m, midpoint marker at 125 m.

### `viz.py` (new)
- `python -m stage1.viz --layout random --k 8 --topology 1` renders one
  panel per router: all in-range links, each drone's chosen next-hop arrow,
  the 250 m circle around the GS, unreachable drones (hollow red), and the
  PDR of one simulated episode (seeded identically to realization 0 of
  `stage1.evaluate`). A per-router summary (PDR, drop reasons, unreachable
  fraction, mean delay) is printed to stdout.

### Tests
- `test_channel.py` rewritten for the new curve (pinned endpoints, midpoint,
  monotonicity inside the range, saturation beyond it, symmetry).
- `test_world.py` / `test_routing.py` / `test_sim.py` geometries updated:
  scenarios that relied on 350–773 m links were rebuilt at ≤ 250 m scale;
  prune tests use a dense synthetic cluster (the real graphs are now too
  sparse for the 5-edge cap to bind). New tests: no link at exactly
  `COMM_RANGE_M`; ring C-drones have no direct GS edge.

### Observed impact (quick run, 2026-07-06)
- The 1750 m arena is now mostly disconnected: with 50 static drones and a
  250 m range, global PDR collapses to ~0.5% and 30–36 of 36 M-drones are
  typically unreachable. The `direct <= greedy <= dijkstra` ordering still
  holds on average but is within seed noise at this PDR level. If Stage 1 is
  meant to show meaningful routing differences, consider more drones, a
  smaller arena, or C-drone layouts that form relay chains toward the GS.
