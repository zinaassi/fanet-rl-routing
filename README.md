# FANET drone-network simulator — Phase 1a

A simulator for a flying ad-hoc network (FANET): a fleet of drones relays data
packets back to a single ground station (GS) over multi-hop wireless links.
Student project by Abed and Zena, supervised by Eran, with Prof. Reuven Cohen
involved.

**Goal: minimize total packet loss** — the share of packets created that never
reach the GS. Per-link loss and the breakdown by cause (channel, queue_full,
no_route, ttl, hop_limit) are secondary; they exist to explain the total.

The plan, in the order set by the supervisor, each step starting only after the
previous one is approved:

| Phase | Step | What |
|---|---|---|
| 1 — static world | **1a** | Non-learned baselines: GREEDY and RANDOM routing. **← you are here** |
| | 1b | RL routing agent on every drone. |
| 2 — dynamic world | 2a–2d | M-drones fly; C-drone movement rules, then RL routing and topology agents. |

**Only Phase 1a exists in this repository.** There is no learning of any kind in
the code, and none of the archived RL code is reused.

---

## 1. What the simulator models

### World

A 2-D square 900 × 900 m with the ground station at its centre, (450, 450).
Twenty-five drones are placed uniformly at random and **never move**:

- **18 M-drones** (mission) create packets *and* relay other drones' packets.
- **7 C-drones** (communication) create nothing; they only relay.

One simulation step is 100 ms, and a run is 1000 steps.

### Radio and channel

Link existence is decided by a free-space path loss (FSPL) model: two endpoints
can hear each other when the received power clears the receiver sensitivity of
−54 dBm. With 30 dBm transmit power, 2 dBi antennas and a 2.4 GHz carrier, that
works out to an effective range of **249.69 m**. The GS uses the same radio as
the drones and is treated as just another endpoint.

A link existing does not mean a transmission succeeds. Every send draws against

```
channel_loss = exp(-k · M)      M = received power − sensitivity, in dB
```

so loss is 1 exactly at the range edge (M = 0) and falls away fast as endpoints
close in — with k = 0.4 it is about 0.46 at 200 m, 0.042 at 100 m and 1.4e-5 at
10 m. `k` is a modelling choice, not a measurement. Run
`scripts/plot_ploss.py` to see the curve.

### Link queues

Each drone keeps **one FIFO queue per outgoing link** — one per neighbour, plus
one for the GS link when the GS is in range. Each queue holds at most **10**
packets and releases at most **1** packet per step, oldest first.

All of a drone's link queues send in the same step. There is no collision model,
so there is no per-drone send limit: a drone with five links may send five
packets in one step. Receiving is likewise unlimited — a drone accepts every
packet that reaches it.

A packet placed into a full link queue is dropped at that drone, cause
`queue_full`.

### Routing happens at enqueue time

When a packet is **created** at a drone or **arrives** at one, that drone picks
the next hop immediately and puts the packet in that link's queue. The choice is
never revisited: a packet waiting on link X→Y is sent to Y. A packet that
arrives during a step is routed at once but cannot be sent before the next step.

Arrivals within a step are processed in a **random order**, drawn from the run's
channel/traffic stream. Processing by drone id would hand the same low-id drones
the last free slot in a nearly full queue every step. The order is reproducible
for a given run seed, and identical for both routing rules.

Both rules use local information only. The GS counts as a neighbour whenever it
is in range, and is not automatically preferred.

**GREEDY.** Consider only neighbours **strictly closer** to the GS than the
current holder. Among those, minimise

```
link_loss    = 1 − (1 − channel_loss) · (1 − queue_full)
channel_loss = exp(-k · M) for that link's distance
queue_full   = 1 if the holder's OWN queue for that link is at capacity, else 0
```

Ties break on the lower `channel_loss`, then on the candidate closer to the GS.
Both inputs are things the drone knows by itself: the channel figure from the
signal margin it could measure, the queue figure exactly, because the queue is
its own. So greedy routes around its own congestion when another qualifying link
still has room. If no neighbour is strictly closer, the packet is dropped
`no_route`.

**RANDOM.** Uniform over all current neighbours, GS included when in range. No
progress condition, no regard for queues. It draws from its own random stream,
so switching rules never shifts the channel or traffic draws.

### Ground station

The GS receives without limit: no queue, no per-step cap. The last hop into it
runs the same channel draw as any other link.

### ACKs and per-link counters

ACKs are ideal in Phase 1a — instant, never lost, and they consume no capacity.

- **End-to-end:** every packet the GS receives produces an ACK to its source.
- **Hop-by-hop:** after each send the sender learns `ok` or `channel`. There is
  no third outcome, because the receiver never refuses a packet.

Each drone keeps counters built **only** from local knowledge and ACKs, never
from the simulator's global view. Per outgoing link X→Y:

| Counter | Meaning |
|---|---|
| `offered` | packets this drone tried to put in that queue — the denominator of the link's loss |
| `dropped_queue_full` | refused because the queue was full |
| `sent` | transmissions attempted |
| `acked` | transmissions the next hop confirmed |
| `lost_channel` | transmissions lost in the channel |
| `expired_in_queue` | packets that died of TTL or hop limit while waiting |

Each M-drone also tracks packets it created, packets the GS acknowledged, and —
via a **source-side timer**, knowing its own creation times and the TTL —
packets whose TTL ran out unacknowledged.

Every report gives both views, and **they must match exactly**. Any gap is a
bug; `scripts/metrics_1a.py` checks it on every run and the experiment stops if
one appears.

### Packet lifetime

A packet lives 50 steps (TTL) and may take at most 10 hops. A packet that
arrives at a drone having already used up either budget never enters a queue —
it would occupy a slot it could never leave, inflating `queue_full`. It is
retired on arrival and counted in `expired_on_arrival`, which keeps

```
ttl + hop_limit = expired_in_queue + expired_on_arrival
```

true across a run.

### Step order

1. Move drones — skipped entirely, since Phase 1a is static — then refresh
   neighbour sets.
2. Retire packets that died waiting in a link queue; run each source's ACK
   timer. First, so a dead packet holds neither a queue slot nor a send slot.
3. Create this step's packets at the M-drones and route each into a link queue.
4. Transmit: snapshot the head of every link queue, then run each channel draw.
   Lost → `channel`. Aimed at the GS → delivered and acknowledged. Aimed at a
   drone → arrives there.
5. Route the arrivals into the receivers' link queues, in random order.
6. Charge radio energy.

Because the transmission snapshot is taken before arrivals are routed, a packet
cannot be received and re-sent in the same step. A packet created in stage 3
*can* be sent in stage 4 of the same step.

### Traffic and the measurement window

Each M-drone creates one packet every 1000, 500, 200 or 100 ms (10, 5, 2 or 1
steps). Each gets a random start offset inside its interval, drawn once at
reset, so they do not all create on the same step.

Metrics count only packets **created in steps [50, 950)** of the 1000. The
warm-up cut of 50 steps lets queues fill before anything is measured; the drain
cut of 50 steps — one full TTL — means every measured packet has finished.
Per-link and queue figures are instantaneous per-step quantities, so they use
the warm-up cut only; a transmission does not need time to finish the way a
packet does. The two views are only ever compared like with like.

### Seeds

Two independent seeds, so one layout can be replayed under different randomness:

- `placement_seed` — drone positions and speeds.
- `run_seed` — channel-loss draws, traffic offsets, and arrival order.

The random routing rule draws from a **third** stream spawned off `run_seed`, so
its choices never shift the channel or traffic draws. Greedy and random
therefore run on identical layouts with identical packet schedules, and differ
only in their routing decisions.

---

## 2. Install and run

```bash
pip install -r requirements.txt     # numpy, matplotlib, networkx, pytest
```

All commands below run from the `rl_sim/` directory:

```bash
cd rl_sim
```

**Tests** — 115 of them, a few seconds:

```bash
python -m pytest tests/ -q
```

**A single run**, printing both views of the metrics:

```bash
python main.py --no-anim
python main.py --no-anim --routing random --load-ms 200
python main.py --no-anim --routing greedy --load-ms 100 --placement-seed 7 --run-seed 7
```

Without `--no-anim`, `main.py` renders a matplotlib animation to `episode.gif`;
`--show` opens it interactively instead. `--steps` overrides the run length and
`--log` the event-log path.

**The Phase-1a experiment** — 2 rules × 4 loads × 20 placements = 160 runs, a
few minutes:

```bash
python scripts/experiment_1a.py
python scripts/experiment_1a.py --placements 5      # quicker smoke run
```

**The range check** — pure geometry over 100 layouts; changes no config:

```bash
python scripts/range_check.py
```

**The channel-loss curve**:

```bash
python scripts/plot_ploss.py
```

---

## 3. Configuration

Everything lives in [`rl_sim/fanet_sim/config.py`](rl_sim/fanet_sim/config.py),
except the radio parameters, which are at the top of
[`rl_sim/fanet_sim/envs/channel.py`](rl_sim/fanet_sim/envs/channel.py).
**DECIDED** values are fixed for Phase 1a; **PROVISIONAL** ones are temporary
defaults and carry a `PROVISIONAL - to confirm` comment in the file.

| Parameter | Value | Meaning | Status |
|---|---|---|---|
| `WIDTH`, `HEIGHT` | 900.0 m | Square arena. | DECIDED |
| `GS_POSITION` | (450.0, 450.0) | Ground station, at the centre. | DECIDED |
| `NUM_M_DRONES` | 18 | Mission drones: create and relay. | DECIDED |
| `NUM_C_DRONES` | 7 | Communication drones: relay only. | DECIDED |
| `STATIC_MODE` | True | Nothing moves; mobility code is kept for Phase 2. | DECIDED |
| `QUEUE_CAPACITY` | 10 | Packets per **link** queue. | DECIDED |
| `MAX_TX_PER_STEP` | 1 | Packets each **link** queue sends per step. | DECIDED |
| `CHANNEL_LOSS_K` | 0.4 | Decay rate `k` in `exp(-k·M)`. | PROVISIONAL |
| `PACKET_TTL` | 50 steps | Packet lifetime. | DECIDED |
| `MAX_HOPS` | 10 | Most hops a packet may take. | DECIDED |
| `PACKET_INTERVAL_STEPS` | 2 (200 ms) | Steps between packets at each M-drone. | DECIDED set |
| `TRAFFIC_LOADS_MS` | (1000, 500, 200, 100) | The four loads in the experiment grid. | DECIDED |
| `RANDOM_TRAFFIC_OFFSETS` | True | Give each M-drone a random phase in its interval. | PROVISIONAL |
| `PACKET_SIZE` | 512 bytes | Informational; nothing depends on it. | DECIDED |
| `TIMESTEP` | 0.1 s | Simulation step. | DECIDED |
| `MAX_STEPS` | 1000 | Steps per run. | PROVISIONAL |
| `WARMUP_STEPS` | 50 | Steps excluded at the start of the window. | PROVISIONAL |
| `DRAIN_STEPS` | 50 | Steps excluded at the end (= one TTL). | PROVISIONAL |
| `PLACEMENT_SEED` | 42 | Default layout seed. | — |
| `RUN_SEED` | 42 | Default channel/traffic seed. | — |
| `DRONE_SPEED_MIN/MAX` | 10.0 / 30.0 m/s | Drawn per drone; unused while static. | DECIDED |
| `LOG_DIR` | `logs` | Where per-episode JSONL event logs go. | — |
| `PT_DBM` | 30.0 dBm | Transmit power (1 W). | DECIDED |
| `GT_DBI`, `GR_DBI` | 2.0 dBi | Antenna gains. | DECIDED |
| `F_HZ` | 2.4 GHz | Carrier frequency. | DECIDED |
| `RX_SENSITIVITY_DBM` | −54.0 dBm | Receiver sensitivity; sets the range. | DECIDED |
| `MAX_LINK_DISTANCE_M` | 249.69 m | *Derived* from the four above. | derived |

The radio values come from the IQMR paper (Sharvari et al., 2024,
arXiv:2408.09109). The sensitivity is derived rather than picked, so that the
FSPL test reproduces IQMR's stated 250 m range at its 1 W transmit power. The
arena size, the 18 + 7 fleet split and the 2-D simplification are **our**
choices, not IQMR's.

---

## 4. Outputs

Everything written by the scripts lands in `rl_sim/out/`, which is gitignored —
all of it regenerates from the commands above.

| File | Contents |
|---|---|
| `1a_runs.csv` | One row per experiment run (160): seeds, rule, load, created/delivered, total loss and its five causes as shares of created, pooled per-link loss and its three parts, queue occupancy, delay, hops, isolated M-drone count, drones in GS range, and whether the two views agreed. |
| `1a_summary.csv` | Mean and standard deviation of every numeric column, per (routing, load). |
| `1a_paired.csv` | Per load: mean and std of `greedy − random` total loss on identical seeds, and how many of the 20 placements greedy lost less on. |
| `1a_total_loss_vs_load.png` | Total packet loss against offered load, one line per rule, mean with std error bars. |
| `1a_loss_heatmaps.png` | Two heatmaps (greedy, random): rows are the four loads, columns are total loss, channel, queue_full, no_route, ttl+hop and pooled per-link loss. Mean % with std beneath, same 0–100 % scale in both panels. |
| `1a_link_matrix_placement1.png` | Per-link loss for placement 1: eight matrices (2 rules × 4 loads), senders as rows, receivers plus GS as columns, grey where a link was never used. |
| `range_check.csv` | One row per candidate range, with the connectivity figures and the transmit power that range would need. |
| `ploss_vs_distance.png` | The channel-loss curve for k = 0.2, 0.4, 0.8. |

`main.py` also writes a JSONL event log per episode under `logs/` (one record
per packet event, per-step network state and per-drone state).

---

## 5. Assumptions and limitations

- **2-D.** No altitude. The real scenario is three-dimensional.
- **No MAC layer, no interference, no collisions.** Nothing contends for the
  medium, which is exactly why each link gets its own queue and its own
  per-step budget, and why a drone can send on all its links at once. A real
  radio could not.
- **The channel-loss curve is a modelling choice.** `exp(-k · M)` with k = 0.4
  is a plausible shape, not a measured one, and k is marked PROVISIONAL. It
  drives the results directly, so it deserves scrutiny before any conclusion
  rests on it.
- **Ideal ACKs.** Instant, never lost, no capacity used, no retransmissions.
  Real ACKs would cost airtime and could themselves be lost.
- **Neighbor knowledge:** each drone knows its own position (GPS), the GS
  position, and its current neighbors' positions and link quality, as if
  learned from periodic hello messages and measured signal strength. The hello
  messages themselves are not simulated: they are free, never lost and always
  up to date. This matters little in the static Phase 1 and will be revisited
  for the dynamic Phase 2.
- **Isolated drones are not handled specially.** In a random 900 × 900 m layout
  at ~250 m range, some M-drones have no path to the GS at all; every packet
  they create is lost as `no_route`. `scripts/range_check.py` quantifies how
  often. What to do about it is an open question.
- **Energy is tracked but constrains nothing.** No drone ever runs out.

---

## 6. Folder structure

```
.
├── CLAUDE.md               Project context and working rules for agent sessions.
├── README.md               This file.
├── requirements.txt        The four third-party packages actually imported.
│
├── rl_sim/                 THE SIMULATOR — the only live code.
│   ├── main.py             Entry point: run one episode, print both views.
│   ├── fanet_sim/
│   │   ├── config.py       Every parameter, with DECIDED/PROVISIONAL status.
│   │   ├── envs/
│   │   │   ├── fanet_env.py  The environment: step order, routing rules,
│   │   │   │                 transmission, drops, ground-truth counters.
│   │   │   ├── drone.py      A drone: position, link queues, ACK counters.
│   │   │   ├── channel.py    FSPL, link existence, p_loss.
│   │   │   └── packet.py     Packet lifecycle and the five drop reasons.
│   │   └── utils/
│   │       ├── metrics.py       Connectivity graphs (networkx).
│   │       ├── event_log.py     JSONL event logger.
│   │       └── visualization.py Matplotlib animation of a run.
│   ├── scripts/
│   │   ├── metrics_1a.py     Both views of one run, and the check that they
│   │   │                     agree. Used by main.py and the experiment.
│   │   ├── experiment_1a.py  The 160-run grid, its CSVs and figures.
│   │   ├── range_check.py    Connectivity at candidate radio ranges.
│   │   ├── plot_ploss.py     The channel-loss curve.
│   │   └── analyze.py        Older stand-alone analyser for the JSONL logs.
│   │                         Nothing in the Phase-1a flow calls it.
│   ├── tests/              115 tests: channel, queues, routing, conservation,
│   │                       ACK-vs-ground-truth, static world, reproducibility.
│   └── out/                Generated results (gitignored).
│
├── archive/                NOT PART OF THE PROJECT. Kept for reference only.
│   ├── rl/                 The earlier PPO work: link-selection and topology
│   │                       policies, their trainer, the C-drone topology agent,
│   │                       the Q-routing baseline, and a trained checkpoint.
│   │                       Superseded — future RL will be written fresh.
│   ├── stage1/             An earlier stand-alone Dijkstra-vs-greedy study,
│   │                       with its own simulator, tests and results.
│   └── docs/               The outdated CLAUDE.md, README and PROJECT_STATE.
│
└── docs/                   The project proposal document.
```

Nothing in `rl_sim/` imports from `archive/`, and the live simulator does not
depend on torch. Both are worth keeping true.
