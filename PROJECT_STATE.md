# FANET RL Routing — Project State

_Last updated: 2026-06-11_

This document is the single place to understand **what the simulator is, how it
works, what has been built, and what has not.** Read this first when returning to
the project.

---

## 1. What this project is

A research simulator for a **Flying Ad-hoc Network (FANET)**: a fleet of drones
that relay data packets back to a ground station (GS) over a constantly-changing
multi-hop wireless topology. It is the foundation for reinforcement-learning
work on routing and topology control.

There are two drone types:

| Type | Role | Mobility | Generates packets? |
|------|------|----------|--------------------|
| **M-drone** (mission) | flies a mission **and** relays packets | straight line start → end, then stops | yes (1/step) |
| **C-drone** (comm)    | only improves connectivity | steered by the topology policy | no |

The ground station sits at the centre of the arena. A packet is **delivered**
when it reaches the GS, and **dropped** on TTL expiry, hop-limit, or a routing
void (no neighbour closer to the GS).

---

## 2. Current state at a glance

| Capability | Status |
|------------|--------|
| Phase-1 simulator (mobility, FSPL channel, packets, greedy routing, metrics, visualisation) | ✅ Built |
| K-link selection MLP on every drone | ✅ Built + **PPO-trained** |
| Topology movement MLP on C-drones | ✅ Built + **PPO-trained** |
| Q-routing baseline | ✅ Built (optional, `--routing qroute`) |
| Train → save weights → evaluate workflow | ✅ Built (`train.py`, `eval.py`) |
| RL **routing** agent (replace greedy) | ❌ Not built (routing stays greedy by design) |
| GNN state embeddings | ❌ Not built |
| 3-D space, interference/MAC, fading | ❌ Not built |

**Measured effect of training** (seed 42, 300 steps, greedy routing, same map):

| Metric | Random policies | After 40 PPO episodes |
|--------|----------------|-----------------------|
| **PDR** | 0.298 | **0.417** |
| drop_rate | 0.699 | 0.580 |
| frac connected to GS | 0.631 | 0.706 |

Absolute PDR is connectivity-bound (50 drones over 1750×1750 m at 250 m range is
sparse); see §8.

> ⚠️ These numbers predate the **C-drone relay link reward** (added 2026-06-11,
> §5.1) and the C-drone reward rework, so the saved `trained_policies.pt` is
> stale. **A 1000-step retrain is pending** before re-quoting metrics.

---

## 3. Repository map

The RL simulator (this document's subject) lives under `rl_sim/`, kept
separate from `stage1/` (the standalone classical routing baseline) and
top-level docs.

> **Stage 1 status (finalized, see `stage1/README.md` for detail):** the
> greedy-vs-dijkstra comparison protocol is finalized in `stage1/compare.py`.
> `restricted_pdr` (delivery ratio over graph-reachable M-drones only, shared
> denominator across routers) is the **primary** comparison metric — global
> PDR and %-unreachable are context only. Protocol: paired sign test over
> per-topology deltas + a normal-approximation 95% CI, no queues (isolates
> channel × routing), random placement for both M- and C-drones (ring/grid
> layouts are out of scope for this comparison — they matter only for future
> C-drone-placement work). `PRACTICAL_SIGNIFICANCE_THRESHOLD = 0.10` is
> carried from planning and **not yet confirmed by the team**.

```
fanet-rl-routing/
├── PROJECT_STATE.md          # this file
├── stage1/                   # classical routing baseline (standalone, see stage1/README.md)
├── docs/                     # proposal / reference documents
└── rl_sim/                   # RL simulator + training/eval pipeline
    ├── train.py               # PPO training entry point (NEW)
    ├── eval.py                 # run the sim with trained/random policies (NEW)
    ├── main.py                 # phase-1 sim entry point (random policies; protected)
    ├── trained_policies.pt     # example saved checkpoint (produced by train.py)
    ├── logs/                   # JSONL event logs (gitignored)
    ├── fanet_sim/
    │   ├── config.py             # ALL parameters (sim + PPO hyper-params)
    │   ├── envs/
    │   │   ├── drone.py          # Drone: state, mobility, link selection, policies
    │   │   ├── channel.py        # FSPL link model (protected)
    │   │   ├── packet.py         # Packet + PacketFactory (protected)
    │   │   ├── policies.py       # the 4 MLPs: 2 actors + 2 critics
    │   │   ├── topology_agent.py # C-drone controller + local topology reward
    │   │   └── fanet_env.py      # environment: reset()/step(), routing, rollouts
    │   ├── rl/                   # PPO support (NEW package)
    │   │   ├── sampling.py       # Plackett–Luce sample / log-prob maths
    │   │   └── ppo.py            # PPOConfig, GAE, updates, PolicyBank, save/load
    │   └── utils/
    │       ├── metrics.py        # connectivity sampling helpers (protected)
    │       ├── event_log.py      # JSONL event logger
    │       └── visualization.py  # matplotlib animation (protected)
    └── scripts/
        └── analyze.py             # Stage-2 metric computation from the JSONL log (protected)
```

**Protected files** (must not be modified per project constraint): `metrics.py`,
`channel.py`, `packet.py`, `scripts/analyze.py`, `visualization.py`, `main.py`.

---

## 4. How the simulator works

### 4.1 The step loop (`fanet_env.py: FANETEnv.step`)

Each timestep (`TIMESTEP = 0.1 s`):

1. **Move M-drones** in a straight line toward their end point (stop on arrival).
2. **Move C-drones** by the topology policy's `[dx, dy]` (capped to max speed).
3. **Recompute links**: every drone finds in-range *candidates* (FSPL test), then
   keeps **K active links** chosen by its link policy.
4. **Topology reward** for each C-drone is computed (local, see §5.2).
5. **Generate packets** (each M-drone emits `PACKET_RATE` packets for the GS).
6. **Route packets** one hop via **greedy geographic forwarding**: send to the
   neighbour closest to the GS; if none is closer, drop (routing void).
7. **Expire** stale packets (TTL / hop limit).
8. **Charge energy** (radio idle + rx; motion energy was charged on move).
9. **Log** per-step network + per-drone state to a JSONL event log.

The episode ends when **all M-drones have arrived** or `MAX_STEPS` is reached.

### 4.2 Channel (`channel.py`)

Free-Space Path Loss model. Two endpoints are "connected" if received power
clears the receiver sensitivity (−54 dBm), which reproduces IQMR's 250 m range
at 1 W. Link quality ∈ [0,1] degrades with the SNR margin. No interference, MAC,
or fading (deferred).

### 4.3 Metrics (`scripts/analyze.py`)

The simulator **does not aggregate metrics in-loop**. It writes raw events to a
JSONL log; `scripts/analyze.py` reads that log and computes PDR, delay, hops,
NRL, connectivity, energy, and a per-window PDR-over-time breakdown. Adding a
metric is a change to that file only.

### 4.4 Interface

`reset() -> {id: state}` and `step(actions) -> (obs, rewards, dones, infos)`,
PettingZoo-style. In normal runs `actions` is ignored (policies act internally).

---

## 5. The learned policies (the RL feature)

Four small MLPs live in `policies.py`. Each drone owns its **own** instances
(independent training). All start random; `train.py` trains them with PPO.

### 5.1 K-link selection — **every drone**

- **Actor** `LinkScorePolicy`: `Input(5) → 16 → 16 → 1`. Scores one candidate
  neighbour from 5 features (link quality, distance to it, its distance to GS,
  its queue length, its degree).
- **Action**: choose **which K of N candidates to keep** as active links.
  - *Eval* (deterministic): keep the top-K scores.
  - *Training* (stochastic): **Plackett–Luce** — sample K without replacement
    using the scores as logits (`rl/sampling.py`). This is the differentiable
    stochastic version of top-K.
- **Critic** `LinkValue`: `Input(6) → 16 → 16 → 1` over a fixed drone summary
  (dist-to-GS, queue, #candidates, mean link quality, energy, GS-reachable).
- **Reward** — **role-specific, because M- and C-drones keep links for different
  reasons** (`config.LINK_REWARD_*`, magnitudes ±1 for both so one LR fits):
  - **M-drones — mission-traffic delivery (source-attributed):** `+1` when a
    packet **that originated at this drone** is delivered, `-1` when one of its
    packets is dropped, credited the step the packet resolves. An M-drone's link
    choice is judged by whether *its own* traffic gets through.
  - **C-drones — relay usefulness (holder-attributed):** a C-drone generates no
    packets, so it is judged on how good a *relay* it is. `+1` each step for each
    packet it **forwards onward or delivers** to the GS, `-1` for each packet that
    **voids or expires while it is holding it** (a void = its kept links left no
    neighbour closer to the GS). This is local and causal: the blame for a dead
    end lands on the drone whose links caused it, not on innocent upstream relays.
  - *Why split (was source-only for everyone):* source-only gave C-drones a
    permanent **zero** reward (they own no packets), so their link nets never
    moved off random init. The relay reward makes "K-link training on **all**
    drones" actually true. The signal is sparse until a C-drone is positioned on a
    traffic path — but the topology policy (§5.2) drives it there, so the two
    C-drone policies compose: movement gets it onto paths, link selection then
    learns which links relay well. (Env: `_step_src_reward` vs
    `_step_relay_reward` in `fanet_env.py`; assigned per drone type at step 6b.)

### 5.2 Topology movement — **C-drones only**

- **Actor** `TopologyPolicy`: `Input(10) → 32 → 32 → 2`. Maps the C-drone's local
  state to a movement **mean** `[dx, dy]`. A learnable `log_std` makes it a
  diagonal Gaussian for PPO; eval uses the mean directly.
  - **Inputs are normalised to ~[0,1]** (positions by arena size, distances by
    range, etc.) — raw 0–1750 m positions otherwise saturate the MLP and the
    policy collapses to a constant move that drifts C-drones into the walls.
  - The last 2 of the 10 features are the **unit bearing to the nearest M-drone**
    (the relay target) — without it the policy can't tell which way to go.
  - The output is **`tanh`-scaled to ±`TOPOLOGY_MAX_STEP_M`**, so the policy
    commands full-speed moves from modest weights instead of barely nudging.
- **Critic** `TopologyValue`: `Input(10) → 32 → 32 → 1`.
- **Reward** (in `topology_agent.py`, dense, local) — **relay-coverage**:
  `+ coverage (M-drones in range) + 3·progress (metres reduced toward nearest
  M-drone, scaled by step size) − 0.2·motion energy`. The progress term is
  potential-based shaping: it's the dense per-step signal that overcomes the
  arena-size-vs-3 m-step scale problem and drives the C-drones to home in on the
  mission drones. Only M-drones count, so C-drones don't chase each other.
  - *Measured (seed 42):* trained C-drones move 1.28 m/step (random 0.39) and
    cover 3.21 M-drones each (random 2.14).
  - *Reward design history* (each fixed a real failure, in order): differential
    own-neighbour reward → border trap; absolute own-neighbour → froze in place;
    static M-proximity → too weak per step; **progress-shaped relay coverage +
    bearing feature + tanh output** → actually moves.

### 5.3 PPO (`rl/ppo.py`)

Standard PPO: clipped surrogate objective, **GAE** advantages, value-function
loss, entropy bonus. One Adam optimiser per drone per policy.

`PolicyBank` owns every drone's networks + optimisers and **persists across
episodes**. The environment recreates drones each `reset()`, so the bank
**injects** the persistent weights back into each fresh drone — this is what lets
learning accumulate. After each episode `train.py` calls
`bank.update_link(...)` and `bank.update_topology(...)`.

**Rollout collection**: when the env is built with `training=True`, `step()`
uses the stochastic action paths and records per-drone transitions into buffers
(`get_link_rollouts()` / `get_topology_rollouts()`). Link rewards are filled in
after routing each step once deliveries/drops are known.

---

## 6. Workflow — how to use it

The RL simulator lives in `rl_sim/`; run all commands below from inside
that directory (`cd rl_sim && ...`).

```bash
pip install numpy matplotlib networkx pillow torch
```

**1. Train** → writes a checkpoint of all policy weights:
```bash
python train.py                                  # config defaults (60 ep × 300 steps)
python train.py --episodes 40 --steps 300 --seed 42 --save trained_policies.pt
python train.py --vary-seed                      # different scenario each episode (generalise)
```
- Prints PDR every episode. By **default every episode replays the same
  scenario** (with fresh exploration noise) so the PDR curve reflects learning,
  not scenario luck. `--vary-seed` trades a clean curve for generalisation.

**2. Evaluate** → run the deterministic simulator with the trained weights:
```bash
python eval.py --steps 300 --seed 42 --weights trained_policies.pt
python eval.py --steps 300 --seed 42                       # random baseline (A/B)
```
- Use the **same `--seed` and `--steps`** for a fair before/after.
- `eval.py` exists because `main.py` is protected and always builds random
  policies; `eval.py` is the bridge that loads a checkpoint into the normal sim.

**Phase-1 sim, unchanged** (random policies, animation, etc.):
```bash
python main.py --no-anim
python main.py            # saves episode.gif
```

---

## 7. Key configuration (`config.py`)

| Group | Constants |
|-------|-----------|
| Fleet | `NUM_M_DRONES=36`, `NUM_C_DRONES=14`, speed 10–30 m/s |
| Space | `WIDTH=HEIGHT=1750`, `GS_POSITION=(875,875)` |
| Radio | tuned in `channel.py`; effective range ≈ 250 m |
| Links | `K_LINKS=3`, bounds `[K_LINKS_MIN=2, K_LINKS_MAX=5]` |
| Packets | `PACKET_RATE=1`, `PACKET_TTL=50`, `MAX_HOPS=10` |
| Episode | `MAX_STEPS=1000` (training overrides via `--steps`) |
| PPO | `PPO_LR=3e-4`, `PPO_GAMMA=0.99`, `PPO_LAMBDA=0.95`, `PPO_CLIP=0.2`, `PPO_EPOCHS=4`, value/entropy coefs |
| Link reward | `LINK_REWARD_DELIVERED=+1`, `LINK_REWARD_DROPPED=−1` (role-specific: M=source, C=relay — §5.1) |

---

## 8. Why absolute PDR is low (and how to raise it)

PDR is **connectivity-bound**, not a training bug:

- 50 drones spread over 1750×1750 m with a 250 m range is sparse — many drones
  have no neighbour closer to the GS, so **greedy routing voids** and drops the
  packet. (The old PDR≈0.93 you may recall was a denser 9-drone / 1000×1000 m
  config.)
- **Greedy routing is intentionally frozen.** PPO link-selection can only pick
  among neighbours that already exist; it cannot create connectivity.

Levers that raise PDR (config only, no code changes): more C-drones, larger
radio range, smaller area, more drones. To improve *learning*: more episodes, or
`--vary-seed` to generalise.

---

## 9. Design decisions & gotchas

- **Routing untouched.** All learning is in link-selection and C-drone movement;
  greedy geographic routing is the fixed baseline.
- **Per-drone independent policies.** No weight sharing; each drone has its own
  actor/critic/optimiser in `PolicyBank`.
- **Role-specific link reward** (§5.1). M-drones: source-attributed (their own
  packets' end-to-end fate). C-drones: holder-attributed relay reward (forwarded
  vs. voided/expired-at-me). Earlier the reward was source-only for everyone,
  which left C-drone link policies with a zero signal; the relay reward fixes that
  so all drones learn link selection for their actual job.
- **Topology policy = homing relay** (§5.2): normalised features + a nearest-M
  bearing input + `tanh`-scaled output + a progress-shaped relay-coverage reward.
  This stack was reached by fixing, in turn, border-drift, freeze-in-place, a
  too-weak distance signal, and a too-small output magnitude. If C-drones ever
  go static or wander again, that's the chain of knobs to check.
- **Fixed scenario by default** in `train.py` for a readable learning curve;
  torch is re-seeded after `reset()` so exploration noise still varies.
- **Training vs eval gap.** Training PDR is stochastic (exploration); the
  deterministic `eval.py` number is the one to report.
- **No metric aggregation in the loop** — always go through the JSONL log +
  `scripts/analyze.py`.

---

## 10. Roadmap (not yet built)

1. **RL routing agent** (MAPPO/MADDPG) to replace greedy forwarding.
2. **GNN** state embeddings feeding the policies.
3. **Topology reward upgrade** to a global connectivity proxy (e.g. Fiedler
   value) instead of the local placeholder.
4. **Richer channel**: 3-D space, interference/MAC, fading.
5. **Generalisation**: train across many scenarios; held-out evaluation maps.
