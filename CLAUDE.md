# CLAUDE.md — FANET drone-network project

## The project
Student project (Abed and Zena), supervised by Eran, with Prof. Reuven
Cohen involved. We simulate a drone network that relays data to a single
ground station (GS) and study how routing, and later drone movement,
affect packet loss. The fleet has two kinds of drones:
- M-drones (mission): create packets AND relay other drones' packets.
- C-drones (communication): create no packets, only relay; in later
  phases they can move to improve the network.

## Goal
Minimize TOTAL PACKET LOSS: the share of created packets that never
reach the GS. Per-link loss and loss by cause (channel, queue_full,
no_route, dead_end, ttl, hop_limit) are secondary, used to explain the
total.

## Plan (order set by the supervisor; each step starts only after the
## previous one is approved)
Phase 1 — STATIC world (nothing moves):
  1a. Non-learned baselines: GREEDY and RANDOM routing.   done
  1b. RL routing agent on every drone, reward = delivery rate. <- CURRENT
Phase 2 — DYNAMIC world (M-drones fly), only after Phase 1:
  2a. Baselines again, with C-drone movement rules (hover, random walk,
      move toward the busiest neighbor).
  2b. RL routing agent alone.
  2c. RL topology agent alone (C-drone movement), greedy routing.
  2d. Both agents together.
Current work is 1a only: no learning of any kind. Later RL agents will
be written NEW; the archived RL code is not to be reused.

## Current model (full details and DECIDED/PROVISIONAL status in
## README.md at the repo root, written at Checkpoint 5)
- World: 2D, 900x900 m, GS at (450,450), 18 M + 7 C drones, placed
  uniformly at random. Static. Time step 100 ms.
- Layout filter: only placements where EVERY M-drone has a path to the
  GS are used (same link rule as the simulator, paths may run through
  any drone, C-drones included; C-drones themselves need no path). A
  rejected layout is discarded and another drawn from the same
  placement stream, so a seed still names one accepted layout.
  Greedy can still hit a dead end -- a drone with a path to the GS but
  no neighbor closer to it -- and those still drop as no_route.
- Link queues: one FIFO queue per outgoing link (per neighbor, plus
  the GS link when in range). Capacity 10 per link; each link sends at
  most 1 packet per step; a drone's links all send in the same step
  (no collisions modeled). A packet placed in a full link queue is
  dropped (queue_full).
- Routing decision at enqueue time: when a packet is created or
  arrives, the drone picks the next hop and puts it in that link's
  queue. Arrivals within a step are processed in random (reproducible)
  order. Arrived packets wait at least one step.
- Channel loss: a logistic curve in DISTANCE, fitted to measured UAV
  links in Rosati et al., "Dynamic Routing for Flying Ad Hoc Networks"
  (arXiv:1406.4399):
      p_loss(d) = 1 / (1 + exp(-s*(d - D50)))
  with D50 = 356 m and s = 0.025 (both PROVISIONAL). Loss is ~0.2% at
  100 m, 6.6% at 250 m, 50% at 356 m, 99% at 540 m. The same curve
  covers the last hop into the GS.
  REPLACED the earlier FSPL model (-54 dBm sensitivity, a hard 249.69 m
  range, loss exp(-k*M) with k = 0.8), which hit ~100% loss right at
  250 m. The FSPL code is gone; scripts/plot_ploss.py keeps a local copy
  of the old curve only to draw the two together.
- Link existence: a link exists while p_loss < LINK_MAX_LOSS = 0.5
  (PROVISIONAL), i.e. out to exactly 356 m. Justification: a drone lists
  a neighbor only when at least half of its hello messages get through.
  ONE rule everywhere: neighbors, the GS link, the connected-layout
  filter and the range check.
  This was 0.99 (~539.8 m) when the curve was introduced at Checkpoint
  7; that reach was a side effect of the cutoff, not a choice, and it
  made the network very dense. At 0.5 a drone has ~8.6 neighbors and
  ~12 of 25 drones link straight to the GS.
- Link loss = 1 - (1 - channel loss) * (1 - queue_full), where
  queue_full = 1 if the sender's own queue for that link is full.
- Loop guard, a design assumption of the NETWORK (not of a routing
  rule): the packet header carries the list of drones it has visited,
  and no drone may forward a packet to a drone already on that list. It
  applies to EVERY routing rule. Cheap (~4 bytes with 25 drones and at
  most 10 hops) and standard practice: a BGP route carries its AS_PATH
  and a router drops any route already containing its own AS number.
  config.LOOP_GUARD (PROVISIONAL) selects:
    "path"     -- the whole visited list. THE DEFAULT.
    "previous" -- only the drone the packet just came from. What Phase
                  1a and Phase 1b up to CHECKPOINT 11 ran; kept so
                  those results stay reproducible.
  The GS is never excluded under either setting.
  A packet that has links but whose every reachable neighbor is already
  on its path drops as dead_end -- distinct from no_route, which means
  it had no candidate at all.
  GREEDY is unaffected: it only moves strictly closer to the GS, so it
  cannot revisit a drone. Identical results under both settings, which
  tests/test_loop_guard.py checks. RANDOM, rate and rl all change.
- GREEDY: among neighbors strictly closer to the GS, the lowest link
  loss; ties: lower channel loss, then closer to the GS; none closer ->
  drop (no_route).
- RANDOM: uniform over current neighbors, own random stream.
- GS: receives without limit.
- ACKs (ideal in Phase 1): GS end-to-end ACK to the source; hop-by-hop
  ACK (ok/channel) to the sender. Drones keep per-link counters built
  only from local knowledge and ACKs. These must always match the
  simulator's ground truth.
- Packets that arrive already dead are retired on arrival
  (expired_on_arrival).
- Loads: each M-drone creates one packet every 1000/500/200/100 ms.
  200 ms is the working load for the next phases; the experiment still
  sweeps all four. Metrics use packets created in steps [50, 950) of
  1000.
- Seeds: separate placement, channel/traffic and routing streams;
  greedy and random run on identical placement and traffic.
- Neighbor knowledge: each drone knows its neighbors' positions and
  link quality, as if from hello messages; hello messages are not
  simulated.
- Phase 1b adds two routing rules beside greedy/random: "rate"
  (delivery_rate * (1 - queue_full), epsilon 0.05 in training AND at
  test) and "rl" (a shared 3->32->32->1 network scoring each option
  from 3 local inputs: delivery_rate, own queue fill, channel loss;
  epsilon 0.1 training, 0 at test). Options are the current neighbors
  plus the GS when in range, minus what the loop guard above excludes;
  no candidate at all -> no_route, all candidates guarded -> dead_end.
  An end-to-end ACK walks back along the packet's path when it reaches
  the GS; a packet unacked by created_at + TTL is recorded lost by
  every drone that decided on it. Per-link delivery rate = mean of the
  last 50 resolved outcomes, seeded at 1 - channel_loss. A decision
  refused by a full queue is a training example but does NOT move the
  delivery rate: it was never sent.
  These rules do NOT use the hop-by-hop ACK. The simulator's ACK
  counters and the views_agree check stay as measurement tools only.

## Open questions (do NOT decide these; ask)
Whether LOOP_GUARD stays "path" (the header-carried visited list) is
pending supervisor confirmation; "previous" is kept as the baseline.
Pooling every drone's training examples into ONE shared network is an
ASSUMPTION pending supervisor confirmation; each drone still decides
from its own local inputs only. Also: ACK details (routed or ideal, capacity use, retransmissions); RL reward
form; whether the topology agent joins the static phase; link-break
handling and hello messages in the dynamic phase; a near-full load
(e.g. 90 ms).

## IGNORE / LEAVE BEHIND
We keep ONLY the simulator: world, drones, placement/mobility, channel,
packets, queues, logging, metrics, analysis. Everything below is out of
scope. Do not use it, extend it, or rely on it:
- PPO K-link (link selection) policy and PPO topology policy
- train.py, eval.py, any old training/reward code
- Q-routing
- any LSTM / GNN / trajectory-prediction code or plans
- any stage1/ folder or Dijkstra-vs-greedy study code and its results
- the old CLAUDE.md, PROJECT_STATE.md and the old README: outdated
- anything from earlier sessions not restated in a current prompt
None of it is deleted; it lives under archive/. Nothing in rl_sim/
imports from archive/. torch is allowed ONLY in rl_sim/agents/:
fanet_sim/ stays torch-free and never imports agents/ -- the router is
injected into FANETEnv and called through five hooks. Keep all of that
true.

## Keeping results consistent
Every figure and CSV carries a settings stamp (curve, cutoff, reach, world,
queues, load, placements, window, git commit) read from config.py. After
changing anything in config.py, rerun `python scripts/regenerate_all.py` so
every output comes from one config -- a stale figure beside a fresh table is
worse than neither.

## RULES
0. Never commit to main. Work on a branch; commit per approved
   checkpoint.
1. Read before writing. Before changing anything, read the code that
   actually runs. Do not trust docs or earlier session summaries.
2. Never guess. If something is unclear, or the code differs from the
   prompt in a way the prompt does not cover, STOP and ask.
3. No scope creep. Build only what was asked.
4. Values marked [DECIDED] are fixed. Values marked [PROVISIONAL] carry
   a "PROVISIONAL - to confirm" comment in config.
5. List edits and wait for OK. Before editing any file, list what you
   will change in it and why, then wait for approval.
