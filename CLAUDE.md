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

## Decided by the supervisor (not open for revisiting)
- The objective is TOTAL PACKET LOSS. Everything else is explanation.
- One FIFO queue per OUTGOING LINK, not one queue per drone.
- The static world comes first (greedy, random, then RL); the dynamic
  world only after Phase 1 is approved.
- The RL reward is the end-to-end delivery outcome, carried by the GS
  ACK -- not a per-hop signal.

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
Current work is 1b: the RL routing agent, in the static world. Phase 1a
is done. The Phase-1b agents were written NEW; the archived RL code is
not to be reused, and the same holds for the Phase-2 agents.

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
- Channel loss [DECIDED 2026-10-13]: a free-space path loss budget gives
  each link a MARGIN above the receiver's sensitivity, and that margin
  drives an S-curve:
      FSPL_dB(d) = 20*log10(d) + 20*log10(F_HZ) + 20*log10(4*pi/c)
      M(d)       = PT_DBM + GT_DBI + GR_DBI - FSPL_dB(d) - RX_SENSITIVITY_DBM
      p_loss(M)  = 1 / (1 + A * exp(b * M)),  A = (1 - SENS_LOSS)/SENS_LOSS
  with PT_DBM = 30 (1 W), GT_DBI = GR_DBI = 2, F_HZ = 2.4e9,
  RX_SENSITIVITY_DBM = -54, SENS_LOSS = 0.10, LOSS_50_DISTANCE_M = 350.
  A = 9 and b = 0.749045 per dB; b is DERIVED at import time from the edge
  and must never be hardcoded.
  The sensitivity is NOT a wall: it is the point where 10% of transmissions
  are lost, following 802.11 OFDM (10% packet error rate on 1000-octet
  frames). That point is 249.689108 m. Links between 249.7 m and 350 m have
  a NEGATIVE margin and still exist; they simply lose more. Loss is 0.03%
  at 100 m, 2.6% at 200 m, 10% at 249.7 m, 26.8% at 300 m, 50% at 350 m.
  The same curve covers the last hop into the GS.
  History: an earlier FSPL loss exp(-0.8*M) put 100% loss at the
  sensitivity, a wall at 250 m; replaced by an S-curve where the
  sensitivity means 10% loss (802.11).
- Link existence: a link exists while p_loss < LINK_MAX_LOSS = 0.5, i.e.
  out to exactly LOSS_50_DISTANCE_M = 350 m. ONE rule everywhere:
  neighbours, the GS link, the connected-layout filter and the range check.
  The link edge is a design choice, not a physical constant: with 1 W in
  free space the radio would reach kilometres. We compared edges from 300
  to 450 m on 3000 random layouts (out/edge_scan.csv) and chose 350 m,
  where about 95% of random layouts are connected and about half of the
  M-drones (52%) still need a relay to reach the ground station. Shorter
  edges often leave drones cut off; longer edges let most drones reach the
  GS directly, which leaves little to route. Chosen from geometry alone,
  before any routing experiment, and FIXED.
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
  The experiment sweeps all four. Metrics use packets created in steps
  [50, 950) of 1000.
- Working load 200 ms [DECIDED 2026-10-13]. Reason: the heaviest load
  where the network is congested (queue_full drops are a real part of
  the loss) but not yet saturated (greedy beats random on every
  placement). Chosen by the team after running all four loads, and
  re-checked after the channel change: at 200 ms greedy queue_full =
  40.0% and greedy beats random on 20/20 placements by 21.8 points; at
  100 ms only 18/20 (saturated).
- Seeds: separate placement, channel/traffic and routing streams;
  greedy and random run on identical placement and traffic.
- Neighbor knowledge: each drone knows its neighbors' positions and
  link quality, as if from hello messages; hello messages are not
  simulated.
- Phase 1b adds two routing rules beside greedy/random: "rate"
  (delivery_rate * (1 - queue_full), NO exploration -- deterministic
  given the layout and the traffic, like greedy) and "rl" (a shared 3->32->32->1 network scoring each option
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
- RL reward form [DECIDED by the supervisor]: a per-packet END-TO-END
  reward, 1 if the packet reached the GS and 0 if it did not. When a
  packet arrives, the GS ACK walks back along its path and every drone
  that decided on that packet is told the outcome; a packet still
  unacked at created_at + TTL is recorded lost the same way. There is
  no per-hop reward and no discounting: every decision on a packet gets
  that packet's single outcome.
- CRITICAL implementation note -- do NOT simplify this away:
  Packet.path records only the hops that SUCCEEDED. A drone whose
  transmission failed never appears there, and those are exactly the
  examples the network most needs. So rl/rate decisions are stored
  SEPARATELY, in the tracker, as a LIST per packet (a list, not one
  entry: under LOOP_GUARD = "previous" a packet can decide at the same
  drone more than once), and outcomes are resolved from that store.
  Never reconstruct who decided on a packet from its path.
- Agent seeding: the exploration stream comes from the agent's OWN
  np.random.SeedSequence([run_seed, config.AGENT_STREAM_TAG]) with
  AGENT_STREAM_TAG = 0xA6E17 -- NOT a third spawn off the environment's
  two streams. Spawning a third would shift the channel stream and move
  every Phase-1a result with it. A test checks greedy and random still
  reproduce out/1a_runs.csv to 1e-12.
- RL training protocol:
    layouts  training 101-300 (cycled, one per run), validation
             301-310 (picks the kept model), train-check 101-110 (the
             overfitting comparison only), TEST 1-50 (seeds 1-20 are
             kept as the original CHECKPOINT-12 set).
    schedule 1000 steps per run at 200 ms; freeze and evaluate every 10
             runs; stop at 300 runs or early when validation has not
             improved by 0.5 points over five evaluations; repeated for
             network init seeds 0, 1 and 2.
    learning Monte Carlo targets (the packet's own outcome, no
             bootstrapping), binary cross-entropy, Adam at 1e-3,
             batches of 32, NO replay buffer and NO target network.
    models   saved under out/models/<loop_guard>/, so a change of guard
             does not overwrite the previous models.

## Status
Phase 1a and Phase 1b are COMPLETE on the new channel. Everything in
out/ is a current result; every figure and CSV carries the new stamp.

Headline, 50 TEST layouts at 200 ms, mean % of packets created:
  rl 22.29   rate 25.05   greedy 52.76   random 74.17
  rl - rate: -2.76 points, lower on 41 of 50 layouts, p < 0.0001
    (exact two-sided sign test and paired t-test)
At 500 ms: rl 12.56, rate 20.96, greedy 27.42, random 74.28; rl - rate
-8.40 points, 50/50, p < 0.0001.

Supporting checks, all done on the new channel:
- Loop guard: "path" loses fewer packets than "previous" for random
  (-4.53 points at both loads), rate (-1.75 / -2.52) and rl (-2.14 /
  -3.48), all p < 0.0001; greedy is identical under both, on every
  layout. rl was trained separately under each guard, so no model is
  evaluated under a guard it did not train with.
- Channel diagnosis: of rl's channel losses, 45.6% were geometry-forced,
  7.1% congestion-forced and 47.3% had a cleaner idle option; within
  that last group 99.95% were trade-offs where the cleaner link had a
  worse delivery rate. (out/1b_channel_diagnosis.csv)
- Queue size Q = 5 / 10 / 20: rl beats rate at every Q. Q = 10
  reproduces the committed results to 1e-12.
- Range check: at the 350 m edge, 99.7% of M-drones have a path to the
  GS and 4.0% of random layouts strand at least one.

Next: the written report, then Phase 2. The report under docs/report/
is still written against the old channel and is rewritten next.

## Open questions (do NOT decide these; ask)
- Whether LOOP_GUARD stays "path" (the header-carried visited list);
  "previous" is kept as the baseline either way.
- Pooling every drone's training examples into ONE shared network is an
  ASSUMPTION pending confirmation; each drone still decides from its
  own local inputs only.
- Retransmissions, or a milder channel instead. Most of rl's remaining
  loss is channel loss on links it had no cleaner alternative to; both
  answers are MODEL CHANGES and are the supervisor's call.
- Whether the S-curve already absorbs MAC-layer retries. If it does,
  adding retransmissions on top would double-count them.
- A hop count as a 4th network input. Deferred, optional.
- An optional robustness check on denser layouts (more drones, or a
  smaller arena), to see whether the ordering holds.
- ACK details: routed or ideal, whether they consume capacity.
- Whether the topology agent joins the static phase.
- Link-break handling and hello messages in the dynamic phase.
- A near-full load (e.g. 90 ms).

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
