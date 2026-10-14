# FANET drone-network simulator — Phase 1

A simulator for a flying ad-hoc network (FANET): a fleet of drones relays data
packets back to a single ground station (GS) over multi-hop wireless links.
Student project by Abed and Zena, supervised by Eran, with Prof. Reuven Cohen
involved.

**Goal: minimize total packet loss** — the share of packets created that never
reach the GS. Per-link loss and the breakdown by cause (channel, queue_full,
no_route, dead_end, ttl, hop_limit) are secondary; they exist to explain the
total.

The plan, in the order set by the supervisor, each step starting only after the
previous one is approved:

| Phase | Step | What |
|---|---|---|
| 1 — static world | **1a** | Non-learned baselines: GREEDY and RANDOM routing. **done** |
| 1 — static world | **1b** | RL routing agent on every drone. **done ← you are here** |
| 2 — dynamic world | 2a–2d | M-drones fly; C-drone movement rules, then RL routing and topology agents. |

**Phase 1 is complete: both the baselines and the learned router.** The
Phase-1b agents were written new; none of the archived RL code is reused.
PyTorch is used only in `rl_sim/agents/`; the simulator itself stays
torch-free.

---

## Status

**Phase 1a and Phase 1b are complete on the current channel model.**
Everything in `rl_sim/out/` is a current result, and every figure and CSV
carries the settings stamp of the code that produced it.

Headline, on the 50 held-back TEST layouts at the 200 ms working load, as a
mean % of the packets created:

| rule | 200 ms | 500 ms |
|---|---|---|
| random | 74.17 | 74.28 |
| greedy | 52.76 | 27.42 |
| rate | 25.05 | 20.96 |
| **rl** | **22.29** | **12.56** |

Paired on the same 50 layouts, `rl − rate` is **−2.76 points at 200 ms**,
lower on **41 of 50** layouts, and **−8.40 points at 500 ms**, lower on
**50 of 50**; both p < 0.0001 under an exact two-sided sign test and a paired
t-test.

Supporting checks, all on the current channel: the loop guard (`"path"` loses
fewer packets than `"previous"` for random, rate and rl, and greedy is
identical under both), the channel-loss diagnosis, a queue-size sweep at
Q = 5 / 10 / 20, and the geometry range check.

The written report under `docs/report/` is the one thing still out of date:
it was written against the previous channel model and is being rewritten.

---

## 1. What the simulator models

### World

A 2-D square 900 × 900 m with the ground station at its centre, (450, 450).
Twenty-five drones are placed uniformly at random and **never move**:

- **18 M-drones** (mission) create packets *and* relay other drones' packets.
- **7 C-drones** (communication) create nothing; they only relay.

One simulation step is 100 ms, and a run is 1000 steps.

**Not every random layout is used.** A layout is accepted only if **every
M-drone has a path to the GS**, using the same link rule as the simulator and
allowing the path to run through any drone, C-drones included. C-drones
themselves need no path. A rejected layout is discarded and another is drawn
from the same placement stream, so a given `placement_seed` still names one
specific accepted layout; `FANETEnv.placement_draws` records how many draws it
took.

The filter exists because a stranded M-drone loses every packet it creates as
`no_route`, whatever the routing rule does — that is a property of the layout,
not of the routing, and it swamps the comparison. It does **not** remove every
`no_route` drop: greedy can still reach a dead end, a drone that has a path to
the GS but no neighbour closer to it. `scripts/range_check.py` deliberately
runs **unfiltered**, since measuring how often isolation happens is its job.

At the current 350 m edge about **5 % of random layouts** strand at least one
M-drone (`out/edge_scan.csv`, 3000 layouts), so the filter is rarely invoked
but not inert.

### Channel and links

Loss comes from a **free-space path loss budget turned into an S-curve**. The
budget gives each link a *margin* above the receiver's sensitivity, and the
margin sets the chance one transmission is lost:

```
FSPL_dB(d) = 20·log10(d) + 20·log10(F_HZ) + 20·log10(4π/c)
M(d)       = PT_DBM + GT_DBI + GR_DBI − FSPL_dB(d) − RX_SENSITIVITY_DBM
p_loss(M)  = 1 / (1 + A · exp(b · M))        A = (1 − SENS_LOSS)/SENS_LOSS
```

with 1 W at 2.4 GHz, 2 dBi antennas and a −54 dBm receiver. `A` = 9 and
`b` = 0.749045 per dB. **`b` is derived, never written down**: it is whatever
value puts 50 % loss at the link edge, computed when `channel.py` is imported,
so moving the edge moves `b` with it.

The receiver sensitivity is **not a wall**. It is the point where 10 % of
transmissions are lost, following 802.11 OFDM, which defines sensitivity as
the power giving a 10 % packet error rate on 1000-octet frames. That point is
**249.7 m**.

| d (m) | 50 | 100 | 150 | 200 | 249.7 | 250 | 300 | 350 |
|---|---|---|---|---|---|---|---|---|
| margin (dB) | 13.97 | 7.95 | 4.43 | 1.93 | 0.00 | −0.01 | −1.59 | −2.93 |
| p_loss | 0.000 | 0.000 | 0.004 | 0.026 | 0.100 | 0.101 | 0.268 | 0.500 |

Links between 249.7 m and the edge have a **negative margin and still exist**;
they simply lose more. The GS uses the same model — it is just another
endpoint, and the last hop into it draws like any other.

**A link exists while its loss stays under `LINK_MAX_LOSS` = 0.5**, which puts
the edge at exactly **350 m**. That is the single rule used everywhere:
neighbour sets, the GS link, the connected-layout filter and the range check.
Distance alone decides it, through the curve.

The edge is a **design choice, not a physical constant**: with 1 W in free
space the radio would reach kilometres. We compared edges from 300 to 450 m on
3000 random layouts (`out/edge_scan.csv`) and chose 350 m, where about 95 % of
random layouts are connected and about half of the M-drones (52 %) still need
a relay to reach the ground station. Shorter edges often leave drones cut off;
longer edges let most drones reach the GS directly, which leaves little to
route. Chosen from geometry alone, before any routing experiment, and fixed.

At this cutoff, across the 20 experiment layouts, a drone has about **7.9
neighbours** and **12.2 of 25** drones hold a direct link to the GS. The range
check, which samples 100 layouts and does not filter them, gives 8.6 and 12.4.

Run `scripts/plot_ploss.py` to see the curve against the one it replaced.

> **This replaced an earlier model** (up to Checkpoint 7): free-space path loss
> with a −54 dBm receiver sensitivity, giving a hard 249.69 m range, and loss
> `exp(-k · M)` on the margin above sensitivity with k = 0.8. That curve reached
> ~100% loss exactly at 250 m, which measured UAV links do not show, and 250 m
> is short against a 900 × 900 m area. The FSPL code is gone; only
> `scripts/plot_ploss.py` keeps a copy of the old formula, to draw both curves
> together.
>
> The cutoff was first set to 0.99, which reached 539.8 m and made the network
> very dense (~15 neighbours per drone). That reach was a side effect of the
> cutoff rather than a deliberate choice, and it handed the RANDOM rule a pile
> of links losing over 90% of what was sent on them. It is now 0.5.

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
channel_loss = p_loss(that link's distance)   — the S-curve above
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

### The loop guard

**The packet header carries the list of drones the packet has visited, and no
drone may forward a packet to a drone already on that list.** This is a property
of the *network*, not of a routing rule: it applies to greedy, random and both
Phase-1b rules alike, and a rule cannot opt out of it.

It is cheap. With 25 drones an id fits in 5 bits and a packet takes at most
`MAX_HOPS` = 10 hops, so the whole visited list is about 4 bytes beside a
512-byte payload. The same idea runs the Internet's inter-domain routing: a BGP
route carries its AS_PATH and a router discards any route whose path already
contains its own AS number.

A packet that still has a link but whose every reachable neighbour is already on
its path has nowhere legal to go. It is dropped, with its own reason:

| Reason | Meaning |
|---|---|
| `no_route` | No candidate at all. Greedy's progress condition found nothing closer, or the drone had no link. |
| `dead_end` | Candidates existed, but the loop guard excluded every one of them. |

The two are kept apart because they say different things: `no_route` is a
property of the layout and the rule, `dead_end` is the price of the guard.

`config.LOOP_GUARD` selects the rule (PROVISIONAL):

- `"path"` — the whole visited list. **The default.**
- `"previous"` — only the drone the packet just came from. This is what Phase 1a
  and Phase 1b up to CHECKPOINT 11 ran, and it is kept so those results stay
  reproducible. Under it a packet can cycle, and does: see section 6.

The GS is never excluded, under either setting. It is the destination, not a hop
to avoid.

Greedy is **unaffected** by the choice: it only ever hands a packet to a drone
strictly closer to the GS, so distance to the GS falls at every hop and it can
never return to a drone it has already visited. Its results are identical under
both settings, and `tests/test_loop_guard.py` checks that rather than assuming
it.

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
reset, so they do not all create on the same step. **200 ms is the working
load** for the next phases and is the default; the experiment still sweeps all
four.

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

**Tests** — 202 of them, about a minute:

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

**Everything at once.** After changing anything in `config.py`, rebuild every
figure and table from the new settings in one go. Each script runs in a fresh
process, and the run stops at the first failure or any `views_agree` mismatch:

```bash
python scripts/regenerate_all.py
```

Every figure footer and every CSV's first line carries a **settings stamp** —
the loss curve, the link cutoff and reach, the world, the queue limits, the
load, the placement count, the measurement window and the git commit — all
read from `config.py`. A figure or table found on its own can therefore be
traced back to the model version that produced it. CSVs written this way have
a leading `#` line, so read them with `scripts.stamp.read_stamped_csv`.

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
| `REQUIRE_CONNECTED_M` | True | Only accept layouts where every M-drone can reach the GS. | DECIDED |
| `MAX_PLACEMENT_DRAWS` | 10000 | Give up rather than loop forever looking for one. | — |
| `QUEUE_CAPACITY` | 10 | Packets per **link** queue. | DECIDED |
| `MAX_TX_PER_STEP` | 1 | Packets each **link** queue sends per step. | DECIDED |
| `PT_DBM` | 30.0 dBm | Transmit power (1 W). | DECIDED |
| `GT_DBI`, `GR_DBI` | 2.0 dBi | Antenna gains. | DECIDED |
| `F_HZ` | 2.4 GHz | Carrier frequency. | DECIDED |
| `RX_SENSITIVITY_DBM` | −54.0 dBm | Receiver sensitivity. | DECIDED |
| `SENS_LOSS` | 0.10 | Loss **at** the sensitivity point (802.11 OFDM's 10 % PER). | DECIDED |
| `LOSS_50_DISTANCE_M` | 350.0 m | Distance of 50 % loss, which is the link edge. A design choice, chosen from geometry. | DECIDED |
| `LINK_MAX_LOSS` | 0.5 | A link exists while loss is under this (→ 350 m). | DECIDED |
| `LOOP_GUARD` | `"path"` | Which previously visited drones a packet may not be sent back to: `"path"` (all of them) or `"previous"` (only the last). Applies to every routing rule. | PROVISIONAL |
| `PACKET_TTL` | 50 steps | Packet lifetime. | DECIDED |
| `MAX_HOPS` | 10 | Most hops a packet may take. | DECIDED |
| `PACKET_INTERVAL_STEPS` | 2 (200 ms) | Steps between packets at each M-drone; the working load. | DECIDED |
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

`channel.max_link_distance()` derives the 350 m link reach from the loss
curve, so there is no separate range constant to keep in step.

The radio constants and the speed range come from IQMR (Sharvari et al., 2024,
arXiv:2408.09109); the 10 % sensitivity definition follows 802.11 OFDM. The
arena size, the 18 + 7 fleet split, the link edge and the 2-D simplification
are **our** choices.

---

## 4. Outputs

Everything written by the scripts lands in `rl_sim/out/`. The figures and the
small result tables are committed, so the numbers quoted here can be checked
without rerunning anything; the big per-run CSVs are gitignored and regenerate
from the commands above.

| File | Contents |
|---|---|
| `ploss_curve.png` | The channel loss curve against distance, with the receiver sensitivity and the link edge marked and the link margin on a second axis. |
| `edge_scan.csv` | What each candidate link edge (300–450 m) does to the geometry over 3000 random layouts, with the connected-layout filter off: share of connected layouts, share of M-drones with a direct link to the GS, mean neighbours, mean minimum hops, and the loss at 200 m and 300 m. |
| `1a_runs.csv` | *(gitignored)* One row per experiment run (160): seeds, rule, load, created/delivered, total loss and its six causes as shares of created, pooled per-link loss and its three parts, queue occupancy, delay, hops, isolated M-drone count (0 under the filter), drones in GS range, how many layout draws the seed needed, and whether the two views agreed. |
| `1a_summary.csv` | Mean and standard deviation of every numeric column, per (routing, load). |
| `1a_paired.csv` | Per load: mean and std of `greedy − random` total loss on identical seeds, and how many of the 20 placements greedy lost less on. |
| `1a_total_loss_vs_load.png` | Total packet loss against offered load, one line per rule, mean with std error bars. |
| `1a_loss_heatmaps.png` | Three panels: loss causes for greedy and for random, sharing a "% of packets created" scale, then per-link loss on its own "% of link attempts" scale. Mean % with std beneath. |
| `1a_link_matrix_placement1.png` | Per-link loss for placement 1: eight matrices (2 rules × 4 loads), senders as rows, receivers plus GS as columns, grey where a link was never used. |

| `range_check.csv` | Fleet connectivity at several link cutoffs over 100 unfiltered random layouts: reach, share of M-drones with a path to the GS, share of layouts stranding one, neighbour counts, good-link counts and hops on the most reliable path. |
| `1b_training_log.csv` | One row per training run (its total loss and BCE) and per evaluation (validation and train-check total loss, plus the validation loss by cause), with the reference rules measured on the same validation layouts. |
| `1b_eval_log.csv` | The same evaluations, one row per layout rather than per mean, flagged where a new best model was saved. |
| `1b_training_log_previous.csv`, `1b_eval_log_previous.csv` | The same two logs for the agent trained under `LOOP_GUARD = "previous"`, used only by the loop-guard comparison. |
| `1b_learning_curve.png` | Total loss against training runs — validation, train-check, the faint per-run training loss, and dashed reference lines; below it, the BCE per run. |
| `1b_learning_curve_seeds.png` | The same training with one line per init seed: validation with a standard-error band and a star at every saved checkpoint, train-check beneath it, and the per-run BCE with a moving average. |
| `1b_validation_causes.png` | Where the validation loss goes at each evaluation: channel, queue_full, no_route, dead_end, ttl+hop. |
| `1b_test50_runs.csv` | One row per (rule, load, layout, init seed) over the 50 TEST layouts: total loss, all six causes kept apart, and mean hops. |
| `1b_test50_paired.csv` | Paired comparisons on layouts 1–50, 1–20 and 21–50: mean difference, std, layouts won, sign-test and paired t-test p-values. |
| `1b_comparison_total_loss_50.png` | Total loss per rule on the 50 TEST layouts, every layout a dot, at both loads. |
| `1b_loss_heatmaps_50.png` | A panel per rule: total loss and all six causes, both loads, mean with std, on one shared scale. |
| `1b_loop_guard.csv` | One row per (loop guard, rule, load, layout): total loss, its six causes, the revisit share and mean hops, under both guards. |
| `1b_loop_guard_paired.csv` | Per-layout differences between rules under `"path"`. |
| `1b_loop_guard_by_guard.csv` | Paired `"path" − "previous"` per rule and load: mean difference, std, layouts won, and both p-values. |
| `1b_channel_diagnosis.csv` | Every rl and rate channel loss classified by what the deciding drone was offered — avoidable, congestion-forced or geometry-forced — plus the split of the avoidable ones by delivery rate. |
| `queue_size_runs.csv` | *(gitignored)* One row per Phase-1b run at each link-queue capacity Q = 5 / 10 / 20. |
| `queue_size_1a.csv` | The Phase-1a load curve at each Q. |
| `queue_size_paired.csv` | Paired `rl − rate` and `rl − greedy` at each Q and load, with both p-values. |
| `queue_size_total_loss.png` | Total loss against Q, one line per rule, one panel per load. |
| `queue_size_load_curve.png` | Greedy's loss against offered load, one line per Q. |
| `out/models/<guard>/` | *(gitignored)* The best-validation network per init seed, kept per loop-guard setting. |

`main.py` also writes a JSONL event log per episode under `logs/` (one record
per packet event, per-step network state and per-drone state).

---

## 5. Observed baseline behaviour

Over 160 runs — two rules, four loads, 20 placements, identical seeds for
both rules. The two views agreed in every run and no M-drone was ever
isolated. Mean % of the packets created:

| rule | load | total | channel | queue_full | no_route | dead_end | ttl | hop_limit |
|---|---|---|---|---|---|---|---|---|
| greedy | 1000 ms | 11.63 | 8.71 | 2.92 | 0.00 | 0.00 | 0.00 | 0.00 |
| greedy | 500 ms | 26.47 | 9.91 | 16.55 | 0.00 | 0.00 | 0.00 | 0.00 |
| greedy | 200 ms | 52.72 | 11.14 | 40.02 | 0.00 | 0.00 | 1.56 | 0.00 |
| greedy | 100 ms | 66.34 | 13.16 | 51.61 | 0.00 | 0.00 | 1.58 | 0.00 |
| random | 1000 ms | 74.38 | 60.95 | 0.00 | 0.00 | 4.29 | 0.00 | 9.14 |
| random | 500 ms | 74.45 | 60.98 | 0.00 | 0.00 | 4.32 | 0.00 | 9.15 |
| random | 200 ms | 74.51 | 60.98 | 0.00 | 0.00 | 4.23 | 0.00 | 9.31 |
| random | 100 ms | 74.33 | 60.44 | 0.47 | 0.00 | 4.17 | 0.00 | 9.25 |
<!-- source: rl_sim/out/1a_runs.csv -->

**Greedy's loss is congestion; random's is the radio.** Greedy sends toward
whichever qualifying neighbour has the lowest link loss, so traffic converges
on the drones nearest the GS and `queue_full` grows from 2.92 % at the
lightest load to 51.61 % at the heaviest. Random spreads traffic over every
neighbour, so its queues stay empty and its loss is almost all `channel`,
flat near 74 % whatever the load.

Paired on identical placements, greedy loses less at every load: −62.74
points at 1000 ms, −47.98 at 500 ms, −21.80 at 200 ms and −7.99 at 100 ms,
on 20, 20, 20 and 18 of the 20 placements.
<!-- source: rl_sim/out/1a_paired.csv -->

---

## 6. Phase 1b — the learned routing rules

Phase 1b adds two more rules beside greedy and random, in a new package
`rl_sim/agents/`. **PyTorch is allowed only there.** `fanet_sim/` stays
torch-free and never imports `agents/`: a router object is injected into
`FANETEnv` and called through five hooks. A test blocks both `torch` and
`agents` and imports the simulator to keep that true.

### What each drone learns

When a packet reaches the GS, an **end-to-end ACK walks back along its path**,
so every drone that forwarded it credits the link it used. A packet that is
not acknowledged by `created_at + TTL` is recorded as lost by every drone that
decided on it — each knows the creation step and the TTL from the packet
header, so that is a local decision, not an announcement. These ACKs are ideal
in this phase: instant, never lost, no capacity used.

From those outcomes each drone keeps a **per-link delivery rate**: the mean of
the last `DELIVERY_RATE_WINDOW` = 50 resolved packets it *sent* on that link,
starting at `1 − channel_loss` before anything has resolved. A decision refused
by a full queue becomes a training example but does **not** move the rate — it
was never sent.

> Decisions are stored as they are made, not reconstructed from `Packet.path`.
> The path records only hops that *succeeded*, since a send lost in the channel
> never reaches `relay_to`. For a lost packet the decision that matters most —
> the one whose transmission failed — is therefore absent from the path, and
> reading it back would discard exactly the examples the network most needs.

### The two rules

Both choose among the current neighbours plus the GS when in range, minus
whatever [the loop guard](#the-loop-guard) excludes — the whole visited path
under the default `LOOP_GUARD = "path"`. No candidate at all is a `no_route`
drop; candidates that the guard removes entirely is a `dead_end` drop.

**`rate`** — no network:

```
score = delivery_rate × (1 − queue_full)
```

Highest score wins; ties go to the lower channel loss. With probability
`EPSILON_RATE` = 0.05 it picks uniformly instead, so links it never chooses
still get measured. **That exploration is on at test time too**, so `rate`'s
reported numbers carry noise that greedy's do not.

**`rl`** — one small network, `3 → 32 → 32 → 1` with ReLU and a sigmoid output,
**shared by every drone**. It scores one option from three local inputs — that
link's delivery rate, the fill of the deciding drone's own queue for it, and
its channel loss — and estimates the chance a packet sent that way reaches the
GS. Highest output wins. While training it explores with probability 0.1; at
evaluation there is none, though the delivery rates keep updating, since those
are observations rather than learning.

Every decision becomes one training example once its packet resolves: target 1
if the GS acknowledged it, 0 if the deadline passed. Examples accumulate across
steps; once at least 32 have gathered, one Adam step is taken on their binary
cross-entropy. No replay buffer, no target network.

> **Assumption, pending supervisor confirmation.** Pooling every drone's
> examples into one shared network is our choice, not a given. Each drone still
> *decides* from its own local inputs only — nothing in a decision reads another
> drone's state — but the weights those decisions share are trained on
> everyone's experience at once.

### How it is trained and tested

All at the 200 ms working load, with the layouts split so nothing is measured
on what it was fitted to:

| Set | Seeds | Used for |
|---|---|---|
| training | 101–300, cycled | one layout per run, 1000 steps, learning on |
| validation | 301–310 | choosing the kept model; never trained on |
| train-check | 101–110 | the overfitting comparison only |
| test | 1–50 | used at the very end; seeds 1–20 are the original set |

Every 10 training runs the network is frozen and measured on the validation and
train-check layouts. The best validation model is kept under `out/models/`
(gitignored). Training stops after 300 runs, or early when validation has not
improved by 0.5 points over five evaluations. The whole thing is repeated for
network init seeds 0, 1 and 2.

Evaluations run in parallel — the network is frozen and the layouts are
independent — and a test asserts a worker pool returns exactly the sequential
numbers.

The test layouts are also measured at **500 ms**, a load nothing was trained on,
as a generalisation check.

```bash
cd rl_sim
python scripts/train_1b.py --init-seeds 0 1 2       # tens of minutes
python scripts/evaluate_1b_large.py                # the 50 held-back layouts
python scripts/plot_1b.py                          # learning curve + causes
```

`regenerate_all.py` runs `evaluate_1b_large.py` and `plot_1b.py` by default
and skips them when no trained model is present; `--with-training` retrains
first. `evaluate_1b.py` (20 layouts) is kept as a script but is not in that
plan: the 50-layout evaluation is the official test.

### Results

On the 50 held-back TEST layouts, with `rl` the mean of three init seeds.
The two views agreed in all 600 runs. Mean % of the packets created:

| load | rule | total | std | channel | queue_full | no_route | dead_end | ttl | hop_limit | hops |
|---|---|---|---|---|---|---|---|---|---|---|
| 200 ms | random | 74.17 | 2.77 | 61.85 | 0.00 | 0.00 | 3.36 | 0.00 | 8.96 | 4.32 |
| 200 ms | greedy | 52.76 | 6.38 | 9.81 | 41.36 | 0.00 | 0.00 | 1.59 | 0.00 | 3.40 |
| 200 ms | rate | 25.05 | 7.15 | 22.94 | 0.53 | 0.00 | 0.54 | 0.47 | 0.57 | 3.12 |
| 200 ms | **rl** | **22.29** | 9.59 | 18.48 | 0.03 | 0.00 | 1.03 | 1.48 | 1.27 | 3.46 |
| 500 ms | random | 74.28 | 2.95 | 61.94 | 0.00 | 0.00 | 3.41 | 0.00 | 8.93 | 4.32 |
| 500 ms | greedy | 27.42 | 9.55 | 8.04 | 19.34 | 0.00 | 0.00 | 0.05 | 0.00 | 3.84 |
| 500 ms | rate | 20.96 | 4.31 | 20.13 | 0.03 | 0.00 | 0.36 | 0.00 | 0.44 | 2.97 |
| 500 ms | **rl** | **12.56** | 5.69 | 11.20 | 0.00 | 0.00 | 0.67 | 0.02 | 0.67 | 3.32 |
<!-- source: rl_sim/out/1b_test50_runs.csv -->

Paired on the same 50 layouts. The sign test is the exact two-sided binomial
on the layouts won; the t-test is two-sided on the per-layout differences.

| load | comparison | mean diff | std | wins | p (sign) | p (t-test) |
|---|---|---|---|---|---|---|
| 200 ms | rl − rate | −2.76 | 3.15 | 41 / 50 | <0.0001 | <0.0001 |
| 200 ms | rl − greedy | −30.47 | 9.85 | 50 / 50 | <0.0001 | <0.0001 |
| 200 ms | rate − greedy | −27.71 | 8.09 | 50 / 50 | <0.0001 | <0.0001 |
| 200 ms | rate − random | −49.12 | 6.15 | 50 / 50 | <0.0001 | <0.0001 |
| 500 ms | rl − rate | −8.40 | 2.95 | 50 / 50 | <0.0001 | <0.0001 |
| 500 ms | rl − greedy | −14.86 | 11.52 | 45 / 50 | <0.0001 | <0.0001 |
| 500 ms | rate − greedy | −6.46 | 11.12 | 33 / 50 | 0.0328 | 0.0002 |
| 500 ms | rate − random | −53.32 | 3.94 | 50 / 50 | <0.0001 | <0.0001 |
<!-- source: rl_sim/out/1b_test50_paired.csv -->

**What the loop guard changed.** `rl` was trained separately under each
setting, so no model is evaluated under a guard it did not train with.
`"path"` loses fewer packets than `"previous"` for random (−4.53 points at
both loads), rate (−1.75 and −2.52) and rl (−2.14 and −3.48), all
p < 0.0001; greedy is identical under both on every layout. Under
`"previous"` the share of packets that revisit a drone is 43.0 % for random,
13.1 % for rl and 6.3 % for rate; under `"path"` it is zero by construction.
<!-- source: rl_sim/out/1b_loop_guard.csv, rl_sim/out/1b_loop_guard_by_guard.csv -->

**Where rl's channel losses come from.** Of its channel losses at 200 ms,
45.6 % were geometry-forced — nothing at least 10 points cleaner was allowed
— 7.1 % congestion-forced, and 47.3 % had a cleaner option whose queue was
under half full. Within that last group, 99.95 % were trade-offs where the
cleaner link had a worse measured delivery rate.
<!-- source: rl_sim/out/1b_channel_diagnosis.csv -->

**Queue size.** Re-running everything at link-queue capacities Q = 5, 10 and
20 leaves the ordering unchanged: rl beats rate at every Q, on 42, 41 and 35
of 50 layouts at 200 ms and 50, 50 and 48 at 500 ms.
<!-- source: rl_sim/out/queue_size_paired.csv -->

```bash
cd rl_sim
python scripts/train_1b.py --init-seeds 0 1 2     # tens of minutes
python scripts/evaluate_1b_large.py               # the 50 TEST layouts
python scripts/compare_loop_guard.py              # both guards side by side
python scripts/diagnose_1b_channel.py             # why the channel losses happen
python scripts/queue_size_check.py                # Q = 5, 10 and 20
```

### Seeding

The agent's exploration stream comes from its **own** `SeedSequence([run_seed,
tag])`, not from spawning a third stream off the environment's two. Spawning
would have shifted the channel stream and moved every Phase-1a result with it.
A test reproduces greedy and random at seed 1 against the committed
`out/1a_runs.csv` to 1e-12.

---

## 7. Assumptions and limitations

- **2-D.** No altitude. The real scenario is three-dimensional.
- **No MAC layer, no interference, no collisions.** Nothing contends for the
  medium, which is exactly why each link gets its own queue and its own
  per-step budget, and why a drone can send on all its links at once. A real
  radio could not.
- **The loss curve is a model, not a measurement.** Free-space path loss with
  no terrain, obstruction or fading, turned into an S-curve whose only fitted
  point is the 10 % sensitivity definition. The link edge is our design
  choice. Both drive the results directly.
- **Loss depends on distance alone.** Not on traffic, antenna orientation, or
  interference from other transmissions.
- **Ideal ACKs.** Instant, never lost, no capacity used, no retransmissions.
  Real ACKs would cost airtime and could themselves be lost.
- **Neighbor knowledge:** each drone knows its own position (GPS), the GS
  position, and its current neighbors' positions and link quality, as if
  learned from periodic hello messages and measured signal strength. The hello
  messages themselves are not simulated: they are free, never lost and always
  up to date. This matters little in the static Phase 1 and will be revisited
  for the dynamic Phase 2.
- **Layouts are filtered, so the results are conditional.** Only placements in
  which every M-drone can reach the GS are simulated. At the current 350 m
  edge about 5% of random layouts strand at least one M-drone
  (`out/edge_scan.csv`), and those layouts are excluded. Every number here
  therefore describes a *connected* deployment, not an arbitrary one.
- **The loop guard is a design assumption.** We assert that the header can
  carry the visited list and that every drone honours it. That is cheap (about
  4 bytes) and standard practice (BGP's AS_PATH), but it is still an assumption
  about the protocol, not a result, and `LOOP_GUARD` is PROVISIONAL. A real
  network would also have to cope with the header growing in the dynamic phase,
  where paths can be longer.
- **Energy is tracked but constrains nothing.** No drone ever runs out.
- **The Phase-1b network is shared, and trained on pooled experience.** Each
  drone decides from its own local inputs, but one set of weights is fitted to
  every drone's examples at once. That is an assumption, not a result.
- **`rate` explores at test time.** Its 5% exploration is on in evaluation as
  well as training, so its reported numbers carry noise that greedy's and
  `rl`'s do not.
- **The Phase-1b ACKs are ideal.** Instant, never lost, no capacity used — the
  return path costs nothing, which a real network could not offer.

---

## 8. Folder structure

```
.
├── CLAUDE.md               Project context and working rules for agent sessions.
├── README.md               This file.
├── requirements.txt        The third-party packages actually imported.
│                           torch is needed only by agents/ and the 1b scripts.
│
├── rl_sim/                 THE SIMULATOR — the only live code.
│   ├── main.py             Entry point: run one episode, print both views.
│   ├── fanet_sim/
│   │   ├── config.py       Every parameter, with DECIDED/PROVISIONAL status.
│   │   ├── envs/
│   │   │   ├── fanet_env.py  The environment: step order, routing rules,
│   │   │   │                 transmission, drops, ground-truth counters.
│   │   │   ├── drone.py      A drone: position, link queues, ACK counters.
│   │   │   ├── channel.py    The loss curve, link existence, link quality.
│   │   │   └── packet.py     Packet lifecycle and the six drop reasons.
│   │   └── utils/
│   │       ├── metrics.py       Connectivity graphs (networkx).
│   │       ├── event_log.py     JSONL event logger.
│   │       └── visualization.py Matplotlib animation of a run.
│   ├── scripts/
│   │   ├── metrics_1a.py     Both views of one run, and the check that they
│   │   │                     agree. Used by main.py and the experiment.
│   │   ├── experiment_1a.py  The 160-run grid, its CSVs and figures.
│   │   ├── range_check.py    Connectivity at candidate link cutoffs.
│   │   ├── plot_ploss.py     The channel-loss curve.
│   │   ├── edge_scan.py      What each candidate link edge does to the
│   │   │                     geometry; report only, changes nothing.
│   │   ├── train_1b.py       Phase-1b training protocol.
│   │   ├── evaluate_1b.py    Phase 1b on the 20 held-back TEST layouts.
│   │   ├── evaluate_1b_large.py
│   │   │                     The same on 50 layouts, with the sign test and
│   │   │                     paired t-test, and the heatmaps.
│   │   ├── compare_loop_guard.py
│   │   │                     Every rule under both LOOP_GUARD settings.
│   │   ├── diagnose_1b*.py   Read-only diagnostics: what the policy chose,
│   │   │                     the packet's state at each decision, and why
│   │   │                     the channel losses happened.
│   │   ├── plot_1b.py        The learning curve and validation causes.
│   │   ├── regenerate_all.py Rebuild every output from the current config.
│   │   ├── stamp.py          The settings stamp on every figure and CSV.
│   │   └── analyze.py        Older stand-alone analyser for the JSONL logs.
│   │                         Nothing in the Phase-1a flow calls it.
│   ├── agents/             PHASE 1B. The only place torch is allowed.
│   │   ├── tracker.py        Delivery rates, decisions, ACKs and deadlines.
│   │   ├── rate_router.py    The shared base, and the "rate" rule.
│   │   └── rl_router.py      The network, the "rl" rule, and its learning.
│   ├── tests/              214 tests: channel, queues, routing, conservation,
│   │                       ACK-vs-ground-truth, layout filter, static world,
│   │                       reproducibility, the loop guard, and the
│   │                       Phase-1b agents.
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
