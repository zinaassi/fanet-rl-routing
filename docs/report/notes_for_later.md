# Notes for later

Things noticed while writing the report that belong in a section written
after Phase 2. One line each, with where it came from. Nothing here is
written into the report yet.

## Difficulties we hit

- The first channel model (free-space path loss, receiver sensitivity
  -54 dBm, hard 249.69 m range, loss `exp(-k*M)` with k = 0.8) reached
  ~100% loss right at its own range edge, which measured UAV links do not
  show. Replaced at Checkpoint 7. <!-- source: rl_sim/fanet_sim/config.py lines 130-142; commit bd7cb3e -->
- When the logistic curve was introduced, the link cutoff was 0.99, giving a
  ~539.8 m reach as a side effect rather than a choice. The network became
  very dense (~15 neighbours per drone) and RANDOM was handed links losing
  over 90% of what was sent on them. Changed to 0.5 at Checkpoint 7b. <!-- source: rl_sim/fanet_sim/config.py lines 144-154; commit bc0bcaf -->
- A stale figure was once committed beside a fresh table after a config
  change. This is why every figure and CSV now carries a settings stamp and
  why `regenerate_all.py` exists. <!-- source: rl_sim/scripts/stamp.py; rl_sim/scripts/regenerate_all.py -->
- `Packet.path` records only the hops that succeeded, so the drone whose
  transmission failed never appears in it. Reconstructing who decided on a
  packet from its path would silently discard exactly the failures the
  network most needs to learn from. Decisions are stored separately in the
  tracker instead. <!-- source: rl_sim/agents/tracker.py; CLAUDE.md "CRITICAL implementation note" -->
- Adding the agent's exploration stream as a third spawn off the
  environment's seed would have shifted the channel stream and moved every
  Phase-1a result. It uses its own `SeedSequence` instead. <!-- source: rl_sim/agents/rate_router.py line 58 -->

## Ideas for further work

- Retransmissions, or a milder channel instead. Most of rl's remaining loss
  is channel loss on links it had no cleaner alternative to. Both are model
  changes and the supervisor's call. <!-- source: CLAUDE.md open questions; rl_sim/scripts/diagnose_1b_channel.py output -->
- Whether the Rosati curve already includes MAC-layer retries; if it does,
  adding retransmissions on top would double-count them. <!-- source: CLAUDE.md open questions -->
- A hop count as a 4th network input (deferred, optional). <!-- source: CLAUDE.md open questions -->
- A robustness check on denser layouts (more drones, or a smaller arena). <!-- source: CLAUDE.md open questions -->
- A near-full load, e.g. 90 ms. <!-- source: CLAUDE.md open questions -->
- Pooling every drone's training examples into one shared network is an
  assumption, not a result; each drone still decides from local inputs only. <!-- source: CLAUDE.md open questions -->

## Background topics to cover

- FANET routing families and where greedy geographic routing sits among
  them. [CITE: a FANET routing survey]
- The measured UAV link study the loss curve is fitted to: Rosati et al.,
  "Dynamic Routing for Flying Ad Hoc Networks", arXiv:1406.4399. <!-- source: rl_sim/fanet_sim/config.py lines 125-132 -->
- The drone speed range comes from IQMR (Sharvari et al., 2024,
  arXiv:2408.09109); unused while the world is static. <!-- source: rl_sim/fanet_sim/config.py line 42 -->
- BGP's AS_PATH as the standard precedent for a path-carried loop guard. <!-- source: CLAUDE.md loop guard section -->

## Checkpoint dates and milestones

- Phase 1a baselines finished through Checkpoint 7b. <!-- source: commit bc0bcaf -->
- Phase 1b RL routing agent, Checkpoints 8-11. <!-- source: commits c0cc4c3, 0187e94 -->
- Path loop guard, Checkpoint 12. <!-- source: commit 18f1c63 -->
- 50-layout test set with significance tests, Checkpoint 13. <!-- source: commit 963021e -->
- Queue-size robustness check and per-seed learning curves. <!-- source: commit add16da -->

## Things to double-check before submitting

- `AREA_SIZE`, the energy constants and the mobility constants are present
  in config but unused while the world is static and energy constrains
  nothing. Decide whether the report mentions them at all. <!-- source: rl_sim/fanet_sim/config.py lines 32, 79-83, 225-226 -->
- The report's parameter table marks PROVISIONAL values as modelling
  choices; keep it in step with config.py if any are confirmed.
