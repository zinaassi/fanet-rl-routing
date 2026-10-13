# Notes for later

Things noticed while writing the report that belong in a section written
after Phase 2. One line each, with where it came from. Nothing here is
written into the report yet.

## URGENT

- **docs/report/ is STALE: written with the old channel; rewrite after the
  Phase 1 re-runs.** <!-- source: the commit replacing the channel, 2026-10-13 -->

## Caveats to state in the write-up

- The step-4a RL training runs (250 of them) were NOT checked for
  views_agree: `run_once` computed it but `train_one_seed` never read it,
  so only the 530 frozen evaluation and reference runs were inspected. The
  assertion was added to `train_one_seed` afterwards, so any FUTURE
  training is checked, but the models trained at that step were not. <!-- source: rl_sim/scripts/train_1b.py; commit fe94b73 -->

## Difficulties we hit

- The channel model was changed twice. An earlier FSPL loss `exp(-0.8*M)`
  put 100% loss at the receiver sensitivity, a wall at 250 m. A curve in
  distance replaced it, and was itself replaced on 2026-10-13 by the current
  S-curve on the FSPL margin, where the sensitivity means 10% loss (802.11
  OFDM). Every result from the first two models was deleted. <!-- source: rl_sim/fanet_sim/config.py; rl_sim/fanet_sim/envs/channel.py -->
- The link edge is a design choice, not a physical constant, and had to be
  chosen from geometry alone before any routing experiment so that it could
  not be tuned to a result. <!-- source: rl_sim/out/edge_scan.csv; CLAUDE.md -->
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
- Whether the S-curve already absorbs MAC-layer retries; if it does, adding
  retransmissions on top would double-count them. <!-- source: CLAUDE.md open questions -->
- A hop count as a 4th network input (deferred, optional). <!-- source: CLAUDE.md open questions -->
- A robustness check on denser layouts (more drones, or a smaller arena). <!-- source: CLAUDE.md open questions -->
- A near-full load, e.g. 90 ms. <!-- source: CLAUDE.md open questions -->
- Pooling every drone's training examples into one shared network is an
  assumption, not a result; each drone still decides from local inputs only. <!-- source: CLAUDE.md open questions -->

## Background topics to cover

- FANET routing families and where greedy geographic routing sits among
  them. [CITE: a FANET routing survey]
- 802.11 OFDM defines receiver sensitivity as the power giving a 10% packet
  error rate on 1000-octet frames, which is what anchors the S-curve.
  [CITE: the 802.11 clause defining receiver minimum input sensitivity] <!-- source: rl_sim/fanet_sim/config.py -->
- The radio constants (1 W, 2.4 GHz, -54 dBm) and the drone speed range come
  from IQMR (Sharvari et al., 2024, arXiv:2408.09109); the speed range is
  unused while the world is static. <!-- source: rl_sim/fanet_sim/config.py -->
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
