# CLAUDE.md — FANET Phase 1A

## Project goal

Minimize per-link packet loss in a drone network that relays data to a single
ground station (GS). The fleet is M-drones (create AND relay packets) plus
C-drones (relay only). Phase 1A is the simulator with non-learned routing on a
static world — no learning of any kind.

See `README.md` for the design, the config parameters, and the DECIDED /
PROVISIONAL status of every value.

## IGNORE / LEAVE BEHIND

We keep ONLY the simulator: world, drones, placement/mobility, channel,
packets, queues, logging, metrics, analysis. Everything below is out of scope
— do not use it, extend it, or rely on it:

- PPO K-link (link selection) policy and PPO topology policy
- `train.py`, `eval.py`, any training/reward code
- Q-routing
- any LSTM / GNN / trajectory-prediction code or plans
- any `stage1/` folder or Dijkstra-vs-greedy study code and its results
- the old `CLAUDE.md`, `PROJECT_STATE.md` and the old README: outdated
- anything from earlier sessions not restated in a current prompt

None of it is deleted. It all lives under `archive/` (see `README.md` for the
layout). Nothing in `rl_sim/` imports from `archive/`, and the live simulator
does not depend on torch — keep both of those true.

## RULES

0. Never commit to `main`. Work on a branch; commit per approved checkpoint.
1. **Read before writing.** Before changing anything, read the code that
   actually runs (entry point, config, env, channel, packet, metrics,
   analysis). Do not trust docs or earlier session summaries.
2. **Never guess.** If something is unclear, or the code differs from the
   prompt in a way the prompt does not cover, STOP and ask. Do not quietly
   pick an answer.
3. **No scope creep.** Build only what was asked: no extra features, sweeps,
   plots or refactors.
4. Values marked [DECIDED] are fixed. Values marked [PROVISIONAL] are
   temporary defaults and carry a `PROVISIONAL - to confirm` comment in
   config, so they stay easy to change.
5. **List edits and wait for OK.** Before editing any file, list what you will
   change in it and why, then wait for approval.
