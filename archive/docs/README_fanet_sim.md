# FANET Simulator — Phase 1

Research simulator for a Flying Ad-hoc Network (FANET) built as the
foundation for a reinforcement-learning routing and topology-control project.

## Quick start

All commands below are run from the `rl_sim/` directory (this package's
parent), e.g. `cd rl_sim && python main.py --no-anim`.

```bash
pip install numpy matplotlib networkx pillow

# Headless run (fastest)
python main.py --no-anim

# Save animation to episode.gif
python main.py

# Interactive window
python main.py --show

# Q-routing baseline instead of greedy
python main.py --no-anim --routing qroute

# Override step count
python main.py --no-anim --steps 500
```

## PPO training

Two policies are trained with PPO (routing stays greedy):

* **K-link selection** (every drone) — picks which K neighbours to keep each
  step. Reward: +1 / −1 per packet from that drone delivered / dropped.
* **Topology movement** (C-drones only) — picks each C-drone's [dx, dy] move.
  Reward: the local topology reward (neighbours gained + link-quality gain −
  motion energy).

Each drone trains its own policy independently; weights persist across episodes
and are saved at the end. PDR is printed every episode so improvement is visible.

```bash
pip install torch

python train.py                            # config defaults
python train.py --episodes 100 --steps 400 # longer run
python train.py --save my_policies.pt      # custom checkpoint path
```

## Project structure

```
fanet_sim/
    config.py           All simulation parameters
    envs/
        drone.py        Drone class — position, energy, queue, get_state()
        channel.py      Link quality function (distance-based)
        packet.py       Packet lifecycle and PacketFactory
        fanet_env.py    Main environment: reset(), step(), routing baselines
    utils/
        metrics.py      PDR, delay, hops, connectivity, throughput …
        visualization.py  Matplotlib animation
main.py                 Entry point
```

## Key design decisions

* **Pure Python + NumPy** — no RL framework in phase 1.
* **PettingZoo-style interface** — `reset()` / `step(actions)` ready for MARL.
* **Two routing baselines** — greedy geographic and Q-routing (selectable via CLI).
* **All constants in config.py** — change once, affects everything.
