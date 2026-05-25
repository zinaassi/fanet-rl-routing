# FANET Simulator — Phase 1

Research simulator for a Flying Ad-hoc Network (FANET) built as the
foundation for a reinforcement-learning routing and topology-control project.

## Quick start

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
