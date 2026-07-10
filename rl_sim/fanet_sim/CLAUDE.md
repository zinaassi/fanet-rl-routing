# FANET Simulator — Project Context for Claude Code

## What this project is

This is a research simulator for a Flying Ad-hoc Network (FANET) built for an undergraduate
research project at the Technion. The goal is to simulate a fleet of drones that must route
data packets back to a ground station through multi-hop communication. The network topology
changes constantly because drones are moving. The simulator is the foundation for later work
on reinforcement learning based routing and topology control.

The simulator must be built first, before any RL or GNN work begins. The supervisor
confirmed this order explicitly.

---

## Project structure

```
fanet_sim/
    envs/
        fanet_env.py        # main environment class
        drone.py            # drone agent (position, energy, links)
        channel.py          # wireless link model
        packet.py           # packet generation and routing
    utils/
        metrics.py          # PDR, delay, hop count, connectivity
        visualization.py    # matplotlib animation of the network
    config.py               # all simulation parameters in one place
    main.py                 # entry point to run a simulation episode
    README.md
```

---

## Drone types

There are two types of drones. This distinction is central to the entire project.

**M-drones (Mission drones)**
- Follow pre-planned waypoint paths. They do not choose where to fly.
- Carry out the actual mission (sensing, imaging). This is not simulated — just their movement.
- Also participate in relaying packets to the ground station.
- Generate packets at regular intervals that must reach the ground station.

**C-drones (Communication drones)**
- Have no mission. Their only job is to improve network connectivity.
- In the simulator (phase 1), they follow scripted paths, just like M-drones.
- In later phases, a topology RL agent will control their movement.
- Do not generate packets themselves.

Both types relay packets for other drones.

---

## Simulation parameters (from literary review)

These values come from IQMR (2024) and related FANET papers. Use these as defaults in config.py.
All parameters must be easy to change from config.py without touching other files.

```python
# Space
AREA_SIZE = 1000          # radius in meters (3D sphere), start with 2D square for simplicity
USE_3D = False            # start False, extend later
WIDTH = 1000              # meters (2D mode)
HEIGHT = 1000             # meters (2D mode)

# Drones
NUM_M_DRONES = 6          # mission drones
NUM_C_DRONES = 3          # communication drones
DRONE_SPEED_MIN = 10      # m/s
DRONE_SPEED_MAX = 30      # m/s
COMM_RANGE = 250          # meters — two drones can communicate if closer than this
TX_POWER = 1.0            # watts (uniform across all drones)

# Energy
INITIAL_ENERGY = 207792   # joules (11.1V * 5200mAh, from IQMR)
ENERGY_PER_MOVE = 0.1     # joules per meter flown (approximate)
ENERGY_PER_TX = 0.01      # joules per packet transmitted

# Packets
PACKET_RATE = 1           # packets generated per drone per timestep
PACKET_SIZE = 512         # bytes
MAX_HOPS = 10             # drop packet if it exceeds this hop count
PACKET_TTL = 50           # timesteps before packet expires

# Simulation
TIMESTEP = 0.1            # seconds per step
MAX_STEPS = 1000          # steps per episode
NUM_EPISODES = 1          # for testing

# Ground station
GS_POSITION = (500, 500)  # center of the area in 2D mode
```

---

## Node features (state vector per drone)

These are the features each drone must expose at every timestep. They come directly from the
literary review of IQMR, DGATR, Spectral RL, and topology control papers. These will later
become the input to the GNN and RL agents.

```python
drone.get_state() -> dict with:
    position          # (x, y) or (x, y, z) — current coordinates
    velocity          # (vx, vy) — current velocity vector
    distance_to_gs    # float — euclidean distance to ground station
    residual_energy   # float — remaining energy in joules
    queue_length      # int — number of packets currently buffered
    num_neighbors     # int — number of drones within comm range right now
    neighbor_ids      # list of drone IDs within comm range
    neighbor_positions    # list of (x,y) for each neighbor
    neighbor_distances    # list of distances to each neighbor
    link_quality      # dict {neighbor_id: float 0-1} — quality per link
    is_connected_to_gs    # bool — can this drone reach GS in one hop?
```

Link quality is computed from distance: quality = 1 - (distance / COMM_RANGE).
A link exists only if quality > 0, i.e., distance < COMM_RANGE.

---

## Mobility models

**M-drones:** Waypoint model. Each M-drone gets a list of (x, y) waypoints at the start of
the episode. It flies toward the current waypoint at constant speed. When it arrives, it moves
to the next waypoint. Waypoints wrap around (cyclic). Waypoints are randomly generated inside
the area at episode start, or can be provided explicitly for reproducibility.

**C-drones (phase 1):** Same waypoint model as M-drones, with randomly generated waypoints.
In later phases, their movement will be replaced by the topology RL agent's action output.

**Movement update per timestep:**
```
direction = normalize(target_waypoint - current_position)
new_position = current_position + direction * speed * TIMESTEP
energy -= ENERGY_PER_MOVE * distance_moved
```

---

## Channel model

Simple distance-based model. Two drones are connected if their euclidean distance is less
than COMM_RANGE. Link quality degrades linearly with distance.

```python
def link_quality(dist, comm_range):
    if dist >= comm_range:
        return 0.0
    return 1.0 - (dist / comm_range)
```

No path loss exponents or noise models in phase 1. These can be added later.

---

## Routing (baseline — greedy geographic routing)

The simulator ships with a greedy geographic routing baseline. This is NOT the RL agent.
It is the classical baseline that the RL agent will later need to outperform.

Greedy geographic routing: at each drone, forward the packet to whichever neighbor is
closest to the ground station (in euclidean distance). If no neighbor is closer to the GS
than the current drone, drop the packet (local maximum / void).

```python
def greedy_next_hop(drone, neighbors, gs_position):
    best = min(neighbors, key=lambda n: distance(n.position, gs_position), default=None)
    if best and distance(best.position, gs_position) < distance(drone.position, gs_position):
        return best
    return None  # void — packet dropped
```

Also implement Q-routing as a second baseline (simple Q-table, state = current drone ID,
action = next hop neighbor ID, reward = -delay).

---

## Packet lifecycle

1. At each timestep, each M-drone generates PACKET_RATE new packets destined for the GS.
2. Each drone maintains a queue (list) of packets to forward.
3. At each timestep, each drone attempts to forward its queued packets to the next hop.
4. A packet is delivered when it reaches the GS.
5. A packet is dropped if: TTL expires, hop count exceeds MAX_HOPS, or no valid next hop.
6. Track each packet: creation time, creation drone, hops taken, delivery time or drop reason.

---

## Metrics to compute (from literary review)

All metrics must be computed at the end of each episode and optionally per timestep.
These are the standard metrics reported by IQMR, Spectral RL, and DGATR.

```python
metrics = {
    "PDR": delivered / total_generated,           # packet delivery ratio (most important)
    "avg_delay": mean(delivery_time - creation_time),   # end-to-end delay in timesteps
    "avg_hops": mean(hop_counts),                 # average hops per delivered packet
    "drop_rate": dropped / total_generated,       # fraction dropped
    "avg_energy": mean(residual_energy per drone),# energy remaining
    "network_connected": is_graph_connected(),    # bool — are all drones reachable?
    "throughput": delivered_per_timestep,         # packets per timestep
    "routing_load": control_packets / data_packets,  # overhead ratio
}
```

---

## Visualization

The simulator must produce a matplotlib animation showing:
- Drone positions as colored dots (blue for M-drones, orange for C-drones, green star for GS)
- Active wireless links as thin gray lines between connected drones
- Packet transmissions as brief flashes on links
- A live metric panel showing PDR and delay updating each timestep

This is essential for debugging and for showing the supervisor.

```python
# example call
from utils.visualization import FANETVisualizer
vis = FANETVisualizer(env)
vis.animate(save_path="episode.gif")  # or vis.show() for interactive
```

---

## What must work at the end of phase 1

1. Run one full episode with N drones moving on waypoint paths.
2. Packets are generated by M-drones and routed to GS using greedy geographic routing.
3. All metrics are computed and printed at the end of the episode.
4. A matplotlib visualization shows the network evolving over time.
5. Config values can be changed and the simulation reruns correctly.
6. The environment exposes a step() function compatible with future RL integration.

The step() function signature must follow this interface so it can be wrapped by a MARL
framework later (PettingZoo style):

```python
env.reset() -> observations: dict[drone_id, state_dict]
env.step(actions: dict[drone_id, action]) -> (observations, rewards, dones, infos)
```

For now, actions are ignored and greedy routing runs internally. The interface is just
prepared for the RL agent to plug in later.

---

## What to avoid in phase 1

- Do not implement RL agents yet. Placeholder interfaces only.
- Do not implement GNN yet.
- Do not implement the topology RL agent for C-drones yet. They move on scripted paths.
- Do not use any external MARL frameworks yet (PettingZoo, RLlib). Pure Python + NumPy only.
- Do not over-engineer. The goal is a working, readable, extensible simulator.

---

## Tech stack

- Python 3.10+
- NumPy for all math and state computation
- Matplotlib for visualization and animation
- NetworkX for graph connectivity checks (is_connected, shortest path)
- No PyTorch, no gym, no external RL libraries in phase 1

Install:
```bash
pip install numpy matplotlib networkx
```

---

## Code style

- Every function must have a clear docstring explaining inputs and outputs.
- No magic numbers. Every constant goes in config.py.
- Each class in its own file.
- Keep functions short. If a function is more than 30 lines, split it.
- Use type hints throughout.
- Write simple, readable code. This is a research project, not production software.

---

## Context for future phases

After the simulator is working, the project will add:

**Phase 2:** Add a GNN layer that reads the network graph and produces state embeddings.
The GNN input will be the node feature vectors defined above. Output is a richer embedding
per drone fed to the RL policy.

**Phase 3:** Add a routing RL agent (MAPPO or MADDPG) that replaces greedy routing.
The agent uses the GNN embeddings as its state and outputs next-hop decisions.

**Phase 4:** Add a topology RL agent that controls C-drone positions.
Reward for topology agent: network connectivity (Fiedler value or simpler proxy),
load balance across links, reduction in routing voids.

The simulator must be designed so these phases can be added without rewriting the core.
