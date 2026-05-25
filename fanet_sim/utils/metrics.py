"""
metrics.py — Episode and per-timestep metric computation for the FANET simulator.

Metrics follow the definitions from IQMR (2024), Spectral RL, and DGATR:

    PDR             Packet delivery ratio = delivered / total_generated.
    avg_delay       Mean end-to-end delay in timesteps (delivered packets only).
    avg_hops        Mean hop count per delivered packet.
    drop_rate       Fraction of packets dropped before reaching the GS.
    avg_energy      Mean residual energy across all drones in joules.
    network_connected  Whether the drone graph is fully connected (bool).
    throughput      Delivered packets per simulation timestep.
    routing_load    Ratio of forwarded (relay) transmissions to original packets.
"""

from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING, Dict, List

import networkx as nx

from fanet_sim import config

if TYPE_CHECKING:
    from fanet_sim.envs.drone import Drone
    from fanet_sim.envs.fanet_env import FANETEnv
    from fanet_sim.envs.packet import Packet


# ---------------------------------------------------------------------------
# Graph connectivity
# ---------------------------------------------------------------------------

def build_network_graph(drones: List["Drone"]) -> nx.Graph:
    """Build a NetworkX graph of the current drone topology.

    Nodes are drone IDs; edges exist between drones that are within
    COMM_RANGE of each other.

    Args:
        drones: All Drone objects in the simulation.

    Returns:
        An undirected NetworkX Graph.
    """
    G = nx.Graph()
    G.add_nodes_from(d.drone_id for d in drones)
    for drone in drones:
        for nid in drone.neighbors:
            G.add_edge(drone.drone_id, nid)
    return G


def is_network_connected(drones: List["Drone"]) -> bool:
    """Return True if every drone can reach every other drone via relay links.

    Args:
        drones: All Drone objects in the simulation.

    Returns:
        True if the induced communication graph is connected.
    """
    G = build_network_graph(drones)
    return nx.is_connected(G) if len(G.nodes) > 0 else False


# ---------------------------------------------------------------------------
# Episode-level metrics
# ---------------------------------------------------------------------------

def compute_episode_metrics(env: "FANETEnv") -> Dict[str, object]:
    """Compute all standard metrics at the end of an episode.

    Args:
        env: The FANETEnv instance after running a full episode.

    Returns:
        A dict with the following keys:

        - ``PDR``               float  — packet delivery ratio.
        - ``avg_delay``         float  — mean delivery delay in timesteps.
        - ``avg_hops``          float  — mean hops per delivered packet.
        - ``drop_rate``         float  — fraction of packets dropped.
        - ``avg_energy``        float  — mean residual energy per drone (J).
        - ``network_connected`` bool   — True if graph is connected at episode end.
        - ``throughput``        float  — delivered packets per timestep.
        - ``routing_load``      float  — relay tx count / total packets generated.
        - ``total_generated``   int    — packets created this episode.
        - ``total_delivered``   int    — packets that reached the GS.
        - ``total_dropped``     int    — packets that were dropped.
    """
    total_generated = len(env.all_packets)
    total_delivered = len(env.delivered)
    total_dropped = len(env.dropped)

    pdr = total_delivered / total_generated if total_generated > 0 else 0.0
    drop_rate = total_dropped / total_generated if total_generated > 0 else 0.0

    delays = [p.delay() for p in env.delivered if p.delay() is not None]
    avg_delay = mean(delays) if delays else 0.0

    hops = [p.hop_count for p in env.delivered]
    avg_hops = mean(hops) if hops else 0.0

    energies = [d.energy for d in env.drones]
    avg_energy = mean(energies) if energies else 0.0

    connected = is_network_connected(env.drones)

    steps = max(env.step_count, 1)
    throughput = total_delivered / steps

    # routing_load: relay hops beyond the first hop count as "control" overhead
    total_relay_hops = sum(p.hop_count for p in env.all_packets)
    routing_load = total_relay_hops / total_generated if total_generated > 0 else 0.0

    return {
        "PDR": pdr,
        "avg_delay": avg_delay,
        "avg_hops": avg_hops,
        "drop_rate": drop_rate,
        "avg_energy": avg_energy,
        "network_connected": connected,
        "throughput": throughput,
        "routing_load": routing_load,
        "total_generated": total_generated,
        "total_delivered": total_delivered,
        "total_dropped": total_dropped,
    }


# ---------------------------------------------------------------------------
# Per-timestep snapshot
# ---------------------------------------------------------------------------

def compute_step_metrics(env: "FANETEnv") -> Dict[str, object]:
    """Compute a lightweight snapshot of metrics at the current timestep.

    Useful for live visualisation and debugging.

    Args:
        env: The FANETEnv instance mid-episode.

    Returns:
        A dict with:

        - ``step``          int   — current timestep index.
        - ``delivered``     int   — cumulative packets delivered so far.
        - ``dropped``       int   — cumulative packets dropped so far.
        - ``generated``     int   — cumulative packets generated so far.
        - ``PDR``           float — running PDR.
        - ``avg_delay``     float — running mean delay (delivered only).
        - ``connected``     bool  — current graph connectivity.
        - ``active_links``  int   — number of wireless links active this step.
    """
    gen = len(env.all_packets)
    dlv = len(env.delivered)
    drp = len(env.dropped)
    pdr = dlv / gen if gen > 0 else 0.0

    delays = [p.delay() for p in env.delivered if p.delay() is not None]
    avg_delay = mean(delays) if delays else 0.0

    return {
        "step": env.step_count,
        "delivered": dlv,
        "dropped": drp,
        "generated": gen,
        "PDR": pdr,
        "avg_delay": avg_delay,
        "connected": is_network_connected(env.drones),
        "active_links": len(env.active_links),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_episode_summary(metrics: Dict[str, object]) -> None:
    """Print a formatted summary of episode metrics to stdout.

    Args:
        metrics: Dict returned by compute_episode_metrics().
    """
    sep = "=" * 52
    print(sep)
    print("  FANET Episode Summary")
    print(sep)
    print(f"  Packets generated :  {metrics['total_generated']}")
    print(f"  Packets delivered :  {metrics['total_delivered']}")
    print(f"  Packets dropped   :  {metrics['total_dropped']}")
    print(sep)
    print(f"  PDR               :  {metrics['PDR']:.4f}")
    print(f"  Drop rate         :  {metrics['drop_rate']:.4f}")
    print(f"  Avg delay (steps) :  {metrics['avg_delay']:.2f}")
    print(f"  Avg hops          :  {metrics['avg_hops']:.2f}")
    print(f"  Throughput (pkt/s):  {metrics['throughput']:.4f}")
    print(f"  Routing load      :  {metrics['routing_load']:.4f}")
    print(f"  Avg residual E (J):  {metrics['avg_energy']:.1f}")
    print(f"  Network connected :  {metrics['network_connected']}")
    print(sep)
