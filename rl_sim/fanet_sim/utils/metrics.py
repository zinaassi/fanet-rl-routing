"""
metrics.py — Live, in-loop metric helpers ONLY.

Per the metrics & logging spec, all reported episode metrics (PDR, delay,
connectivity, energy, etc.) are computed in Stage 2 by scripts/analyze.py
from the raw event log. This module is restricted to:

  * Building the current communication graph (used by the visualiser and by
    the per-step connectivity sample written to the log).
  * Computing the graded connectivity sample (frac_connected_to_gs and
    num_components) that the env logs once per step.
  * A small live HUD snapshot for the visualiser — explicitly NOT the source
    of any reported metric.

Do NOT add episode aggregation back into this module. New metrics belong
in scripts/analyze.py so they can be added without re-running simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import networkx as nx
import numpy as np

from fanet_sim import config
from fanet_sim.envs.channel import are_connected, euclidean_distance

if TYPE_CHECKING:
    from fanet_sim.envs.drone import Drone
    from fanet_sim.envs.fanet_env import FANETEnv


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_network_graph(drones: List["Drone"]) -> nx.Graph:
    """Build the current undirected communication graph (drones only).

    Args:
        drones: All Drone objects in the simulation.

    Returns:
        Graph with drone IDs as nodes and an edge between any two drones
        whose neighbour sets include each other.
    """
    G = nx.Graph()
    G.add_nodes_from(d.drone_id for d in drones)
    for drone in drones:
        for nid in drone.neighbors:
            G.add_edge(drone.drone_id, nid)
    return G


def build_graph_with_gs(
    drones: List["Drone"],
    gs_position: np.ndarray,
) -> Tuple[nx.Graph, str]:
    """Build the communication graph including the ground station as a node.

    A drone is linked to the GS node if the FSPL received-signal test in
    :func:`fanet_sim.envs.channel.are_connected` passes.

    Args:
        drones:      All Drone objects.
        gs_position: Ground-station position.

    Returns:
        (graph, gs_node_label) — the graph with one extra node for the GS
        and the label used for it (so callers do not collide with int IDs).
    """
    gs_label = "GS"
    G = build_network_graph(drones)
    G.add_node(gs_label)
    for d in drones:
        if are_connected(d.position, gs_position):
            G.add_edge(d.drone_id, gs_label)
    return G, gs_label


# ---------------------------------------------------------------------------
# Graded connectivity (logged per-step by the env)
# ---------------------------------------------------------------------------

def connectivity_sample(
    drones: List["Drone"],
    gs_position: np.ndarray,
) -> Tuple[float, int]:
    """Return (frac_connected_to_gs, num_components) for the current step.

    ``frac_connected_to_gs`` is the fraction of drones with *any* multi-hop
    path to the ground station. This is the graded measure required by §B.3
    of the metrics spec — do NOT collapse it to a binary flag.

    Args:
        drones:      All Drone objects.
        gs_position: Ground-station position.

    Returns:
        (fraction in [0, 1], number of connected components in the drone-only
        graph).
    """
    if not drones:
        return 0.0, 0

    G, gs = build_graph_with_gs(drones, gs_position)
    if gs in G:
        reachable = nx.node_connected_component(G, gs)
        connected = sum(1 for d in drones if d.drone_id in reachable)
    else:
        connected = 0
    frac = connected / len(drones)

    drone_only = build_network_graph(drones)
    num_components = nx.number_connected_components(drone_only)
    return frac, num_components


# ---------------------------------------------------------------------------
# Live HUD snapshot (visualiser only — NOT for reporting)
# ---------------------------------------------------------------------------

def live_hud(env: "FANETEnv") -> Dict[str, object]:
    """Return a lightweight snapshot for live display in the visualiser.

    These numbers are intentionally cheap and approximate. Reported metrics
    must come from the Stage 2 analyser, not from this function.

    Args:
        env: The FANETEnv instance mid-episode.

    Returns:
        Dict with: ``step``, ``delivered``, ``dropped``, ``generated``,
        ``running_pdr``, ``active_links``, ``frac_connected_to_gs``.
    """
    frac, _ = connectivity_sample(env.drones, env.gs_position)
    gen = len(env.all_packets)
    dlv = len(env.delivered)
    # Delivered-only running delay (None when no deliveries yet — never
    # substitute 0 or infinity; that is the bug §B.2 of the spec warns about).
    delays = [p.delay() for p in env.delivered if p.delay() is not None]
    running_delay = sum(delays) / len(delays) if delays else None
    return {
        "step": env.step_count,
        "delivered": dlv,
        "dropped": len(env.dropped),
        "generated": gen,
        "running_pdr": (dlv / gen) if gen > 0 else 0.0,
        "running_avg_delay": running_delay,
        "active_links": len(env.active_links),
        "frac_connected_to_gs": frac,
    }
