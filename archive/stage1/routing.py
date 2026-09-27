"""One-shot routers, all sharing the interface ``next_hop(drone_id, graph)``.

A router returns the fixed next hop a drone will use for the whole episode,
or ``None`` if the drone has no usable hop (it then drops its traffic —
"unreachable" for that router). Routing is computed once at t=0 on the
frozen pruned graph.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import networkx as nx


NextHops = Dict[int, Optional[int]]


def _dist_to_gs(graph: nx.DiGraph, node: int) -> float:
    gx, gy = graph.nodes[graph.graph["gs_id"]]["pos"]
    x, y = graph.nodes[node]["pos"]
    return math.hypot(x - gx, y - gy)


class DirectRouter:
    """GS if the drone has a direct edge to it, else None."""

    name = "direct"

    def next_hop(self, drone_id: int, graph: nx.DiGraph) -> Optional[int]:
        gs_id = graph.graph["gs_id"]
        return gs_id if graph.has_edge(drone_id, gs_id) else None


class GreedyRouter:
    """Forward to the out-neighbour geometrically closest to the GS.

    Drop-on-no-progress: returns None unless some neighbour is strictly
    closer to the GS than the drone itself (the GS counts as distance 0,
    so a direct edge always wins). Ties break by neighbour id.
    """

    name = "greedy"

    def next_hop(self, drone_id: int, graph: nx.DiGraph) -> Optional[int]:
        own = _dist_to_gs(graph, drone_id)
        best: Optional[tuple[float, int]] = None
        for nbr in graph.successors(drone_id):
            d = _dist_to_gs(graph, nbr)
            if d < own and (best is None or (d, nbr) < best):
                best = (d, nbr)
        return None if best is None else best[1]


class DijkstraRouter:
    """First hop of the minimum-weight (= max delivery probability) path to GS.

    Implemented as ONE Dijkstra run from the GS on the reversed graph; the
    resulting shortest-path tree is memoised per graph object, so the
    per-drone ``next_hop`` interface stays O(1) after the first call.
    """

    name = "dijkstra"

    def __init__(self) -> None:
        self._graph: Optional[nx.DiGraph] = None
        self._table: NextHops = {}

    def next_hop(self, drone_id: int, graph: nx.DiGraph) -> Optional[int]:
        if graph is not self._graph:
            self._table = self._compute_table(graph)
            self._graph = graph
        return self._table.get(drone_id)

    @staticmethod
    def _compute_table(graph: nx.DiGraph) -> NextHops:
        gs_id = graph.graph["gs_id"]
        reversed_view = graph.reverse(copy=False)
        _, paths = nx.single_source_dijkstra(reversed_view, gs_id, weight="weight")
        # paths[u] is [gs, ..., u] in the reversed graph, i.e. the reversed
        # route u -> ... -> gs; u's first hop is therefore paths[u][-2].
        table: NextHops = {
            u: p[-2] for u, p in paths.items() if u != gs_id and len(p) >= 2
        }
        _assert_loop_free(table, gs_id, len(graph))
        return table


def _assert_loop_free(table: NextHops, gs_id: int, n_nodes: int) -> None:
    """Every next-hop chain must reach the GS within n_nodes hops."""
    for start in table:
        node, hops = start, 0
        while node != gs_id:
            assert hops <= n_nodes and node in table, (
                f"routing loop or dead end starting at drone {start}"
            )
            node, hops = table[node], hops + 1  # type: ignore[assignment]


ROUTERS = {r.name: r for r in (DirectRouter, GreedyRouter, DijkstraRouter)}


def make_router(name: str):
    """Instantiate a router by name ('direct' | 'greedy' | 'dijkstra')."""
    return ROUTERS[name]()


def routing_table(router, graph: nx.DiGraph) -> NextHops:
    """Evaluate ``router.next_hop`` for every drone in the graph."""
    gs_id = graph.graph["gs_id"]
    return {u: router.next_hop(u, graph) for u in graph.nodes if u != gs_id}


def routed_drones(next_hops: NextHops, graph: nx.DiGraph) -> frozenset[int]:
    """Drones whose next-hop chain terminates at the GS.

    A drone whose chain hits a None hop (or a cycle) is unreachable for the
    router that produced the table, even if it has a next hop itself.
    """
    gs_id = graph.graph["gs_id"]
    routed: set[int] = set()
    for start in next_hops:
        chain = []
        node: Optional[int] = start
        seen: set[int] = set()
        while node is not None and node != gs_id and node not in seen and node not in routed:
            seen.add(node)
            chain.append(node)
            node = next_hops.get(node)
        if node == gs_id or node in routed:
            routed.update(chain)
    return frozenset(routed)
