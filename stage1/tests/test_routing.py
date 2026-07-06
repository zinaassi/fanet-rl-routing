"""Routers: dijkstra composability, greedy no-progress rule, direct rule."""
import networkx as nx
import numpy as np
import pytest

from stage1 import config, world
from stage1.routing import (
    DijkstraRouter,
    DirectRouter,
    GreedyRouter,
    make_router,
    routed_drones,
    routing_table,
)


@pytest.fixture(scope="module")
def built_world():
    return world.build_world("random", 0.6, 123)


def test_dijkstra_first_hop_composability(built_world):
    """dist(u) == w(u, next_hop) + dist(next_hop): first hops compose into
    shortest paths, so per-drone decisions are globally consistent."""
    g = built_world.graph
    table = routing_table(DijkstraRouter(), g)
    dist = nx.single_source_dijkstra_path_length(
        g.reverse(copy=False), config.GS_ID, weight="weight"
    )
    reachable = [u for u, nh in table.items() if nh is not None]
    assert reachable, "fixture world should have routed drones"
    for u in reachable:
        nh = table[u]
        assert dist[u] == pytest.approx(g.edges[u, nh]["weight"] + dist[nh])


def test_dijkstra_chains_terminate_at_gs(built_world):
    g = built_world.graph
    table = routing_table(DijkstraRouter(), g)
    for u, nh in table.items():
        if nh is None:
            continue
        node, hops = u, 0
        while node != config.GS_ID:
            node = table[node]
            hops += 1
            assert hops <= len(g), f"loop in dijkstra chain from {u}"


def test_dijkstra_none_when_no_path():
    positions = np.array([[0.0, 0.0], [9000.0, 9000.0]])  # drone far from GS
    g = world.candidate_graph(positions, k=1.0)
    assert DijkstraRouter().next_hop(0, g) is None


def test_greedy_prefers_gs_when_direct_edge_exists():
    positions = np.array([[300.0, 0.0], [250.0, 0.0], [0.0, 0.0]])
    g = world.candidate_graph(positions, k=0.3)
    assert g.has_edge(0, 2)
    assert GreedyRouter().next_hop(0, g) == 2  # GS is at distance 0 of itself


def test_greedy_picks_neighbor_closest_to_gs():
    # Drone 0 at 1000 m (beyond the k=0.3 direct range); neighbours at
    # 500 m and 700 m from the GS: greedy must pick the 500 m one.
    positions = np.array([[1000.0, 0.0], [500.0, 0.0], [700.0, 0.0], [0.0, 0.0]])
    g = world.candidate_graph(positions, k=0.3)
    assert not g.has_edge(0, 3)
    assert GreedyRouter().next_hop(0, g) == 1


def test_greedy_drop_on_no_progress():
    # Drone 0's only in-range neighbour is farther from the GS than itself.
    positions = np.array([[800.0, 0.0], [1000.0, 0.0], [0.0, 0.0]])
    g = world.candidate_graph(positions, k=1.0)
    assert g.has_edge(0, 1)  # neighbour is in range...
    assert GreedyRouter().next_hop(0, g) is None  # ...but offers no progress


def test_direct_router_rule():
    positions = np.array([[300.0, 0.0], [2000.0, 0.0], [0.0, 0.0]])
    g = world.candidate_graph(positions, k=0.3)
    router = DirectRouter()
    assert router.next_hop(0, g) == 2
    assert router.next_hop(1, g) is None


def test_make_router_names():
    for name in config.ROUTERS:
        assert make_router(name).name == name


def test_routed_drones_follows_chains():
    g = nx.DiGraph(gs_id=0)
    g.add_nodes_from(range(5))
    table = {1: 2, 2: None, 3: 0, 4: 3}
    assert routed_drones(table, g) == {3, 4}


def test_routed_drones_handles_cycles():
    g = nx.DiGraph(gs_id=0)
    g.add_nodes_from(range(4))
    table = {1: 2, 2: 1, 3: 0}
    assert routed_drones(table, g) == {3}
