"""restricted_pdr: shared-population delivery ratio for router comparison."""
import math

import networkx as nx
import numpy as np
import pytest

from stage1 import metrics, sim
from stage1.routing import GreedyRouter, DijkstraRouter, routing_table


def _add_edge(g, u, v, p_loss):
    w = -math.log1p(-p_loss) if p_loss < 1.0 else float("inf")
    g.add_edge(u, v, p_loss=p_loss, weight=w, dist=0.0)


def _greedy_void_graph():
    """X (M-drone) is graph-reachable but sits in a greedy void.

    Geometry (GS at origin): X at 300 m; its ONLY out-neighbour Y is at
    350 m — farther from the GS — so greedy (drop-on-no-progress) strands X.
    A real path X -> Y -> W -> GS exists (W at 150 m), so X IS
    graph-reachable and Dijkstra routes it. Y, W are C-drones.
    """
    gs = 3
    g = nx.DiGraph(gs_id=gs)
    g.add_node(0, kind="M", pos=(300.0, 0.0))   # X: the stranded M-drone
    g.add_node(1, kind="C", pos=(350.0, 0.0))   # Y: farther from GS than X
    g.add_node(2, kind="C", pos=(150.0, 0.0))   # W: closer, relays to GS
    g.add_node(gs, kind="GS", pos=(0.0, 0.0))
    _add_edge(g, 0, 1, 0.0)   # X -> Y (only exit from X; Y is farther)
    _add_edge(g, 1, 2, 0.0)   # Y -> W
    _add_edge(g, 2, gs, 0.0)  # W -> GS
    return g


def test_restricted_pdr_penalizes_stranded_reachable_drone():
    g = _greedy_void_graph()

    # graph_reachable is router-INDEPENDENT: same population either way.
    reach = metrics.reachable_m_drones(g)
    assert reach == frozenset({0})  # only M-drone, and it has a path to GS

    greedy_tbl = routing_table(GreedyRouter(), g)
    dijkstra_tbl = routing_table(DijkstraRouter(), g)
    assert greedy_tbl[0] is None          # greedy strands the reachable drone
    assert dijkstra_tbl[0] == 1           # dijkstra routes X -> Y -> W -> GS

    greedy_res = sim.run_sim(g, greedy_tbl, np.random.default_rng(0),
                             n_steps=20, max_tx_per_step=None)
    dijkstra_res = sim.run_sim(g, dijkstra_tbl, np.random.default_rng(0),
                               n_steps=20, max_tx_per_step=None)

    rp_greedy = metrics.restricted_pdr(greedy_res, g)
    rp_dijkstra = metrics.restricted_pdr(dijkstra_res, g)

    # Same denominator population; greedy delivers 0 of X's packets.
    assert rp_greedy == 0.0
    assert rp_dijkstra > rp_greedy  # dijkstra is rewarded for routing X


def test_restricted_pdr_equals_global_pdr_when_all_reachable_and_routed():
    """With every M-drone reachable AND routed AND no in-flight packets,
    restricted_pdr reduces to the ordinary global PDR."""
    gs = 2
    g = nx.DiGraph(gs_id=gs)
    g.add_node(0, kind="M", pos=(100.0, 0.0))
    g.add_node(1, kind="M", pos=(0.0, 100.0))
    g.add_node(gs, kind="GS", pos=(0.0, 0.0))
    _add_edge(g, 0, gs, 0.3)   # lossy but DIRECT: every packet resolves the
    _add_edge(g, 1, gs, 0.3)   # same step it is emitted -> zero in-flight

    table = {0: gs, 1: gs}
    assert metrics.reachable_m_drones(g) == frozenset({0, 1})

    res = sim.run_sim(g, table, np.random.default_rng(5),
                      n_steps=200, max_tx_per_step=None)
    assert (res.status == sim.IN_FLIGHT).sum() == 0  # precondition: all resolved

    m = metrics.sim_metrics(res, g, table)
    assert metrics.restricted_pdr(res, g) == pytest.approx(m["pdr_global"])
    assert 0.0 < metrics.restricted_pdr(res, g) < 1.0  # non-degenerate


def test_restricted_pdr_nan_when_no_reachable_source():
    """Fully disconnected sources -> undefined (NaN), not a divide error."""
    gs = 1
    g = nx.DiGraph(gs_id=gs)
    g.add_node(0, kind="M", pos=(9000.0, 0.0))  # no edges at all
    g.add_node(gs, kind="GS", pos=(0.0, 0.0))
    assert metrics.reachable_m_drones(g) == frozenset()
    res = sim.run_sim(g, {0: None}, np.random.default_rng(0), n_steps=5)
    assert math.isnan(metrics.restricted_pdr(res, g))
