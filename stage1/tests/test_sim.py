"""Queue mechanics: FIFO order, one tx/step, drop-not-requeue, PDR accounting."""
import math

import networkx as nx
import numpy as np
import pytest

from stage1 import metrics, sim, world
from stage1.routing import DijkstraRouter, routing_table


def make_graph(edges: dict, gs_id: int, kinds: dict) -> nx.DiGraph:
    """Tiny hand-built graph for deterministic sims (p_loss set directly)."""
    g = nx.DiGraph(gs_id=gs_id)
    for node, kind in kinds.items():
        g.add_node(node, kind=kind, pos=(0.0, 0.0))
    for (u, v), p in edges.items():
        w = -math.log1p(-p) if p < 1.0 else float("inf")
        g.add_edge(u, v, p_loss=p, weight=w, dist=0.0)
    return g


def relay_world():
    """Two sources feed one relay: S1(0), S2(1) -> R(2) -> GS(3), lossless."""
    g = make_graph(
        {(0, 2): 0.0, (1, 2): 0.0, (2, 3): 0.0},
        gs_id=3,
        kinds={0: "M", 1: "M", 2: "C", 3: "GS"},
    )
    return g, {0: 2, 1: 2, 2: 3}


def test_fifo_order_and_one_tx_per_step():
    g, hops = relay_world()
    res = sim.run_sim(g, hops, np.random.default_rng(0), n_steps=6)
    delivered = np.flatnonzero(res.delivered)
    # The relay receives 2 packets/step but forwards exactly one: deliveries
    # happen one per step from step 1, in emission order (FIFO; same-step
    # arrivals ordered by sender id).
    assert list(delivered) == [0, 1, 2, 3, 4]
    assert list(res.end_step[delivered]) == [1, 2, 3, 4, 5]


def test_delay_decomposes_into_hops_plus_queue_wait():
    g, hops = relay_world()
    res = sim.run_sim(g, hops, np.random.default_rng(0), n_steps=6)
    d = res.delivered
    assert np.array_equal(
        res.delay_steps[d], res.hops[d] + res.queue_wait_steps[d]
    )
    # Packet 0: emitted step 0, relayed step 0, delivered step 1 -> 2 hops, 0 wait.
    assert res.delay_steps[0] == 2 and res.hops[0] == 2 and res.queue_wait_steps[0] == 0
    # Packet 1 arrived the same step but queued one step behind packet 0.
    assert res.delay_steps[1] == 3 and res.queue_wait_steps[1] == 1


def test_relay_queue_grows_and_is_flagged_unstable():
    g, hops = relay_world()
    res = sim.run_sim(g, hops, np.random.default_rng(0), n_steps=50)
    r_col = res.drones.index(2)
    depths = res.queue_depths[:, r_col]
    assert depths[-1] > depths[0]  # +2 in, -1 out per step
    assert metrics.unstable_drones(res, window=20, slope_min=0.5) == (2,)
    # The sources drain every step and must not be flagged.
    assert 0 not in metrics.unstable_drones(res, window=20, slope_min=0.05)


def test_drop_not_requeue_on_lossy_link():
    # S(0) -> R(1) lossless, R -> GS(2) always fails: R drops one head-of-queue
    # packet per step, permanently, and its queue never grows.
    g = make_graph(
        {(0, 1): 0.0, (1, 2): 1.0}, gs_id=2, kinds={0: "M", 1: "C", 2: "GS"}
    )
    res = sim.run_sim(g, {0: 1, 1: 2}, np.random.default_rng(0), n_steps=5)
    dropped = res.status == sim.DROPPED_CHANNEL
    assert dropped.sum() == 4  # packets 0..3; packet 4 still in flight at R
    assert np.all(res.end_node[dropped] == 1)
    assert np.all(res.hops[dropped] == 1)  # the S->R hop succeeded, then died
    assert (res.status == sim.IN_FLIGHT).sum() == 1
    assert res.delivered.sum() == 0
    r_col = res.drones.index(1)
    assert np.all(res.queue_depths[:, r_col] <= 1)  # never re-queued


def test_no_route_drops_one_packet_per_step():
    g = make_graph({}, gs_id=1, kinds={0: "M", 1: "GS"})
    res = sim.run_sim(g, {0: None}, np.random.default_rng(0), n_steps=4)
    assert (res.status == sim.DROPPED_NO_ROUTE).sum() == 4
    assert np.all(res.end_node[res.resolved] == 0)


def test_pdr_excludes_in_flight_packets():
    # Chain S(0) -> R(1) -> GS(2), lossless, 3 steps: the packet emitted in
    # the last step is still at R and must not count against PDR.
    g = make_graph(
        {(0, 1): 0.0, (1, 2): 0.0}, gs_id=2, kinds={0: "M", 1: "C", 2: "GS"}
    )
    table = {0: 1, 1: 2}
    res = sim.run_sim(g, table, np.random.default_rng(0), n_steps=3)
    assert res.delivered.sum() == 2
    assert (res.status == sim.IN_FLIGHT).sum() == 1
    m = metrics.sim_metrics(res, g, table)
    assert m["n_in_flight"] == 1
    assert m["pdr_global"] == 1.0  # 2 delivered / 2 resolved


def test_global_vs_routed_pdr_split():
    # One routed source (lossless direct link) and one source with no route.
    g = make_graph({(0, 2): 0.0}, gs_id=2, kinds={0: "M", 1: "M", 2: "GS"})
    table = {0: 2, 1: None}
    res = sim.run_sim(g, table, np.random.default_rng(0), n_steps=2)
    m = metrics.sim_metrics(res, g, table)
    assert m["pdr_global"] == pytest.approx(0.5)
    assert m["pdr_routed"] == 1.0
    assert m["unreachable_frac_m"] == pytest.approx(0.5)
    assert m["drop_frac_at_m"] == 1.0


def test_emission_rate_one_packet_per_m_drone_per_step():
    g, hops = relay_world()
    res = sim.run_sim(g, hops, np.random.default_rng(0), n_steps=7)
    assert len(res.src) == 2 * 7  # only the two M-drones emit
    assert np.all(np.isin(res.src, [0, 1]))


def test_channel_seed_reproducible_and_varying():
    w = world.build_world("random", 0.6, 1)
    table = routing_table(DijkstraRouter(), w.graph)
    a = sim.run_sim(w.graph, table, np.random.default_rng(7), n_steps=100)
    b = sim.run_sim(w.graph, table, np.random.default_rng(7), n_steps=100)
    c = sim.run_sim(w.graph, table, np.random.default_rng(8), n_steps=100)
    assert np.array_equal(a.status, b.status)
    assert np.array_equal(a.end_step, b.end_step)
    assert not np.array_equal(a.status, c.status)


def test_delay_invariant_on_real_world():
    w = world.build_world("random", 0.6, 1)
    table = routing_table(DijkstraRouter(), w.graph)
    res = sim.run_sim(w.graph, table, np.random.default_rng(7), n_steps=100)
    d = res.delivered
    assert d.any()
    assert np.array_equal(res.delay_steps[d], res.hops[d] + res.queue_wait_steps[d])
    assert np.all(res.hops[d] >= 1)
    assert np.all(res.queue_wait_steps[d] >= 0)
