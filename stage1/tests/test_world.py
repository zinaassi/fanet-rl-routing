"""Topology sampling and graph construction: cutoff edges, pruning, GS exemption."""
import math

import numpy as np
import pytest

from stage1 import channel, config, world


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------------------------------------------------------- positions

def test_positions_shape_bounds_and_gs():
    for layout in config.LAYOUTS:
        pos = world.sample_positions(layout, 3)
        assert pos.shape == (config.N_DRONES + 1, 2)
        assert np.all(pos >= 0.0) and np.all(pos <= config.AREA_SIZE_M)
        assert tuple(pos[config.GS_ID]) == config.GS_POS


def test_m_positions_identical_across_layouts():
    ring = world.sample_positions("ring", 11)
    grid = world.sample_positions("grid", 11)
    rand = world.sample_positions("random", 11)
    m = slice(0, config.N_M_DRONES)
    assert np.array_equal(ring[m], grid[m])
    assert np.array_equal(ring[m], rand[m])


def test_topology_seed_reproducible_and_distinct():
    a = world.sample_positions("random", 5)
    b = world.sample_positions("random", 5)
    c = world.sample_positions("random", 6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_ring_layout_geometry():
    pos = world.sample_positions("ring", 0)
    for c in pos[config.N_M_DRONES : config.N_DRONES]:
        assert _dist(c, config.GS_POS) == pytest.approx(config.RING_RADIUS_M)


def test_grid_layout_is_deterministic_lattice():
    a = world.sample_positions("grid", 1)
    b = world.sample_positions("grid", 2)
    c_slice = slice(config.N_M_DRONES, config.N_DRONES)
    assert np.array_equal(a[c_slice], b[c_slice])
    assert len({tuple(p) for p in a[c_slice]}) == config.N_C_DRONES


def test_random_c_positions_resampled_per_seed():
    c_slice = slice(config.N_M_DRONES, config.N_DRONES)
    a = world.sample_positions("random", 1)[c_slice]
    b = world.sample_positions("random", 2)[c_slice]
    assert not np.array_equal(a, b)


# ------------------------------------------------------------- range edges

def test_gs_has_no_outgoing_edges():
    w = world.build_world("random", 8.0, 0)
    assert w.raw_graph.out_degree(config.GS_ID) == 0
    assert w.graph.out_degree(config.GS_ID) == 0


def test_range_limit_removes_edges():
    # One drone within the hard 250 m range of the GS, one beyond it.
    positions = np.array([[100.0, 0.0], [600.0, 0.0], [0.0, 0.0]])
    g = world.candidate_graph(positions, k=8.0)
    assert g.has_edge(0, 2)
    assert not g.has_edge(1, 2)
    assert not g.has_edge(0, 1)  # drone-drone edges obey the same range


def test_no_link_at_exactly_comm_range():
    # At exactly COMM_RANGE_M the loss probability is 1 (the link could never
    # deliver a packet), so candidate_graph refuses to create it.
    at_edge = np.array([[config.COMM_RANGE_M, 0.0], [0.0, 0.0]])
    assert not world.candidate_graph(at_edge, k=8.0).has_edge(0, 1)
    inside = np.array([[config.COMM_RANGE_M - 1.0, 0.0], [0.0, 0.0]])
    g = world.candidate_graph(inside, k=8.0)
    assert g.has_edge(0, 1)
    assert g.edges[0, 1]["p_loss"] < 1.0


def test_candidate_edges_match_range_rule_exactly():
    w = world.build_world("random", 8.0, 4)
    pos = w.positions
    for i in range(config.N_DRONES):
        for j in range(config.N_DRONES + 1):
            if i == j:
                continue
            d = _dist(pos[i], pos[j])
            pl = float(channel.p_loss(d, 8.0))
            expected = d <= config.COMM_RANGE_M and pl < 1.0 - world._MIN_DELIVERY_PROB
            assert w.raw_graph.has_edge(i, j) == expected


def test_edge_weight_formula():
    w = world.build_world("random", 8.0, 4)
    for _, _, data in w.graph.edges(data=True):
        assert data["weight"] == pytest.approx(-math.log1p(-data["p_loss"]))


# ----------------------------------------------------------------- pruning

def _dense_cluster():
    """12 drones packed in a 150 m box (all mutually in range) plus a GS at
    the box centre: every drone sees 11 drone neighbours, so the 5-edge cap
    must bind."""
    rng = np.random.default_rng(0)
    drones = rng.uniform(0.0, 150.0, size=(12, 2))
    positions = np.vstack([drones, [[75.0, 75.0]]])
    raw = world.candidate_graph(positions, k=8.0)
    return raw, world.prune_out_edges(raw)


def test_prune_caps_drone_to_drone_out_degree():
    raw, pruned = _dense_cluster()
    gs = raw.graph["gs_id"]
    for u in range(12):
        assert raw.out_degree(u) == 12  # 11 drones + GS: dense by construction
        non_gs = [v for v in pruned.successors(u) if v != gs]
        assert len(non_gs) == config.MAX_OUT_EDGES


def test_prune_cap_holds_on_real_world():
    w = world.build_world("random", 8.0, 2)
    for u in range(config.N_DRONES):
        non_gs = [v for v in w.graph.successors(u) if v != config.GS_ID]
        assert len(non_gs) <= config.MAX_OUT_EDGES


def test_prune_keeps_lowest_ploss_edges():
    raw, pruned = _dense_cluster()
    gs = raw.graph["gs_id"]
    for u in range(12):
        kept = {v for v in pruned.successors(u) if v != gs}
        all_edges = sorted((d["p_loss"], v) for v, d in raw[u].items() if v != gs)
        expected = {v for _, v in all_edges[: config.MAX_OUT_EDGES]}
        assert kept == expected


def test_gs_edge_exempt_from_cap():
    # Drone 0 sits 200 m from the GS with six much closer drone neighbours.
    # Without the exemption the GS edge would lose the top-5 contest.
    d0 = np.array([200.0, 0.0])
    nbrs = [d0 + [10.0 + i, 5.0] for i in range(6)]
    positions = np.array([d0] + nbrs + [[0.0, 0.0]])
    pruned = world.prune_out_edges(world.candidate_graph(positions, k=8.0))
    gs = len(positions) - 1
    assert pruned.has_edge(0, gs)
    assert pruned.out_degree(0) == config.MAX_OUT_EDGES + 1


def test_prune_tie_break_by_neighbor_id():
    # Seven neighbours at exactly 100 m (exact in float): keep ids 1..5.
    offsets = [(100, 0), (-100, 0), (0, 100), (0, -100), (60, 80), (80, 60), (-60, 80)]
    positions = np.array(
        [[0.0, 0.0]] + [[float(x), float(y)] for x, y in offsets] + [[9000.0, 9000.0]]
    )
    pruned = world.prune_out_edges(world.candidate_graph(positions, k=8.0))
    assert set(pruned.successors(0)) == {1, 2, 3, 4, 5}


def test_prune_disconnect_detection():
    # Six mutually-close drones whose only way out is a "bridge" drone; the
    # 5-edge cap fills up with fellow cluster members and severs the bridge.
    cluster_center = np.array([400.0, 0.0])
    offsets = [(0, 0), (20, 0), (-20, 0), (0, 20), (0, -20), (15, 15)]
    cluster = [cluster_center + off for off in offsets]
    positions = np.array(cluster + [[200.0, 0.0], [0.0, 0.0]])  # bridge=6, GS=7
    raw = world.candidate_graph(positions, k=8.0)
    pruned = world.prune_out_edges(raw)
    lost = world.drones_reaching_gs(raw) - world.drones_reaching_gs(pruned)
    assert lost == {0, 1, 2, 3, 4, 5}
    assert 6 in world.drones_reaching_gs(pruned)  # the bridge itself survives


def test_build_world_reports_prune_disconnects():
    w = world.build_world("random", 8.0, 9)
    lost = world.drones_reaching_gs(w.raw_graph) - world.drones_reaching_gs(w.graph)
    assert w.prune_disconnected == tuple(sorted(lost))


def test_ring_c_drones_have_no_direct_gs_link():
    # Ring radius == COMM_RANGE_M: C-drones sit exactly at the GS's range
    # edge (p_loss = 1), so they get no direct GS edge and can only relay.
    w = world.build_world("ring", 8.0, 0)
    for c in range(config.N_M_DRONES, config.N_DRONES):
        assert not w.raw_graph.has_edge(c, config.GS_ID)
