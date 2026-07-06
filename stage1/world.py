"""Topology sampling and communication-graph construction.

Node convention (default world): drones are 0..49 (M-drones 0..35,
C-drones 36..49) and the ground station is node 50. Every graph built here
stores ``graph.graph["gs_id"]`` and per-node ``pos`` / ``kind`` attributes
("M", "C" or "GS"), so the routers and the simulator never need to guess.

Seeding: ``sample_positions`` derives two independent RNG streams from the
topology seed — one for M-drone positions, one for C-drone positions — so
M-drone positions are identical across layouts for the same topology seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import networkx as nx
import numpy as np

from . import channel, config

Seed = Union[int, Sequence[int]]

# A link whose delivery probability (1 - p_loss) is below this floor can never
# carry a packet in practice, so no edge is created for it. This also makes
# edge existence at the range boundary (d ~= COMM_RANGE_M, where p_loss -> 1)
# immune to floating-point dust in recomputed distances.
_MIN_DELIVERY_PROB: float = 1e-12


def _ring_positions() -> np.ndarray:
    """C-drone ring: evenly spaced on a circle of RING_RADIUS_M around GS."""
    n = config.N_C_DRONES
    angles = 2.0 * math.pi * np.arange(n) / n
    gs = np.asarray(config.GS_POS)
    return gs + config.RING_RADIUS_M * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _grid_positions() -> np.ndarray:
    """C-drone lattice: staggered rows (GRID_ROW_SIZES) of cell centres."""
    rows = config.GRID_ROW_SIZES
    assert sum(rows) == config.N_C_DRONES, "GRID_ROW_SIZES must sum to N_C_DRONES"
    pts = []
    n_rows = len(rows)
    for i, row_size in enumerate(rows):
        y = config.AREA_SIZE_M * (i + 0.5) / n_rows
        for j in range(row_size):
            x = config.AREA_SIZE_M * (j + 0.5) / row_size
            pts.append((x, y))
    return np.asarray(pts, dtype=float)


def sample_positions(layout: str, topology_seed: Seed) -> np.ndarray:
    """Sample all node positions for one frozen topology.

    Returns an array of shape (N_DRONES + 1, 2): rows 0..35 are M-drones,
    36..49 are C-drones, and the last row is the GS at the arena centre.
    """
    if layout not in config.LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {config.LAYOUTS}")
    m_ss, c_ss = np.random.SeedSequence(topology_seed).spawn(2)
    rng_m = np.random.default_rng(m_ss)
    m_pos = rng_m.uniform(0.0, config.AREA_SIZE_M, size=(config.N_M_DRONES, 2))
    if layout == "ring":
        c_pos = _ring_positions()
    elif layout == "grid":
        c_pos = _grid_positions()
    else:  # random
        rng_c = np.random.default_rng(c_ss)
        c_pos = rng_c.uniform(0.0, config.AREA_SIZE_M, size=(config.N_C_DRONES, 2))
    gs = np.asarray([config.GS_POS], dtype=float)
    return np.concatenate([m_pos, c_pos, gs], axis=0)


def default_kinds(n_nodes: int) -> tuple[str, ...]:
    """Node kinds for an arbitrary position array: all "M" except a final "GS"."""
    return ("M",) * (n_nodes - 1) + ("GS",)


def candidate_graph(
    positions: np.ndarray,
    k: float,
    kinds: Optional[Sequence[str]] = None,
) -> nx.DiGraph:
    """Directed range graph: edge i->j iff d_ij <= COMM_RANGE_M and p_loss < 1.

    The hard communication range is COMM_RANGE_M (250 m): nodes farther
    apart than that have no link at all and can never exchange packets. A
    boundary link at exactly the range edge has p_loss = 1 (it could never
    deliver), so it is not created either — ring C-drones placed exactly at
    COMM_RANGE_M from the GS therefore have no direct GS edge.

    The last row of ``positions`` (or the node whose kind is "GS") is the
    ground station; it is a pure sink and never a source, so it has no
    outgoing edges. Edge attributes: ``dist``, ``p_loss`` and
    ``weight = -log(1 - p_loss)``.
    """
    n = len(positions)
    if kinds is None:
        kinds = default_kinds(n)
    if len(kinds) != n or kinds.count("GS") != 1:
        raise ValueError("kinds must match positions and contain exactly one GS")
    gs_id = kinds.index("GS")

    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=2))
    ploss = np.asarray(channel.p_loss(dists, k))

    g = nx.DiGraph(gs_id=gs_id, k=k)
    for i in range(n):
        g.add_node(i, pos=(float(positions[i, 0]), float(positions[i, 1])), kind=kinds[i])
    for i in range(n):
        if i == gs_id:
            continue  # GS never transmits
        for j in range(n):
            if j == i or dists[i, j] > config.COMM_RANGE_M:
                continue
            p = float(ploss[i, j])
            if p >= 1.0 - _MIN_DELIVERY_PROB:
                continue  # boundary link (d ~= COMM_RANGE_M): can never deliver
            g.add_edge(i, j, dist=float(dists[i, j]), p_loss=p, weight=-math.log1p(-p))
    return g


def prune_out_edges(g: nx.DiGraph, cap: int = config.MAX_OUT_EDGES) -> nx.DiGraph:
    """Keep each drone's ``cap`` lowest-p_loss drone->drone edges.

    Ties are broken by neighbour id (ascending). The drone->GS edge is
    exempt from the cap and always kept, so GS in-degree stays unbounded.
    Returns a new graph; the input is not modified.
    """
    gs_id = g.graph["gs_id"]
    pruned = nx.DiGraph(**g.graph)
    pruned.add_nodes_from(g.nodes(data=True))
    for u in g.nodes:
        if u == gs_id:
            continue
        to_gs = [(u, v, d) for v, d in g[u].items() if v == gs_id]
        others = sorted(
            ((u, v, d) for v, d in g[u].items() if v != gs_id),
            key=lambda e: (e[2]["p_loss"], e[1]),
        )
        for u_, v_, d_ in to_gs + others[:cap]:
            pruned.add_edge(u_, v_, **d_)
    return pruned


def drones_reaching_gs(g: nx.DiGraph) -> frozenset[int]:
    """Set of drones with at least one directed path to the GS."""
    return frozenset(nx.ancestors(g, g.graph["gs_id"]))


@dataclass(frozen=True)
class World:
    """One frozen topology plus its raw and pruned communication graphs."""

    layout: str
    k: float
    positions: np.ndarray
    raw_graph: nx.DiGraph        # range-only candidate graph (no out-edge cap)
    graph: nx.DiGraph            # after the per-drone out-edge prune
    prune_disconnected: tuple[int, ...]  # drones connected in raw_graph but not in graph


def build_world(layout: str, k: float, topology_seed: Seed) -> World:
    """Sample a topology and build its pruned communication graph."""
    positions = sample_positions(layout, topology_seed)
    kinds = ("M",) * config.N_M_DRONES + ("C",) * config.N_C_DRONES + ("GS",)
    raw = candidate_graph(positions, k, kinds)
    pruned = prune_out_edges(raw)
    lost = tuple(sorted(drones_reaching_gs(raw) - drones_reaching_gs(pruned)))
    return World(
        layout=layout,
        k=k,
        positions=positions,
        raw_graph=raw,
        graph=pruned,
        prune_disconnected=lost,
    )
