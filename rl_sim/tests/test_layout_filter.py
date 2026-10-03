"""Layout-filter tests (Checkpoint 6, item 4).

Only placements in which EVERY M-drone has a path to the GS are used. A
rejected layout is discarded and another drawn from the same placement stream,
so a seed still names one specific accepted layout.
"""

from __future__ import annotations

import os

import networkx as nx
import pytest

from fanet_sim import config
from fanet_sim.envs.channel import are_connected
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.utils.metrics import build_graph_with_gs

from .conftest import run_episode

SEEDS = [1, 2, 3, 4, 5, 9, 17, 42]


def _env(placement_seed: int, run_seed: int = 1) -> FANETEnv:
    """Return a reset environment on one placement seed."""
    env = FANETEnv(
        routing="greedy",
        log_path=os.devnull,
        placement_seed=placement_seed,
        run_seed=run_seed,
    )
    env.reset()
    env.close_logger()
    return env


def _m_drones_reaching_gs(env: FANETEnv) -> int:
    """Count M-drones with any path to the GS, derived independently.

    Rebuilds the graph from raw positions with the FSPL link test, rather than
    reusing the env's own acceptance check.
    """
    graph = nx.Graph()
    graph.add_nodes_from(d.drone_id for d in env.drones)
    graph.add_node("GS")
    for i, first in enumerate(env.drones):
        if are_connected(first.position, env.gs_position):
            graph.add_edge(first.drone_id, "GS")
        for second in env.drones[i + 1:]:
            if are_connected(first.position, second.position):
                graph.add_edge(first.drone_id, second.drone_id)

    reachable = nx.node_connected_component(graph, "GS")
    return sum(
        1 for d in env.drones if d.drone_type == "M" and d.drone_id in reachable
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_every_accepted_layout_connects_all_m_drones(seed: int) -> None:
    """Every M-drone in an accepted layout has a path to the GS."""
    assert config.REQUIRE_CONNECTED_M, "this test assumes the filter is on"
    env = _env(seed)
    assert _m_drones_reaching_gs(env) == config.NUM_M_DRONES


@pytest.mark.parametrize("seed", SEEDS)
def test_no_drone_is_left_without_any_neighbour(seed: int) -> None:
    """A connected M-drone must have at least one link or a direct GS shot."""
    env = _env(seed)
    for drone in env.drones:
        if drone.drone_type != "M":
            continue
        has_link = bool(drone.neighbors) or are_connected(
            drone.position, env.gs_position
        )
        assert has_link, f"M-drone {drone.drone_id} has nowhere to send"


def test_the_filter_actually_rejects_some_layouts() -> None:
    """Guard: if no seed ever needed a second draw, the filter is untested."""
    draws = [_env(seed).placement_draws for seed in range(1, 21)]
    assert all(d >= 1 for d in draws)
    assert max(draws) > 1, "no layout was ever rejected — filter not exercised"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_filter_is_reproducible(seed: int) -> None:
    """Same placement seed: same accepted layout, same number of draws."""
    import numpy as np

    first = _env(seed)
    second = _env(seed)

    assert first.placement_draws == second.placement_draws
    assert np.array_equal(
        np.array([d.position for d in first.drones]),
        np.array([d.position for d in second.drones]),
    )
    assert [d.drone_type for d in first.drones] == [
        d.drone_type for d in second.drones
    ]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_accepted_layout_does_not_depend_on_the_run_seed(seed: int) -> None:
    """Changing only the run seed leaves the layout and the draw count alone."""
    import numpy as np

    first = _env(seed, run_seed=1)
    second = _env(seed, run_seed=99)

    assert first.placement_draws == second.placement_draws
    assert np.array_equal(
        np.array([d.position for d in first.drones]),
        np.array([d.position for d in second.drones]),
    )


def test_the_env_accepts_its_own_layout() -> None:
    """The env's acceptance check agrees with the independent graph rebuild."""
    for seed in SEEDS:
        env = _env(seed)
        assert env._all_m_reach_gs(env.drones)
        assert _m_drones_reaching_gs(env) == config.NUM_M_DRONES


def test_a_filtered_run_strands_no_packets_at_the_source() -> None:
    """With every M-drone connected, no packet dies at hop 0 for want of a link."""
    env = run_episode(steps=200, routing="greedy", placement_seed=2, run_seed=2).env

    for drone in env.drones:
        if drone.drone_type == "M":
            assert drone.neighbors or are_connected(
                drone.position, env.gs_position
            )


def test_turning_the_filter_off_restores_single_draw_placement() -> None:
    """With the flag off, the first layout drawn is the one used."""
    previous = config.REQUIRE_CONNECTED_M
    try:
        config.REQUIRE_CONNECTED_M = False
        env = _env(2)
        assert env.placement_draws == 1
    finally:
        config.REQUIRE_CONNECTED_M = previous
