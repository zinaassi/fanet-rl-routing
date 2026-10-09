"""Option sets, the loop guard, inputs and determinism (Phase 1b)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.channel import euclidean_distance
from fanet_sim.envs.fanet_env import FANETEnv, ROUTER_RULES

from agents import make_router
from agents.rate_router import BaseRouter, agent_rng

RULES = ["rate", "rl"]


def _env(rule: str, seed: int = 4, training: bool = False) -> FANETEnv:
    """Return a reset environment running *rule*."""
    router = make_router(rule, run_seed=seed, training=training, init_seed=0)
    env = FANETEnv(routing=rule, log_path=os.devnull,
                   placement_seed=seed, run_seed=seed, router=router)
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Options and the loop guard
# ---------------------------------------------------------------------------

def test_options_drop_the_drone_the_packet_came_from() -> None:
    """The loop guard removes exactly one key, and keeps the order fixed."""
    candidates = {5: np.zeros(2), 2: np.zeros(2), 9: np.zeros(2),
                  "GS": np.zeros(2)}
    assert BaseRouter.options(candidates, came_from=None) == [2, 5, 9, "GS"]
    assert BaseRouter.options(candidates, came_from=5) == [2, 9, "GS"]
    assert BaseRouter.options(candidates, came_from=99) == [2, 5, 9, "GS"]


def test_options_are_empty_when_the_only_link_is_the_one_it_came_from() -> None:
    """A dead-end chain leaves nothing, which the caller drops as no_route."""
    assert BaseRouter.options({7: np.zeros(2)}, came_from=7) == []


def test_the_gs_is_never_excluded_by_the_loop_guard() -> None:
    """The GS never sends, so it can never be where a packet came from."""
    candidates = {3: np.zeros(2), "GS": np.zeros(2)}
    assert "GS" in BaseRouter.options(candidates, came_from=3)


@pytest.mark.parametrize("rule", RULES)
def test_choices_are_always_current_options(rule: str) -> None:
    """Over a run, every chosen hop is a current neighbour or the in-range GS."""
    env = _env(rule)
    checked = 0
    for _ in range(80):
        env.step()
        for drone_id, next_key in env.tx_events:
            drone = env.get_drone_by_id(drone_id)
            assert next_key == "GS" or next_key in drone.neighbors
            checked += 1
    env.close_logger()
    assert checked > 0, "no transmission happened — test proved nothing"


@pytest.mark.parametrize("rule", RULES)
def test_a_packet_is_never_sent_straight_back(rule: str) -> None:
    """No packet's path contains the same pair twice in a row (A->B->A)."""
    env = _env(rule)
    for _ in range(200):
        env.step()
    env.close_logger()

    bounced = [p for p in env.all_packets
               if any(p.path[i] == p.path[i + 2] for i in range(len(p.path) - 2))]
    assert not bounced, f"{len(bounced)} packets were bounced straight back"


# ---------------------------------------------------------------------------
# The three local inputs
# ---------------------------------------------------------------------------

def test_features_are_exactly_the_three_local_values() -> None:
    """delivery_rate, own queue fill, channel loss — derived, not guessed."""
    env = _env("rl")
    router = env.router
    drone = next(d for d in env.drones if d.neighbors)
    key = sorted(drone.neighbors)[0]
    position = drone.neighbors[key].position

    # Put three packets on that link so the fill is non-zero.
    from fanet_sim.envs.packet import PacketFactory
    factory = PacketFactory(ttl=config.PACKET_TTL,
                            max_hops=config.MAX_HOPS, size_bytes=512)
    for _ in range(3):
        drone.enqueue(key, factory.create(source_id=0, created_at=0))

    delivery_rate, queue_fill, channel_loss = router.features(drone, key, position)

    expected_loss = channel.p_loss(euclidean_distance(drone.position, position))
    assert channel_loss == pytest.approx(expected_loss)
    assert queue_fill == pytest.approx(3 / config.QUEUE_CAPACITY)
    assert delivery_rate == pytest.approx(1.0 - expected_loss)  # no outcomes yet
    env.close_logger()


def test_features_read_only_the_deciding_drones_own_rates() -> None:
    """Another drone's measurements on the same link are not visible."""
    env = _env("rl")
    router = env.router
    drone = next(d for d in env.drones if d.neighbors)
    key = sorted(drone.neighbors)[0]
    position = drone.neighbors[key].position

    other = next(d for d in env.drones if d.drone_id != drone.drone_id)
    for _ in range(config.DELIVERY_RATE_WINDOW):
        router.tracker._record_outcome(other.drone_id, key, False)

    delivery_rate, _, channel_loss = router.features(drone, key, position)
    assert delivery_rate == pytest.approx(1.0 - channel_loss), (
        "the deciding drone saw another drone's outcomes"
    )
    env.close_logger()


# ---------------------------------------------------------------------------
# Determinism and seeding
# ---------------------------------------------------------------------------

def test_the_agent_stream_does_not_disturb_the_simulator_streams() -> None:
    """Building an agent stream leaves the env's two streams byte-identical."""
    plain = FANETEnv(routing="greedy", log_path=os.devnull,
                     placement_seed=3, run_seed=3)
    plain.reset()
    before_chan = [plain._rng_chan.random() for _ in range(5)]
    before_route = [plain._rng_route.random() for _ in range(5)]
    plain.close_logger()

    agent_rng(3)   # the agent's own SeedSequence, not a third spawn

    again = FANETEnv(routing="greedy", log_path=os.devnull,
                     placement_seed=3, run_seed=3)
    again.reset()
    assert [again._rng_chan.random() for _ in range(5)] == before_chan
    assert [again._rng_route.random() for _ in range(5)] == before_route
    again.close_logger()


@pytest.mark.parametrize("rule", RULES)
def test_same_seeds_reproduce_the_run(rule: str) -> None:
    """With evaluation settings, a rule replays exactly."""
    results = []
    for _ in range(2):
        env = _env(rule, seed=6)
        for _ in range(150):
            env.step()
        env.close_logger()
        results.append((
            len(env.delivered), len(env.dropped),
            [p.path for p in env.all_packets],
        ))
    assert results[0] == results[1]


def test_rl_with_no_exploration_is_deterministic_for_a_fixed_network() -> None:
    """Evaluation epsilon is 0, so the same net and seeds give the same run."""
    from agents.rl_router import RLRouter

    import torch
    torch.manual_seed(0)
    from agents.rl_router import LinkScoreNet
    net = LinkScoreNet()

    runs = []
    for _ in range(2):
        router = RLRouter(run_seed=8, net=net, training=False)
        assert router.epsilon == 0.0
        env = FANETEnv(routing="rl", log_path=os.devnull,
                       placement_seed=8, run_seed=8, router=router)
        env.reset()
        for _ in range(120):
            env.step()
        env.close_logger()
        runs.append([p.path for p in env.all_packets])
    assert runs[0] == runs[1]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_a_router_rule_without_a_router_fails_loudly(rule: str) -> None:
    """Forgetting the router is an error, not a silent fall back to greedy."""
    with pytest.raises(ValueError, match="needs a router object"):
        FANETEnv(routing=rule, log_path=os.devnull)


def test_router_rules_are_selectable_like_the_others() -> None:
    """rate and rl sit in ROUTING_RULES beside greedy and random."""
    from fanet_sim.envs.fanet_env import ROUTING_RULES
    assert set(ROUTER_RULES) <= set(ROUTING_RULES)
    assert {"greedy", "random", "rate", "rl"} == set(ROUTING_RULES)


def test_phase_1a_results_are_unchanged_by_the_1b_work() -> None:
    """greedy and random still reproduce the committed Phase-1a numbers.

    The agent's exploration stream is derived from its OWN SeedSequence rather
    than by spawning a third stream off the environment's two. Had it spawned,
    the channel stream would have shifted and every Phase-1a result with it.
    This checks against the committed out/1a_runs.csv, not against a value
    copied into the test.

    Pinned to LOOP_GUARD="previous": random is one of the rules the path guard
    changes, so the Phase-1a baseline is reproducible only under the guard it
    was run with. See tests/test_loop_guard.py.
    """
    from scripts.metrics_1a import compute_metrics
    from scripts.stamp import read_stamped_csv

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = read_stamped_csv(os.path.join(here, "out", "1a_runs.csv"))

    previous = config.PACKET_INTERVAL_STEPS
    previous_guard = config.LOOP_GUARD
    try:
        config.PACKET_INTERVAL_STEPS = 2          # the 200 ms rows
        config.LOOP_GUARD = "previous"
        for rule in ("greedy", "random"):
            expected = next(
                r for r in runs
                if r["routing"] == rule and r["load_ms"] == "200"
                and r["placement_seed"] == "1"
            )
            env = FANETEnv(routing=rule, log_path=os.devnull,
                           placement_seed=1, run_seed=1)
            env.reset()
            for _ in range(config.MAX_STEPS):
                env.step()
            env.close_logger()

            actual = compute_metrics(env)["ground_truth"]["total_loss"]
            assert actual == pytest.approx(float(expected["total_loss"]), abs=1e-12), (
                f"{rule} no longer reproduces its committed Phase-1a result"
            )
    finally:
        config.PACKET_INTERVAL_STEPS = previous
        config.LOOP_GUARD = previous_guard
