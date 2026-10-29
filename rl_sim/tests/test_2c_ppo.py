"""
test_2c_ppo.py — Phase 2c M2: the learning parts.

The environment side is covered by tests/test_2c_interface.py. What is new
here is everything that only exists while training: a centralised critic
that may see the whole world, a Gaussian actor that explores, the GAE pass
that turns one episode into targets, and the schedule that lets go of the
midpoint rule.

The critic's input is the one place in Phase 2c where something non-local is
allowed, so there is a test that says exactly that, and a matching one on
the actor's side saying the actor still cannot see it.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs import c_interface
from fanet_sim.envs.c_scripted import MidpointAgent
from fanet_sim.envs.fanet_env import FANETEnv

torch = pytest.importorskip("torch")


@contextmanager
def m0_world() -> Iterator[None]:
    """The world M0-M3 use: static mission, C-drones flying, hellos on."""
    saved = (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
             config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
             config.SHARE_TARGETS, config.AVOID)
    config.C_RULE = "rl"
    config.STATIC_MODE = True
    config.IDEAL_HELLO = False
    config.HELLO_INTERVAL = 1.0
    config.SHARE_TARGETS = True
    config.AVOID = True
    config.PACKET_INTERVAL_STEPS = 2
    try:
        yield
    finally:
        (config.C_RULE, config.STATIC_MODE, config.IDEAL_HELLO,
         config.PACKET_INTERVAL_STEPS, config.HELLO_INTERVAL,
         config.SHARE_TARGETS, config.AVOID) = saved


def _stepped_env(seed: int = 101, steps: int = 60) -> FANETEnv:
    """Return an environment that has run for a while under midpoint."""
    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    env.c_agent = MidpointAgent(env)
    for _ in range(steps):
        env.step()
    return env


# ---------------------------------------------------------------------------
# The inputs that were dropped
# ---------------------------------------------------------------------------

def test_the_ack_inputs_are_gone() -> None:
    """15 self features and 14 per neighbour, after dropping three."""
    assert c_interface.SELF_FEATURES == 15
    assert c_interface.NEIGHBOUR_FEATURES == 14

    with m0_world():
        env = _stepped_env()
        drone = next(d for d in env.drones if d.drone_type == "C")
        observation = c_interface.build_observation(env, drone)
        env.close_logger()
    assert observation["self"].shape == (15,)
    assert observation["neighbours"].shape[1] == 14


def test_nothing_in_the_observation_needs_an_end_to_end_ack() -> None:
    """Wiping the ACK books leaves the observation unchanged.

    The point of dropping those three inputs: the observation must not
    depend on anything an end-to-end ACK carries, because in the moving
    world an ACK may not get home.
    """
    with m0_world():
        env = _stepped_env()
        drone = next(d for d in env.drones if d.drone_type == "C")
        before = c_interface.build_observation(env, drone)

        for other in env.drones:
            other.packets_gs_acked = 0
            other.packets_gs_acked_measured = 0
            other.packets_lost_no_ack = 0
            other.outstanding = {}
        after = c_interface.build_observation(env, drone)
        env.close_logger()

    assert np.array_equal(before["self"], after["self"])
    assert np.array_equal(before["neighbours"], after["neighbours"])


# ---------------------------------------------------------------------------
# The critic sees the world; the actor does not
# ---------------------------------------------------------------------------

def test_the_critic_state_is_the_whole_world() -> None:
    """Moving any drone changes the critic's input.

    The opposite of the actor's test, and deliberately so: this is the one
    thing in Phase 2c allowed to be non-local, because it is only ever used
    while training.
    """
    with m0_world():
        env = _stepped_env()
        drone = next(d for d in env.drones if d.drone_type == "C")
        before = c_interface.build_global_state(env, drone)

        other = next(d for d in env.drones if d.drone_id != drone.drone_id)
        other.position = other.position + np.array([120.0, 0.0])
        after = c_interface.build_global_state(env, drone)
        env.close_logger()

    assert before.shape == (c_interface.global_state_width(25),)
    assert not np.array_equal(before, after), (
        "the critic did not notice a drone moving 120 m"
    )


def test_the_critic_state_says_which_drone_is_asking() -> None:
    """Two C-drones at the same moment get different critic inputs."""
    with m0_world():
        env = _stepped_env()
        c_drones = [d for d in env.drones if d.drone_type == "C"]
        first = c_interface.build_global_state(env, c_drones[0])
        second = c_interface.build_global_state(env, c_drones[1])
        env.close_logger()
    assert not np.array_equal(first, second)
    # The world part is shared; only the tail differs.
    shared = 25 * c_interface.GLOBAL_PER_DRONE
    assert np.array_equal(first[:shared], second[:shared])


def test_the_recorded_critic_state_matches_the_moment() -> None:
    """Each decision carries the global state as it was when taken."""
    with m0_world():
        env = _stepped_env(steps=40)
        assert env.c_decisions
        for row in env.c_decisions:
            assert row["global_state"].shape == (
                c_interface.global_state_width(25),
            )
            assert np.isfinite(row["global_state"]).all()
        env.close_logger()


# ---------------------------------------------------------------------------
# The actor
# ---------------------------------------------------------------------------

def test_the_actor_starts_exploring_at_about_forty_five_metres() -> None:
    """std = 0.1 in action units is 45 m on the ground."""
    from agents.c_actor import GaussianActor

    actor = GaussianActor()
    spread = float(actor.log_std.exp().mean().detach()) * c_interface.ACTION_SCALE_M
    assert spread == pytest.approx(45.0, abs=0.5)


def test_the_actor_mean_is_what_the_clone_would_say() -> None:
    """Wrapping a trained CActor in the Gaussian does not move its mean."""
    from agents.c_actor import CActor, GaussianActor, to_tensors

    plain = CActor()
    gaussian = GaussianActor()
    gaussian.mean.load_state_dict(plain.state_dict())

    batch = c_interface.pad_observations([{
        "self": np.zeros(c_interface.SELF_FEATURES),
        "neighbours": np.zeros((2, c_interface.NEIGHBOUR_FEATURES)),
        "mask": np.ones(2),
    }])
    own, neighbours, mask = to_tensors(batch)
    with torch.no_grad():
        assert torch.allclose(plain(own, neighbours, mask),
                              gaussian(own, neighbours, mask))


def test_sampling_spreads_around_the_mean() -> None:
    """Samples scatter about the mean with roughly the learned spread."""
    from agents.c_actor import GaussianActor, to_tensors

    actor = GaussianActor()
    batch = c_interface.pad_observations([{
        "self": np.zeros(c_interface.SELF_FEATURES),
        "neighbours": np.zeros((1, c_interface.NEIGHBOUR_FEATURES)),
        "mask": np.ones(1),
    }])
    own, neighbours, mask = to_tensors(batch)
    torch.manual_seed(0)
    with torch.no_grad():
        distribution = actor.distribution(own, neighbours, mask)
        drawn = distribution.sample((4000,)).squeeze(1)
        centre = distribution.mean.squeeze(0)
    assert float((drawn.mean(dim=0) - centre).abs().max()) < 0.02
    assert float(drawn.std(dim=0).mean()) == pytest.approx(0.1, abs=0.02)


def test_the_pooling_ignores_padding() -> None:
    """Padded slots change nothing: a table of 2 reads the same at width 5."""
    from agents.c_actor import CActor, to_tensors

    torch.manual_seed(0)
    actor = CActor()
    observation = {
        "self": np.linspace(0, 1, c_interface.SELF_FEATURES),
        "neighbours": np.linspace(
            0, 1, 2 * c_interface.NEIGHBOUR_FEATURES
        ).reshape(2, c_interface.NEIGHBOUR_FEATURES),
        "mask": np.ones(2),
    }
    narrow = c_interface.pad_observations([observation], width=2)
    wide = c_interface.pad_observations([observation], width=5)
    with torch.no_grad():
        a = actor(*to_tensors(narrow))
        b = actor(*to_tensors(wide))
    assert torch.allclose(a, b, atol=1e-6), (
        "padding leaked into the pooled neighbourhood"
    )


def test_an_empty_table_still_gives_an_answer() -> None:
    """A drone that has heard nobody must not produce NaN."""
    from agents.c_actor import CActor, to_tensors

    actor = CActor()
    batch = c_interface.pad_observations([{
        "self": np.zeros(c_interface.SELF_FEATURES),
        "neighbours": np.zeros((0, c_interface.NEIGHBOUR_FEATURES)),
        "mask": np.zeros(0),
    }], width=3)
    with torch.no_grad():
        out = actor(*to_tensors(batch))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# The update's arithmetic
# ---------------------------------------------------------------------------

def test_gae_on_one_trajectory_matches_the_hand_sum() -> None:
    """Checked against the discounted sum written out longhand."""
    from scripts.train_c_ppo import GAE_LAMBDA, GAMMA, advantages

    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, 0.5, 0.5])
    drones = np.array([7, 7, 7])
    order = np.array([0, 1, 2])

    advantage, returns = advantages(rewards, values, drones, order)

    deltas = [
        rewards[0] + GAMMA * values[1] - values[0],
        rewards[1] + GAMMA * values[2] - values[1],
        rewards[2] + GAMMA * 0.0 - values[2],
    ]
    step = GAMMA * GAE_LAMBDA
    expected = [
        deltas[0] + step * (deltas[1] + step * deltas[2]),
        deltas[1] + step * deltas[2],
        deltas[2],
    ]
    assert advantage == pytest.approx(expected, abs=1e-12)
    assert returns == pytest.approx(np.array(expected) + values, abs=1e-12)


def test_each_drone_gets_its_own_trajectory() -> None:
    """One drone's rewards never leak into another's advantage."""
    from scripts.train_c_ppo import advantages

    rewards = np.array([1.0, 100.0, 1.0, 100.0])
    values = np.zeros(4)
    drones = np.array([0, 1, 0, 1])
    order = np.array([0, 0, 1, 1])

    advantage, _ = advantages(rewards, values, drones, order)
    # Drone 0 never sees a 100, so neither of its advantages can approach one.
    assert advantage[0] < 10.0 and advantage[2] < 10.0
    assert advantage[1] > 90.0 and advantage[3] > 90.0


def test_the_anchor_fades_on_schedule() -> None:
    """Full weight while the critic warms up, then linearly to nothing."""
    from scripts.train_c_ppo import (ANCHOR_UPDATES, ANCHOR_WEIGHT,
                                     CRITIC_ONLY_UPDATES, anchor_weight)

    assert anchor_weight(1) == ANCHOR_WEIGHT
    assert anchor_weight(CRITIC_ONLY_UPDATES) == ANCHOR_WEIGHT
    half = CRITIC_ONLY_UPDATES + ANCHOR_UPDATES // 2
    assert anchor_weight(half) == pytest.approx(ANCHOR_WEIGHT / 2, abs=1e-9)
    assert anchor_weight(CRITIC_ONLY_UPDATES + ANCHOR_UPDATES) == 0.0
    assert anchor_weight(CRITIC_ONLY_UPDATES + ANCHOR_UPDATES + 50) == 0.0


def test_the_value_normaliser_tracks_and_inverts() -> None:
    """Normalising then denormalising returns the original numbers."""
    from agents.c_actor import RunningNorm

    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 2.0, size=5000)
    normaliser = RunningNorm()
    normaliser.update(values)

    assert normaliser.mean == pytest.approx(5.0, abs=0.1)
    assert np.sqrt(normaliser.var) == pytest.approx(2.0, abs=0.1)
    roundtrip = normaliser.denormalise(normaliser.normalise(values))
    assert roundtrip == pytest.approx(values, abs=1e-9)


def test_a_rollout_produces_aligned_arrays() -> None:
    """Every array from an episode has one row per usable decision."""
    from agents.c_actor import GaussianActor
    from scripts.train_c_ppo import MAX_NEIGHBOURS, _run_episode

    actor = GaussianActor()
    out = _run_episode(101, 9001, actor.state_dict(), "rl", greedy=False)

    rows = len(out["reward"])
    assert rows > 0
    for key in ("self", "neighbours", "mask", "global", "action",
                "log_prob", "anchor", "drone", "order"):
        assert len(out[key]) == rows, f"{key} has {len(out[key])} of {rows}"
    assert out["neighbours"].shape[1] == MAX_NEIGHBOURS
    assert out["self"].shape[1] == c_interface.SELF_FEATURES
    assert np.isfinite(out["reward"]).all()
    assert out["views_agree"]
