"""Static-world and reproducibility tests (spec item J, last two bullets)."""

from __future__ import annotations

import numpy as np

from fanet_sim import config

from .conftest import EpisodeTrace, run_episode


def test_no_drone_moves_in_static_mode(episode: EpisodeTrace) -> None:
    """J: no drone position changes during a run while STATIC_MODE is True."""
    assert config.STATIC_MODE, "this test assumes the Phase-1A static world"
    first = episode.positions[0]
    for step, snapshot in enumerate(episode.positions):
        assert np.array_equal(snapshot, first), f"a drone moved by step {step}"


def test_static_mode_burns_no_motion_energy(episode: EpisodeTrace) -> None:
    """Nothing moved, so the motion-energy budget is untouched."""
    for drone in episode.env.drones:
        assert drone.energy_motion == config.INITIAL_ENERGY


def test_static_episode_runs_the_full_step_budget() -> None:
    """The episode ends on the step budget, not on the all-arrived shortcut.

    A static M-drone is trivially "arrived", so the mobile-world shortcut would
    otherwise end the episode on step 1.
    """
    trace = run_episode(steps=120)
    assert trace.env.step_count == 120


def test_same_seeds_reproduce_the_run() -> None:
    """J: running twice with the same seeds gives identical results."""
    a = run_episode(steps=150, placement_seed=7, run_seed=11)
    b = run_episode(steps=150, placement_seed=7, run_seed=11)

    assert len(a.env.all_packets) == len(b.env.all_packets)
    assert len(a.env.delivered) == len(b.env.delivered)
    assert len(a.env.dropped) == len(b.env.dropped)
    assert a.tx_total == b.tx_total

    # Identical layout, and identical per-drone ACK counters.
    assert np.array_equal(a.positions[0], b.positions[0])
    for da, db in zip(a.env.drones, b.env.drones):
        assert dict(da.link_sent) == dict(db.link_sent)
        assert dict(da.link_acked) == dict(db.link_acked)
        assert dict(da.link_lost_channel) == dict(db.link_lost_channel)
        assert dict(da.link_lost_queue_full) == dict(db.link_lost_queue_full)
        assert da.packets_gs_acked == db.packets_gs_acked

    # Right down to each packet's route and fate.
    for pa, pb in zip(a.env.all_packets, b.env.all_packets):
        assert pa.path == pb.path
        assert pa.delivered == pb.delivered
        assert pa.drop_reason == pb.drop_reason


def test_placement_seed_alone_fixes_the_layout() -> None:
    """Same placement seed, different run seed: identical layout, different draws."""
    a = run_episode(steps=150, placement_seed=7, run_seed=11)
    b = run_episode(steps=150, placement_seed=7, run_seed=12)

    assert np.array_equal(a.positions[0], b.positions[0]), "layout must not move"
    assert [d.speed for d in a.env.drones] == [d.speed for d in b.env.drones]
    # The channel/traffic randomness differs, so the outcome should differ.
    assert len(a.env.delivered) != len(b.env.delivered)


def test_placement_seed_changes_the_layout() -> None:
    """A different placement seed really does move the drones."""
    a = run_episode(steps=20, placement_seed=7, run_seed=11)
    b = run_episode(steps=20, placement_seed=8, run_seed=11)
    assert not np.array_equal(a.positions[0], b.positions[0])


def test_routing_stream_is_separate_from_the_channel_stream() -> None:
    """Draws from the RANDOM-routing stream never disturb the channel stream.

    The two streams are spawned independently off run_seed, so draining one
    leaves the other's sequence untouched. This is what lets greedy and random
    run on identical channel and traffic randomness.
    """
    quiet = run_episode(steps=10, placement_seed=3, run_seed=5).env
    drained = run_episode(steps=10, placement_seed=3, run_seed=5).env

    # Heavily consume one env's routing stream; the other's is untouched.
    for _ in range(1000):
        drained._rng_route.random()

    # Both channel streams have seen exactly the same 10 steps of draws, so
    # their next values must still agree.
    assert [quiet._rng_chan.random() for _ in range(20)] == [
        drained._rng_chan.random() for _ in range(20)
    ]
