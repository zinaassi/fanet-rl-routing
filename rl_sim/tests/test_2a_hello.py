"""
test_2a_hello.py — Neighbour knowledge learned from hellos, not handed over.

Phase 2a step 2. Until now a drone was simply given its neighbours' positions.
Now it broadcasts a hello every ``HELLO_INTERVAL``, hears other drones' hellos
through the same loss curve the data uses, and routes on nothing but the table
that builds up. The table is therefore always a little wrong, in two
directions at once: it lists neighbours that have left, and misses ones that
have arrived.

The checks fall into four groups:

* ``IDEAL_HELLO`` restores the old perfect knowledge exactly — this is what
  keeps every Phase-1 and step-1 result reproducible;
* the hello schedule itself — rate, jitter, start times;
* the table — what goes in, what times out, what the router sees;
* the costs the table's errors create, and that the books still balance.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.channel import are_connected
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import DropReason


@contextmanager
def world(static: bool = True, ideal: bool = True,
          interval: float = 1.0, load: int = 200) -> Iterator[None]:
    """Run the body in one configuration, then put every setting back."""
    saved = (config.STATIC_MODE, config.IDEAL_HELLO, config.HELLO_INTERVAL,
             config.PACKET_INTERVAL_STEPS)
    config.STATIC_MODE = static
    config.IDEAL_HELLO = ideal
    config.HELLO_INTERVAL = interval
    config.PACKET_INTERVAL_STEPS = int(round(load / 1000.0 / config.TIMESTEP))
    try:
        yield
    finally:
        (config.STATIC_MODE, config.IDEAL_HELLO, config.HELLO_INTERVAL,
         config.PACKET_INTERVAL_STEPS) = saved


def _run(rule: str, seed: int, steps: int = config.MAX_STEPS) -> FANETEnv:
    """Run one episode and return the finished environment."""
    env = FANETEnv(routing=rule, log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    for _ in range(steps):
        env.step()
    env.close_logger()
    return env


def _causes(env: FANETEnv) -> dict:
    """Return the loss split by cause for a finished run."""
    from scripts.metrics_1a import compute_metrics
    return compute_metrics(env)["ground_truth"]["loss_rate_by_reason"]


# ---------------------------------------------------------------------------
# IDEAL_HELLO changes nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ("greedy", "random"))
def test_ideal_hello_and_no_movement_is_still_phase_1(rule: str) -> None:
    """The frozen static reference, with the hello code in place."""
    from scripts.metrics_1a import compute_metrics
    from scripts.stamp import read_stamped_csv

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "out", "2a_reference.csv")
    if not os.path.exists(path):
        pytest.skip("out/2a_reference.csv is missing")
    expected = next(
        r for r in read_stamped_csv(path)
        if r["rule"] == rule and int(r["placement_seed"]) == 3
    )

    with world(static=True, ideal=True):
        env = _run(rule, 3)
    truth = compute_metrics(env)["ground_truth"]

    assert (truth["total_loss"] or 0.0) == pytest.approx(
        float(expected["total_loss"]), abs=1e-12
    )
    for cause in ("channel", "queue_full", "no_route", "dead_end",
                  "ttl", "hop_limit"):
        assert truth["loss_rate_by_reason"][cause] == pytest.approx(
            float(expected[cause]), abs=1e-12
        )
    assert truth["loss_rate_by_reason"]["stale_link"] == 0.0


@pytest.mark.parametrize("rule", ("greedy", "random"))
def test_ideal_hello_with_movement_reproduces_step_1(rule: str) -> None:
    """With the fleet flying, IDEAL_HELLO gives the step-1 numbers back.

    Checked against out/2a_runs.csv, written before any hello code existed.
    """
    from scripts.metrics_1a import compute_metrics
    from scripts.stamp import read_stamped_csv

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "out", "2a_runs.csv")
    if not os.path.exists(path):
        pytest.skip("out/2a_runs.csv is missing; run scripts/experiment_2a.py")
    rows = [r for r in read_stamped_csv(path)
            if r["rule"] == rule and int(r["placement_seed"]) == 3
            and float(r["max_start_delay_s"]) == 50.0]
    if not rows:
        pytest.skip("no 50 s row for this rule in out/2a_runs.csv")
    expected = rows[0]

    saved = config.MAX_START_DELAY_S
    config.MAX_START_DELAY_S = 50.0
    try:
        with world(static=False, ideal=True):
            env = _run(rule, 3)
    finally:
        config.MAX_START_DELAY_S = saved
    truth = compute_metrics(env)["ground_truth"]

    assert (truth["total_loss"] or 0.0) == pytest.approx(
        float(expected["total_loss"]), abs=1e-12
    ), f"{rule} no longer reproduces its step-1 result"
    for cause in ("channel", "queue_full", "no_route", "dead_end",
                  "ttl", "hop_limit", "link_broken"):
        assert truth["loss_rate_by_reason"][cause] == pytest.approx(
            float(expected[cause]), abs=1e-12
        )
    assert env.stale_sends == 0


def test_ideal_hello_draws_nothing_from_the_hello_stream() -> None:
    """A run with perfect knowledge leaves the hello stream untouched."""
    with world(static=False, ideal=True):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=4, run_seed=4)
        env.reset()
        before = env._rng_hello.bit_generator.state["state"]["state"]
        for _ in range(200):
            env.step()
        after = env._rng_hello.bit_generator.state["state"]["state"]
        env.close_logger()
    assert before == after, "IDEAL_HELLO consumed a hello draw"
    assert all(d.hellos_sent == 0 for d in env.drones)


def test_hellos_do_not_disturb_the_other_streams() -> None:
    """Turning hellos on changes no layout, trajectory or channel draw.

    The layout and the flight come from the placement and mobility streams,
    and the data channel from the channel stream. None of the three may move
    because a hello was drawn.
    """
    with world(static=False, ideal=True):
        ideal = _run("greedy", 6, steps=0)
    with world(static=False, ideal=False):
        real = _run("greedy", 6, steps=0)

    for a, b in zip(sorted(ideal.drones, key=lambda d: d.drone_id),
                    sorted(real.drones, key=lambda d: d.drone_id)):
        assert np.array_equal(a.start_point, b.start_point)
        assert np.array_equal(a.end_point, b.end_point)
        assert a.speed == b.speed
        assert a.start_delay_s == b.start_delay_s
    assert ideal.placement_draws == real.placement_draws
    # The channel stream has not been touched by either, so both sit at the
    # same point before any traffic runs.
    assert (ideal._rng_chan.bit_generator.state["state"]["state"]
            == real._rng_chan.bit_generator.state["state"]["state"])


# ---------------------------------------------------------------------------
# The hello schedule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval", (0.2, 0.5, 1.0, 2.0))
def test_the_hello_rate_matches_the_interval(interval: float) -> None:
    """Each drone sends about one hello per interval, jitter included.

    The jitter only ever SHORTENS an interval, by up to
    HELLO_JITTER_FRACTION, so the rate sits between 1/interval and
    1/(interval * (1 - fraction)).
    """
    with world(static=False, ideal=False, interval=interval):
        env = _run("greedy", 7, steps=600)

    seconds = 600 * config.TIMESTEP
    low = 1.0 / interval
    high = 1.0 / (interval * (1.0 - config.HELLO_JITTER_FRACTION))
    for drone in env.drones:
        rate = drone.hellos_sent / seconds
        assert low - 0.2 <= rate <= high + 0.2, (
            f"drone {drone.drone_id} sent {rate:.2f} hellos/s at a "
            f"{interval}s interval, expected {low:.2f}-{high:.2f}"
        )
    gs_rate = env._gs_hellos_sent / seconds
    assert low - 0.2 <= gs_rate <= high + 0.2


def test_hellos_do_not_all_start_together() -> None:
    """Start times are spread over the first interval."""
    with world(static=False, ideal=False, interval=1.0):
        env = _run("greedy", 8, steps=0)
    firsts = sorted(d.next_hello_step for d in env.drones)
    assert len(set(firsts)) > 1, "every drone would broadcast on the same step"
    assert min(firsts) >= 0
    assert max(firsts) < config.HELLO_INTERVAL / config.TIMESTEP


def test_a_drone_hears_more_hellos_than_it_sends() -> None:
    """With 25 drones in range of several others, reception outnumbers sending."""
    with world(static=False, ideal=False):
        env = _run("greedy", 9, steps=400)
    assert sum(d.hellos_received for d in env.drones) > sum(
        d.hellos_sent for d in env.drones
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_the_table_only_holds_senders_that_claimed_to_be_in_range() -> None:
    """An entry is taken only if the advertised position is within reach.

    Checked on the step the entry was written, which is the moment the rule
    applies: the receiver moves afterwards, so an older entry may well be out
    of range by now — that is the staleness this step is about.
    """
    with world(static=False, ideal=False):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=10, run_seed=10)
        env.reset()
        fresh_seen = 0
        for _ in range(300):
            env.step()
            for drone in env.drones:
                for key, hello in drone.hello_table.items():
                    assert key != drone.drone_id, "a drone heard itself"
                    if hello.sent_at != env.step_count - 1:
                        continue        # not written on the last step
                    fresh_seen += 1
                    assert are_connected(drone.position, hello.position), (
                        f"drone {drone.drone_id} accepted {key} advertising "
                        f"a position out of range"
                    )
        env.close_logger()
    assert fresh_seen > 0, "no entry was written at all, so nothing was checked"


def test_a_hello_advertising_an_out_of_range_position_is_ignored() -> None:
    """Put a sender far away and confirm nobody takes it into a table."""
    from fanet_sim.envs import channel

    with world(static=False, ideal=False):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=10, run_seed=10)
        env.reset()
        for drone in env.drones:
            drone.hello_table.clear()
            drone.next_hello_step = 10 ** 9      # nobody else broadcasts
        env._gs_next_hello_step = 10 ** 9

        # One sender, parked beyond the edge from everyone, broadcasting now.
        far = env.drones[0]
        far.position = np.array([0.0, 0.0], dtype=np.float64)
        for other in env.drones[1:]:
            other.position = np.array([config.WIDTH, config.HEIGHT],
                                      dtype=np.float64)
        far.next_hello_step = env.step_count
        env._exchange_hellos(measured=True)
        env.close_logger()

    corner = float(np.hypot(config.WIDTH, config.HEIGHT))
    assert corner > channel.max_link_distance()
    assert all(not d.hello_table for d in env.drones), (
        "a hello from beyond the edge was taken into a table"
    )


def test_an_entry_times_out_after_three_intervals_of_silence() -> None:
    """Nothing older than the timeout survives in any table."""
    interval = 1.0
    with world(static=False, ideal=False, interval=interval):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=11, run_seed=11)
        env.reset()
        timeout = int(round(config.HELLO_TIMEOUT_INTERVALS
                            * interval / config.TIMESTEP))
        for _ in range(400):
            env.step()
            for drone in env.drones:
                for hello in drone.hello_table.values():
                    age = env.step_count - hello.sent_at
                    assert age <= timeout + 1, (
                        f"a {age}-step-old entry survived a {timeout}-step "
                        f"timeout"
                    )
        env.close_logger()


def test_the_router_sees_the_table_and_nothing_else() -> None:
    """Every next hop a drone considers comes from its own table."""
    with world(static=False, ideal=False):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=12, run_seed=12)
        env.reset()
        for _ in range(250):
            env.step()
            for drone in env.drones:
                believed = set(env._believed_links(drone))
                assert believed == set(drone.hello_table), (
                    "the router is reading something other than the table"
                )
        env.close_logger()


def test_the_table_is_not_perfect_once_the_fleet_moves() -> None:
    """The point of the step: knowledge lags, in both directions.

    Without this, every other table check could pass on a table that happened
    to be exactly right all run.
    """
    from scripts.metrics_1a import compute_metrics

    with world(static=False, ideal=False):
        env = _run("greedy", 5)
    truth = compute_metrics(env)["ground_truth"]
    assert truth["table_precision"] < 1.0, "no stale entry ever appeared"
    assert truth["table_recall"] < 1.0, "no neighbour was ever missed"
    assert 0.0 <= truth["table_precision"] <= 1.0
    assert 0.0 <= truth["table_recall"] <= 1.0


# ---------------------------------------------------------------------------
# What the errors cost, and the books
# ---------------------------------------------------------------------------

def test_no_stale_sends_while_nothing_moves() -> None:
    """With real hellos but a still world, nobody is ever really gone.

    Entries can still time out and come back, but a send can never be aimed
    at a drone that has moved out of range, because none of them moves.
    """
    with world(static=True, ideal=False):
        env = _run("greedy", 13)
    assert env.stale_sends == 0
    assert env.stale_link_drops == 0
    assert _causes(env)["stale_link"] == 0.0


def test_moving_with_real_hellos_produces_stale_sends() -> None:
    """The stale path is exercised: sends go to neighbours that have left."""
    with world(static=False, ideal=False):
        env = _run("greedy", 5)
    assert env.stale_sends > 0
    assert env.stale_link_drops > 0
    assert any(p.drop_reason is DropReason.STALE_LINK
               for p in env.all_packets if p.dropped)
    # A stale send is not automatically lost — the curve still decides.
    assert env.stale_link_drops <= env.stale_sends


def test_packets_add_up_with_real_hellos() -> None:
    """created = delivered + dropped + still in flight."""
    with world(static=False, ideal=False):
        env = _run("greedy", 5)
    created = len(env.all_packets)
    delivered = sum(1 for p in env.all_packets if p.delivered)
    dropped = sum(1 for p in env.all_packets if p.dropped)
    in_flight = sum(1 for p in env.all_packets
                    if not p.delivered and not p.dropped)
    assert created == delivered + dropped + in_flight
    assert created > 0 and delivered > 0


@pytest.mark.parametrize("rule", ("greedy", "random"))
def test_the_two_views_still_agree_with_real_hellos(rule: str) -> None:
    """ACK counters match ground truth once stale losses are in the mix."""
    from scripts.metrics_1a import compute_metrics

    with world(static=False, ideal=False):
        for seed in (2, 5):
            env = _run(rule, seed)
            metrics = compute_metrics(env)
            assert not metrics["discrepancies"], metrics["discrepancies"]


def test_no_queue_outlives_its_table_entry() -> None:
    """A drone never holds packets for a neighbour it has forgotten."""
    with world(static=False, ideal=False):
        env = FANETEnv(routing="greedy", log_path=os.devnull,
                       placement_seed=5, run_seed=5)
        env.reset()
        for _ in range(500):
            env.step()
            for drone in env.drones:
                for key, queue in drone.link_queues.items():
                    if queue:
                        assert key in drone.hello_table, (
                            f"drone {drone.drone_id} still queues for {key}, "
                            f"which is no longer in its table"
                        )
        env.close_logger()


def test_nothing_crashes_at_the_top_speed_with_hellos() -> None:
    """200 runs at 30 m/s with real hellos."""
    saved = (config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX)
    with world(static=False, ideal=False):
        config.DRONE_SPEED_MIN = config.DRONE_SPEED_MAX = 30.0
        try:
            for seed in range(1, 201):
                env = FANETEnv(routing="greedy", log_path=os.devnull,
                               placement_seed=seed, run_seed=seed)
                env.reset()
                for _ in range(60):
                    env.step()
                env.close_logger()
        finally:
            config.DRONE_SPEED_MIN, config.DRONE_SPEED_MAX = saved
