"""Routing-rule tests (spec item F).

Both rules use local information only and choose the next hop at SEND time.
Greedy keeps the candidates strictly closer to the GS and takes the lowest
channel loss among them; random draws uniformly over all neighbours. The GS is
an ordinary candidate whenever it is in range.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.channel import euclidean_distance
from fanet_sim.envs.fanet_env import (
    FANETEnv,
    greedy_next_hop,
    random_next_hop,
)

from .conftest import run_episode

GS = np.array(config.GS_POSITION, dtype=np.float64)


def _pos(x: float, y: float) -> np.ndarray:
    """Return a position array."""
    return np.array([x, y], dtype=np.float64)


# ---------------------------------------------------------------------------
# greedy
# ---------------------------------------------------------------------------

def test_greedy_returns_none_without_candidates() -> None:
    """No neighbours at all is a routing void."""
    assert greedy_next_hop(_pos(100.0, 100.0), {}, GS) is None


def test_greedy_returns_none_when_no_candidate_makes_progress() -> None:
    """Neighbours that are all further from the GS are a routing void."""
    holder = _pos(450.0, 300.0)          # 150 m from the GS
    candidates = {
        1: _pos(450.0, 250.0),           # 200 m — further out
        2: _pos(450.0, 200.0),           # 250 m — further out
    }
    assert greedy_next_hop(holder, candidates, GS) is None


def test_greedy_rejects_a_neighbour_at_the_same_distance() -> None:
    """The progress condition is STRICT: equally distant is not progress."""
    holder = _pos(450.0, 350.0)          # 100 m from the GS
    candidates = {1: _pos(450.0, 550.0)}  # also 100 m from the GS
    assert greedy_next_hop(holder, candidates, GS) is None


def test_greedy_picks_the_lowest_loss_among_closer_candidates() -> None:
    """Of the candidates that make progress, the cleanest link wins.

    Candidate 1 is closer to the GS but far from the holder (a lossy link);
    candidate 2 makes less progress but sits right next to the holder. The
    redefined rule takes the reliable link.
    """
    holder = _pos(450.0, 150.0)          # 300 m from the GS
    candidates = {
        1: _pos(450.0, 330.0),           # 120 m from GS, 180 m hop — lossy
        2: _pos(450.0, 170.0),           # 280 m from GS,  20 m hop — clean
    }
    assert channel.p_loss(180.0) > channel.p_loss(20.0)
    assert greedy_next_hop(holder, candidates, GS) == 2


def test_greedy_breaks_ties_toward_the_gs() -> None:
    """Two equally clean links: the one closer to the GS wins."""
    holder = _pos(450.0, 250.0)          # 200 m from the GS
    candidates = {
        1: _pos(450.0, 300.0),           # 50 m hop (0+50),   150 m from the GS
        2: _pos(410.0, 280.0),           # 50 m hop (40, 30), ~163 m from the GS
    }
    # Both hops are exactly 50 m, so the p_loss stage is an exact tie...
    assert euclidean_distance(holder, candidates[1]) == pytest.approx(50.0)
    assert euclidean_distance(holder, candidates[2]) == pytest.approx(50.0)
    # ...and both make progress, so the tie-break decides.
    holder_dist = euclidean_distance(holder, GS)
    assert euclidean_distance(candidates[1], GS) < holder_dist
    assert euclidean_distance(candidates[2], GS) < holder_dist
    assert euclidean_distance(candidates[1], GS) < euclidean_distance(
        candidates[2], GS
    )

    assert greedy_next_hop(holder, candidates, GS) == 1


def test_greedy_may_prefer_a_relay_over_a_marginal_direct_gs_shot() -> None:
    """The GS does not automatically win just for being in range."""
    holder = _pos(450.0, 210.0)          # 240 m from the GS — a marginal link
    candidates = {
        "GS": GS,
        1: _pos(450.0, 220.0),           # 10 m hop, 230 m from the GS
    }
    # The direct shot is far more likely to be lost than the short relay hop.
    assert channel.p_loss(240.0) > channel.p_loss(10.0)
    assert greedy_next_hop(holder, candidates, GS) == 1


def test_greedy_takes_the_gs_when_the_link_is_clean() -> None:
    """A short direct shot at the GS beats relaying."""
    holder = _pos(450.0, 430.0)          # 20 m from the GS
    candidates = {
        "GS": GS,
        1: _pos(450.0, 425.0),           # 25 m hop, 25 m from the GS
    }
    assert greedy_next_hop(holder, candidates, GS) == "GS"


def _link_loss(drone, pos: np.ndarray, key, gs: np.ndarray) -> float:
    """Re-derive the greedy score for one candidate, from the spec formula."""
    channel_loss = channel.p_loss(euclidean_distance(drone.position, pos))
    queue_loss = 1.0 if drone.queue_is_full(key) else 0.0
    return 1.0 - (1.0 - channel_loss) * (1.0 - queue_loss)


def test_greedy_choices_hold_during_a_real_run() -> None:
    """Over a full run, every greedy choice satisfies both stages of the rule.

    Re-derives the rule from each holder's own candidate set and queue state
    and checks the env's choice against it: strictly closer to the GS, and
    minimum link_loss among the candidates that qualify.
    """
    env = run_episode(steps=120, routing="greedy", placement_seed=4, run_seed=4).env

    checked = 0
    for drone in env.drones:
        candidates = env._next_hop_candidates(drone)
        choice = env._select_next_hop(drone, candidates)
        holder_dist = euclidean_distance(drone.position, env.gs_position)

        closer = {
            key: pos for key, pos in candidates.items()
            if euclidean_distance(pos, env.gs_position) < holder_dist
        }
        if not closer:
            assert choice is None
            continue

        assert choice is not None
        # Stage 1: the choice makes progress.
        assert euclidean_distance(candidates[choice], env.gs_position) < holder_dist
        # Stage 2: nothing that qualifies scores better.
        best = min(
            _link_loss(drone, pos, key, env.gs_position)
            for key, pos in closer.items()
        )
        chosen = _link_loss(drone, candidates[choice], choice, env.gs_position)
        assert chosen == pytest.approx(best)
        checked += 1

    assert checked > 0, "no drone had a usable next hop — test proved nothing"


# ---------------------------------------------------------------------------
# random
# ---------------------------------------------------------------------------

def test_random_returns_none_without_candidates() -> None:
    """No neighbours at all is a routing void for random too."""
    rng = np.random.default_rng(0)
    assert random_next_hop({}, rng) is None


def test_random_only_ever_picks_a_current_candidate() -> None:
    """Every draw is one of the offered keys."""
    rng = np.random.default_rng(0)
    candidates: Dict = {3: _pos(1.0, 1.0), 7: _pos(2.0, 2.0), "GS": GS}
    for _ in range(300):
        assert random_next_hop(candidates, rng) in candidates


def test_random_ignores_the_progress_condition() -> None:
    """Random will happily pick a neighbour further from the GS."""
    rng = np.random.default_rng(0)
    holder = _pos(450.0, 350.0)
    candidates = {1: _pos(450.0, 100.0)}   # strictly further from the GS
    assert greedy_next_hop(holder, candidates, GS) is None
    assert random_next_hop(candidates, rng) == 1


def test_random_covers_every_candidate() -> None:
    """Given enough draws, each candidate is chosen at least once."""
    rng = np.random.default_rng(0)
    candidates: Dict = {1: _pos(1.0, 1.0), 2: _pos(2.0, 2.0), "GS": GS}
    seen = {random_next_hop(candidates, rng) for _ in range(500)}
    assert seen == set(candidates)


def test_random_is_reproducible_for_a_given_stream() -> None:
    """The same stream and candidate set replay the same sequence."""
    candidates: Dict = {5: _pos(1.0, 1.0), 2: _pos(2.0, 2.0), "GS": GS}
    first = [random_next_hop(candidates, np.random.default_rng(3)) for _ in range(1)]
    second = [random_next_hop(candidates, np.random.default_rng(3)) for _ in range(1)]
    assert first == second

    rng = np.random.default_rng(9)
    seq_a = [random_next_hop(candidates, rng) for _ in range(50)]
    rng = np.random.default_rng(9)
    seq_b = [random_next_hop(candidates, rng) for _ in range(50)]
    assert seq_a == seq_b


def test_random_choices_hold_during_a_real_run() -> None:
    """Over a full run, every random choice is a current neighbour or the GS."""
    env = run_episode(steps=120, routing="random", placement_seed=4, run_seed=4).env

    checked = 0
    for drone in env.drones:
        candidates = env._next_hop_candidates(drone)
        choice = env._select_next_hop(drone, candidates)
        if not candidates:
            assert choice is None
            continue
        assert choice in candidates
        if choice != "GS":
            assert choice in drone.neighbors
        checked += 1

    assert checked > 0, "no drone had any neighbour — test proved nothing"


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("routing", ["greedy", "random"])
def test_a_holder_with_no_usable_next_hop_drops_as_no_route(routing: str) -> None:
    """A packet at a drone with nothing to forward to is dropped "no_route".

    The situation is CONSTRUCTED, not hunted for in a layout: with links
    reaching ~540 m in a 900x900 m area, essentially every drone can reach the
    GS directly, so no real placement contains such a drone.
    """
    import os

    from fanet_sim.envs.packet import DropReason

    env = FANETEnv(routing=routing, log_path=os.devnull)
    env.reset()

    # Strand one drone far outside everything: no neighbours, no GS link.
    drone = env.drones[0]
    drone.position = np.array([50_000.0, 50_000.0])
    drone.neighbors = {}
    drone.candidates = {}
    assert env._next_hop_candidates(drone) == {}

    pkt = _factory_for(env).create(source_id=drone.drone_id, created_at=0)
    env.all_packets.append(pkt)
    env._route_into_queue(drone, pkt, measured=False)

    assert pkt.dropped
    assert pkt.drop_reason is DropReason.NO_ROUTE
    assert drone.total_queued() == 0


def test_greedy_dead_end_drops_as_no_route() -> None:
    """A drone with neighbours but none closer to the GS drops "no_route".

    This is the case the layout filter does NOT remove: the drone is part of
    the graph, but greedy's progress condition finds nothing to hand the packet
    to. Constructed directly, since the dense network makes it rare in practice.
    """
    import os

    from fanet_sim.envs.packet import DropReason

    env = FANETEnv(routing="greedy", log_path=os.devnull)
    env.reset()

    reach = channel.max_link_distance()
    holder, neighbour = env.drones[0], env.drones[1]

    # Holder just beyond the GS's reach, so the GS is not a candidate...
    holder.position = env.gs_position + np.array([reach + 20.0, 0.0])
    # ...and its one neighbour is further from the GS still, but close enough
    # to the holder to be linked.
    neighbour.position = holder.position + np.array([50.0, 0.0])

    holder.neighbors = {neighbour.drone_id: neighbour}
    holder.candidates = dict(holder.neighbors)

    candidates = env._next_hop_candidates(holder)
    assert "GS" not in candidates, "the holder must not reach the GS directly"
    assert neighbour.drone_id in candidates, "the holder must still have a link"
    assert euclidean_distance(
        neighbour.position, env.gs_position
    ) > euclidean_distance(holder.position, env.gs_position)

    pkt = _factory_for(env).create(source_id=holder.drone_id, created_at=0)
    env.all_packets.append(pkt)
    env._route_into_queue(holder, pkt, measured=False)

    assert pkt.dropped
    assert pkt.drop_reason is DropReason.NO_ROUTE


def test_unknown_routing_rule_is_rejected() -> None:
    """A typo in the routing name fails loudly instead of silently defaulting."""
    import os

    with pytest.raises(ValueError, match="unknown routing rule"):
        FANETEnv(routing="qroute", log_path=os.devnull)


def test_both_rules_share_placement_and_traffic() -> None:
    """Greedy and random run on the same layout and the same traffic timing.

    The routing rule draws from its own stream, and the traffic offsets are
    drawn once at reset before any routing happens, so swapping the rule
    changes the routing decisions and nothing else about the setup.
    """
    greedy = run_episode(steps=120, routing="greedy", placement_seed=9, run_seed=9)
    random_ = run_episode(steps=120, routing="random", placement_seed=9, run_seed=9)

    assert np.array_equal(greedy.positions[0], random_.positions[0])
    assert greedy.env._traffic_offset == random_.env._traffic_offset
    # Same layout and same traffic schedule means the same packets exist.
    assert len(greedy.env.all_packets) == len(random_.env.all_packets)
    assert [p.created_at for p in greedy.env.all_packets] == [
        p.created_at for p in random_.env.all_packets
    ]
    assert [p.source_id for p in greedy.env.all_packets] == [
        p.source_id for p in random_.env.all_packets
    ]
    # But the routing decisions differ, so the outcomes must too.
    assert len(greedy.env.delivered) != len(random_.env.delivered)


def test_traffic_offsets_spread_creation_across_the_interval() -> None:
    """M-drones do not all create their packets on the same step."""
    env = run_episode(steps=60, placement_seed=9, run_seed=9).env
    interval = config.PACKET_INTERVAL_STEPS
    offsets = set(env._traffic_offset.values())

    assert set(env._traffic_offset) == {
        d.drone_id for d in env.drones if d.drone_type == "M"
    }
    assert all(0 <= off < interval for off in offsets)
    if interval > 1:
        assert len(offsets) > 1, "every M-drone landed on the same offset"


# ---------------------------------------------------------------------------
# greedy: queue awareness (the new link_loss term)
# ---------------------------------------------------------------------------

def _env() -> FANETEnv:
    """A reset environment for routing tests that need real drones."""
    import os

    env = FANETEnv(routing="greedy", log_path=os.devnull)
    env.reset()
    return env


def test_greedy_avoids_its_own_full_link_queue() -> None:
    """A full queue makes that link's loss 1, so a non-full link wins.

    Candidate 1 has the cleaner channel and would win outright, but this
    drone's queue for it is full; candidate 2 still has room.
    """
    holder = _pos(450.0, 150.0)          # 300 m from the GS
    candidates = {
        1: _pos(450.0, 180.0),           #  30 m hop — clean channel
        2: _pos(450.0, 230.0),           #  80 m hop — worse channel
    }
    # With both queues free, the clean channel wins.
    assert greedy_next_hop(holder, candidates, GS, {1: False, 2: False}) == 1
    # With link 1's queue full, greedy routes around its own congestion.
    assert greedy_next_hop(holder, candidates, GS, {1: True, 2: False}) == 2


def test_greedy_falls_back_to_the_cleanest_channel_when_all_links_are_full() -> None:
    """If every qualifying link is full they all score 1, so channel decides."""
    holder = _pos(450.0, 150.0)
    candidates = {
        1: _pos(450.0, 180.0),           # 30 m hop — cleanest
        2: _pos(450.0, 230.0),           # 80 m hop
    }
    choice = greedy_next_hop(holder, candidates, GS, {1: True, 2: True})
    assert choice == 1


def test_greedy_full_queue_does_not_override_the_progress_condition() -> None:
    """A non-full link that makes no progress is still not eligible."""
    holder = _pos(450.0, 350.0)          # 100 m from the GS
    candidates = {
        1: _pos(450.0, 200.0),           # further from the GS — never eligible
    }
    assert greedy_next_hop(holder, candidates, GS, {1: False}) is None


def test_greedy_reads_the_holders_own_queues() -> None:
    """The env feeds greedy this drone's real queue state, not a guess."""
    env = _env()
    drone = next(d for d in env.drones if len(d.neighbors) >= 2)
    candidates = env._next_hop_candidates(drone)

    holder_dist = euclidean_distance(drone.position, env.gs_position)
    closer = [
        key for key, pos in candidates.items()
        if euclidean_distance(pos, env.gs_position) < holder_dist
    ]
    if len(closer) < 2:
        pytest.skip("this drone has fewer than two qualifying links")

    first = env._select_next_hop(drone, candidates)
    assert first in closer

    # Fill the winner's queue; greedy must now pick something else.
    factory = _factory_for(env)
    for i in range(config.QUEUE_CAPACITY):
        drone.link_queues[first].append(factory.create(source_id=0, created_at=i))
    assert drone.queue_is_full(first)

    second = env._select_next_hop(drone, candidates)
    assert second != first
    assert second in closer


def _factory_for(env: FANETEnv):
    """Return a packet factory matching the env's TTL and hop limit."""
    from fanet_sim.envs.packet import PacketFactory

    return PacketFactory(
        ttl=config.PACKET_TTL, max_hops=config.MAX_HOPS, size_bytes=512
    )


# ---------------------------------------------------------------------------
# routing happens at enqueue time
# ---------------------------------------------------------------------------

def test_next_hop_is_fixed_when_the_packet_is_queued() -> None:
    """A queued packet's link is decided on enqueue and never revisited.

    A packet sitting in the queue for link L is sent to L, even if the routing
    rule would now prefer a different link.
    """
    env = _env()
    drone = next(d for d in env.drones if len(d.neighbors) >= 2)
    candidates = env._next_hop_candidates(drone)
    chosen = env._select_next_hop(drone, candidates)
    if chosen is None:
        pytest.skip("this drone has no qualifying next hop")

    factory = _factory_for(env)
    pkt = factory.create(source_id=drone.drone_id, created_at=env.step_count)
    env.all_packets.append(pkt)
    env._route_into_queue(drone, pkt, measured=False)

    # The packet is waiting on exactly the link that was chosen for it.
    assert pkt in drone.link_queues[chosen]
    assert all(
        pkt not in queue
        for key, queue in drone.link_queues.items()
        if key != chosen
    )

    # Now make that link look terrible by filling its queue. The packet does
    # not move: the decision was already made.
    for i in range(config.QUEUE_CAPACITY):
        drone.link_queues[chosen].append(
            factory.create(source_id=0, created_at=i)
        )
    assert pkt in drone.link_queues[chosen]


def test_a_packet_arriving_cannot_be_sent_in_the_same_step() -> None:
    """Arrivals are routed on arrival but wait a step before being sent.

    The transmission snapshot is taken before arrivals are routed, so a packet
    that lands this step is still in its new queue when the step ends.
    """
    trace = run_episode(steps=40, routing="greedy", placement_seed=4, run_seed=4)
    env = trace.env

    # Every packet that took at least two hops must have spent at least one
    # step waiting between them.
    multi_hop = [p for p in env.delivered if p.hop_count >= 2]
    if not multi_hop:
        pytest.skip("no multi-hop delivery in this short run")
    for pkt in multi_hop:
        delay = pkt.delay()
        assert delay is not None
        assert delay >= pkt.hop_count - 1


# ---------------------------------------------------------------------------
# arrival processing order
# ---------------------------------------------------------------------------

def test_arrival_order_is_reproducible_for_the_same_seeds() -> None:
    """Same seeds replay the same arrival order, so runs are identical."""
    a = run_episode(steps=150, routing="greedy", placement_seed=6, run_seed=6).env
    b = run_episode(steps=150, routing="greedy", placement_seed=6, run_seed=6).env

    assert len(a.delivered) == len(b.delivered)
    assert len(a.dropped) == len(b.dropped)
    for pa, pb in zip(a.all_packets, b.all_packets):
        assert pa.path == pb.path
        assert pa.drop_reason == pb.drop_reason


def test_arrival_order_changes_with_the_run_seed() -> None:
    """A different run seed gives a different ordering, and different outcomes."""
    a = run_episode(steps=150, routing="greedy", placement_seed=6, run_seed=6).env
    b = run_episode(steps=150, routing="greedy", placement_seed=6, run_seed=7).env

    assert [p.path for p in a.all_packets] != [p.path for p in b.all_packets]


def test_arrivals_are_not_processed_in_drone_id_order() -> None:
    """The order really is shuffled, not just the identity permutation.

    Ordering by drone id would hand the same drones the last free slot in a
    nearly full queue every step.
    """
    env = _env()
    order_seen = []
    real = FANETEnv._route_into_queue

    def spy(self, drone, pkt, measured):
        order_seen.append(drone.drone_id)
        return real(self, drone, pkt, measured)

    FANETEnv._route_into_queue = spy  # type: ignore[method-assign]
    try:
        # Drive _route_arrivals directly with a known, id-ordered input.
        env._route_arrivals(  # type: ignore[arg-type]
            [(d, _factory_for(env).create(source_id=d.drone_id, created_at=0))
             for d in env.drones],
            measured=False,
        )
    finally:
        FANETEnv._route_into_queue = real  # type: ignore[method-assign]

    assert sorted(order_seen) == [d.drone_id for d in env.drones]
    assert order_seen != [d.drone_id for d in env.drones], (
        "arrivals were processed in drone-id order"
    )
