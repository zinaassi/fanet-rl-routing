"""The path-based loop guard (CHECKPOINT 12).

The guard is a property of the NETWORK, not of a routing rule: the packet
header carries the list of drones it has visited, and every rule is forbidden
to choose a hop already on that list. ``config.LOOP_GUARD`` selects between
``"path"`` (the whole visited list) and ``"previous"`` (only the drone the
packet just came from, the Phase-1a/CHECKPOINT-11 behaviour), so the older
results stay reproducible.

A packet whose every reachable neighbour is already on its path has nowhere
legal to go and is dropped as ``dead_end`` — distinct from ``no_route``, which
means it had no link at all.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import DropReason, PacketFactory

from agents import make_router


@contextlib.contextmanager
def loop_guard(mode: str) -> Iterator[None]:
    """Run the block with ``config.LOOP_GUARD`` set to *mode*."""
    previous = config.LOOP_GUARD
    config.LOOP_GUARD = mode
    try:
        yield
    finally:
        config.LOOP_GUARD = previous


def _run(rule: str, placement_seed: int = 3, run_seed: int = 3,
         steps: int = 400) -> FANETEnv:
    """Run *steps* of *rule* and return the finished environment."""
    router = (make_router(rule, run_seed=run_seed, training=False, init_seed=0)
              if rule in ("rate", "rl") else None)
    env = FANETEnv(routing=rule, log_path=os.devnull,
                   placement_seed=placement_seed, run_seed=run_seed,
                   router=router)
    env.reset()
    for _ in range(steps):
        env.step()
    env.close_logger()
    return env


def _revisits(env: FANETEnv) -> List[Tuple[int, List[Any]]]:
    """Return (packet_id, path) for every packet that visited a drone twice."""
    found = []
    for pkt in env.all_packets:
        hops = [h for h in pkt.path if h != "GS"]
        if len(hops) != len(set(hops)):
            found.append((pkt.packet_id, list(pkt.path)))
    return found


# ---------------------------------------------------------------------------
# What the guard forbids
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["random", "rate", "rl"])
def test_no_packet_ever_visits_the_same_drone_twice(rule: str) -> None:
    """Under "path", a loop cannot happen at all, for any rule.

    Checked over a full run rather than on a constructed case: a loop is a
    property of the whole trajectory, and the rules that loop in practice
    (random, rate, rl) only do so under load.
    """
    with loop_guard("path"):
        env = _run(rule)
    assert env.all_packets, "the run produced no packets to check"
    assert _revisits(env) == [], f"{rule} revisited a drone under the guard"


@pytest.mark.parametrize("rule", ["random", "rate"])
def test_the_same_rules_do_loop_without_the_path_guard(rule: str) -> None:
    """The guard is what removes the loops, not the seeds.

    Without this, the test above would pass on a run that simply never looped.
    """
    with loop_guard("previous"):
        env = _run(rule)
    assert _revisits(env), f"{rule} never looped, so nothing was being guarded"


def test_the_gs_is_never_excluded_by_the_guard() -> None:
    """The GS is the destination, never a hop to avoid.

    It cannot appear on a path before delivery, but the guard must not depend
    on that: it keeps "GS" unconditionally.
    """
    env = FANETEnv(routing="greedy", log_path=os.devnull)
    env.reset()

    holder = env.drones[0]
    holder.position = env.gs_position.copy()
    neighbour = env.drones[1]
    neighbour.position = env.gs_position + np.array([10.0, 0.0])
    holder.neighbors = {neighbour.drone_id: neighbour}
    holder.candidates = dict(holder.neighbors)

    pkt = PacketFactory(ttl=config.PACKET_TTL, max_hops=config.MAX_HOPS,
                        size_bytes=512).create(source_id=holder.drone_id,
                                               created_at=0)
    # Pretend the packet has already been everywhere, the GS included.
    pkt.path = [holder.drone_id, neighbour.drone_id, "GS"]

    with loop_guard("path"):
        allowed = env._allowed_candidates(holder, pkt)

    assert "GS" in allowed, "the guard removed the destination"
    assert neighbour.drone_id not in allowed, "a visited drone survived"


# ---------------------------------------------------------------------------
# dead_end
# ---------------------------------------------------------------------------

def test_a_packet_with_every_neighbour_visited_is_a_dead_end() -> None:
    """All links lead back onto the path: dead_end, not no_route.

    Constructed directly. The drone has links, so it is not a routing void;
    what leaves it with nothing is the guard.
    """
    # random, not greedy: greedy would refuse both neighbours anyway, since
    # neither is closer to the GS, and that would hide the guard's own effect.
    env = FANETEnv(routing="random", log_path=os.devnull, run_seed=7)
    env.reset()

    reach = channel.max_link_distance()
    holder, left, right = env.drones[0], env.drones[1], env.drones[2]

    # Out of the GS's reach, so "GS" is not among the candidates.
    holder.position = env.gs_position + np.array([reach + 20.0, 0.0])
    left.position = holder.position + np.array([30.0, 0.0])
    right.position = holder.position + np.array([0.0, 30.0])
    holder.neighbors = {left.drone_id: left, right.drone_id: right}
    holder.candidates = dict(holder.neighbors)

    assert "GS" not in env._next_hop_candidates(holder)

    factory = PacketFactory(ttl=config.PACKET_TTL, max_hops=config.MAX_HOPS,
                            size_bytes=512)

    with loop_guard("path"):
        pkt = factory.create(source_id=holder.drone_id, created_at=0)
        pkt.path = [left.drone_id, right.drone_id, holder.drone_id]
        env.all_packets.append(pkt)
        env._route_into_queue(holder, pkt, measured=False)
        assert pkt.dropped
        assert pkt.drop_reason is DropReason.DEAD_END

    with loop_guard("previous"):
        other = factory.create(source_id=holder.drone_id, created_at=0)
        other.path = [left.drone_id, right.drone_id, holder.drone_id]
        env.all_packets.append(other)
        env._route_into_queue(holder, other, measured=False)
        assert not other.dropped, (
            "the previous-hop guard should let it go back to an earlier drone"
        )
        assert other.current_holder == holder.drone_id, "it was not queued"


def test_dead_end_is_counted_in_both_metric_views() -> None:
    """dead_end packets are lost packets, and the two views still agree.

    A dead_end drop happens while the drone holds the packet, so no link
    counter moves and the source simply never gets its ACK — exactly like
    no_route. The invariant to protect is that the drones' own books still
    reconcile with the simulator's.
    """
    from scripts.metrics_1a import compute_metrics

    with loop_guard("path"):
        env = _run("random", steps=config.MAX_STEPS)
        metrics = compute_metrics(env)

    truth = metrics["ground_truth"]
    assert truth["loss_by_reason"]["dead_end"] > 0, "no dead_end drop to check"
    assert metrics["discrepancies"] == [], metrics["discrepancies"]

    # Every dropped packet carries a reason, and dead_end is one of them.
    assert sum(truth["loss_by_reason"].values()) == truth["dropped"]

    # The drones' own books agree on how many packets were lost, dead_end
    # drops included: the source never gets an ACK, exactly as for no_route.
    assert metrics["acks"]["lost_no_ack"] == truth["dropped"] + \
        truth["still_in_flight"]


# ---------------------------------------------------------------------------
# What the guard must NOT change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("placement_seed", [1, 2, 3])
def test_greedy_is_identical_under_both_modes(placement_seed: int) -> None:
    """Greedy only ever moves strictly closer to the GS, so it cannot revisit.

    Its results must therefore be untouched by the guard — not merely similar.
    """
    from scripts.metrics_1a import compute_metrics

    results = {}
    for mode in ("previous", "path"):
        with loop_guard(mode):
            env = _run("greedy", placement_seed=placement_seed,
                       run_seed=placement_seed, steps=config.MAX_STEPS)
            truth = compute_metrics(env)["ground_truth"]
        results[mode] = (truth["created"], truth["delivered"],
                         truth["total_loss"], truth["mean_hops"],
                         dict(truth["loss_by_reason"]))

    assert results["previous"] == results["path"]
    assert results["path"][4]["dead_end"] == 0


def test_checkpoint_11_is_reproduced_under_the_previous_guard() -> None:
    """rate and rl still match the CHECKPOINT-11 TEST numbers.

    Checked against out/1b_test_runs_PREVIOUS.csv and the models kept in
    out/models/previous/, so the pre-guard results stay falsifiable.
    """
    from scripts.stamp import read_stamped_csv
    from scripts.train_1b import run_once

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(here, "out", "1b_test_runs_PREVIOUS.csv")
    model_dir = os.path.join(here, "out", "models", "previous")
    if not os.path.exists(csv_path) or not os.path.isdir(model_dir):
        pytest.skip("CHECKPOINT-11 artefacts are not present")

    rows = read_stamped_csv(csv_path)
    checked = 0

    with loop_guard("previous"):
        for rule, init_seed in (("rate", -1), ("rl", 0)):
            expected = next(
                r for r in rows
                if r["rule"] == rule and r["load_ms"] == "200"
                and r["placement_seed"] == "1"
                and int(r["init_seed"]) == init_seed
            )
            kwargs: Dict[str, Any] = {}
            if rule == "rl":
                import torch

                kwargs["state_dict"] = torch.load(
                    os.path.join(model_dir, f"rl_seed{init_seed}.pt"),
                    map_location="cpu",
                )
                kwargs["init_seed"] = init_seed
            result = run_once(routing=rule, placement_seed=1, run_seed=1,
                              load_ms=200, training=False, **kwargs)
            assert result["total_loss"] == pytest.approx(
                float(expected["total_loss"]), abs=1e-12
            ), f"{rule} no longer reproduces its CHECKPOINT-11 result"
            checked += 1

    assert checked == 2
