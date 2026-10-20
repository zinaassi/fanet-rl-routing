"""
test_2a_reference.py — The static world must survive Phase 2a untouched.

Phase 2a makes the M-drones fly. Everything it adds has to leave the static
world exactly where it was, or every Phase-1 number stops meaning anything.
Two checks, both against files committed BEFORE any Phase-2a code existed:

* the layout hash — every drone's start position, from the placement stream;
* the results — greedy and random at 200 ms on the 20 Phase-1a layouts.

The layout hash is the sharper of the two, and it is the one that catches the
mistake Phase 2a is most likely to make: drawing a direction, a speed or a
start delay from the placement stream instead of the mobility stream. One
extra draw there shifts every position after it, and the digest moves.

Both checks run with movement OFF, which is what the reference recorded. The
"movement on reproduces movement off" check lives in tests/test_2a_mobility.py
and is a different question.
"""

from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv

#: Checked one layout at a time rather than all twenty: a full sweep is a
#: couple of minutes, and a stream shift moves every seed, not just one.
CHECKED_SEEDS = (1, 7, 20)

#: The rules the reference holds. Both are re-run for the checked seeds.
RULES = ("greedy", "random")


def _repo_out(name: str) -> str:
    """Return the path of *name* inside the project's out/ directory."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "out", name)


def _reference_rows():
    """Return the frozen reference rows, or skip if they are not present."""
    from scripts.stamp import read_stamped_csv

    path = _repo_out("2a_reference.csv")
    if not os.path.exists(path):
        pytest.skip("out/2a_reference.csv is missing; run "
                    "scripts/reference_2a.py to record it")
    return read_stamped_csv(path)


def _hash_rows():
    """Return the frozen layout hashes, or skip if they are not present."""
    from scripts.stamp import read_stamped_csv

    path = _repo_out("2a_layout_hash.csv")
    if not os.path.exists(path):
        pytest.skip("out/2a_layout_hash.csv is missing; run "
                    "scripts/reference_2a.py to record it")
    return read_stamped_csv(path)


def _layout_hash(seed: int) -> str:
    """Return the SHA-256 over every drone's start position for one layout."""
    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=seed, run_seed=seed)
    env.reset()
    drones = sorted(env.drones, key=lambda d: d.drone_id)
    positions = np.array([d.position for d in drones], dtype=np.float64)
    env.close_logger()
    return hashlib.sha256(positions.tobytes()).hexdigest()


@pytest.mark.parametrize("seed", CHECKED_SEEDS)
def test_the_layout_is_the_one_phase_1_drew(seed: int) -> None:
    """Every drone still starts where the placement stream put it.

    This is what fails first if mobility randomness is drawn from the
    placement stream: one extra draw shifts every position after it.
    """
    rows = _hash_rows()
    expected = next(
        r for r in rows if int(r["placement_seed"]) == seed
    )
    assert _layout_hash(seed) == expected["positions_sha256"], (
        f"layout {seed} is not the one the reference recorded — something "
        f"is drawing from the placement stream that did not before"
    )


@pytest.mark.parametrize("seed", CHECKED_SEEDS)
@pytest.mark.parametrize("rule", RULES)
def test_the_static_results_are_unchanged(rule: str, seed: int) -> None:
    """greedy and random reproduce the frozen reference to 1e-12.

    Run with movement off, which is the world the reference recorded.
    """
    from scripts.metrics_1a import compute_metrics

    rows = _reference_rows()
    expected = next(
        r for r in rows
        if r["rule"] == rule and int(r["placement_seed"]) == seed
    )

    assert config.STATIC_MODE, (
        "this check records the STATIC world; config.STATIC_MODE is False"
    )

    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = 2          # the 200 ms reference
    try:
        env = FANETEnv(routing=rule, log_path=os.devnull,
                       placement_seed=seed, run_seed=seed)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()
        metrics = compute_metrics(env)
    finally:
        config.PACKET_INTERVAL_STEPS = previous

    truth = metrics["ground_truth"]
    assert not metrics["discrepancies"], metrics["discrepancies"]

    assert (truth["total_loss"] or 0.0) == pytest.approx(
        float(expected["total_loss"]), abs=1e-12
    ), f"{rule} seed {seed} no longer reproduces the static reference"

    by_reason = truth["loss_rate_by_reason"]
    for cause in ("channel", "queue_full", "no_route", "dead_end",
                  "ttl", "hop_limit"):
        assert by_reason[cause] == pytest.approx(
            float(expected[cause]), abs=1e-12
        ), f"{rule} seed {seed} moved on {cause}"


def test_the_reference_covers_every_phase_1a_layout() -> None:
    """The frozen files describe all 20 layouts and both rules.

    A reference with holes in it would pass the checks above while saying
    nothing about the layouts it left out.
    """
    rows = _reference_rows()
    hashes = _hash_rows()

    seeds = {int(r["placement_seed"]) for r in rows}
    assert seeds == set(range(1, 21)), f"missing layouts: {set(range(1, 21)) - seeds}"
    assert {r["rule"] for r in rows} == set(RULES)
    assert len(rows) == 2 * 20

    assert {int(r["placement_seed"]) for r in hashes} == set(range(1, 21))
    assert all(len(r["positions_sha256"]) == 64 for r in hashes)
    assert all(int(r["num_drones"]) == config.NUM_M_DRONES + config.NUM_C_DRONES
               for r in hashes)
