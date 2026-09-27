"""Channel-model tests (spec item J, third bullet).

p_loss must be a probability everywhere, must be exactly 1 where no link
exists, and must be negligible on a short link.
"""

from __future__ import annotations

import math

import pytest

from fanet_sim import config
from fanet_sim.envs import channel


# Distances spanning collocated, short, mid, the range edge, and out of range.
DISTANCES = [0.0, 1e-12, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 249.0,
             channel.MAX_LINK_DISTANCE_M, 250.0, 400.0, 2000.0]


@pytest.mark.parametrize("dist", DISTANCES)
@pytest.mark.parametrize("k", [0.2, 0.4, 0.8])
def test_p_loss_is_a_probability(dist: float, k: float) -> None:
    """p_loss stays inside [0, 1] for every distance and every k."""
    value = channel.p_loss(dist, k=k)
    assert 0.0 <= value <= 1.0, f"p_loss({dist}, k={k}) = {value}"
    assert not math.isnan(value)


@pytest.mark.parametrize("dist", [channel.MAX_LINK_DISTANCE_M, 250.0, 400.0, 2000.0])
def test_p_loss_is_one_without_a_link(dist: float) -> None:
    """At or beyond the range edge the margin is <= 0 and p_loss is exactly 1."""
    assert channel.link_margin_db(dist) <= 0.0
    assert channel.p_loss(dist) == 1.0


def test_p_loss_is_negligible_at_10m() -> None:
    """A 10 m link loses less than 0.1% of transmissions at the default k."""
    assert channel.p_loss(10.0) < 0.001


def test_p_loss_is_monotonic_in_distance() -> None:
    """Loss never decreases as the endpoints move apart."""
    values = [channel.p_loss(d) for d in [1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 240.0]]
    assert values == sorted(values)


def test_p_loss_matches_the_formula_inside_range() -> None:
    """p_loss is exactly exp(-k * margin) where a link exists."""
    k = config.CHANNEL_LOSS_K
    for dist in [10.0, 50.0, 100.0, 200.0]:
        margin = channel.link_margin_db(dist)
        assert margin > 0.0
        assert channel.p_loss(dist, k=k) == pytest.approx(math.exp(-k * margin))


def test_larger_k_means_less_loss() -> None:
    """k scales how fast loss decays with margin, so bigger k loses less."""
    assert channel.p_loss(100.0, k=0.8) < channel.p_loss(100.0, k=0.4)
    assert channel.p_loss(100.0, k=0.4) < channel.p_loss(100.0, k=0.2)
