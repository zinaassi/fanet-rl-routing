"""Channel-model tests (Checkpoint 7, item 4).

The loss curve is logistic in distance, fitted to measured UAV links in Rosati
et al. (arXiv:1406.4399). It is a probability everywhere, rises with distance,
and approaches but never exceeds 1. A link exists while loss is below
``config.LINK_MAX_LOSS``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs import channel

#: Distances spanning short links, the knee, the cutoff and far beyond.
DISTANCES = [0.0, 1e-12, 1.0, 10.0, 50.0, 100.0, 200.0, 250.0, 270.0, 300.0,
             356.0, 400.0, 450.0, 500.0, 539.8, 636.4, 800.0, 2000.0, 1e6]

#: Values the supervisor gave for checking the curve.
EXPECTED = {
    100.0: 0.0017,
    200.0: 0.0198,
    250.0: 0.066,
    270.0: 0.104,
    300.0: 0.198,
    356.0: 0.5,
    400.0: 0.750,
    540.0: 0.990,
}


@pytest.mark.parametrize("dist", DISTANCES)
def test_p_loss_is_a_probability(dist: float) -> None:
    """p_loss stays inside [0, 1] at every distance."""
    value = channel.p_loss(dist)
    assert 0.0 <= value <= 1.0, f"p_loss({dist}) = {value}"
    assert not math.isnan(value)


def test_p_loss_rises_with_distance() -> None:
    """Loss is strictly increasing in distance."""
    values = [channel.p_loss(d) for d in
              [0.0, 50.0, 100.0, 200.0, 300.0, 356.0, 400.0, 500.0, 636.0]]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "the curve should be strict"


def test_p_loss_is_exactly_one_half_at_d50() -> None:
    """p_loss(356) = 0.5 — the fitted midpoint."""
    assert channel.p_loss(config.LOSS_50_DISTANCE_M) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("dist,expected", sorted(EXPECTED.items()))
def test_p_loss_matches_the_fitted_values(dist: float, expected: float) -> None:
    """The curve reproduces the values from the paper's fit."""
    assert channel.p_loss(dist) == pytest.approx(expected, abs=1e-3)


def test_p_loss_matches_the_formula() -> None:
    """p_loss is exactly the logistic 1 / (1 + exp(-s*(d - D50)))."""
    d50 = config.LOSS_50_DISTANCE_M
    slope = config.LOSS_SLOPE_PER_M
    for dist in [10.0, 100.0, 250.0, 356.0, 450.0, 600.0]:
        expected = 1.0 / (1.0 + math.exp(-slope * (dist - d50)))
        assert channel.p_loss(dist) == pytest.approx(expected, rel=1e-12)


def test_p_loss_approaches_one_but_never_exceeds_it() -> None:
    """Loss tends to 1 at great distance and never goes past it."""
    far = [600.0, 800.0, 1200.0, 5000.0, 1e6]
    values = [channel.p_loss(d) for d in far]
    assert values == sorted(values)
    assert all(v <= 1.0 for v in values)
    assert channel.p_loss(1200.0) > 0.999


def test_p_loss_approaches_zero_at_touching_range() -> None:
    """Loss tends to 0 for a very short link, and never goes below it."""
    assert channel.p_loss(0.0) >= 0.0
    assert channel.p_loss(0.0) < 0.001
    assert channel.p_loss(-1000.0) >= 0.0   # the formula is defined there too


def test_the_curve_is_unlike_the_model_it_replaced() -> None:
    """At 250 m the old model lost ~everything; the new one loses ~6.6%.

    This is the whole point of the change, so it gets an explicit guard.
    """
    assert channel.p_loss(250.0) < 0.10
    assert channel.p_loss(250.0) == pytest.approx(0.066, abs=1e-3)


# ---------------------------------------------------------------------------
# Inverse and link existence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss", [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
def test_distance_for_loss_inverts_p_loss(loss: float) -> None:
    """distance_for_loss and p_loss are inverses of each other."""
    dist = channel.distance_for_loss(loss)
    assert channel.p_loss(dist) == pytest.approx(loss, abs=1e-9)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_distance_for_loss_rejects_impossible_targets(bad: float) -> None:
    """The curve has no finite inverse at or outside 0 and 1."""
    with pytest.raises(ValueError):
        channel.distance_for_loss(bad)


def test_max_link_distance_is_where_loss_hits_the_cutoff() -> None:
    """The link edge sits exactly at the configured maximum loss.

    Derived from the cutoff rather than pinned to one distance, so changing
    LINK_MAX_LOSS moves the edge instead of breaking the test.
    """
    edge = channel.max_link_distance()
    assert channel.p_loss(edge) == pytest.approx(config.LINK_MAX_LOSS, abs=1e-9)


@pytest.mark.parametrize("cutoff,reach", [(0.99, 539.8), (0.5, 356.0), (0.2, 300.5)])
def test_the_reach_of_each_candidate_cutoff(cutoff: float, reach: float) -> None:
    """The three cutoffs in the sensitivity study reach the distances expected."""
    assert channel.max_link_distance(cutoff) == pytest.approx(reach, abs=0.1)


def test_a_half_loss_cutoff_reaches_exactly_the_curve_midpoint() -> None:
    """At the configured 0.5 cutoff the edge is D50 itself.

    A drone lists a neighbour only when at least half its hello messages get
    through, which by construction is the 50% point of the curve.
    """
    assert config.LINK_MAX_LOSS == 0.5, "this test documents the configured cutoff"
    assert channel.max_link_distance() == pytest.approx(
        config.LOSS_50_DISTANCE_M, abs=1e-9
    )


def test_a_link_exists_exactly_while_loss_is_under_the_cutoff() -> None:
    """link_exists is p_loss < LINK_MAX_LOSS, with nothing else involved."""
    for dist in DISTANCES:
        assert channel.link_exists(dist) == (
            channel.p_loss(dist) < config.LINK_MAX_LOSS
        )


def test_the_link_edge_is_a_clean_boundary() -> None:
    """Just inside the edge there is a link; just outside there is not."""
    edge = channel.max_link_distance()
    assert channel.link_exists(edge - 0.1)
    assert not channel.link_exists(edge + 0.1)
    assert not channel.link_exists(edge)       # loss == cutoff is not "under"


def test_are_connected_uses_the_same_rule_as_link_exists() -> None:
    """The position-based test agrees with the distance-based one."""
    origin = np.array([0.0, 0.0])
    for dist in [10.0, 250.0, 400.0, 539.0, 540.0, 700.0]:
        other = np.array([dist, 0.0])
        assert channel.are_connected(origin, other) == channel.link_exists(dist)


def test_link_quality_is_the_chance_a_send_gets_through() -> None:
    """link_quality is 1 - p_loss, and falls as distance grows."""
    for dist in [10.0, 100.0, 250.0, 356.0, 500.0]:
        assert channel.link_quality(dist) == pytest.approx(
            1.0 - channel.p_loss(dist)
        )
    assert channel.link_quality(356.0) == pytest.approx(0.5, abs=1e-6)
    qualities = [channel.link_quality(d) for d in [10.0, 200.0, 400.0, 600.0]]
    assert qualities == sorted(qualities, reverse=True)


def test_the_curve_parameters_are_the_configured_ones() -> None:
    """Passing the config values explicitly changes nothing."""
    for dist in [100.0, 356.0, 540.0]:
        assert channel.p_loss(
            dist,
            d50=config.LOSS_50_DISTANCE_M,
            slope=config.LOSS_SLOPE_PER_M,
        ) == pytest.approx(channel.p_loss(dist))


def test_a_shallower_slope_spreads_the_curve_out() -> None:
    """The slope controls how abruptly loss climbs around D50."""
    d50 = config.LOSS_50_DISTANCE_M
    # Below D50 a shallower curve loses more; above it, less.
    assert channel.p_loss(250.0, slope=0.010) > channel.p_loss(250.0, slope=0.025)
    assert channel.p_loss(450.0, slope=0.010) < channel.p_loss(450.0, slope=0.025)
    # And both still pass through 0.5 at D50.
    assert channel.p_loss(d50, slope=0.010) == pytest.approx(0.5, abs=1e-9)


def test_moving_d50_shifts_the_whole_curve() -> None:
    """A larger D50 means less loss at every distance."""
    for dist in [100.0, 250.0, 400.0]:
        assert channel.p_loss(dist, d50=450.0) < channel.p_loss(dist, d50=356.0)
