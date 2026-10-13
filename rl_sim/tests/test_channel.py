"""The channel: loss against distance, its inverse, and what counts as a link.

The model is a free-space path loss budget turned into an S-curve on the
margin above receiver sensitivity. The sensitivity is the 10% loss point, not
a wall, so links past it still exist and simply lose more.

The steepness ``b`` is DERIVED from the link edge rather than written down, so
the tests check that moving the edge moves ``b`` — a hardcoded constant would
silently survive a change to ``config.LOSS_50_DISTANCE_M``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs import channel

#: Distances spanning short links, the sensitivity point, the edge and beyond.
DISTANCES = [0.0, 1e-12, 1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 249.689108,
             250.0, 300.0, 322.0, 350.0, 400.0, 450.0, 500.0, 800.0, 2000.0,
             1e6]

#: Reference values computed independently from the specification. Every one
#: must hold to 1e-5; they are what pins the model down.
REFERENCE = {
    50.0: (13.968592, 0.000003),
    100.0: (7.947992, 0.000288),
    150.0: (4.426167, 0.004019),
    200.0: (1.927392, 0.025558),
    249.689108: (0.000000, 0.100000),
    250.0: (-0.010808, 0.100731),
    300.0: (-1.594433, 0.268367),
    322.0: (-2.209125, 0.367608),
    350.0: (-2.933369, 0.500000),
}

#: The derived constants, also computed independently.
FSPL_CONST_DB = 40.052008
SENSITIVITY_DISTANCE_M = 249.689108
MARGIN_AT_EDGE_DB = -2.933369
B_PER_DB = 0.749045


# ---------------------------------------------------------------------------
# The reference table
# ---------------------------------------------------------------------------

def test_the_fspl_frequency_constant() -> None:
    """The distance-independent part of free-space path loss."""
    assert channel.FSPL_CONST_DB == pytest.approx(FSPL_CONST_DB, abs=1e-5)


def test_the_sensitivity_distance_is_where_the_margin_is_zero() -> None:
    """The 10% loss point sits exactly where received power meets sensitivity."""
    assert channel.SENSITIVITY_DISTANCE_M == pytest.approx(
        SENSITIVITY_DISTANCE_M, abs=1e-5
    )
    assert channel.margin_db(channel.SENSITIVITY_DISTANCE_M) == pytest.approx(
        0.0, abs=1e-9
    )
    assert channel.p_loss(channel.SENSITIVITY_DISTANCE_M) == pytest.approx(
        config.SENS_LOSS, abs=1e-9
    )


def test_a_is_fixed_by_the_sensitivity_definition() -> None:
    """A follows from SENS_LOSS alone: 10% loss at zero margin gives A = 9."""
    assert channel.A == pytest.approx(
        (1.0 - config.SENS_LOSS) / config.SENS_LOSS, rel=1e-12
    )
    assert channel.A == pytest.approx(9.0, abs=1e-9)


def test_the_margin_at_the_edge() -> None:
    """The margin at the link edge, which is what b is derived from."""
    assert channel.margin_db(config.LOSS_50_DISTANCE_M) == pytest.approx(
        MARGIN_AT_EDGE_DB, abs=1e-5
    )


def test_b_matches_the_reference_value() -> None:
    """b is the steepness that puts 50% loss at the edge."""
    assert channel.B_PER_DB == pytest.approx(B_PER_DB, abs=1e-5)


@pytest.mark.parametrize("dist,expected", sorted(REFERENCE.items()))
def test_the_reference_table(dist: float, expected: tuple) -> None:
    """Margin and loss at every reference distance, to 1e-5."""
    margin, loss = expected
    assert channel.margin_db(dist) == pytest.approx(margin, abs=1e-5)
    assert channel.p_loss(dist) == pytest.approx(loss, abs=1e-5)


def test_p_loss_matches_the_formula() -> None:
    """p_loss is exactly 1 / (1 + A * exp(b * margin))."""
    for dist in [10.0, 100.0, 250.0, 350.0, 450.0, 600.0]:
        expected = 1.0 / (
            1.0 + channel.A * math.exp(channel.B_PER_DB * channel.margin_db(dist))
        )
        assert channel.p_loss(dist) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# b is derived, not written down
# ---------------------------------------------------------------------------

def test_b_is_derived_from_the_edge() -> None:
    """Changing the edge changes b, and keeps 50% loss at the new edge.

    A hardcoded b would pass every other test in this file while quietly
    putting the 50% point somewhere other than the configured edge.
    """
    for edge in (300.0, 325.0, 400.0, 450.0):
        assert channel.p_loss(edge, edge_m=edge) == pytest.approx(0.5, abs=1e-9)

    # And the derived steepness really differs between edges.
    steepnesses = [channel._slope_for_edge(e) for e in (300.0, 350.0, 450.0)]
    assert len(set(steepnesses)) == len(steepnesses)
    # A nearer edge means a steeper curve.
    assert steepnesses == sorted(steepnesses, reverse=True)


def test_moving_the_edge_in_config_moves_b() -> None:
    """B_PER_DB tracks config.LOSS_50_DISTANCE_M, not a literal."""
    assert channel.B_PER_DB == pytest.approx(
        math.log(channel.A) / (-channel.margin_db(config.LOSS_50_DISTANCE_M)),
        rel=1e-12,
    )


def test_the_sensitivity_point_does_not_move_with_the_edge() -> None:
    """Whatever the edge, zero margin still means SENS_LOSS."""
    for edge in (300.0, 350.0, 450.0):
        assert channel.p_loss(
            channel.SENSITIVITY_DISTANCE_M, edge_m=edge
        ) == pytest.approx(config.SENS_LOSS, abs=1e-9)


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dist", DISTANCES)
def test_p_loss_is_a_probability(dist: float) -> None:
    """p_loss stays inside [0, 1] at every distance."""
    value = channel.p_loss(dist)
    assert 0.0 <= value <= 1.0, f"p_loss({dist}) = {value}"
    assert not math.isnan(value)


def test_p_loss_is_strictly_increasing_and_inside_the_open_interval() -> None:
    """Loss rises strictly with distance and never reaches 0 or 1 exactly."""
    distances = [1.0, 10.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0,
                 400.0, 500.0]
    values = [channel.p_loss(d) for d in distances]
    assert values == sorted(values)
    assert len(set(values)) == len(values), "the curve should be strict"
    for value in values:
        assert 0.0 < value < 1.0


def test_links_past_the_sensitivity_have_a_negative_margin_and_still_exist() -> None:
    """Between the sensitivity point and the edge the margin is below zero.

    This is the point of the change: the sensitivity is not a wall.
    """
    for dist in (260.0, 300.0, 322.0, 349.0):
        assert channel.margin_db(dist) < 0.0
        assert channel.link_exists(dist)
        assert config.SENS_LOSS < channel.p_loss(dist) < config.LINK_MAX_LOSS


def test_p_loss_approaches_one_but_never_exceeds_it() -> None:
    """Loss stays strictly under 1 at every distance the arena can produce.

    Mathematically the curve only approaches 1. In float64 the exponential
    underflows eventually and p_loss becomes exactly 1.0, but that happens
    around 99 km — about 78 times the arena diagonal, so the simulator can
    never reach it. The diagonal is the longest link that can exist.
    """
    diagonal = math.hypot(config.WIDTH, config.HEIGHT)
    assert channel.p_loss(diagonal) < 1.0
    assert channel.p_loss(diagonal) > 0.99
    assert channel.p_loss(1e6) <= 1.0


def test_p_loss_is_zero_at_touching_range() -> None:
    """At zero distance the margin is infinite, so nothing is lost."""
    assert channel.p_loss(0.0) == 0.0
    assert channel.p_loss(1e-12) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Inverse and link existence
# ---------------------------------------------------------------------------

def test_distance_for_loss_at_the_edge_and_the_sensitivity() -> None:
    """The inverse lands exactly on the two anchored points."""
    assert channel.distance_for_loss(0.5) == pytest.approx(350.0, abs=1e-5)
    assert channel.distance_for_loss(0.1) == pytest.approx(
        SENSITIVITY_DISTANCE_M, abs=1e-5
    )


@pytest.mark.parametrize("loss", [0.05, 0.2, 0.3, 0.45])
def test_p_loss_undoes_distance_for_loss(loss: float) -> None:
    """p_loss(distance_for_loss(p)) == p."""
    assert channel.p_loss(channel.distance_for_loss(loss)) == pytest.approx(
        loss, abs=1e-12
    )


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


def test_a_half_loss_cutoff_reaches_exactly_the_configured_edge() -> None:
    """At the configured 0.5 cutoff the reach is LOSS_50_DISTANCE_M itself."""
    assert config.LINK_MAX_LOSS == 0.5, "this test documents the configured cutoff"
    assert channel.max_link_distance() == pytest.approx(
        config.LOSS_50_DISTANCE_M, abs=1e-9
    )
    assert channel.max_link_distance() == pytest.approx(350.0, abs=1e-5)


def test_a_link_exists_exactly_while_loss_is_under_the_cutoff() -> None:
    """link_exists is p_loss < LINK_MAX_LOSS, with nothing else involved."""
    for dist in DISTANCES:
        assert channel.link_exists(dist) == (
            channel.p_loss(dist) < config.LINK_MAX_LOSS
        )


def test_the_link_edge_is_a_clean_boundary() -> None:
    """A link exists at 349.9 m and not at 350.1 m."""
    assert channel.link_exists(349.9)
    assert not channel.link_exists(350.1)
    assert not channel.link_exists(350.0)      # loss == cutoff is not "under"


def test_are_connected_uses_the_same_rule_as_link_exists() -> None:
    """The position-based test agrees with the distance-based one."""
    origin = np.array([0.0, 0.0])
    for dist in [10.0, 250.0, 349.9, 350.0, 350.1, 400.0, 700.0]:
        other = np.array([dist, 0.0])
        assert channel.are_connected(origin, other) == channel.link_exists(dist)


def test_link_quality_is_the_chance_a_send_gets_through() -> None:
    """link_quality is 1 - p_loss, and falls as distance grows."""
    for dist in [10.0, 100.0, 250.0, 350.0, 500.0]:
        assert channel.link_quality(dist) == pytest.approx(
            1.0 - channel.p_loss(dist)
        )
    assert channel.link_quality(350.0) == pytest.approx(0.5, abs=1e-6)
    qualities = [channel.link_quality(d) for d in [10.0, 200.0, 400.0, 600.0]]
    assert qualities == sorted(qualities, reverse=True)


def test_the_curve_parameters_are_the_configured_ones() -> None:
    """Passing the configured edge explicitly changes nothing."""
    for dist in [100.0, 350.0, 450.0]:
        assert channel.p_loss(
            dist, edge_m=config.LOSS_50_DISTANCE_M
        ) == pytest.approx(channel.p_loss(dist))


def test_a_farther_edge_means_less_loss_everywhere_past_the_sensitivity() -> None:
    """Moving the edge out flattens the curve beyond the sensitivity point."""
    for dist in [300.0, 350.0, 400.0]:
        assert channel.p_loss(dist, edge_m=450.0) < channel.p_loss(
            dist, edge_m=350.0
        )
