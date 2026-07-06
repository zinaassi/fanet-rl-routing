"""Channel-model math: pinned endpoints, midpoint, monotonicity, saturation."""
import numpy as np
import pytest

from stage1 import channel, config


def test_ploss_endpoints_pinned_for_all_k():
    """The rescaling maps 0 m -> 0 and COMM_RANGE_M -> 1 exactly, every k."""
    for k in config.K_SWEEP:
        assert channel.p_loss(0.0, k) == 0.0
        assert channel.p_loss(config.COMM_RANGE_M, k) == 1.0


def test_ploss_midpoint_is_half_for_all_k():
    for k in config.K_SWEEP:
        assert channel.p_loss(config.COMM_RANGE_M / 2.0, k) == pytest.approx(0.5)


def test_ploss_strictly_monotonic_within_range():
    d = np.linspace(0.0, config.COMM_RANGE_M, 1000)
    for k in config.K_SWEEP:
        pl = np.asarray(channel.p_loss(d, k))
        assert np.all(np.diff(pl) > 0.0)


def test_ploss_saturates_at_one_beyond_range():
    d = np.array([config.COMM_RANGE_M + 1e-9, 300.0, 1000.0, 1e6])
    for k in config.K_SWEEP:
        assert np.all(np.asarray(channel.p_loss(d, k)) == 1.0)


def test_larger_k_is_steeper_around_the_midpoint():
    """Steeper k => lower loss on short links, higher loss on long links."""
    short, long_ = 50.0, 200.0
    ks = sorted(config.K_SWEEP)
    p_short = [float(channel.p_loss(short, k)) for k in ks]
    p_long = [float(channel.p_loss(long_, k)) for k in ks]
    assert p_short == sorted(p_short, reverse=True)
    assert p_long == sorted(p_long)


def test_short_links_are_finite_and_nearly_lossless():
    pl = np.asarray(channel.p_loss(np.array([0.0, 1e-9, 1.0]), config.K_SWEEP[-1]))
    assert np.all(np.isfinite(pl))
    assert pl[2] < 1e-3
    for k in config.K_SWEEP:
        assert channel.p_loss(1.0, k) < 0.01


def test_ploss_symmetric_about_midpoint():
    """p(mid - x) + p(mid + x) == 1: the rescaled logistic keeps its symmetry."""
    mid = config.COMM_RANGE_M / 2.0
    for k in config.K_SWEEP:
        for x in (10.0, 50.0, 100.0):
            total = float(channel.p_loss(mid - x, k)) + float(channel.p_loss(mid + x, k))
            assert total == pytest.approx(1.0)
