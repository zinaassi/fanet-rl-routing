"""Channel-model math: calibration point, monotonicity, cutoff behaviour."""
import numpy as np
import pytest

from stage1 import channel, config


def test_ploss_at_250m_is_half_for_all_k():
    for k in config.K_SWEEP:
        assert channel.p_loss(250.0, k) == pytest.approx(0.5, abs=5e-3)


def test_ploss_strictly_monotonic_in_distance():
    d = np.linspace(1.0, 3000.0, 1000)
    for k in config.K_SWEEP:
        pl = np.asarray(channel.p_loss(d, k))
        assert np.all(np.diff(pl) > 0.0)


def test_margin_crosses_zero_near_250m():
    assert channel.margin_db(249.0) > 0.0 > channel.margin_db(250.0)


def test_received_power_spot_value():
    expected = (
        config.P_TX_DBM + config.G_TX_DBI + config.G_RX_DBI
        - 20.0 * np.log10(100.0) - 20.0 * np.log10(config.FREQ_HZ) + 147.55
    )
    assert channel.received_power_dbm(100.0) == pytest.approx(expected)


def test_max_link_range_brackets_the_cutoff():
    for k in config.K_SWEEP:
        r = channel.max_link_range_m(k)
        assert channel.p_loss(0.999 * r, k) < config.P_LOSS_CUTOFF
        assert channel.p_loss(1.001 * r, k) > config.P_LOSS_CUTOFF


def test_max_link_range_shrinks_with_k():
    ranges = [channel.max_link_range_m(k) for k in sorted(config.K_SWEEP)]
    assert ranges == sorted(ranges, reverse=True)


def test_short_links_are_finite_and_lossless():
    pl = np.asarray(channel.p_loss(np.array([0.0, 1e-9, 1.0]), 1.0))
    assert np.all(np.isfinite(pl))
    assert pl[2] < 1e-6
