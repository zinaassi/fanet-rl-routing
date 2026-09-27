"""Channel-model math: exp(-k*M) properties, valid domain, FSPL calibration."""
import numpy as np
import pytest

from stage1 import channel, config


def test_ploss_tends_to_one_at_range_edge():
    """p_loss(d) -> 1 as d -> RANGE_M from below (M -> 0)."""
    for k in config.K_SWEEP:
        just_inside = config.RANGE_M * (1.0 - 1e-12)
        assert float(channel.p_loss(just_inside, k)) == pytest.approx(1.0, abs=1e-9)
        assert float(channel.p_loss(0.999 * config.RANGE_M, k)) > 0.99


def test_ploss_tends_to_zero_at_zero_distance():
    """p_loss(d) -> 0 as d -> 0 (M -> +inf)."""
    for k in config.K_SWEEP:
        assert float(channel.p_loss(0.0, k)) < 1e-3
    seq = [float(channel.p_loss(d, min(config.K_SWEEP))) for d in (100.0, 10.0, 1.0, 0.01)]
    assert seq == sorted(seq, reverse=True)  # monotone approach to 0


def test_ploss_strictly_increasing_within_range():
    d = np.linspace(0.5, config.RANGE_M * 0.9999, 1000)
    for k in config.K_SWEEP:
        pl = np.asarray(channel.p_loss(d, k))
        assert np.all(np.diff(pl) > 0.0)
        assert np.all((pl > 0.0) & (pl < 1.0))


def test_ploss_undefined_at_and_beyond_range():
    """d >= RANGE_M is outside the valid domain and must never be evaluated:
    the margin is <= 0 there and exp(-k*M) >= 1 is not a probability."""
    for d in (config.RANGE_M, config.RANGE_M + 1e-9, 2.0 * config.RANGE_M):
        with pytest.raises(ValueError):
            channel.p_loss(d, 0.1)
    with pytest.raises(ValueError):  # arrays too, even if only one entry is out
        channel.p_loss(np.array([10.0, config.RANGE_M]), 0.1)


def test_margin_zero_exactly_at_range():
    assert float(channel.margin_db(config.RANGE_M)) == 0.0
    assert float(channel.margin_db(0.5 * config.RANGE_M)) > 0.0
    assert float(channel.margin_db(2.0 * config.RANGE_M)) < 0.0


def test_p_sens_derivation_matches_legacy_250m_value():
    """Deriving P_sens from a 250 m range reproduces the original -54 dBm
    FSPL calibration."""
    assert channel.p_sens_dbm(250.0) == pytest.approx(-54.0, abs=0.1)


def test_received_power_spot_value():
    expected = (
        config.P_TX_DBM + config.G_TX_DBI + config.G_RX_DBI
        - 20.0 * np.log10(100.0) - 20.0 * np.log10(config.FREQ_HZ) + 147.55
    )
    assert float(channel.received_power_dbm(100.0)) == pytest.approx(expected)


def test_custom_range_parameter():
    """The calibration sweep evaluates candidate ranges via range_m."""
    p = float(channel.p_loss(300.0, 0.1, range_m=400.0))
    assert 0.0 < p < 1.0
    with pytest.raises(ValueError):
        channel.p_loss(config.RANGE_M, 0.1)  # invalid at/beyond the default RANGE_M


def test_larger_k_gives_cleaner_short_links():
    """Inside the range M > 0, so a larger decay k means lower loss."""
    ks = sorted(config.K_SWEEP)
    for d in (50.0, 125.0, 200.0):
        pl = [float(channel.p_loss(d, k)) for k in ks]
        assert pl == sorted(pl, reverse=True)


def test_ploss_equals_power_law_form():
    """exp(-k * 20*log10(R/d)) == (d/R)^(20k/ln10) — handy closed form."""
    d, k = 100.0, 0.1
    expected = (d / config.RANGE_M) ** (20.0 * k / np.log(10.0))
    assert float(channel.p_loss(d, k)) == pytest.approx(expected)
