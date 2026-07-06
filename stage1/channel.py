"""Free-space path-loss channel with a logistic packet-loss curve.

    P_rx(d) = P_tx + G_tx + G_rx - 20*log10(d) - 20*log10(f) + 147.55   [dBm]
    M(d)    = P_rx(d) - P_sens                                          [dB]
    p_loss(d) = 1 / (1 + exp(k * M(d)))

With the Stage-1 constants the margin crosses zero at d ~= 249.6 m, so
p_loss(250 m) ~= 0.5 for every k. All functions are numpy-vectorised and
accept scalars or arrays.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from . import config

FloatOrArray = Union[float, np.ndarray]

# Distances are clamped to this floor so log10 never sees zero (co-located
# nodes would otherwise produce -inf path loss).
_MIN_DISTANCE_M: float = 1e-6


def received_power_dbm(distance_m: FloatOrArray) -> FloatOrArray:
    """Received power [dBm] at the given link distance [m] (Friis / FSPL)."""
    d = np.maximum(np.asarray(distance_m, dtype=float), _MIN_DISTANCE_M)
    return (
        config.P_TX_DBM
        + config.G_TX_DBI
        + config.G_RX_DBI
        - 20.0 * np.log10(d)
        - 20.0 * np.log10(config.FREQ_HZ)
        + 147.55
    )


def margin_db(distance_m: FloatOrArray) -> FloatOrArray:
    """Link margin M(d) = P_rx(d) - P_sens [dB]; zero at ~249.6 m."""
    return received_power_dbm(distance_m) - config.P_SENS_DBM


def p_loss(distance_m: FloatOrArray, k: float) -> FloatOrArray:
    """Per-attempt packet-loss probability, logistic in the link margin.

    Written as 0.5*(1 - tanh(k*M/2)), which equals 1/(1+exp(k*M)) but does
    not overflow for very short links (large positive margins).
    """
    m = margin_db(distance_m)
    return 0.5 * (1.0 - np.tanh(0.5 * k * np.asarray(m, dtype=float)))


def max_link_range_m(k: float, cutoff: float = config.P_LOSS_CUTOFF) -> float:
    """Largest distance whose p_loss does not exceed ``cutoff``.

    Inverts the logistic: p_loss(d) = cutoff  <=>  k*M(d) = log((1-c)/c).
    """
    margin_at_cutoff = np.log((1.0 - cutoff) / cutoff) / k
    const = (
        config.P_TX_DBM
        + config.G_TX_DBI
        + config.G_RX_DBI
        - 20.0 * np.log10(config.FREQ_HZ)
        + 147.55
        - config.P_SENS_DBM
    )
    return float(10.0 ** ((const - margin_at_cutoff) / 20.0))
