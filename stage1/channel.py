"""FSPL-margin channel with exponential packet loss on a hard range.

    P_rx(d)   = P_tx + G_tx + G_rx - 20*log10(d) - 20*log10(f) + 147.55  [dBm]
    P_sens    = P_rx(RANGE_M)      (derived: the margin is 0 exactly at RANGE_M)
    M(d)      = P_rx(d) - P_sens = 20*log10(RANGE_M / d)                 [dB]
    p_loss(d) = exp(-k * M(d))     valid ONLY for 0 < d < RANGE_M

Properties (unit-tested):
    p_loss(d) -> 1 as d -> RANGE_M from below   (M -> 0)
    p_loss(d) -> 0 as d -> 0                    (M -> +inf)
    p_loss strictly increasing in d on (0, RANGE_M)

Domain: for d >= RANGE_M the margin is <= 0, so exp(-k*M) >= 1 — not a
valid probability. The hard-range rule in ``world.candidate_graph`` (an
edge exists iff d < RANGE_M) is therefore REQUIRED by this formula, not
redundant with it: it is what keeps p_loss inside its valid domain.
``p_loss`` raises ``ValueError`` if evaluated at d >= RANGE_M so the
invariant cannot be violated silently.

All functions are numpy-vectorised and accept scalars or arrays; the
optional ``range_m`` argument (used by the calibration sweep in
``stage1/calibrate.py``) overrides ``config.RANGE_M``.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from . import config

FloatOrArray = Union[float, np.ndarray]

# Distances are clamped to this floor so log10 never sees zero (co-located
# nodes would otherwise produce an infinite margin; the clamp keeps
# p_loss(0) == p_loss(1e-6 m) ~ 0, preserving the d -> 0 limit).
_MIN_DISTANCE_M: float = 1e-6


def received_power_dbm(distance_m: FloatOrArray) -> np.ndarray:
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


def p_sens_dbm(range_m: Optional[float] = None) -> float:
    """Receiver sensitivity implied by the hard range: P_sens = P_rx(range_m).

    Solving M(range_m) = 0 for P_sens — the same FSPL calibration approach
    as the original -54 dBm @ 250 m assumption, but parameterised so the
    calibration sweep can derive P_sens for any candidate range.
    """
    return float(received_power_dbm(range_m if range_m is not None else config.RANGE_M))


def margin_db(distance_m: FloatOrArray, range_m: Optional[float] = None) -> np.ndarray:
    """Link margin M(d) = P_rx(d) - P_sens = 20*log10(range_m / d) [dB].

    Positive inside the range, exactly zero at d = range_m.
    """
    return received_power_dbm(distance_m) - p_sens_dbm(range_m)


def p_loss(
    distance_m: FloatOrArray, k: float, range_m: Optional[float] = None
) -> np.ndarray:
    """Per-attempt packet-loss probability exp(-k * M(d)), for 0 < d < range_m.

    Raises ``ValueError`` for any d >= range_m: there the margin is <= 0 and
    exp(-k*M) >= 1 is not a probability. Callers must apply the hard-range
    rule (no link at d >= range_m) BEFORE evaluating this function.
    """
    r = float(range_m if range_m is not None else config.RANGE_M)
    d = np.asarray(distance_m, dtype=float)
    if np.any(d >= r):
        raise ValueError(
            f"p_loss is undefined for d >= RANGE_M ({r:g} m): out-of-range "
            "links must be excluded by the hard-range rule, never evaluated"
        )
    return np.exp(-k * margin_db(d, r))
