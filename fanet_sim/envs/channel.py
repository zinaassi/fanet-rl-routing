"""
channel.py — Free-Space Path Loss (FSPL) channel model.

Replaces the previous distance-cutoff model. Link existence is no longer a
fixed 250 m sphere; it is now determined by whether the received signal
power clears the receiver sensitivity threshold.

Assumptions
-----------
* 2-D free-space propagation. No obstruction, no terrain, no multipath,
  no fading. "Drones can see each other" means "received FSPL signal
  >= receiver sensitivity".
* No MAC layer, no interference model, no SNR-based bit-rate adaptation.
  These are deferred to a later phase.
* All drones use identical radios (same Pt, Gt, Gr, sensitivity). The GS
  uses the same parameters too — it is treated as just another endpoint
  for the FSPL test.

Parameter choices (anchored to IQMR (Sharvari et al., 2024, arXiv:2408.09109))
-----------------------------------------------------------------------------
    PT_DBM                  =  30 dBm    1 W — matches IQMR transmit power
    GT_DBI = GR_DBI         =   2 dBi    small omnidirectional dipole
    F_HZ                    = 2.4 GHz    common ISM / UAV telemetry band
    RX_SENSITIVITY_DBM      = -54 dBm    derived (not picked) — see below
    MARGIN_FULL_QUALITY_DB  =  30 dB     link_quality saturates to 1.0
                                          once Pr exceeds sensitivity by
                                          this margin

Derivation of sensitivity
-------------------------
IQMR reports a radio range of 250 m at 1 W transmit power. To reproduce
that range under our FSPL model with Pt = 30 dBm, Gt = Gr = 2 dBi,
f = 2.4 GHz, the receiver sensitivity is fully determined:

    FSPL(250 m, 2.4 GHz)  = 20*log10(250) + 20*log10(2.4e9) + 20*log10(4*pi/c)
                          = 47.96 + 187.60 + (-147.56)
                          = 88.00 dB
    sensitivity required  = Pt + Gt + Gr - FSPL = 30 + 2 + 2 - 88 = -54 dBm

So -54 dBm is the sensitivity that makes our FSPL link-existence test
agree with IQMR's stated 250 m range at IQMR's 1 W transmit power. This
is now an IQMR-derived value, not an arbitrary radio-class pick.

Link budget = PT_DBM + GT_DBI + GR_DBI - RX_SENSITIVITY_DBM
            = 30 + 2 + 2 - (-54) = 88 dB
Effective max link distance with these parameters is ~250 m (printed at
startup via channel.MAX_LINK_DISTANCE_M), matching IQMR's reported range.

FSPL formula
------------
    FSPL_dB(d) = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c)
    Pr_dBm     = Pt + Gt + Gr - FSPL_dB
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Radio parameters — change these to retune the channel.
# ---------------------------------------------------------------------------
PT_DBM: float = 30.0               # transmit power (dBm) — 1 W, matches IQMR
GT_DBI: float = 2.0                # transmit antenna gain (dBi)
GR_DBI: float = 2.0                # receive antenna gain (dBi)
F_HZ: float = 2.4e9                # carrier frequency (Hz)
RX_SENSITIVITY_DBM: float = -54.0  # receiver sensitivity (dBm) — derived to give 250 m at IQMR's 1 W
MARGIN_FULL_QUALITY_DB: float = 30.0  # margin at which link_quality saturates

# Physical constant — speed of light in vacuum (m/s).
SPEED_OF_LIGHT_M_S: float = 299_792_458.0


# ---------------------------------------------------------------------------
# Derived constants (do not edit directly — driven by the parameters above)
# ---------------------------------------------------------------------------

# FSPL_dB(d) = 20*log10(d) + _FSPL_FREQ_CONST_DB
_FSPL_FREQ_CONST_DB: float = (
    20.0 * math.log10(F_HZ)
    + 20.0 * math.log10(4.0 * math.pi / SPEED_OF_LIGHT_M_S)
)

# Total dB of path loss the link can tolerate.
LINK_BUDGET_DB: float = PT_DBM + GT_DBI + GR_DBI - RX_SENSITIVITY_DBM

# Effective max link distance (m): distance at which Pr == sensitivity exactly.
# This is the new source of truth for "can two endpoints communicate" and
# replaces the old config.COMM_RANGE in every link-existence test.
MAX_LINK_DISTANCE_M: float = 10.0 ** ((LINK_BUDGET_DB - _FSPL_FREQ_CONST_DB) / 20.0)


# ---------------------------------------------------------------------------
# FSPL math
# ---------------------------------------------------------------------------

def fspl_db(dist_m: float) -> float:
    """Free-space path loss in dB at distance *dist_m*.

    Returns ``-inf`` for distances at or below 1e-9 m (collocated endpoints)
    so the link is treated as unconditionally available rather than dividing
    by zero. Callers should not pass negative distances.

    Args:
        dist_m: Euclidean separation between transmitter and receiver (m).

    Returns:
        Path loss in decibels.
    """
    if dist_m <= 1e-9:
        return -float("inf")
    return 20.0 * math.log10(dist_m) + _FSPL_FREQ_CONST_DB


def received_power_dbm(dist_m: float) -> float:
    """Received signal power at the receiver, in dBm.

    Pr = Pt + Gt + Gr - FSPL(d).

    Args:
        dist_m: Euclidean separation between transmitter and receiver (m).

    Returns:
        Received power in dBm. ``+inf`` for collocated endpoints.
    """
    return PT_DBM + GT_DBI + GR_DBI - fspl_db(dist_m)


# ---------------------------------------------------------------------------
# Link existence + quality (public API used by drone.py, env, metrics)
# ---------------------------------------------------------------------------

def link_quality(dist_m: float) -> float:
    """Link quality in [0, 1], monotonically non-increasing in distance.

    Derived from the SNR margin above receiver sensitivity:

        margin  = Pr - RX_SENSITIVITY_DBM   (in dB)
        quality = clamp(margin / MARGIN_FULL_QUALITY_DB, 0, 1)

    A quality of 0 means the link does not exist (margin <= 0).
    A quality of 1 means the link has at least MARGIN_FULL_QUALITY_DB of
    headroom above sensitivity.

    Args:
        dist_m: Euclidean distance between endpoints (m).

    Returns:
        Float in [0.0, 1.0].
    """
    margin = received_power_dbm(dist_m) - RX_SENSITIVITY_DBM
    if margin <= 0.0:
        return 0.0
    if margin >= MARGIN_FULL_QUALITY_DB:
        return 1.0
    return margin / MARGIN_FULL_QUALITY_DB


def are_connected(pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
    """True iff the received signal between two positions clears sensitivity.

    This is the FSPL-based replacement for the old ``distance < COMM_RANGE``
    test. Use this anywhere a binary link-existence answer is needed.

    Args:
        pos_a: Position of endpoint A as a NumPy array.
        pos_b: Position of endpoint B as a NumPy array.

    Returns:
        True if Pr(d) >= RX_SENSITIVITY_DBM, else False.
    """
    dist = float(np.linalg.norm(pos_a - pos_b))
    return received_power_dbm(dist) >= RX_SENSITIVITY_DBM


def euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Return the euclidean distance between two position arrays (m).

    Args:
        pos_a: Position vector (x, y) or (x, y, z).
        pos_b: Position vector of the same dimension.

    Returns:
        Non-negative float distance in metres.
    """
    return float(np.linalg.norm(pos_a - pos_b))
