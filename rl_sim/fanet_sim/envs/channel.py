"""
channel.py — Packet loss versus distance, and what counts as a link.

Loss model
----------
A free-space path loss budget gives each link a MARGIN above the receiver's
sensitivity, and that margin drives an S-curve for the chance one
transmission is lost:

    FSPL_dB(d) = 20*log10(d) + 20*log10(F_HZ) + 20*log10(4*pi/c)
    M(d)       = PT_DBM + GT_DBI + GR_DBI - FSPL_dB(d) - RX_SENSITIVITY_DBM
    p_loss(M)  = 1 / (1 + A * exp(b * M))

        A = (1 - SENS_LOSS) / SENS_LOSS, so that p_loss is exactly SENS_LOSS
            where the margin is zero
        b = ln(A) / -M(LOSS_50_DISTANCE_M), DERIVED at import time so that
            p_loss is exactly 0.5 at the link edge. Never hardcode it: change
            LOSS_50_DISTANCE_M and b follows.

The receiver sensitivity is NOT a wall. It is the point where 10% of
transmissions are lost, following 802.11 OFDM, which defines sensitivity as
the power giving a 10% packet error rate on 1000-octet frames. With the
configured constants that point is 249.69 m:

    d (m)      50     100     150     200   249.7     250     300     350
    p_loss  0.0000  0.0003  0.0040  0.0256  0.1000  0.1007  0.2684  0.5000

Links between 249.69 m and the edge have a NEGATIVE margin and still exist;
they simply lose more. The same curve governs the last hop into the ground
station — the GS is just another endpoint.

p_loss returns exactly 0.0 at zero distance, where the margin is infinite.
Mathematically it only approaches 1 and never reaches it; in float64 the
exponential underflows around 99 km and it becomes exactly 1.0, which is far
beyond any distance this arena can produce.

Link existence
--------------
A link exists while its loss stays under ``config.LINK_MAX_LOSS``, which puts
the edge at ``max_link_distance()``. That cutoff is 0.5, so the edge is
exactly ``config.LOSS_50_DISTANCE_M``. This is the ONE rule used everywhere:
neighbour sets, the GS link, the connected-layout filter and the range check.
Distance alone decides it, through the curve.

The edge is a design choice, not a physical constant — see CLAUDE.md.

Assumptions
-----------
* 2-D. No altitude, terrain or obstruction.
* No MAC layer, no interference model, no collisions.
* Every endpoint has the same radio, the GS included.
* Loss depends on distance only — not on traffic, antenna orientation or
  interference from other transmissions.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from fanet_sim import config


# ---------------------------------------------------------------------------
# The loss curve
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT = 299_792_458.0

# The distance-independent part of free-space path loss, in dB.
FSPL_CONST_DB = (
    20.0 * math.log10(config.F_HZ)
    + 20.0 * math.log10(4.0 * math.pi / SPEED_OF_LIGHT)
)

# Fixed by the sensitivity definition: p_loss is SENS_LOSS at margin 0.
A = (1.0 - config.SENS_LOSS) / config.SENS_LOSS


def fspl_db(dist_m: float) -> float:
    """Free-space path loss over *dist_m* metres, in dB.

    Args:
        dist_m: Distance between sender and receiver, in metres.

    Returns:
        The path loss in dB. Undefined at 0 m, where it diverges.
    """
    return 20.0 * math.log10(dist_m) + FSPL_CONST_DB


def margin_db(dist_m: float) -> float:
    """How far the received power is above the receiver's sensitivity, in dB.

    Negative past the sensitivity distance. Such a link still exists — it
    simply loses more than ``config.SENS_LOSS`` of what is sent on it.

    Args:
        dist_m: Distance between sender and receiver, in metres.

    Returns:
        The margin in dB.
    """
    return (config.PT_DBM + config.GT_DBI + config.GR_DBI
            - fspl_db(dist_m) - config.RX_SENSITIVITY_DBM)


# Where the margin is exactly zero, i.e. the sensitivity point. Everything
# else is anchored to it, so it is computed once here.
SENSITIVITY_DISTANCE_M = 10.0 ** (
    (config.PT_DBM + config.GT_DBI + config.GR_DBI
     - config.RX_SENSITIVITY_DBM - FSPL_CONST_DB) / 20.0
)


def _slope_for_edge(edge_m: float) -> float:
    """The S-curve steepness b that puts 50% loss at *edge_m*.

    Args:
        edge_m: The distance that should lose half of what is sent on it.

    Returns:
        b, in units of 1/dB.
    """
    return math.log(A) / (-margin_db(edge_m))


# DERIVED, never hardcoded: move LOSS_50_DISTANCE_M and b follows.
B_PER_DB = _slope_for_edge(config.LOSS_50_DISTANCE_M)


def p_loss(dist_m: float, edge_m: Optional[float] = None) -> float:
    """Probability that a transmission over *dist_m* metres is lost.

    Strictly increasing in distance, and always strictly inside (0, 1).

    Args:
        dist_m: Distance between sender and receiver, in metres.
        edge_m: Distance of 50% loss. Defaults to
                ``config.LOSS_50_DISTANCE_M``; pass it only to ask what a
                different edge would do.

    Returns:
        A loss in [0.0, 1.0]: exactly 0.0 at zero distance, and strictly
        inside (0, 1) at every distance this arena can produce.
    """
    if dist_m <= 0.0:
        # Zero distance has infinite margin, so nothing is lost.
        return 0.0
    b = B_PER_DB if edge_m is None else _slope_for_edge(edge_m)

    exponent = b * margin_db(dist_m)
    # exp() overflows for a very short link; the limit there is a loss of 0.
    if exponent > 700.0:
        return 0.0
    return 1.0 / (1.0 + A * math.exp(exponent))


def link_quality(dist_m: float) -> float:
    """Probability that a transmission over *dist_m* metres gets through.

    Simply ``1 - p_loss``. This is what a drone reports about a link in its
    state vector, and what "a good link" is measured against.

    Args:
        dist_m: Distance between endpoints, in metres.

    Returns:
        Float in (0.0, 1.0); higher is better.
    """
    return 1.0 - p_loss(dist_m)


def distance_for_loss(
    target_loss: float,
    edge_m: Optional[float] = None,
) -> float:
    """Return the distance at which loss equals *target_loss*.

    The inverse of :func:`p_loss`. Solving the S-curve for the margin and
    then the margin for the distance:

        M = ln((1 - L) / (A * L)) / b
        d = SENSITIVITY_DISTANCE_M * 10 ** (-M / 20)

    Args:
        target_loss: A loss strictly between 0 and 1.
        edge_m:      Distance of 50% loss. Defaults to
                     ``config.LOSS_50_DISTANCE_M``.

    Returns:
        The distance in metres.

    Raises:
        ValueError: If *target_loss* is not strictly inside (0, 1), where the
            curve has no finite inverse.
    """
    if not 0.0 < target_loss < 1.0:
        raise ValueError(
            f"target_loss must be strictly between 0 and 1, got {target_loss}"
        )
    b = B_PER_DB if edge_m is None else _slope_for_edge(edge_m)

    margin = math.log((1.0 - target_loss) / (A * target_loss)) / b
    return SENSITIVITY_DISTANCE_M * 10.0 ** (-margin / 20.0)


def max_link_distance(max_loss: Optional[float] = None) -> float:
    """Return how far a link reaches: the distance where loss hits the cutoff.

    Args:
        max_loss: Loss at which a link stops existing. Defaults to
                  ``config.LINK_MAX_LOSS``.

    Returns:
        The distance in metres. At the configured cutoff of 0.5 this is
        exactly ``config.LOSS_50_DISTANCE_M``, the curve's midpoint.
    """
    if max_loss is None:
        max_loss = config.LINK_MAX_LOSS
    return distance_for_loss(max_loss)


# ---------------------------------------------------------------------------
# Link existence
# ---------------------------------------------------------------------------

def link_exists(dist_m: float, max_loss: Optional[float] = None) -> bool:
    """True if two endpoints *dist_m* apart have a link at all.

    Args:
        dist_m:   Distance between endpoints, in metres.
        max_loss: Loss at which a link stops existing. Defaults to
                  ``config.LINK_MAX_LOSS``.

    Returns:
        Whether ``p_loss(dist_m) < max_loss``.
    """
    if max_loss is None:
        max_loss = config.LINK_MAX_LOSS
    return p_loss(dist_m) < max_loss


def are_connected(pos_a: np.ndarray, pos_b: np.ndarray) -> bool:
    """True iff the two positions have a link.

    The single link-existence test used across the simulator: neighbour sets,
    the GS link, the connected-layout filter and the range check all come
    through here.

    Args:
        pos_a: Position of endpoint A as a NumPy array.
        pos_b: Position of endpoint B as a NumPy array.

    Returns:
        Whether a link exists between them.
    """
    return link_exists(float(np.linalg.norm(pos_a - pos_b)))


def euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Return the euclidean distance between two position arrays (m).

    Args:
        pos_a: Position vector (x, y) or (x, y, z).
        pos_b: Position vector of the same dimension.

    Returns:
        Non-negative float distance in metres.
    """
    return float(np.linalg.norm(pos_a - pos_b))
