"""
channel.py — Packet loss versus distance, and what counts as a link.

Loss model
----------
A logistic curve in DISTANCE, fitted to measured UAV-to-UAV links in

    Rosati et al., "Dynamic Routing for Flying Ad Hoc Networks",
    arXiv:1406.4399

    p_loss(d) = 1 / (1 + exp(-s * (d - D50)))

        d   = distance between sender and receiver, in metres
        D50 = config.LOSS_50_DISTANCE_M, the distance where loss is 50%
        s   = config.LOSS_SLOPE_PER_M, the measured steepness

With D50 = 356 m and s = 0.025 this is the paper's
``1 / (1 + exp(-(0.025*d - 8.9)))``:

    d (m)     100     200     250     270     300     356     400     540
    p_loss  0.0017  0.0198  0.0661  0.1044  0.1978  0.5000  0.7503  0.9900

The curve never reaches 0 or 1 exactly; it only approaches them. The same
curve governs the last hop into the ground station — the GS is just another
endpoint.

Link existence
--------------
A link exists while its loss stays under ``config.LINK_MAX_LOSS``, which puts
the edge at ``max_link_distance()``. Currently that cutoff is 0.5 and the edge
is 356 m — see config.py, which is the single source of truth. This is the ONE
rule used everywhere: neighbour sets, the GS link, the connected-layout filter
and the range check. Distance alone decides it, through the curve.

Assumptions
-----------
* 2-D. No altitude, terrain or obstruction.
* No MAC layer, no interference model, no collisions.
* Every endpoint has the same radio, the GS included.
* Loss depends on distance only — not on traffic, antenna orientation or
  interference from other transmissions.

What this replaced
------------------
Until Checkpoint 7 the model was free-space path loss (FSPL): transmit power
30 dBm, 2 dBi antennas, 2.4 GHz carrier and a receiver sensitivity of -54 dBm,
giving a hard 249.69 m range, with loss ``exp(-k * M)`` on the margin M above
sensitivity (k = 0.8). That curve hit ~100% loss exactly at 250 m, which
measured UAV links do not show, and 250 m is short against a 900x900 m area.
The FSPL code is removed — nothing in the simulator needed it once link
existence stopped being power-based. ``scripts/plot_ploss.py`` carries its own
copy of the old formula, only so the two curves can be drawn together.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from fanet_sim import config


# ---------------------------------------------------------------------------
# The loss curve
# ---------------------------------------------------------------------------

def p_loss(
    dist_m: float,
    d50: Optional[float] = None,
    slope: Optional[float] = None,
) -> float:
    """Probability that a transmission over *dist_m* metres is lost.

    ``1 / (1 + exp(-slope * (dist_m - d50)))`` — strictly increasing in
    distance, and always strictly inside (0, 1).

    Args:
        dist_m: Distance between sender and receiver, in metres.
        d50:    Distance at which loss is 50%. Defaults to
                ``config.LOSS_50_DISTANCE_M``.
        slope:  Steepness per metre. Defaults to ``config.LOSS_SLOPE_PER_M``.

    Returns:
        Float in (0.0, 1.0).
    """
    if d50 is None:
        d50 = config.LOSS_50_DISTANCE_M
    if slope is None:
        slope = config.LOSS_SLOPE_PER_M

    exponent = -slope * (dist_m - d50)
    # exp() overflows for a very short link; the limit there is a loss of 0.
    if exponent > 700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(exponent))


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
    d50: Optional[float] = None,
    slope: Optional[float] = None,
) -> float:
    """Return the distance at which loss equals *target_loss*.

    The inverse of :func:`p_loss`:
    ``d = d50 + ln(L / (1 - L)) / slope``.

    Args:
        target_loss: A loss strictly between 0 and 1.
        d50:         Defaults to ``config.LOSS_50_DISTANCE_M``.
        slope:       Defaults to ``config.LOSS_SLOPE_PER_M``.

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
    if d50 is None:
        d50 = config.LOSS_50_DISTANCE_M
    if slope is None:
        slope = config.LOSS_SLOPE_PER_M

    return d50 + math.log(target_loss / (1.0 - target_loss)) / slope


def max_link_distance(max_loss: Optional[float] = None) -> float:
    """Return how far a link reaches: the distance where loss hits the cutoff.

    Args:
        max_loss: Loss at which a link stops existing. Defaults to
                  ``config.LINK_MAX_LOSS``.

    Returns:
        The distance in metres. At the configured cutoff of 0.5 this is 356 m,
        the curve's midpoint.
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
