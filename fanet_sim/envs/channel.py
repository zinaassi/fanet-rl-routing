"""
channel.py — Wireless link model for the FANET simulator.

Phase-1 model: distance-based link quality that degrades linearly with
distance. No path-loss exponents or noise models (those come in a later
phase).
"""

from __future__ import annotations
import numpy as np
from fanet_sim import config


def link_quality(dist: float, comm_range: float = config.COMM_RANGE) -> float:
    """Compute the quality of a wireless link given the euclidean distance.

    Quality is 1.0 at zero distance and falls linearly to 0.0 at
    *comm_range*. A link is considered non-existent (quality == 0) when the
    distance equals or exceeds *comm_range*.

    Args:
        dist:       Euclidean distance between the two endpoints in metres.
        comm_range: Maximum communication range in metres.

    Returns:
        A float in [0.0, 1.0].  0.0 means no link.
    """
    if dist >= comm_range:
        return 0.0
    return float(1.0 - dist / comm_range)


def are_connected(pos_a: np.ndarray, pos_b: np.ndarray,
                  comm_range: float = config.COMM_RANGE) -> bool:
    """Return True if two nodes at *pos_a* and *pos_b* can communicate.

    Args:
        pos_a:      Position of node A as a NumPy array (x, y) or (x, y, z).
        pos_b:      Position of node B as a NumPy array.
        comm_range: Maximum communication range in metres.

    Returns:
        True if the euclidean distance < comm_range.
    """
    dist = float(np.linalg.norm(pos_a - pos_b))
    return dist < comm_range


def euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Return the euclidean distance between two position arrays.

    Args:
        pos_a: Position vector (x, y) or (x, y, z).
        pos_b: Position vector of the same dimension.

    Returns:
        Non-negative float distance in metres.
    """
    return float(np.linalg.norm(pos_a - pos_b))
