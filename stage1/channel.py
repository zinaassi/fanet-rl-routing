"""Distance-normalised logistic packet-loss channel with a hard 250 m range.

    u(d)      = clip(d / COMM_RANGE_M, 0, 1)          normalised distance
    sigma(x)  = 1 / (1 + exp(-x))
    p_loss(d) = (sigma(k*(u - 1/2)) - sigma(-k/2)) / (sigma(k/2) - sigma(-k/2))

A logistic in normalised distance, rescaled so that for EVERY steepness k:

    p_loss(0)            = 0     (a zero-length link never loses a packet)
    p_loss(COMM_RANGE/2) = 0.5   (midpoint at 125 m)
    p_loss(COMM_RANGE)   = 1     (a link at the range edge never delivers)

Beyond COMM_RANGE_M the curve saturates at 1: out-of-range nodes cannot
exchange packets, and ``world.candidate_graph`` creates no edge there. The
steepness k only shapes the curve inside the range — k -> 0 approaches a
linear ramp, large k approaches a step at the midpoint. All functions are
numpy-vectorised and accept scalars or arrays.
"""
from __future__ import annotations

from typing import Union

import numpy as np

from . import config

FloatOrArray = Union[float, np.ndarray]


def _sigmoid(x: FloatOrArray) -> np.ndarray:
    """Numerically stable logistic 1/(1+exp(-x)), via tanh (never overflows)."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def normalized_distance(distance_m: FloatOrArray) -> np.ndarray:
    """d / COMM_RANGE_M, clipped to [0, 1] (saturates beyond the hard range)."""
    d = np.asarray(distance_m, dtype=float)
    return np.clip(d / config.COMM_RANGE_M, 0.0, 1.0)


def p_loss(distance_m: FloatOrArray, k: float) -> FloatOrArray:
    """Per-attempt packet-loss probability; exactly 0 at d=0 and 1 at the range edge.

    Rescaled logistic in normalised distance (see module docstring); the
    rescaling pins the endpoints to 0 and 1 for every k, so k is a pure
    shape knob and never changes the communication range.
    """
    u = normalized_distance(distance_m)
    lo = _sigmoid(-0.5 * k)
    hi = _sigmoid(0.5 * k)
    return (_sigmoid(k * (u - 0.5)) - lo) / (hi - lo)
