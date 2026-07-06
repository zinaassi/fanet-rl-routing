"""Per-simulation metrics and between/within-topology aggregation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import networkx as nx
import numpy as np

from . import config, sim
from .routing import NextHops, routed_drones


def queue_slopes(queue_depths: np.ndarray, window: int) -> np.ndarray:
    """Least-squares slope [packets/step] of each drone's queue depth
    over the last ``window`` recorded steps."""
    w = queue_depths[-window:].astype(float)
    t = np.arange(w.shape[0], dtype=float)
    t -= t.mean()
    denom = float(t @ t)
    if denom == 0.0:
        return np.zeros(w.shape[1])
    return (t @ w) / denom


def unstable_drones(
    result: sim.SimResult,
    window: int = config.INSTABILITY_WINDOW,
    slope_min: float = config.INSTABILITY_SLOPE_MIN,
) -> tuple[int, ...]:
    """Drones whose queue depth is still trending upward at episode end."""
    slopes = queue_slopes(result.queue_depths, window)
    return tuple(result.drones[i] for i in np.flatnonzero(slopes > slope_min))


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def sim_metrics(
    result: sim.SimResult,
    graph: nx.DiGraph,
    next_hops: NextHops,
) -> Dict[str, float]:
    """Flatten one simulation into a dict of scalar metrics.

    PDR denominators exclude packets still in flight at episode end.
    ``pdr_global`` counts every resolved packet; ``pdr_routed`` only packets
    emitted by routed sources (drones whose next-hop chain reaches the GS).
    Delay metrics are over delivered packets only.
    """
    routed = routed_drones(next_hops, graph)
    drones = result.drones
    m_drones = [d for d in drones if graph.nodes[d].get("kind", "M") == "M"]

    resolved = result.resolved
    delivered = result.delivered
    routed_src = np.isin(result.src, list(routed))

    delays = result.delay_steps[delivered]
    hops = result.hops[delivered]
    waits = result.queue_wait_steps[delivered]

    dropped = resolved & ~delivered
    kinds = {d: graph.nodes[d].get("kind", "M") for d in drones}
    drop_nodes = result.end_node[dropped]
    n_dropped = int(dropped.sum())
    drops_at_m = sum(1 for n in drop_nodes if kinds.get(int(n)) == "M")
    drops_at_c = sum(1 for n in drop_nodes if kinds.get(int(n)) == "C")

    unstable = unstable_drones(result)

    return {
        "n_emitted": float(len(result.src)),
        "n_delivered": float(delivered.sum()),
        "n_dropped_channel": float((result.status == sim.DROPPED_CHANNEL).sum()),
        "n_dropped_no_route": float((result.status == sim.DROPPED_NO_ROUTE).sum()),
        "n_in_flight": float((~resolved).sum()),
        "pdr_global": _safe_ratio(float(delivered.sum()), float(resolved.sum())),
        "pdr_routed": _safe_ratio(
            float((delivered & routed_src).sum()), float((resolved & routed_src).sum())
        ),
        "unreachable_frac_all": 1.0 - len(routed) / len(drones),
        "unreachable_frac_m": 1.0 - sum(d in routed for d in m_drones) / len(m_drones),
        "mean_delay_steps": float(delays.mean()) if delays.size else float("nan"),
        "mean_delay_ms": float(delays.mean() * config.STEP_MS) if delays.size else float("nan"),
        "mean_hops": float(hops.mean()) if hops.size else float("nan"),
        "mean_queue_wait_steps": float(waits.mean()) if waits.size else float("nan"),
        "max_queue_depth": float(result.queue_depths.max()),
        "mean_queue_depth": float(result.queue_depths.mean()),
        "n_unstable_drones": float(len(unstable)),
        "drop_frac_at_m": _safe_ratio(float(drops_at_m), float(n_dropped)),
        "drop_frac_at_c": _safe_ratio(float(drops_at_c), float(n_dropped)),
    }


def drop_histogram(result: sim.SimResult) -> Dict[int, int]:
    """Drop counts keyed by the node id where each drop happened."""
    dropped = result.resolved & ~result.delivered
    nodes, counts = np.unique(result.end_node[dropped], return_counts=True)
    return {int(n): int(c) for n, c in zip(nodes, counts)}


@dataclass(frozen=True)
class AggStats:
    """Mean plus variability split into between- and within-topology parts."""

    mean: float
    between_std: float  # std over per-topology means
    within_std: float   # mean over topologies of the per-topology std

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.mean:.3f} ±{self.between_std:.3f}b ±{self.within_std:.3f}w"


def aggregate(values: np.ndarray) -> AggStats:
    """Aggregate a (n_topologies, n_channel_realizations) metric array.

    NaN cells (undefined metrics, e.g. delay with zero deliveries) are
    ignored. With a single topology or realization the corresponding std
    is NaN.
    """
    v = np.asarray(values, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        topo_means = np.nanmean(v, axis=1)
        mean = float(np.nanmean(topo_means))
        between = float(np.nanstd(topo_means, ddof=1)) if v.shape[0] > 1 else float("nan")
        within = (
            float(np.nanmean(np.nanstd(v, axis=1, ddof=1))) if v.shape[1] > 1 else float("nan")
        )
    return AggStats(mean=mean, between_std=between, within_std=within)
