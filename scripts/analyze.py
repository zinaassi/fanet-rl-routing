"""
analyze.py — Stage 2 metric computation for the FANET simulator.

Reads one JSONL event log produced by Stage 1 (the simulator) and computes
all reported metrics. The simulator never aggregates metrics in-loop; this
script is the single source of truth for what is reported.

Adding a new metric (delay percentiles, throughput, Fiedler connectivity,
path efficiency, …) is a change to THIS file only — no re-simulation needed,
as long as the raw fields are already in the log.

Usage:
    python scripts/analyze.py logs/episode_0.jsonl
    python scripts/analyze.py logs/episode_0.jsonl --json out.json

Implements the Tier-1 and Tier-2 metrics from the spec:

  Tier 1
    PDR                       delivered / generated  (raw counts always printed)
    avg_delay                 mean of arrival - generation over DELIVERED only
                              (dropped / in-flight excluded — never 0 or inf)
    frac_connected_to_gs      mean and min across the per-step samples

  Tier 2
    avg_hops                  mean hop count over DELIVERED only
    NRL                       control transmissions / delivered data packets
                              (is_control flag drives the split)
    energy                    radio and motion totals reported SEPARATELY
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield one parsed record per line of *path*.

    Args:
        path: Path to a Stage-1 JSONL event log.

    Yields:
        Dict for each non-empty line.
    """
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all reported metrics from a stream of raw event records.

    Args:
        records: Iterable of parsed JSON records (one per event).

    Returns:
        Dict containing raw counts, derived ratios, energy totals, and the
        per-episode metadata. Field naming follows the spec; ``None`` is used
        for metrics that are undefined for this episode (e.g. avg_delay when
        no packets were delivered).
    """
    meta: Optional[Dict[str, Any]] = None

    # Packet-event counters
    generated_ids: set = set()
    delivered_ids: set = set()
    dropped_ids: set = set()
    delays_steps: List[float] = []     # delivered-only, in timesteps
    delays_seconds: List[float] = []   # delivered-only, in seconds
    hops_delivered: List[int] = []     # delivered-only

    # Per-packet generation times (steps + seconds) to compute delay
    gen_time_s: Dict[int, float] = {}
    # Used only to recover timestep-units for delays from the meta record.
    timestep_s: Optional[float] = None

    # Drop-reason breakdown (diagnostic)
    drops_by_reason: Dict[str, int] = defaultdict(int)

    # Transmission tallies — split by is_control so NRL is honest.
    data_tx_count = 0
    control_tx_count = 0

    # Per-step connectivity samples
    frac_samples: List[float] = []
    components_samples: List[int] = []

    # First and last drone-state record seen per drone. Energy is monotone
    # non-increasing, so first = initial and last = final.
    first_drone_state: Dict[int, Dict[str, Any]] = {}
    last_drone_state: Dict[int, Dict[str, Any]] = {}

    for rec in records:
        rt = rec.get("record_type")

        if rt == "episode_meta":
            meta = rec
            cm = rec.get("connectivity_model_params") or {}
            ts = cm.get("timestep_s")
            if ts is not None:
                timestep_s = float(ts)
            continue

        if rt == "packet_event":
            event = rec["event"]
            pid = rec["packet_id"]
            is_control = bool(rec.get("is_control", False))
            t = float(rec["time"])

            if event == "generated":
                if not is_control:
                    generated_ids.add(pid)
                    gen_time_s[pid] = t
            elif event == "forwarded":
                if is_control:
                    control_tx_count += 1
                else:
                    data_tx_count += 1
            elif event == "delivered":
                # The hop into the GS is itself a transmission.
                if is_control:
                    control_tx_count += 1
                else:
                    data_tx_count += 1
                    delivered_ids.add(pid)
                    hops_delivered.append(int(rec["hop_index"]))
                    if pid in gen_time_s:
                        delta_s = t - gen_time_s[pid]
                        delays_seconds.append(delta_s)
                        if timestep_s and timestep_s > 0:
                            delays_steps.append(delta_s / timestep_s)
            elif event == "dropped":
                if not is_control:
                    dropped_ids.add(pid)
                    drops_by_reason[rec.get("drop_reason") or "unknown"] += 1
            continue

        if rt == "step_state":
            frac_samples.append(float(rec["frac_connected_to_gs"]))
            components_samples.append(int(rec["num_components"]))
            continue

        if rt == "drone_state":
            did = int(rec["drone_id"])
            if did not in first_drone_state:
                first_drone_state[did] = rec
            last_drone_state[did] = rec
            continue

    # ---------------- Derived metrics ----------------

    generated = len(generated_ids)
    delivered = len(delivered_ids)
    dropped = len(dropped_ids)
    # A packet was generated but neither delivered nor dropped → still in flight.
    in_flight_at_end = len(generated_ids - delivered_ids - dropped_ids)

    pdr = delivered / generated if generated > 0 else None
    drop_rate = dropped / generated if generated > 0 else None

    # Delay: delivered-only. Never substitute 0 or infinity when empty.
    avg_delay_steps = _mean_or_none(delays_steps)
    avg_delay_seconds = _mean_or_none(delays_seconds)

    avg_hops = _mean_or_none(hops_delivered)

    # NRL: control transmissions per delivered data packet. Undefined when
    # nothing was delivered.
    nrl = (control_tx_count / delivered) if delivered > 0 else None

    # Connectivity summaries — mean and min, NOT a binary flag.
    conn_mean = _mean_or_none(frac_samples)
    conn_min = min(frac_samples) if frac_samples else None
    components_mean = _mean_or_none(components_samples)

    # Energy totals, kept SEPARATE per spec §C.3.
    initial_radio = sum(d.get("energy_radio", 0.0) for d in first_drone_state.values())
    initial_motion = sum(d.get("energy_motion", 0.0) for d in first_drone_state.values())
    final_radio = sum(d.get("energy_radio", 0.0) for d in last_drone_state.values())
    final_motion = sum(d.get("energy_motion", 0.0) for d in last_drone_state.values())

    energy = {
        "total_radio_consumed_J": _safe_diff(initial_radio, final_radio),
        "total_motion_consumed_J": _safe_diff(initial_motion, final_motion),
        "avg_radio_remaining_J": _mean_or_none(
            [d.get("energy_radio", 0.0) for d in last_drone_state.values()]
        ),
        "avg_motion_remaining_J": _mean_or_none(
            [d.get("energy_motion", 0.0) for d in last_drone_state.values()]
        ),
    }

    # PDR over time — analyse the episode's TIME EVOLUTION of delivery, not
    # just the single cumulative number. For each generation step we measure
    # the fraction of packets generated at that step that were eventually
    # delivered. Late-episode generation steps are excluded because their
    # packets may not have had time to deliver before the episode ended.
    episode_length = int((meta or {}).get("episode_length", 0))
    ttl_steps = int(((meta or {}).get("traffic_load", {}) or {}).get("ttl", 50))
    pdr_over_time = _pdr_over_time(
        gen_time_s=gen_time_s,
        delivered_ids=delivered_ids,
        episode_length=episode_length,
        timestep_s=timestep_s or 0.1,
        ttl_steps=ttl_steps,
    )

    return {
        "episode_meta": meta,
        "raw_counts": {
            "generated": generated,
            "delivered": delivered,
            "dropped": dropped,
            "in_flight_at_end": in_flight_at_end,
            "data_transmissions": data_tx_count,
            "control_transmissions": control_tx_count,
            "drops_by_reason": dict(drops_by_reason),
        },
        "tier1": {
            "PDR": pdr,
            "avg_delay_steps": avg_delay_steps,
            "avg_delay_seconds": avg_delay_seconds,
            "frac_connected_to_gs_mean": conn_mean,
            "frac_connected_to_gs_min": conn_min,
        },
        "tier2": {
            "avg_hops": avg_hops,
            "NRL": nrl,
            "drop_rate": drop_rate,
            "num_components_mean": components_mean,
            "energy": energy,
        },
        "pdr_over_time": pdr_over_time,
    }


def _pdr_over_time(
    *,
    gen_time_s: Dict[int, float],
    delivered_ids: set,
    episode_length: int,
    timestep_s: float,
    ttl_steps: int,
    n_windows: int = 10,
) -> Dict[str, Any]:
    """Per-generation-step PDR plus a windowed view across the episode.

    For each generation step ``s``, we compute
        per_step_pdr(s) = | { p : gen_step(p)=s AND p delivered } | /
                          | { p : gen_step(p)=s } |
    then aggregate (mean / median / min / max / std) across all steps
    ``s < episode_length - ttl_steps`` (later steps are excluded because
    packets generated there had no time to deliver before MAX_STEPS).

    The windowed view buckets generation steps into ``n_windows`` equal
    time-bins and reports delivered/generated per bin. The very last bin
    may include late-episode steps whose packets ran out of time; this is
    flagged in the returned dict via ``last_window_truncated``.

    Args:
        gen_time_s:     Map packet_id -> generation time in seconds.
        delivered_ids:  Set of packet IDs that were delivered.
        episode_length: Total simulated steps (from episode_meta).
        timestep_s:     Seconds per simulated step.
        ttl_steps:      Packet TTL in steps (defines the cutoff).
        n_windows:      Number of equal-time bins for the windowed view.

    Returns:
        Dict with keys ``per_step`` (summary stats), ``windowed`` (list of
        per-bin records), ``cutoff_step``, and ``last_window_truncated``.
    """
    empty = {
        "per_step": {"mean": None, "median": None, "min": None, "max": None,
                     "std": None, "n_steps_used": 0},
        "windowed": [],
        "cutoff_step": None,
        "last_window_truncated": False,
    }
    if not gen_time_s or timestep_s <= 0 or episode_length <= 0:
        return empty

    cutoff_step = max(0, episode_length - ttl_steps)

    # Bucket each packet by its generation step.
    per_step: Dict[int, list] = defaultdict(lambda: [0, 0])  # step -> [delivered, total]
    for pid, t in gen_time_s.items():
        s = int(round(t / timestep_s))
        per_step[s][1] += 1
        if pid in delivered_ids:
            per_step[s][0] += 1

    # Per-step PDR summary, excluding the post-cutoff steps.
    pdrs_used: List[float] = []
    for s, (d, n) in per_step.items():
        if s >= cutoff_step or n == 0:
            continue
        pdrs_used.append(d / n)

    per_step_summary = {
        "mean":   statistics.fmean(pdrs_used) if pdrs_used else None,
        "median": statistics.median(pdrs_used) if pdrs_used else None,
        "min":    min(pdrs_used) if pdrs_used else None,
        "max":    max(pdrs_used) if pdrs_used else None,
        "std":    statistics.pstdev(pdrs_used) if len(pdrs_used) > 1 else 0.0,
        "n_steps_used": len(pdrs_used),
    }

    # Windowed view across the FULL episode (last bin may be truncated).
    win_size = max(1, episode_length // n_windows)
    windowed: List[Dict[str, Any]] = []
    for w in range(n_windows):
        lo = w * win_size
        hi = (w + 1) * win_size - 1 if w < n_windows - 1 else episode_length - 1
        gen = dlv = 0
        for s in range(lo, hi + 1):
            if s in per_step:
                d, n = per_step[s]
                gen += n
                dlv += d
        pdr = (dlv / gen) if gen > 0 else None
        windowed.append({
            "start_step": lo,
            "end_step": hi,
            "generated": gen,
            "delivered": dlv,
            "pdr": pdr,
        })

    return {
        "per_step": per_step_summary,
        "windowed": windowed,
        "cutoff_step": cutoff_step,
        "last_window_truncated": windowed[-1]["end_step"] >= cutoff_step if windowed else False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_or_none(xs: List[float]) -> Optional[float]:
    """Return the mean of *xs*, or None if empty.

    Empty means undefined — never substitute 0 or infinity (spec §B.2).
    """
    return statistics.fmean(xs) if xs else None


def _safe_diff(a: float, b: float) -> Optional[float]:
    """Return ``a - b`` if both are finite, else None."""
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    return a - b


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    """Format a metric value for the text report."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_report(metrics: Dict[str, Any]) -> None:
    """Print a human-readable report to stdout."""
    meta = metrics["episode_meta"] or {}
    raw = metrics["raw_counts"]
    t1 = metrics["tier1"]
    t2 = metrics["tier2"]
    energy = t2["energy"]

    sep = "=" * 56
    print(sep)
    print(f"  FANET Episode Analysis  (episode_id={meta.get('episode_id')})")
    print(sep)
    print(f"  seed              : {meta.get('seed')}")
    print(f"  num_drones        : {meta.get('num_drones')}  "
          f"(M={meta.get('num_M')}, C={meta.get('num_C')})")
    print(f"  episode_length    : {meta.get('episode_length')}")
    print(sep)
    print("  Raw counts")
    print(f"    generated        : {raw['generated']}")
    print(f"    delivered        : {raw['delivered']}")
    print(f"    dropped          : {raw['dropped']}")
    print(f"    in_flight_at_end : {raw['in_flight_at_end']}")
    print(f"    data tx          : {raw['data_transmissions']}")
    print(f"    control tx       : {raw['control_transmissions']}")
    if raw["drops_by_reason"]:
        print("    drops by reason  :")
        for reason, n in sorted(raw["drops_by_reason"].items()):
            print(f"      - {reason}: {n}")
    print(sep)
    print("  Tier 1")
    print(f"    PDR                       : {_fmt(t1['PDR'])}")
    print(f"    avg_delay (steps)         : {_fmt(t1['avg_delay_steps'])}")
    print(f"    avg_delay (seconds)       : {_fmt(t1['avg_delay_seconds'])}")
    print(f"    frac_connected_to_gs mean : {_fmt(t1['frac_connected_to_gs_mean'])}")
    print(f"    frac_connected_to_gs min  : {_fmt(t1['frac_connected_to_gs_min'])}")
    print(sep)
    print("  Tier 2")
    print(f"    avg_hops          : {_fmt(t2['avg_hops'])}")
    print(f"    NRL               : {_fmt(t2['NRL'])}")
    print(f"    drop_rate         : {_fmt(t2['drop_rate'])}")
    print(f"    num_components    : {_fmt(t2['num_components_mean'])}")
    print(f"    radio energy used : {_fmt(energy['total_radio_consumed_J'])} J  "
          f"(remaining avg {_fmt(energy['avg_radio_remaining_J'])} J)")
    print(f"    motion energy used: {_fmt(energy['total_motion_consumed_J'])} J  "
          f"(remaining avg {_fmt(energy['avg_motion_remaining_J'])} J)")
    print(sep)

    pot = metrics.get("pdr_over_time") or {}
    ps = pot.get("per_step") or {}
    windowed = pot.get("windowed") or []
    if ps.get("n_steps_used"):
        print("  PDR over time")
        print(f"    cutoff step (TTL margin) : {pot.get('cutoff_step')}  "
              f"(generation steps >= this are excluded from per-step stats)")
        print(f"    per-step PDR — n         : {ps['n_steps_used']}")
        print(f"    per-step PDR — mean      : {_fmt(ps['mean'])}")
        print(f"    per-step PDR — median    : {_fmt(ps['median'])}")
        print(f"    per-step PDR — min / max : {_fmt(ps['min'])} / {_fmt(ps['max'])}")
        print(f"    per-step PDR — std       : {_fmt(ps['std'])}")
        if windowed:
            print("    windowed PDR (delivered / generated per window of generation steps)")
            for w in windowed:
                lo, hi = w["start_step"], w["end_step"]
                pdr_str = _fmt(w["pdr"])
                trunc = "  *truncated" if (
                    pot.get("last_window_truncated")
                    and hi == windowed[-1]["end_step"]
                ) else ""
                print(f"      steps {lo:4d}-{hi:4d} : PDR={pdr_str}  "
                      f"({w['delivered']:>5d} / {w['generated']:>5d}){trunc}")
        print(sep)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Analyse a Stage-1 FANET event log.")
    parser.add_argument("log_path", type=Path, help="Path to the JSONL event log.")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to also write the metrics dict as JSON.")
    args = parser.parse_args(argv)

    if not args.log_path.exists():
        print(f"error: {args.log_path} does not exist", file=sys.stderr)
        return 2

    metrics = analyze(read_jsonl(args.log_path))
    print_report(metrics)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
