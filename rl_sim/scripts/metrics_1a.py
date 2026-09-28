"""
metrics_1a.py — The Phase-1A metrics report for one run.

Reads a finished :class:`FANETEnv` and reports the same run twice:

    ground truth — what the simulator knows happened.
    ACK-based    — what the drones themselves concluded, from end-to-end GS
                   ACKs and hop-by-hop ACKs only.

With ideal ACKs the two views must agree exactly; any gap is a bug, and
:func:`compare_views` is what surfaces it.

Two measurement windows
-----------------------
The headline number is TOTAL PACKET LOSS: the share of packets created in the
window that never reached the GS. The split by cause and the per-link figures
are there to explain it.

Packet-level figures (total loss, loss by cause, delay, hop count) count
only packets CREATED in ``config.measurement_window()`` — after the warm-up, and
early enough that a full TTL remains before the episode ends, so every measured
packet has finished.

Per-link and queue figures are instantaneous per-step quantities: a
transmission does not need time to finish the way a packet does. They count
every EVENT after the warm-up, via ``config.is_warm``.

These are different populations — a transmission after the drain cut belongs to
the second but not the first — so the two views are only ever compared like
with like: packet counts against packet counts, link counters against link
counters.

Usage:
    from scripts.metrics_1a import compute_metrics, print_metrics_report
    print_metrics_report(compute_metrics(env))
"""

from __future__ import annotations

import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.drone import LINK_FIELDS
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import DROP_REASON_LABELS, Packet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rate(numerator: float, denominator: float) -> Optional[float]:
    """Return numerator/denominator, or None when the denominator is zero."""
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: List[float]) -> Optional[float]:
    """Return the mean of *values*, or None if empty."""
    return statistics.fmean(values) if values else None


def measured_packets(env: FANETEnv) -> List[Packet]:
    """Return the packets whose creation step falls in the measurement window.

    Args:
        env: A finished environment.

    Returns:
        The subset of ``env.all_packets`` the report is computed over.
    """
    return [
        pkt for pkt in env.all_packets
        if config.in_measurement_window(pkt.created_at)
    ]


#: The three ways a packet offered to a link fails to get across.
LOSS_PARTS = ("dropped_queue_full", "lost_channel", "expired_in_queue")


def _link_loss_rows(
    stats: Dict[Any, Dict[str, int]]
) -> List[Dict[str, Any]]:
    """Turn per-link counter rows into per-link loss records.

    For the link X->Y, loss is the share of packets OFFERED to that link which
    never got across:

        loss = (dropped_queue_full + lost_channel + expired_in_queue) / offered

    Each part is kept alongside the total. Packets still waiting in the queue
    when the episode ended are in the denominator but in none of the parts, so
    the parts can sum to slightly less than the total offered.

    Args:
        stats: Per-link counter rows keyed by link.

    Returns:
        One record per link that was offered at least one packet.
    """
    rows: List[Dict[str, Any]] = []
    for link, row in stats.items():
        offered = row["offered"]
        if offered == 0:
            continue
        lost = sum(row[part] for part in LOSS_PARTS)
        record: Dict[str, Any] = {
            "link": link,
            "offered": offered,
            "sent": row["sent"],
            "acked": row["acked"],
            "loss": lost / offered,
        }
        for part in LOSS_PARTS:
            record[f"loss_{part}"] = row[part] / offered
        rows.append(record)
    return rows


def _link_summary(stats: Dict[Any, Dict[str, int]]) -> Dict[str, Any]:
    """Summarise per-link loss: the overall average and the worst single link.

    Args:
        stats: Per-link counter rows keyed by link.

    Returns:
        A dict with pooled totals, the unweighted mean across links, and the
        worst link (by loss, among links offered at least one packet).
    """
    rows = _link_loss_rows(stats)
    total_offered = sum(row["offered"] for row in rows)
    part_totals = {
        part: sum(row[part] for row in stats.values()) for part in LOSS_PARTS
    }
    total_lost = sum(part_totals.values())

    worst = max(rows, key=lambda r: (r["loss"], r["offered"]), default=None)

    summary: Dict[str, Any] = {
        "raw": {link: dict(row) for link, row in stats.items()},
        "num_links": len(rows),
        "total_offered": total_offered,
        # Pooled over every offered packet, so busy links carry more weight.
        "pooled_loss": _rate(total_lost, total_offered),
        # Unweighted across links, so a rarely-used bad link still shows.
        "mean_link_loss": _mean([r["loss"] for r in rows]),
        "worst_link": worst,
        "rows": rows,
    }
    for part in LOSS_PARTS:
        summary[f"pooled_loss_{part}"] = _rate(part_totals[part], total_offered)
    return summary


# ---------------------------------------------------------------------------
# The two views
# ---------------------------------------------------------------------------

def ground_truth_view(env: FANETEnv) -> Dict[str, Any]:
    """Compute the simulator's own view of the run.

    Args:
        env: A finished environment.

    Returns:
        A dict of counts and rates over the measurement window.
    """
    packets = measured_packets(env)
    created = len(packets)
    delivered = [p for p in packets if p.delivered]
    dropped = [p for p in packets if p.dropped]
    in_flight = [p for p in packets if not p.delivered and not p.dropped]

    by_reason = {label: 0 for label in DROP_REASON_LABELS}
    for pkt in dropped:
        assert pkt.drop_reason is not None
        by_reason[pkt.drop_reason.label] += 1

    delays = [d for d in (p.delay() for p in delivered) if d is not None]
    hops = [p.hop_count for p in delivered]

    return {
        "created": created,
        "delivered": len(delivered),
        "dropped": len(dropped),
        "still_in_flight": len(in_flight),
        "delivery_rate": _rate(len(delivered), created),
        # THE headline metric: share of created packets that never reached the GS.
        "total_loss": _rate(len(dropped) + len(in_flight), created),
        "loss_rate": _rate(len(dropped), created),
        "loss_by_reason": by_reason,
        "loss_rate_by_reason": {
            label: _rate(count, created) for label, count in by_reason.items()
        },
        "mean_delay_steps": _mean(delays),
        "mean_delay_ms": (
            None if _mean(delays) is None
            else _mean(delays) * config.TIMESTEP * 1000
        ),
        "mean_hops": _mean(hops),
        "max_hops": max(hops) if hops else None,
        "links": _link_summary(env.link_stats_measured),
        "mean_queue": _mean([float(q) for q in env.queue_samples]),
        "max_queue": max(env.queue_samples) if env.queue_samples else None,
    }


def ack_view(env: FANETEnv) -> Dict[str, Any]:
    """Compute the drones' own view, built only from ACK information.

    Sources know what they created, what the GS acknowledged, and what timed
    out unacknowledged. Senders know, per outgoing link, how many sends were
    accepted and how each failure went. Nothing here reads the simulator's
    packet list.

    Args:
        env: A finished environment.

    Returns:
        A dict shaped to line up with :func:`ground_truth_view`.
    """
    created = sum(d.packets_created_measured for d in env.drones)
    acked = sum(d.packets_gs_acked_measured for d in env.drones)
    lost = sum(d.packets_lost_no_ack_measured for d in env.drones)

    # Per-link rows, as the drones that own those links recorded them.
    link_stats: Dict[Any, Dict[str, int]] = {}
    for drone in env.drones:
        for next_hop, row in drone.link_stats_measured.items():
            link_stats[(drone.drone_id, next_hop)] = dict(row)

    return {
        "created": created,
        "delivered": acked,
        "lost_no_ack": lost,
        # Not windowed: these are packets the sources are still waiting on at
        # the end of the episode, most of them created after the drain cut.
        "still_waiting": sum(len(d.outstanding) for d in env.drones),
        "delivery_rate": _rate(acked, created),
        "total_loss": _rate(lost, created),
        "loss_rate": _rate(lost, created),
        # What the drones' own link books say went wrong, by cause.
        "lost_queue_full": sum(
            r["dropped_queue_full"] for r in link_stats.values()
        ),
        "lost_channel": sum(r["lost_channel"] for r in link_stats.values()),
        "expired_in_queue": sum(
            r["expired_in_queue"] for r in link_stats.values()
        ),
        "links": _link_summary(link_stats),
    }


def compare_views(truth: Dict[str, Any], acks: Dict[str, Any]) -> List[str]:
    """Return a list of discrepancies between the two views.

    With ideal ACKs this must come back empty.

    Only like-for-like quantities are compared. Packet counts (creation
    window) go against packet counts; per-link counters (event window) go
    against per-link counters. Comparing a creation-windowed packet count with
    an event-windowed link counter would flag a difference that is only the two
    windows disagreeing, not a bookkeeping error.

    Args:
        truth: Output of :func:`ground_truth_view`.
        acks:  Output of :func:`ack_view`.

    Returns:
        Human-readable difference descriptions, empty when the views agree.
    """
    problems: List[str] = []

    # --- packet level, both sides over the creation window ---
    if truth["created"] != acks["created"]:
        problems.append(
            f"created: ground truth {truth['created']} vs ACK {acks['created']}"
        )
    if truth["delivered"] != acks["delivered"]:
        problems.append(
            f"delivered: ground truth {truth['delivered']} vs ACK {acks['delivered']}"
        )

    # --- per link, both sides over the event window ---
    truth_rows = truth["links"]["raw"]
    ack_rows = acks["links"]["raw"]

    if set(truth_rows) != set(ack_rows):
        only_truth = set(truth_rows) - set(ack_rows)
        only_acks = set(ack_rows) - set(truth_rows)
        problems.append(
            f"link sets differ: simulator-only {sorted(map(str, only_truth))}, "
            f"ACK-only {sorted(map(str, only_acks))}"
        )

    for link in sorted(set(truth_rows) & set(ack_rows), key=str):
        for field in LINK_FIELDS:
            gt_value = truth_rows[link][field]
            ack_value = ack_rows[link][field]
            if gt_value != ack_value:
                problems.append(
                    f"link {link} {field}: ground truth {gt_value} "
                    f"vs ACK {ack_value}"
                )

    return problems


def compute_metrics(env: FANETEnv) -> Dict[str, Any]:
    """Build the full Phase-1A metrics record for one finished run.

    Args:
        env: A finished environment.

    Returns:
        A dict with ``run``, ``ground_truth``, ``acks`` and ``discrepancies``.
    """
    truth = ground_truth_view(env)
    acks = ack_view(env)
    start, stop = config.measurement_window()
    return {
        "run": {
            "routing": env.routing,
            "placement_seed": env.placement_seed,
            "run_seed": env.run_seed,
            "steps": env.step_count,
            "load_interval_steps": config.PACKET_INTERVAL_STEPS,
            "load_ms": config.PACKET_INTERVAL_STEPS * config.TIMESTEP * 1000,
            "window": (start, stop),
        },
        "ground_truth": truth,
        "acks": acks,
        "discrepancies": compare_views(truth, acks),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _pct(value: Optional[float]) -> str:
    """Format a rate in [0, 1] as a percentage, or "n/a"."""
    return "n/a" if value is None else f"{value * 100:6.2f}%"


def _num(value: Optional[float], places: int = 2) -> str:
    """Format a number, or "n/a"."""
    return "n/a" if value is None else f"{value:.{places}f}"


def _link_name(link: Tuple[int, Any]) -> str:
    """Render a (from, to) link key as "A->B"."""
    return f"{link[0]}->{link[1]}"


def _print_links(links: Dict[str, Any], indent: str) -> None:
    """Print one view's per-link loss block.

    Args:
        links:  A :func:`_link_summary` result.
        indent: Leading whitespace for every line.
    """
    print(f"{indent}per-link loss  : {links['num_links']} links, "
          f"{links['total_offered']} packets offered"
          f"   [from step {config.WARMUP_STEPS}]")
    print(f"{indent}  overall (pooled) : {_pct(links['pooled_loss'])}"
          f"   queue_full {_pct(links['pooled_loss_dropped_queue_full'])}"
          f"   channel {_pct(links['pooled_loss_lost_channel'])}"
          f"   expired {_pct(links['pooled_loss_expired_in_queue'])}")
    print(f"{indent}  mean over links  : {_pct(links['mean_link_loss'])}")
    worst = links["worst_link"]
    if worst is not None:
        print(f"{indent}  worst link       : {_link_name(worst['link'])}"
              f"  loss {_pct(worst['loss'])}"
              f"  (queue_full {_pct(worst['loss_dropped_queue_full'])},"
              f" channel {_pct(worst['loss_lost_channel'])},"
              f" expired {_pct(worst['loss_expired_in_queue'])})"
              f"  over {worst['offered']} offered")


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print the Phase-1A report for one run.

    Args:
        metrics: Output of :func:`compute_metrics`.
    """
    run = metrics["run"]
    truth = metrics["ground_truth"]
    acks = metrics["acks"]

    width = 66
    print("=" * width)
    print(f"  Phase-1A run report — routing = {run['routing']}")
    print("=" * width)
    print(f"  seeds          : placement={run['placement_seed']}  run={run['run_seed']}")
    print(f"  steps          : {run['steps']}")
    print(f"  load           : 1 packet / M-drone / {run['load_interval_steps']} steps"
          f"  ({run['load_ms']:.0f} ms)")
    print(f"  measured over  : packets created in steps "
          f"[{run['window'][0]}, {run['window'][1]})")

    print("-" * width)
    print("  GROUND TRUTH (simulator's view)")
    print(f"    created           : {truth['created']}")
    print(f"    delivered         : {truth['delivered']}   "
          f"delivery rate {_pct(truth['delivery_rate'])}")
    print(f"    TOTAL PACKET LOSS : {_pct(truth['total_loss'])}"
          f"   ({truth['dropped'] + truth['still_in_flight']} packets)")
    if truth["still_in_flight"]:
        print(f"      of which still in flight: {truth['still_in_flight']}"
              "  (the drain cut should have emptied these)")

    print("    loss by cause  :")
    for label in DROP_REASON_LABELS:
        count = truth["loss_by_reason"][label]
        rate = truth["loss_rate_by_reason"][label]
        print(f"      {label:<12} {count:>7}   {_pct(rate)}")

    print(f"    delay (delivered) : mean {_num(truth['mean_delay_steps'])} steps"
          f"  ({_num(truth['mean_delay_ms'], 1)} ms)")
    print(f"    hops  (delivered) : mean {_num(truth['mean_hops'])}"
          f"   max {truth['max_hops']}")
    print(f"    link-queue occupancy : mean {_num(truth['mean_queue'])}"
          f"   max {truth['max_queue']}  (cap {config.QUEUE_CAPACITY} per link)")

    _print_links(truth["links"], "    ")

    print("-" * width)
    print("  ACK-BASED / LOCAL (drones' view)")
    print(f"    created           : {acks['created']}")
    print(f"    GS-acked          : {acks['delivered']}   "
          f"delivery rate {_pct(acks['delivery_rate'])}")
    print(f"    TOTAL PACKET LOSS : {_pct(acks['total_loss'])}"
          f"   ({acks['lost_no_ack']} packets, no GS ACK before TTL)")
    print(f"    still awaiting    : {acks['still_waiting']}"
          f"   [all packets, incl. those created after the window]")
    print(f"    own link books    : queue_full {acks['lost_queue_full']}"
          f"   channel {acks['lost_channel']}"
          f"   expired_in_queue {acks['expired_in_queue']}"
          f"   [from step {config.WARMUP_STEPS}]")

    _print_links(acks["links"], "    ")

    print("-" * width)
    problems = metrics["discrepancies"]
    if problems:
        print("  VIEWS DISAGREE — with ideal ACKs they must not:")
        for problem in problems:
            print(f"    ! {problem}")
    else:
        print("  views agree: ACK-based counts match ground truth exactly.")
    print("=" * width)
