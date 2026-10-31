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
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

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


def measured_packets(env: FANETEnv,
                     window: Optional[Tuple[int, int]] = None) -> List[Packet]:
    """Return the packets whose creation step falls in the measurement window.

    Args:
        env:    A finished environment.
        window: A ``(start, stop)`` pair of creation steps to use instead of
                the configured measurement window, so a report can be cut
                into slices of the run. Phase 2c M4 uses it to ask what the
                loss was in each ten-second window while the mission drones
                were still flying. ``None``, the default, is the configured
                window, so every earlier caller is unaffected.

    Returns:
        The subset of ``env.all_packets`` the report is computed over.
    """
    if window is None:
        return [
            pkt for pkt in env.all_packets
            if config.in_measurement_window(pkt.created_at)
        ]
    start, stop = window
    return [pkt for pkt in env.all_packets if start <= pkt.created_at < stop]


def time_window_labels(span_s: float = 10.0) -> List[str]:
    """Return the label of each slice the measurement window cuts into.

    Args:
        span_s: Length of a slice, in seconds.

    Returns:
        Labels like "5-15s", in order.
    """
    start, stop = config.measurement_window()
    span = int(round(span_s / config.TIMESTEP))
    return [
        f"{first * config.TIMESTEP:.0f}-"
        f"{min(first + span, stop) * config.TIMESTEP:.0f}s"
        for first in range(start, stop, span)
    ]


def loss_by_time_window(env: FANETEnv,
                        span_s: float = 10.0) -> Dict[str, float]:
    """Return the total loss inside each slice of the measurement window.

    A run where the mission drones are still flying is not one world but a
    sequence of them, and a single average hides whether the loss came from
    the upheaval or from where the drones ended up. Packets are counted by
    the slice they were CREATED in, which is how total_loss counts them.

    Args:
        env:    A finished environment.
        span_s: Length of a slice, in seconds.

    Returns:
        Slice label -> share of packets created in it that never arrived.
    """
    start, stop = config.measurement_window()
    span = int(round(span_s / config.TIMESTEP))
    out: Dict[str, float] = {}
    for first, label in zip(range(start, stop, span),
                            time_window_labels(span_s)):
        packets = measured_packets(env, (first, min(first + span, stop)))
        lost = [pkt for pkt in packets if not pkt.delivered]
        out[label] = _rate(len(lost), len(packets))
    return out


#: The three ways a packet offered to a link fails to get across.
LOSS_PARTS = ("dropped_queue_full", "lost_channel", "expired_in_queue")


def mission_spread(env: FANETEnv) -> Dict[str, Any]:
    """Return how far the mission got, measured at the last step.

    The reach is read from the loss curve, never typed in: it is the distance
    at which a link stops existing, so "beyond the reach" means "cannot talk
    to the GS directly, whatever the geometry in between".

    A flight the border clipped ends exactly ON the margin line in at least
    one axis. One that was not clipped ends at whatever real coordinate its
    drawn distance reached, so landing on the margin otherwise has
    probability zero. Nothing is clipped in a static world, where no end
    point was ever drawn.

    Args:
        env: A finished environment.

    Returns:
        Mean M-drone distance from the GS, the share beyond the reach, the
        share the border cut short, and the reach itself.
    """
    from fanet_sim.envs import channel

    gs = np.asarray(config.GS_POSITION, dtype=np.float64)
    reach = channel.max_link_distance()
    margin = config.BORDER_MARGIN_M
    m_drones = [drone for drone in env.drones if drone.drone_type == "M"]
    distances = [float(np.linalg.norm(drone.position - gs))
                 for drone in m_drones]

    def _clipped(drone: Any) -> bool:
        return any(abs(drone.end_point[axis] - edge) < 1e-9
                   for axis, extent in ((0, config.WIDTH),
                                        (1, config.HEIGHT))
                   for edge in (margin, extent - margin))

    clipped = (0 if config.STATIC_MODE
               else sum(1 for drone in m_drones if _clipped(drone)))

    return {
        "m_gs_distance_end_mean": statistics.fmean(distances),
        "pct_m_beyond_reach_end": 100.0 * sum(
            1 for d in distances if d > reach) / len(distances),
        "pct_m_clipped_end": 100.0 * clipped / len(m_drones),
        "reach_m": reach,
    }


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
        # --- Phase 2a: what the moving world adds ---
        # Mean over M-drones of the share of measured steps on which that
        # drone had SOME path to the GS. 1.0 in the static world.
        "pct_time_connected": _mean([
            count / env.measured_steps
            for count in env.connected_steps.values()
        ]) if env.measured_steps else None,
        # Loss over the packets whose source could still reach the GS when it
        # created them. Separates "the routing lost it" from "there was
        # nowhere for it to go".
        **_connected_loss(packets),
        "reroutes": env.reroutes,
        "link_broken_drops": env.link_broken_drops,
        # --- Phase 2a step 2: how good the neighbour tables were ---
        # Of the neighbours a drone believed it had, the share that were
        # real; and of its real neighbours, the share it knew about. Both 1.0
        # under IDEAL_HELLO, by construction.
        "table_precision": (
            env._accuracy_precision / env._accuracy_samples
            if env._accuracy_samples else None
        ),
        "table_recall": (
            env._accuracy_recall / env._accuracy_samples
            if env._accuracy_samples else None
        ),
        # Sends aimed at a neighbour that had really moved out of range, and
        # the ones the channel then lost.
        "stale_sends": env.stale_sends,
        "stale_link_drops": env.stale_link_drops,
        **_hello_rates(env),
        # --- Phase 2a step 3: what the C-drones did ---
        "c_distance_flown_mean": _mean([
            d.distance_flown for d in env.drones if d.drone_type == "C"
        ]),
        "c_distance_from_gs": list(env.c_distance_series),
        "avoid_turns": sum(d.avoid_turns for d in env.drones),
        "avoid_waits": sum(d.avoid_waits for d in env.drones),
        # Each approach inside C_SAFE_DISTANCE_M counted once, by kind.
        "close_passes_cc": env.close_pass_events.get("CC", 0),
        "close_passes_cm": env.close_pass_events.get("CM", 0),
        "close_passes_mm": env.close_pass_events.get("MM", 0),
        "close_passes": sum(env.close_pass_events.values()),
        "closest_approach_m": (
            None if env.closest_approach_m == float("inf")
            else env.closest_approach_m
        ),
    }


def _hello_rates(env: FANETEnv) -> Dict[str, Any]:
    """Return hellos sent and heard, per drone per second.

    Counted over the whole run rather than the measurement window: a hello
    schedule does not warm up, and the rate is a property of the setting, not
    of the traffic.

    Args:
        env: A finished environment.

    Returns:
        ``hellos_sent_per_drone_s`` and ``hellos_received_per_drone_s``.
    """
    seconds = config.MAX_STEPS * config.TIMESTEP
    drones = len(env.drones)
    if not drones or seconds <= 0:
        return {"hellos_sent_per_drone_s": None,
                "hellos_received_per_drone_s": None}
    sent = sum(d.hellos_sent for d in env.drones)
    received = sum(d.hellos_received for d in env.drones)
    return {
        "hellos_sent_per_drone_s": sent / drones / seconds,
        "hellos_received_per_drone_s": received / drones / seconds,
    }


def _connected_loss(packets: Sequence[Packet]) -> Dict[str, Any]:
    """Split the loss by whether the source could reach the GS at all.

    A packet whose source had no path to the GS at the moment it was created
    could not have been delivered by any routing rule, so charging it to the
    routing says nothing. Reported both ways, with the share of packets in
    each group, so that

        total_loss = c * loss_while_connected
                     + (1 - c) * loss_while_disconnected

    holds exactly, with ``c = pct_packets_connected``. tests/test_2a_fixes.py
    checks that identity on every run.

    "% connected" is measured PER PACKET, at the moment of creation — not as
    a share of steps. A drone that is cut off while it happens to be silent
    costs the network nothing, and should not read as if it did.

    Args:
        packets: The packets inside the measurement window.

    Returns:
        The counts and rates for both groups.
    """
    connected = [p for p in packets if p.source_connected]
    cut_off = [p for p in packets if not p.source_connected]
    lost_connected = [p for p in connected if not p.delivered]
    lost_cut_off = [p for p in cut_off if not p.delivered]
    return {
        "pct_packets_connected": _rate(len(connected), len(packets)),
        "created_while_connected": len(connected),
        "lost_while_connected": len(lost_connected),
        "loss_while_connected": _rate(len(lost_connected), len(connected)),
        "created_while_disconnected": len(cut_off),
        "lost_while_disconnected": len(lost_cut_off),
        "loss_while_disconnected": _rate(len(lost_cut_off), len(cut_off)),
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
        "lost_link_broken": sum(
            r["dropped_link_broken"] for r in link_stats.values()
        ),
        "lost_stale_link": sum(
            r["lost_stale_link"] for r in link_stats.values()
        ),
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
