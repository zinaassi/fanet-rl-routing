"""
check_curve.py — What a STEEPER loss curve would do to the Phase 1a results.

A QUICK CHECK. Everything it writes goes to ``out/check_curve/``. It changes
no default, no committed CSV and no committed figure: the curve is patched
into :mod:`fanet_sim.envs.channel` for the duration of a run and restored
afterwards.

The curve
---------
Today's curve is pinned by two points: 10% loss where the margin is zero (the
receiver sensitivity, 249.69 m) and 50% loss at the 350 m link edge. The
check keeps the 50% point exactly where it is and moves the 10% point
outwards, to 275 m and then 300 m. The transition from "good" to "gone" then
happens over 75 m or 50 m instead of 100 m, which is a steeper curve.

Both anchors are distances, so the fit is:

    p_loss(M) = 1 / (1 + A * exp(b * M))

    p_loss(d10) = 0.10  ->  9 = A * exp(b * M(d10))
    p_loss(350) = 0.50  ->  1 = A * exp(b * M(350))

    b = ln(9) / (M(d10) - M(350))        A = exp(-b * M(350))

so both ``b`` and ``A`` move. ``A`` is not free: it is what puts the 50%
point at 350 m once ``b`` is chosen. Feeding the sensitivity distance back in
as ``d10`` reproduces today's ``b`` and ``A = 9`` exactly, which
:func:`check_the_fit` asserts before anything is run.

Because every curve crosses 50% at exactly 350 m, and a link exists while
``p_loss < LINK_MAX_LOSS = 0.5``, the link edge is 350 m under all three.
The layouts are therefore the same for all three curves — measured, not
assumed, by :func:`layout_fingerprints`.

No RL. Only the two Phase 1a rules, greedy (the classic geographic rule) and
random, on the same 20 placement seeds at all four loads.

Writes:
    out/check_curve/runs.csv       one row per run
    out/check_curve/paired.csv     greedy vs random per curve and load
    out/check_curve/geometry.csv   the layout fingerprint per curve
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.evaluate_1b_large import sign_test
from scripts.stamp import csv_comment
from scripts.train_1b import run_once

#: The loss the near anchor stands for. 802.11 defines the receiver
#: sensitivity as the power giving this packet error rate, which is where
#: today's curve gets its near anchor.
NEAR_LOSS = 0.10

#: The 10% distances to compare. ``None`` means "leave the defaults alone",
#: so the baseline row is today's curve itself rather than a refit of it.
CURVES: Tuple[Tuple[str, Optional[float]], ...] = (
    ("250 (today)", None),
    ("275", 275.0),
    ("300", 300.0),
)

LOADS_MS = (1000, 500, 200, 100)
RULES = ("greedy", "random")
PLACEMENTS = tuple(range(1, 21))

CAUSES = ("channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit")

RUN_COLUMNS = (
    "curve", "d10_m", "b_per_db", "a", "rule", "load_ms", "placement_seed",
    "total_loss", *CAUSES, "mean_hops", "views_agree",
)


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------

def curve_params(d10_m: float) -> Tuple[float, float]:
    """Return ``(b, A)`` for the curve through *d10_m* and the link edge.

    Args:
        d10_m: Distance that should lose exactly ``NEAR_LOSS`` of what is
               sent on it. Must be inside the link edge.

    Returns:
        The steepness ``b`` in 1/dB and the prefactor ``A``.

    Raises:
        ValueError: If *d10_m* is not strictly inside the link edge, where
            the two anchors would coincide or cross.
    """
    if not 0.0 < d10_m < config.LOSS_50_DISTANCE_M:
        raise ValueError(
            f"d10_m must lie strictly inside the {config.LOSS_50_DISTANCE_M} m "
            f"edge, got {d10_m}"
        )
    near_odds = (1.0 - NEAR_LOSS) / NEAR_LOSS
    margin_near = channel.margin_db(d10_m)
    margin_edge = channel.margin_db(config.LOSS_50_DISTANCE_M)

    b = math.log(near_odds) / (margin_near - margin_edge)
    a = math.exp(-b * margin_edge)
    return b, a


def apply_curve(d10_m: Optional[float]) -> Tuple[float, float]:
    """Patch the channel module to the curve through *d10_m*.

    Args:
        d10_m: The 10% distance, or ``None`` to leave today's curve alone.

    Returns:
        The ``(b, A)`` now in force, so the caller can record them.
    """
    if d10_m is None:
        return channel.B_PER_DB, channel.A
    b, a = curve_params(d10_m)
    channel.B_PER_DB = b
    channel.A = a
    return b, a


def restore_curve(b: float, a: float) -> None:
    """Put the channel module back to the curve it had.

    Args:
        b: The steepness to restore.
        a: The prefactor to restore.
    """
    channel.B_PER_DB = b
    channel.A = a


def near_distance(d10_m: Optional[float]) -> float:
    """Return the 10% distance a curve spec stands for, in metres.

    Args:
        d10_m: The spec, where ``None`` means today's curve.

    Returns:
        The distance where the curve loses ``NEAR_LOSS``.
    """
    return channel.SENSITIVITY_DISTANCE_M if d10_m is None else d10_m


def check_the_fit() -> None:
    """Verify the fit before any run uses it.

    Three things have to hold, or the comparison is not what it claims:
    refitting at today's near anchor must reproduce today's curve, every
    curve must pass through both of its anchors, and the link edge must stay
    at 350 m so the layouts cannot move.

    Raises:
        SystemExit: If any of them fails.
    """
    default_b, default_a = channel.B_PER_DB, channel.A
    problems: List[str] = []

    refit_b, refit_a = curve_params(channel.SENSITIVITY_DISTANCE_M)
    if abs(refit_b - default_b) > 1e-12 or abs(refit_a - default_a) > 1e-12:
        problems.append(
            f"refitting at the sensitivity distance gives b={refit_b!r}, "
            f"A={refit_a!r}, not the configured b={default_b!r}, A={default_a!r}"
        )

    for label, spec in CURVES:
        b, a = apply_curve(spec)
        try:
            near = near_distance(spec)
            if abs(channel.p_loss(near) - NEAR_LOSS) > 1e-12:
                problems.append(
                    f"curve {label}: loss at {near:.3f} m is "
                    f"{channel.p_loss(near)!r}, not {NEAR_LOSS}"
                )
            if abs(channel.p_loss(config.LOSS_50_DISTANCE_M) - 0.5) > 1e-12:
                problems.append(
                    f"curve {label}: loss at the edge is "
                    f"{channel.p_loss(config.LOSS_50_DISTANCE_M)!r}, not 0.5"
                )
            edge = channel.max_link_distance()
            if abs(edge - config.LOSS_50_DISTANCE_M) > 1e-9:
                problems.append(
                    f"curve {label}: the link edge moved to {edge!r} m"
                )
            if b <= 0.0 or a <= 0.0:
                problems.append(f"curve {label}: b={b!r}, A={a!r}")
        finally:
            restore_curve(default_b, default_a)

    if problems:
        for problem in problems:
            print(f"STOP: {problem}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# The runs
# ---------------------------------------------------------------------------

def _job(args: Tuple[str, Optional[float], str, int, int]) -> Dict[str, Any]:
    """Worker entry point for one run under one curve.

    Args:
        args: ``(label, d10_m, rule, load_ms, placement_seed)``.

    Returns:
        That run's row.
    """
    label, spec, rule, load_ms, seed = args
    default_b, default_a = channel.B_PER_DB, channel.A
    b, a = apply_curve(spec)
    try:
        result = run_once(rule, seed, seed, load_ms, training=False)
    finally:
        # Restored so a sequential run does not leave the last job's curve
        # behind in the parent.
        restore_curve(default_b, default_a)
    return {
        "curve": label,
        "d10_m": near_distance(spec),
        "b_per_db": b,
        "a": a,
        "rule": rule,
        "load_ms": load_ms,
        "placement_seed": seed,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        "mean_hops": result["mean_hops"],
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def run_grid(workers: int) -> List[Dict[str, Any]]:
    """Run every (curve, rule, load, placement) combination.

    Args:
        workers: Parallel workers; 1 runs sequentially.

    Returns:
        One row per run.

    Raises:
        SystemExit: If any run's two views of the traffic disagree.
    """
    jobs = [
        (label, spec, rule, load_ms, seed)
        for label, spec in CURVES
        for rule in RULES
        for load_ms in LOADS_MS
        for seed in PLACEMENTS
    ]
    print(f"Running {len(jobs)} runs "
          f"({len(CURVES)} curves x {len(RULES)} rules x {len(LOADS_MS)} "
          f"loads x {len(PLACEMENTS)} layouts) ...")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [row for row in rows if not row["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — curve {row['curve']} "
                  f"{row['rule']} seed {row['placement_seed']}: "
                  f"{row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


# ---------------------------------------------------------------------------
# Layout identity
# ---------------------------------------------------------------------------

def layout_fingerprints(spec: Optional[float]) -> Dict[int, Tuple[str, int]]:
    """Return a fingerprint of every layout as drawn under one curve.

    The fingerprint is a hash of all 25 positions in drone-id order together
    with the number of links, so a moved drone and a changed neighbour set
    both show up.

    Args:
        spec: The curve's 10% distance, or ``None`` for today's.

    Returns:
        ``{placement_seed: (positions_hash, link_count)}``.
    """
    default_b, default_a = channel.B_PER_DB, channel.A
    apply_curve(spec)
    try:
        prints: Dict[int, Tuple[str, int]] = {}
        for seed in PLACEMENTS:
            env = FANETEnv(routing="greedy", log_path=os.devnull,
                           placement_seed=seed, run_seed=seed)
            env.reset()
            drones = sorted(env.drones, key=lambda d: d.drone_id)
            positions = np.array([d.position for d in drones],
                                 dtype=np.float64)
            digest = hashlib.sha256(positions.tobytes()).hexdigest()[:16]
            links = sum(len(d.candidates) for d in drones)
            prints[seed] = (digest, links)
            env.close_logger()
        return prints
    finally:
        restore_curve(default_b, default_a)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    """Return the mean, or 0.0 for an empty sequence."""
    return statistics.fmean(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    """Return the sample standard deviation, or 0.0 below two values."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _select(rows: Sequence[Dict[str, Any]], **where: Any) -> List[Dict[str, Any]]:
    """Return the rows matching every ``column=value`` in *where*."""
    return [r for r in rows
            if all(r[column] == value for column, value in where.items())]


def print_causes(rows: Sequence[Dict[str, Any]]) -> None:
    """Print table (a): total loss and loss by cause, per curve, load, rule."""
    print("\n  (a) 20 layouts — mean % of packets created")
    width = 112
    print("  " + "-" * width)
    print(f"  {'curve':>12} {'load':>8} {'rule':>8} {'total':>8} {'std':>6}"
          + "".join(f"{cause.replace('_', ' '):>11}" for cause in CAUSES))
    print("  " + "-" * width)
    for label, _ in CURVES:
        for load_ms in LOADS_MS:
            for rule in RULES:
                chosen = _select(rows, curve=label, load_ms=load_ms, rule=rule)
                totals = [r["total_loss"] * 100 for r in chosen]
                line = (f"  {label:>12} {str(load_ms) + ' ms':>8} {rule:>8} "
                        f"{_mean(totals):7.2f}% {_std(totals):6.2f}")
                for cause in CAUSES:
                    share = _mean([r[cause] * 100 for r in chosen])
                    line += f"{share:10.2f}%"
                print(line)
        print("  " + "-" * width)


def paired_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return greedy minus random per curve and load, paired on the layout."""
    out: List[Dict[str, Any]] = []
    for label, _ in CURVES:
        for load_ms in LOADS_MS:
            diffs = []
            for seed in PLACEMENTS:
                greedy = _select(rows, curve=label, load_ms=load_ms,
                                 rule="greedy", placement_seed=seed)
                random_ = _select(rows, curve=label, load_ms=load_ms,
                                  rule="random", placement_seed=seed)
                diffs.append(greedy[0]["total_loss"] - random_[0]["total_loss"])
            wins, used, p_sign = sign_test(diffs)
            out.append({
                "curve": label,
                "load_ms": load_ms,
                "comparison": "greedy - random",
                "n": len(diffs),
                "mean_diff": _mean(diffs),
                "std_diff": _std(diffs),
                "wins": wins,
                "ties": len(diffs) - used,
                "p_sign": p_sign,
            })
    return out


def print_paired(paired: Sequence[Dict[str, Any]]) -> None:
    """Print table (b): greedy against random, paired on the same layouts."""
    print("\n  (b) Paired on the same layouts: greedy - random")
    print("      negative = greedy loses fewer packets")
    width = 74
    print("  " + "-" * width)
    print(f"  {'curve':>12} {'load':>8} {'mean diff':>11} {'std':>8}"
          f" {'greedy lower':>14} {'p (sign)':>12}")
    print("  " + "-" * width)
    for row in paired:
        print(f"  {row['curve']:>12} {str(row['load_ms']) + ' ms':>8} "
              f"{row['mean_diff'] * 100:10.2f}% {row['std_diff'] * 100:7.2f}% "
              f"{str(row['wins']) + ' / ' + str(row['n']):>14} "
              f"{_p_text(row['p_sign']):>12}")
    print("  " + "-" * width)


def _p_text(p_value: float) -> str:
    """Format a p-value the way the other scripts do."""
    return "<0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def print_hops(rows: Sequence[Dict[str, Any]]) -> None:
    """Print table (c): hops per delivered packet."""
    print("\n  (c) Hops per delivered packet")
    width = 52
    print("  " + "-" * width)
    print(f"  {'curve':>12} {'load':>8} {'greedy':>10} {'random':>10}")
    print("  " + "-" * width)
    for label, _ in CURVES:
        for load_ms in LOADS_MS:
            cells = []
            for rule in RULES:
                chosen = _select(rows, curve=label, load_ms=load_ms, rule=rule)
                cells.append(_mean([r["mean_hops"] for r in chosen
                                    if r["mean_hops"] is not None]))
            print(f"  {label:>12} {str(load_ms) + ' ms':>8} "
                  f"{cells[0]:10.2f} {cells[1]:10.2f}")
        print("  " + "-" * width)


def print_curves() -> None:
    """Print what each curve is, before the results that used it."""
    default_b, default_a = channel.B_PER_DB, channel.A
    print("\n  The three curves (50% loss at 350 m in all three)")
    width = 78
    print("  " + "-" * width)
    print(f"  {'curve':>12} {'10% at':>10} {'b (1/dB)':>10} {'A':>10}"
          f" {'loss @200m':>12} {'loss @300m':>12}")
    print("  " + "-" * width)
    for label, spec in CURVES:
        b, a = apply_curve(spec)
        try:
            print(f"  {label:>12} {near_distance(spec):9.3f}m {b:10.4f} "
                  f"{a:10.4f} {channel.p_loss(200.0) * 100:11.2f}% "
                  f"{channel.p_loss(300.0) * 100:11.2f}%")
        finally:
            restore_curve(default_b, default_a)
    print("  " + "-" * width)


def print_geometry(prints: Dict[str, Dict[int, Tuple[str, int]]]) -> None:
    """Print table (d): whether the layouts are the same under every curve."""
    print("\n  (d) The 20 layouts, drawn under each curve")
    baseline_label = CURVES[0][0]
    baseline = prints[baseline_label]
    width = 66
    print("  " + "-" * width)
    print(f"  {'curve':>12} {'layouts':>9} {'links':>8} "
          f"{'same as ' + baseline_label:>24}")
    print("  " + "-" * width)
    for label, _ in CURVES:
        current = prints[label]
        same = sum(1 for seed in PLACEMENTS
                   if current[seed] == baseline[seed])
        links = sum(current[seed][1] for seed in PLACEMENTS)
        print(f"  {label:>12} {len(current):9d} {links:8d} "
              f"{str(same) + ' / ' + str(len(PLACEMENTS)):>24}")
    print("  " + "-" * width)
    identical = all(prints[label][seed] == baseline[seed]
                    for label, _ in CURVES for seed in PLACEMENTS)
    print(f"  identical positions and link counts across all three curves: "
          f"{identical}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str], note: str) -> None:
    """Write *rows* to *path* under a settings stamp.

    Args:
        path:    Destination file.
        rows:    The rows to write.
        columns: Column order.
        note:    Appended to the stamp, saying what the file is.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=list(LOADS_MS),
                                 placements=len(PLACEMENTS), extra=note))
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    print(f"  wrote {path}")


def main() -> None:
    """Run the three curves and print every table."""
    parser = argparse.ArgumentParser(
        description="Phase 1a under a steeper loss curve (quick check)."
    )
    parser.add_argument("--out-dir", type=str, default="out/check_curve",
                        help="Where the outputs go. Nothing else is touched.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    args = parser.parse_args()

    check_the_fit()
    print_curves()

    prints = {label: layout_fingerprints(spec) for label, spec in CURVES}

    rows = run_grid(args.workers)
    print_causes(rows)
    paired = paired_rows(rows)
    print_paired(paired)
    print_hops(rows)
    print_geometry(prints)

    note = (f"curve check: 10% loss at "
            + "/".join(f"{near_distance(spec):.0f}" for _, spec in CURVES)
            + " m, 50% at 350 m")
    write_csv(os.path.join(args.out_dir, "runs.csv"), rows, RUN_COLUMNS, note)
    write_csv(os.path.join(args.out_dir, "paired.csv"), paired,
              ("curve", "load_ms", "comparison", "n", "mean_diff", "std_diff",
               "wins", "ties", "p_sign"), note)
    geometry = [
        {"curve": label, "placement_seed": seed,
         "positions_sha256_16": prints[label][seed][0],
         "links": prints[label][seed][1]}
        for label, _ in CURVES for seed in PLACEMENTS
    ]
    write_csv(os.path.join(args.out_dir, "geometry.csv"), geometry,
              ("curve", "placement_seed", "positions_sha256_16", "links"),
              note)


if __name__ == "__main__":
    main()
