"""
curriculum_step1.py — Phase 2c curriculum step 1: the new mission, measured.

No training. This runs the two hand-written C-drone rules over the "near_gs"
mission at four settings of the difficulty dial, so the rungs of the
curriculum can be read off before anything is trained on them.

The mission: the whole fleet starts inside 100 m of the ground station, and
each M-drone then flies a distance drawn from [D/2, D] in a random direction
and hovers. D = config.MISSION_DISTANCE_M is the dial. D = 0 is a world where
nothing moves at all, which is the easiest rung and the sanity check: it
should look like the static world.

What makes the rungs comparable is that D is drawn LAST among a drone's
mobility draws, so a layout flies the same directions at the same speeds
after the same delays at every D -- only further.

Settings: validation layouts 301-330, 200 ms, greedy routing, hellos every
1 s (so the neighbour tables are real, not ideal), C-drones sharing targets
at a 150 m spread, run seed 9100 + (layout - 301).

Writes:
    out/2c_curriculum1.csv          one row per (D, rule, layout)
    out/2c_curriculum1_summary.csv  one row per (D, rule)
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
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
from scripts.metrics_1a import (
    ack_view,
    compare_views,
    ground_truth_view,
    loss_by_time_window,
    mission_spread,
    time_window_labels,
)
from scripts.stamp import csv_comment

#: The layouts the curriculum is measured on. Held back from M3's training,
#: and the same thirty every stage validates on.
VALIDATION_LAYOUTS: Tuple[int, ...] = tuple(range(301, 331))

#: Run seed of a validation layout, as every 2c stage has numbered them.
VALIDATION_RUN_SEED_BASE: int = 9100

#: The rungs of the dial, in metres. 0 is "nobody moves".
DISTANCES: Tuple[float, ...] = (0.0, 150.0, 250.0, 350.0)

#: Stands in for "the static world" in the distance list: the Phase-1 layout,
#: uniform over the whole arena, with nothing moving at all. Not a rung of
#: this curriculum -- it is the world M0-M3 trained in, carried along as a
#: reference row so the rungs can be read against something known.
STATIC: float = -1.0

#: The two hand-written rules. No RL here: nothing has been trained on this
#: mission yet, and that is the point of the step.
RULES: Tuple[str, ...] = ("hover", "midpoint")

LOAD_MS: int = 200
ROUTING: str = "greedy"
HELLO_INTERVAL_S: float = 1.0
TARGET_SPREAD_M: float = 150.0

#: C-drone collision avoidance. True is what M3 and M4 ran and what the
#: design keeps; it is set explicitly here because leaving it to the config
#: default silently ran step 1 with it OFF, against every other 2c result.
AVOID: bool = True

CAUSES: Tuple[str, ...] = (
    "channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit",
    "link_broken", "stale_link",
)


def run_seed_for(layout: int) -> int:
    """Return the run seed of a validation layout.

    Args:
        layout: A placement seed in VALIDATION_LAYOUTS.

    Returns:
        9100 + (layout - 301).
    """
    return VALIDATION_RUN_SEED_BASE + layout - VALIDATION_LAYOUTS[0]


def _world(distance: float, rule: str, avoid: bool = AVOID) -> Tuple:
    """Put config on the curriculum mission and return what it was.

    Set inside the worker, because a spawned process imports config fresh and
    inherits nothing the parent changed.

    Args:
        distance: D in metres, or STATIC for the Phase-1 static world.
        rule:     "hover" or "midpoint".
        avoid:    C-drone collision avoidance.

    Returns:
        The previous values, for :func:`_restore`.
    """
    previous = (config.MISSION, config.MISSION_DISTANCE_M, config.STATIC_MODE,
                config.C_RULE, config.IDEAL_HELLO, config.HELLO_INTERVAL,
                config.SHARE_TARGETS, config.C_TARGET_SPREAD_M,
                config.PACKET_INTERVAL_STEPS, config.AVOID)
    if distance == STATIC:
        # The world M0-M3 ran in: uniform placement over the whole arena and
        # no mission movement. MISSION is left alone because nothing reads it
        # while STATIC_MODE is on.
        config.STATIC_MODE = True
    else:
        config.MISSION = "near_gs"
        config.MISSION_DISTANCE_M = float(distance)
        config.STATIC_MODE = False  # the MISSION flies; D = 0 means 0 m
    config.C_RULE = rule
    config.IDEAL_HELLO = False
    config.HELLO_INTERVAL = HELLO_INTERVAL_S
    config.SHARE_TARGETS = True     # inert for hover: only midpoint and
    config.C_TARGET_SPREAD_M = TARGET_SPREAD_M   # busiest choose a target
    config.AVOID = avoid
    config.PACKET_INTERVAL_STEPS = int(
        round(LOAD_MS / 1000.0 / config.TIMESTEP)
    )
    return previous


def _restore(previous: Tuple) -> None:
    """Put config back.

    Args:
        previous: What :func:`_world` returned.
    """
    (config.MISSION, config.MISSION_DISTANCE_M, config.STATIC_MODE,
     config.C_RULE, config.IDEAL_HELLO, config.HELLO_INTERVAL,
     config.SHARE_TARGETS, config.C_TARGET_SPREAD_M,
     config.PACKET_INTERVAL_STEPS, config.AVOID) = previous


def label_for(distance: float) -> str:
    """Return how a distance is printed.

    Args:
        distance: D in metres, or STATIC.

    Returns:
        "static", or the distance in metres.
    """
    return "static" if distance == STATIC else f"{distance:.0f}"


def _spread(env: FANETEnv) -> Dict[str, float]:
    """Return how far the mission got, measured at the end of the run.

    One line, because the measurement itself lives in metrics_1a and the
    trainer reports the same numbers from the same function. Keeping two
    copies is how the two tables would drift apart.

    Args:
        env: A finished environment.

    Returns:
        What :func:`scripts.metrics_1a.mission_spread` returns.
    """
    return mission_spread(env)


def run_once(distance: float, rule: str, layout: int,
             avoid: bool = AVOID) -> Dict[str, Any]:
    """Run one layout under one rule at one D.

    Args:
        distance: D in metres, or STATIC.
        rule:     "hover" or "midpoint".
        layout:   Placement seed.
        avoid:    C-drone collision avoidance.

    Returns:
        The row for the runs CSV.
    """
    previous = _world(distance, rule, avoid)
    try:
        run_seed = run_seed_for(layout)
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed)
        env.reset()
        # MAX_STEPS regardless of the done flag, the way experiment_2a runs:
        # the measurement window is a property of the episode length, not of
        # whether anyone is still flying.
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        truth = ground_truth_view(env)
        acks = ack_view(env)
        by_reason = truth["loss_rate_by_reason"]
        windows = loss_by_time_window(env)

        return {
            "mission_distance_m": distance,
            "setting": label_for(distance),
            "avoid": avoid,
            "rule": rule,
            "placement_seed": layout,
            "run_seed": run_seed,
            "total_loss": 100.0 * (truth["total_loss"] or 0.0),
            **{cause: 100.0 * (by_reason[cause] or 0.0) for cause in CAUSES},
            "pct_packets_connected": 100.0 * (
                truth["pct_packets_connected"] or 0.0),
            "loss_while_connected": 100.0 * (
                truth["loss_while_connected"] or 0.0),
            **{f"loss_{label}": 100.0 * (value if value is not None else 0.0)
               for label, value in windows.items()},
            **_spread(env),
            "c_distance_flown_mean": truth["c_distance_flown_mean"] or 0.0,
            "views_agree": not compare_views(truth, acks),
        }
    finally:
        _restore(previous)


def _job(args: Tuple) -> Dict[str, Any]:
    """Unpack one job for the pool.

    Args:
        args: (distance, rule, layout, avoid).

    Returns:
        The run's row.
    """
    return run_once(*args)


def build_jobs(distances: Sequence[float] = DISTANCES,
               avoid: bool = AVOID) -> List[Tuple]:
    """Return every (D, rule, layout, avoid) to run.

    Args:
        distances: The settings to sweep, STATIC included if wanted.
        avoid:     C-drone collision avoidance.

    Returns:
        One job per (setting, rule, layout).
    """
    return [(distance, rule, layout, avoid)
            for distance in distances
            for rule in RULES
            for layout in VALIDATION_LAYOUTS]


def run_grid(workers: int, distances: Sequence[float] = DISTANCES,
             avoid: bool = AVOID) -> List[Dict[str, Any]]:
    """Run every job and return its rows.

    Args:
        workers:   Pool size.
        distances: The settings to sweep.
        avoid:     C-drone collision avoidance.

    Returns:
        One row per job.

    Raises:
        RuntimeError: If any run's ACK view disagreed with the simulator.
    """
    jobs = build_jobs(distances, avoid)
    print(f"Running {len(jobs)} runs: {len(distances)} settings x "
          f"{len(RULES)} rules x {len(VALIDATION_LAYOUTS)} layouts, "
          f"avoid={avoid}, no training ...", flush=True)

    if workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=context) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    disagreed = [row for row in rows if not row["views_agree"]]
    print(f"  views agreed in {len(rows) - len(disagreed)} of {len(rows)} runs")
    if disagreed:
        raise RuntimeError("the ACK views disagreed with the simulator")
    return rows


def _mean(values: Sequence[Optional[float]]) -> float:
    """Mean of the values that exist.

    Args:
        values: Possibly containing None.

    Returns:
        The mean, or 0.0 if nothing is there.
    """
    present = [v for v in values if v is not None]
    return statistics.fmean(present) if present else 0.0


def _std(values: Sequence[Optional[float]]) -> float:
    """Sample standard deviation of the values that exist.

    Args:
        values: Possibly containing None.

    Returns:
        The deviation, or 0.0 if fewer than two values are there.
    """
    present = [v for v in values if v is not None]
    return statistics.stdev(present) if len(present) > 1 else 0.0


SUMMARY_MEANS: Tuple[str, ...] = (
    ("total_loss",) + CAUSES
    + ("pct_packets_connected", "loss_while_connected")
    + tuple(f"loss_{label}" for label in time_window_labels())
    + ("m_gs_distance_end_mean", "pct_m_beyond_reach_end",
       "pct_m_clipped_end", "c_distance_flown_mean")
)


def summarise(rows: Sequence[Dict[str, Any]],
              distances: Sequence[float] = DISTANCES
              ) -> List[Dict[str, Any]]:
    """Collapse the per-layout rows to one row per (D, rule).

    Args:
        rows:      Every run's row.
        distances: The settings, in the order to report them.

    Returns:
        One row per setting, in the order the grid was built.
    """
    out: List[Dict[str, Any]] = []
    for distance in distances:
        for rule in RULES:
            group = [row for row in rows
                     if row["mission_distance_m"] == distance
                     and row["rule"] == rule]
            summary: Dict[str, Any] = {
                "mission_distance_m": distance,
                "setting": label_for(distance),
                "avoid": group[0]["avoid"] if group else None,
                "rule": rule,
                "layouts": len(group),
                "total_loss_std": _std([row["total_loss"] for row in group]),
            }
            for column in SUMMARY_MEANS:
                summary[column] = _mean([row[column] for row in group])
            out.append(summary)
    return out


def print_totals(summary: Sequence[Dict[str, Any]]) -> None:
    """Print total loss and loss by cause.

    Args:
        summary: The per-setting rows.
    """
    print("\n  Validation 301-330, near_gs mission — % of packets created")
    header = f"  {'D':>6} {'rule':>9} {'total':>8} {'std':>6}"
    header += "".join(f"{cause[:9]:>11}" for cause in CAUSES)
    header += f"{'conn%':>9}{'loss|conn':>11}"
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    for row in summary:
        line = (f"  {row['setting']:>6} {row['rule']:>9} "
                f"{row['total_loss']:>7.2f}% {row['total_loss_std']:>6.2f}")
        line += "".join(f"{row[cause]:>10.2f}%" for cause in CAUSES)
        line += (f"{row['pct_packets_connected']:>8.2f}%"
                 f"{row['loss_while_connected']:>10.2f}%")
        print(line)
    print("  " + "-" * (len(header) + 2))


def print_windows(summary: Sequence[Dict[str, Any]]) -> None:
    """Print loss per ten-second slice.

    Args:
        summary: The per-setting rows.
    """
    labels = time_window_labels()
    print("\n  Loss per ten seconds, by the slice a packet was created in"
          " — % of packets created in that slice")
    header = f"  {'D':>6} {'rule':>9}" + "".join(f"{l:>9}" for l in labels)
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    for row in summary:
        line = f"  {row['setting']:>6} {row['rule']:>9}"
        line += "".join(f"{row[f'loss_{label}']:>8.2f}%" for label in labels)
        print(line)
    print("  " + "-" * (len(header) + 2))


def print_spread(summary: Sequence[Dict[str, Any]],
                 rows: Sequence[Dict[str, Any]]) -> None:
    """Print how far the mission got by the end of the run.

    Args:
        summary: The per-setting rows.
        rows:    The per-layout rows, for the reach.
    """
    reach = rows[0]["reach_m"] if rows else channel.max_link_distance()
    print(f"\n  Where the mission ended up, at the last step "
          f"(reach = {reach:.0f} m)")
    header = (f"  {'D':>6} {'rule':>9} {'mean M from GS':>16} "
              f"{'% M beyond reach':>18} {'% M clipped':>13} {'C flown':>10}")
    print("  " + "-" * (len(header) + 2))
    print(header)
    print("  " + "-" * (len(header) + 2))
    for row in summary:
        print(f"  {row['setting']:>6} {row['rule']:>9} "
              f"{row['m_gs_distance_end_mean']:>15.1f}m "
              f"{row['pct_m_beyond_reach_end']:>17.1f}% "
              f"{row['pct_m_clipped_end']:>12.1f}% "
              f"{row['c_distance_flown_mean']:>9.0f}m")
    print("  " + "-" * (len(header) + 2))


def print_against_hover(summary: Sequence[Dict[str, Any]]) -> None:
    """Print what moving the C-drones is worth at each D.

    Args:
        summary: The per-setting rows.
    """
    print("\n  What midpoint is worth at each D — points of total loss")
    print("  " + "-" * 52)
    print(f"  {'D':>6} {'hover':>9} {'midpoint':>10} {'difference':>12}")
    print("  " + "-" * 52)
    seen = []
    for row in summary:
        if row["mission_distance_m"] in seen:
            continue
        seen.append(row["mission_distance_m"])
        rules = {other["rule"]: other["total_loss"] for other in summary
                 if other["mission_distance_m"] == row["mission_distance_m"]}
        difference = rules["midpoint"] - rules["hover"]
        print(f"  {row['setting']:>6} {rules['hover']:>8.2f}% "
              f"{rules['midpoint']:>9.2f}% {difference:>11.2f}")
    print("  " + "-" * 52)


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              columns: Sequence[str], note: str) -> None:
    """Write *rows* to *path* under a settings stamp.

    Args:
        path:    Where to write.
        rows:    The rows.
        columns: Column order.
        note:    Appended to the stamp.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=[LOAD_MS],
                                 placements=len(VALIDATION_LAYOUTS),
                                 extra=note))
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in columns})
    print(f"  wrote {path}")


def stamp_note(distances: Sequence[float] = DISTANCES,
               avoid: bool = AVOID) -> str:
    """Return the mission settings, for the stamp.

    Args:
        distances: The settings this sweep covered.
        avoid:     C-drone collision avoidance.

    Returns:
        A short description of the mission this sweep flew.
    """
    return (f"phase 2c curriculum step 1, mission=near_gs, "
            f"D={'/'.join(label_for(d) for d in distances)} m, "
            f"start_radius={config.START_RADIUS_M:.0f} m, "
            f"min_spacing={config.MIN_SPACING_M:.0f} m, "
            f"validation {VALIDATION_LAYOUTS[0]}-{VALIDATION_LAYOUTS[-1]}, "
            f"hellos {HELLO_INTERVAL_S:.0f} s, target spread "
            f"{TARGET_SPREAD_M:.0f} m, avoid={avoid}, greedy routing, "
            f"NO training")


def main() -> None:
    """Run the sweep and print the tables."""
    parser = argparse.ArgumentParser(
        description="Phase 2c curriculum step 1: the near_gs mission."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1))
    parser.add_argument("--distances", type=float, nargs="+",
                        default=list(DISTANCES),
                        help="D in metres; -1 means the static world")
    parser.add_argument("--out-prefix", type=str, default="2c_curriculum1")
    parser.add_argument("--no-avoid", dest="avoid", action="store_false",
                        default=AVOID,
                        help="C-drone collision avoidance off, as step 1 "
                             "ran before it was set explicitly")
    args = parser.parse_args()

    distances = tuple(args.distances)
    rows = run_grid(args.workers, distances, args.avoid)
    summary = summarise(rows, distances)

    print_totals(summary)
    print_windows(summary)
    print_spread(summary, rows)
    print_against_hover(summary)

    run_columns = (["mission_distance_m", "setting", "avoid", "rule",
                    "placement_seed", "run_seed", "total_loss"] + list(CAUSES)
                   + ["pct_packets_connected", "loss_while_connected"]
                   + [f"loss_{label}" for label in time_window_labels()]
                   + ["m_gs_distance_end_mean", "pct_m_beyond_reach_end",
                      "pct_m_clipped_end", "reach_m",
                      "c_distance_flown_mean", "views_agree"])
    summary_columns = (["mission_distance_m", "setting", "avoid", "rule",
                        "layouts", "total_loss", "total_loss_std"]
                       + list(CAUSES)
                       + ["pct_packets_connected", "loss_while_connected"]
                       + [f"loss_{label}" for label in time_window_labels()]
                       + ["m_gs_distance_end_mean", "pct_m_beyond_reach_end",
                          "pct_m_clipped_end", "c_distance_flown_mean"])

    note = stamp_note(distances, args.avoid)
    write_csv(os.path.join(args.out_dir, f"{args.out_prefix}.csv"),
              rows, run_columns, note)
    write_csv(os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv"),
              summary, summary_columns, note)


if __name__ == "__main__":
    main()
