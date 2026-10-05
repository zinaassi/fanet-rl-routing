"""
experiment_1a.py — The Phase-1A experiment grid (spec item K).

Runs every combination of

    routing    : greedy, random
    load       : one packet per M-drone every 1000 / 500 / 200 / 100 ms
    placement  : placement_seed 1..20   (run_seed = placement_seed)

which is 2 x 4 x 20 = 160 runs of 1000 steps each. Both routing rules run on
identical placements and identical channel/traffic randomness, so the pairs are
directly comparable.

Every run is checked: if the ACK-based view ever disagrees with the simulator's
ground truth, the script stops immediately and says which run.

Outputs, all under out/:
    1a_runs.csv                    one row per run
    1a_summary.csv                 mean and std per (routing, load)
    1a_paired.csv                  greedy - random, per load, same seeds
    1a_total_loss_vs_load.png      total loss vs load, one line per rule
    1a_loss_heatmaps.png           loss and its causes, both rules
    1a_link_matrix_placement1.png  per-link loss matrices, placement 1

Usage:
    python scripts/experiment_1a.py
    python scripts/experiment_1a.py --placements 5      # a quicker smoke run
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from fanet_sim import config
from fanet_sim.envs.channel import are_connected
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.utils.metrics import build_graph_with_gs
from scripts.metrics_1a import compute_metrics

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

ROUTING_RULES = ("greedy", "random")
LOADS_MS = (1000, 500, 200, 100)
DEFAULT_PLACEMENTS = 20
OUT_DIR = "out"

#: Column order of 1a_runs.csv.
RUN_COLUMNS = [
    "placement_seed", "run_seed", "routing", "load_ms",
    "created", "delivered", "total_loss",
    "loss_channel", "loss_queue_full", "loss_no_route",
    "loss_ttl", "loss_hop_limit",
    "per_link_loss_pooled", "per_link_queue_full", "per_link_channel",
    "per_link_expired",
    "mean_queue_occupancy", "max_queue_occupancy",
    "mean_delay_steps", "mean_hops",
    "isolated_M_count", "drones_in_GS_range", "mean_neighbors_per_drone",
    "placement_draws", "views_agree",
]

#: The numeric columns that get a mean and std in the summary.
NUMERIC_COLUMNS = [c for c in RUN_COLUMNS if c not in
                   ("routing", "views_agree")]

# ---------------------------------------------------------------------------
# Figure styling. Categorical hues validated as a 2-slot categorical palette.
# ---------------------------------------------------------------------------
SERIES_COLORS = {"greedy": "#2a78d6", "random": "#eb6834"}
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
SURFACE = "#fcfcfb"
UNUSED_GRAY = "#d6d5cf"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Phase-1A experiment grid.")
    parser.add_argument(
        "--placements",
        type=int,
        default=DEFAULT_PLACEMENTS,
        help=f"How many placement seeds, 1..N (default {DEFAULT_PLACEMENTS}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR,
        help="Directory for the CSVs and figures.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Layout facts (pure geometry on the run's own placement)
# ---------------------------------------------------------------------------

def layout_facts(env: FANETEnv) -> Tuple[int, int, float]:
    """Describe the layout: isolation, GS reach and link density.

    Args:
        env: A reset environment.

    Returns:
        ``(isolated_M_count, drones_linked_to_GS, mean_neighbors_per_drone)``.
    """
    graph, gs_label = build_graph_with_gs(env.drones, env.gs_position)
    reachable = nx.node_connected_component(graph, gs_label)

    isolated_m = sum(
        1 for d in env.drones
        if d.drone_type == "M" and d.drone_id not in reachable
    )
    in_gs_range = sum(
        1 for d in env.drones if are_connected(d.position, env.gs_position)
    )
    mean_neighbors = statistics.fmean(len(d.neighbors) for d in env.drones)
    return isolated_m, in_gs_range, mean_neighbors


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run_one(
    routing: str, load_ms: int, placement_seed: int, run_seed: int
) -> Tuple[Dict[str, Any], Dict[Any, Dict[str, int]]]:
    """Run one grid cell and return its CSV row plus its per-link counters.

    Args:
        routing:        "greedy" or "random".
        load_ms:        Milliseconds between packets at each M-drone.
        placement_seed: Seed for the layout.
        run_seed:       Seed for channel and traffic randomness.

    Returns:
        ``(row, link_stats_measured)`` — the row for 1a_runs.csv, and the
        simulator's post-warm-up per-link counters (used by the link matrix).
    """
    previous_interval = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(round(load_ms / 1000.0 / config.TIMESTEP))
    try:
        env = FANETEnv(
            routing=routing,
            log_path=os.devnull,
            placement_seed=placement_seed,
            run_seed=run_seed,
        )
        env.reset()
        isolated_m, in_gs_range, mean_neighbors = layout_facts(env)

        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        links = truth["links"]
        created = truth["created"]

        def share(count: int) -> float:
            """Return *count* as a share of the packets created."""
            return count / created if created else 0.0

        row: Dict[str, Any] = {
            "placement_seed": placement_seed,
            "run_seed": run_seed,
            "routing": routing,
            "load_ms": load_ms,
            "created": created,
            "delivered": truth["delivered"],
            "total_loss": truth["total_loss"] or 0.0,
            "loss_channel": share(truth["loss_by_reason"]["channel"]),
            "loss_queue_full": share(truth["loss_by_reason"]["queue_full"]),
            "loss_no_route": share(truth["loss_by_reason"]["no_route"]),
            "loss_ttl": share(truth["loss_by_reason"]["ttl"]),
            "loss_hop_limit": share(truth["loss_by_reason"]["hop_limit"]),
            "per_link_loss_pooled": links["pooled_loss"] or 0.0,
            "per_link_queue_full": links["pooled_loss_dropped_queue_full"] or 0.0,
            "per_link_channel": links["pooled_loss_lost_channel"] or 0.0,
            "per_link_expired": links["pooled_loss_expired_in_queue"] or 0.0,
            "mean_queue_occupancy": truth["mean_queue"] or 0.0,
            "max_queue_occupancy": truth["max_queue"] or 0,
            "mean_delay_steps": truth["mean_delay_steps"] or 0.0,
            "mean_hops": truth["mean_hops"] or 0.0,
            "isolated_M_count": isolated_m,
            "drones_in_GS_range": in_gs_range,
            "mean_neighbors_per_drone": mean_neighbors,
            "placement_draws": env.placement_draws,
            "views_agree": not metrics["discrepancies"],
        }
        # The five causes are shares of the same denominator, so they must add
        # up to the headline loss. They only would not if a measured packet
        # were still in flight at the end, which the drain cut prevents.
        row["_still_in_flight"] = truth["still_in_flight"]
        row["_discrepancies"] = metrics["discrepancies"]

        return row, {k: dict(v) for k, v in env.link_stats_measured.items()}
    finally:
        config.PACKET_INTERVAL_STEPS = previous_interval


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------

def run_grid(placements: int) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], Dict]]:
    """Run every cell of the grid.

    Args:
        placements: How many placement seeds to use (1..placements).

    Returns:
        ``(rows, link_stats_placement1)`` — every run's row, and the per-link
        counters for placement 1 keyed by ``(routing, load_ms)``.

    Raises:
        SystemExit: If any run's two views disagree.
    """
    rows: List[Dict[str, Any]] = []
    link_stats_p1: Dict[Tuple[str, int], Dict] = {}

    total = len(ROUTING_RULES) * len(LOADS_MS) * placements
    done = 0
    started = time.perf_counter()
    print(f"Running {total} runs ({len(ROUTING_RULES)} rules x "
          f"{len(LOADS_MS)} loads x {placements} placements) ...")

    for placement_seed in range(1, placements + 1):
        for load_ms in LOADS_MS:
            for routing in ROUTING_RULES:
                row, link_stats = run_one(
                    routing=routing,
                    load_ms=load_ms,
                    placement_seed=placement_seed,
                    run_seed=placement_seed,
                )

                if row["isolated_M_count"] != 0:
                    print("\nSTOP: a run had an M-drone with no path to the GS,"
                          " which the layout filter should have excluded.")
                    print(f"  run: routing={routing} load={load_ms}ms "
                          f"placement_seed={placement_seed}   "
                          f"isolated_M_count={row['isolated_M_count']}")
                    raise SystemExit(1)

                if not row["views_agree"]:
                    print("\nSTOP: the ACK-based view disagrees with ground truth.")
                    print(f"  run: routing={routing} load={load_ms}ms "
                          f"placement_seed={placement_seed}")
                    for problem in row["_discrepancies"]:
                        print(f"    ! {problem}")
                    raise SystemExit(1)

                if placement_seed == 1:
                    link_stats_p1[(routing, load_ms)] = link_stats

                rows.append(row)
                done += 1

        elapsed = time.perf_counter() - started
        print(f"  placement {placement_seed:>3}/{placements}   "
              f"{done:>4}/{total} runs   {elapsed:6.1f}s")

    return rows, link_stats_p1


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_runs_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write one row per run.

    Args:
        rows: Every run's row.
        path: Output CSV path.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in RUN_COLUMNS})


def summarise(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mean and std of every numeric column, per (routing, load).

    Args:
        rows: Every run's row.

    Returns:
        One record per (routing, load), with ``<column>_mean`` and
        ``<column>_std`` entries.
    """
    summary: List[Dict[str, Any]] = []
    for routing in ROUTING_RULES:
        for load_ms in LOADS_MS:
            cell = [
                r for r in rows
                if r["routing"] == routing and r["load_ms"] == load_ms
            ]
            record: Dict[str, Any] = {
                "routing": routing,
                "load_ms": load_ms,
                "runs": len(cell),
            }
            for column in NUMERIC_COLUMNS:
                if column in ("placement_seed", "run_seed", "load_ms"):
                    continue
                values = [float(r[column]) for r in cell]
                record[f"{column}_mean"] = statistics.fmean(values)
                record[f"{column}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            summary.append(record)
    return summary


def write_summary_csv(summary: List[Dict[str, Any]], path: str) -> None:
    """Write the per-cell mean/std summary.

    Args:
        summary: Output of :func:`summarise`.
        path:    Output CSV path.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def paired_comparison(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare greedy against random on identical seeds, per load.

    Args:
        rows: Every run's row.

    Returns:
        One record per load with the mean and std of
        ``greedy.total_loss - random.total_loss`` and how often greedy was
        lower.
    """
    records: List[Dict[str, Any]] = []
    for load_ms in LOADS_MS:
        by_seed: Dict[int, Dict[str, float]] = {}
        for row in rows:
            if row["load_ms"] != load_ms:
                continue
            by_seed.setdefault(row["placement_seed"], {})[row["routing"]] = (
                float(row["total_loss"])
            )

        diffs = [
            pair["greedy"] - pair["random"]
            for pair in by_seed.values()
            if "greedy" in pair and "random" in pair
        ]
        records.append({
            "load_ms": load_ms,
            "placements": len(diffs),
            "mean_greedy_minus_random": statistics.fmean(diffs) if diffs else 0.0,
            "std_greedy_minus_random": (
                statistics.stdev(diffs) if len(diffs) > 1 else 0.0
            ),
            "placements_greedy_lower": sum(1 for d in diffs if d < 0),
        })
    return records


def write_paired_csv(records: List[Dict[str, Any]], path: str) -> None:
    """Write the paired greedy-vs-random comparison.

    Args:
        records: Output of :func:`paired_comparison`.
        path:    Output CSV path.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


# ---------------------------------------------------------------------------
# Printed tables
# ---------------------------------------------------------------------------

def print_summary(summary: List[Dict[str, Any]]) -> None:
    """Print the per-cell summary as a table.

    Args:
        summary: Output of :func:`summarise`.
    """
    header = (
        f"{'rule':<8}{'load':>7}{'runs':>6}"
        f"{'total loss':>18}{'channel':>16}{'queue_full':>16}"
        f"{'no_route':>16}{'ttl+hop':>15}{'per-link':>16}"
        f"{'delay':>13}{'hops':>12}{'queue':>12}"
    )
    print()
    print("  Phase-1A summary — mean (std) over placements, in %")
    print("  " + "-" * (len(header) + 1))
    print("  " + header)
    for record in summary:
        ttl_hop_mean = record["loss_ttl_mean"] + record["loss_hop_limit_mean"]
        ttl_hop_std = record["loss_ttl_std"] + record["loss_hop_limit_std"]
        print(
            "  "
            f"{record['routing']:<8}{record['load_ms']:>5} ms{record['runs']:>6}"
            f"{_pm(record['total_loss_mean'], record['total_loss_std']):>18}"
            f"{_pm(record['loss_channel_mean'], record['loss_channel_std']):>16}"
            f"{_pm(record['loss_queue_full_mean'], record['loss_queue_full_std']):>16}"
            f"{_pm(record['loss_no_route_mean'], record['loss_no_route_std']):>16}"
            f"{_pm(ttl_hop_mean, ttl_hop_std):>15}"
            f"{_pm(record['per_link_loss_pooled_mean'], record['per_link_loss_pooled_std']):>16}"
            f"{record['mean_delay_steps_mean']:>9.2f} st"
            f"{record['mean_hops_mean']:>12.2f}"
            f"{record['mean_queue_occupancy_mean']:>12.2f}"
        )


#: Where the previous model's summary is kept, for the side-by-side table.
BASELINE_FILENAME = "1a_summary_OLD_k08.csv"

#: Columns compared between the old and new channel models.
COMPARISON_ROWS = [
    ("total_loss", "total loss"),
    ("loss_channel", "  channel"),
    ("loss_queue_full", "  queue_full"),
    ("loss_no_route", "  no_route"),
    ("loss_ttl", "  ttl"),
    ("loss_hop_limit", "  hop_limit"),
]


def load_baseline(path: str) -> Optional[List[Dict[str, Any]]]:
    """Load the previous model's summary, if it is there.

    Args:
        path: Path to the baseline summary CSV.

    Returns:
        The rows, or None when the file is missing.
    """
    if not os.path.exists(path):
        return None
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def print_old_vs_new(
    summary: List[Dict[str, Any]], baseline: Optional[List[Dict[str, Any]]]
) -> None:
    """Print the old channel model beside the new one.

    Args:
        summary:  This run's summary.
        baseline: The previous model's summary, or None.
    """
    print()
    if baseline is None:
        print(f"  (no {BASELINE_FILENAME} found — skipping the old-vs-new table)")
        return

    print("  OLD channel model (FSPL, exp(-0.8*margin), 250 m wall)"
          "  vs  NEW (logistic in distance)")
    print(f"  Mean % of packets created, over the same "
          f"{summary[0]['runs']} placements.")
    print("  " + "-" * 92)
    header = f"  {'':<14}"
    for load in LOADS_MS:
        header += f"{str(load) + ' ms':>19}"
    print(header + "\n  " + f"{'':<14}" + "".join(
        f"{'old':>9}{'new':>10}" for _ in LOADS_MS
    ))
    print("  " + "-" * 92)

    for routing in ROUTING_RULES:
        print(f"  {routing}")
        for key, label in COMPARISON_ROWS:
            line = f"  {label:<14}"
            for load in LOADS_MS:
                new_cell = next(
                    r for r in summary
                    if r["routing"] == routing and r["load_ms"] == load
                )
                old_cell = next(
                    (r for r in baseline
                     if r["routing"] == routing and int(r["load_ms"]) == load),
                    None,
                )
                old_value = (
                    float(old_cell[f"{key}_mean"]) * 100
                    if old_cell is not None else float("nan")
                )
                line += f"{old_value:>9.1f}{new_cell[f'{key}_mean'] * 100:>10.1f}"
            print(line)
        print("  " + "-" * 92)


def _pm(mean: float, std: float) -> str:
    """Format a mean and std as percentages, e.g. "64.5 (3.2)"."""
    return f"{mean * 100:.1f} ({std * 100:.1f})"


def print_paired(records: List[Dict[str, Any]]) -> None:
    """Print the paired comparison as a table.

    Args:
        records: Output of :func:`paired_comparison`.
    """
    print()
    print("  Paired greedy - random on identical seeds (negative = greedy loses less)")
    print("  " + "-" * 76)
    print(f"  {'load':>8}{'placements':>13}{'mean diff':>14}{'std':>12}"
          f"{'greedy lower':>16}")
    for record in records:
        print(
            f"  {record['load_ms']:>5} ms{record['placements']:>13}"
            f"{record['mean_greedy_minus_random'] * 100:>13.2f}%"
            f"{record['std_greedy_minus_random'] * 100:>11.2f}%"
            f"{record['placements_greedy_lower']:>10} / {record['placements']}"
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes) -> None:
    """Apply the shared recessive axis styling."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_total_loss_vs_load(
    summary: List[Dict[str, Any]], path: str
) -> None:
    """Draw total packet loss against offered load, one line per rule.

    Args:
        summary: Output of :func:`summarise`.
        path:    Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Heaviest load on the right: 1000 ms -> 100 ms.
    x_positions = list(range(len(LOADS_MS)))

    # Label each line above or below its last point depending on which line
    # ends higher, so the two labels never sit on top of each other.
    finals = {
        routing: next(
            r for r in summary
            if r["routing"] == routing and r["load_ms"] == LOADS_MS[-1]
        )["total_loss_mean"]
        for routing in ROUTING_RULES
    }
    topmost = max(finals, key=lambda k: finals[k])

    for routing in ROUTING_RULES:
        cells = [
            next(r for r in summary
                 if r["routing"] == routing and r["load_ms"] == load)
            for load in LOADS_MS
        ]
        means = [c["total_loss_mean"] * 100 for c in cells]
        stds = [c["total_loss_std"] * 100 for c in cells]
        ax.errorbar(
            x_positions, means, yerr=stds,
            color=SERIES_COLORS[routing], linewidth=2.2,
            marker="o", markersize=8, capsize=4, elinewidth=1.2,
            label=routing, zorder=3,
        )
        ax.annotate(
            routing,
            xy=(x_positions[-1], means[-1]),
            xytext=(-6, 12 if routing == topmost else -20),
            textcoords="offset points",
            color=SERIES_COLORS[routing],
            fontsize=10, fontweight="bold", ha="right", zorder=5,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{load} ms" for load in LOADS_MS])
    ax.set_xlabel("Load — one packet per M-drone every …  (heavier to the right)",
                  color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("Total packet loss (%)", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Total packet loss vs offered load",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=14, loc="left",
    )
    _style_axes(ax)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=10,
                       labelcolor=TEXT_SECONDARY)
    legend.set_zorder(6)

    n_placements = summary[0]["runs"]
    fig.text(0.01, 0.015,
             f"mean +/- std over {n_placements} placements; "
             f"{config.MAX_STEPS} steps; window "
             f"[{config.measurement_window()[0]}, {config.measurement_window()[1]})",
             color=TEXT_SECONDARY, fontsize=8)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


#: Panels measured as a share of PACKETS CREATED. These share one colour bar.
HEATMAP_COLUMNS = [
    ("total_loss", "total\nloss"),
    ("loss_channel", "channel"),
    ("loss_queue_full", "queue\nfull"),
    ("loss_no_route", "no\nroute"),
    ("_ttl_hop", "ttl +\nhop"),
]

#: Per-link loss has a different denominator — link ATTEMPTS, not packets
#: created — so it gets its own panel and its own colour bar rather than
#: sharing a scale with numbers it is not comparable to.
LINK_COLUMN = ("per_link_loss_pooled", "per-link\nloss")


def _cell_value(cell: Dict[str, Any], key: str) -> Tuple[float, float]:
    """Return one panel cell's (mean, std) as percentages.

    Args:
        cell: A summary record.
        key:  A column key, or the synthetic "_ttl_hop".

    Returns:
        ``(mean_pct, std_pct)``.
    """
    if key == "_ttl_hop":
        return (
            (cell["loss_ttl_mean"] + cell["loss_hop_limit_mean"]) * 100,
            (cell["loss_ttl_std"] + cell["loss_hop_limit_std"]) * 100,
        )
    return cell[f"{key}_mean"] * 100, cell[f"{key}_std"] * 100


def _draw_panel(
    ax: plt.Axes,
    means: np.ndarray,
    stds: np.ndarray,
    labels: List[str],
    title: str,
    show_loads: bool = True,
) -> Any:
    """Draw one heatmap panel with mean and std in each cell.

    Args:
        ax:         Axes to draw on.
        means:      loads x columns of mean percentages.
        stds:       Matching standard deviations.
        labels:     Column labels.
        title:      Panel title.
        show_loads: Whether to label the rows. Off for panels that sit to the
                    right of a colour bar, where row labels would collide with
                    it; the rows are in the same order in every panel.

    Returns:
        The image handle, for attaching a colour bar.
    """
    image = ax.imshow(means, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(means.shape[0]))
    if show_loads:
        ax.set_yticklabels([f"{load} ms" for load in LOADS_MS], fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11,
                 fontweight="bold", pad=10)

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            # Keep the label readable against a dark cell.
            ink = "#ffffff" if means[i, j] > 55 else TEXT_PRIMARY
            ax.text(j, i - 0.10, f"{means[i, j]:.1f}", ha="center",
                    va="center", color=ink, fontsize=10, fontweight="bold")
            ax.text(j, i + 0.22, f"({stds[i, j]:.1f})", ha="center",
                    va="center", color=ink, fontsize=7)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_loss_heatmaps(summary: List[Dict[str, Any]], path: str) -> None:
    """Draw the loss heatmaps: causes per rule, then per-link loss apart.

    The cause panels are shares of packets created and share one colour bar.
    Per-link loss counts link ATTEMPTS instead, so it sits in its own panel
    with its own bar rather than being read against a scale it does not belong
    to.

    Args:
        summary: Output of :func:`summarise`.
        path:    Output PNG path.
    """
    fig = plt.figure(figsize=(16.0, 5.4), dpi=150)
    grid = fig.add_gridspec(
        1, 5, width_ratios=[len(HEATMAP_COLUMNS), len(HEATMAP_COLUMNS),
                            0.20, 1.3, 0.20],
        wspace=0.55,
    )
    fig.patch.set_facecolor(SURFACE)

    cause_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    cause_bar_ax = fig.add_subplot(grid[0, 2])
    link_ax = fig.add_subplot(grid[0, 3])
    link_bar_ax = fig.add_subplot(grid[0, 4])

    cause_image = None
    for ax, routing in zip(cause_axes, ROUTING_RULES):
        means = np.zeros((len(LOADS_MS), len(HEATMAP_COLUMNS)))
        stds = np.zeros_like(means)
        for i, load in enumerate(LOADS_MS):
            cell = next(r for r in summary
                        if r["routing"] == routing and r["load_ms"] == load)
            for j, (key, _) in enumerate(HEATMAP_COLUMNS):
                means[i, j], stds[i, j] = _cell_value(cell, key)
        cause_image = _draw_panel(
            ax, means, stds, [label for _, label in HEATMAP_COLUMNS], routing
        )

    bar = fig.colorbar(cause_image, cax=cause_bar_ax)
    bar.set_label("% of packets created", color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)

    # Per-link loss: one column per rule, its own denominator, its own bar.
    link_means = np.zeros((len(LOADS_MS), len(ROUTING_RULES)))
    link_stds = np.zeros_like(link_means)
    for i, load in enumerate(LOADS_MS):
        for j, routing in enumerate(ROUTING_RULES):
            cell = next(r for r in summary
                        if r["routing"] == routing and r["load_ms"] == load)
            link_means[i, j], link_stds[i, j] = _cell_value(
                cell, LINK_COLUMN[0]
            )
    link_image = _draw_panel(
        link_ax, link_means, link_stds, list(ROUTING_RULES), "per-link loss",
        show_loads=False,
    )

    link_bar = fig.colorbar(link_image, cax=link_bar_ax)
    link_bar.set_label("% of link attempts", color=TEXT_SECONDARY, fontsize=9)
    link_bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    link_bar.outline.set_visible(False)

    n_placements = summary[0]["runs"]
    fig.suptitle(
        "Loss and its causes, by load and routing rule  —  mean % (std)",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.text(0.01, 0.015,
             f"mean over {n_placements} placements; identical seeds for both "
             f"rules. Left panels: share of packets created. Right panel: "
             f"share of link attempts, pooled over all links — a different "
             f"denominator, hence its own scale.",
             color=TEXT_SECONDARY, fontsize=8)

    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_link_matrices(
    link_stats: Dict[Tuple[str, int], Dict],
    labels: List[str],
    node_index: Dict[Any, int],
    path: str,
) -> None:
    """Draw the per-link loss matrix for placement 1, for every rule and load.

    Args:
        link_stats: Per-link counters keyed by ``(routing, load_ms)``.
        labels:     Axis labels, e.g. ``["M0", ..., "C24", "GS"]``.
        node_index: Node key → row/column index.
        path:       Output PNG path.
    """
    n_rows = len(ROUTING_RULES)
    n_cols = len(LOADS_MS)
    size = len(labels)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 4.2 * n_rows), dpi=150
    )
    fig.patch.set_facecolor(SURFACE)

    colormap = plt.get_cmap("Blues").copy()
    colormap.set_bad(UNUSED_GRAY)   # gray = link never used

    image = None
    for row_i, routing in enumerate(ROUTING_RULES):
        for col_i, load in enumerate(LOADS_MS):
            ax = axes[row_i][col_i]
            matrix = np.full((size, size), np.nan)

            for (sender, receiver), counters in link_stats[(routing, load)].items():
                offered = counters["offered"]
                if offered == 0:
                    continue
                lost = (counters["dropped_queue_full"]
                        + counters["lost_channel"]
                        + counters["expired_in_queue"])
                matrix[node_index[sender], node_index[receiver]] = (
                    100.0 * lost / offered
                )

            image = ax.imshow(
                np.ma.masked_invalid(matrix),
                cmap=colormap, vmin=0, vmax=100, aspect="equal",
            )
            ax.set_title(f"{routing} — {load} ms", color=TEXT_PRIMARY,
                         fontsize=10, fontweight="bold", pad=6)

            step = max(1, size // 13)
            ticks = list(range(0, size, step))
            if ticks[-1] != size - 1:
                ticks.append(size - 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([labels[t] for t in ticks], fontsize=6, rotation=90)
            ax.set_yticks(ticks)
            ax.set_yticklabels([labels[t] for t in ticks], fontsize=6)
            ax.tick_params(colors=TEXT_SECONDARY, length=2)
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_i == 0:
                ax.set_ylabel("sender", color=TEXT_SECONDARY, fontsize=9)
            if row_i == n_rows - 1:
                ax.set_xlabel("receiver", color=TEXT_SECONDARY, fontsize=9)

    bar = fig.colorbar(image, ax=axes, fraction=0.015, pad=0.02)
    bar.set_label("per-link loss (%)", color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)

    fig.suptitle(
        "Per-link loss, placement_seed 1  —  gray = link not used",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def placement_labels(placement_seed: int) -> Tuple[List[str], Dict[Any, int]]:
    """Build axis labels and index lookup for one placement's drones plus GS.

    Args:
        placement_seed: Which layout to describe.

    Returns:
        ``(labels, node_index)`` where labels look like "M3" / "C20" / "GS".
    """
    env = FANETEnv(routing="greedy", log_path=os.devnull,
                   placement_seed=placement_seed, run_seed=placement_seed)
    env.reset()
    env.close_logger()

    labels = [f"{d.drone_type}{d.drone_id}" for d in env.drones]
    node_index: Dict[Any, int] = {d.drone_id: i for i, d in enumerate(env.drones)}
    labels.append("GS")
    node_index["GS"] = len(labels) - 1
    return labels, node_index


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the grid, write every output, and print the two tables."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows, link_stats_p1 = run_grid(args.placements)

    stranded = sum(r["_still_in_flight"] for r in rows)
    if stranded:
        print(f"\nNOTE: {stranded} measured packets were still in flight at the "
              "end of a run, so the causes do not sum exactly to total loss.")

    runs_path = os.path.join(args.out_dir, "1a_runs.csv")
    summary_path = os.path.join(args.out_dir, "1a_summary.csv")
    paired_path = os.path.join(args.out_dir, "1a_paired.csv")

    write_runs_csv(rows, runs_path)
    summary = summarise(rows)
    write_summary_csv(summary, summary_path)
    paired = paired_comparison(rows)
    write_paired_csv(paired, paired_path)

    print_summary(summary)
    print_paired(paired)
    print_old_vs_new(
        summary, load_baseline(os.path.join(args.out_dir, BASELINE_FILENAME))
    )

    line_path = os.path.join(args.out_dir, "1a_total_loss_vs_load.png")
    heat_path = os.path.join(args.out_dir, "1a_loss_heatmaps.png")
    matrix_path = os.path.join(args.out_dir, "1a_link_matrix_placement1.png")

    plot_total_loss_vs_load(summary, line_path)
    plot_loss_heatmaps(summary, heat_path)
    labels, node_index = placement_labels(1)
    plot_link_matrices(link_stats_p1, labels, node_index, matrix_path)

    draws = [row["placement_draws"] for row in rows]
    print()
    print(f"  mean neighbours per drone: "
          f"{statistics.fmean(r['mean_neighbors_per_drone'] for r in rows):.2f}")
    print(f"  drones linked directly to the GS: "
          f"{statistics.fmean(r['drones_in_GS_range'] for r in rows):.2f}"
          f" of {config.NUM_M_DRONES + config.NUM_C_DRONES}")
    print(f"  placement draws to find a connected layout: "
          f"mean {statistics.fmean(draws):.2f}, max {max(draws)}")
    print(f"  isolated M-drones: 0 in all {len(rows)} runs")
    print(f"  views agreed in all {len(rows)} runs")
    for path in (runs_path, summary_path, paired_path,
                 line_path, heat_path, matrix_path):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
