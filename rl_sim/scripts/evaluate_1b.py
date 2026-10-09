"""
evaluate_1b.py — Measure Phase-1b on the held-back TEST layouts.

The TEST layouts are placement seeds 1-20, the same ones Phase 1a reports.
They were never trained on and never used to choose a model: the best
checkpoint for each init seed was picked on the VALIDATION layouts alone.
This script uses them once.

Measured here:
    random, greedy, rate     one run per layout
    rl                       the best-validation model of each init seed,
                             reported as the mean over those seeds

At the 200 ms working load, and again at 500 ms — a load nothing was trained
on, so that column is a generalisation check.

Outputs, all stamped:
    out/1b_comparison_total_loss.png  total loss per rule, every layout a dot
    out/1b_loss_by_cause.png          where the loss goes, at 200 ms
    out/1b_paired.csv                 rl - greedy and rl - rate on the same
                                      seeds, per load
    out/1b_test_runs.csv              one row per (rule, load, layout)

views_agree must hold in every run; the script stops if it ever does not.

Usage:
    python scripts/evaluate_1b.py
    python scripts/evaluate_1b.py --init-seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fanet_sim import config
from scripts.stamp import csv_comment, stamp_lines
from scripts.train_1b import TEST_SEEDS, run_once

LOADS_MS = (200, 500)
MAIN_LOAD_MS = 200
RULES = ("random", "greedy", "rate", "rl")
CAUSES = ("channel", "queue_full", "no_route", "dead_end", "ttl_hop")

RULE_COLORS = {
    "random": "#e34948",
    "greedy": "#eb6834",
    "rate": "#4a3aa7",
    "rl": "#2a78d6",
}
CAUSE_COLORS = {
    "channel": "#2a78d6",
    "queue_full": "#eb6834",
    "no_route": "#1baf7a",
    "dead_end": "#e87ba4",
    "ttl_hop": "#eda100",
}
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
SURFACE = "#fcfcfb"

RUN_COLUMNS = ["rule", "load_ms", "placement_seed", "init_seed", "total_loss",
               *CAUSES, "views_agree"]
PAIRED_COLUMNS = ["load_ms", "comparison", "layouts", "mean_difference",
                  "std_difference", "layouts_rl_lower"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate Phase-1b on TEST layouts.")
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Which trained models to evaluate (default 0 1 2).")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the models live and the outputs go.")
    parser.add_argument("--suffix", type=str, default="",
                        help="Appended to every output filename, so runs under "
                             "different loop guards do not overwrite each other.")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    return parser.parse_args()


def _style(ax: plt.Axes) -> None:
    """Apply the shared recessive axis styling."""
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


# ---------------------------------------------------------------------------
# Running the grid
# ---------------------------------------------------------------------------

def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one test run.

    Args:
        args: ``(rule, load_ms, seed, init_seed, state_dict)``.

    Returns:
        That run's row.
    """
    rule, load_ms, seed, init_seed, state = args
    result = run_once(rule, seed, seed, load_ms, state_dict=state, training=False)
    return {
        "rule": rule, "load_ms": load_ms, "placement_seed": seed,
        "init_seed": init_seed if rule == "rl" else -1,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def run_grid(
    init_seeds: Sequence[int], out_dir: str, workers: int
) -> List[Dict[str, Any]]:
    """Run every rule on every TEST layout, at both loads.

    Args:
        init_seeds: Which trained models to evaluate.
        out_dir:    Where the saved models live.
        workers:    Parallel workers.

    Returns:
        One row per run.

    Raises:
        SystemExit: If a model is missing, or any run's views disagree.
    """
    from concurrent.futures import ProcessPoolExecutor

    import torch

    states: Dict[int, Dict[str, Any]] = {}
    for init_seed in init_seeds:
        path = os.path.join(out_dir, "models", config.LOOP_GUARD,
                            f"rl_seed{init_seed}.pt")
        if not os.path.exists(path):
            print(f"STOP: no trained model at {path}. Run train_1b.py first.")
            raise SystemExit(1)
        states[init_seed] = torch.load(path, weights_only=True)

    jobs: List[Tuple] = []
    for load_ms in LOADS_MS:
        for seed in TEST_SEEDS:
            for rule in RULES:
                if rule == "rl":
                    for init_seed in init_seeds:
                        jobs.append((rule, load_ms, seed, init_seed,
                                     states[init_seed]))
                else:
                    jobs.append((rule, load_ms, seed, -1, None))

    print(f"Running {len(jobs)} test runs "
          f"({len(RULES)} rules x {len(LOADS_MS)} loads x "
          f"{len(TEST_SEEDS)} layouts, rl x {len(init_seeds)} init seeds) ...")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [row for row in rows if not row["views_agree"]]
    if bad:
        print("\nSTOP: the ACK view disagreed with ground truth.")
        for row in bad[:5]:
            print(f"  {row['rule']} load {row['load_ms']} "
                  f"seed {row['placement_seed']}: {row['discrepancies']}")
        raise SystemExit(1)

    print(f"  views agreed in all {len(rows)} runs")
    return rows


def per_layout(
    rows: Sequence[Dict[str, Any]], rule: str, load_ms: int, key: str = "total_loss"
) -> List[float]:
    """Return one value per TEST layout, averaging rl over its init seeds.

    Args:
        rows:    Every run's row.
        rule:    Which rule.
        load_ms: Which load.
        key:     Which measured value.

    Returns:
        One number per layout, in TEST_SEEDS order.
    """
    values: List[float] = []
    for seed in TEST_SEEDS:
        matching = [r[key] for r in rows
                    if r["rule"] == rule and r["load_ms"] == load_ms
                    and r["placement_seed"] == seed]
        values.append(statistics.fmean(matching))
    return values


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_comparison(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Draw total loss per rule, with every layout shown as a dot.

    Args:
        rows: Every run's row.
        path: Output PNG path.
    """
    fig, axes = plt.subplots(1, len(LOADS_MS), figsize=(12.0, 5.6), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, load_ms in zip(axes, LOADS_MS):
        ax.set_facecolor(SURFACE)
        # Percentages everywhere: the boxes and the dots must share units.
        series = [[value * 100 for value in per_layout(rows, rule, load_ms)]
                  for rule in RULES]

        parts = ax.boxplot(series, widths=0.55, patch_artist=True,
                           medianprops={"color": TEXT_PRIMARY, "linewidth": 1.6},
                           whiskerprops={"color": TEXT_SECONDARY},
                           capprops={"color": TEXT_SECONDARY},
                           flierprops={"marker": ""}, zorder=2)
        for patch, rule in zip(parts["boxes"], RULES):
            patch.set_facecolor(RULE_COLORS[rule])
            patch.set_alpha(0.25)
            patch.set_edgecolor(RULE_COLORS[rule])

        rng = np.random.default_rng(0)       # fixed jitter, so the plot replays
        for index, (rule, values) in enumerate(zip(RULES, series), start=1):
            jitter = rng.uniform(-0.16, 0.16, size=len(values))
            ax.scatter(index + jitter, values, s=22,
                       color=RULE_COLORS[rule], alpha=0.75, zorder=3,
                       edgecolors="none")

        # The mean goes under the tick label, where nothing can collide with it.
        ax.set_xticks(range(1, len(RULES) + 1))
        ax.set_xticklabels([
            f"{rule}\n{statistics.fmean(values):.1f}%"
            for rule, values in zip(RULES, series)
        ])
        heading = f"{load_ms} ms"
        if load_ms != MAIN_LOAD_MS:
            heading += "  (never trained on)"
        ax.set_title(heading, color=TEXT_PRIMARY, fontsize=12,
                     fontweight="bold", pad=10)
        _style(ax)

    axes[0].set_ylabel("Total packet loss (%)", color=TEXT_SECONDARY, fontsize=11)
    axes[0].set_ylim(0, 100)

    fig.suptitle(
        f"Phase-1b on the {len(TEST_SEEDS)} held-back TEST layouts — "
        "one dot per layout, label = mean",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.text(0.01, 0.012, stamp_lines(width=140, loads_ms=list(LOADS_MS),
                                      placements=len(TEST_SEEDS)),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_loss_by_cause(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Draw stacked bars of where the loss goes at the main load.

    Args:
        rows: Every run's row.
        path: Output PNG path.
    """
    rules = ("greedy", "rate", "rl")

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bottoms = np.zeros(len(rules))
    for cause in CAUSES:
        heights = np.array([
            statistics.fmean(per_layout(rows, rule, MAIN_LOAD_MS, cause)) * 100
            for rule in rules
        ])
        ax.bar(rules, heights, bottom=bottoms, width=0.55,
               color=CAUSE_COLORS[cause], label=cause, zorder=2)
        for index, (height, base) in enumerate(zip(heights, bottoms)):
            if height >= 1.5:          # only label a slice big enough to read
                ax.text(index, base + height / 2, f"{height:.1f}",
                        ha="center", va="center", color="#ffffff",
                        fontsize=9, fontweight="bold", zorder=4)
        bottoms += heights

    for index, total in enumerate(bottoms):
        ax.annotate(f"{total:.1f}%", xy=(index, total), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=TEXT_PRIMARY, fontsize=10, fontweight="bold")

    ax.set_ylabel("% of packets created", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylim(0, max(bottoms) * 1.18)
    ax.set_title(
        f"Where the loss goes at {MAIN_LOAD_MS} ms — TEST layouts, mean of "
        f"{len(TEST_SEEDS)}",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", pad=12, loc="left",
    )
    _style(ax)
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.012, stamp_lines(width=112, loads_ms=[MAIN_LOAD_MS],
                                      placements=len(TEST_SEEDS)),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The paired table
# ---------------------------------------------------------------------------

def paired_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare rl against greedy and rate on the same layouts.

    Args:
        rows: Every run's row.

    Returns:
        One record per (load, comparison).
    """
    records: List[Dict[str, Any]] = []
    for load_ms in LOADS_MS:
        rl = per_layout(rows, "rl", load_ms)
        for other in ("greedy", "rate"):
            baseline = per_layout(rows, other, load_ms)
            differences = [a - b for a, b in zip(rl, baseline)]
            records.append({
                "load_ms": load_ms,
                "comparison": f"rl - {other}",
                "layouts": len(differences),
                "mean_difference": statistics.fmean(differences),
                "std_difference": (statistics.stdev(differences)
                                   if len(differences) > 1 else 0.0),
                "layouts_rl_lower": sum(1 for d in differences if d < 0),
            })
    return records


def print_tables(
    rows: Sequence[Dict[str, Any]], paired: Sequence[Dict[str, Any]]
) -> None:
    """Print the test results and the paired comparison.

    Args:
        rows:   Every run's row.
        paired: Output of :func:`paired_table`.
    """
    print()
    print(f"  TEST layouts (seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}), "
          "mean % of packets created")
    print("  " + "-" * 84)
    print(f"  {'load':>8}{'rule':>9}{'total':>9}{'channel':>10}"
          f"{'queue_full':>12}{'no_route':>10}{'dead_end':>11}"
          f"{'ttl+hop':>10}{'std':>9}")
    print("  " + "-" * 84)
    for load_ms in LOADS_MS:
        for rule in RULES:
            totals = per_layout(rows, rule, load_ms)
            causes = [statistics.fmean(per_layout(rows, rule, load_ms, c))
                      for c in CAUSES]
            print(f"  {load_ms:>5} ms{rule:>9}"
                  f"{statistics.fmean(totals)*100:>8.2f}%"
                  f"{causes[0]*100:>9.2f}%{causes[1]*100:>11.2f}%"
                  f"{causes[2]*100:>9.2f}%{causes[3]*100:>10.2f}%"
                  f"{causes[4]*100:>9.2f}%"
                  f"{statistics.stdev(totals)*100:>8.2f}")
        print("  " + "-" * 84)

    print()
    print("  Paired on identical layouts (negative = rl loses less)")
    print("  " + "-" * 72)
    print(f"  {'load':>8}{'comparison':>14}{'layouts':>10}{'mean diff':>13}"
          f"{'std':>10}{'rl lower':>13}")
    for record in paired:
        print(f"  {record['load_ms']:>5} ms{record['comparison']:>14}"
              f"{record['layouts']:>10}"
              f"{record['mean_difference']*100:>12.2f}%"
              f"{record['std_difference']*100:>9.2f}%"
              f"{record['layouts_rl_lower']:>8} / {record['layouts']}")
    print("  " + "-" * 72)


def main() -> None:
    """Run the test grid, write every output and print the tables."""
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows = run_grid(args.init_seeds, args.out_dir, args.workers)
    paired = paired_table(rows)

    runs_path = os.path.join(args.out_dir, f"1b_test_runs{args.suffix}.csv")
    with open(runs_path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=list(LOADS_MS),
                                 placements=len(TEST_SEEDS),
                                 extra="phase 1b TEST layouts"))
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    paired_path = os.path.join(args.out_dir, f"1b_paired{args.suffix}.csv")
    with open(paired_path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=list(LOADS_MS),
                                 placements=len(TEST_SEEDS),
                                 extra="phase 1b paired comparison"))
        writer = csv.DictWriter(handle, fieldnames=PAIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(paired)

    comparison_path = os.path.join(args.out_dir, f"1b_comparison_total_loss{args.suffix}.png")
    causes_path = os.path.join(args.out_dir, f"1b_loss_by_cause{args.suffix}.png")
    plot_comparison(rows, comparison_path)
    plot_loss_by_cause(rows, causes_path)

    print_tables(rows, paired)
    print()
    for path in (runs_path, paired_path, comparison_path, causes_path):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
