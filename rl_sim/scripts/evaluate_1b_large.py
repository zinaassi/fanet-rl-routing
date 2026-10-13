"""
evaluate_1b_large.py — Phase 1b on 50 held-back TEST layouts (CHECKPOINT 13).

The CHECKPOINT-12 evaluation used 20 TEST layouts, which left the paired
rl-against-rate comparison close to the layout-to-layout spread. This runs the
same frozen models on 50 layouts — the original 1-20 plus 30 more — and adds
two significance tests to each paired comparison:

  sign test     exact two-sided binomial on the number of layouts won,
                assuming no difference makes either sign equally likely.
                It uses only the SIGN of each difference, so one freak layout
                cannot carry it.
  paired t-test two-sided, on the per-layout differences. It uses their sizes,
                which is sharper when they are roughly normal and less
                trustworthy when they are not.

Nothing is trained, and no model changes: these are the CHECKPOINT-12 "path"
models, the best-validation checkpoint of each init seed, reported as the
3-seed mean. LOOP_GUARD must be "path".

Writes:
    out/1b_test50_runs.csv              one row per run
    out/1b_test50_paired.csv            one row per comparison per layout set
    out/1b_comparison_total_loss_50.png the comparison figure, 50 layouts
    out/1b_loss_heatmaps_50.png         every rule, every cause, both loads

``--from-csv`` redraws the tables and figures from a previous run's CSV
instead of running the 600 runs again.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fanet_sim import config
from scripts.stamp import csv_comment, stamp_lines
from scripts.train_1b import run_once

TEST50_SEEDS = tuple(range(1, 51))
ORIGINAL_SEEDS = tuple(range(1, 21))      # the CHECKPOINT-12 set
ADDED_SEEDS = tuple(range(21, 51))        # the 30 new ones

LOADS_MS = (200, 500)
MAIN_LOAD_MS = 200
RULES = ("random", "greedy", "rate", "rl")
#: Only "rl" needs a trained network; the others run on their own, so a
#: subset without it can be measured with no model present.
MODEL_RULES = ("rl",)
CAUSES = ("channel", "queue_full", "no_route", "dead_end", "ttl", "hop_limit")
COMPARISONS = (("rl", "rate"), ("rl", "greedy"), ("rate", "greedy"))

SURFACE = "#faf9f5"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
RULE_COLORS = {
    "random": "#e0534e",
    "greedy": "#eb6834",
    "rate": "#5a4fcf",
    "rl": "#2a78d6",
}


# ---------------------------------------------------------------------------
# Running the grid
# ---------------------------------------------------------------------------

def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one run.

    Args:
        args: ``(rule, load_ms, seed, init_seed, state_dict)``.

    Returns:
        That run's row.
    """
    rule, load_ms, seed, init_seed, state = args
    result = run_once(rule, seed, seed, load_ms, state_dict=state,
                      training=False)
    return {
        "rule": rule, "load_ms": load_ms, "placement_seed": seed,
        "init_seed": init_seed if rule == "rl" else -1,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        "mean_hops": result["mean_hops"],
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def run_grid(out_dir: str, init_seeds: Sequence[int], workers: int,
             rules: Sequence[str] = RULES) -> List[Dict[str, Any]]:
    """Run the chosen rules on every layout, at both loads.

    Args:
        out_dir:    Where the trained models live.
        init_seeds: Which trained models to evaluate.
        workers:    Parallel workers.
        rules:      Which rules to run. Models are loaded ONLY when a rule
                    needing one is among them, so greedy, random and rate
                    can be measured with no trained network present.

    Returns:
        One row per run.

    Raises:
        SystemExit: If a needed model is missing, or any run's views disagree.
    """
    states: Dict[int, Dict[str, Any]] = {}
    if any(rule in MODEL_RULES for rule in rules):
        import torch

        for init_seed in init_seeds:
            path = os.path.join(out_dir, "models", config.LOOP_GUARD,
                                f"rl_seed{init_seed}.pt")
            if not os.path.exists(path):
                print(f"STOP: no trained model at {path}.")
                raise SystemExit(1)
            states[init_seed] = torch.load(path, weights_only=True)

    jobs: List[Tuple] = []
    for load_ms in LOADS_MS:
        for seed in TEST50_SEEDS:
            for rule in rules:
                if rule in MODEL_RULES:
                    for init_seed in init_seeds:
                        jobs.append((rule, load_ms, seed, init_seed,
                                     states[init_seed]))
                else:
                    jobs.append((rule, load_ms, seed, -1, None))

    print(f"Running {len(jobs)} runs ({len(rules)} rules x {len(LOADS_MS)} "
          f"loads x {len(TEST50_SEEDS)} layouts, rl x {len(init_seeds)} "
          f"init seeds) ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [r for r in rows if not r["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — {row['rule']} "
                  f"seed {row['placement_seed']}: {row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def per_layout(rows: Sequence[Dict[str, Any]], rule: str, load_ms: int,
               seeds: Sequence[int] = TEST50_SEEDS,
               key: str = "total_loss") -> List[float]:
    """Return one value per layout, averaging rl over its init seeds.

    Args:
        rows:    Every run's row.
        rule:    Which rule.
        load_ms: Which load.
        seeds:   Which layouts, in order.
        key:     Which measured value.

    Returns:
        One number per layout, in *seeds* order.
    """
    values: List[float] = []
    for seed in seeds:
        matching = [r[key] for r in rows
                    if r["rule"] == rule and r["load_ms"] == load_ms
                    and r["placement_seed"] == seed and r[key] is not None]
        values.append(statistics.fmean(matching))
    return values


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def sign_test(diffs: Sequence[float]) -> Tuple[int, int, float]:
    """Exact two-sided sign test on paired differences.

    Ties contribute to neither side and are dropped, which is the standard
    treatment and makes the test conservative.

    Args:
        diffs: Per-layout differences.

    Returns:
        ``(wins, used, p)`` — wins is how many differences were negative
        (the first rule lost fewer packets), used is how many were not ties.
    """
    from scipy import stats

    wins = sum(1 for d in diffs if d < 0)
    losses = sum(1 for d in diffs if d > 0)
    used = wins + losses
    if used == 0:
        return 0, 0, 1.0
    p = float(stats.binomtest(wins, used, 0.5, alternative="two-sided").pvalue)
    return wins, used, p


def paired_t_test(diffs: Sequence[float]) -> float:
    """Two-sided paired t-test p-value on the differences.

    Args:
        diffs: Per-layout differences.

    Returns:
        The p-value, or 1.0 if it is not defined (fewer than two values, or
        no spread at all).
    """
    from scipy import stats

    if len(diffs) < 2 or statistics.pstdev(diffs) == 0:
        return 1.0
    return float(stats.ttest_1samp(np.asarray(diffs, dtype=float), 0.0).pvalue)


def paired_rows(rows: Sequence[Dict[str, Any]], seeds: Sequence[int],
                label: str,
                rules: Sequence[str] = RULES) -> List[Dict[str, Any]]:
    """Build the paired comparison rows for one set of layouts.

    Args:
        rows:  Every run's row.
        seeds: The layouts to compare on.
        label: A name for this set, carried into the CSV.
        rules: Which rules were run; a comparison needing a rule that was
               not run is skipped rather than compared against nothing.

    Returns:
        One row per (load, comparison).
    """
    comparisons = [(a, b) for a, b in COMPARISONS
                   if a in rules and b in rules]
    out: List[Dict[str, Any]] = []
    for load_ms in LOADS_MS:
        for first, second in comparisons:
            a = per_layout(rows, first, load_ms, seeds)
            b = per_layout(rows, second, load_ms, seeds)
            diffs = [x - y for x, y in zip(a, b)]
            wins, used, p_sign = sign_test(diffs)
            out.append({
                "layouts": label, "n": len(seeds), "load_ms": load_ms,
                "comparison": f"{first} - {second}",
                "mean_diff": statistics.fmean(diffs),
                "std_diff": statistics.stdev(diffs),
                "wins": wins, "ties": len(diffs) - used,
                "p_sign": p_sign, "p_ttest": paired_t_test(diffs),
            })
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_summary(rows: Sequence[Dict[str, Any]],
                  rules: Sequence[str] = RULES) -> None:
    """Print table (a): every rule at every load, all six causes."""
    width = 106
    print()
    print(f"  (a) {len(TEST50_SEEDS)} TEST layouts (seeds "
          f"{TEST50_SEEDS[0]}-{TEST50_SEEDS[-1]}), mean % of packets created")
    print("  " + "-" * width)
    print(f"  {'load':>7}{'rule':>9}{'total':>9}{'std':>8}{'channel':>10}"
          f"{'queue_full':>12}{'no_route':>10}{'dead_end':>10}{'ttl':>8}"
          f"{'hop_limit':>11}{'hops':>7}")
    print("  " + "-" * width)
    for load_ms in LOADS_MS:
        for rule in rules:
            totals = per_layout(rows, rule, load_ms)
            causes = [statistics.fmean(
                per_layout(rows, rule, load_ms, TEST50_SEEDS, c)
            ) for c in CAUSES]
            hops = statistics.fmean(
                per_layout(rows, rule, load_ms, TEST50_SEEDS, "mean_hops")
            )
            print(f"  {load_ms:>5} ms{rule:>9}"
                  f"{statistics.fmean(totals)*100:>8.2f}%"
                  f"{statistics.stdev(totals)*100:>8.2f}"
                  f"{causes[0]*100:>9.2f}%{causes[1]*100:>11.2f}%"
                  f"{causes[2]*100:>9.2f}%{causes[3]*100:>9.2f}%"
                  f"{causes[4]*100:>7.2f}%{causes[5]*100:>10.2f}%"
                  f"{hops:>7.2f}")
        print("  " + "-" * width)


def print_paired(paired: Sequence[Dict[str, Any]], label: str,
                 heading: str) -> None:
    """Print one paired table, for the rows whose ``layouts`` is *label*."""
    width = 94
    print()
    print(f"  {heading}")
    print("      negative = the first rule loses fewer packets")
    print("  " + "-" * width)
    print(f"  {'load':>7}{'comparison':>18}{'mean diff':>12}{'std':>9}"
          f"{'wins':>10}{'ties':>7}{'p (sign)':>12}{'p (t-test)':>13}")
    print("  " + "-" * width)
    for row in paired:
        if row["layouts"] != label:
            continue
        wins = f"{row['wins']} / {row['n']}"
        print(f"  {row['load_ms']:>5} ms{row['comparison']:>18}"
              f"{row['mean_diff']*100:>11.2f}%{row['std_diff']*100:>8.2f}%"
              f"{wins:>10}{row['ties']:>7}"
              f"{_fmt_p(row['p_sign']):>12}{_fmt_p(row['p_ttest']):>13}")
    print("  " + "-" * width)


def _fmt_p(p: float) -> str:
    """Format a p-value without pretending to more precision than it has."""
    if p < 1e-4:
        return "<0.0001"
    return f"{p:.4f}"


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _style(ax: plt.Axes) -> None:
    """Apply the shared recessive axis styling."""
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_comparison(rows: Sequence[Dict[str, Any]], path: str,
                    rules: Sequence[str] = RULES) -> None:
    """Draw total loss per rule, one dot per layout, at both loads."""
    fig, axes = plt.subplots(1, len(LOADS_MS), figsize=(12.0, 5.6), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, load_ms in zip(axes, LOADS_MS):
        ax.set_facecolor(SURFACE)
        # Percentages everywhere: the boxes and the dots must share units.
        series = [[value * 100 for value in per_layout(rows, rule, load_ms)]
                  for rule in rules]

        parts = ax.boxplot(series, widths=0.55, patch_artist=True,
                           medianprops={"color": TEXT_PRIMARY,
                                        "linewidth": 1.6},
                           whiskerprops={"color": TEXT_SECONDARY},
                           capprops={"color": TEXT_SECONDARY},
                           flierprops={"marker": ""}, zorder=2)
        for patch, rule in zip(parts["boxes"], rules):
            patch.set_facecolor(RULE_COLORS[rule])
            patch.set_alpha(0.25)
            patch.set_edgecolor(RULE_COLORS[rule])

        rng = np.random.default_rng(0)       # fixed jitter, so the plot replays
        for index, (rule, values) in enumerate(zip(rules, series), start=1):
            jitter = rng.uniform(-0.18, 0.18, size=len(values))
            ax.scatter(index + jitter, values, s=16,
                       color=RULE_COLORS[rule], alpha=0.7, zorder=3,
                       edgecolors="none")

        # The mean goes under the tick label, where nothing can collide.
        ax.set_xticks(range(1, len(rules) + 1))
        ax.set_xticklabels([
            f"{rule}\n{statistics.fmean(values):.1f}%"
            for rule, values in zip(rules, series)
        ])
        heading = f"{load_ms} ms"
        if load_ms != MAIN_LOAD_MS:
            heading += "  (never trained on)"
        ax.set_title(heading, color=TEXT_PRIMARY, fontsize=12,
                     fontweight="bold", pad=10)
        _style(ax)

    axes[0].set_ylabel("Total packet loss (%)", color=TEXT_SECONDARY,
                       fontsize=11)
    axes[0].set_ylim(0, 100)

    fig.suptitle(
        f"Phase-1b on {len(TEST50_SEEDS)} held-back TEST layouts — "
        "one dot per layout, label = mean",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.text(0.01, 0.012,
             stamp_lines(width=140, loads_ms=list(LOADS_MS),
                         placements=len(TEST50_SEEDS)),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def _draw_panel(ax: plt.Axes, means: np.ndarray, stds: np.ndarray,
                labels: Sequence[str], title: str) -> Any:
    """Draw one rule's panel: loads down the side, causes across the top.

    Args:
        ax:     Axes to draw on.
        means:  loads x columns of mean percentages.
        stds:   Matching standard deviations across layouts.
        labels: Column labels, already broken over lines where needed.
        title:  The rule's name.

    Returns:
        The image handle, for attaching the shared colour bar.
    """
    image = ax.imshow(means, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(means.shape[0]))
    ax.set_yticklabels([f"{load} ms" for load in LOADS_MS], fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, fontweight="bold",
                 pad=12)

    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            # Keep the label readable against a dark cell.
            ink = "#ffffff" if means[i, j] > 55 else TEXT_PRIMARY
            ax.text(j, i - 0.09, f"{means[i, j]:.1f}", ha="center",
                    va="center", color=ink, fontsize=11, fontweight="bold")
            ax.text(j, i + 0.20, f"({stds[i, j]:.1f})", ha="center",
                    va="center", color=ink, fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_heatmaps(rows: Sequence[Dict[str, Any]], path: str,
                  rules: Sequence[str] = RULES) -> None:
    """Draw total loss and every cause, one panel per rule, both loads.

    Every number is a share of the packets created, so all four panels share
    one colour bar.

    Args:
        rows: Every run's row.
        path: Output PNG path.
    """
    columns = [("total_loss", "total\nloss"), ("channel", "channel"),
               ("queue_full", "queue\nfull"), ("no_route", "no\nroute"),
               ("dead_end", "dead\nend"), ("ttl", "ttl"),
               ("hop_limit", "hop\nlimit")]

    fig = plt.figure(figsize=(19.5, 5.2), dpi=150)
    grid = fig.add_gridspec(
        1, len(rules) + 1,
        width_ratios=[len(columns)] * len(rules) + [0.30],
        wspace=0.28, bottom=0.26, top=0.80, left=0.045, right=0.965,
    )
    fig.patch.set_facecolor(SURFACE)

    axes = [fig.add_subplot(grid[0, i]) for i in range(len(rules))]
    bar_ax = fig.add_subplot(grid[0, len(rules)])

    image = None
    for ax, rule in zip(axes, rules):
        means = np.zeros((len(LOADS_MS), len(columns)))
        stds = np.zeros_like(means)
        for i, load_ms in enumerate(LOADS_MS):
            for j, (key, _) in enumerate(columns):
                values = [v * 100 for v in
                          per_layout(rows, rule, load_ms, TEST50_SEEDS, key)]
                means[i, j] = statistics.fmean(values)
                stds[i, j] = statistics.stdev(values)
        image = _draw_panel(ax, means, stds,
                            [label for _, label in columns], rule)

    bar = fig.colorbar(image, cax=bar_ax)
    bar.set_label("% of packets created", color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)

    fig.suptitle(
        f"Loss and its causes, by load and routing rule, on "
        f"{len(TEST50_SEEDS)} TEST layouts  \u2014  mean % (std)",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.text(0.01, 0.105,
             "Identical layouts and traffic for every rule; rl is the mean of "
             "3 init seeds. Every panel is a share of the packets created, so "
             "they share one scale. The six causes sum to the total loss.",
             color=TEXT_SECONDARY, fontsize=8)
    fig.text(0.01, 0.02,
             stamp_lines(width=170, loads_ms=list(LOADS_MS),
                         placements=len(TEST50_SEEDS)),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")

    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _read_rows(path: str) -> List[Dict[str, Any]]:
    """Read a saved run grid back, restoring the numeric types.

    Raises:
        SystemExit: If the file is not there.
    """
    from scripts.stamp import read_stamped_csv

    if not os.path.exists(path):
        print(f"STOP: {path} not found; run without --from-csv first.")
        raise SystemExit(1)
    rows: List[Dict[str, Any]] = []
    for raw in read_stamped_csv(path):
        row: Dict[str, Any] = {
            "rule": raw["rule"], "load_ms": int(raw["load_ms"]),
            "placement_seed": int(raw["placement_seed"]),
            "init_seed": int(raw["init_seed"]),
            "views_agree": raw["views_agree"] == "True",
            "discrepancies": [],
        }
        for key in ("total_loss", *CAUSES, "mean_hops"):
            row[key] = float(raw[key]) if raw[key] else None
        rows.append(row)
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              fields: Sequence[str]) -> None:
    """Write *rows* with the settings stamp as a leading comment."""
    with open(path, "w", newline="") as handle:
        handle.write(csv_comment(loads_ms=list(LOADS_MS),
                                 placements=len(TEST50_SEEDS),
                                 extra="phase 1b, 50 TEST layouts"))
        writer = csv.DictWriter(handle, fieldnames=list(fields),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}")


def parse_rules(values: Sequence[str]) -> List[str]:
    """Normalise the --rules argument into a list in RULES order.

    Accepts comma-separated ("greedy,random,rate") and space-separated
    ("greedy random rate") forms, since both read naturally on a command
    line.

    Args:
        values: The raw argument values.

    Returns:
        The chosen rules, in the canonical RULES order.

    Raises:
        SystemExit: If a name is not a known rule.
    """
    chosen: List[str] = []
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if not name:
                continue
            if name not in RULES:
                print(f"STOP: unknown rule {name!r}; known: {', '.join(RULES)}")
                raise SystemExit(1)
            if name not in chosen:
                chosen.append(name)
    if not chosen:
        print("STOP: --rules selected nothing")
        raise SystemExit(1)
    return [rule for rule in RULES if rule in chosen]


def main() -> None:
    """Run the 50-layout evaluation and print every table."""
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 1b on 50 held-back TEST layouts."
    )
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Which trained models to evaluate.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the models live and the outputs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--rules", type=str, nargs="+", default=list(RULES),
                        help="Which rules to run, comma- or space-separated. "
                             "Models are loaded only when a rule needing one "
                             "is among them, so greedy, random and rate can "
                             "be measured with no trained network present. "
                             "Default: all four.")
    parser.add_argument("--from-csv", action="store_true",
                        help="Redraw from out/1b_test50_runs.csv instead of "
                             "running the 600 runs again.")
    args = parser.parse_args()

    if config.LOOP_GUARD != "path":
        print(f"STOP: LOOP_GUARD is {config.LOOP_GUARD!r}, expected 'path'.")
        raise SystemExit(1)

    rules = parse_rules(args.rules)

    if args.from_csv:
        rows = _read_rows(os.path.join(args.out_dir, "1b_test50_runs.csv"))
        print(f"Redrawing from {len(rows)} saved runs "
              f"(nothing re-simulated).")
    else:
        rows = run_grid(args.out_dir, args.init_seeds, args.workers, rules)
    print_summary(rows, rules)

    paired = (paired_rows(rows, TEST50_SEEDS, "1-50", rules)
              + paired_rows(rows, ORIGINAL_SEEDS, "1-20", rules)
              + paired_rows(rows, ADDED_SEEDS, "21-50", rules))

    print_paired(paired, "1-50",
                 "(b) Paired on the same 50 layouts (seeds 1-50)")
    print_paired(paired, "1-20",
                 "(c1) Layouts 1-20 only — the CHECKPOINT-12 set")
    print_paired(paired, "21-50",
                 "(c2) Layouts 21-50 only — the 30 added layouts")

    print()
    if not args.from_csv:
        write_csv(os.path.join(args.out_dir, "1b_test50_runs.csv"), rows,
                  ["rule", "load_ms", "placement_seed", "init_seed",
                   "total_loss", *CAUSES, "mean_hops", "views_agree"])
    write_csv(os.path.join(args.out_dir, "1b_test50_paired.csv"), paired,
              ["layouts", "n", "load_ms", "comparison", "mean_diff",
               "std_diff", "wins", "ties", "p_sign", "p_ttest"])
    plot_comparison(rows, os.path.join(
        args.out_dir, "1b_comparison_total_loss_50.png"), rules)
    plot_heatmaps(rows, os.path.join(
        args.out_dir, "1b_loss_heatmaps_50.png"), rules)


if __name__ == "__main__":
    main()
