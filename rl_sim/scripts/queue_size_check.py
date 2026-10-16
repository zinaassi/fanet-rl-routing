"""
queue_size_check.py — How much the link-queue capacity Q decides the results.

Q = 10 is a PROVISIONAL choice, so every conclusion drawn at Q = 10 is
conditional on it. This reruns the key results at Q = 5, 10 and 20 and puts
them side by side.

Nothing is trained and no model changes. The "rl" models are the current
out/models/path/ networks, trained at Q = 10 and used unchanged at every Q.
Note that
rl's second input is the queue FILL, length / Q, computed with the Q of the
run — so at Q = 5 a queue holding 3 reads as 0.6 where at Q = 20 it reads as
0.15. The network was fitted against fills computed at Q = 10.

``config.QUEUE_CAPACITY`` is set per run and restored afterwards; the
default in config.py stays 10.

Writes:
    out/queue_size_runs.csv        Phase 1b, one row per run
    out/queue_size_paired.csv      paired comparisons per Q and load
    out/queue_size_1a.csv          Phase 1a load curve per Q
    out/queue_size_total_loss.png  total loss against Q
    out/queue_size_load_curve.png  greedy's load curve, one line per Q
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fanet_sim import config
from scripts.evaluate_1b_large import (
    CAUSES, RULE_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, GRID_COLOR,
    TEST50_SEEDS, paired_t_test, sign_test,
)
from scripts.stamp import csv_comment, stamp_lines
from scripts.train_1b import run_once

QUEUE_SIZES = (5, 10, 20)
BASELINE_Q = 10

# Phase 1b
LOADS_MS = (200, 500)
RULES = ("random", "greedy", "rate", "rl")
COMPARISONS = (("rl", "rate"), ("rl", "greedy"))

# Phase 1a
LOADS_1A_MS = (1000, 500, 200, 100)
RULES_1A = ("greedy", "random")
PLACEMENTS_1A = tuple(range(1, 21))

EXTRAS = ("mean_hops", "mean_delay_ms", "mean_queue", "max_queue")


def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one run at one queue capacity.

    Args:
        args: ``(queue, rule, load_ms, seed, init_seed, state_dict)``.

    Returns:
        That run's row.
    """
    queue, rule, load_ms, seed, init_seed, state = args
    previous = config.QUEUE_CAPACITY
    config.QUEUE_CAPACITY = queue
    try:
        result = run_once(rule, seed, seed, load_ms, state_dict=state,
                          training=False)
    finally:
        # Restored so a sequential run does not leave the parent's Q wherever
        # the last job happened to set it.
        config.QUEUE_CAPACITY = previous
    return {
        "queue_capacity": queue, "rule": rule, "load_ms": load_ms,
        "placement_seed": seed,
        "init_seed": init_seed if rule == "rl" else -1,
        "total_loss": result["total_loss"],
        **{cause: result[cause] for cause in CAUSES},
        **{extra: result[extra] for extra in EXTRAS},
        "views_agree": result["views_agree"],
        "discrepancies": result["discrepancies"],
    }


def _run_jobs(jobs: Sequence[Tuple], workers: int,
              what: str) -> List[Dict[str, Any]]:
    """Run *jobs* and stop if any run's two views disagree."""
    print(f"Running {len(jobs)} {what} runs ...")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_job, jobs))
    else:
        rows = [_job(job) for job in jobs]

    bad = [r for r in rows if not r["views_agree"]]
    if bad:
        for row in bad[:5]:
            print(f"STOP: views disagree — Q={row['queue_capacity']} "
                  f"{row['rule']} seed {row['placement_seed']}: "
                  f"{row['discrepancies']}")
        raise SystemExit(1)
    print(f"  views agreed in all {len(rows)} runs")
    return rows


def _load_states(out_dir: str,
                 init_seeds: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Return the trained weights, keyed by init seed.

    Raises:
        SystemExit: If a model is missing.
    """
    import torch

    states: Dict[int, Dict[str, Any]] = {}
    for init_seed in init_seeds:
        path = os.path.join(out_dir, "models", config.LOOP_GUARD,
                            f"rl_seed{init_seed}.pt")
        if not os.path.exists(path):
            print(f"STOP: no trained model at {path}.")
            raise SystemExit(1)
        states[init_seed] = torch.load(path, weights_only=True)
    return states


def run_phase_1b(out_dir: str, init_seeds: Sequence[int],
                 workers: int) -> List[Dict[str, Any]]:
    """Every rule on the 50 TEST layouts, at every Q and both loads."""
    states = _load_states(out_dir, init_seeds)
    jobs: List[Tuple] = []
    for queue in QUEUE_SIZES:
        for load_ms in LOADS_MS:
            for seed in TEST50_SEEDS:
                for rule in RULES:
                    if rule == "rl":
                        for init_seed in init_seeds:
                            jobs.append((queue, rule, load_ms, seed,
                                         init_seed, states[init_seed]))
                    else:
                        jobs.append((queue, rule, load_ms, seed, -1, None))
    return _run_jobs(jobs, workers, "Phase-1b")


def run_phase_1a(workers: int) -> List[Dict[str, Any]]:
    """greedy and random over the Phase-1a placements, every load and Q."""
    jobs: List[Tuple] = []
    for queue in QUEUE_SIZES:
        for load_ms in LOADS_1A_MS:
            for seed in PLACEMENTS_1A:
                for rule in RULES_1A:
                    jobs.append((queue, rule, load_ms, seed, -1, None))
    return _run_jobs(jobs, workers, "Phase-1a")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_layout(rows: Sequence[Dict[str, Any]], queue: int, rule: str,
               load_ms: int, seeds: Sequence[int],
               key: str = "total_loss") -> List[float]:
    """Return one value per layout, averaging rl over its init seeds."""
    values: List[float] = []
    for seed in seeds:
        matching = [r[key] for r in rows
                    if r["queue_capacity"] == queue and r["rule"] == rule
                    and r["load_ms"] == load_ms
                    and r["placement_seed"] == seed and r[key] is not None]
        values.append(statistics.fmean(matching))
    return values


def _mean(rows: Sequence[Dict[str, Any]], queue: int, rule: str,
          load_ms: int, seeds: Sequence[int], key: str) -> float:
    """Mean over layouts of *key*."""
    return statistics.fmean(per_layout(rows, queue, rule, load_ms, seeds, key))


def _max(rows: Sequence[Dict[str, Any]], queue: int, rule: str,
         load_ms: int, seeds: Sequence[int], key: str) -> float:
    """Largest value of *key* over every run, not a mean of maxima."""
    values = [r[key] for r in rows
              if r["queue_capacity"] == queue and r["rule"] == rule
              and r["load_ms"] == load_ms and r["placement_seed"] in seeds
              and r[key] is not None]
    return max(values) if values else 0.0


def print_1b_table(rows: Sequence[Dict[str, Any]]) -> None:
    """Phase 1b: every rule at every Q and load."""
    width = 128
    print()
    print(f"  (A) Phase 1b on {len(TEST50_SEEDS)} TEST layouts — "
          f"mean % of packets created, by link-queue capacity Q")
    print("  " + "-" * width)
    print(f"  {'load':>7}{'Q':>4}{'rule':>8}{'total':>9}{'std':>7}"
          f"{'channel':>9}{'queue_full':>12}{'no_route':>10}{'dead_end':>10}"
          f"{'ttl':>7}{'hop_lim':>9}{'hops':>7}{'delay ms':>10}"
          f"{'q mean':>9}{'q max':>8}")
    print("  " + "-" * width)
    for load_ms in LOADS_MS:
        for rule in RULES:
            for queue in QUEUE_SIZES:
                totals = per_layout(rows, queue, rule, load_ms, TEST50_SEEDS)
                causes = [_mean(rows, queue, rule, load_ms, TEST50_SEEDS, c)
                          for c in CAUSES]
                print(f"  {load_ms:>5} ms{queue:>4}{rule:>8}"
                      f"{statistics.fmean(totals)*100:>8.2f}%"
                      f"{statistics.stdev(totals)*100:>7.2f}"
                      f"{causes[0]*100:>8.2f}%{causes[1]*100:>11.2f}%"
                      f"{causes[2]*100:>9.2f}%{causes[3]*100:>9.2f}%"
                      f"{causes[4]*100:>6.2f}%{causes[5]*100:>8.2f}%"
                      f"{_mean(rows, queue, rule, load_ms, TEST50_SEEDS, 'mean_hops'):>7.2f}"
                      f"{_mean(rows, queue, rule, load_ms, TEST50_SEEDS, 'mean_delay_ms'):>10.1f}"
                      f"{_mean(rows, queue, rule, load_ms, TEST50_SEEDS, 'mean_queue'):>9.2f}"
                      f"{_max(rows, queue, rule, load_ms, TEST50_SEEDS, 'max_queue'):>8.0f}")
            print("  " + "-" * width)


def paired_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Paired comparisons per Q and load, over the 50 TEST layouts."""
    out: List[Dict[str, Any]] = []
    for queue in QUEUE_SIZES:
        for load_ms in LOADS_MS:
            for first, second in COMPARISONS:
                a = per_layout(rows, queue, first, load_ms, TEST50_SEEDS)
                b = per_layout(rows, queue, second, load_ms, TEST50_SEEDS)
                diffs = [x - y for x, y in zip(a, b)]
                wins, used, p_sign = sign_test(diffs)
                out.append({
                    "queue_capacity": queue, "load_ms": load_ms,
                    "comparison": f"{first} - {second}",
                    "n": len(diffs),
                    "mean_diff": statistics.fmean(diffs),
                    "std_diff": statistics.stdev(diffs),
                    "wins": wins, "ties": len(diffs) - used,
                    "p_sign": p_sign, "p_ttest": paired_t_test(diffs),
                })
    return out


def _fmt_p(p: float) -> str:
    """Format a p-value without pretending to more precision than it has."""
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def print_paired(paired: Sequence[Dict[str, Any]]) -> None:
    """The paired table, per Q and load."""
    width = 92
    print()
    print(f"  (A) Paired on the same {len(TEST50_SEEDS)} layouts — "
          "negative = the first rule loses fewer packets")
    print("  " + "-" * width)
    print(f"  {'load':>7}{'Q':>4}{'comparison':>16}{'mean diff':>12}"
          f"{'std':>9}{'wins':>10}{'p (sign)':>12}{'p (t-test)':>13}")
    print("  " + "-" * width)
    for load_ms in LOADS_MS:
        for row in paired:
            if row["load_ms"] != load_ms:
                continue
            wins = f"{row['wins']} / {row['n']}"
            print(f"  {load_ms:>5} ms{row['queue_capacity']:>4}"
                  f"{row['comparison']:>16}{row['mean_diff']*100:>11.2f}%"
                  f"{row['std_diff']*100:>8.2f}%{wins:>10}"
                  f"{_fmt_p(row['p_sign']):>12}{_fmt_p(row['p_ttest']):>13}")
        print("  " + "-" * width)


def print_1a_table(rows: Sequence[Dict[str, Any]]) -> None:
    """Phase 1a: the load curve at every Q."""
    width = 104
    print()
    print(f"  (B) Phase 1a on {len(PLACEMENTS_1A)} placements — "
          "mean % of packets created, by link-queue capacity Q")
    print("  " + "-" * width)
    print(f"  {'rule':>8}{'Q':>4}{'load':>9}{'total':>9}{'std':>7}"
          f"{'channel':>9}{'queue_full':>12}{'no_route':>10}{'dead_end':>10}"
          f"{'ttl':>7}{'hop_lim':>9}")
    print("  " + "-" * width)
    for rule in RULES_1A:
        for queue in QUEUE_SIZES:
            for load_ms in LOADS_1A_MS:
                totals = per_layout(rows, queue, rule, load_ms, PLACEMENTS_1A)
                causes = [_mean(rows, queue, rule, load_ms, PLACEMENTS_1A, c)
                          for c in CAUSES]
                print(f"  {rule:>8}{queue:>4}{load_ms:>6} ms"
                      f"{statistics.fmean(totals)*100:>8.2f}%"
                      f"{statistics.stdev(totals)*100:>7.2f}"
                      f"{causes[0]*100:>8.2f}%{causes[1]*100:>11.2f}%"
                      f"{causes[2]*100:>9.2f}%{causes[3]*100:>9.2f}%"
                      f"{causes[4]*100:>6.2f}%{causes[5]*100:>8.2f}%")
            print("  " + "-" * width)


# ---------------------------------------------------------------------------
# The reproduction check
# ---------------------------------------------------------------------------

def check_baseline(rows: Sequence[Dict[str, Any]], out_dir: str) -> bool:
    """Check that Q = 10 reproduces the committed 50-layout numbers.

    Compared per layout, not on the means, so a compensating pair of errors
    cannot hide.

    Returns:
        True if every value matches to 1e-12.
    """
    from scripts.stamp import read_stamped_csv

    path = os.path.join(out_dir, "1b_test50_runs.csv")
    if not os.path.exists(path):
        print(f"  (no {path}; skipping the Q = 10 reproduction check)")
        return True

    saved: Dict[Tuple, float] = {}
    for raw in read_stamped_csv(path):
        key = (raw["rule"], int(raw["load_ms"]), int(raw["placement_seed"]),
               int(raw["init_seed"]))
        saved[key] = float(raw["total_loss"])

    checked = 0
    bad: List[str] = []
    for row in rows:
        if row["queue_capacity"] != BASELINE_Q:
            continue
        key = (row["rule"], row["load_ms"], row["placement_seed"],
               row["init_seed"])
        if key not in saved:
            continue
        checked += 1
        if abs(saved[key] - row["total_loss"]) > 1e-12:
            bad.append(f"    {key}: saved {saved[key]!r}, now "
                       f"{row['total_loss']!r}")

    print()
    print(f"  Q = {BASELINE_Q} against the committed 50-layout runs: "
          f"{checked} runs compared per layout")
    if bad:
        print(f"  MISMATCH in {len(bad)} of them:")
        for line in bad[:10]:
            print(line)
        return False
    print("  every one identical to 1e-12")
    return True


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _sweep_stamp(**kwargs: Any) -> str:
    """The settings stamp, with Q naming the sweep instead of the default.

    These figures cover three capacities, so a stamp reading "Q=10/link"
    would name one of them as though it applied to the whole figure.
    """
    previous = config.QUEUE_CAPACITY
    config.QUEUE_CAPACITY = "/".join(str(q) for q in QUEUE_SIZES)  # type: ignore[assignment]
    try:
        return stamp_lines(**kwargs)
    finally:
        config.QUEUE_CAPACITY = previous


def _spread(values: Sequence[float], gap: float) -> List[float]:
    """Return label positions at least *gap* apart, closest to *values*.

    End-of-line labels collide whenever two lines finish close together, and
    two numbers printed on top of each other are worse than none. This nudges
    them apart while keeping their order.

    Args:
        values: The y positions the labels would ideally sit at.
        gap:    Minimum separation, in the same units.

    Returns:
        Adjusted positions, in the same order as *values*.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for position, index in enumerate(order):
        if position == 0:
            continue
        below = placed[order[position - 1]]
        if placed[index] - below < gap:
            placed[index] = below + gap
    return placed


def _style(ax: plt.Axes) -> None:
    """Apply the shared recessive axis styling."""
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_total_loss(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Total loss against Q, one line per rule, one panel per load."""
    fig, axes = plt.subplots(1, len(LOADS_MS), figsize=(11.5, 5.0), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, load_ms in zip(axes, LOADS_MS):
        ax.set_facecolor(SURFACE)
        ends: List[float] = []
        for rule in RULES:
            values = [_mean(rows, q, rule, load_ms, TEST50_SEEDS,
                            "total_loss") * 100 for q in QUEUE_SIZES]
            ax.plot(QUEUE_SIZES, values, marker="o", markersize=6,
                    linewidth=2.0, color=RULE_COLORS[rule], label=rule,
                    zorder=3)
            ends.append(values[-1])

        # rate and rl finish within a point or two of each other, so the
        # end labels have to be pushed apart before they are drawn.
        for rule, end, y in zip(RULES, ends, _spread(ends, 4.5)):
            ax.annotate(f"{end:.1f}%", (QUEUE_SIZES[-1], y),
                        textcoords="offset points", xytext=(8, 0),
                        color=RULE_COLORS[rule], fontsize=9,
                        fontweight="bold", va="center")
        ax.set_xscale("log", base=2)
        ax.set_xticks(QUEUE_SIZES)
        ax.set_xticklabels([str(q) for q in QUEUE_SIZES])
        ax.set_xlim(QUEUE_SIZES[0] * 0.85, QUEUE_SIZES[-1] * 1.35)
        ax.set_xlabel("link-queue capacity Q (packets)",
                      color=TEXT_SECONDARY, fontsize=10)
        ax.set_title(f"{load_ms} ms", color=TEXT_PRIMARY, fontsize=12,
                     fontweight="bold", pad=10)
        _style(ax)

    axes[0].set_ylabel("Total packet loss (%)  —  lower is better",
                       color=TEXT_SECONDARY, fontsize=10)
    axes[0].set_ylim(0, 100)
    # Above the lines, not over them: greedy sits around 50-58%.
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
                   loc="upper center", ncol=len(RULES),
                   bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        f"Total loss against link-queue capacity — {len(TEST50_SEEDS)} TEST "
        "layouts, rl trained at Q = 10 and used unchanged",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.text(0.01, 0.012,
             _sweep_stamp(width=140, loads_ms=list(LOADS_MS),
                          placements=len(TEST50_SEEDS)),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_load_curve(rows: Sequence[Dict[str, Any]], path: str) -> None:
    """Greedy's loss against offered load, one line per Q."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    shades = {5: "#9cc2ea", 10: "#2a78d6", 20: "#15406f"}
    # Q = 5 and Q = 10 sit almost on top of each other at every load, so a
    # plain solid line would simply hide one of them.
    dashes = {5: (1.5, 2.0), 10: (6.0, 2.0), 20: ()}
    widths = {5: 3.4, 10: 2.2, 20: 1.8}
    loads = sorted(LOADS_1A_MS)
    ends: List[float] = []
    for queue in QUEUE_SIZES:
        values = [_mean(rows, queue, "greedy", load, PLACEMENTS_1A,
                        "total_loss") * 100 for load in loads]
        ax.plot(loads, values, marker="o", markersize=6,
                linewidth=widths[queue], dashes=dashes[queue],
                color=shades[queue], label=f"Q = {queue}", zorder=3)
        ends.append(values[0])

    # Labels at the heaviest load, pushed apart where the lines converge.
    for queue, end, y in zip(QUEUE_SIZES, ends, _spread(ends, 4.5)):
        ax.annotate(f"{end:.1f}%", (loads[0], y),
                    textcoords="offset points", xytext=(-8, 0),
                    color=shades[queue], fontsize=9, fontweight="bold",
                    ha="right", va="center")

    ax.set_xscale("log")
    ax.set_xticks(loads)
    ax.set_xticklabels([str(load) for load in loads])
    ax.set_xlim(loads[0] * 0.72, loads[-1] * 1.12)
    # Heavier load is a shorter interval, so the axis runs right to left.
    ax.invert_xaxis()
    ax.set_xlabel("packet interval per M-drone (ms)  —  heavier load to the "
                  "right", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel("Total packet loss (%)  —  lower is better",
                  color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)
    _style(ax)

    fig.suptitle(
        f"Greedy's load curve at three link-queue capacities — "
        f"{len(PLACEMENTS_1A)} Phase-1a placements",
        color=TEXT_PRIMARY, fontsize=13, fontweight="bold", x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.text(0.01, 0.012,
             _sweep_stamp(width=130, loads_ms=list(loads),
                          placements=len(PLACEMENTS_1A)),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _read_rows(path: str, has_init_seed: bool) -> List[Dict[str, Any]]:
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
            "queue_capacity": int(raw["queue_capacity"]),
            "rule": raw["rule"], "load_ms": int(raw["load_ms"]),
            "placement_seed": int(raw["placement_seed"]),
            "init_seed": int(raw["init_seed"]) if has_init_seed else -1,
            "views_agree": raw["views_agree"] == "True",
            "discrepancies": [],
        }
        for key in ("total_loss", *CAUSES, *EXTRAS):
            row[key] = float(raw[key]) if raw[key] else None
        rows.append(row)
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, Any]],
              fields: Sequence[str], loads: Sequence[int],
              placements: int) -> None:
    """Write *rows* with the settings stamp as a leading comment.

    The stamp's Q is overridden: these files cover several capacities, so a
    stamp naming one of them would be wrong. The queue_capacity column says
    which Q each row ran at.
    """
    previous = config.QUEUE_CAPACITY
    config.QUEUE_CAPACITY = "/".join(str(q) for q in QUEUE_SIZES)  # type: ignore[assignment]
    try:
        with open(path, "w", newline="") as handle:
            handle.write(csv_comment(loads_ms=list(loads),
                                     placements=placements,
                                     extra="queue-size check"))
            writer = csv.DictWriter(handle, fieldnames=list(fields),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    finally:
        config.QUEUE_CAPACITY = previous
    print(f"  wrote {path}")


def main() -> None:
    """Run both sweeps, print every table, draw both figures."""
    parser = argparse.ArgumentParser(
        description="Rerun the key results at several link-queue capacities."
    )
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Which trained models to evaluate.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the models live and the outputs go.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--from-csv", action="store_true",
                        help="Redraw from the saved CSVs instead of running "
                             "every layout again. This mode SKIPS the Q = 10 "
                             "reproduction check, since nothing is "
                             "re-simulated.")
    args = parser.parse_args()

    print(f"Queue-size check — Q in {QUEUE_SIZES}, default stays "
          f"{config.QUEUE_CAPACITY}")
    print(f"  rl's queue-fill input is length / Q, computed with the Q of "
          f"the run; the models were trained at Q = {BASELINE_Q}.")

    matched = True
    if args.from_csv:
        rows_1b = _read_rows(os.path.join(args.out_dir,
                                          "queue_size_runs.csv"), True)
        rows_1a = _read_rows(os.path.join(args.out_dir,
                                          "queue_size_1a.csv"), False)
        print(f"Redrawing from {len(rows_1b)} + {len(rows_1a)} saved runs "
              f"(nothing re-simulated).")
    else:
        rows_1b = run_phase_1b(args.out_dir, args.init_seeds, args.workers)
        matched = check_baseline(rows_1b, args.out_dir)

    print_1b_table(rows_1b)
    paired = paired_rows(rows_1b)
    print_paired(paired)

    if not args.from_csv:
        rows_1a = run_phase_1a(args.workers)
    print_1a_table(rows_1a)

    print()
    if not args.from_csv:
        write_csv(os.path.join(args.out_dir, "queue_size_runs.csv"), rows_1b,
                  ["queue_capacity", "rule", "load_ms", "placement_seed",
                   "init_seed", "total_loss", *CAUSES, *EXTRAS, "views_agree"],
                  LOADS_MS, len(TEST50_SEEDS))
        write_csv(os.path.join(args.out_dir, "queue_size_1a.csv"), rows_1a,
                  ["queue_capacity", "rule", "load_ms", "placement_seed",
                   "total_loss", *CAUSES, *EXTRAS, "views_agree"],
                  LOADS_1A_MS, len(PLACEMENTS_1A))
    write_csv(os.path.join(args.out_dir, "queue_size_paired.csv"), paired,
              ["queue_capacity", "load_ms", "comparison", "n", "mean_diff",
               "std_diff", "wins", "ties", "p_sign", "p_ttest"],
              LOADS_MS, len(TEST50_SEEDS))

    plot_total_loss(rows_1b, os.path.join(args.out_dir,
                                          "queue_size_total_loss.png"))
    plot_load_curve(rows_1a, os.path.join(args.out_dir,
                                          "queue_size_load_curve.png"))

    if not matched:
        print("\nSTOP: Q = 10 did not reproduce the committed results.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
