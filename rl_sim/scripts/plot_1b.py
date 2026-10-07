"""
plot_1b.py — The Phase-1b learning curve and the validation loss by cause.

Reads ``out/1b_training_log.csv`` and draws:

    out/1b_learning_curve.png
        top    total packet loss against training runs: validation,
               train-check, the faint per-run training loss (which carries
               exploration noise), and dashed reference lines for greedy,
               rate and random measured on the same validation layouts.
               With several init seeds, the mean line is drawn with a
               shaded min-max band.
        bottom the training loss (BCE) per run.

    out/1b_validation_causes.png
        where the loss goes on the validation layouts at each evaluation:
        channel, queue_full, no_route and ttl+hop, as a share of packets
        created.

Both carry the settings stamp.

Usage:
    python scripts/plot_1b.py
    python scripts/plot_1b.py --log out/1b_training_log.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.stamp import read_stamped_csv, stamp_lines

LOAD_MS = 200

# Categorical hues, validated as a categorical palette.
RL_COLOR = "#2a78d6"
TRAIN_CHECK_COLOR = "#1baf7a"
TRAIN_RUN_COLOR = "#9fc4ea"
REFERENCE_COLORS = {"greedy": "#eb6834", "rate": "#4a3aa7", "random": "#e34948"}
CAUSE_COLORS = {
    "channel": "#2a78d6",
    "queue_full": "#eb6834",
    "no_route": "#1baf7a",
    "ttl_hop": "#eda100",
}
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5c5b54"
GRID_COLOR = "#e4e3dd"
SURFACE = "#fcfcfb"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Plot the Phase-1b training.")
    parser.add_argument("--log", type=str,
                        default=os.path.join("out", "1b_training_log.csv"),
                        help="The training log to read.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the figures go.")
    return parser.parse_args()


def _number(value: str) -> Optional[float]:
    """Return a CSV cell as a float, or None when it is blank."""
    return float(value) if value not in ("", "None") else None


def load_log(path: str) -> Dict[str, Any]:
    """Split the training log into the pieces the figures need.

    Args:
        path: The CSV written by train_1b.py.

    Returns:
        ``train`` and ``evaluation`` rows grouped by init seed, plus the
        ``reference`` rules measured on the validation layouts.
    """
    rows = read_stamped_csv(path)
    train: Dict[int, List[Tuple[int, float, Optional[float]]]] = {}
    evaluation: Dict[int, List[Dict[str, Any]]] = {}
    references: Dict[str, Dict[str, float]] = {}

    for row in rows:
        if row["kind"] == "reference":
            references[row["routing"]] = {
                "total_loss": float(row["validation_total_loss"]),
                "channel": float(row["validation_channel"]),
                "queue_full": float(row["validation_queue_full"]),
                "no_route": float(row["validation_no_route"]),
                "ttl_hop": float(row["validation_ttl_hop"]),
            }
            continue

        seed = int(row["init_seed"])
        index = int(row["run_index"])
        if row["kind"] == "train":
            train.setdefault(seed, []).append(
                (index, float(row["train_total_loss"]), _number(row["train_bce"]))
            )
        else:
            evaluation.setdefault(seed, []).append({
                "run_index": index,
                "validation": float(row["validation_total_loss"]),
                "train_check": float(row["train_check_total_loss"]),
                "channel": float(row["validation_channel"]),
                "queue_full": float(row["validation_queue_full"]),
                "no_route": float(row["validation_no_route"]),
                "ttl_hop": float(row["validation_ttl_hop"]),
            })
    return {"train": train, "evaluation": evaluation, "references": references}


def _band(
    series: Dict[int, List[Dict[str, Any]]], key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, mean, min and max of one evaluation series across init seeds.

    Seeds may stop at different runs, so the series is truncated to the
    shortest: a mean over a changing set of seeds would bend for a reason
    that has nothing to do with learning.

    Args:
        series: Evaluation rows per init seed.
        key:    Which value to summarise.

    Returns:
        ``(run_index, mean, minimum, maximum)``, all as percentages.
    """
    shortest = min(len(rows) for rows in series.values())
    runs = np.array([row["run_index"]
                     for row in next(iter(series.values()))[:shortest]])
    stacked = np.array([[row[key] for row in rows[:shortest]]
                        for rows in series.values()]) * 100.0
    return runs, stacked.mean(axis=0), stacked.min(axis=0), stacked.max(axis=0)


def _style(ax: plt.Axes) -> None:
    """Apply the shared recessive axis styling."""
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def plot_learning_curve(log: Dict[str, Any], path: str) -> None:
    """Draw the two-panel learning curve.

    Args:
        log:  Output of :func:`load_log`.
        path: Output PNG path.
    """
    train, evaluation, references = log["train"], log["evaluation"], log["references"]
    seeds = sorted(evaluation)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10.0, 8.0), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)

    # --- per-run training loss, faint: it carries exploration noise ---
    for seed in sorted(train):
        runs = [index for index, _, _ in train[seed]]
        losses = [value * 100 for _, value, _ in train[seed]]
        top.plot(runs, losses, color=TRAIN_RUN_COLOR, linewidth=0.9,
                 alpha=0.8, zorder=2,
                 label="training runs (with exploration)" if seed == seeds[0] else None)

    # --- validation and train-check ---
    for key, color, label in (
        ("validation", RL_COLOR, "rl — validation"),
        ("train_check", TRAIN_CHECK_COLOR, "rl — train-check"),
    ):
        runs, mean, low, high = _band(evaluation, key)
        if len(seeds) > 1:
            top.fill_between(runs, low, high, color=color, alpha=0.18, zorder=3)
        top.plot(runs, mean, color=color, linewidth=2.4, marker="o",
                 markersize=5, label=label, zorder=4)

    # --- reference rules on the same validation layouts ---
    for rule, values in references.items():
        top.axhline(values["total_loss"] * 100,
                    color=REFERENCE_COLORS.get(rule, TEXT_SECONDARY),
                    linewidth=1.6, linestyle="--", zorder=3,
                    label=f"{rule} (validation)")

    top.set_ylabel("Total packet loss (%)", color=TEXT_SECONDARY, fontsize=11)
    top.set_ylim(0, 100)
    title = "Phase-1b training"
    if len(seeds) > 1:
        title += f" — mean of {len(seeds)} init seeds, band = min-max"
    else:
        title += f" — init seed {seeds[0]}"
    top.set_title(title, color=TEXT_PRIMARY, fontsize=13,
                  fontweight="bold", pad=12, loc="left")
    _style(top)
    top.legend(loc="upper right", frameon=False, fontsize=9,
               labelcolor=TEXT_SECONDARY, ncol=2)

    # --- bottom: the BCE actually optimised ---
    for seed in sorted(train):
        runs = [index for index, _, bce in train[seed] if bce is not None]
        values = [bce for _, _, bce in train[seed] if bce is not None]
        bottom.plot(runs, values, color=RL_COLOR, linewidth=1.2, alpha=0.9,
                    label=f"init seed {seed}" if len(seeds) > 1 else None)
    bottom.set_xlabel("Training runs", color=TEXT_SECONDARY, fontsize=11)
    bottom.set_ylabel("Training loss (BCE)", color=TEXT_SECONDARY, fontsize=11)
    _style(bottom)
    if len(seeds) > 1:
        bottom.legend(loc="upper right", frameon=False, fontsize=9,
                      labelcolor=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.012, stamp_lines(width=112, loads_ms=[LOAD_MS]),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_validation_causes(log: Dict[str, Any], path: str) -> None:
    """Draw where the validation loss goes, evaluation by evaluation.

    Args:
        log:  Output of :func:`load_log`.
        path: Output PNG path.
    """
    evaluation = log["evaluation"]
    seeds = sorted(evaluation)

    fig, ax = plt.subplots(figsize=(10.0, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for cause, color in CAUSE_COLORS.items():
        runs, mean, low, high = _band(evaluation, cause)
        if len(seeds) > 1:
            ax.fill_between(runs, low, high, color=color, alpha=0.15, zorder=2)
        # Legend only, no direct labels: queue_full and no_route both sit on
        # zero here, so labels at the line ends would land on top of each other.
        ax.plot(runs, mean, color=color, linewidth=2.2, marker="o",
                markersize=5, label=cause, zorder=3)

    ax.set_xlabel("Training runs", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylabel("% of packets created", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=max(row["run_index"] for row in evaluation[seeds[0]]) * 1.04)
    subtitle = (f"mean of {len(seeds)} init seeds" if len(seeds) > 1
                else f"init seed {seeds[0]}")
    ax.set_title(f"Where the validation loss goes — {subtitle}",
                 color=TEXT_PRIMARY, fontsize=13, fontweight="bold",
                 pad=12, loc="left")
    _style(ax)
    # Centre-right is the empty band between the channel line and the rest.
    ax.legend(loc="center right", frameon=False, fontsize=9,
              labelcolor=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.012, stamp_lines(width=112, loads_ms=[LOAD_MS]),
             color=TEXT_SECONDARY, fontsize=7.5, va="bottom")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def best_evaluation(evaluation: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Return the evaluation row with the lowest validation total loss.

    Averaged across init seeds at the same run index, so "best" means the
    checkpoint the protocol would keep, not one seed's lucky run.

    Args:
        evaluation: Evaluation rows per init seed.

    Returns:
        The best row, with an extra ``init_seeds`` count.
    """
    seeds = sorted(evaluation)
    shortest = min(len(rows) for rows in evaluation.values())
    keys = ("validation", "train_check", "channel", "queue_full",
            "no_route", "ttl_hop")

    best: Optional[Dict[str, Any]] = None
    for position in range(shortest):
        averaged = {
            key: float(np.mean([evaluation[seed][position][key] for seed in seeds]))
            for key in keys
        }
        averaged["run_index"] = evaluation[seeds[0]][position]["run_index"]
        averaged["init_seeds"] = len(seeds)
        if best is None or averaged["validation"] < best["validation"]:
            best = averaged
    assert best is not None
    return best


def print_cause_comparison(log: Dict[str, Any]) -> None:
    """Print the validation loss by cause for greedy, rate and the best rl.

    Args:
        log: Output of :func:`load_log`.
    """
    references = log["references"]
    best = best_evaluation(log["evaluation"])

    print()
    print("  Validation layouts — total loss and where it goes, % of packets created")
    plural = "s" if best["init_seeds"] > 1 else ""
    print(f"  best rl checkpoint: run {best['run_index']}"
          f"  ({best['init_seeds']} init seed{plural})")
    print("  " + "-" * 74)
    print(f"  {'rule':<10}{'total':>10}{'channel':>12}{'queue_full':>13}"
          f"{'no_route':>11}{'ttl+hop':>11}")
    print("  " + "-" * 74)

    for rule in ("random", "greedy", "rate"):
        if rule not in references:
            continue
        values = references[rule]
        print(f"  {rule:<10}{values['total_loss']*100:>9.2f}%"
              f"{values['channel']*100:>11.2f}%{values['queue_full']*100:>12.2f}%"
              f"{values['no_route']*100:>10.2f}%{values['ttl_hop']*100:>10.2f}%")

    print(f"  {'rl (best)':<10}{best['validation']*100:>9.2f}%"
          f"{best['channel']*100:>11.2f}%{best['queue_full']*100:>12.2f}%"
          f"{best['no_route']*100:>10.2f}%{best['ttl_hop']*100:>10.2f}%")
    print("  " + "-" * 74)

    if "rate" in references:
        rate = references["rate"]
        print(f"  rl - rate   {(best['validation']-rate['total_loss'])*100:>+9.2f}"
              f"{(best['channel']-rate['channel'])*100:>+11.2f}"
              f"{(best['queue_full']-rate['queue_full'])*100:>+12.2f}"
              f"{(best['no_route']-rate['no_route'])*100:>+10.2f}"
              f"{(best['ttl_hop']-rate['ttl_hop'])*100:>+10.2f}")
    if "greedy" in references:
        greedy = references["greedy"]
        print(f"  rl - greedy {(best['validation']-greedy['total_loss'])*100:>+9.2f}"
              f"{(best['channel']-greedy['channel'])*100:>+11.2f}"
              f"{(best['queue_full']-greedy['queue_full'])*100:>+12.2f}"
              f"{(best['no_route']-greedy['no_route'])*100:>+10.2f}"
              f"{(best['ttl_hop']-greedy['ttl_hop'])*100:>+10.2f}")
    print("  " + "-" * 74)


def main() -> None:
    """Draw both figures from the training log."""
    args = parse_args()
    log = load_log(args.log)
    if not log["evaluation"]:
        raise SystemExit(f"{args.log} has no evaluation rows yet")

    os.makedirs(args.out_dir, exist_ok=True)
    curve = os.path.join(args.out_dir, "1b_learning_curve.png")
    causes = os.path.join(args.out_dir, "1b_validation_causes.png")
    plot_learning_curve(log, curve)
    plot_validation_causes(log, causes)
    print_cause_comparison(log)
    print()
    print(f"  wrote {curve}")
    print(f"  wrote {causes}")


if __name__ == "__main__":
    main()
