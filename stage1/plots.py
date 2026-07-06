"""Summary figures (static PNG, light surface).

Colors follow a validated categorical palette: hues are assigned to routers
in a fixed order (never cycled), value labels are printed directly on the
marks, grids are hairlines, and text always uses ink colors rather than
series colors.
"""
from __future__ import annotations

import os
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import channel, config
from .metrics import AggStats

# Validated reference palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SLOTS = ("#2a78d6", "#1baf7a", "#eda100")  # categorical slots 1-3, fixed order

ROUTER_COLOR = {"direct": SLOTS[0], "greedy": SLOTS[1], "dijkstra": SLOTS[2]}
DELAY_COLOR = {"hops": SLOTS[0], "queue wait": SLOTS[1]}

CellKey = Tuple[str, float, str]  # (layout, k, router)
CellAgg = Mapping[CellKey, Mapping[str, AggStats]]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": BASELINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.axisbelow": True,
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "legend.frameon": False,
        }
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def calibration_plot(out_path: str, ks: Sequence[float] = config.K_SWEEP) -> None:
    """p_loss vs distance for each k, with the link-existence cutoff."""
    _style()
    d = np.linspace(1.0, 1200.0, 600)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, color in zip(ks, SLOTS):
        pl = channel.p_loss(d, k)
        rng = channel.max_link_range_m(k)
        ax.plot(d, pl, color=color, linewidth=2, label=f"k = {k} (cutoff at {rng:.0f} m)")
    ax.axhline(
        config.P_LOSS_CUTOFF, color=MUTED, linewidth=1, linestyle=(0, (4, 3))
    )
    ax.text(
        1180,
        config.P_LOSS_CUTOFF - 0.02,
        f"link cutoff (p_loss = {config.P_LOSS_CUTOFF})",
        fontsize=8,
        color=MUTED,
        ha="right",
        va="top",
    )
    ax.axvline(250, color=GRID, linewidth=0.8)
    ax.text(255, 0.03, "250 m: p_loss = 0.5", fontsize=8, color=MUTED)
    ax.set_xlabel("link distance d [m]")
    ax.set_ylabel("p_loss(d)")
    ax.set_title("Channel calibration: logistic packet loss vs distance")
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", labelcolor=INK_2)
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _cell_grid(
    agg: CellAgg,
    layouts: Sequence[str],
    ks: Sequence[float],
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        len(ks),
        len(layouts),
        figsize=(3.1 * len(layouts), 2.5 * len(ks)),
        sharey=True,
        squeeze=False,
    )
    for row, k in enumerate(ks):
        for col, layout in enumerate(layouts):
            ax = axes[row, col]
            ax.set_title(f"{layout}, k={k}", fontsize=9)
    return fig, axes


def metric_bar_grid(
    agg: CellAgg,
    metric: str,
    layouts: Sequence[str],
    ks: Sequence[float],
    routers: Sequence[str],
    out_path: str,
    title: str,
    ylabel: str,
    as_percent: bool = False,
) -> None:
    """One panel per (k, layout); bars per router with between-topology error bars."""
    _style()
    fig, axes = _cell_grid(agg, layouts, ks)
    scale = 100.0 if as_percent else 1.0
    for row, k in enumerate(ks):
        for col, layout in enumerate(layouts):
            ax = axes[row, col]
            xs = np.arange(len(routers))
            for x, r in zip(xs, routers):
                stats = agg.get((layout, k, r), {}).get(metric)
                if stats is None or np.isnan(stats.mean):
                    continue
                v = stats.mean * scale
                err = stats.between_std * scale
                ax.bar(
                    x, v, width=0.62, color=ROUTER_COLOR[r],
                    edgecolor=SURFACE, linewidth=1.5,
                )
                label_y = v
                if np.isfinite(err) and err > 0:
                    ax.errorbar(
                        x, v, yerr=err, fmt="none",
                        ecolor=INK_2, elinewidth=1, capsize=2.5,
                    )
                    label_y = v + err
                label = f"{v:.0f}" if as_percent else f"{v:.2f}"
                ax.annotate(
                    label, xy=(x, label_y), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels(routers, fontsize=8)
            if col == 0:
                ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ROUTER_COLOR[r]) for r in routers]
    fig.legend(handles, routers, loc="lower center", ncol=len(routers),
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def delay_decomposition_grid(
    agg: CellAgg,
    layouts: Sequence[str],
    ks: Sequence[float],
    routers: Sequence[str],
    out_path: str,
) -> None:
    """Stacked mean delay (delivered packets): transmission hops + queue wait, in ms."""
    _style()
    fig, axes = _cell_grid(agg, layouts, ks)
    for row, k in enumerate(ks):
        for col, layout in enumerate(layouts):
            ax = axes[row, col]
            xs = np.arange(len(routers))
            for x, r in zip(xs, routers):
                cell = agg.get((layout, k, r), {})
                hops = cell.get("mean_hops")
                wait = cell.get("mean_queue_wait_steps")
                if hops is None or np.isnan(hops.mean):
                    continue
                hop_ms = hops.mean * config.STEP_MS
                wait_ms = (wait.mean if wait and np.isfinite(wait.mean) else 0.0) * config.STEP_MS
                ax.bar(x, hop_ms, width=0.62, color=DELAY_COLOR["hops"],
                       edgecolor=SURFACE, linewidth=1.5)
                ax.bar(x, wait_ms, bottom=hop_ms, width=0.62,
                       color=DELAY_COLOR["queue wait"], edgecolor=SURFACE, linewidth=1.5)
                ax.text(x, hop_ms + wait_ms, f" {hop_ms + wait_ms:.0f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)
            ax.set_xticks(xs)
            ax.set_xticklabels(routers, fontsize=8)
            if col == 0:
                ax.set_ylabel("mean delay [ms]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DELAY_COLOR.values()]
    fig.legend(handles, list(DELAY_COLOR), loc="lower center", ncol=2,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Delay decomposition (delivered packets)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_summary_plots(
    agg: CellAgg,
    layouts: Sequence[str],
    ks: Sequence[float],
    routers: Sequence[str],
    out_dir: str,
) -> Dict[str, str]:
    """Write the standard figure set; returns {figure name: path}."""
    paths = {
        "calibration": os.path.join(out_dir, "calibration.png"),
        "pdr": os.path.join(out_dir, "pdr.png"),
        "delay": os.path.join(out_dir, "delay_decomposition.png"),
        "unreachable": os.path.join(out_dir, "unreachable.png"),
    }
    calibration_plot(paths["calibration"], ks)
    metric_bar_grid(
        agg, "pdr_global", layouts, ks, routers, paths["pdr"],
        title="Global PDR (in-flight packets excluded)", ylabel="PDR",
    )
    delay_decomposition_grid(agg, layouts, ks, routers, paths["delay"])
    metric_bar_grid(
        agg, "unreachable_frac_m", layouts, ks, routers, paths["unreachable"],
        title="Unreachable M-drones (no route to GS)", ylabel="% of M-drones",
        as_percent=True,
    )
    return paths
