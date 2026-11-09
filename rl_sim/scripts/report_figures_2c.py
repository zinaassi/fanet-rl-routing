"""
report_figures_2c.py — The Phase 2c figures for the report's section 11.

    2c_training_curve.png   the varied-mission run (B): full-fleet and varied
                            validation loss against update, from
                            out/2c_r2_curve_B.csv, with the kept update marked
    GIFs                    copied unchanged from the committed edge-case
                            recordings (see GIFS below)

The curve is written stamped to out/ and, with --report, unstamped to
../docs/report/figures/ together with the GIFs.

Usage:
    python scripts/report_figures_2c.py [--report]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scripts.report_figures import (C_COLOR, M_COLOR, SURFACE, TEXT_PRIMARY,
                                    TEXT_SECONDARY, _stamp, _style)
from scripts.train_c_ppo_round1 import read_csv

FIGURES_DIR = os.path.join("..", "docs", "report", "figures")
CURVE = "2c_training_curve.png"

#: report name -> source under out/. Drawn from the saved recordings by
#: plot_edge_cases_2c.py --write-dir out/2c_report_gifs/... (no new runs):
#:   case 1 (script scenario 1), run seed 9901: the first RL agent (round-1
#:     control) and midpoint, recordings in out/2c_edge_r1_control/rec;
#:   case 6 (script scenario 9), run seed 9900: the first RL agent
#:     (out/2c_edge_bigfleet_r2_control/rec) and the agent after varied
#:     training (run B, out/2c_edge_bigfleet_r2_B/rec). 9900 is a typical
#:     seed: the first agent loses 86.9% there, as on 3 of its 5 seeds.
GIFS = {
    "2c_case1_rl.gif": "2c_report_gifs/case1/s1_control.gif",
    "2c_case1_midpoint.gif": "2c_report_gifs/case1/s1_midpoint.gif",
    "2c_case6_before.gif": "2c_report_gifs/case6_before/s9_control.gif",
    "2c_case6_after.gif": "2c_report_gifs/case6_after/s9_B.gif",
}


def curve(out_dir: str, path: str, stamp: bool) -> None:
    rows = read_csv(os.path.join(out_dir, "2c_r2_curve_B.csv"))
    x = [int(r["update"]) for r in rows]
    full = [float(r["full_loss"]) for r in rows]
    varied = [float(r["varied_loss"]) for r in rows]
    score = [float(r["eval_loss"]) for r in rows]
    best = min(range(len(rows)), key=lambda i: score[i])

    fig, ax = plt.subplots(figsize=(7.2, 3.9 if stamp else 3.5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(x, full, color=M_COLOR, lw=2, label="full fleet")
    ax.plot(x, varied, color=C_COLOR, lw=2, label="varied missions")
    ax.axvline(x[best], color=TEXT_SECONDARY, lw=1, ls="--")
    ax.text(x[best] + 6, max(full) + 0.2, f"kept: update {x[best]}",
            color=TEXT_SECONDARY, fontsize=9, va="top")
    for series, name in ((full, "full fleet"), (varied, "varied missions")):
        ax.text(x[-1] + 6, series[-1], name, color=TEXT_PRIMARY, fontsize=9,
                va="center")
    ax.set_xlim(0, x[-1] + 95)
    ax.set_xlabel("update", color=TEXT_SECONDARY)
    ax.set_ylabel("validation loss (% of packets)", color=TEXT_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="center right",
              bbox_to_anchor=(1.0, 0.55))
    if stamp:
        fig.subplots_adjust(bottom=0.25)
        _stamp(fig)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}  (kept update {x[best]}: full {full[best]:.2f}, "
          f"varied {varied[best]:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2c report figures.")
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    curve(args.out_dir, os.path.join(args.out_dir, CURVE), stamp=True)
    if args.report:
        curve(args.out_dir, os.path.join(FIGURES_DIR, CURVE), stamp=False)
        for name, source in GIFS.items():
            shutil.copy2(os.path.join(args.out_dir, source),
                         os.path.join(FIGURES_DIR, name))
            print(f"  copied {source} -> {name}")


if __name__ == "__main__":
    main()
