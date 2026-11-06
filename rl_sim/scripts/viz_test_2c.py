"""
viz_test_2c.py — Phase 2c: two full-fleet test episodes, drawn.

Pictures only, no training and no new numbers for the report. Re-runs two of
the direct actor's test episodes (layouts 401-450, D = 650, the run seeds the
test run used) with the direct actor and with midpoint, records every step of
the scored run, checks the direct loss against the test run's, and draws:

    out/2c_viz_test/L{layout}_direct.gif, L{layout}_midpoint.gif
    out/2c_viz_test/L{layout}_snapshots.png   t = 0/30/60/95 s, midpoint over direct
    out/2c_viz_test/L{layout}_c_distance.png  each C-drone's distance from the GS
    out/2c_viz_test/L{layout}_loss.csv        total loss and loss by cause

The episodes: the layout whose direct loss is closest to the test median
(a tie goes to the lower layout number) and the layout with the highest
direct loss, both read from the test run's per-layout CSV.

The drawing is plot_edge_cases_2c's, and the recording is edge_cases_2c's.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from fanet_sim import config
from fanet_sim.envs.c_scripted import MidpointAgent
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.edge_cases_2c import DIRECT_MODEL, _record_step
from scripts.evaluate_2c_m4_test import run_seed_for
from scripts.plot_edge_cases_2c import (FRAME_EVERY,
                                        draw_step, legend_handles,
                                        setup_axis)
from scripts.stamp import settings_stamp
from scripts.train_c_ppo import (CAUSES, ROUTING, SamplingAgent, _restore,
                                 _world, write_csv)
from scripts.metrics_1a import compute_metrics

DISTANCE_M: float = 650.0
POLICIES: Tuple[str, ...] = ("direct", "midpoint")
SNAPSHOT_TIMES_S = (0.0, 30.0, 60.0, 95.0)
MATCH_TOLERANCE: float = 1e-9

#: Half the edge-case GIFs' speed: one 0.5 s frame every 0.1 s, so 5x slower
#: than real time.
GIF_FPS: int = 10


def without_links(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the recording with its links emptied, for drawing.

    With the full fleet every link drawn turns the picture into a mesh that
    hides the drones, so the pictures show positions and trails only.
    """
    empty_link = np.zeros((0, 4))
    empty_gs = np.zeros((0, 3))
    return {**rec,
            "links_at": [empty_link] * len(rec["links_at"]),
            "gs_at": [empty_gs] * len(rec["gs_at"])}


def pick_layouts(test_csv: str) -> List[Tuple[int, float, str]]:
    """Return (layout, direct loss %, why) for the median and worst episodes."""
    with open(test_csv) as handle:
        rows = [r for r in csv.DictReader(line for line in handle
                                          if not line.startswith("#"))
                if r["policy"] == "direct_final"]
    loss = {int(r["placement_seed"]): float(r["total_loss"]) for r in rows}
    median = statistics.median(loss.values())
    near = min(sorted(loss), key=lambda layout: abs(loss[layout] - median))
    worst = max(sorted(loss), key=lambda layout: loss[layout])
    ties = [layout for layout in sorted(loss)
            if abs(abs(loss[layout] - median) - abs(loss[near] - median))
            < 1e-12]
    return [
        (near, loss[near],
         f"closest to the median {median:.2f}% of {len(loss)} layouts"
         + (f" (tied with {ties[1:]}; lower layout number taken)"
            if len(ties) > 1 else "")),
        (worst, loss[worst], "highest direct loss"),
    ]


def run(layout: int, policy: str, state: Any) -> Dict[str, Any]:
    """Run one test episode exactly as the test run did, and record it."""
    import torch

    run_seed = run_seed_for(layout)
    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [run_seed, config.AGENT_STREAM_TAG]).generate_state(1)[0]))
    previous = _world(moving=True, distance=DISTANCE_M)
    try:
        env = FANETEnv(routing=ROUTING, log_path=os.devnull,
                       placement_seed=layout, run_seed=run_seed)
        env.reset()
        if policy == "midpoint":
            env.c_agent = MidpointAgent(env)
        else:
            from agents.c_actor import GaussianActor

            actor = GaussianActor()
            actor.load_state_dict(state)
            actor.eval()
            env.c_agent = SamplingAgent(env, actor, greedy=True)

        store: Dict[str, list] = {"pos": [], "link": [], "gs": []}
        _record_step(env, 0, store)
        for step in range(1, config.MAX_STEPS + 1):
            env.step()
            _record_step(env, step, store)
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        links = np.array(store["link"], dtype=np.float64).reshape(-1, 4)
        gs_links = np.array(store["gs"], dtype=np.float64).reshape(-1, 3)
        steps = len(store["pos"])
        return {
            "policy": policy,
            "run_seed": run_seed,
            "total_loss": 100.0 * (truth["total_loss"] or 0.0),
            **{c: 100.0 * truth["loss_rate_by_reason"][c] for c in CAUSES},
            "c_distance_flown_mean": truth["c_distance_flown_mean"],
            "views_agree": not metrics["discrepancies"],
            "rec": {
                "pos": np.array(store["pos"]),
                "types": np.array([d.drone_type for d in env.drones]),
                "gs": np.asarray(env.gs_position, dtype=np.float64),
                "m_end": np.array([d.end_point for d in env.drones
                                   if d.drone_type == "M"]),
                "links_at": np.split(links, np.searchsorted(
                    links[:, 0], np.arange(1, steps))),
                "gs_at": np.split(gs_links, np.searchsorted(
                    gs_links[:, 0], np.arange(1, steps))),
            },
        }
    finally:
        _restore(previous)


def gif(rec: Dict[str, Any], title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    setup_axis(ax, rec)
    ax.set_title(title, fontsize=8)
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=5.5,
              framealpha=0.8)
    fig.text(0.5, 0.005, settings_stamp(extra="test episode"), ha="center",
             fontsize=3.5, color="#666666", wrap=True)
    artists: Dict[str, Any] = {}

    def update(step: int):
        draw_step(ax, rec, step, artists)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, len(rec["pos"]), FRAME_EVERY),
        blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=GIF_FPS), dpi=80)
    plt.close(fig)


def snapshots(recs: Dict[str, Dict[str, Any]], title: str, path: str) -> None:
    fig, axes = plt.subplots(2, len(SNAPSHOT_TIMES_S), figsize=(11, 6.2))
    for row, policy in enumerate(("midpoint", "direct")):
        for col, t in enumerate(SNAPSHOT_TIMES_S):
            ax = axes[row, col]
            setup_axis(ax, recs[policy])
            draw_step(ax, recs[policy], int(round(t / config.TIMESTEP)), {})
            if col == 0:
                ax.set_ylabel(policy, fontsize=10)
    fig.suptitle(title + "; top midpoint, bottom direct actor", fontsize=10)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(0.5, 0.005, settings_stamp(extra="test episode"), ha="center",
             fontsize=4.5, color="#666666", wrap=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def c_distance_plot(recs: Dict[str, Dict[str, Any]], title: str,
                    path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, policy in zip(axes, POLICIES):
        rec = recs[policy]
        c = [i for i, t in enumerate(rec["types"]) if t == "C"]
        m = [i for i, t in enumerate(rec["types"]) if t == "M"]
        time = np.arange(len(rec["pos"])) * config.TIMESTEP
        dist = np.linalg.norm(rec["pos"] - rec["gs"], axis=-1)
        ax.fill_between(time, dist[:, m].min(axis=1), dist[:, m].max(axis=1),
                        color="#3a86ff", alpha=0.12, label="M-drones (range)")
        for k, i in enumerate(c):
            ax.plot(time, dist[:, i], lw=1.0, label=f"C {i}")
        ax.axhline(350.0, ls="--", lw=0.7, color="#9aa0a6",
                   label="GS reach 350 m")
        ax.set_title(policy, fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("distance from GS (m)", fontsize=8)
    axes[1].legend(fontsize=6, ncol=2, loc="upper left")
    fig.suptitle(title + ": each C-drone's distance from the GS", fontsize=9)
    fig.text(0.5, 0.005, settings_stamp(extra="test episode"), ha="center",
             fontsize=4.0, color="#666666", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(
        description="Phase 2c: two full-fleet test episodes, drawn."
    )
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--test-csv", type=str, required=True,
                        help="the test run's per-layout CSV")
    args = parser.parse_args()
    out = os.path.join(args.out_dir, "2c_viz_test")
    os.makedirs(out, exist_ok=True)
    state = torch.load(os.path.join(args.out_dir, DIRECT_MODEL),
                       weights_only=True)

    for layout, expected, why in pick_layouts(args.test_csv):
        print(f"\nLayout {layout}: {why}; test-run direct loss "
              f"{expected:.4f}%", flush=True)
        results = {policy: run(layout, policy, state) for policy in POLICIES}
        got = results["direct"]["total_loss"]
        if abs(got - expected) > MATCH_TOLERANCE:
            raise SystemExit(f"STOP: direct loss {got:.6f}% does not match "
                             f"the test run's {expected:.6f}%")
        print(f"  direct loss reproduced: {got:.4f}%")
        if not all(r["views_agree"] for r in results.values()):
            raise SystemExit("STOP: the ACK views disagreed")

        loss_path = os.path.join(out, f"L{layout}_loss.csv")
        write_csv(loss_path, list(results.values()),
                  ["policy", "run_seed", "total_loss"] + list(CAUSES)
                  + ["c_distance_flown_mean", "views_agree"],
                  f"phase 2c test layout {layout}, near_gs D = 650, "
                  f"18M+7C, pictures only")
        with open(loss_path) as handle:
            saved = list(csv.DictReader(line for line in handle
                                        if not line.startswith("#")))
        print(f"  {'policy':>9} {'total':>6} "
              + " ".join(f"{c[:9]:>9}" for c in CAUSES) + "   C flown")
        for row in saved:
            print(f"  {row['policy']:>9} {float(row['total_loss']):6.2f} "
                  + " ".join(f"{float(row[c]):9.2f}" for c in CAUSES)
                  + f"   {float(row['c_distance_flown_mean']):5.0f} m")

        recs = {p: without_links(results[p]["rec"]) for p in POLICIES}
        title = f"Test layout {layout}, D = 650 m, run seed " \
                f"{results['direct']['run_seed']}"
        for policy in POLICIES:
            path = os.path.join(out, f"L{layout}_{policy}.gif")
            gif(recs[policy], f"{title}\n{policy}", path)
            print(f"  wrote {path}", flush=True)
        path = os.path.join(out, f"L{layout}_snapshots.png")
        snapshots(recs, title, path)
        print(f"  wrote {path}")
        path = os.path.join(out, f"L{layout}_c_distance.png")
        c_distance_plot(recs, title, path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
