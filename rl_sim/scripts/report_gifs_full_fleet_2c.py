"""
report_gifs_full_fleet_2c.py — Full-fleet GIFs and stills for report section 11.

Pictures only: no training, and no number in the report comes from here.

1. A full-fleet TEST episode (out/2c_r2_test.csv, set "full", layouts
   601-650): the layout whose loss under the RL agent trained on the full
   fleet ("control", models/2c/c_actor_r1_control.pt) is closest to that
   agent's median (a tie goes to the lower layout). That episode is re-run
   under control and midpoint with the test run's own run seed, exactly as
   evaluate_r2_test.py ran it (train_c_ppo_round2.run_episode, normal plan),
   with every step recorded. Each policy's total loss must reproduce its row
   in 2c_r2_test.csv to MATCH_TOLERANCE, or the script stops.

2. The full-fleet edge cases, report cases 6-9 = script scenarios 9, 10, 12,
   13, drawn from the recordings saved while Table 13's runs were scored:
   the RL agent before varied training (out/2c_edge_bigfleet_r2_control/rec,
   c_actor_r1_control.pt) and after it (out/2c_edge_bigfleet_r2_B/rec,
   c_actor_r2_B.pt). Nothing is re-simulated. One run seed per case: the
   seed whose two losses (before, after) are together closest to the two
   five-run means, a tie to the lower seed. Case 6 keeps seed 9900, which
   Figure 9's GIFs already use (chosen earlier as typical of the agent before
   varied training: 86.9% on 3 of its 5 seeds).

Every picture uses plot_edge_cases_2c's big-fleet drawing: links off, flying
M-drones in their own colour, C-drone trails, and each flying M-drone's
delivered-packet route as a thin line; half speed (10 fps, one frame per
0.5 s). Stills at t = 30, 60 and 95 s, one row per GIF.

Writes, stamped, under out/2c_report_gifs/full_fleet/:
    test_L{layout}_{control,midpoint}.gif, test_L{layout}_snapshots.png,
    test_L{layout}_loss.csv, rec/test_L{layout}_{policy}.npz
    case{6..9}_{before,after}.gif, case{6..9}_snapshots.png
and, with --report, copies the new GIFs and stills into
../docs/report/figures/ as 2c_full_test_{midpoint,rl}.gif,
2c_full_test_snapshots.png, 2c_case{7,8,9}_{before,after}.gif and
2c_case{6..9}_snapshots.png (case 6's GIFs are already there).

Usage:
    python scripts/report_gifs_full_fleet_2c.py [--report]
"""

from __future__ import annotations

import argparse
import os
import shutil
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
from scripts.edge_cases_2c import _record_step
from scripts.metrics_1a import compute_metrics
from scripts.plot_edge_cases_2c import (BIG_FPS, FRAME_EVERY, big_legend,
                                        big_setup_axis, draw_big_step)
from scripts.stamp import settings_stamp
from scripts.train_c_ppo import (ROUTING, SamplingAgent, _restore, _world,
                                 write_csv)
from scripts.train_c_ppo_round1 import read_csv
from scripts.train_c_ppo_round2 import CONTROL_ACTOR, NORMAL_PLAN

B_ACTOR: str = os.path.join("models", "2c", "c_actor_r2_B.pt")
TEST_CSV: str = "2c_r2_test.csv"
MATCH_TOLERANCE: float = 1e-9
SNAPSHOT_TIMES_S: Tuple[float, ...] = (30.0, 60.0, 95.0)
FIGURES_DIR: str = os.path.join("..", "docs", "report", "figures")

#: Report case -> (script scenario, what happens, as Table 13 says it).
CASES: Dict[int, Tuple[int, str]] = {
    6: (9, "full fleet; 1 M-drone flies to a corner"),
    7: (10, "full fleet; 3 M-drones fly to three different corners"),
    8: (12, "full fleet; 1 M-drone flies to a corner and back"),
    9: (13, "full fleet; 9 M-drones fly together to one side"),
}
#: Case 6 keeps the seed Figure 9's GIFs were drawn at.
FIXED_SEED: Dict[int, int] = {6: 9900}
EDGE_SEEDS: range = range(9900, 9905)
EDGE_SOURCES: Dict[str, Tuple[str, str, str]] = {
    # name -> (recording folder under out/, policy label in it, model)
    "before": ("2c_edge_bigfleet_r2_control", "control", CONTROL_ACTOR),
    "after": ("2c_edge_bigfleet_r2_B", "B", B_ACTOR),
}
TITLES = {"control": "RL agent (trained on the full fleet)",
          "midpoint": "midpoint rule",
          "before": "RL agent before varied training",
          "after": "RL agent after varied training"}


# ---------------------------------------------------------------------------
# Recordings
# ---------------------------------------------------------------------------

def index_links(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Split the flat link arrays by step, as plot_edge_cases_2c.load does."""
    steps = len(rec["pos"])
    links, gs_links = rec["link"], rec["gs_link"]
    rec["links_at"] = np.split(links, np.searchsorted(links[:, 0],
                                                      np.arange(1, steps)))
    rec["gs_at"] = np.split(gs_links, np.searchsorted(gs_links[:, 0],
                                                      np.arange(1, steps)))
    return rec


def load_npz(path: str) -> Dict[str, Any]:
    return index_links(dict(np.load(path)))


def pick_test_layout(out_dir: str) -> Tuple[int, int, float, float]:
    """(layout, run seed, control loss, median) of the median control run."""
    rows = [r for r in read_csv(os.path.join(out_dir, TEST_CSV))
            if r["set"] == "full" and r["policy"] == "control"]
    loss = {int(r["placement_seed"]): float(r["total_loss"]) for r in rows}
    seed = {int(r["placement_seed"]): int(r["run_seed"]) for r in rows}
    median = statistics.median(loss.values())
    layout = min(sorted(loss), key=lambda k: abs(loss[k] - median))
    return layout, seed[layout], loss[layout], median


def record_test_episode(layout: int, run_seed: int, policy: str,
                        state: Any) -> Dict[str, Any]:
    """train_c_ppo_round2.run_episode for the normal plan, recorded.

    Same seeding, world, fleet, agent and step loop as the test run; the
    normal plan has no flyer overrides and no return leg."""
    import torch

    plan = NORMAL_PLAN
    torch.set_num_threads(1)
    torch.manual_seed(int(np.random.SeedSequence(
        [run_seed, config.AGENT_STREAM_TAG]).generate_state(1)[0]))
    previous = _world(True, plan["distance"])
    counts = (config.NUM_M_DRONES, config.NUM_C_DRONES)
    config.NUM_M_DRONES, config.NUM_C_DRONES = plan["n_m"], plan["n_c"]
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
        n_m = plan["n_m"]
        delivered = [p for p in env.delivered if p.source_id in range(n_m)]
        rec = {
            "pos": np.array(store["pos"]),
            "types": np.array([d.drone_type for d in env.drones]),
            "gs": np.asarray(env.gs_position, dtype=np.float64),
            "link": np.array(store["link"], dtype=np.float64).reshape(-1, 4),
            "gs_link": np.array(store["gs"], dtype=np.float64).reshape(-1, 3),
            "pkt_source": np.array([p.source_id for p in delivered],
                                   dtype=int),
            "pkt_created": np.array([p.created_at for p in delivered],
                                    dtype=int),
            "pkt_path": np.array(["-".join(str(h) for h in p.path)
                                  for p in delivered], dtype=str),
            "m_end": np.array([d.end_point for d in env.drones
                               if d.drone_type == "M"]),
            # In the full-fleet mission every M-drone flies.
            "m_flies": np.ones(n_m, dtype=bool),
        }
        return {"total_loss": 100.0 * (truth["total_loss"] or 0.0),
                "views_agree": not metrics["discrepancies"], "rec": rec}
    finally:
        config.NUM_M_DRONES, config.NUM_C_DRONES = counts
        _restore(previous)


def edge_seed(out_dir: str, case: int) -> int:
    """The case's run seed: fixed for case 6, else the most typical one."""
    if case in FIXED_SEED:
        return FIXED_SEED[case]
    scenario = CASES[case][0]
    loss: Dict[Tuple[str, int], float] = {}
    for name, (folder, label, _) in EDGE_SOURCES.items():
        for r in read_csv(os.path.join(out_dir, folder, "edge_runs.csv")):
            if int(r["scenario"]) == scenario and r["policy"] == label:
                loss[(name, int(r["run_seed"]))] = float(r["loss_flying"])
    mean = {n: statistics.fmean(loss[(n, s)] for s in EDGE_SEEDS)
            for n in EDGE_SOURCES}
    return min(EDGE_SEEDS, key=lambda s: (sum(abs(loss[(n, s)] - mean[n])
                                              for n in EDGE_SOURCES), s))


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def gif(rec: Dict[str, Any], title: str, stamp_note: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    big_setup_axis(ax, rec)
    ax.set_title(title, fontsize=8)
    ax.legend(handles=big_legend(), loc="lower right", fontsize=5.0,
              framealpha=0.8)
    fig.text(0.5, 0.005, settings_stamp(extra=stamp_note), ha="center",
             fontsize=3.5, color="#666666", wrap=True)
    artists: Dict[str, Any] = {}

    def update(step: int):
        draw_big_step(ax, rec, step, artists)
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=list(range(0, len(rec["pos"]), FRAME_EVERY)),
        blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=BIG_FPS), dpi=80)
    plt.close(fig)


def snapshots(recs: List[Tuple[str, Dict[str, Any]]], title: str,
              stamp_note: str, path: str) -> None:
    fig, axes = plt.subplots(len(recs), len(SNAPSHOT_TIMES_S),
                             figsize=(8.4, 3.0 * len(recs)), squeeze=False)
    for row, (label, rec) in enumerate(recs):
        for col, t in enumerate(SNAPSHOT_TIMES_S):
            ax = axes[row, col]
            big_setup_axis(ax, rec)
            draw_big_step(ax, rec, int(round(t / config.TIMESTEP)), {})
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.legend(handles=big_legend(), loc="lower center", ncol=4,
               fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.005, settings_stamp(extra=stamp_note), ha="center",
             fontsize=4.0, color="#666666", wrap=True)
    fig.subplots_adjust(bottom=0.13, top=0.92, hspace=0.25)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--out-dir", type=str, default="out")
    parser.add_argument("--report", action="store_true",
                        help="also copy into ../docs/report/figures/")
    args = parser.parse_args()
    out = os.path.join(args.out_dir, "2c_report_gifs", "full_fleet")
    os.makedirs(os.path.join(out, "rec"), exist_ok=True)
    copies: Dict[str, str] = {}

    print("Models:")
    print(f"  RL agent trained on the full fleet: "
          f"{os.path.join(args.out_dir, CONTROL_ACTOR)}")
    print(f"  RL agent after varied training:     "
          f"{os.path.join(args.out_dir, B_ACTOR)}")

    # 1. The median full-fleet test episode.
    layout, run_seed, expected, median = pick_test_layout(args.out_dir)
    print(f"\nTest layout {layout} (run seed {run_seed}): control loss "
          f"{expected:.4f}%, closest to the median {median:.4f}% of 601-650")
    saved = {r["policy"]: float(r["total_loss"])
             for r in read_csv(os.path.join(args.out_dir, TEST_CSV))
             if r["set"] == "full" and int(r["placement_seed"]) == layout}
    states = {"control": torch.load(os.path.join(args.out_dir, CONTROL_ACTOR),
                                    weights_only=True),
              "midpoint": None}
    recs: Dict[str, Dict[str, Any]] = {}
    rows = []
    for policy in ("midpoint", "control"):
        result = record_test_episode(layout, run_seed, policy, states[policy])
        got, want = result["total_loss"], saved[policy]
        if abs(got - want) > MATCH_TOLERANCE or not result["views_agree"]:
            raise SystemExit(f"STOP: {policy} loss {got:.9f}% does not match "
                             f"the test run's {want:.9f}% (or views disagree)")
        print(f"  {policy:>8}: loss reproduced, {got:.4f}%")
        rows.append({"policy": policy, "placement_seed": layout,
                     "run_seed": run_seed, "total_loss": got,
                     "test_csv_total_loss": want})
        np.savez_compressed(os.path.join(out, "rec",
                                         f"test_L{layout}_{policy}.npz"),
                            **result["rec"])
        recs[policy] = index_links(result["rec"])
    write_csv(os.path.join(out, f"test_L{layout}_loss.csv"), rows, list(rows[0]),
              f"phase 2c full-fleet test layout {layout}, run seed "
              f"{run_seed}, re-run for pictures; must equal {TEST_CSV}")
    note = f"test layout {layout}, run seed {run_seed}"
    for policy in ("midpoint", "control"):
        path = os.path.join(out, f"test_L{layout}_{policy}.gif")
        gif(recs[policy], f"Full fleet, test layout {layout}: "
            f"{TITLES[policy]}\n{saved[policy]:.1f}% of packets lost",
            note, path)
        print(f"  wrote {path} ({size_mb(path):.2f} MB)")
        copies[path] = f"2c_full_test_{'rl' if policy == 'control' else policy}.gif"
    path = os.path.join(out, f"test_L{layout}_snapshots.png")
    snapshots([(TITLES[p], recs[p]) for p in ("midpoint", "control")],
              f"Full fleet, test layout {layout}, t = 30, 60, 95 s", note,
              path)
    print(f"  wrote {path}")
    copies[path] = "2c_full_test_snapshots.png"

    # 2. The full-fleet edge cases, from the scored runs' recordings.
    print("\nReport case -> script scenario (Table 13):")
    for case, (scenario, text) in CASES.items():
        print(f"  case {case} = scenario {scenario}: {text}")
    for case, (scenario, text) in CASES.items():
        seed = edge_seed(args.out_dir, case)
        print(f"\nCase {case} (scenario {scenario}), run seed {seed}")
        case_recs = []
        for name, (folder, label, model) in EDGE_SOURCES.items():
            rec_path = os.path.join(args.out_dir, folder, "rec",
                                    f"s{scenario}_{label}_{seed}.npz")
            loss = [float(r["loss_flying"]) for r in read_csv(
                os.path.join(args.out_dir, folder, "edge_runs.csv"))
                if int(r["scenario"]) == scenario and r["policy"] == label
                and int(r["run_seed"]) == seed][0]
            print(f"  {name:>6}: {rec_path} (model {model}), flying "
                  f"M-drones lose {loss:.1f}%")
            rec = load_npz(rec_path)
            case_recs.append((TITLES[name], rec))
            path = os.path.join(out, f"case{case}_{name}.gif")
            gif(rec, f"Edge case {case}: {text}\n{TITLES[name]}, run seed "
                f"{seed}: flying M-drones lose {loss:.1f}%",
                f"edge case {case} (scenario {scenario})", path)
            print(f"  wrote {path} ({size_mb(path):.2f} MB)")
            if case not in FIXED_SEED:
                copies[path] = f"2c_case{case}_{name}.gif"
        path = os.path.join(out, f"case{case}_snapshots.png")
        snapshots(case_recs, f"Edge case {case}: {text}; run seed {seed}; "
                  f"t = 30, 60, 95 s",
                  f"edge case {case} (scenario {scenario})", path)
        print(f"  wrote {path}")
        copies[path] = f"2c_case{case}_snapshots.png"

    if args.report:
        print("\nCopies for the report:")
        for source, name in copies.items():
            shutil.copy2(source, os.path.join(FIGURES_DIR, name))
            print(f"  {source} -> {os.path.join(FIGURES_DIR, name)}")


if __name__ == "__main__":
    main()
