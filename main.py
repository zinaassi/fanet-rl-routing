"""
main.py — Entry point for the FANET phase-1 simulator.

Runs one full episode with:
    - NUM_M_DRONES mission drones following random waypoint paths
    - NUM_C_DRONES communication drones following random waypoint paths
    - Greedy geographic routing as the baseline
    - Raw events logged to {LOG_DIR}/episode_{id}.jsonl
    - Stage-2 metrics printed by reading that log
    - A matplotlib animation saved to episode.gif (or shown live)

Usage:
    python main.py                   # run & save animation to episode.gif
    python main.py --no-anim         # run without animation
    python main.py --show            # run with interactive animation window
    python main.py --routing qroute  # use Q-routing baseline instead
    python main.py --log custom.jsonl   # override the event log path
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.utils.visualization import FANETVisualizer
from scripts.analyze import analyze, print_report, read_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="FANET Phase-1 Simulator")
    parser.add_argument(
        "--no-anim",
        action="store_true",
        help="Disable animation (faster headless run).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show animation interactively instead of saving to GIF.",
    )
    parser.add_argument(
        "--routing",
        choices=["greedy", "qroute"],
        default="greedy",
        help="Routing baseline: greedy (default) or qroute.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override MAX_STEPS from config.py.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=0,
        help="Episode identifier (used in the log filename and records).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to write the Stage-1 JSONL event log. "
             "Defaults to {LOG_DIR}/episode_{id}.jsonl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for this run. Defaults to config.RANDOM_SEED. "
             "Use a different seed to vary mobility / waypoint draws while "
             "keeping all other parameters fixed.",
    )
    return parser.parse_args()


def run_headless(env: FANETEnv) -> None:
    """Run a full episode without animation and report progress every 10%.

    Args:
        env: A reset FANETEnv instance.
    """
    max_steps = config.MAX_STEPS
    report_every = max(1, max_steps // 10)

    print(f"Running {max_steps} steps (headless) …")
    t0 = time.perf_counter()

    for step in range(max_steps):
        _, _, dones, _ = env.step()
        if (step + 1) % report_every == 0:
            pct = 100 * (step + 1) / max_steps
            elapsed = time.perf_counter() - t0
            gen = len(env.all_packets)
            dlv = len(env.delivered)
            running_pdr = dlv / gen if gen > 0 else 0.0
            print(
                f"  [{pct:5.1f}%] step={step+1:>5}  "
                f"gen={gen:>5}  dlv={dlv:>5}  pdr~={running_pdr:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )
        if all(dones.values()):
            break

    elapsed = time.perf_counter() - t0
    print(f"Episode finished in {elapsed:.2f}s\n")


def main() -> None:
    """Entry point: parse args, run episode, then run the Stage-2 analyser."""
    args = parse_args()

    routing = "q-routing" if args.routing == "qroute" else "greedy"
    if args.steps:
        config.MAX_STEPS = args.steps

    log_path = args.log or os.path.join(
        config.LOG_DIR, f"episode_{args.episode_id}.jsonl"
    )

    print("=" * 56)
    print("  FANET Simulator — Phase 1")
    print("=" * 56)
    print(f"  M-drones  : {config.NUM_M_DRONES}")
    print(f"  C-drones  : {config.NUM_C_DRONES}")
    print(f"  Steps     : {config.MAX_STEPS}")
    print(f"  Routing   : {routing}")
    print(f"  Seed      : {args.seed if args.seed is not None else config.RANDOM_SEED}")
    print(f"  Log file  : {log_path}")
    print("  Channel   : FSPL (free-space path loss)")
    print(f"    Pt={channel.PT_DBM:.1f} dBm  "
          f"Gt={channel.GT_DBI:.1f} dBi  "
          f"Gr={channel.GR_DBI:.1f} dBi  "
          f"f={channel.F_HZ/1e9:.2f} GHz")
    print(f"    sensitivity = {channel.RX_SENSITIVITY_DBM:.1f} dBm  "
          f"link budget = {channel.LINK_BUDGET_DB:.1f} dB")
    print(f"    => effective max link distance = "
          f"{channel.MAX_LINK_DISTANCE_M:.1f} m")
    print("=" * 56 + "\n")

    env = FANETEnv(
        routing=routing,
        log_path=log_path,
        episode_id=args.episode_id,
        seed=args.seed,
    )
    env.reset()
    print(f"Environment reset — {env.num_drones} drones created.\n")

    if args.no_anim:
        run_headless(env)
    elif args.show:
        vis = FANETVisualizer(env, interval_ms=50)
        print("Showing interactive animation …")
        vis.show()
    else:
        vis = FANETVisualizer(env, interval_ms=50)
        vis.animate(save_path="episode.gif")

    # Ensure the log file is flushed even if MAX_STEPS was not reached
    # (e.g. closed the interactive window early).
    env.close_logger()

    # Stage 2 — read the raw log and report metrics.
    print()
    metrics = analyze(read_jsonl(Path(log_path)))
    print_report(metrics)


if __name__ == "__main__":
    main()
