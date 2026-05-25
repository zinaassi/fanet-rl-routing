"""
main.py — Entry point for the FANET phase-1 simulator.

Runs one full episode with:
    - NUM_M_DRONES mission drones following random waypoint paths
    - NUM_C_DRONES communication drones following random waypoint paths
    - Greedy geographic routing as the baseline
    - All standard metrics printed at episode end
    - A matplotlib animation saved to episode.gif (or shown live)

Usage:
    python main.py                   # run & save animation to episode.gif
    python main.py --no-anim         # run without animation
    python main.py --show            # run with interactive animation window
    python main.py --routing qroute  # use Q-routing baseline instead
"""

from __future__ import annotations

import argparse
import sys
import time

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.utils.metrics import compute_episode_metrics, print_episode_summary
from fanet_sim.utils.visualization import FANETVisualizer


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
    return parser.parse_args()


def run_headless(env: FANETEnv) -> None:
    """Run a full episode without animation and report progress every 100 steps.

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
            pdr = dlv / gen if gen > 0 else 0.0
            print(
                f"  [{pct:5.1f}%] step={step+1:>5}  "
                f"gen={gen:>5}  dlv={dlv:>5}  PDR={pdr:.3f}  "
                f"elapsed={elapsed:.1f}s"
            )
        if all(dones.values()):
            break

    elapsed = time.perf_counter() - t0
    print(f"Episode finished in {elapsed:.2f}s\n")


def main() -> None:
    """Entrypoint: parse args, run episode, print metrics, optionally animate."""
    args = parse_args()

    routing = "q-routing" if args.routing == "qroute" else "greedy"
    if args.steps:
        config.MAX_STEPS = args.steps

    print("=" * 52)
    print("  FANET Simulator — Phase 1")
    print("=" * 52)
    print(f"  M-drones  : {config.NUM_M_DRONES}")
    print(f"  C-drones  : {config.NUM_C_DRONES}")
    print(f"  Steps     : {config.MAX_STEPS}")
    print(f"  Routing   : {routing}")
    print(f"  Seed      : {config.RANDOM_SEED}")
    print("=" * 52 + "\n")

    env = FANETEnv(routing=routing)
    obs = env.reset()
    print(f"Environment reset — {env.num_drones} drones created.\n")

    if args.no_anim:
        # ----------------------------------------------------------------
        # Headless run
        # ----------------------------------------------------------------
        run_headless(env)
        metrics = compute_episode_metrics(env)
        print_episode_summary(metrics)

    elif args.show:
        # ----------------------------------------------------------------
        # Interactive animation (blocks until window is closed)
        # ----------------------------------------------------------------
        vis = FANETVisualizer(env, interval_ms=50)
        print("Showing interactive animation …")
        vis.show()
        metrics = compute_episode_metrics(env)
        print_episode_summary(metrics)

    else:
        # ----------------------------------------------------------------
        # Save animation to GIF
        # ----------------------------------------------------------------
        vis = FANETVisualizer(env, interval_ms=50)
        vis.animate(save_path="episode.gif")
        metrics = compute_episode_metrics(env)
        print_episode_summary(metrics)


if __name__ == "__main__":
    main()
