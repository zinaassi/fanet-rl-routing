"""
eval.py — run the simulator with trained (or random) policies and report metrics.

This is the bridge between ``train.py`` and the phase-1 simulator. ``train.py``
saves a weights checkpoint; this script loads it into a normal (deterministic,
non-training) environment, runs one full episode of greedy routing, and prints
the standard metrics via the Stage-2 analyser — exactly like ``main.py``, but
with the *trained* policies plugged in.

With ``--weights`` the K-link and topology MLPs use the trained weights (links =
deterministic top-K of the learned scores; C-drone moves = the learned mean).
Without ``--weights`` the policies are random, so you can A/B the same scenario:

    python eval.py --steps 300 --seed 42                       # random baseline
    python eval.py --steps 300 --seed 42 --weights trained_policies.pt

Routing is greedy in both cases (unchanged). Use the same --seed/--steps for a
fair before/after comparison.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.rl.ppo import PolicyBank, PPOConfig
from fanet_sim.utils.visualization import FANETVisualizer
from scripts.analyze import analyze, print_report, read_jsonl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained/random FANET policies in the simulator."
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to a checkpoint saved by train.py. Omit for random policies.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override MAX_STEPS from config.py.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed (scenario). Defaults to config.RANDOM_SEED.",
    )
    parser.add_argument(
        "--episode-id", type=int, default=0,
        help="Episode identifier (used in the log filename and records).",
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Path for the JSONL event log. Defaults to {LOG_DIR}/eval_{id}.jsonl.",
    )
    parser.add_argument(
        "--save-anim", type=str, default=None, metavar="PATH",
        help="Save a matplotlib animation of the run to PATH (e.g. eval.gif). "
             "Shows the trained C-drone movement and link choices.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show the animation in an interactive window instead of saving.",
    )
    return parser.parse_args()


def main() -> None:
    """Load policies (if any), run one episode, and print the metric report."""
    args = parse_args()
    if args.steps:
        config.MAX_STEPS = args.steps

    log_path = args.log or os.path.join(
        config.LOG_DIR, f"eval_{args.episode_id}.jsonl"
    )

    bank = None
    if args.weights:
        bank = PolicyBank(config.NUM_M_DRONES, config.NUM_C_DRONES, PPOConfig())
        bank.load(args.weights)

    print("=" * 56)
    print("  FANET Simulator — Policy Evaluation")
    print("=" * 56)
    print(f"  policies  : {'trained (' + args.weights + ')' if args.weights else 'random (untrained)'}")
    print(f"  drones    : {config.NUM_M_DRONES} M + {config.NUM_C_DRONES} C")
    print(f"  steps     : {config.MAX_STEPS}")
    print(f"  seed      : {args.seed if args.seed is not None else config.RANDOM_SEED}")
    print("=" * 56 + "\n")

    env = FANETEnv(
        routing="greedy",
        log_path=log_path,
        episode_id=args.episode_id,
        seed=args.seed,
        training=False,      # deterministic top-K links / mean moves
        policy_bank=bank,    # injects trained weights when provided
    )
    env.reset()

    t0 = time.perf_counter()
    if args.show or args.save_anim:
        # The visualiser drives env.step() itself, one call per frame.
        vis = FANETVisualizer(env, interval_ms=50)
        if args.show:
            print("Showing interactive animation …")
            vis.show()
        else:
            vis.animate(save_path=args.save_anim)
    else:
        for _ in range(config.MAX_STEPS):
            _, _, dones, _ = env.step()
            if all(dones.values()):
                break
    env.close_logger()
    print(f"Episode finished in {time.perf_counter() - t0:.2f}s\n")

    metrics = analyze(read_jsonl(Path(log_path)))
    print_report(metrics)


if __name__ == "__main__":
    main()
