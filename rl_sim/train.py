"""
train.py — PPO training entry point for the two FANET drone policies.

Trains, with PPO, the two policies that already exist in the simulator:

  * the **K-link selection** MLP on EVERY drone — its action each step is which
    K of its in-range candidates to keep as links. Reward: +1 when a packet that
    originated at the drone is delivered to the ground station, -1 when one of
    its packets is dropped (config.LINK_REWARD_*).
  * the **topology** movement MLP on every **C-drone** — its action is a
    [dx, dy] displacement. Reward: the local topology reward already implemented
    in ``fanet_sim/envs/topology_agent.py`` (gain in neighbours + improvement in
    mean link quality − motion energy).

Routing is left untouched (greedy geographic forwarding).

Each drone trains its OWN policy independently; weights, critics, and optimiser
state live in a :class:`PolicyBank` that persists across episodes. Every episode
runs until all M-drones arrive or MAX_STEPS is reached; both policies are then
PPO-updated, and the episode's PDR is printed so improvement is visible. Trained
weights are saved at the end.

Usage:
    python train.py                          # train with config defaults
    python train.py --episodes 100 --steps 400
    python train.py --save my_policies.pt --seed 7
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.rl.ppo import PolicyBank, PPOConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="PPO training for FANET policies")
    parser.add_argument(
        "--episodes", type=int, default=config.TRAIN_EPISODES,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--steps", type=int, default=config.TRAIN_MAX_STEPS,
        help="Max steps per episode (overrides config.MAX_STEPS for training).",
    )
    parser.add_argument(
        "--seed", type=int, default=config.RANDOM_SEED,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--vary-seed", action="store_true",
        help="Use a different scenario each episode (seed + episode). By "
             "default every episode replays the SAME scenario so the per-episode "
             "PDR curve cleanly reflects learning rather than scenario luck.",
    )
    parser.add_argument(
        "--save", type=str, default=config.POLICY_SAVE_PATH,
        help="Path to write the trained policy weights.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint to CONTINUE training from (loads weights and "
             "Adam optimiser state). Omit to start from random weights.",
    )
    parser.add_argument(
        "--log-dir", type=str, default=os.path.join(config.LOG_DIR, "train"),
        help="Directory for per-episode event logs.",
    )
    return parser.parse_args()


def run_episode(env: FANETEnv, max_steps: int) -> None:
    """Run one full training episode to termination.

    Args:
        env:       A freshly reset training environment.
        max_steps: Hard cap on steps (the env also stops once all M-drones
                   arrive).
    """
    for _ in range(max_steps):
        _, _, dones, _ = env.step()
        if all(dones.values()):
            break
    env.close_logger()


def main() -> None:
    """Train both policies with PPO across many episodes and save the weights."""
    args = parse_args()

    # Training episodes use the requested step budget.
    config.MAX_STEPS = args.steps
    os.makedirs(args.log_dir, exist_ok=True)

    ppo_cfg = PPOConfig()
    bank = PolicyBank(config.NUM_M_DRONES, config.NUM_C_DRONES, ppo_cfg)
    if args.resume:
        bank.load(args.resume)  # warm-start weights + optimiser state

    print("=" * 64)
    print("  FANET PPO Training")
    print("=" * 64)
    print(f"  drones        : {config.NUM_M_DRONES} M + {config.NUM_C_DRONES} C")
    print(f"  resume from   : {args.resume if args.resume else '(none — random init)'}")
    print(f"  episodes      : {args.episodes}")
    print(f"  max steps     : {args.steps}")
    print(f"  base seed     : {args.seed}")
    print(f"  link reward   : +{config.LINK_REWARD_DELIVERED:g} / "
          f"{config.LINK_REWARD_DROPPED:g}  (M-drones: source packets | "
          f"C-drones: relay forwarded vs. voided)")
    print(f"  lr={ppo_cfg.lr:g}  gamma={ppo_cfg.gamma:g}  lam={ppo_cfg.lam:g}  "
          f"clip={ppo_cfg.clip:g}  epochs={ppo_cfg.epochs}")
    print(f"  save to       : {args.save}")
    print("=" * 64)

    t0 = time.perf_counter()
    for ep in range(args.episodes):
        episode_seed = args.seed + ep if args.vary_seed else args.seed
        env = FANETEnv(
            routing="greedy",
            training=True,
            policy_bank=bank,
            episode_id=ep,
            seed=episode_seed,
            log_path=os.path.join(args.log_dir, f"train_ep_{ep}.jsonl"),
        )
        env.reset()
        # reset() seeds torch from the (possibly fixed) scenario seed; re-seed
        # it here so the PPO exploration noise still varies every episode even
        # when the scenario is held fixed.
        torch.manual_seed(args.seed + ep)
        run_episode(env, args.steps)

        # PPO updates — each drone updates its own policy independently.
        link_signal = bank.update_link(env.get_link_rollouts())
        topo_signal = bank.update_topology(env.get_topology_rollouts())

        generated = len(env.all_packets)
        delivered = len(env.delivered)
        pdr = delivered / generated if generated > 0 else 0.0
        elapsed = time.perf_counter() - t0
        print(
            f"  episode {ep:4d} | steps {env.step_count:4d} | "
            f"gen {generated:5d} dlv {delivered:5d} | PDR {pdr:6.3f} | "
            f"|adv| link {link_signal:.3f} topo {topo_signal:.3f} | "
            f"{elapsed:6.1f}s"
        )

    bank.save(args.save)
    print("=" * 64)
    print(f"  training complete — weights saved to {args.save}")
    print("=" * 64)


if __name__ == "__main__":
    main()
