"""
diagnose_1b_band.py — A close look at one band of the rl policy's predictions.

Read-only. No model is changed and no config is written: the traffic load is
set per run and restored.

Describes every rl decision whose CHOSEN option carried a predicted delivery
chance inside a given band (0.4-0.5 by default), over the 20 TEST layouts at
200 ms with the seed-0 best-validation model:

    * the chosen option's queue fill, and how often it was completely full
    * the chosen option's channel loss
    * how many other options were on the table, and the best prediction
      among them
    * how often EVERY option was completely full

"Queue fill" is the length of the deciding drone's own queue for that link,
before the packet joins it, divided by config.QUEUE_CAPACITY. A fill of 1.0
means that queue was at capacity.

Usage:
    python scripts/diagnose_1b_band.py
    python scripts/diagnose_1b_band.py --band 0.9 1.0
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.drone import Drone, NextHop
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.stamp import settings_stamp
from scripts.train_1b import TEST_SEEDS

LOAD_MS = 200
FULL = 1.0 - 1e-9        # queue_fill at or above this means the queue is full


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Describe one band of the rl policy's predictions."
    )
    parser.add_argument("--band", type=float, nargs=2, default=[0.4, 0.5],
                        metavar=("LOW", "HIGH"),
                        help="Predicted-value band to describe (default 0.4 0.5).")
    parser.add_argument("--init-seed", type=int, default=0,
                        help="Which trained rl model to use (default 0).")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the trained models live.")
    return parser.parse_args()


def _build_router(state: Dict[str, Any], run_seed: int, band: Tuple[float, float]):
    """Return an rl router that records decisions inside *band*.

    Built inside the worker so the subclass is defined where torch is already
    imported, and so nothing in agents/ has to change.

    Args:
        state:    The trained weights.
        run_seed: The run's channel/traffic seed.
        band:     ``(low, high)`` predicted-value band to record.

    Returns:
        A router carrying the band's running totals.
    """
    import torch

    from agents.rl_router import RLRouter

    low, high = band

    class BandRouter(RLRouter):
        """rl, instrumented to describe decisions inside one band."""

        def __init__(self) -> None:
            super().__init__(run_seed=run_seed, training=False)
            self.net.load_state_dict(state)
            self.total = 0
            self.in_band = 0
            self.fill_sum = 0.0
            self.loss_sum = 0.0
            self.chosen_full = 0
            self.other_count_sum = 0
            self.best_other_sum = 0.0
            self.with_others = 0
            self.only_option = 0
            self.all_full = 0

        def _choose(self, drone: Drone, keys, feature_rows) -> NextHop:
            """Choose as rl does, recording the ones that land in the band."""
            with torch.no_grad():
                scores = self.net(
                    torch.tensor(feature_rows, dtype=torch.float32)
                )
            best = int(torch.argmax(scores).item())
            prediction = float(scores[best].item())

            self.total += 1
            if low <= prediction < high:
                self.in_band += 1
                _, fill, channel_loss = feature_rows[best]
                self.fill_sum += fill
                self.loss_sum += channel_loss
                if fill >= FULL:
                    self.chosen_full += 1
                if all(row[1] >= FULL for row in feature_rows):
                    self.all_full += 1

                others = [float(scores[i].item())
                          for i in range(len(keys)) if i != best]
                self.other_count_sum += len(others)
                if others:
                    self.with_others += 1
                    self.best_other_sum += max(others)
                else:
                    self.only_option += 1

            return keys[best]

    return BandRouter()


def _run(args: Tuple) -> Dict[str, Any]:
    """Worker: run one layout and return its band totals.

    Args:
        args: ``(placement_seed, state, band)``.

    Returns:
        The running totals for that layout.
    """
    placement_seed, state, band = args

    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(round(LOAD_MS / 1000.0 / config.TIMESTEP))
    try:
        router = _build_router(state, placement_seed, band)
        env = FANETEnv(routing="rl", log_path=os.devnull,
                       placement_seed=placement_seed, run_seed=placement_seed,
                       router=router)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        return {
            "total": router.total, "in_band": router.in_band,
            "fill_sum": router.fill_sum, "loss_sum": router.loss_sum,
            "chosen_full": router.chosen_full,
            "other_count_sum": router.other_count_sum,
            "best_other_sum": router.best_other_sum,
            "with_others": router.with_others,
            "only_option": router.only_option,
            "all_full": router.all_full,
            "views_agree": not metrics["discrepancies"],
        }
    finally:
        config.PACKET_INTERVAL_STEPS = previous


def main() -> None:
    """Run the band diagnostic and print its table."""
    args = parse_args()
    band = (args.band[0], args.band[1])

    import torch

    path = os.path.join(args.out_dir, "models", f"rl_seed{args.init_seed}.pt")
    if not os.path.exists(path):
        print(f"STOP: no trained model at {path}")
        raise SystemExit(1)
    state = torch.load(path, weights_only=True)

    jobs = [(seed, state, band) for seed in TEST_SEEDS]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            runs = list(pool.map(_run, jobs))
    else:
        runs = [_run(job) for job in jobs]

    if not all(run["views_agree"] for run in runs):
        print("STOP: views disagreed with ground truth")
        raise SystemExit(1)

    def total(key: str) -> float:
        """Sum one field over every layout."""
        return sum(run[key] for run in runs)

    decisions = total("total")
    in_band = int(total("in_band"))

    print(f"rl decisions with a chosen-option prediction in "
          f"[{band[0]:.1f}, {band[1]:.1f})")
    print(f"  seed-{args.init_seed} model, {len(TEST_SEEDS)} TEST layouts, "
          f"{LOAD_MS} ms")
    print(f"  {settings_stamp(loads_ms=[LOAD_MS], placements=len(TEST_SEEDS))}")
    print()

    if in_band == 0:
        print("  no decisions fell in this band")
        return

    with_others = int(total("with_others"))
    print("  " + "-" * 70)
    print(f"  {'decisions in band':<44}{in_band:>13}")
    print(f"  {'as a share of all rl decisions':<44}"
          f"{100*in_band/decisions:>12.2f}%")
    print("  " + "-" * 70)
    print(f"  {'mean queue fill of the chosen option':<44}"
          f"{100*total('fill_sum')/in_band:>12.2f}%")
    print(f"  {'share where the chosen queue was full':<44}"
          f"{100*total('chosen_full')/in_band:>12.2f}%")
    print(f"  {'mean channel loss of the chosen option':<44}"
          f"{100*total('loss_sum')/in_band:>12.2f}%")
    print("  " + "-" * 70)
    print(f"  {'mean number of OTHER options available':<44}"
          f"{total('other_count_sum')/in_band:>13.2f}")
    print(f"  {'decisions with no other option':<44}"
          f"{int(total('only_option')):>13}")
    if with_others:
        print(f"  {'mean best prediction among the others':<44}"
              f"{total('best_other_sum')/with_others:>13.3f}")
    print("  " + "-" * 70)
    print(f"  {'share where EVERY option was full':<44}"
          f"{100*total('all_full')/in_band:>12.2f}%")
    print("  " + "-" * 70)


if __name__ == "__main__":
    main()
