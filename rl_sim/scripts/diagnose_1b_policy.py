"""
diagnose_1b_policy.py — What the rl policy does, and how well it knows itself.

Read-only. No model is changed and no config is written: the traffic load is
set per run and restored. Uses the seed-0 best-validation model on the 20 TEST
layouts at 200 ms, except table 3 which uses the validation layouts.

Three tables:

1. AGREEMENT. At every rl decision, work out what "rate" would have chosen
   from exactly the same state — same delivery rates, same queues, same
   option set, no exploration on either side. Report how often they agree,
   and when they do not, the mean channel loss and queue fill of each one's
   pick.

2. CALIBRATION. Group rl decisions by the delivery chance the network
   predicted for the option it chose, and report how many fell in each band
   and what share of them actually arrived. Decisions still unresolved when
   the run ends are not counted — there is no outcome to compare against.

3. VALIDATION NOISE. Re-run the same model on the same 10 validation layouts
   under three different channel/traffic seeds, to show how much the reported
   figure moves for reasons that have nothing to do with the policy.

Usage:
    python scripts/diagnose_1b_policy.py
    python scripts/diagnose_1b_policy.py --init-seed 0 --workers 20
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
from fanet_sim.envs.packet import Packet
from scripts.metrics_1a import compute_metrics
from scripts.stamp import settings_stamp
from scripts.train_1b import TEST_SEEDS, VALIDATION_SEEDS

LOAD_MS = 200
BINS = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
RUN_SEED_OFFSETS = (0, 1000, 2000)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Phase-1b policy diagnostics.")
    parser.add_argument("--init-seed", type=int, default=0,
                        help="Which trained rl model to use (default 0).")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the trained models live.")
    return parser.parse_args()


def _build_router(state: Dict[str, Any], run_seed: int):
    """Return an rl router that records what rate would have done.

    Defined inside the worker so the subclasses are built where torch is
    already imported, and so nothing in agents/ has to change.

    Args:
        state:    The trained weights.
        run_seed: The run's channel/traffic seed.

    Returns:
        A router with ``agreement`` and ``calibration`` records on it.
    """
    import torch

    from agents.rl_router import RLRouter
    from agents.tracker import Tracker

    class RecordingTracker(Tracker):
        """A tracker that pairs each prediction with the outcome it got."""

        def __init__(self) -> None:
            super().__init__()
            self.predictions: Dict[int, List[float]] = {}
            self.resolved: List[Tuple[float, float]] = []

        def resolve(self, packet_id: int, delivered: bool):
            """Settle a packet and bank (prediction, outcome) for each decision."""
            predictions = self.predictions.pop(packet_id, [])
            examples = super().resolve(packet_id, delivered)
            target = 1.0 if delivered else 0.0
            self.resolved.extend((p, target) for p in predictions)
            return examples

        def reset(self) -> None:
            """Clear everything, including the prediction records."""
            super().reset()
            self.predictions = {}
            self.resolved = []

    class ComparingRouter(RLRouter):
        """rl, but it also works out what rate would have chosen."""

        def __init__(self) -> None:
            super().__init__(run_seed=run_seed, tracker=RecordingTracker(),
                             training=False)
            self.net.load_state_dict(state)
            self.agree = 0
            self.decisions = 0
            # Sums over disagreements, for the means.
            self.rl_loss_sum = 0.0
            self.rl_fill_sum = 0.0
            self.rate_loss_sum = 0.0
            self.rate_fill_sum = 0.0
            self._last_prediction: Optional[float] = None

        def _rate_choice(self, drone: Drone, keys, feature_rows) -> NextHop:
            """What RateRouter._choose would return with no exploration."""
            def rank(index: int):
                delivery_rate, _, channel_loss = feature_rows[index]
                queue_full = 1.0 if drone.queue_is_full(keys[index]) else 0.0
                return (-(delivery_rate * (1.0 - queue_full)), channel_loss)

            return keys[min(range(len(keys)), key=rank)]

        def _choose(self, drone: Drone, keys, feature_rows) -> NextHop:
            """Choose as rl does, and record what rate would have done."""
            with torch.no_grad():
                scores = self.net(
                    torch.tensor(feature_rows, dtype=torch.float32)
                )
            best = int(torch.argmax(scores).item())
            self._last_prediction = float(scores[best].item())

            chosen = keys[best]
            alternative = self._rate_choice(drone, keys, feature_rows)

            self.decisions += 1
            if chosen == alternative:
                self.agree += 1
            else:
                _, rl_fill, rl_loss = feature_rows[best]
                _, rate_fill, rate_loss = feature_rows[keys.index(alternative)]
                self.rl_loss_sum += rl_loss
                self.rl_fill_sum += rl_fill
                self.rate_loss_sum += rate_loss
                self.rate_fill_sum += rate_fill
            return chosen

        def select_next_hop(self, drone, candidates, pkt: Packet, came_from):
            """Route, then file the prediction beside the decision."""
            self._last_prediction = None
            chosen = super().select_next_hop(drone, candidates, pkt, came_from)
            if chosen is not None and self._last_prediction is not None:
                self.tracker.predictions.setdefault(
                    pkt.packet_id, []
                ).append(self._last_prediction)
            return chosen

    return ComparingRouter()


def _run(args: Tuple) -> Dict[str, Any]:
    """Worker: run one layout and aggregate its records.

    Args:
        args: ``(placement_seed, run_seed, state)``.

    Returns:
        Agreement counts, calibration bins and the run's total loss.
    """
    placement_seed, run_seed, state = args

    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(round(LOAD_MS / 1000.0 / config.TIMESTEP))
    try:
        router = _build_router(state, run_seed)
        env = FANETEnv(routing="rl", log_path=os.devnull,
                       placement_seed=placement_seed, run_seed=run_seed,
                       router=router)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)

        counts = [0] * len(BINS)
        delivered = [0] * len(BINS)
        for prediction, target in router.tracker.resolved:
            index = min(int(prediction * 10), len(BINS) - 1)
            counts[index] += 1
            delivered[index] += int(target)

        return {
            "decisions": router.decisions,
            "agree": router.agree,
            "rl_loss_sum": router.rl_loss_sum,
            "rl_fill_sum": router.rl_fill_sum,
            "rate_loss_sum": router.rate_loss_sum,
            "rate_fill_sum": router.rate_fill_sum,
            "bin_counts": counts,
            "bin_delivered": delivered,
            "total_loss": metrics["ground_truth"]["total_loss"] or 0.0,
            "views_agree": not metrics["discrepancies"],
        }
    finally:
        config.PACKET_INTERVAL_STEPS = previous


def sweep(
    seeds: Sequence[int], state: Dict[str, Any], workers: int,
    run_seed_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Run one model over a set of layouts.

    Args:
        seeds:           Placement seeds.
        state:           The trained weights.
        workers:         Parallel workers.
        run_seed_offset: Added to each placement seed to vary the
                         channel/traffic randomness while keeping the layout.

    Returns:
        One record per layout.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    jobs = [(seed, seed + run_seed_offset, state) for seed in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(_run, jobs))
    else:
        runs = [_run(job) for job in jobs]

    if not all(run["views_agree"] for run in runs):
        print("STOP: views disagreed with ground truth")
        raise SystemExit(1)
    return runs


def main() -> None:
    """Run the three diagnostics and print their tables."""
    args = parse_args()

    import torch

    path = os.path.join(args.out_dir, "models", f"rl_seed{args.init_seed}.pt")
    if not os.path.exists(path):
        print(f"STOP: no trained model at {path}")
        raise SystemExit(1)
    state = torch.load(path, weights_only=True)

    print(f"Phase-1b policy diagnostics — seed-{args.init_seed} model, "
          f"{LOAD_MS} ms")
    print(f"  {settings_stamp(loads_ms=[LOAD_MS], placements=len(TEST_SEEDS))}")

    runs = sweep(TEST_SEEDS, state, args.workers)

    # --- 1. agreement ---
    decisions = sum(r["decisions"] for r in runs)
    agree = sum(r["agree"] for r in runs)
    differ = decisions - agree
    print()
    print(f"  1. Agreement with rate, over {len(TEST_SEEDS)} TEST layouts")
    print("     Same delivery rates, queues and options; no exploration either side.")
    print("  " + "-" * 76)
    print(f"  {'decisions':>12}{'agree':>12}{'agree %':>11}{'differ':>12}"
          f"{'differ %':>11}")
    print(f"  {decisions:>12}{agree:>12}{100*agree/decisions:>10.2f}%"
          f"{differ:>12}{100*differ/decisions:>10.2f}%")
    print("  " + "-" * 76)
    if differ:
        print()
        print("     Where they differ, the choice each one made:")
        print("  " + "-" * 60)
        print(f"  {'chose':<10}{'mean channel loss':>22}{'mean queue fill':>20}")
        print(f"  {'rl':<10}"
              f"{100*sum(r['rl_loss_sum'] for r in runs)/differ:>21.2f}%"
              f"{100*sum(r['rl_fill_sum'] for r in runs)/differ:>19.2f}%")
        print(f"  {'rate':<10}"
              f"{100*sum(r['rate_loss_sum'] for r in runs)/differ:>21.2f}%"
              f"{100*sum(r['rate_fill_sum'] for r in runs)/differ:>19.2f}%")
        print("  " + "-" * 60)

    # --- 2. calibration ---
    print()
    print("  2. Calibration — predicted delivery chance against what happened")
    print("     Decisions still in flight at the end of a run are not counted.")
    print("  " + "-" * 72)
    print(f"  {'predicted':<14}{'decisions':>14}{'share':>10}"
          f"{'actually delivered':>22}")
    print("  " + "-" * 72)
    total_resolved = sum(sum(r["bin_counts"]) for r in runs)
    for index, (low, high) in enumerate(BINS):
        count = sum(r["bin_counts"][index] for r in runs)
        if count == 0:
            print(f"  {low:.1f} - {high:.1f}{'':<5}{count:>14}"
                  f"{'':>10}{'n/a':>22}")
            continue
        delivered = sum(r["bin_delivered"][index] for r in runs)
        print(f"  {low:.1f} - {high:.1f}{'':<5}{count:>14}"
              f"{100*count/total_resolved:>9.1f}%{100*delivered/count:>21.2f}%")
    print("  " + "-" * 72)
    print(f"  {'all':<14}{total_resolved:>14}{100.0:>9.1f}%"
          f"{100*sum(sum(r['bin_delivered']) for r in runs)/total_resolved:>21.2f}%")
    print("  " + "-" * 72)

    # --- 3. validation noise ---
    print()
    print(f"  3. Validation noise — same {len(VALIDATION_SEEDS)} layouts, "
          "three channel/traffic seeds")
    print("  " + "-" * 72)
    print(f"  {'run seed':<20}{'mean total loss':>20}{'std over layouts':>22}")
    print("  " + "-" * 72)
    means: List[float] = []
    for offset in RUN_SEED_OFFSETS:
        validation = sweep(VALIDATION_SEEDS, state, args.workers, offset)
        totals = [run["total_loss"] for run in validation]
        means.append(statistics.fmean(totals))
        label = ("placement seed" if offset == 0
                 else f"placement seed + {offset}")
        print(f"  {label:<20}{means[-1]*100:>19.2f}%"
              f"{statistics.stdev(totals)*100:>21.2f}%")
    print("  " + "-" * 72)
    print(f"  spread across the three: {min(means)*100:.2f}% to "
          f"{max(means)*100:.2f}%  "
          f"(range {(max(means)-min(means))*100:.2f} points)")
    print("  " + "-" * 72)


if __name__ == "__main__":
    main()
