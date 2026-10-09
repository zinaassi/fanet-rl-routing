"""
diagnose_1b_context.py — The state of a packet when the rl policy decides,
and how that packet eventually ends.

Read-only. No model is changed and no config is written: the traffic load is
set per run and restored.

For two groups of rl decisions — those whose chosen option carried a predicted
delivery chance inside a band (0.4-0.5 by default), and ALL decisions as a
reference — it reports:

    * the packet's age at the decision, in steps since it was created, and
      the share older than AGE_THRESHOLD steps
    * the packet's hop count at the decision
    * the share whose packet had already passed through this drone earlier
    * how those packets finally ended

Every share is over DECISIONS, not packets: a packet routed five times
contributes five rows, each carrying that packet's eventual ending. A packet
still in flight when its run ends is reported as "in flight", since it has no
ending yet.

Usage:
    python scripts/diagnose_1b_context.py
    python scripts/diagnose_1b_context.py --band 0.9 1.0
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.drone import Drone, NextHop
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import Packet
from scripts.metrics_1a import compute_metrics
from scripts.stamp import settings_stamp
from scripts.train_1b import TEST_SEEDS

LOAD_MS = 200
AGE_THRESHOLD = 40          # steps; with TTL 50 this leaves under 1 s
ENDINGS = ("delivered", "channel", "queue_full", "no_route", "dead_end",
           "ttl", "hop_limit", "in flight")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Packet state at each rl decision, and how it ended."
    )
    parser.add_argument("--band", type=float, nargs=2, default=[0.4, 0.5],
                        metavar=("LOW", "HIGH"),
                        help="Predicted-value band to single out (default 0.4 0.5).")
    parser.add_argument("--init-seed", type=int, default=0,
                        help="Which trained rl model to use (default 0).")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the trained models live.")
    return parser.parse_args()


def _build_router(state: Dict[str, Any], run_seed: int, band: Tuple[float, float]):
    """Return an rl router that logs the context of every decision.

    Built inside the worker so the subclass is defined where torch is already
    imported, and so nothing in agents/ has to change.

    Args:
        state:    The trained weights.
        run_seed: The run's channel/traffic seed.
        band:     ``(low, high)`` predicted-value band to single out.

    Returns:
        A router carrying one record per decision.
    """
    import torch

    from agents.rl_router import RLRouter

    low, high = band

    class ContextRouter(RLRouter):
        """rl, logging what the packet looked like at each decision."""

        def __init__(self) -> None:
            super().__init__(run_seed=run_seed, training=False)
            self.net.load_state_dict(state)
            self.env: Optional[FANETEnv] = None
            # (packet_id, in_band, age, hop_count, revisited)
            self.records: List[Tuple[int, bool, int, int, bool]] = []
            self._in_band = False

        def _choose(self, drone: Drone, keys, feature_rows) -> NextHop:
            """Choose as rl does, noting whether it lands in the band."""
            with torch.no_grad():
                scores = self.net(
                    torch.tensor(feature_rows, dtype=torch.float32)
                )
            best = int(torch.argmax(scores).item())
            self._in_band = low <= float(scores[best].item()) < high
            return keys[best]

        def select_next_hop(self, drone, candidates, pkt: Packet, came_from):
            """Route, then log the packet's state at this decision."""
            chosen = super().select_next_hop(drone, candidates, pkt, came_from)
            if chosen is None:
                return None

            step = self.env.step_count if self.env is not None else 0
            # path is [source, hop1, ..., this drone]; anything before the last
            # entry is somewhere the packet has already been.
            revisited = drone.drone_id in pkt.path[:-1]
            self.records.append((
                pkt.packet_id, self._in_band, step - pkt.created_at,
                pkt.hop_count, revisited,
            ))
            return chosen

    return ContextRouter()


def _run(args: Tuple) -> Dict[str, Any]:
    """Worker: run one layout and aggregate its decision records.

    Args:
        args: ``(placement_seed, state, band)``.

    Returns:
        Totals for the band group and for all decisions.
    """
    placement_seed, state, band = args

    previous = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(round(LOAD_MS / 1000.0 / config.TIMESTEP))
    try:
        router = _build_router(state, placement_seed, band)
        env = FANETEnv(routing="rl", log_path=os.devnull,
                       placement_seed=placement_seed, run_seed=placement_seed,
                       router=router)
        router.env = env
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)

        ending: Dict[int, str] = {}
        for pkt in env.all_packets:
            if pkt.delivered:
                ending[pkt.packet_id] = "delivered"
            elif pkt.dropped and pkt.drop_reason is not None:
                ending[pkt.packet_id] = pkt.drop_reason.label
            else:
                ending[pkt.packet_id] = "in flight"

        groups = {
            "band": {"count": 0, "age_sum": 0, "old": 0, "hop_sum": 0,
                     "loops": 0, "endings": Counter()},
            "all": {"count": 0, "age_sum": 0, "old": 0, "hop_sum": 0,
                    "loops": 0, "endings": Counter()},
        }
        for packet_id, in_band, age, hops, revisited in router.records:
            names = ("all", "band") if in_band else ("all",)
            for name in names:
                bucket = groups[name]
                bucket["count"] += 1
                bucket["age_sum"] += age
                bucket["hop_sum"] += hops
                if age > AGE_THRESHOLD:
                    bucket["old"] += 1
                if revisited:
                    bucket["loops"] += 1
                bucket["endings"][ending[packet_id]] += 1

        for bucket in groups.values():
            bucket["endings"] = dict(bucket["endings"])
        groups["views_agree"] = not metrics["discrepancies"]
        return groups
    finally:
        config.PACKET_INTERVAL_STEPS = previous


def main() -> None:
    """Run the diagnostic and print its table."""
    args = parse_args()
    band = (args.band[0], args.band[1])

    import torch

    path = os.path.join(args.out_dir, "models", config.LOOP_GUARD,
                        f"rl_seed{args.init_seed}.pt")
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

    totals: Dict[str, Dict[str, Any]] = {}
    for name in ("band", "all"):
        merged: Dict[str, Any] = {"count": 0, "age_sum": 0, "old": 0,
                                  "hop_sum": 0, "loops": 0,
                                  "endings": Counter()}
        for run in runs:
            bucket = run[name]
            for key in ("count", "age_sum", "old", "hop_sum", "loops"):
                merged[key] += bucket[key]
            merged["endings"].update(bucket["endings"])
        totals[name] = merged

    labels = {"band": f"[{band[0]:.1f}, {band[1]:.1f})", "all": "all decisions"}

    print(f"Packet state at each rl decision — seed-{args.init_seed} model, "
          f"{len(TEST_SEEDS)} TEST layouts, {LOAD_MS} ms")
    print(f"  {settings_stamp(loads_ms=[LOAD_MS], placements=len(TEST_SEEDS))}")
    print(f"  Every share is over DECISIONS. TTL is {config.PACKET_TTL} steps; "
          f'"old" means age > {AGE_THRESHOLD}.')
    print()
    print("  " + "-" * 72)
    print(f"  {'':<42}{labels['band']:>14}{labels['all']:>16}")
    print("  " + "-" * 72)

    def row(title: str, key: str, fmt: str = "{:.2f}") -> None:
        """Print one metric for both groups."""
        values = []
        for name in ("band", "all"):
            bucket = totals[name]
            count = bucket["count"] or 1
            if key == "count":
                values.append(f"{bucket['count']}")
            elif key == "age":
                values.append(fmt.format(bucket["age_sum"] / count))
            elif key == "old":
                values.append(fmt.format(100 * bucket["old"] / count) + "%")
            elif key == "hops":
                values.append(fmt.format(bucket["hop_sum"] / count))
            elif key == "loops":
                values.append(fmt.format(100 * bucket["loops"] / count) + "%")
        print(f"  {title:<42}{values[0]:>14}{values[1]:>16}")

    row("decisions", "count")
    row("mean packet age at the decision (steps)", "age")
    row(f"share with age > {AGE_THRESHOLD} steps", "old")
    row("mean hop count at the decision", "hops")
    row("share whose packet had been here before", "loops")
    print("  " + "-" * 72)
    print(f"  {'how those packets finally ended:':<42}")
    for label in ENDINGS:
        shares = []
        for name in ("band", "all"):
            bucket = totals[name]
            count = bucket["count"] or 1
            shares.append(f"{100 * bucket['endings'].get(label, 0) / count:.2f}%")
        print(f"    {label:<40}{shares[0]:>14}{shares[1]:>16}")
    print("  " + "-" * 72)


if __name__ == "__main__":
    main()
