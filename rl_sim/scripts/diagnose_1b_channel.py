"""
diagnose_1b_channel.py — Why the channel losses happened (diagnostic only).

For every transmission that was LOST IN THE CHANNEL, this records what the
deciding drone could see at the moment it chose that link: the chosen link's
channel loss, and the cleanest option the loop guard still allowed — its
channel loss and how full the deciding drone's own queue for it was.

A packet lost in the channel is lost on exactly one transmission, its last,
so the decision responsible for a channel loss is that packet's last recorded
decision. Nothing here is reconstructed from the packet's path.

Each channel loss is then classified:

  A) avoidable         an allowed option at least 10 points cleaner existed,
                       and its queue fill was < 0.5
  B) congestion-forced such an option existed, but its queue fill was >= 0.5
  C) geometry-forced   no allowed option was 10 or more points cleaner

Class A is then split by the cleaner option's measured delivery rate:

  A1) the cleaner option's delivery rate was HIGHER or equal — the chosen
      link was worse on both channel loss and delivery rate
  A2) the cleaner option's delivery rate was LOWER — the chosen link traded
      channel loss for the better end-to-end record

"Allowed" means what the router could actually pick: the loop guard's
candidates, minus the drone the packet came from.

Nothing is trained and no model changes. Read-only.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv
from fanet_sim.envs.packet import DropReason
from scripts.stamp import settings_stamp

TEST_SEEDS = tuple(range(1, 51))
LOAD_MS = 200
RULES = ("rl", "rate")
CLASSES = ("A", "B", "C")
CLASS_LABELS = {
    "A": "A  avoidable (cleaner option, queue < 0.5)",
    "B": "B  congestion-forced (cleaner option, queue >= 0.5)",
    "C": "C  geometry-forced (nothing 10+ points cleaner)",
}
A_SPLIT = ("A1", "A2")
A_SPLIT_LABELS = {
    "A1": "A1  cleaner option's delivery rate >= chosen",
    "A2": "A2  cleaner option's delivery rate <  chosen",
}
CLEANER_BY = 0.10          # "at least 10 points lower"
BUSY_QUEUE = 0.5           # "queue fill >= 0.5"
DIRTY_LINK = 0.20          # "links losing > 20%"


def _build_router(rule: str, state: Dict[str, Any], run_seed: int):
    """Return *rule*'s router, logging each packet's latest decision.

    Built inside the worker, so the subclass is defined where torch is already
    imported and nothing in agents/ has to change.

    Args:
        rule:     "rl" or "rate".
        state:    The trained weights, for "rl".
        run_seed: The run's channel/traffic seed.

    Returns:
        A router carrying ``latest``: packet id → what that decision saw.
    """
    if rule == "rl":
        from agents.rl_router import RLRouter as Base
    else:
        from agents.rate_router import RateRouter as Base

    class ProbeRouter(Base):                 # type: ignore[valid-type,misc]
        """The rule, unchanged, plus a log of what each decision was offered."""

        def __init__(self) -> None:
            if rule == "rl":
                super().__init__(run_seed=run_seed, training=False)
                self.net.load_state_dict(state)
            else:
                super().__init__(run_seed=run_seed)
            # packet id → what that decision was offered:
            # (chosen loss, chosen rate, cleanest loss, cleanest rate,
            #  cleanest queue fill)
            self.latest: Dict[int, Tuple[float, ...]] = {}

        def select_next_hop(self, drone, candidates, pkt, came_from):
            """Route exactly as the rule does, then log the option set.

            The options are read BEFORE the choice is made, so the queue fills
            are the ones the decision actually saw.
            """
            keys = self.options(candidates, came_from)
            if not keys:
                return super().select_next_hop(drone, candidates, pkt,
                                               came_from)
            rows = [self.features(drone, key, candidates[key]) for key in keys]

            chosen = super().select_next_hop(drone, candidates, pkt, came_from)
            if chosen is None:
                return None

            cleanest = min(range(len(keys)), key=lambda i: rows[i][2])
            picked = rows[keys.index(chosen)]
            # features are (delivery_rate, queue_fill, channel_loss).
            self.latest[pkt.packet_id] = (
                picked[2], picked[0],          # chosen link
                rows[cleanest][2], rows[cleanest][0], rows[cleanest][1],
            )
            return chosen

    return ProbeRouter()


def _classify(chosen_loss: float, clean_loss: float,
              clean_fill: float) -> str:
    """Return "A", "B" or "C" for one channel loss."""
    if chosen_loss - clean_loss < CLEANER_BY:
        return "C"
    return "A" if clean_fill < BUSY_QUEUE else "B"


def _run(args: Tuple) -> Dict[str, Any]:
    """Run one layout and summarise its channel losses.

    Args:
        args: ``(rule, seed, state)``.

    Returns:
        Per-class counts and summed chosen-link channel loss, plus how many
        channel losses happened on links losing more than 20%.
    """
    rule, seed, state = args

    previous_interval = config.PACKET_INTERVAL_STEPS
    config.PACKET_INTERVAL_STEPS = int(
        round(LOAD_MS / 1000.0 / config.TIMESTEP)
    )
    try:
        router = _build_router(rule, state, seed)
        env = FANETEnv(routing=rule, log_path=os.devnull,
                       placement_seed=seed, run_seed=seed, router=router)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        from scripts.metrics_1a import compute_metrics, measured_packets

        counts = {name: 0 for name in CLASSES}
        loss_sums = {name: 0.0 for name in CLASSES}
        # For the class-A split: counts and the four means it needs.
        a_counts = {name: 0 for name in A_SPLIT}
        a_sums = {name: {"chosen_loss": 0.0, "chosen_rate": 0.0,
                         "clean_loss": 0.0, "clean_rate": 0.0}
                  for name in A_SPLIT}
        dirty = 0
        unmatched = 0

        for pkt in measured_packets(env):
            if not pkt.dropped or pkt.drop_reason is not DropReason.CHANNEL:
                continue
            seen = router.latest.get(pkt.packet_id)
            if seen is None:
                unmatched += 1
                continue
            chosen_loss, chosen_rate, clean_loss, clean_rate, clean_fill = seen
            name = _classify(chosen_loss, clean_loss, clean_fill)
            counts[name] += 1
            loss_sums[name] += chosen_loss
            if chosen_loss > DIRTY_LINK:
                dirty += 1

            if name == "A":
                split = "A1" if clean_rate >= chosen_rate else "A2"
                a_counts[split] += 1
                a_sums[split]["chosen_loss"] += chosen_loss
                a_sums[split]["chosen_rate"] += chosen_rate
                a_sums[split]["clean_loss"] += clean_loss
                a_sums[split]["clean_rate"] += clean_rate

        return {
            "counts": counts, "loss_sums": loss_sums, "dirty": dirty,
            "a_counts": a_counts, "a_sums": a_sums,
            "unmatched": unmatched,
            "views_agree": not compute_metrics(env)["discrepancies"],
        }
    finally:
        config.PACKET_INTERVAL_STEPS = previous_interval


def collect(rule: str, state: Dict[str, Any],
            workers: int) -> Dict[str, Any]:
    """Run every layout for *rule* and pool the results.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    jobs = [(rule, seed, state) for seed in TEST_SEEDS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run, jobs))
    else:
        results = [_run(job) for job in jobs]

    if not all(r["views_agree"] for r in results):
        print("STOP: views disagreed with ground truth")
        raise SystemExit(1)

    counts = {name: sum(r["counts"][name] for r in results)
              for name in CLASSES}
    loss_sums = {name: sum(r["loss_sums"][name] for r in results)
                 for name in CLASSES}
    a_counts = {name: sum(r["a_counts"][name] for r in results)
                for name in A_SPLIT}
    a_sums = {
        name: {field: sum(r["a_sums"][name][field] for r in results)
               for field in ("chosen_loss", "chosen_rate",
                             "clean_loss", "clean_rate")}
        for name in A_SPLIT
    }
    return {
        "counts": counts, "loss_sums": loss_sums,
        "a_counts": a_counts, "a_sums": a_sums,
        "dirty": sum(r["dirty"] for r in results),
        "unmatched": sum(r["unmatched"] for r in results),
        "total": sum(counts.values()),
    }


def print_table(rule: str, data: Dict[str, Any]) -> None:
    """Print one rule's classification table."""
    total = data["total"]
    width = 86
    print()
    print(f"  {rule} — {total} channel losses over "
          f"{len(TEST_SEEDS)} layouts at {LOAD_MS} ms")
    print("  " + "-" * width)
    print(f"  {'class':<52}{'share':>10}{'mean chosen loss':>22}")
    print("  " + "-" * width)
    for name in CLASSES:
        count = data["counts"][name]
        share = 100.0 * count / total if total else 0.0
        mean = (100.0 * data["loss_sums"][name] / count) if count else 0.0
        print(f"  {CLASS_LABELS[name]:<52}{share:>9.2f}%{mean:>21.2f}%")
    print("  " + "-" * width)
    dirty_share = 100.0 * data["dirty"] / total if total else 0.0
    print(f"  {'on links losing > 20%':<52}{dirty_share:>9.2f}%")
    print("  " + "-" * width)
    if data["unmatched"]:
        print(f"  ({data['unmatched']} channel losses had no recorded "
              f"decision and are excluded)")


def print_a_split(rule: str, data: Dict[str, Any]) -> None:
    """Print the class-A split, chosen link against the cleaner option."""
    a_total = data["counts"]["A"]
    width = 100
    print()
    print(f"  {rule} — class A only ({a_total} channel losses), split by the "
          f"cleaner option's delivery rate")
    print("  " + "-" * width)
    print(f"  {'':<44}{'share':>9}{'chosen':>21}{'cleaner option':>25}")
    print(f"  {'':<44}{'of A':>9}{'rate':>11}{'loss':>10}"
          f"{'rate':>13}{'loss':>11}")
    print("  " + "-" * width)
    for name in A_SPLIT:
        count = data["a_counts"][name]
        sums = data["a_sums"][name]
        share = 100.0 * count / a_total if a_total else 0.0
        if count:
            values = (100.0 * sums["chosen_rate"] / count,
                      100.0 * sums["chosen_loss"] / count,
                      100.0 * sums["clean_rate"] / count,
                      100.0 * sums["clean_loss"] / count)
        else:
            values = (0.0, 0.0, 0.0, 0.0)
        print(f"  {A_SPLIT_LABELS[name]:<44}{share:>8.2f}%"
              f"{values[0]:>10.2f}%{values[1]:>9.2f}%"
              f"{values[2]:>12.2f}%{values[3]:>10.2f}%")
    print("  " + "-" * width)


def main() -> None:
    """Run the diagnostic for both rules and print the tables."""
    parser = argparse.ArgumentParser(
        description="Classify Phase-1b channel losses by what was on offer."
    )
    parser.add_argument("--init-seed", type=int, default=0,
                        help="Which trained rl model to use.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the models live.")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--rules", type=str, nargs="+", default=list(RULES),
                        choices=list(RULES),
                        help="Which rules to measure (default: both).")
    args = parser.parse_args()

    if config.LOOP_GUARD != "path":
        print(f"STOP: LOOP_GUARD is {config.LOOP_GUARD!r}, expected 'path'.")
        raise SystemExit(1)

    import torch

    model = os.path.join(args.out_dir, "models", config.LOOP_GUARD,
                         f"rl_seed{args.init_seed}.pt")
    if not os.path.exists(model):
        print(f"STOP: no trained model at {model}")
        raise SystemExit(1)
    state = torch.load(model, weights_only=True)

    print(f"Channel losses by what the decision was offered — "
          f"seed-{args.init_seed} model, {len(TEST_SEEDS)} TEST layouts, "
          f"{LOAD_MS} ms")
    print(f"  {settings_stamp(loads_ms=[LOAD_MS], placements=len(TEST_SEEDS))}")
    print(f"  A cleaner option means at least "
          f"{CLEANER_BY * 100:.0f} points less channel loss; a busy queue "
          f"means fill >= {BUSY_QUEUE}.")

    for rule in args.rules:
        data = collect(rule, state, args.workers)
        print_table(rule, data)
        print_a_split(rule, data)


if __name__ == "__main__":
    main()
