"""
diagnose_1b.py — What links the rules actually choose, and how they behave
at tighter link cutoffs.

A read-only diagnostic. It changes no model and writes no config: where a
cutoff is varied it is set for the duration of one run and restored.

Four tables, all on the 20 TEST layouts at the 200 ms working load:

1. The channel loss of the links actually CHOSEN — mean, median, and the
   share of transmissions over links losing more than 10 / 20 / 40%.
   The world is static, so a link's channel loss is fixed by the distance
   between its endpoints; weighting each link by how many transmissions it
   carried gives the exact distribution over transmissions, with no
   instrumentation inside the simulator.

2. Mean hops per delivered packet, and mean channel loss per hop — once over
   every transmission, and once over only the hops that delivered packets
   took. Those are different populations and can differ.

3. The same for "rate" with EPSILON_RATE = 0, to separate the loss its
   exploration causes from the loss its policy causes.

4. A cutoff check: "rate" and "rl" at LINK_MAX_LOSS 0.2 and 0.3.
   NOTE: the cutoff also feeds the connected-layout filter, so a tighter one
   can reject a layout and redraw it. The layouts are therefore NOT
   guaranteed identical across cutoffs; the table reports the mean placement
   draws so that is visible.

Usage:
    python scripts/diagnose_1b.py
    python scripts/diagnose_1b.py --init-seed 0 --workers 20
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
from fanet_sim.envs import channel
from fanet_sim.envs.channel import euclidean_distance
from fanet_sim.envs.fanet_env import FANETEnv
from scripts.metrics_1a import compute_metrics
from scripts.stamp import settings_stamp
from scripts.train_1b import TEST_SEEDS

LOAD_MS = 200
THRESHOLDS = (0.10, 0.20, 0.40)
CAUSES = ("channel", "queue_full", "no_route", "ttl_hop")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Phase-1b routing diagnostics.")
    parser.add_argument("--init-seed", type=int, default=0,
                        help="Which trained rl model to use (default 0).")
    parser.add_argument("--workers", type=int,
                        default=min(20, os.cpu_count() or 1),
                        help="Parallel workers; 1 runs sequentially.")
    parser.add_argument("--out-dir", type=str, default="out",
                        help="Where the trained models live.")
    return parser.parse_args()


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Return the weighted median of *values*.

    Args:
        values:  The observations.
        weights: How many times each was observed.

    Returns:
        The value at which half the weight lies below.
    """
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    seen = 0.0
    for value, weight in pairs:
        seen += weight
        if seen >= total / 2.0:
            return value
    return pairs[-1][0]


def profile_run(
    routing: str,
    placement_seed: int,
    state_dict: Optional[Dict[str, Any]] = None,
    epsilon: Optional[float] = None,
    link_max_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one layout and describe the links it chose.

    Args:
        routing:        The rule to run.
        placement_seed: Layout seed; run_seed is the same.
        state_dict:     For "rl", the weights to use.
        epsilon:        Override the rule's exploration probability.
        link_max_loss:  Override config.LINK_MAX_LOSS for this run only.

    Returns:
        Totals and the per-link transmission profile for this layout.
    """
    previous_interval = config.PACKET_INTERVAL_STEPS
    previous_cutoff = config.LINK_MAX_LOSS
    config.PACKET_INTERVAL_STEPS = int(round(LOAD_MS / 1000.0 / config.TIMESTEP))
    if link_max_loss is not None:
        config.LINK_MAX_LOSS = link_max_loss
    try:
        router = None
        if routing in ("rate", "rl"):
            if routing == "rate":
                from agents.rate_router import RateRouter

                router = RateRouter(run_seed=placement_seed, epsilon=epsilon)
            else:
                from agents import make_router

                router = make_router("rl", run_seed=placement_seed,
                                     training=False, init_seed=0)
                if state_dict is not None:
                    router.net.load_state_dict(state_dict)

        env = FANETEnv(routing=routing, log_path=os.devnull,
                       placement_seed=placement_seed, run_seed=placement_seed,
                       router=router)
        env.reset()
        for _ in range(config.MAX_STEPS):
            env.step()
        env.close_logger()

        metrics = compute_metrics(env)
        truth = metrics["ground_truth"]
        by_reason = truth["loss_rate_by_reason"]

        # --- the loss of every link that carried traffic, by transmission ---
        positions = {d.drone_id: d.position for d in env.drones}
        losses: List[float] = []
        weights: List[float] = []
        for (from_id, to_key), row in env.link_stats_measured.items():
            if row["sent"] == 0:
                continue
            target = (env.gs_position if to_key == "GS" else positions[to_key])
            losses.append(channel.p_loss(
                euclidean_distance(positions[from_id], target)
            ))
            weights.append(float(row["sent"]))

        # --- the hops delivered packets actually took ---
        delivered_hop_losses: List[float] = []
        for pkt in env.delivered:
            for first, second in zip(pkt.path, pkt.path[1:]):
                start = positions[first]
                end = env.gs_position if second == "GS" else positions[second]
                delivered_hop_losses.append(
                    channel.p_loss(euclidean_distance(start, end))
                )

        return {
            "total_loss": truth["total_loss"] or 0.0,
            "channel": by_reason["channel"],
            "queue_full": by_reason["queue_full"],
            "no_route": by_reason["no_route"],
            "ttl_hop": by_reason["ttl"] + by_reason["hop_limit"],
            "mean_hops": truth["mean_hops"] or 0.0,
            "placement_draws": env.placement_draws,
            "views_agree": not metrics["discrepancies"],
            "link_losses": losses,
            "link_weights": weights,
            "delivered_hop_losses": delivered_hop_losses,
        }
    finally:
        config.PACKET_INTERVAL_STEPS = previous_interval
        config.LINK_MAX_LOSS = previous_cutoff


def _job(args: Tuple) -> Dict[str, Any]:
    """Worker entry point for one profiled run."""
    routing, seed, state, epsilon, cutoff = args
    return profile_run(routing, seed, state, epsilon, cutoff)


def profile(
    routing: str,
    state: Optional[Dict[str, Any]],
    workers: int,
    epsilon: Optional[float] = None,
    cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one rule over every TEST layout and pool the profiles.

    Args:
        routing: The rule.
        state:   For "rl", the weights.
        workers: Parallel workers.
        epsilon: Exploration override.
        cutoff:  LINK_MAX_LOSS override.

    Returns:
        The pooled summary for this rule.

    Raises:
        SystemExit: If any run's ACK view disagrees with ground truth.
    """
    jobs = [(routing, seed, state, epsilon, cutoff) for seed in TEST_SEEDS]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            runs = list(pool.map(_job, jobs))
    else:
        runs = [_job(job) for job in jobs]

    if not all(run["views_agree"] for run in runs):
        print(f"STOP: views disagreed for {routing}")
        raise SystemExit(1)

    losses = [loss for run in runs for loss in run["link_losses"]]
    weights = [weight for run in runs for weight in run["link_weights"]]
    total_weight = sum(weights)
    hop_losses = [loss for run in runs for loss in run["delivered_hop_losses"]]

    summary: Dict[str, Any] = {
        "transmissions": total_weight,
        "mean_link_loss": (
            sum(loss * weight for loss, weight in zip(losses, weights))
            / total_weight if total_weight else float("nan")
        ),
        "median_link_loss": _weighted_median(losses, weights) if losses else float("nan"),
        "mean_delivered_hop_loss": (
            statistics.fmean(hop_losses) if hop_losses else float("nan")
        ),
        "mean_hops": statistics.fmean(run["mean_hops"] for run in runs),
        "placement_draws": statistics.fmean(run["placement_draws"] for run in runs),
        "total_loss": statistics.fmean(run["total_loss"] for run in runs),
    }
    for cause in CAUSES:
        summary[cause] = statistics.fmean(run[cause] for run in runs)
    for threshold in THRESHOLDS:
        share = sum(
            weight for loss, weight in zip(losses, weights) if loss > threshold
        ) / total_weight if total_weight else float("nan")
        summary[f"above_{int(threshold*100)}"] = share
    return summary


def main() -> None:
    """Run every diagnostic and print the four tables."""
    args = parse_args()

    import torch

    model_path = os.path.join(args.out_dir, "models",
                              f"rl_seed{args.init_seed}.pt")
    if not os.path.exists(model_path):
        print(f"STOP: no trained model at {model_path}")
        raise SystemExit(1)
    state = torch.load(model_path, weights_only=True)

    print(f"Phase-1b diagnostics — {len(TEST_SEEDS)} TEST layouts at {LOAD_MS} ms")
    print(f"  {settings_stamp(loads_ms=[LOAD_MS], placements=len(TEST_SEEDS))}")
    print(f"  rl model: {model_path}")

    cases: List[Tuple[str, Dict[str, Any]]] = []
    for rule in ("greedy", "rate", "rl"):
        cases.append((rule, profile(rule, state if rule == "rl" else None,
                                    args.workers)))
    cases.append(("rate (eps=0)",
                  profile("rate", None, args.workers, epsilon=0.0)))

    # --- 1 and 3: the chosen links ---
    print()
    print("  1+3. Channel loss of the links actually chosen, weighted by transmissions")
    print("  " + "-" * 86)
    print(f"  {'rule':<14}{'sends':>10}{'mean':>9}{'median':>9}"
          f"{'>10%':>9}{'>20%':>9}{'>40%':>9}")
    print("  " + "-" * 86)
    for name, values in cases:
        print(f"  {name:<14}{values['transmissions']:>10.0f}"
              f"{values['mean_link_loss']*100:>8.2f}%"
              f"{values['median_link_loss']*100:>8.2f}%"
              f"{values['above_10']*100:>8.1f}%"
              f"{values['above_20']*100:>8.1f}%"
              f"{values['above_40']*100:>8.1f}%")
    print("  " + "-" * 86)

    # --- 2: hops ---
    print()
    print("  2. Hops and the loss per hop")
    print("  " + "-" * 78)
    print(f"  {'rule':<14}{'mean hops':>12}{'mean loss/hop':>16}"
          f"{'same, delivered':>18}")
    print(f"  {'':<14}{'(delivered)':>12}{'(all sends)':>16}{'hops only':>18}")
    print("  " + "-" * 78)
    for name, values in cases:
        print(f"  {name:<14}{values['mean_hops']:>12.2f}"
              f"{values['mean_link_loss']*100:>15.2f}%"
              f"{values['mean_delivered_hop_loss']*100:>17.2f}%")
    print("  " + "-" * 78)

    # --- 4: the cutoff check ---
    print()
    print("  4. Tighter link cutoffs — total loss and where it goes")
    print("     NOTE: the cutoff also drives the layout filter, so layouts may")
    print("     be redrawn and are NOT guaranteed identical across rows.")
    print("  " + "-" * 92)
    print(f"  {'cutoff':>8}{'reach':>9}{'rule':>7}{'total':>9}{'channel':>10}"
          f"{'queue_full':>12}{'no_route':>10}{'ttl+hop':>10}{'draws':>9}")
    print("  " + "-" * 92)
    for cutoff in (config.LINK_MAX_LOSS, 0.3, 0.2):
        for rule in ("rate", "rl"):
            values = profile(rule, state if rule == "rl" else None,
                             args.workers, cutoff=cutoff)
            marker = "  <- configured" if cutoff == config.LINK_MAX_LOSS else ""
            print(f"  {cutoff*100:>7.0f}%{channel.max_link_distance(cutoff):>9.1f}"
                  f"{rule:>7}{values['total_loss']*100:>8.2f}%"
                  f"{values['channel']*100:>9.2f}%{values['queue_full']*100:>11.2f}%"
                  f"{values['no_route']*100:>9.2f}%{values['ttl_hop']*100:>9.2f}%"
                  f"{values['placement_draws']:>9.2f}{marker}")
        print("  " + "-" * 92)


if __name__ == "__main__":
    main()
