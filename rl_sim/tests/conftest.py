"""Shared fixtures and helpers for the Phase-1A test suite.

Every test runs a real episode through :class:`FANETEnv`. ``run_episode``
keeps that boilerplate in one place and also records the per-step facts the
invariant tests need (queue occupancy, send counts, positions), since those
cannot be recovered after the episode ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pytest

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv


@dataclass
class EpisodeTrace:
    """An episode plus the per-step observations taken while it ran.

    Attributes:
        env:            The environment, after the episode finished.
        max_queue_seen: Largest queue length observed on any drone at any step.
        max_sent_step:  Largest number of transmissions one drone attempted in
                        a single step.
        max_released_step: Largest batch any drone released from its queue in a
                        single step, counted at Drone.dequeue_for_send — so it
                        includes failed sends and "no_route" drops, not just
                        the transmissions that went out.
        positions:      Per-step snapshot of every drone position.
        tx_total:       Total accepted transmissions over the episode.
    """

    env: FANETEnv
    max_queue_seen: int = 0
    max_sent_step: int = 0
    max_released_step: int = 0
    positions: List[np.ndarray] = field(default_factory=list)
    tx_total: int = 0


def run_episode(
    steps: int = 300,
    routing: str = "greedy",
    placement_seed: int = 1,
    run_seed: int = 1,
    log_path: Optional[str] = None,
) -> EpisodeTrace:
    """Run one episode and return it with its per-step observations.

    Args:
        steps:          How many steps to run (also sets config.MAX_STEPS, so
                        the env's own termination matches).
        routing:        Routing rule to use.
        placement_seed: Drone-placement seed.
        run_seed:       Channel/traffic seed.
        log_path:       Where to write the event log. Defaults to os.devnull,
                        so tests do not litter the working tree.

    Returns:
        An :class:`EpisodeTrace`.
    """
    import os

    from fanet_sim.envs.drone import Drone

    prev_max_steps = config.MAX_STEPS
    real_dequeue = Drone.dequeue_for_send
    config.MAX_STEPS = steps
    try:
        env = FANETEnv(
            routing=routing,
            log_path=os.devnull if log_path is None else log_path,
            placement_seed=placement_seed,
            run_seed=run_seed,
        )
        env.reset()
        trace = EpisodeTrace(env=env)

        # Spy on the queue release itself: this is the ONLY place packets leave
        # a queue to be sent, so it is where the per-step budget must hold.
        released: Dict[int, int] = {}

        def spy(self: Drone, n: Optional[int] = None) -> list:
            batch = real_dequeue(self, n)
            released[self.drone_id] = released.get(self.drone_id, 0) + len(batch)
            return batch

        Drone.dequeue_for_send = spy  # type: ignore[method-assign]

        trace.positions.append(
            np.array([d.position.copy() for d in env.drones])
        )

        for _ in range(steps):
            before_sent = {
                d.drone_id: sum(d.link_sent.values()) for d in env.drones
            }
            released.clear()
            env.step()
            if released:
                trace.max_released_step = max(
                    trace.max_released_step, max(released.values())
                )

            trace.tx_total += len(env.tx_events)
            for d in env.drones:
                trace.max_queue_seen = max(trace.max_queue_seen, len(d.queue))
                sent = sum(d.link_sent.values()) - before_sent[d.drone_id]
                trace.max_sent_step = max(trace.max_sent_step, sent)
            trace.positions.append(
                np.array([d.position.copy() for d in env.drones])
            )

        env.close_logger()
        return trace
    finally:
        config.MAX_STEPS = prev_max_steps
        Drone.dequeue_for_send = real_dequeue  # type: ignore[method-assign]


def drop_counts_by_reason(env: FANETEnv) -> Dict[str, int]:
    """Count dropped packets per drop-reason label.

    Args:
        env: A finished environment.

    Returns:
        Dict mapping reason label → count. Reasons that never fired are absent.
    """
    counts: Dict[str, int] = {}
    for pkt in env.dropped:
        assert pkt.drop_reason is not None, "a dropped packet has no reason"
        counts[pkt.drop_reason.label] = counts.get(pkt.drop_reason.label, 0) + 1
    return counts


@pytest.fixture(scope="module")
def episode() -> EpisodeTrace:
    """A single 300-step greedy episode, shared by the read-only tests."""
    return run_episode(steps=300)
