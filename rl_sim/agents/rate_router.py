"""
rate_router.py — The "rate" routing rule, and the plumbing both rules share.

"rate" uses no neural network. Among the options it scores

    score = delivery_rate * (1 - queue_full)

where ``queue_full`` is 1 when the drone's own queue for that link is already
at capacity, and picks the highest; ties go to the lower channel loss. With
probability ``config.EPSILON_RATE`` it picks uniformly at random instead, so
links it never chooses still get measured. That exploration is on in training
AND at test, so its reported numbers carry that noise — unlike greedy's.

:class:`BaseRouter` holds everything the "rl" rule also needs: the option set,
the three local inputs, the decision store and the hooks the environment
calls. No torch here — "rate" needs none, and keeping it out means this file
can be tested without it.

The environment never imports this module. It is handed a router object and
calls five hooks on it:

    select_next_hop(drone, candidates, pkt, came_from) -> key | None
    on_sent(drone_id, next_hop, packet_id)
    on_delivered(pkt)
    on_step_end(step)
    on_reset()
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fanet_sim import config
from fanet_sim.envs import channel
from fanet_sim.envs.channel import euclidean_distance
from fanet_sim.envs.drone import Drone, NextHop
from fanet_sim.envs.packet import Packet

from agents.tracker import Example, Features, Tracker


def agent_rng(run_seed: int) -> np.random.Generator:
    """Return the agent's own exploration stream for a run.

    Derived from ``run_seed`` through a SEPARATE SeedSequence rather than by
    spawning a third stream off the environment's two. Spawning a third would
    change the channel stream and silently move every Phase-1a result.

    Args:
        run_seed: The run's channel/traffic seed.

    Returns:
        A generator used only for exploration draws.
    """
    return np.random.default_rng(
        np.random.SeedSequence([run_seed, config.AGENT_STREAM_TAG])
    )


class BaseRouter:
    """Shared machinery: options, local inputs, decisions, and the hooks.

    Attributes:
        tracker: Per-link delivery rates and the pending decisions.
        rng:     The exploration stream.
        name:    The routing rule's name.
    """

    name = "base"

    def __init__(self, run_seed: int, tracker: Optional[Tracker] = None) -> None:
        """Create a router.

        Args:
            run_seed: The run's channel/traffic seed; the exploration stream
                      is derived from it.
            tracker:  An existing tracker to use, or None for a fresh one.
        """
        self.tracker = tracker if tracker is not None else Tracker()
        self.run_seed = run_seed
        self.rng = agent_rng(run_seed)

    # ------------------------------------------------------------------
    # Options and inputs
    # ------------------------------------------------------------------

    @staticmethod
    def options(
        candidates: Dict[NextHop, np.ndarray], came_from: Optional[int]
    ) -> List[NextHop]:
        """Return the keys this packet may be sent to, in a fixed order.

        Everything currently reachable, minus the drone the packet just came
        from — the loop guard. The order is drone ids ascending then "GS", so
        a random draw over them is reproducible whatever order the candidate
        dict happens to have.

        Args:
            candidates: Reachable next hops, key → position.
            came_from:  The drone this packet arrived from, or None if it was
                        created here.

        Returns:
            The allowed keys. Empty means the caller must drop "no_route".
        """
        keys = sorted(
            key for key in candidates
            if key != "GS" and key != came_from
        )
        if "GS" in candidates:
            keys.append("GS")
        return keys

    def features(
        self, drone: Drone, key: NextHop, position: np.ndarray
    ) -> Features:
        """Return the three local inputs describing one option.

        In order: the link's measured delivery rate, the fill of this drone's
        own queue for that link BEFORE this packet joins it, and the link's
        channel loss. All three come from the drone itself.

        Args:
            drone:    The drone deciding.
            key:      The option.
            position: That endpoint's position.

        Returns:
            ``(delivery_rate, queue_fill, channel_loss)``.
        """
        channel_loss = channel.p_loss(
            euclidean_distance(drone.position, position)
        )
        delivery_rate = self.tracker.delivery_rate(
            drone.drone_id, key, default=1.0 - channel_loss
        )
        queue_fill = drone.queue_len(key) / config.QUEUE_CAPACITY
        return (delivery_rate, queue_fill, channel_loss)

    # ------------------------------------------------------------------
    # The choice
    # ------------------------------------------------------------------

    def _choose(
        self,
        drone: Drone,
        keys: Sequence[NextHop],
        feature_rows: Sequence[Features],
    ) -> NextHop:
        """Pick one option. Overridden by each rule.

        Args:
            drone:        The drone deciding.
            keys:         The allowed options, in fixed order.
            feature_rows: Their local inputs, in the same order.

        Returns:
            The chosen key.

        Raises:
            NotImplementedError: Always; subclasses implement this.
        """
        raise NotImplementedError

    def select_next_hop(
        self,
        drone: Drone,
        candidates: Dict[NextHop, np.ndarray],
        pkt: Packet,
        came_from: Optional[int],
    ) -> Optional[NextHop]:
        """Choose this packet's next hop and record the decision.

        Args:
            drone:      The drone deciding.
            candidates: Reachable next hops, key → position.
            pkt:        The packet being routed.
            came_from:  The drone it arrived from, or None.

        Returns:
            The chosen key, or None when the loop guard leaves nothing — the
            caller then drops the packet "no_route".
        """
        keys = self.options(candidates, came_from)
        if not keys:
            return None

        feature_rows = [
            self.features(drone, key, candidates[key]) for key in keys
        ]
        chosen = self._choose(drone, keys, feature_rows)

        self.tracker.record_decision(
            packet_id=pkt.packet_id,
            created_at=pkt.created_at,
            drone_id=drone.drone_id,
            next_hop=chosen,
            features=feature_rows[keys.index(chosen)],
            ttl=pkt.ttl,
        )
        return chosen

    # ------------------------------------------------------------------
    # Hooks the environment calls
    # ------------------------------------------------------------------

    def on_sent(
        self, drone_id: int, next_hop: NextHop, packet_id: int
    ) -> None:
        """Note that a packet was actually transmitted on its chosen link."""
        self.tracker.mark_sent(packet_id, drone_id, next_hop)

    def on_delivered(self, pkt: Packet) -> None:
        """Walk the GS ACK back along the path, settling every decision."""
        self._learn(self.tracker.resolve(pkt.packet_id, delivered=True))

    def on_step_end(self, step: int) -> None:
        """Give up on packets whose TTL ran out this step."""
        self._learn(self.tracker.resolve_deadlines(step))

    def on_reset(self) -> None:
        """Clear everything for a new run."""
        self.tracker.reset()
        self.rng = agent_rng(self.run_seed)

    def _learn(self, examples: List[Example]) -> None:
        """Consume resolved examples. "rate" does not learn, so this is a no-op.

        Args:
            examples: Newly resolved ``(features, target)`` pairs.
        """


class RateRouter(BaseRouter):
    """Pick the link with the best measured delivery rate that has room."""

    name = "rate"

    def __init__(
        self,
        run_seed: int,
        tracker: Optional[Tracker] = None,
        epsilon: Optional[float] = None,
    ) -> None:
        """Create the rate router.

        Args:
            run_seed: The run's channel/traffic seed.
            tracker:  An existing tracker, or None for a fresh one.
            epsilon:  Exploration probability. Defaults to
                      ``config.EPSILON_RATE``, and applies at test too.
        """
        super().__init__(run_seed, tracker)
        self.epsilon = config.EPSILON_RATE if epsilon is None else epsilon

    def _choose(
        self,
        drone: Drone,
        keys: Sequence[NextHop],
        feature_rows: Sequence[Features],
    ) -> NextHop:
        """Score each option and take the best, or explore.

        Args:
            drone:        The drone deciding.
            keys:         The allowed options.
            feature_rows: Their local inputs.

        Returns:
            The chosen key.
        """
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return keys[int(self.rng.integers(len(keys)))]

        def rank(index: int) -> Tuple[float, float]:
            """Higher score first; ties go to the lower channel loss."""
            delivery_rate, _, channel_loss = feature_rows[index]
            queue_full = 1.0 if drone.queue_is_full(keys[index]) else 0.0
            score = delivery_rate * (1.0 - queue_full)
            return (-score, channel_loss)

        return keys[min(range(len(keys)), key=rank)]
