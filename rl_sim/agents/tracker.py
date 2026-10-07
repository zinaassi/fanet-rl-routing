"""
tracker.py — What each drone learns from end-to-end ACKs.

Holds the two pieces of state the Phase-1b routing rules are built on, and
nothing else:

* **A decision store.** Every routing choice is recorded with the three local
  inputs it was made from. The record waits until the packet's fate is known.
* **Per-link delivery rates.** For each link, the mean of the last
  ``config.DELIVERY_RATE_WINDOW`` RESOLVED outcomes a drone sent on it.

How a packet resolves
---------------------
* **Delivered.** The GS ACK walks back along the packet's path, so every drone
  that decided on that packet learns "delivered" for the link it used.
* **Lost.** Nothing arrives, and at ``created_at + PACKET_TTL`` every drone
  that decided on the packet records "lost". Each drone knows the creation
  step and the TTL from the packet header, so this is a local decision, not a
  global announcement.

Why decisions are stored rather than read back off the path
-----------------------------------------------------------
``Packet.path`` records only hops that SUCCEEDED — a send lost in the channel
never reaches ``relay_to``. The decision that matters most for a lost packet,
the one whose transmission failed, is therefore absent from the path. Walking
the path would silently drop exactly the examples the network most needs, so
every decision is kept here instead.

Locality
--------
Rates are keyed by ``(drone_id, next_hop)``. A decision made at drone X reads
only X's own entries; nothing here lets one drone see another's. Keeping the
dictionary in one object is a placement choice, not a widening of what a drone
knows.

No torch: the "rate" rule needs none of it, and this module is shared.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from fanet_sim import config
from fanet_sim.envs.drone import NextHop

#: The three local inputs every routing decision is made from, in order:
#: delivery_rate, own queue fill, channel loss.
Features = Tuple[float, float, float]

#: One resolved decision, ready to train on.
Example = Tuple[Features, float]


@dataclass
class Decision:
    """One routing choice, waiting for its packet's fate.

    Attributes:
        drone_id: The drone that chose.
        next_hop: The link it chose — a drone id, or "GS".
        features: The three local inputs the choice was made from.
        sent:     Whether the packet was ever actually transmitted on that
                  link. False when the link's queue refused it, or when it
                  expired while waiting. Only sent decisions move the link's
                  delivery rate; all of them become training examples.
    """

    drone_id: int
    next_hop: NextHop
    features: Features
    sent: bool = False


@dataclass
class Tracker:
    """Per-link delivery rates and the decisions still awaiting an outcome.

    Attributes:
        window:    How many resolved outcomes each link keeps.
        rates:     (drone_id, next_hop) → recent outcomes, newest last.
        decisions: packet id → every decision made about it, in order. A
                   packet can revisit a drone, so one drone may appear more
                   than once; all its entries resolve to the same target.
        deadlines: step → packet ids whose TTL runs out then.
    """

    window: int = field(default_factory=lambda: config.DELIVERY_RATE_WINDOW)
    rates: Dict[Tuple[int, NextHop], Deque[float]] = field(default_factory=dict)
    decisions: Dict[int, List[Decision]] = field(default_factory=dict)
    deadlines: Dict[int, List[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Delivery rates
    # ------------------------------------------------------------------

    def delivery_rate(
        self, drone_id: int, next_hop: NextHop, default: float
    ) -> float:
        """Return the measured delivery rate of one link.

        Args:
            drone_id: The drone whose link it is.
            next_hop: The far end — a drone id, or "GS".
            default:  What to report before any outcome has resolved. The
                      callers pass ``1 - channel_loss`` for that link.

        Returns:
            The mean of the last resolved outcomes, or *default* if there are
            none yet.
        """
        outcomes = self.rates.get((drone_id, next_hop))
        if not outcomes:
            return default
        return sum(outcomes) / len(outcomes)

    def _record_outcome(
        self, drone_id: int, next_hop: NextHop, delivered: bool
    ) -> None:
        """Add one resolved outcome to a link's window.

        Args:
            drone_id:  The drone whose link it is.
            next_hop:  The far end.
            delivered: Whether the packet reached the GS.
        """
        key = (drone_id, next_hop)
        if key not in self.rates:
            self.rates[key] = deque(maxlen=self.window)
        self.rates[key].append(1.0 if delivered else 0.0)

    # ------------------------------------------------------------------
    # Decisions
    # ------------------------------------------------------------------

    def record_decision(
        self,
        packet_id: int,
        created_at: int,
        drone_id: int,
        next_hop: NextHop,
        features: Features,
        ttl: Optional[int] = None,
    ) -> None:
        """Store one routing choice until the packet's fate is known.

        Args:
            packet_id:  Which packet.
            created_at: The step it was created, from its header.
            drone_id:   The drone choosing.
            next_hop:   The link chosen.
            features:   The three local inputs behind the choice.
            ttl:        Packet lifetime in steps. Defaults to
                        ``config.PACKET_TTL``.
        """
        if ttl is None:
            ttl = config.PACKET_TTL

        if packet_id not in self.decisions:
            self.decisions[packet_id] = []
            self.deadlines.setdefault(created_at + ttl, []).append(packet_id)
        self.decisions[packet_id].append(
            Decision(drone_id=drone_id, next_hop=next_hop, features=features)
        )

    def mark_sent(
        self, packet_id: int, drone_id: int, next_hop: NextHop
    ) -> None:
        """Note that a packet really was transmitted on the link it was given.

        Only sent decisions move a link's delivery rate. A packet refused by a
        full queue, or one that expired while waiting, was never sent.

        Args:
            packet_id: Which packet.
            drone_id:  The sending drone.
            next_hop:  The link used.
        """
        for decision in reversed(self.decisions.get(packet_id, ())):
            if decision.drone_id == drone_id and decision.next_hop == next_hop:
                decision.sent = True
                return

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, packet_id: int, delivered: bool) -> List[Example]:
        """Settle every decision made about one packet.

        Args:
            packet_id: The packet whose fate is now known.
            delivered: True if it reached the GS.

        Returns:
            One training example per decision, each ``(features, target)``
            with target 1.0 for delivered and 0.0 for lost. Empty if the
            packet was already resolved or never routed.
        """
        decisions = self.decisions.pop(packet_id, None)
        if decisions is None:
            return []

        target = 1.0 if delivered else 0.0
        for decision in decisions:
            if decision.sent:
                self._record_outcome(
                    decision.drone_id, decision.next_hop, delivered
                )
        return [(decision.features, target) for decision in decisions]

    def resolve_deadlines(self, step: int) -> List[Example]:
        """Give up on every packet whose TTL ran out at *step*.

        Args:
            step: The current simulation step.

        Returns:
            The training examples from those packets, all with target 0.0.
        """
        examples: List[Example] = []
        for packet_id in self.deadlines.pop(step, ()):
            examples.extend(self.resolve(packet_id, delivered=False))
        return examples

    def reset(self) -> None:
        """Forget everything. Called at the start of every run."""
        self.rates.clear()
        self.decisions.clear()
        self.deadlines.clear()
