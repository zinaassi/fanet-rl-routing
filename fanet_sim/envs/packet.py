"""
packet.py — Packet class and lifecycle tracking for the FANET simulator.

Each packet is a lightweight data object that records its full history:
where it was born, when, how many hops it has taken, and how it ended
(delivered, dropped-TTL, dropped-hops, dropped-void).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class DropReason(Enum):
    """Reason a packet was dropped before delivery."""
    TTL_EXPIRED = auto()      # packet lived too long (timestep count)
    MAX_HOPS = auto()         # hop limit exceeded
    NO_NEXT_HOP = auto()      # greedy routing void — no suitable neighbour


@dataclass
class Packet:
    """A single data packet travelling through the FANET.

    Attributes:
        packet_id:      Globally unique integer identifier.
        source_id:      ID of the drone that generated this packet.
        created_at:     Simulation timestep at which the packet was created.
        ttl:            Maximum number of timesteps the packet may live.
        max_hops:       Maximum number of relay hops allowed.
        size_bytes:     Payload size in bytes (informational).
        current_holder: ID of the drone (or 'GS') currently holding the packet.
        hop_count:      Number of hops taken so far (incremented on each relay).
        delivered:      True once the packet has reached the ground station.
        dropped:        True if the packet has been discarded.
        drop_reason:    Why the packet was dropped, if applicable.
        delivered_at:   Timestep at which delivery occurred (None if not yet).
        path:           Ordered list of drone IDs the packet has visited.
    """

    packet_id: int
    source_id: int
    created_at: int
    ttl: int
    max_hops: int
    size_bytes: int = 512
    current_holder: int | str = field(init=False)
    hop_count: int = field(default=0, init=False)
    delivered: bool = field(default=False, init=False)
    dropped: bool = field(default=False, init=False)
    drop_reason: Optional[DropReason] = field(default=None, init=False)
    delivered_at: Optional[int] = field(default=None, init=False)
    path: List[int | str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.current_holder = self.source_id
        self.path.append(self.source_id)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def is_alive(self, current_step: int) -> bool:
        """Return True if the packet has not expired, been delivered, or dropped.

        Args:
            current_step: The current simulation timestep.

        Returns:
            True if the packet is still active.
        """
        if self.delivered or self.dropped:
            return False
        age = current_step - self.created_at
        return age < self.ttl and self.hop_count <= self.max_hops

    def relay_to(self, next_holder: int | str) -> None:
        """Record a hop: move the packet to *next_holder*.

        Args:
            next_holder: Drone ID (int) or 'GS' for the ground station.
        """
        self.current_holder = next_holder
        self.hop_count += 1
        self.path.append(next_holder)

    def mark_delivered(self, timestep: int) -> None:
        """Mark the packet as successfully delivered at *timestep*.

        Args:
            timestep: The simulation step at which delivery occurs.
        """
        self.delivered = True
        self.delivered_at = timestep

    def mark_dropped(self, reason: DropReason) -> None:
        """Mark the packet as dropped for *reason*.

        Args:
            reason: A DropReason enum value explaining why it was dropped.
        """
        self.dropped = True
        self.drop_reason = reason

    def delay(self) -> Optional[int]:
        """Return end-to-end delay in timesteps, or None if not delivered.

        Returns:
            Integer delay (delivered_at - created_at), or None.
        """
        if self.delivered and self.delivered_at is not None:
            return self.delivered_at - self.created_at
        return None

    def __repr__(self) -> str:
        status = "delivered" if self.delivered else ("dropped" if self.dropped else "in-flight")
        return (
            f"Packet(id={self.packet_id}, src={self.source_id}, "
            f"hops={self.hop_count}, status={status})"
        )


class PacketFactory:
    """Creates Packet instances with auto-incrementing IDs.

    Attributes:
        _next_id: Internal counter for packet IDs.
        ttl:      Default TTL used when creating packets.
        max_hops: Default hop limit used when creating packets.
        size:     Default packet size in bytes.
    """

    def __init__(self, ttl: int, max_hops: int, size_bytes: int) -> None:
        """Initialise the factory.

        Args:
            ttl:        Timestep lifetime for each packet.
            max_hops:   Maximum relay hops allowed.
            size_bytes: Payload size in bytes.
        """
        self._next_id: int = 0
        self.ttl = ttl
        self.max_hops = max_hops
        self.size = size_bytes

    def create(self, source_id: int, created_at: int) -> Packet:
        """Create and return a new Packet.

        Args:
            source_id:  ID of the originating drone.
            created_at: Current simulation timestep.

        Returns:
            A freshly initialised Packet object.
        """
        pkt = Packet(
            packet_id=self._next_id,
            source_id=source_id,
            created_at=created_at,
            ttl=self.ttl,
            max_hops=self.max_hops,
            size_bytes=self.size,
        )
        self._next_id += 1
        return pkt

    def reset(self) -> None:
        """Reset the packet ID counter (call at episode start)."""
        self._next_id = 0
