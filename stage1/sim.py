"""Queued packet simulation over a frozen graph and a fixed routing table.

Per 100 ms step:
  1. every source (M-drone) appends one new packet to its own FIFO queue;
  2. every drone with a non-empty queue attempts to transmit exactly its
     head-of-queue packet: with next hop v it draws r ~ U(0,1) and the
     packet is dropped permanently if r < p_loss(u->v), otherwise it is
     delivered (v == GS) or handed to v's queue; a drone whose next hop is
     None drops its head packet instead ("no route");
  3. all hand-offs land at the END of the step, so a packet moves at most
     one hop per step and the outcome is independent of iteration order.

Determinism: drones transmit in ascending id order and one uniform draw is
consumed per queue-active drone per step, so a given rng seed fully
reproduces the episode. Same-step arrivals join a queue in sender-id order.

Delay accounting: a packet emitted in step ``e`` and resolved in step ``x``
spent ``x - e + 1`` steps in the network; every one of those steps was
either its one successful transmission on some hop or a step spent waiting
in a queue, so for delivered packets ``delay == hops + queue_wait`` exactly.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import networkx as nx
import numpy as np

from .routing import NextHops

# Packet status codes
IN_FLIGHT: int = 0
DELIVERED: int = 1
DROPPED_CHANNEL: int = 2
DROPPED_NO_ROUTE: int = 3


@dataclass(frozen=True)
class SimResult:
    """Per-packet outcome arrays plus the queue-depth history.

    All packet arrays are indexed by packet id (emission order). ``end_step``
    is -1 and ``end_node`` is -1 for packets still in flight at episode end.
    ``queue_depths[t, i]`` is drone ``drones[i]``'s queue depth at the end of
    step t.
    """

    n_steps: int
    gs_id: int
    drones: tuple[int, ...]
    src: np.ndarray        # (n_packets,) emitting drone id
    emit_step: np.ndarray  # (n_packets,)
    end_step: np.ndarray   # (n_packets,) delivery/drop step, -1 if in flight
    end_node: np.ndarray   # (n_packets,) GS if delivered, dropping drone if dropped
    hops: np.ndarray       # (n_packets,) successful transmissions so far
    status: np.ndarray     # (n_packets,) one of the status codes above
    queue_depths: np.ndarray  # (n_steps, n_drones) int32

    @property
    def resolved(self) -> np.ndarray:
        """Mask of packets that were delivered or dropped (not in flight)."""
        return self.status != IN_FLIGHT

    @property
    def delivered(self) -> np.ndarray:
        return self.status == DELIVERED

    @property
    def delay_steps(self) -> np.ndarray:
        """Total steps in network for resolved packets (undefined elsewhere)."""
        return self.end_step - self.emit_step + 1

    @property
    def queue_wait_steps(self) -> np.ndarray:
        """Steps spent waiting in queues; exact for delivered packets."""
        return self.delay_steps - self.hops


def run_sim(
    graph: nx.DiGraph,
    next_hops: NextHops,
    rng: np.random.Generator,
    n_steps: int,
    sources: Optional[Sequence[int]] = None,
) -> SimResult:
    """Run one episode; ``sources`` defaults to all nodes with kind == "M"."""
    gs_id: int = graph.graph["gs_id"]
    drones = tuple(sorted(n for n in graph.nodes if n != gs_id))
    index: Dict[int, int] = {d: i for i, d in enumerate(drones)}
    if sources is None:
        sources = [d for d in drones if graph.nodes[d].get("kind", "M") == "M"]
    sources = sorted(sources)

    hop_of = np.full(len(drones), -1, dtype=np.int64)      # -1 = no route
    loss_of = np.zeros(len(drones), dtype=float)
    for i, u in enumerate(drones):
        v = next_hops.get(u)
        if v is not None:
            assert graph.has_edge(u, v), f"next hop {u}->{v} is not a graph edge"
            hop_of[i] = v
            loss_of[i] = graph.edges[u, v]["p_loss"]

    queues: list[deque[int]] = [deque() for _ in drones]
    src_l: list[int] = []
    emit_l: list[int] = []
    end_l: list[int] = []
    end_node_l: list[int] = []
    hops_l: list[int] = []
    status_l: list[int] = []
    depths = np.zeros((n_steps, len(drones)), dtype=np.int32)

    for step in range(n_steps):
        for s in sources:  # emission phase
            pid = len(src_l)
            src_l.append(s)
            emit_l.append(step)
            end_l.append(-1)
            end_node_l.append(-1)
            hops_l.append(0)
            status_l.append(IN_FLIGHT)
            queues[index[s]].append(pid)

        active = [i for i in range(len(drones)) if queues[i]]
        draws = rng.random(len(active))  # one attempt per active drone
        arrivals: list[tuple[int, int]] = []  # (receiver index, packet id)
        for a, i in enumerate(active):
            pid = queues[i].popleft()
            v = hop_of[i]
            if v < 0:  # no route: drop-on-no-progress
                status_l[pid] = DROPPED_NO_ROUTE
                end_l[pid] = step
                end_node_l[pid] = drones[i]
            elif draws[a] < loss_of[i]:  # lost on air: no retransmission
                status_l[pid] = DROPPED_CHANNEL
                end_l[pid] = step
                end_node_l[pid] = drones[i]
            else:
                hops_l[pid] += 1
                if v == gs_id:
                    status_l[pid] = DELIVERED
                    end_l[pid] = step
                    end_node_l[pid] = gs_id
                else:
                    arrivals.append((index[v], pid))

        for j, pid in arrivals:  # hand-offs land after all transmissions
            queues[j].append(pid)
        depths[step] = [len(q) for q in queues]

    return SimResult(
        n_steps=n_steps,
        gs_id=gs_id,
        drones=drones,
        src=np.asarray(src_l, dtype=np.int64),
        emit_step=np.asarray(emit_l, dtype=np.int64),
        end_step=np.asarray(end_l, dtype=np.int64),
        end_node=np.asarray(end_node_l, dtype=np.int64),
        hops=np.asarray(hops_l, dtype=np.int64),
        status=np.asarray(status_l, dtype=np.int64),
        queue_depths=depths,
    )
