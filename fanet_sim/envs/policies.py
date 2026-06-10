"""
policies.py — Randomly-initialised PyTorch MLP policies for the drones.

Two small multilayer perceptrons, both **untrained** — their weights are
random at construction and there is no learning yet. They exist to wire the
full neural pipeline end-to-end; a later phase will train them.

LinkScorePolicy  (one instance per drone, all drones)
    Scores a single candidate neighbour from 5 local features; the drone keeps
    its top-K highest-scoring candidates as active links.
        Input(5) -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> Linear(1)

    The 5 input features (see Drone._link_features) are:
        link quality, distance to neighbour, neighbour's distance to GS,
        neighbour's queue length, neighbour's number of current links.

TopologyPolicy   (one instance per C-drone)
    Maps a C-drone's 8-feature local state to a movement vector [dx, dy].
        Input(8) -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(2)

    The 8 input features (see Drone.policy_move) are:
        pos_x, pos_y, vel_x, vel_y, num_neighbors, mean_link_quality,
        mean_distance_to_neighbours, queue_length.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Architecture dimensions (kept here so they are defined in exactly one place).
LINK_SCORE_INPUT_DIM: int = 5
LINK_SCORE_HIDDEN_DIM: int = 16
TOPOLOGY_INPUT_DIM: int = 8
TOPOLOGY_HIDDEN_DIM: int = 32
TOPOLOGY_OUTPUT_DIM: int = 2


class LinkScorePolicy(nn.Module):
    """Per-drone MLP that scores one candidate link (higher = keep)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LINK_SCORE_INPUT_DIM, LINK_SCORE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(LINK_SCORE_HIDDEN_DIM, LINK_SCORE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(LINK_SCORE_HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of candidates.

        Args:
            x: Tensor of shape ``(N, 5)`` — one row of features per candidate.

        Returns:
            Tensor of shape ``(N,)`` — one scalar score per candidate.
        """
        return self.net(x).squeeze(-1)


class TopologyPolicy(nn.Module):
    """Per-C-drone MLP that maps an 8-feature state to a [dx, dy] move."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOPOLOGY_INPUT_DIM, TOPOLOGY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN_DIM, TOPOLOGY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN_DIM, TOPOLOGY_OUTPUT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a state vector to a raw (unclamped) movement vector.

        Args:
            x: Tensor of shape ``(8,)`` (or ``(N, 8)``).

        Returns:
            Tensor of shape ``(2,)`` (or ``(N, 2)``) — the raw [dx, dy].
        """
        return self.net(x)
