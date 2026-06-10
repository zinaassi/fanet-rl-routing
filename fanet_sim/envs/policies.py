"""
policies.py — PyTorch MLP policies (and their PPO critics) for the drones.

The two actor MLPs are the K-link selector and the topology mover. They start
with random weights; ``train.py`` trains them with PPO. Each network is paired
with a small value network (critic) that PPO uses as a baseline.

LinkScorePolicy  (one instance per drone, all drones)
    Scores a single candidate neighbour from 5 local features. The drone keeps
    K links by sampling K candidates without replacement from these scores
    (used as logits); the deterministic eval path keeps the top-K instead.
        Input(5) -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> Linear(1)

    The 5 input features (see Drone._link_features) are:
        link quality, distance to neighbour, neighbour's distance to GS,
        neighbour's queue length, neighbour's number of current links.

LinkValue        (one instance per drone, all drones)
    PPO critic for the link policy. Maps a fixed 6-feature drone summary (see
    Drone._link_value_features) to a scalar state value.
        Input(6) -> Linear(16) -> ReLU -> Linear(16) -> ReLU -> Linear(1)

TopologyPolicy   (one instance per C-drone)
    Maps a C-drone's 10-feature local state to a movement vector [dx, dy]. For
    PPO the output is the MEAN of a diagonal Gaussian whose per-axis log-std is
    the learnable parameter ``log_std``; the eval path uses the mean directly.
        Input(10) -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(2)

    The 10 input features (see Drone._topology_features) are:
        pos_x, pos_y, vel_x, vel_y, num_neighbors, mean_link_quality,
        mean_distance_to_neighbours, queue_length,
        unit_dx_to_nearest_M, unit_dy_to_nearest_M.

TopologyValue    (one instance per C-drone)
    PPO critic for the topology policy. Maps the same 10-feature state to a
    scalar state value.
        Input(10) -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fanet_sim import config

# Architecture dimensions (kept here so they are defined in exactly one place).
LINK_SCORE_INPUT_DIM: int = 5
LINK_SCORE_HIDDEN_DIM: int = 16
LINK_VALUE_INPUT_DIM: int = 6
TOPOLOGY_INPUT_DIM: int = 10
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
            x: Tensor of shape ``(N, 5)`` (or ``(T, N, 5)``) — one row of
               features per candidate.

        Returns:
            Tensor of shape ``(N,)`` (or ``(T, N)``) — one scalar score per
            candidate. The scores are used as logits for sampling links.
        """
        return self.net(x).squeeze(-1)


class LinkValue(nn.Module):
    """Per-drone critic estimating the state value for the link policy."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LINK_VALUE_INPUT_DIM, LINK_SCORE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(LINK_SCORE_HIDDEN_DIM, LINK_SCORE_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(LINK_SCORE_HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate V(s) from the 6-feature drone summary.

        Args:
            x: Tensor of shape ``(6,)`` (or ``(T, 6)``).

        Returns:
            Tensor of shape ``()`` (or ``(T,)``) — the scalar value estimate.
        """
        return self.net(x).squeeze(-1)


class TopologyPolicy(nn.Module):
    """Per-C-drone MLP that maps an 8-feature state to a [dx, dy] move.

    For PPO the network output is the mean of a diagonal Gaussian over moves;
    :attr:`log_std` holds the (state-independent) per-axis log standard
    deviation and is a learnable parameter trained alongside the mean network.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOPOLOGY_INPUT_DIM, TOPOLOGY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN_DIM, TOPOLOGY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN_DIM, TOPOLOGY_OUTPUT_DIM),
        )
        # Learnable log standard deviation, one per output axis. Starts at 0.5
        # (std ≈ 1.65 m, ~half the 3 m max step) to give healthy initial
        # exploration so the policy discovers that moving toward mission drones
        # pays off; PPO shrinks it as the policy sharpens.
        self.log_std = nn.Parameter(torch.full((TOPOLOGY_OUTPUT_DIM,), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a state vector to a movement MEAN bounded to the max step.

        The raw network output is squashed with ``tanh`` and scaled by
        ``config.TOPOLOGY_MAX_STEP_M``, so the mean already spans the full
        ±max-step range per axis. This lets the policy command a full-speed move
        from modest network weights — the network only has to choose a
        *direction* (handed to it by the nearest-M bearing feature), not learn to
        scale its output up to metres. Without this the raw output stays tiny and
        the C-drones barely move.

        Args:
            x: Tensor of shape ``(10,)`` (or ``(N, 10)``).

        Returns:
            Tensor of shape ``(2,)`` (or ``(N, 2)``) — the mean [dx, dy] in
            metres, each component in ``[-TOPOLOGY_MAX_STEP_M, +…]``.
        """
        return torch.tanh(self.net(x)) * config.TOPOLOGY_MAX_STEP_M


class TopologyValue(nn.Module):
    """Per-C-drone critic estimating the state value for the topology policy."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TOPOLOGY_INPUT_DIM, TOPOLOGY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN_DIM, TOPOLOGY_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TOPOLOGY_HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate V(s) from the 8-feature topology state.

        Args:
            x: Tensor of shape ``(8,)`` (or ``(T, 8)``).

        Returns:
            Tensor of shape ``()`` (or ``(T,)``) — the scalar value estimate.
        """
        return self.net(x).squeeze(-1)
