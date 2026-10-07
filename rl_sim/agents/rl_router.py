"""
rl_router.py — The "rl" routing rule: one small network, shared by all drones.

The network scores a single option from three LOCAL inputs and estimates the
chance a packet sent that way reaches the GS:

    3 -> RL_HIDDEN -> RL_HIDDEN -> 1,  ReLU, sigmoid output

Inputs, in order: the link's measured delivery rate, the fill of the deciding
drone's own queue for that link, and the link's channel loss. The drone picks
the option with the highest output. While training it instead explores
uniformly with probability ``config.RL_EPSILON_TRAIN``; at evaluation there is
no exploration, though the per-link delivery rates keep updating from ACKs —
those are observations, not learning.

Learning
--------
Every decision becomes one training example once its packet resolves: target 1
if the GS acknowledged it, 0 if its TTL ran out. Examples accumulate across
steps; once at least ``config.RL_BATCH_SIZE`` have gathered, one Adam step is
taken on their binary cross-entropy and the batch is cleared. No replay
buffer, no target network.

Pooling every drone's examples into ONE shared network is an ASSUMPTION
pending supervisor confirmation. Each drone still decides from its own local
inputs only; nothing in a decision reads another drone's state.

This is the only module in the project that imports torch.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import nn

from fanet_sim import config
from fanet_sim.envs.drone import Drone, NextHop

from agents.rate_router import BaseRouter
from agents.tracker import Example, Features, Tracker


class LinkScoreNet(nn.Module):
    """Estimates P(delivered) for one option from its three local inputs."""

    def __init__(self, hidden: Optional[int] = None) -> None:
        """Build the network.

        Args:
            hidden: Width of both hidden layers. Defaults to
                    ``config.RL_HIDDEN``.
        """
        super().__init__()
        if hidden is None:
            hidden = config.RL_HIDDEN
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score a batch of options.

        Args:
            features: An (N, 3) tensor of local inputs.

        Returns:
            An (N,) tensor of probabilities in (0, 1).
        """
        return torch.sigmoid(self.net(features)).squeeze(-1)


class RLRouter(BaseRouter):
    """Route by the network's estimate, and learn from resolved packets.

    Attributes:
        net:       The shared scoring network.
        training:  Whether to explore and to take gradient steps.
        epsilon:   Exploration probability while training.
        pending:   Resolved examples not yet used in a gradient step.
        batch_losses: The BCE of every gradient step taken this run.
    """

    name = "rl"

    def __init__(
        self,
        run_seed: int,
        net: Optional[LinkScoreNet] = None,
        tracker: Optional[Tracker] = None,
        training: bool = False,
        init_seed: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Create the RL router.

        Args:
            run_seed:      The run's channel/traffic seed.
            net:           An existing network to carry on with, or None to
                           build a fresh one.
            tracker:       An existing tracker, or None for a fresh one.
            training:      True to explore and learn; False to evaluate.
            init_seed:     Seeds torch before building a fresh network, so an
                           initialisation is reproducible. Ignored when *net*
                           is supplied.
            learning_rate: Adam learning rate. Defaults to
                           ``config.RL_LEARNING_RATE``.
        """
        super().__init__(run_seed, tracker)

        if net is None:
            if init_seed is not None:
                torch.manual_seed(init_seed)
            net = LinkScoreNet()
        self.net = net

        self.training = training
        self.epsilon = (
            config.RL_EPSILON_TRAIN if training else config.RL_EPSILON_EVAL
        )
        self.learning_rate = (
            config.RL_LEARNING_RATE if learning_rate is None else learning_rate
        )
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.learning_rate
        )
        self.loss_fn = nn.BCELoss()

        self.pending: List[Example] = []
        self.batch_losses: List[float] = []

    # ------------------------------------------------------------------
    # The choice
    # ------------------------------------------------------------------

    def _choose(
        self,
        drone: Drone,
        keys: Sequence[NextHop],
        feature_rows: Sequence[Features],
    ) -> NextHop:
        """Take the highest-scoring option, or explore while training.

        Args:
            drone:        The drone deciding.
            keys:         The allowed options, in fixed order.
            feature_rows: Their local inputs, in the same order.

        Returns:
            The chosen key.
        """
        if self.epsilon > 0.0 and self.rng.random() < self.epsilon:
            return keys[int(self.rng.integers(len(keys)))]

        with torch.no_grad():
            scores = self.net(torch.tensor(feature_rows, dtype=torch.float32))
        # argmax over a fixed key order, so ties break deterministically.
        return keys[int(torch.argmax(scores).item())]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _learn(self, examples: List[Example]) -> None:
        """Take a gradient step once enough examples have resolved.

        Args:
            examples: Newly resolved ``(features, target)`` pairs.
        """
        if not self.training or not examples:
            return

        self.pending.extend(examples)
        if len(self.pending) < config.RL_BATCH_SIZE:
            return

        features = torch.tensor(
            [row for row, _ in self.pending], dtype=torch.float32
        )
        targets = torch.tensor(
            [target for _, target in self.pending], dtype=torch.float32
        )
        self.pending = []

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.net(features), targets)
        loss.backward()
        self.optimizer.step()
        self.batch_losses.append(float(loss.item()))

    def mean_batch_loss(self) -> Optional[float]:
        """Return the mean BCE over this run's gradient steps, or None."""
        if not self.batch_losses:
            return None
        return sum(self.batch_losses) / len(self.batch_losses)

    def on_reset(self) -> None:
        """Clear per-run state. The network and optimizer carry over."""
        super().on_reset()
        self.pending = []
        self.batch_losses = []

    # ------------------------------------------------------------------
    # Saving and loading
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the network weights to *path*."""
        torch.save(self.net.state_dict(), path)

    @staticmethod
    def load_net(path: str) -> LinkScoreNet:
        """Return a network with the weights stored at *path*."""
        net = LinkScoreNet()
        net.load_state_dict(torch.load(path, weights_only=True))
        return net
