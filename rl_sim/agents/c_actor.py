"""
c_actor.py — The C-drone policy network, and cloning midpoint with it.

Phase 2c M1. The network has to read a neighbour table, which is a SET: it
has no natural order and no fixed size, and the answer must not depend on
which order the entries happen to come in. So each entry goes through the
same small network, and the results are pooled with a mean and a max before
anything else looks at them. Mean says what the neighbourhood is like on
average; max says whether anything stands out, which is what a rule like
"fly to the farthest M-drone" needs.

    neighbour (16) --> 64 --> 64 ---\\
                                     mean, max  -->  128  --\\
    self      (16) --> 64 ------------------------------------>  192 --> 128 --> 2  --> tanh

The two outputs are the waypoint, relative to the GS and divided by 450 m,
which is exactly what ``c_interface.waypoint_from_action`` expects.

M1 only clones midpoint: the network is trained to predict the waypoint the
Phase-2a rule would have chosen, by least squares. There is no reward term
here and no exploration — the reward is logged by the environment and starts
being used in M2, where this network becomes the actor of a PPO pair.

This is the only Phase-2c file that imports torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from fanet_sim.envs import c_interface

#: Widths, kept in one place so the trainer and the agent cannot disagree.
NEIGHBOUR_HIDDEN: int = 64
SELF_HIDDEN: int = 64
JOINT_HIDDEN: int = 128


class CActor(nn.Module):
    """Maps one C-drone's observation to a waypoint.

    Attributes:
        neighbour: Applied to every table entry, the same weights each time.
        own:       Applied to the drone's own features.
        head:      Applied to the pooled neighbourhood and the self vector.
    """

    def __init__(self) -> None:
        """Build the three parts."""
        super().__init__()
        self.neighbour = nn.Sequential(
            nn.Linear(c_interface.NEIGHBOUR_FEATURES, NEIGHBOUR_HIDDEN),
            nn.ReLU(),
            nn.Linear(NEIGHBOUR_HIDDEN, NEIGHBOUR_HIDDEN),
            nn.ReLU(),
        )
        self.own = nn.Sequential(
            nn.Linear(c_interface.SELF_FEATURES, SELF_HIDDEN),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * NEIGHBOUR_HIDDEN + SELF_HIDDEN, JOINT_HIDDEN),
            nn.ReLU(),
            nn.Linear(JOINT_HIDDEN, 2),
            nn.Tanh(),
        )

    def forward(self, own: torch.Tensor, neighbours: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Return one waypoint per observation.

        Args:
            own:        (B, SELF_FEATURES).
            neighbours: (B, N, NEIGHBOUR_FEATURES).
            mask:       (B, N), 1 for a real entry and 0 for padding.

        Returns:
            (B, 2) in [-1, 1].
        """
        encoded = self.neighbour(neighbours)                  # (B, N, H)
        keep = mask.unsqueeze(-1)                             # (B, N, 1)

        # Padding must not reach either pool. For the mean it is zeroed and
        # divided out; for the max it is pushed to -inf, then any row with no
        # neighbours at all is put back to zero rather than left at -inf.
        total = (encoded * keep).sum(dim=1)
        count = keep.sum(dim=1).clamp(min=1.0)
        pooled_mean = total / count

        very_small = torch.finfo(encoded.dtype).min
        pooled_max = torch.where(keep.bool(), encoded,
                                 torch.full_like(encoded, very_small)
                                 ).max(dim=1).values
        empty = (mask.sum(dim=1) == 0).unsqueeze(-1)
        pooled_max = torch.where(empty, torch.zeros_like(pooled_max),
                                 pooled_max)

        joined = torch.cat([pooled_mean, pooled_max, self.own(own)], dim=-1)
        return self.head(joined)


def to_tensors(batch: Dict[str, np.ndarray],
               device: str = "cpu") -> Tuple[torch.Tensor, ...]:
    """Turn a padded observation batch into float32 tensors.

    Args:
        batch:  Output of :func:`c_interface.pad_observations`.
        device: Where to put them.

    Returns:
        ``(self, neighbours, mask)``.
    """
    return (
        torch.as_tensor(batch["self"], dtype=torch.float32, device=device),
        torch.as_tensor(batch["neighbours"], dtype=torch.float32,
                        device=device),
        torch.as_tensor(batch["mask"], dtype=torch.float32, device=device),
    )


class GaussianActor(nn.Module):
    """A :class:`CActor` plus one learned spread, for PPO.

    The network gives the MEAN waypoint; the spread is a single pair of
    numbers shared by every drone and every situation, which is the usual
    starting point and keeps the thing that is explored from depending on
    what is being observed. It starts at ``std = 0.1``, about 45 m on the
    ground — a little under the C-drones' safe distance, so exploration
    nudges rather than teleports.

    Attributes:
        mean:    The deterministic actor.
        log_std: The learned spread, in action units.
    """

    def __init__(self, start_std: float = 0.1) -> None:
        """Build the actor and its spread.

        Args:
            start_std: The initial standard deviation, in action units.
        """
        super().__init__()
        self.mean = CActor()
        self.log_std = nn.Parameter(
            torch.full((2,), float(np.log(start_std)))
        )

    def forward(self, own: torch.Tensor, neighbours: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Return the mean waypoint.

        Args:
            own:        (B, SELF_FEATURES).
            neighbours: (B, N, NEIGHBOUR_FEATURES).
            mask:       (B, N).

        Returns:
            (B, 2) in [-1, 1].
        """
        return self.mean(own, neighbours, mask)

    def distribution(self, own: torch.Tensor, neighbours: torch.Tensor,
                     mask: torch.Tensor) -> torch.distributions.Normal:
        """Return the action distribution for these observations.

        Args:
            own:        (B, SELF_FEATURES).
            neighbours: (B, N, NEIGHBOUR_FEATURES).
            mask:       (B, N).

        Returns:
            A diagonal Normal over the two action dimensions.
        """
        centre = self.forward(own, neighbours, mask)
        return torch.distributions.Normal(centre, self.log_std.exp())


class CCritic(nn.Module):
    """Values a moment from the whole network's state.

    Used ONLY while training. It is deliberately a plain MLP: the thing it
    reads is already a fixed-length vector, so none of the actor's set
    handling applies.

    Attributes:
        body: 2 x 128 with ReLU, then one output.
    """

    def __init__(self, width: int, hidden: int = 128) -> None:
        """Build the critic.

        Args:
            width:  Length of the global state vector.
            hidden: Width of both hidden layers.
        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return one value per state.

        Args:
            state: (B, width).

        Returns:
            (B,) values.
        """
        return self.body(state).squeeze(-1)


class RunningNorm:
    """Tracks a mean and variance for normalising the value target.

    PPO's value loss is badly scaled when the returns are, and these returns
    start near zero and grow. Keeping a running estimate and dividing by it
    keeps the critic's job the same size throughout.

    Attributes:
        mean:  Running mean.
        var:   Running variance.
        count: How many samples have gone in.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, values: np.ndarray) -> None:
        """Fold a batch into the estimate.

        Args:
            values: The new samples.
        """
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = len(values)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total
                    ) / total
        self.count = total

    def normalise(self, values: np.ndarray) -> np.ndarray:
        """Return *values* in units of the running spread.

        Args:
            values: Raw returns.

        Returns:
            The normalised returns.
        """
        return (values - self.mean) / (np.sqrt(self.var) + 1e-8)

    def denormalise(self, values: np.ndarray) -> np.ndarray:
        """Undo :meth:`normalise`.

        Args:
            values: Normalised values.

        Returns:
            Raw values.
        """
        return values * (np.sqrt(self.var) + 1e-8) + self.mean


class WaypointAgent:
    """Runs a trained :class:`CActor` as the C-drones' agent.

    Args:
        actor: The network to run. Put into eval mode and never trained here.
    """

    def __init__(self, actor: CActor) -> None:
        self.actor = actor
        self.actor.eval()

    def decide(
        self, observations: Sequence[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """Return one action per observation.

        The whole batch goes through the network at once, which is both
        faster and exactly what the trainer did, so a drone's answer cannot
        depend on how many drones asked.

        Args:
            observations: One per C-drone.

        Returns:
            One (2,) action per observation.
        """
        if not observations:
            return []
        batch = c_interface.pad_observations(list(observations))
        with torch.no_grad():
            out = self.actor(*to_tensors(batch))
        return [row.astype(np.float64) for row in out.numpy()]


def train_clone(
    observations: Sequence[Dict[str, np.ndarray]],
    actions: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    holdout: float = 0.10,
    seed: int = 0,
    log: Optional[Any] = None,
) -> Tuple[CActor, Dict[str, Any]]:
    """Fit a :class:`CActor` to someone else's waypoints.

    Least squares on the action, which is the waypoint over 450 m, so a
    reported error of 0.01 is 4.5 m on the ground. The held-out split is by
    DECISION rather than by layout: M1 is asking whether the network can
    express the rule at all, not whether it carries to new layouts — that is
    what the validation layouts are for, and they are never in this data.

    Args:
        observations:  One per decision.
        actions:       (N, 2), the waypoints to copy.
        epochs:        Passes over the training part.
        batch_size:    Decisions per step.
        learning_rate: For Adam.
        holdout:       Share kept back for checking.
        seed:          Seeds the split and the network's initial weights.
        log:           Optional callable for progress lines.

    Returns:
        The trained network, and a dict of the training record.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    width = max((len(o["mask"]) for o in observations), default=1) or 1
    batch = c_interface.pad_observations(list(observations), width=width)
    own, neighbours, mask = to_tensors(batch)
    targets = torch.as_tensor(np.asarray(actions, dtype=np.float64),
                              dtype=torch.float32)

    order = rng.permutation(len(observations))
    cut = int(round(len(order) * (1.0 - holdout)))
    train_index = torch.as_tensor(order[:cut].copy())
    test_index = torch.as_tensor(order[cut:].copy())

    actor = CActor()
    optimiser = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        actor.train()
        shuffled = train_index[torch.randperm(len(train_index))]
        running = 0.0
        for start in range(0, len(shuffled), batch_size):
            picked = shuffled[start:start + batch_size]
            predicted = actor(own[picked], neighbours[picked], mask[picked])
            loss = loss_fn(predicted, targets[picked])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss) * len(picked)
        train_loss = running / max(1, len(shuffled))

        actor.eval()
        with torch.no_grad():
            held = actor(own[test_index], neighbours[test_index],
                         mask[test_index])
            test_loss = float(loss_fn(held, targets[test_index]))
            metres = float(torch.linalg.norm(
                (held - targets[test_index]) * c_interface.ACTION_SCALE_M,
                dim=-1
            ).mean())
        history.append({"epoch": epoch, "train_mse": train_loss,
                        "holdout_mse": test_loss, "holdout_metres": metres})
        if log is not None:
            log(f"    epoch {epoch:>3}  train MSE {train_loss:.6f}  "
                f"held-out MSE {test_loss:.6f}  "
                f"mean error {metres:7.2f} m")

    return actor, {
        "history": history,
        "decisions": len(observations),
        "train_decisions": int(len(train_index)),
        "holdout_decisions": int(len(test_index)),
        "neighbour_slots": width,
        "final": history[-1] if history else {},
    }
