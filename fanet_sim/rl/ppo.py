"""
ppo.py — PPO trainer for the two drone policies.

Implements Proximal Policy Optimization (clipped surrogate objective + GAE
advantages) for the policies defined in ``fanet_sim.envs.policies``:

  * the **K-link** selector (``LinkScorePolicy`` + ``LinkValue``) on every
    drone, whose action is a Plackett–Luce draw of which K links to keep, and
  * the **topology** mover (``TopologyPolicy`` + ``TopologyValue``) on every
    C-drone, whose action is a diagonal-Gaussian [dx, dy] displacement.

Each drone trains its **own** policy independently: weights, critic, and Adam
optimiser state are kept per drone in :class:`PolicyBank`, which survives across
episodes (the environment recreates drones every reset, so the bank injects the
persistent networks back into each fresh drone). After every episode ``train.py``
hands the collected rollouts to :meth:`PolicyBank.update_link` and
:meth:`PolicyBank.update_topology`.

Rollout transition formats (plain dicts, built in ``drone.py`` / ``fanet_env``):
    link:  {cand_feats (N,5), value_feats (6,), selected_order [int],
            logp float, value float, reward float, done bool}
    topo:  {feats (8,), action (2,), logp float, value float,
            reward float, done bool}
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from fanet_sim import config
from fanet_sim.envs.policies import (
    LINK_SCORE_INPUT_DIM,
    LinkScorePolicy,
    LinkValue,
    TopologyPolicy,
    TopologyValue,
)
from fanet_sim.rl.sampling import plackett_luce_log_prob

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fanet_sim.envs.drone import Drone

_NEG: float = -1e9


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """PPO hyper-parameters shared by both policies."""

    lr: float = config.PPO_LR
    gamma: float = config.PPO_GAMMA
    lam: float = config.PPO_LAMBDA
    clip: float = config.PPO_CLIP
    epochs: int = config.PPO_EPOCHS
    value_coef: float = config.PPO_VALUE_COEF
    entropy_coef: float = config.PPO_ENTROPY_COEF


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages and value targets for one trajectory.

    The trajectory is the ordered list of transitions a single drone recorded
    over an episode (steps where it had no decision are simply absent). Only the
    final transition is marked done, so the value bootstrap runs through the
    whole sequence.

    Args:
        rewards: Per-transition rewards.
        values:  Per-transition value estimates V(s_t) from the critic.
        dones:   Per-transition terminal flags (only the last is True).
        gamma:   Discount factor.
        lam:     GAE smoothing parameter.

    Returns:
        A tuple ``(advantages, returns)`` of equal-length lists, where
        ``returns[t] = advantages[t] + values[t]`` are the critic targets.
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_adv = 0.0
    for t in reversed(range(n)):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = values[t + 1] if (t + 1 < n) else 0.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv
    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Standardise advantages to zero mean / unit variance (no-op if tiny)."""
    if x.numel() < 2:
        return x
    return (x - x.mean()) / (x.std() + 1e-8)


# ---------------------------------------------------------------------------
# Per-policy PPO updates
# ---------------------------------------------------------------------------

def update_link_policy(
    policy: LinkScorePolicy,
    value: LinkValue,
    optimizer: torch.optim.Optimizer,
    traj: List[dict],
    cfg: PPOConfig,
) -> Optional[float]:
    """Run a PPO update on one drone's K-link policy from its trajectory.

    Args:
        policy:    The drone's ``LinkScorePolicy`` (actor; produces logits).
        value:     The drone's ``LinkValue`` critic.
        optimizer: Adam over the policy and value parameters together.
        traj:      List of link transition dicts collected this episode.
        cfg:       PPO hyper-parameters.

    Returns:
        The mean absolute advantage (a rough learning-signal magnitude), or
        ``None`` if the trajectory was empty.
    """
    if not traj:
        return None

    rewards = [t["reward"] for t in traj]
    values = [t["value"] for t in traj]
    dones = [t["done"] for t in traj]
    advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.lam)

    n = len(traj)
    max_n = max(t["cand_feats"].shape[0] for t in traj)
    max_k = max(len(t["selected_order"]) for t in traj)

    feats = torch.zeros(n, max_n, LINK_SCORE_INPUT_DIM)
    feat_mask = torch.zeros(n, max_n, dtype=torch.bool)
    sel = torch.zeros(n, max_k, dtype=torch.long)
    sel_mask = torch.zeros(n, max_k, dtype=torch.bool)
    value_feats = torch.stack([t["value_feats"] for t in traj])  # (n, 6)

    for i, tr in enumerate(traj):
        cand = tr["cand_feats"]
        nc = cand.shape[0]
        feats[i, :nc] = cand
        feat_mask[i, :nc] = True
        order = tr["selected_order"]
        for j, idx in enumerate(order):
            sel[i, j] = idx
            sel_mask[i, j] = True

    old_logp = torch.tensor([t["logp"] for t in traj], dtype=torch.float32)
    adv_t = _normalize(torch.tensor(advantages, dtype=torch.float32))
    ret_t = torch.tensor(returns, dtype=torch.float32)

    for _ in range(cfg.epochs):
        logits = policy(feats)                                  # (n, max_n)
        logits = torch.where(feat_mask, logits, logits.new_full((), _NEG))
        new_logp, entropy = plackett_luce_log_prob(logits, sel, sel_mask, feat_mask)
        v_pred = value(value_feats)                             # (n,)

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(v_pred, ret_t)
        entropy_loss = -entropy.mean()

        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(adv_t.abs().mean().item())


def update_topology_policy(
    policy: TopologyPolicy,
    value: TopologyValue,
    optimizer: torch.optim.Optimizer,
    traj: List[dict],
    cfg: PPOConfig,
) -> Optional[float]:
    """Run a PPO update on one C-drone's topology policy from its trajectory.

    Args:
        policy:    The C-drone's ``TopologyPolicy`` (Gaussian actor).
        value:     The C-drone's ``TopologyValue`` critic.
        optimizer: Adam over the policy and value parameters together.
        traj:      List of topology transition dicts collected this episode.
        cfg:       PPO hyper-parameters.

    Returns:
        The mean absolute advantage, or ``None`` if the trajectory was empty.
    """
    if not traj:
        return None

    rewards = [t["reward"] for t in traj]
    values = [t["value"] for t in traj]
    dones = [t["done"] for t in traj]
    advantages, returns = compute_gae(rewards, values, dones, cfg.gamma, cfg.lam)

    feats = torch.stack([t["feats"] for t in traj])     # (n, 8)
    actions = torch.stack([t["action"] for t in traj])  # (n, 2)
    old_logp = torch.tensor([t["logp"] for t in traj], dtype=torch.float32)
    adv_t = _normalize(torch.tensor(advantages, dtype=torch.float32))
    ret_t = torch.tensor(returns, dtype=torch.float32)

    for _ in range(cfg.epochs):
        mean = policy(feats)                            # (n, 2)
        std = torch.exp(policy.log_std)
        dist = Normal(mean, std)
        new_logp = dist.log_prob(actions).sum(dim=-1)   # (n,)
        entropy = dist.entropy().sum(dim=-1)            # (n,)
        v_pred = value(feats)                           # (n,)

        ratio = torch.exp(new_logp - old_logp)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(v_pred, ret_t)
        entropy_loss = -entropy.mean()

        loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(adv_t.abs().mean().item())


# ---------------------------------------------------------------------------
# Policy bank — persistent per-drone networks + optimisers
# ---------------------------------------------------------------------------

class PolicyBank:
    """Owns every drone's trainable policies across episodes.

    The environment recreates ``Drone`` objects (and fresh random MLPs) on every
    ``reset``; :meth:`inject` overwrites those with the bank's persistent
    networks so learning carries over. Drone IDs are stable: ``0 .. num_m-1`` are
    M-drones, ``num_m .. num_m+num_c-1`` are C-drones, matching the environment.
    """

    def __init__(self, num_m: int, num_c: int, cfg: PPOConfig) -> None:
        """Build per-drone networks and their Adam optimisers.

        Args:
            num_m: Number of M-drones.
            num_c: Number of C-drones.
            cfg:   PPO hyper-parameters (provides the learning rate).
        """
        self.num_m = num_m
        self.num_c = num_c
        self.cfg = cfg

        total = num_m + num_c
        self.c_ids = list(range(num_m, num_m + num_c))

        # Link policy + critic on EVERY drone.
        self.link_policies: Dict[int, LinkScorePolicy] = {}
        self.link_values: Dict[int, LinkValue] = {}
        self.link_optims: Dict[int, torch.optim.Optimizer] = {}
        for did in range(total):
            pol, val = LinkScorePolicy(), LinkValue()
            self.link_policies[did] = pol
            self.link_values[did] = val
            self.link_optims[did] = torch.optim.Adam(
                chain(pol.parameters(), val.parameters()), lr=cfg.lr
            )

        # Topology policy + critic on C-drones only.
        self.topo_policies: Dict[int, TopologyPolicy] = {}
        self.topo_values: Dict[int, TopologyValue] = {}
        self.topo_optims: Dict[int, torch.optim.Optimizer] = {}
        for did in self.c_ids:
            pol, val = TopologyPolicy(), TopologyValue()
            self.topo_policies[did] = pol
            self.topo_values[did] = val
            self.topo_optims[did] = torch.optim.Adam(
                chain(pol.parameters(), val.parameters()), lr=cfg.lr
            )

    def inject(self, drones: List["Drone"]) -> None:
        """Replace each drone's policy networks with the bank's persistent ones.

        Args:
            drones: The freshly created drones for the current episode.
        """
        for d in drones:
            d.link_policy = self.link_policies[d.drone_id]
            d.link_value = self.link_values[d.drone_id]
            if d.drone_type == "C":
                d.topo_policy = self.topo_policies[d.drone_id]
                d.topo_value = self.topo_values[d.drone_id]

    def update_link(self, rollouts: Dict[int, List[dict]]) -> float:
        """PPO-update every drone's link policy. Returns the mean signal."""
        signals: List[float] = []
        for did, traj in rollouts.items():
            sig = update_link_policy(
                self.link_policies[did],
                self.link_values[did],
                self.link_optims[did],
                traj,
                self.cfg,
            )
            if sig is not None:
                signals.append(sig)
        return sum(signals) / len(signals) if signals else 0.0

    def update_topology(self, rollouts: Dict[int, List[dict]]) -> float:
        """PPO-update every C-drone's topology policy. Returns the mean signal."""
        signals: List[float] = []
        for did, traj in rollouts.items():
            sig = update_topology_policy(
                self.topo_policies[did],
                self.topo_values[did],
                self.topo_optims[did],
                traj,
                self.cfg,
            )
            if sig is not None:
                signals.append(sig)
        return sum(signals) / len(signals) if signals else 0.0

    def save(self, path: str) -> None:
        """Save all policy and critic weights to a single checkpoint file.

        Args:
            path: Destination ``.pt`` file path.
        """
        torch.save(
            {
                "num_m": self.num_m,
                "num_c": self.num_c,
                "link_policies": {k: v.state_dict() for k, v in self.link_policies.items()},
                "link_values": {k: v.state_dict() for k, v in self.link_values.items()},
                "topo_policies": {k: v.state_dict() for k, v in self.topo_policies.items()},
                "topo_values": {k: v.state_dict() for k, v in self.topo_values.items()},
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load weights previously written by :meth:`save`.

        Args:
            path: Source ``.pt`` file path.
        """
        ckpt = torch.load(path, map_location="cpu")
        for k, sd in ckpt["link_policies"].items():
            self.link_policies[int(k)].load_state_dict(sd)
        for k, sd in ckpt["link_values"].items():
            self.link_values[int(k)].load_state_dict(sd)
        for k, sd in ckpt["topo_policies"].items():
            self.topo_policies[int(k)].load_state_dict(sd)
        for k, sd in ckpt["topo_values"].items():
            self.topo_values[int(k)].load_state_dict(sd)
