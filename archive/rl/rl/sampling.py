"""
sampling.py — Differentiable subset-selection maths for the K-link PPO policy.

The K-link policy must pick *which* K of N candidate links to keep. The
stochastic version of "keep the top-K scores" is **Plackett–Luce sampling**:
draw K items one at a time, each time sampling a candidate with probability
proportional to ``softmax(remaining scores)``. The log-probability of the whole
ordered draw is the sum of the per-draw categorical log-probabilities. This
module provides:

* :func:`sample_k_without_replacement` — used during rollout collection to pick
  the K links and report the chosen order plus its log-probability, and
* :func:`plackett_luce_log_prob` — used during the PPO update to re-evaluate the
  log-probability (and entropy) of a stored selection under the current policy,
  batched over a whole trajectory.

Both functions are pure tensor maths and import nothing from the simulator, so
``drone.py`` can call them without creating an import cycle.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

# Finite stand-in for -inf used to mask unavailable candidates. Large enough
# that masked items get ~0 probability, but finite so log-sum-exp never returns
# NaN (which a true -inf row would).
_NEG: float = -1e9


def sample_k_without_replacement(
    logits: torch.Tensor, k: int
) -> Tuple[List[int], float]:
    """Sample K candidates without replacement (Plackett–Luce).

    Draws ``k`` items sequentially; at each draw the next item is sampled with
    probability ``softmax`` over the candidates not yet chosen. ``k`` is capped
    at the number of candidates available.

    Args:
        logits: 1-D tensor of shape ``(N,)`` — one score per candidate.
        k:      Number of candidates to keep.

    Returns:
        A tuple ``(order, log_prob)`` where ``order`` is the list of chosen
        candidate indices in the order they were drawn, and ``log_prob`` is the
        scalar log-probability of that ordered draw under the current policy.
    """
    n = logits.shape[0]
    k = min(int(k), n)
    avail = torch.ones(n, dtype=torch.bool)
    order: List[int] = []
    log_prob = 0.0

    for _ in range(k):
        masked = torch.where(avail, logits, logits.new_full((), _NEG))
        log_probs = F.log_softmax(masked, dim=0)
        probs = log_probs.exp()
        idx = int(torch.multinomial(probs, 1).item())
        log_prob += float(log_probs[idx].item())
        order.append(idx)
        avail[idx] = False

    return order, log_prob


def plackett_luce_log_prob(
    logits: torch.Tensor,
    sel: torch.Tensor,
    sel_mask: torch.Tensor,
    feat_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-evaluate stored selections under current logits (batched over time).

    Recomputes, for each timestep in a trajectory, the Plackett–Luce
    log-probability and entropy of a previously sampled ordered selection. This
    is the differentiable quantity PPO needs to form the importance ratio and
    the entropy bonus.

    Args:
        logits:    Tensor ``(T, N)`` of candidate scores; entries for padded
                   (non-existent) candidates must already be set to a very
                   negative value via ``feat_mask``.
        sel:       Long tensor ``(T, K)`` of chosen candidate indices per step,
                   in draw order, padded with 0 where a step kept fewer than K.
        sel_mask:  Bool tensor ``(T, K)``; True where ``sel`` is a real draw.
        feat_mask: Bool tensor ``(T, N)``; True where the candidate exists.

    Returns:
        A tuple ``(log_prob, entropy)`` of tensors, each shape ``(T,)`` — the
        total log-probability and summed per-draw entropy for each step.
    """
    t, _ = logits.shape
    k = sel.shape[1]
    avail = feat_mask.clone()
    rows = torch.arange(t)
    log_prob = torch.zeros(t)
    entropy = torch.zeros(t)

    for j in range(k):
        masked = torch.where(avail, logits, logits.new_full((), _NEG))
        log_z = torch.logsumexp(masked, dim=1)              # (T,)
        log_probs = masked - log_z.unsqueeze(1)             # (T, N) log p_i
        probs = log_probs.exp()
        # Per-draw entropy over the still-available candidates.
        step_entropy = -torch.where(
            avail, probs * log_probs, torch.zeros_like(probs)
        ).sum(dim=1)

        chosen = sel[:, j]                                  # (T,)
        chosen_log_prob = log_probs[rows, chosen]           # (T,)
        real = sel_mask[:, j].to(log_prob.dtype)            # 1.0 / 0.0

        log_prob = log_prob + chosen_log_prob * real
        entropy = entropy + step_entropy * real

        # Remove the chosen candidate from the available pool (real draws only).
        # Build the new mask OUT OF PLACE: `avail` was just consumed by
        # ``torch.where`` above, which saves it for backward, so it must not be
        # mutated in place.
        remove = torch.zeros_like(avail)
        remove[rows, chosen] = sel_mask[:, j]
        avail = avail & ~remove

    return log_prob, entropy
