"""
agents — Phase-1b routing rules.

The ONLY place torch is allowed. ``fanet_sim/`` stays torch-free and never
imports from here; a router object is injected into :class:`FANETEnv`, which
calls it through five duck-typed hooks.

Two rules live here:

* ``rate`` — no network. Picks the link with the best measured delivery rate
  that still has queue room, with a little exploration.
* ``rl``   — a small network shared by every drone, scoring each option from
  three local inputs and learning from end-to-end ACKs.

Both are built through :func:`make_router`.
"""

from __future__ import annotations

from typing import Optional

from agents.rate_router import BaseRouter, RateRouter
from agents.tracker import Tracker

__all__ = ["BaseRouter", "RateRouter", "Tracker", "make_router"]


def make_router(
    name: str,
    run_seed: int,
    training: bool = False,
    init_seed: Optional[int] = None,
    net: Optional[object] = None,
) -> BaseRouter:
    """Build the router for a routing rule.

    ``rl`` is imported lazily, so using ``rate`` never pulls in torch.

    Args:
        name:      "rate" or "rl".
        run_seed:  The run's channel/traffic seed.
        training:  For "rl", whether to explore and learn.
        init_seed: For "rl", seeds a fresh network's initialisation.
        net:       For "rl", an existing network to carry on with.

    Returns:
        The router.

    Raises:
        ValueError: If *name* is not a rule built here.
    """
    if name == "rate":
        return RateRouter(run_seed=run_seed)
    if name == "rl":
        from agents.rl_router import RLRouter

        return RLRouter(
            run_seed=run_seed,
            net=net,                      # type: ignore[arg-type]
            training=training,
            init_seed=init_seed,
        )
    raise ValueError(f"no router called {name!r}; expected 'rate' or 'rl'")
