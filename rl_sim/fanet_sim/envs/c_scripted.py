"""
c_scripted.py — The Phase-2a C-drone rules, expressed as agents.

Phase 2c M0 needs to know that the interface did not change the problem. The
way to know is to put the OLD rules through the NEW interface and check the
runs come out the same. These are those rules, written as agents: they take
the observation the network will take, and return the action the network will
return.

They are deliberately not clever. ``HoverAgent`` returns the drone's own
position; ``MidpointAgent`` is the Phase-2a midpoint rule, target sharing and
all. If a run under ``C_RULE = "rl"`` with one of these matches a run under
``C_RULE = "hover"`` or ``"midpoint"``, then the observation carries what the
rule needs and the action can express what the rule wants.

Torch-free, and part of ``fanet_sim`` rather than ``agents/`` for that
reason: nothing here learns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from fanet_sim import config
from fanet_sim.envs import c_interface


class HoverAgent:
    """Stay put: the waypoint is always where the drone already is."""

    def __init__(self, env: Any) -> None:
        """Remember the environment, for the GS position.

        Args:
            env: The environment this agent is attached to.
        """
        self.env = env

    def decide(
        self, observations: Sequence[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """Return one action per observation.

        Args:
            observations: One per C-drone, in drone-id order.

        Returns:
            The action that names each drone's current position.
        """
        actions = []
        for observation in observations:
            # The first two self features are the position over ACTION_SCALE_M.
            position = observation["self"][:2] * c_interface.ACTION_SCALE_M
            actions.append(
                c_interface.action_from_waypoint(position,
                                                 self.env.gs_position)
            )
        return actions


class MidpointAgent:
    """The Phase-2a midpoint rule, as an agent.

    Reads the same neighbour table through the environment, so the choice is
    identical to ``_choose_target``; the only difference is that the answer
    goes out as a normalised action instead of being written straight onto
    the drone.
    """

    def __init__(self, env: Any) -> None:
        """Remember the environment.

        Args:
            env: The environment this agent is attached to.
        """
        self.env = env

    def decide(
        self, observations: Sequence[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """Return one action per observation.

        Args:
            observations: One per C-drone, in drone-id order. Unused: the
                          rule reads the table directly, which is the same
                          information the observation is built from.

        Returns:
            One action per C-drone, or None for a drone the rule would leave
            hovering.
        """
        previous = config.C_RULE
        config.C_RULE = "midpoint"
        try:
            actions = []
            c_drones = [d for d in self.env.drones if d.drone_type == "C"]
            for drone in c_drones:
                target = self.env._choose_target(drone)
                if target is None:
                    # The rule found nothing. Answering None rather than
                    # "my own position" matters: a drone that announces a
                    # target changes what the OTHER C-drones are allowed to
                    # pick, through target sharing.
                    actions.append(None)
                    continue
                actions.append(
                    c_interface.action_from_waypoint(target,
                                                     self.env.gs_position)
                )
            return actions
        finally:
            config.C_RULE = previous


class ReplayAgent:
    """Plays back a fixed list of actions, for tests.

    Args:
        actions: One action per decision, cycled if the run is longer.
    """

    def __init__(self, actions: Sequence[np.ndarray]) -> None:
        self.actions = [np.asarray(a, dtype=np.float64) for a in actions]
        self.calls = 0

    def decide(
        self, observations: Sequence[Dict[str, np.ndarray]]
    ) -> List[np.ndarray]:
        """Return the next action for each observation.

        Args:
            observations: One per C-drone.

        Returns:
            One action per observation.
        """
        out = []
        for _ in observations:
            out.append(self.actions[self.calls % len(self.actions)])
            self.calls += 1
        return out
