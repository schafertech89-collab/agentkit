"""Dyna-Q+ planner.

After every real environment step we:

  1. Update the Q-table with the realised (s, a, r, s') tuple.
  2. Update the learned ``MarketDynamicsModel``.
  3. Run ``k`` planning backups: sample previously-visited (s, a), draw a
     simulated (r, s') from the model, and apply a Q-update with Sutton's
     novelty bonus

         r_plus = r + kappa * sqrt(tau)

     where ``tau`` is the number of steps since (s, a) was last *really*
     sampled. The bonus encourages the policy to re-probe stale parts of
     the action space — exactly the "Dyna-Q+" trick from RL: An Introduction
     §8.3.

The action space here is ``{-1, 0, +1}`` (short / flat / long); the planner
returns a recommended bias that the risk agent can blend with the fused
stance.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..analysis.market_dynamics_model import (
    Action,
    MarketDynamicsModel,
    State,
    stance_bucket,
)


@dataclass
class DynaQPlusConfig:
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1
    kappa: float = 1e-3  # novelty-bonus weight
    plan_steps: int = 25
    actions: Tuple[Action, ...] = (-1, 0, 1)


class DynaQPlus:
    def __init__(self, cfg: DynaQPlusConfig | None = None,
                 model: MarketDynamicsModel | None = None) -> None:
        self.cfg = cfg or DynaQPlusConfig()
        self.model = model or MarketDynamicsModel()
        self.q: dict[Tuple[State, Action], float] = defaultdict(float)
        self._last_seen: dict[Tuple[State, Action], int] = {}
        self.steps = 0

    # ---------------------------------------------------------- choose
    def best_action(self, state: State) -> Action:
        best = self.cfg.actions[0]
        best_q = -math.inf
        for a in self.cfg.actions:
            v = self.q[(state, a)]
            if v > best_q:
                best_q = v
                best = a
        return best

    def choose(self, state: State, rng: np.random.Generator | None = None) -> Action:
        rng = rng or np.random.default_rng()
        if rng.random() < self.cfg.epsilon:
            return int(rng.choice(self.cfg.actions))
        return self.best_action(state)

    # ---------------------------------------------------------- learn
    def step(self, state: State, action: Action, reward: float, next_state: State,
             *, plan: bool = True, rng: np.random.Generator | None = None) -> dict:
        """Apply one real Q-update + ``plan_steps`` planning backups."""
        rng = rng or np.random.default_rng()
        self.steps += 1
        self._last_seen[(state, action)] = self.steps
        # Real Q-update.
        target = reward + self.cfg.gamma * max(self.q[(next_state, a)] for a in self.cfg.actions)
        self.q[(state, action)] += self.cfg.alpha * (target - self.q[(state, action)])
        # Update model.
        self.model.record(state, action, reward, next_state)

        if plan:
            self._plan(rng)
        return {
            "q_value": self.q[(state, action)],
            "best_action": self.best_action(next_state),
            "model_size": len(self.model.known_pairs()),
        }

    # ---------------------------------------------------------- plan
    def _plan(self, rng: np.random.Generator) -> None:
        pairs = list(self.model.known_pairs())
        if not pairs:
            return
        for _ in range(self.cfg.plan_steps):
            idx = int(rng.integers(0, len(pairs)))
            state, action = pairs[idx]
            sim_r, sim_next = self.model.sample(state, action, rng=rng)
            tau = self.steps - self._last_seen.get((state, action), self.steps)
            bonus = self.cfg.kappa * math.sqrt(max(0, tau))
            r_plus = sim_r + bonus
            target = r_plus + self.cfg.gamma * max(
                self.q[(sim_next, a)] for a in self.cfg.actions
            )
            self.q[(state, action)] += self.cfg.alpha * (target - self.q[(state, action)])

    # ---------------------------------------------------------- view
    def policy_bias(self, state: State) -> float:
        """Return the planner's policy bias in [-1, 1] for the given state."""
        return float(self.best_action(state))

    def snapshot(self) -> dict:
        return {
            "q": [
                {"state": list(s), "action": int(a), "q": q}
                for (s, a), q in self.q.items()
            ],
            "steps": self.steps,
            "model": self.model.snapshot(),
        }


__all__ = ["DynaQPlus", "DynaQPlusConfig", "stance_bucket"]
