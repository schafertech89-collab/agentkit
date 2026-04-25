"""TD(lambda) stance learner.

Online learner that replaces the static ``STANCE_WEIGHTS`` dict in the risk
agent with weights tuned by realised PnL. The state representation phi(s) is a
linear concatenation of per-agent stance, regime one-hot, realised vol, and
two trend proxies. The learner keeps two parameter vectors:

    w  : weight vector for the linear value function V(s) = w . phi(s)
    e  : eligibility trace, accumulating phi(s) under the lambda-return rule

At every step we receive a reward (realised PnL minus a turnover/inventory
penalty) and update::

    delta = r + gamma * V(s')   - V(s)
    e     = gamma * lambda * e  + phi(s)
    w     = w + alpha * delta * e

The exposed ``stance_weights()`` projection re-derives the per-agent weights
the risk agent consumes by averaging the learned coefficients of the slots
that correspond to each agent stance, rectified and normalised.

This module is dependency-light (numpy only). It mirrors the torch-optional
fallback pattern of ``ppo_policy.py`` but does not need torch at all because
the learner is linear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


REGIMES = ("trend_up", "trend_down", "chop", "crisis", "squeeze")
DEFAULT_AGENTS = ("technical", "fundamental", "quant", "monte_carlo", "knn",
                  "cta_trend", "crossvenue_arb", "orderflow")


@dataclass
class TDLambdaConfig:
    alpha: float = 0.05
    gamma: float = 0.95
    lam: float = 0.7
    weight_floor: float = 0.05
    agents: Sequence[str] = field(default_factory=lambda: DEFAULT_AGENTS)
    regimes: Sequence[str] = field(default_factory=lambda: REGIMES)


def featurize(
    stances: dict[str, float],
    regime: str | None,
    realized_vol: float,
    trend_short: float,
    trend_long: float,
    *,
    agents: Sequence[str] = DEFAULT_AGENTS,
    regimes: Sequence[str] = REGIMES,
) -> np.ndarray:
    """Build phi(s). Order: [bias, agent stances..., regime one-hot..., vol, ts, tl]."""
    feat = [1.0]
    for a in agents:
        feat.append(float(stances.get(a, 0.0)))
    for r in regimes:
        feat.append(1.0 if regime == r else 0.0)
    feat.append(float(realized_vol))
    feat.append(float(trend_short))
    feat.append(float(trend_long))
    return np.asarray(feat, dtype=float)


def feature_dim(
    agents: Sequence[str] = DEFAULT_AGENTS,
    regimes: Sequence[str] = REGIMES,
) -> int:
    return 1 + len(agents) + len(regimes) + 3


class TDLambdaStanceLearner:
    """Linear TD(lambda) over the stance feature vector.

    The learner is symmetric: positive ``stance_weights`` are produced by
    rectifying and normalising the absolute value of each per-agent
    coefficient. An agent whose stance reliably co-moves with realised PnL
    gets a higher weight, regardless of sign.
    """

    def __init__(self, cfg: TDLambdaConfig | None = None) -> None:
        self.cfg = cfg or TDLambdaConfig()
        self.dim = feature_dim(self.cfg.agents, self.cfg.regimes)
        self.w = np.zeros(self.dim, dtype=float)
        self.e = np.zeros(self.dim, dtype=float)
        self._last_phi: np.ndarray | None = None
        self.steps = 0

    # ------------------------------------------------------------------ value
    def value(self, phi: np.ndarray) -> float:
        return float(self.w @ phi)

    # ------------------------------------------------------------------ update
    def update(self, phi: np.ndarray, reward: float, phi_next: np.ndarray | None) -> float:
        v = self.value(phi)
        v_next = 0.0 if phi_next is None else self.value(phi_next)
        delta = reward + self.cfg.gamma * v_next - v
        self.e = self.cfg.gamma * self.cfg.lam * self.e + phi
        self.w = self.w + self.cfg.alpha * delta * self.e
        self.steps += 1
        return float(delta)

    def reset_trace(self) -> None:
        self.e = np.zeros_like(self.e)

    # ---------------------------------------------------- stance projection
    def stance_weights(self) -> dict[str, float]:
        """Project learned coefficients into the dict the risk agent consumes."""
        out: dict[str, float] = {}
        # Slots in phi are: [bias, agents..., regimes..., vol, ts, tl]
        for i, agent in enumerate(self.cfg.agents, start=1):
            out[agent] = max(self.cfg.weight_floor, abs(float(self.w[i])))
        # Renormalise to sum to 1 so downstream fuse is well-scaled.
        total = sum(out.values()) or 1.0
        for k in list(out):
            out[k] = out[k] / total
        return out

    # ------------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        return {
            "w": self.w.tolist(),
            "e": self.e.tolist(),
            "steps": self.steps,
            "dim": self.dim,
            "weights": self.stance_weights(),
        }

    def load(self, snap: dict) -> None:
        if not snap:
            return
        w = snap.get("w")
        e = snap.get("e")
        if w is not None and len(w) == self.dim:
            self.w = np.asarray(w, dtype=float)
        if e is not None and len(e) == self.dim:
            self.e = np.asarray(e, dtype=float)
        self.steps = int(snap.get("steps", 0))


def reward_from_outcome(
    pnl: float,
    *,
    turnover_usd: float = 0.0,
    inventory_usd: float = 0.0,
    turnover_cost_bps: float = 2.0,
    inventory_penalty_bps: float = 0.5,
) -> float:
    """Shape the realised PnL into a TD reward.

    Sutton-style scaling: subtract a small per-trade transaction-cost term
    (in basis points of turnover) and a tiny inventory carry penalty so the
    learner doesn't reward over-trading or stale positions.
    """
    r = float(pnl)
    r -= turnover_cost_bps * 1e-4 * abs(turnover_usd)
    r -= inventory_penalty_bps * 1e-4 * abs(inventory_usd)
    return r
