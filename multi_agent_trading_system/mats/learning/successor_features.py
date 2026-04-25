"""Successor features (Barreto et al. 2017, building on Dayan 1993).

Decouples *what* you predict (psi) from *how much you care about it* (r_w).
Trains a feature-expectation estimator psi(s) such that

    V(s) = psi(s) . r_w

where r_w is a vector of weights over reward components (PnL, turnover
penalty, drawdown penalty, slippage). Reward re-weighting becomes a dot
product instead of a relearn.

The estimator uses TD(0) on each component independently:

    psi(s) <- psi(s) + alpha * (phi_r(s) + gamma * psi(s') - psi(s))

where phi_r(s) is the reward-component vector observed at s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


DEFAULT_COMPONENTS = ("pnl", "turnover_cost", "drawdown", "slippage")


@dataclass
class SFConfig:
    alpha: float = 0.05
    gamma: float = 0.95
    components: Sequence[str] = field(default_factory=lambda: DEFAULT_COMPONENTS)


class SuccessorFeatureEstimator:
    """Linear successor-feature estimator over a fixed feature vector.

    psi is a (n_components x feature_dim) matrix; psi(s) = psi @ phi(s) is the
    expected discounted feature occupancy. ``value(phi, r_w)`` is then
    ``r_w @ psi @ phi``.
    """

    def __init__(self, feature_dim: int, cfg: SFConfig | None = None) -> None:
        self.cfg = cfg or SFConfig()
        self.feature_dim = feature_dim
        self.n_components = len(self.cfg.components)
        self.psi = np.zeros((self.n_components, feature_dim), dtype=float)

    def expectation(self, phi: np.ndarray) -> np.ndarray:
        """Return psi(s) — discounted occupancy of each reward component."""
        return self.psi @ phi

    def value(self, phi: np.ndarray, r_weights: dict[str, float] | np.ndarray) -> float:
        if isinstance(r_weights, dict):
            rw = np.asarray(
                [float(r_weights.get(c, 0.0)) for c in self.cfg.components],
                dtype=float,
            )
        else:
            rw = np.asarray(r_weights, dtype=float)
        return float(rw @ self.psi @ phi)

    def update(
        self,
        phi: np.ndarray,
        components: dict[str, float],
        phi_next: np.ndarray | None,
    ) -> np.ndarray:
        """One TD(0) step on each reward component.

        ``components`` is a dict like ``{"pnl": 0.003, "drawdown": 0.0, ...}``;
        any missing component is treated as zero. Returns the per-component
        TD error vector for diagnostics.
        """
        comp_vec = np.asarray(
            [float(components.get(c, 0.0)) for c in self.cfg.components],
            dtype=float,
        )
        psi_s = self.psi @ phi
        psi_sp = np.zeros_like(psi_s) if phi_next is None else self.psi @ phi_next
        delta = comp_vec + self.cfg.gamma * psi_sp - psi_s
        self.psi = self.psi + self.cfg.alpha * np.outer(delta, phi)
        return delta

    def snapshot(self) -> dict:
        return {
            "psi": self.psi.tolist(),
            "components": list(self.cfg.components),
            "feature_dim": self.feature_dim,
        }

    def load(self, snap: dict) -> None:
        psi = snap.get("psi") if snap else None
        if psi is not None:
            arr = np.asarray(psi, dtype=float)
            if arr.shape == self.psi.shape:
                self.psi = arr
