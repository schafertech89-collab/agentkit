"""Value head V_theta(s) for bootstrapping Monte Carlo rollouts.

Used in two places:

  * ``MonteCarloAgent`` truncates a path at depth ``d`` and bootstraps with
    ``V_theta(s_d)``, drastically reducing variance.
  * ``DynaQPlus`` uses it as the leaf value during planning rollouts.

Two backends:

  * ``LinearValueHead``  — closed-form / SGD on a linear value function over
    phi(s). Always available (numpy-only).
  * ``TorchValueHead``   — small MLP, available when torch is installed.

Both expose the same ``predict / update`` interface; the orchestrator picks
whichever the environment supports, mirroring the pattern in
``ppo_policy.py``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    _HAS_TORCH = False


class LinearValueHead:
    def __init__(self, feature_dim: int, alpha: float = 0.05) -> None:
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.w = np.zeros(feature_dim, dtype=float)
        self.steps = 0

    def predict(self, phi: np.ndarray) -> float:
        return float(self.w @ phi)

    def update(self, phi: np.ndarray, target: float) -> float:
        pred = self.predict(phi)
        err = target - pred
        self.w = self.w + self.alpha * err * phi
        self.steps += 1
        return float(err)

    def update_batch(
        self,
        phis: Iterable[np.ndarray],
        targets: Iterable[float],
    ) -> float:
        errs = []
        for p, t in zip(phis, targets):
            errs.append(self.update(p, t))
        return float(np.mean(errs)) if errs else 0.0

    def snapshot(self) -> dict:
        return {"w": self.w.tolist(), "feature_dim": self.feature_dim, "steps": self.steps}

    def load(self, snap: dict) -> None:
        if not snap:
            return
        w = snap.get("w")
        if w is not None and len(w) == self.feature_dim:
            self.w = np.asarray(w, dtype=float)
        self.steps = int(snap.get("steps", 0))


if _HAS_TORCH:

    class _MLP(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)


class TorchValueHead:
    def __init__(self, feature_dim: int, hidden: int = 64, lr: float = 3e-4) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch is not available; use LinearValueHead instead")
        self.feature_dim = feature_dim
        self.net = _MLP(feature_dim, hidden=hidden)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()

    def predict(self, phi: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.as_tensor(phi, dtype=torch.float32).unsqueeze(0)
            return float(self.net(x).item())

    def update(self, phi: np.ndarray, target: float) -> float:
        x = torch.as_tensor(phi, dtype=torch.float32).unsqueeze(0)
        y = torch.as_tensor([target], dtype=torch.float32)
        pred = self.net(x)
        loss = self.loss(pred, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())

    def update_batch(
        self,
        phis: Sequence[np.ndarray],
        targets: Sequence[float],
    ) -> float:
        if not list(phis):
            return 0.0
        x = torch.as_tensor(np.asarray(phis), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(targets), dtype=torch.float32)
        pred = self.net(x)
        loss = self.loss(pred, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return float(loss.item())


def make_value_head(feature_dim: int, *, prefer_torch: bool = True):
    """Return a value head, preferring torch when available."""
    if prefer_torch and _HAS_TORCH:
        return TorchValueHead(feature_dim)
    return LinearValueHead(feature_dim)
