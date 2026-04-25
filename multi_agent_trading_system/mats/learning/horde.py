"""Horde of General Value Functions (GVFs).

A GVF is the generalisation of a value function to predicting *any*
cumulant: not just reward, but funding rate, dealer-gamma proxy, the cross
move of an intermarket asset, etc. (Sutton et al., "Horde", 2011.)

Each GVF has::

    cumulant : phi(s) -> float            (the thing being predicted)
    gamma    : phi(s) -> float in [0, 1]  (per-step continuation prob)
    pi       : str (descriptive policy tag — only used for bookkeeping here)

We keep linear value-function estimates per GVF (TD(0)) and expose a bulk
``predict_all`` that yields the current prediction for each GVF given the
shared feature vector. The orchestrator uses this output as a side-channel
of features for the main policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class GVF:
    name: str
    cumulant: Callable[[dict], float]
    gamma: Callable[[dict], float] = field(default=lambda _ctx: 0.95)
    policy_tag: str = "on_policy"


@dataclass
class GVFConfig:
    alpha: float = 0.05


class HordeLearner:
    """Linear TD(0) prediction for a Horde of GVFs sharing one feature vector.

    ``feature_dim`` is the dimensionality of phi(s); we keep a per-GVF
    weight vector and update it on each ``observe`` call.
    """

    def __init__(self, gvfs: Sequence[GVF], feature_dim: int,
                 cfg: GVFConfig | None = None) -> None:
        self.gvfs = list(gvfs)
        self.feature_dim = feature_dim
        self.cfg = cfg or GVFConfig()
        self.w: dict[str, np.ndarray] = {
            g.name: np.zeros(feature_dim, dtype=float) for g in self.gvfs
        }

    def predict(self, gvf_name: str, phi: np.ndarray) -> float:
        return float(self.w[gvf_name] @ phi)

    def predict_all(self, phi: np.ndarray) -> dict[str, float]:
        return {g.name: self.predict(g.name, phi) for g in self.gvfs}

    def observe(self, ctx: dict, phi: np.ndarray, phi_next: np.ndarray | None) -> dict[str, float]:
        """Apply one TD(0) update per GVF. Returns per-GVF TD errors."""
        errs: dict[str, float] = {}
        for g in self.gvfs:
            try:
                z = float(g.cumulant(ctx))
                gam = float(g.gamma(ctx))
            except Exception:
                z, gam = 0.0, 0.95
            v = self.predict(g.name, phi)
            v_next = 0.0 if phi_next is None else self.predict(g.name, phi_next)
            delta = z + gam * v_next - v
            self.w[g.name] = self.w[g.name] + self.cfg.alpha * delta * phi
            errs[g.name] = delta
        return errs


# ------------------------------------------------- standard GVFs (default) --

def _safe(d: dict, *keys: str, default: float = 0.0) -> float:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _gvf_intermarket_xcorr() -> GVF:
    """Predict next-step return of a paired symbol given its current move.

    Cumulant = paired_return. Useful as a feature for index agents.
    """
    return GVF(
        name="intermarket_xcorr",
        cumulant=lambda ctx: _safe(ctx, "paired_return"),
        gamma=lambda _: 0.5,
    )


def _gvf_funding_next_reset() -> GVF:
    """Predict perp funding at next reset window. Cumulant = funding rate."""
    return GVF(
        name="funding_next_reset",
        cumulant=lambda ctx: _safe(ctx, "funding_rate"),
        gamma=lambda _: 0.9,
    )


def _gvf_dealer_gamma_proxy() -> GVF:
    """A *proxy* for dealer gamma, since we have no options chain.

    Cumulant = signed realised vol regime score (positive when realised
    vol is rising fast and price is mean-reverting — a rough analogue of a
    short-gamma dealer environment).
    """

    def cum(ctx: dict) -> float:
        rv = _safe(ctx, "realized_vol")
        rv_prev = _safe(ctx, "realized_vol_prev")
        z = _safe(ctx, "z_score")
        # Higher rv with mean-reverting price -> short-gamma proxy.
        slope = rv - rv_prev
        return float(slope * (-z))

    return GVF(name="dealer_gamma_proxy", cumulant=cum, gamma=lambda _: 0.85)


def default_horde() -> list[GVF]:
    return [
        _gvf_intermarket_xcorr(),
        _gvf_funding_next_reset(),
        _gvf_dealer_gamma_proxy(),
    ]


__all__ = ["GVF", "GVFConfig", "HordeLearner", "default_horde"]
