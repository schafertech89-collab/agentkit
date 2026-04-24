"""Fundamental analysis — on-chain + off-chain factor scoring.

Crypto "fundamentals" are idiosyncratic. This engine composes a weighted
score from the factors that matter across Coinbase-listed assets, Base DeFi
protocols, and prediction-market assets:

    * TVL momentum          (7d change in locked value, DeFiLlama-style)
    * Active-address growth (on-chain users)
    * Dev activity          (commits/PRs — Electric Capital)
    * Supply issuance       (net new supply)
    * Treasury runway       (protocol treasury USD / monthly expenses)
    * Exchange flows        (net inflow to CEX = bearish)
    * Funding rate          (perp funding - positive = longs pay)
    * Stablecoin depeg risk (distance from 1.0)
    * Governance quality    (% circulating supply actively voting)

For prediction markets the factor set pivots to:
    * Liquidity depth
    * Time-decay to resolution
    * Implied vs fair price edge (model-derived)
    * Correlated reference markets
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FundamentalAnalysis:
    asset: str
    factors: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: _default_weights())

    def score(self) -> float:
        """Returns a composite score in [-1, 1]."""
        parts = []
        for key, weight in self.weights.items():
            if key not in self.factors:
                continue
            x = self.factors[key]
            parts.append(weight * _normalize(x, key))
        if not parts:
            return 0.0
        return float(np.clip(np.sum(parts), -1.0, 1.0))

    def explain(self) -> dict:
        breakdown = {}
        for key, weight in self.weights.items():
            if key in self.factors:
                breakdown[key] = {
                    "raw": self.factors[key],
                    "weight": weight,
                    "normalized": _normalize(self.factors[key], key),
                }
        return {"asset": self.asset, "score": self.score(), "breakdown": breakdown}


def _default_weights() -> dict[str, float]:
    return {
        "tvl_change_7d": 0.15,
        "active_addr_growth": 0.10,
        "dev_activity_z": 0.10,
        "supply_inflation_annual": -0.10,
        "treasury_runway_months": 0.05,
        "exchange_netflow": -0.10,
        "funding_rate_8h": -0.05,
        "stablecoin_depeg": -0.15,
        "governance_participation": 0.10,
        # prediction-market specific
        "liquidity_usd": 0.05,
        "implied_vs_fair_edge": 0.20,
        "time_to_resolution_days": 0.05,
    }


def _normalize(value: float, key: str) -> float:
    """Factor-specific squashing into [-1, 1]."""
    scalers = {
        "tvl_change_7d": lambda x: np.tanh(x * 5),
        "active_addr_growth": lambda x: np.tanh(x * 4),
        "dev_activity_z": lambda x: np.tanh(x / 2),
        "supply_inflation_annual": lambda x: np.tanh((x - 0.03) * 20),  # reference 3%
        "treasury_runway_months": lambda x: np.tanh((x - 12) / 24),
        "exchange_netflow": lambda x: np.tanh(x * 2),  # positive = inflow = bearish
        "funding_rate_8h": lambda x: np.tanh(x * 200),
        "stablecoin_depeg": lambda x: np.tanh(abs(x - 1.0) * 50),
        "governance_participation": lambda x: np.tanh((x - 0.05) * 20),
        "liquidity_usd": lambda x: np.tanh(np.log10(max(x, 1)) / 6),
        "implied_vs_fair_edge": lambda x: np.tanh(x * 5),
        "time_to_resolution_days": lambda x: np.tanh(-(x - 14) / 30),
    }
    fn = scalers.get(key, lambda x: float(np.tanh(x)))
    return float(fn(value))
