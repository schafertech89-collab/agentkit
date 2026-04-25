"""Risk / reward sizing.

Core primitives used by the risk agent:

    * Kelly criterion (full and fractional, capped)
    * Volatility-target sizing
    * CVaR / expected-shortfall budget
    * Cross-venue correlation-aware netting

The ``PositionSizer`` is what the execution agent actually calls. It takes the
fused signal from the planner plus the latest risk state in the SCS and emits
a ``SizedOrder`` per venue with capped leverage, stop-loss, take-profit, and a
safety floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .monte_carlo import MonteCarloResult


Side = Literal["long", "short", "flat"]


@dataclass
class SizedOrder:
    venue: str
    symbol: str
    side: Side
    size_usd: float
    leverage: float = 1.0
    stop_loss: float | None = None
    take_profit: float | None = None
    confidence: float = 0.0
    reason: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RiskReward:
    expected_return: float  # fractional expected return over horizon
    win_prob: float  # probability the trade wins
    loss_if_wrong: float  # fractional loss if wrong (positive number)
    reward_if_right: float  # fractional reward if right
    sigma: float  # per-horizon volatility of the trade

    def kelly(self, cap: float = 0.25, fraction: float = 0.5) -> float:
        """Fractional Kelly, capped, never short-negative."""
        b = self.reward_if_right
        q = 1 - self.win_prob
        if b <= 0 or self.loss_if_wrong <= 0:
            return 0.0
        # Kelly fraction for binary outcome:
        f = self.win_prob - q / b
        return float(np.clip(f * fraction, 0.0, cap))

    def sharpe_like(self) -> float:
        if self.sigma <= 0:
            return 0.0
        return float(self.expected_return / self.sigma)


class PositionSizer:
    def __init__(
        self,
        *,
        account_equity: float,
        max_leverage: float = 3.0,
        max_risk_per_trade: float = 0.01,
        cvar_budget: float = 0.05,
        min_confidence: float = 0.15,
    ) -> None:
        self.account_equity = account_equity
        self.max_leverage = max_leverage
        self.max_risk_per_trade = max_risk_per_trade
        self.cvar_budget = cvar_budget
        self.min_confidence = min_confidence

    # -------------------------------------------------------------- API

    def size(
        self,
        *,
        venue: str,
        symbol: str,
        price: float,
        signal: float,
        mc_result: MonteCarloResult,
        rr: RiskReward,
    ) -> SizedOrder:
        """Combine signal, MC distribution, and explicit RR into a SizedOrder.

        ``signal`` is the scalar in [-1, 1] from the planner's stance.
        """
        if abs(signal) < self.min_confidence:
            return SizedOrder(
                venue=venue, symbol=symbol, side="flat", size_usd=0.0, confidence=0.0,
                reason=f"signal {signal:.3f} below min_confidence {self.min_confidence}",
            )

        side: Side = "long" if signal > 0 else "short"
        kelly_f = rr.kelly()
        # scale down kelly by signal magnitude
        f = kelly_f * abs(signal)

        # CVaR budget cap: ES-based position cap
        es_frac = abs(mc_result.expected_shortfall_95 / max(1e-9, price))
        es_cap = 0.0 if es_frac == 0 else (self.cvar_budget / es_frac)
        f = min(f, es_cap)

        # Per-trade risk cap (stop-loss × size ≤ max_risk_per_trade * equity)
        stop_distance = max(mc_result.stdev, 1e-9) * 1.0  # 1σ stop
        per_trade_cap = self.max_risk_per_trade / (stop_distance / max(price, 1e-9))
        f = min(f, per_trade_cap)
        f = float(np.clip(f, 0.0, self.max_leverage))

        size_usd = self.account_equity * f
        leverage = min(self.max_leverage, max(1.0, f))

        stop_loss = price - stop_distance if side == "long" else price + stop_distance
        take_profit = price + (rr.reward_if_right * price) if side == "long" else price - (rr.reward_if_right * price)

        return SizedOrder(
            venue=venue,
            symbol=symbol,
            side=side,
            size_usd=size_usd,
            leverage=leverage,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confidence=float(abs(signal)),
            reason=f"kelly={kelly_f:.3f} es_cap={es_cap:.3f} per_trade_cap={per_trade_cap:.3f}",
            metadata={
                "mean_outcome": mc_result.mean,
                "var_95": mc_result.var_95,
                "es_95": mc_result.expected_shortfall_95,
                "sharpe_like": rr.sharpe_like(),
            },
        )
