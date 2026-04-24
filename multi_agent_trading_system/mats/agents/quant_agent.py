"""Quantitative agent — multi-asset factor/correlation view."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.quantitative import QuantitativeEngine, sharpe
from .base import Agent, AgentContext, AgentOutput


class QuantitativeAgent(Agent):
    name = "quant"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        universe = input_payload.get("universe") or [symbol]
        connector = ctx.get_connector(venue)

        closes: list[list[float]] = []
        for s in universe:
            try:
                candles = await connector.candles(s, granularity="ONE_HOUR", limit=240)
            except Exception:
                candles = []
            if len(candles) < 10:
                continue
            closes.append([c.close for c in candles])

        if not closes:
            cube = self.memcube(title=f"quant {symbol} (no data)", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        matrix = np.array(closes).T
        engine = QuantitativeEngine()
        summary = engine.summary(matrix[:, 0])
        weights = engine.optimal_weights(matrix).tolist() if matrix.shape[1] > 1 else [1.0]

        # Sharpe-anchored stance: if the asset has positive sharpe and positive
        # drift, stance=+sharpe/3 capped; else negative.
        stance = float(np.clip(summary["sharpe"] / 3.0, -1.0, 1.0))

        cube = self.memcube(
            title=f"Quant {symbol} sharpe={summary['sharpe']:+.2f}",
            body=(
                f"daily_ret={summary['mean_daily_return']:+.4%} "
                f"vol={summary['realized_vol']:.4f} "
                f"sortino={summary['sortino']:+.2f} "
                f"calmar={summary['calmar']:+.2f} "
                f"MDD={summary['max_drawdown']:+.2%}"
            ),
            payload={
                "symbol": symbol, "universe": universe, "summary": summary,
                "optimal_weights": weights,
            },
            importance=0.35 + 0.3 * abs(stance),
            tags=[symbol, "quant"],
        )
        await ctx.append("quant_stances", {
            "symbol": symbol, "stance": stance, "summary": summary,
            "optimal_weights": weights,
        })
        return AgentOutput(
            agent=self.name, memcube=cube, stance=stance,
            explain={"summary": summary, "weights": weights},
        )
