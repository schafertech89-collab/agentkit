"""Fundamental analysis agent — composes on-chain + off-chain factors."""

from __future__ import annotations

from typing import Any

from ..analysis.fundamental import FundamentalAnalysis
from .base import Agent, AgentContext, AgentOutput


class FundamentalAgent(Agent):
    name = "fundamental"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        factors = input_payload.get("factors") or await ctx.read(f"fundamentals:{symbol}", {}) or {}
        fa = FundamentalAnalysis(asset=symbol, factors=factors)
        stance = fa.score()
        explain = fa.explain()

        cube = self.memcube(
            title=f"FA {symbol} stance={stance:+.2f}",
            body=f"composite={stance:+.3f} n_factors={len(factors)}",
            payload={"symbol": symbol, "stance": stance, "explain": explain},
            importance=0.3 + 0.4 * abs(stance),
            tags=[symbol, "fundamentals"],
        )
        await ctx.append("fundamental_stances", {
            "agent": self.name, "symbol": symbol, "stance": stance, "explain": explain,
        })
        return AgentOutput(agent=self.name, memcube=cube, stance=stance, explain=explain)
