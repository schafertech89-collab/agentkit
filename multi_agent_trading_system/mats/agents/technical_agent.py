"""Technical analysis agent.

Pulls the most recent candles for every monitored symbol and emits a stance
(-1 to +1) into the SCS. Uses the ``TechnicalAnalysis`` facade.
"""

from __future__ import annotations

from typing import Any

from ..analysis.technical import TechnicalAnalysis
from .base import Agent, AgentContext, AgentOutput


class TechnicalAgent(Agent):
    name = "technical"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        connector = ctx.get_connector(venue)
        candles = await connector.candles(symbol, granularity="ONE_HOUR", limit=200)
        if not candles:
            cube = self.memcube(title=f"{symbol} technical (no data)", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        prices = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [c.volume for c in candles]
        ta = TechnicalAnalysis(prices=prices, highs=highs, lows=lows, volumes=volumes)
        stance = ta.score()
        indicators = ta.indicators()

        cube = self.memcube(
            title=f"TA {symbol} stance={stance:+.2f}",
            body=(
                f"RSI14={indicators['rsi_14']:.2f} "
                f"MACD_hist={indicators['macd']['hist']:+.4f} "
                f"BB_z={indicators['bollinger']['z']:+.2f} "
                f"SMAcross={indicators['sma_cross_20_50']:+.1f}"
            ),
            payload={"indicators": indicators, "stance": stance, "symbol": symbol, "venue": venue},
            importance=0.4 + 0.3 * abs(stance),
            tags=[symbol, venue],
        )
        await ctx.append("technical_stances", {
            "agent": self.name, "symbol": symbol, "venue": venue, "stance": stance,
            "indicators": indicators,
        })
        return AgentOutput(agent=self.name, memcube=cube, stance=stance,
                           explain={"indicators": indicators})
