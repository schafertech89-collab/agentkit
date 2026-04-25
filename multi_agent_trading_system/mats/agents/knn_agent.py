"""KNN regime classifier agent.

Publishes the current market regime as a categorical label in the SCS. The
orchestrator uses the regime to pick strategy templates (e.g. in ``crisis``
the risk agent deleverages; in ``trend_up`` the position sizer increases
its Kelly cap).

The agent maintains an online-updated classifier across restarts by
persisting labelled windows into the vault's semantic tier.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.knn import KNNRegime, engineer_features
from .base import Agent, AgentContext, AgentOutput


REGIME_STANCE = {
    "trend_up": 0.6,
    "trend_down": -0.6,
    "chop": 0.0,
    "crisis": -0.9,
    "squeeze": 0.15,
}


class KNNRegimeAgent(Agent):
    name = "knn"

    def __init__(self, model: KNNRegime | None = None) -> None:
        super().__init__()
        self.model = model or KNNRegime()

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        connector = ctx.get_connector(venue)
        candles = await connector.candles(symbol, granularity="ONE_HOUR", limit=240)
        if len(candles) < 20:
            cube = self.memcube(title=f"KNN {symbol} insufficient data", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        prices = [c.close for c in candles]
        volumes = [c.volume for c in candles]

        # Hydrate any previously labelled windows from SCS
        labelled = await ctx.read("knn_training", [])
        if labelled:
            self.model.fit([(w["prices"], w.get("volumes"), w["label"]) for w in labelled])

        result = self.model.predict(prices, volumes)
        regime = result["regime"]
        stance = REGIME_STANCE.get(regime, 0.0) * result["confidence"]

        cube = self.memcube(
            title=f"KNN {symbol} regime={regime} conf={result['confidence']:.2f}",
            body=f"features={result['features']}",
            payload={"symbol": symbol, "regime": regime, "confidence": result["confidence"],
                     "features": result["features"]},
            importance=0.3 + 0.4 * result["confidence"],
            tags=[symbol, "regime", regime],
        )
        await ctx.write(f"regime:{symbol}", {"regime": regime,
                                              "confidence": result["confidence"],
                                              "stance": stance})
        return AgentOutput(agent=self.name, memcube=cube, stance=stance,
                           explain={"regime": regime, "confidence": result["confidence"]})
