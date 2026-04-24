"""Allora publisher agent.

After the risk agent emits a decision, the Allora publisher extracts the
forward-looking point estimate (median terminal price from the MC simulation,
or the fused stance scaled to the topic's expected output) and posts it to
Allora's inference submission endpoint.
"""

from __future__ import annotations

from typing import Any

from ..connectors.allora import AlloraConnector, Prediction
from .base import Agent, AgentContext, AgentOutput


class AlloraPublisherAgent(Agent):
    name = "allora_publisher"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        allora: AlloraConnector = ctx.get_connector("allora")
        if allora is None:
            cube = self.memcube(title="allora (no connector)", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        predictions_to_make = input_payload.get("predictions") or []
        responses: list[dict] = []
        for pred in predictions_to_make:
            symbol = pred["symbol"]
            topic_id = int(pred["topic_id"])
            mc = await ctx.read(f"mc:{symbol}", {}) or {}
            regime = await ctx.read(f"regime:{symbol}", {}) or {}
            value = float(pred.get("value", mc.get("mean", 0.0)))
            confidence = float(pred.get("confidence", regime.get("confidence", 0.5)))
            resp = await allora.submit_prediction(Prediction(
                topic_id=topic_id, value=value, confidence=confidence,
                extra={
                    "symbol": symbol,
                    "regime": regime.get("regime"),
                    "prob_up": mc.get("prob_up"),
                    "p95": mc.get("p95"),
                    "p05": mc.get("p05"),
                },
            ))
            responses.append({"topic_id": topic_id, "symbol": symbol, "value": value,
                              "confidence": confidence, "response": resp})
            await ctx.append("allora_submissions", {
                "topic_id": topic_id, "symbol": symbol, "value": value,
                "response": resp,
            })

        cube = self.memcube(
            title=f"allora submissions {len(responses)}",
            body=f"topics={[r['topic_id'] for r in responses]}",
            payload={"submissions": responses},
            importance=0.3,
            tags=["allora"],
        )
        return AgentOutput(
            agent=self.name, memcube=cube, stance=0.0,
            explain={"submissions": responses},
        )
