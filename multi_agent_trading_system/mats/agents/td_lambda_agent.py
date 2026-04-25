"""TD(lambda) learner agent.

Per-symbol wrapper around ``TDLambdaStanceLearner``. On each tick:

  1. Build phi(s) from the current SCS state.
  2. If a previous (s, r) pair was recorded for this symbol, apply a TD(lambda)
     update against the realised PnL stored in ``outcomes`` (or, when no
     outcome is yet known, a shaped reward built from the most recent
     reflection's hit_rate / utility).
  3. Stash phi(s) for the next tick and publish learned ``stance_weights``
     to SCS for the risk agent.

Successor features are updated on the same loop so re-weighting the reward
components later (turnover penalty, drawdown) is just a dot-product.
"""

from __future__ import annotations

from typing import Any


from ..learning.successor_features import SuccessorFeatureEstimator
from ..learning.td_lambda import (
    TDLambdaStanceLearner,
    feature_dim,
    featurize,
    reward_from_outcome,
)
from .base import Agent, AgentContext, AgentOutput


class TDLambdaAgent(Agent):
    name = "td_lambda"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.learner = TDLambdaStanceLearner()
        self.sf = SuccessorFeatureEstimator(feature_dim=feature_dim())
        self._last: dict[str, dict[str, Any]] = {}

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        regime_data = await ctx.read(f"regime:{symbol}", {}) or {}
        mc = await ctx.read(f"mc:{symbol}", {}) or {}
        regime = regime_data.get("regime")

        stances = await self._collect_stances(ctx, symbol)
        realized_vol = float(mc.get("stdev", 0.0)) / max(float(mc.get("s0", 1.0)), 1e-9)
        trend_short = float(regime_data.get("trend_short", 0.0))
        trend_long = float(regime_data.get("trend_long", 0.0))
        phi = featurize(stances, regime, realized_vol, trend_short, trend_long)

        # Update against the previous step's reward (if any).
        delta = 0.0
        components: dict[str, float] = {}
        if symbol in self._last:
            prev = self._last[symbol]
            outcomes = await ctx.read("outcomes", {}) or {}
            outcome = outcomes.get(prev["decision_id"]) if prev.get("decision_id") else None
            if outcome:
                pnl = float(outcome.get("pnl", 0.0))
                turnover = float(outcome.get("turnover_usd", abs(pnl) * 5.0))
                inventory = float(outcome.get("inventory_usd", 0.0))
                drawdown = float(outcome.get("drawdown", max(0.0, -pnl)))
                slippage = float(outcome.get("slippage_usd", 0.0))
                reward = reward_from_outcome(
                    pnl, turnover_usd=turnover, inventory_usd=inventory,
                )
                components = {
                    "pnl": pnl,
                    "turnover_cost": -turnover * 1e-4,
                    "drawdown": -drawdown,
                    "slippage": -slippage * 1e-4,
                }
            else:
                # Shaped pseudo-reward from reflection hit_rate when outcomes
                # haven't been settled yet — keeps the learner moving.
                reflections = await ctx.read("reflections", []) or []
                hit = float(reflections[-1]["hit_rate"]) if reflections else 0.5
                reward = (hit - 0.5) * 0.01
                components = {"pnl": reward}
            delta = self.learner.update(prev["phi"], reward, phi)
            self.sf.update(prev["phi"], components, phi)

        weights = self.learner.stance_weights()
        await ctx.write("stance_weights", weights)
        await ctx.write(f"sf:{symbol}", {
            "psi": self.sf.expectation(phi).tolist(),
            "components": list(self.sf.cfg.components),
        })

        # Capture last (phi, decision_id) for next step's reward attribution.
        decisions = await ctx.read("decisions", []) or []
        last_decision_id = None
        for d in reversed(decisions):
            if d.get("symbol") == symbol:
                last_decision_id = d.get("id")
                break
        self._last[symbol] = {"phi": phi, "decision_id": last_decision_id}

        cube = self.memcube(
            title=f"TD(λ) {symbol} delta={delta:+.4f}",
            body=", ".join(f"{k}={v:.3f}" for k, v in weights.items()),
            payload={"symbol": symbol, "weights": weights, "delta": delta,
                     "components": components, "steps": self.learner.steps},
            importance=0.4,
            tags=[symbol, "td_lambda"],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=0.0,
                           explain={"weights": weights, "delta": delta})

    async def _collect_stances(self, ctx: AgentContext, symbol: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, scs_key in [
            ("technical", "technical_stances"),
            ("fundamental", "fundamental_stances"),
            ("quant", "quant_stances"),
            ("cta_trend", "cta_trend_stances"),
            ("crossvenue_arb", "crossvenue_arb_stances"),
            ("orderflow", "orderflow_stances"),
        ]:
            arr = await ctx.read(scs_key, []) or []
            for item in reversed(arr):
                if item.get("symbol") == symbol:
                    out[key] = float(item.get("stance", 0.0))
                    break
        mc = await ctx.read(f"mc:{symbol}", {}) or {}
        if "stance" in mc:
            out["monte_carlo"] = float(mc["stance"])
        regime = await ctx.read(f"regime:{symbol}", {}) or {}
        if "stance" in regime:
            out["knn"] = float(regime["stance"])
        return out
