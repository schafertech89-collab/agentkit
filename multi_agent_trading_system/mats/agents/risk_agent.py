"""Risk agent — fuses stances, computes risk/reward, sizes trades.

The risk agent is the planner's "gatekeeper". It reads stances from every
analytical agent in the SCS, fuses them with regime and reflection feedback,
computes an explicit RiskReward object, and then hands a ``SizedOrder`` per
venue to the execution agent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.monte_carlo import MonteCarloResult
from ..analysis.risk_reward import PositionSizer, RiskReward, SizedOrder
from .base import Agent, AgentContext, AgentOutput


STANCE_WEIGHTS = {
    "technical": 0.25,
    "fundamental": 0.15,
    "quant": 0.2,
    "monte_carlo": 0.2,
    "knn": 0.2,
}


class RiskAgent(Agent):
    name = "risk"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        account_equity = float(input_payload.get("account_equity", 10_000.0))
        max_leverage = float(input_payload.get("max_leverage", 3.0))

        stances = await self._collect_stances(ctx, symbol)
        fused_stance = self._fuse(stances)

        mc_snapshot = await ctx.read(f"mc:{symbol}", {}) or {}
        regime = await ctx.read(f"regime:{symbol}", {}) or {}
        reflections = await ctx.read("reflections", []) or []
        hit_rate = (
            float(reflections[-1]["hit_rate"]) if reflections else 0.5
        )

        # Winners' reward / losers' loss estimated from MC's p05/p95.
        reward_if_right = abs((mc_snapshot.get("p95", 1.05) / max(mc_snapshot.get("s0", 1.0), 1e-9)) - 1.0)
        loss_if_wrong = abs((mc_snapshot.get("p05", 0.95) / max(mc_snapshot.get("s0", 1.0), 1e-9)) - 1.0)
        sigma = float(mc_snapshot.get("stdev", 0.02) / max(mc_snapshot.get("s0", 1.0), 1.0))
        win_prob = 0.5 * (1 + fused_stance) * (0.6 + 0.4 * hit_rate)
        win_prob = float(np.clip(win_prob, 0.02, 0.98))
        expected_return = win_prob * reward_if_right - (1 - win_prob) * loss_if_wrong

        rr = RiskReward(
            expected_return=expected_return, win_prob=win_prob,
            loss_if_wrong=loss_if_wrong, reward_if_right=reward_if_right,
            sigma=sigma,
        )
        sizer = PositionSizer(
            account_equity=account_equity,
            max_leverage=max_leverage,
            max_risk_per_trade=0.01,
            cvar_budget=0.05 if regime.get("regime") != "crisis" else 0.01,
        )

        connector = ctx.get_connector(venue)
        price = 0.0
        try:
            t = await connector.ticker(symbol)
            price = t.price
        except Exception:
            price = mc_snapshot.get("s0") or 1.0

        mc_like = MonteCarloResult(
            samples=np.asarray([mc_snapshot.get("s0", price)]),
            mean=mc_snapshot.get("mean", price), stdev=mc_snapshot.get("stdev", sigma * price),
            median=mc_snapshot.get("median", price),
            p05=mc_snapshot.get("p05", price * 0.95),
            p95=mc_snapshot.get("p95", price * 1.05),
            var_95=mc_snapshot.get("var_95", price * 0.05),
            expected_shortfall_95=mc_snapshot.get("es_95", price * 0.07),
        )
        order = sizer.size(
            venue=venue, symbol=symbol, price=price,
            signal=fused_stance, mc_result=mc_like, rr=rr,
        )

        # Record decision in SCS — picked up by the reflection agent later.
        decision_id = f"dec_{int(ctx.clock()*1000)}_{symbol.replace('-', '').replace('/', '')}"
        await ctx.append("decisions", {
            "id": decision_id, "symbol": symbol, "venue": venue,
            "stance": fused_stance, "order": order.__dict__,
            "timestamp": ctx.clock(),
        })

        cube = self.memcube(
            title=f"risk {symbol} {order.side} ${order.size_usd:.0f} (conf {order.confidence:.2f})",
            body=order.reason,
            payload={
                "decision_id": decision_id,
                "order": order.__dict__,
                "stances": stances,
                "fused_stance": fused_stance,
                "hit_rate": hit_rate,
                "rr": {
                    "win_prob": win_prob,
                    "reward_if_right": reward_if_right,
                    "loss_if_wrong": loss_if_wrong,
                    "sigma": sigma,
                    "expected_return": expected_return,
                },
            },
            importance=0.5 + 0.4 * abs(fused_stance),
            tags=[symbol, venue, order.side],
        )
        return AgentOutput(
            agent=self.name, memcube=cube, stance=fused_stance,
            actions=[{"order": order.__dict__, "decision_id": decision_id}],
            explain={"stances": stances, "rr": rr.__dict__},
        )

    # --------------------------------------------------------- helpers

    async def _collect_stances(self, ctx: AgentContext, symbol: str) -> dict[str, float]:
        sources = [
            ("technical", "technical_stances"),
            ("fundamental", "fundamental_stances"),
            ("quant", "quant_stances"),
        ]
        out: dict[str, float] = {}
        for key, scs_key in sources:
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

    @staticmethod
    def _fuse(stances: dict[str, float]) -> float:
        if not stances:
            return 0.0
        total_w = 0.0
        weighted = 0.0
        for k, s in stances.items():
            w = STANCE_WEIGHTS.get(k, 0.1)
            weighted += w * s
            total_w += w
        return float(np.clip(weighted / max(total_w, 1e-9), -1.0, 1.0))
