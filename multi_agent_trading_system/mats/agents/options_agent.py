"""Macro options selector agent.

Wraps ``learning.options.OptionsSelector``: at each tick, builds an
``OptionState`` per symbol, picks (or rolls) the active option, applies an
intra-option TD step against the most recent realised reward, and writes
``active_option`` and a ``planner_bias:{symbol}`` blend to SCS for the risk
agent to consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..learning.options import OptionState, OptionsSelector
from .base import Agent, AgentContext, AgentOutput


class OptionsSelectorAgent(Agent):
    name = "options"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.selector = OptionsSelector()
        self._last: dict[str, OptionState] = {}

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        regime_data = await ctx.read(f"regime:{symbol}", {}) or {}
        mc = await ctx.read(f"mc:{symbol}", {}) or {}

        regime = regime_data.get("regime", "chop")
        fused = await self._fused_stance(ctx, symbol)
        realized_vol = float(mc.get("stdev", 0.0)) / max(float(mc.get("s0", 1.0)), 1e-9)
        z_score = float(regime_data.get("z_score", 0.0))

        state = OptionState(regime=regime, fused_stance=fused,
                            realized_vol=realized_vol, z_score=z_score)

        # Apply an intra-option TD step against the latest reflection-based reward proxy.
        prev = self._last.get(symbol)
        if prev is not None:
            reflections = await ctx.read("reflections", []) or []
            reward = (float(reflections[-1]["hit_rate"]) - 0.5) * 0.01 if reflections else 0.0
            self.selector.step(prev, reward, state)
        bias = self.selector.primitive_bias(state)
        active = self.selector.active.name if self.selector.active else "stand_aside"

        await ctx.write("active_option", active)
        await ctx.write(f"planner_bias:{symbol}", float(bias))
        self._last[symbol] = state

        cube = self.memcube(
            title=f"option {symbol} active={active} bias={bias:+.2f}",
            body=f"regime={regime} z={z_score:+.2f} rv={realized_vol:.4f}",
            payload={"symbol": symbol, "active": active, "bias": bias,
                     "regime": regime, "q": dict(self.selector.q)},
            importance=0.4,
            tags=[symbol, "options", active],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=float(np.clip(bias, -1.0, 1.0)),
                           explain={"active": active, "bias": bias})

    async def _fused_stance(self, ctx: AgentContext, symbol: str) -> float:
        # Reuse stance reader pattern but compute a quick simple average.
        keys = [
            "technical_stances", "fundamental_stances", "quant_stances",
            "cta_trend_stances", "crossvenue_arb_stances", "orderflow_stances",
        ]
        items: list[float] = []
        for k in keys:
            arr = await ctx.read(k, []) or []
            for item in reversed(arr):
                if item.get("symbol") == symbol:
                    items.append(float(item.get("stance", 0.0)))
                    break
        mc = await ctx.read(f"mc:{symbol}", {}) or {}
        if "stance" in mc:
            items.append(float(mc["stance"]))
        regime = await ctx.read(f"regime:{symbol}", {}) or {}
        if "stance" in regime:
            items.append(float(regime["stance"]))
        if not items:
            return 0.0
        return float(np.clip(np.mean(items), -1.0, 1.0))
