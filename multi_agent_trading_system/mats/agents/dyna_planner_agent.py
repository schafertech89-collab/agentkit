"""Dyna-Q+ planner agent.

Per-tick: builds a coarse ``state = (regime, stance_bucket)`` from SCS,
applies a real Q-update against the previous step's realised PnL, then runs
``plan_steps`` background backups using the learned MarketDynamicsModel.

Outputs a per-symbol ``planner_bias:{symbol}`` (in [-1, 1]) consumed by the
risk agent. Also writes ``planner_q:{symbol}`` for diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.market_dynamics_model import stance_bucket
from ..learning.dyna_q_plus import DynaQPlus
from ..learning.value_head import LinearValueHead
from ..learning.td_lambda import feature_dim, featurize
from .base import Agent, AgentContext, AgentOutput


class DynaPlannerAgent(Agent):
    name = "dyna_planner"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.planner = DynaQPlus()
        self.value_head = LinearValueHead(feature_dim=feature_dim())
        self._last: dict[str, dict[str, Any]] = {}
        self._rng = np.random.default_rng(0)

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        regime_data = await ctx.read(f"regime:{symbol}", {}) or {}
        mc = await ctx.read(f"mc:{symbol}", {}) or {}

        regime = regime_data.get("regime", "chop")
        fused = await self._fused_stance(ctx, symbol)
        bucket = stance_bucket(fused)
        state = (regime, bucket)

        realized_vol = float(mc.get("stdev", 0.0)) / max(float(mc.get("s0", 1.0)), 1e-9)
        trend_short = float(regime_data.get("trend_short", 0.0))
        trend_long = float(regime_data.get("trend_long", 0.0))
        phi = featurize({"_": fused}, regime, realized_vol, trend_short, trend_long)

        # Real-step update from last tick's outcome.
        prev = self._last.get(symbol)
        if prev is not None:
            outcomes = await ctx.read("outcomes", {}) or {}
            outcome = outcomes.get(prev["decision_id"]) if prev.get("decision_id") else None
            reward = float(outcome.get("pnl", 0.0)) if outcome else 0.0
            self.planner.step(prev["state"], prev["action"], reward, state, rng=self._rng)
            # Bootstrap value head off the running TD targets.
            self.value_head.update(prev["phi"], reward + 0.95 * self.value_head.predict(phi))

        action = self.planner.choose(state, rng=self._rng)
        bias = float(action) * 0.6  # scale planner action into stance space
        await ctx.write(f"planner_bias:{symbol}", float(np.clip(bias, -1.0, 1.0)))
        await ctx.write(f"planner_q:{symbol}", {
            "state": list(state),
            "q": {str(a): self.planner.q[(state, a)] for a in self.planner.cfg.actions},
            "best_action": int(self.planner.best_action(state)),
            "model_pairs": len(self.planner.model.known_pairs()),
        })

        # Record for next step.
        decisions = await ctx.read("decisions", []) or []
        last_dec_id = None
        for d in reversed(decisions):
            if d.get("symbol") == symbol:
                last_dec_id = d.get("id")
                break
        self._last[symbol] = {"state": state, "action": int(action),
                              "phi": phi, "decision_id": last_dec_id}

        cube = self.memcube(
            title=f"dyna {symbol} a={action} bias={bias:+.2f}",
            body=f"state={state} model_pairs={len(self.planner.model.known_pairs())}",
            payload={"symbol": symbol, "state": list(state), "action": int(action),
                     "bias": bias, "steps": self.planner.steps,
                     "value_head_steps": self.value_head.steps},
            importance=0.4,
            tags=[symbol, "dyna"],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=float(bias),
                           explain={"state": list(state), "action": int(action), "bias": bias})

    async def _fused_stance(self, ctx: AgentContext, symbol: str) -> float:
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
