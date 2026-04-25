"""Horde agent — runs a Horde of GVF predictors and writes side-channel features.

The agent reads recent SCS state for the symbol, builds a context dict that
the GVFs consume, computes the shared feature vector, calls
``HordeLearner.observe`` to apply a TD(0) update, and writes the current
predictions to ``gvf:{symbol}`` for downstream consumers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..learning.horde import GVF, HordeLearner, default_horde
from ..learning.td_lambda import featurize, feature_dim
from .base import Agent, AgentContext, AgentOutput


class HordeAgent(Agent):
    name = "horde"

    def __init__(self, *, gvfs: list[GVF] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.learner = HordeLearner(gvfs or default_horde(), feature_dim=feature_dim())
        self._last_phi: dict[str, np.ndarray] = {}
        self._last_realized_vol: dict[str, float] = {}

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        paired = input_payload.get("paired_symbol")

        regime_data = await ctx.read(f"regime:{symbol}", {}) or {}
        mc = await ctx.read(f"mc:{symbol}", {}) or {}
        regime = regime_data.get("regime")

        stances: dict[str, float] = {}
        for k, scs_key in [
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
                    stances[k] = float(item.get("stance", 0.0))
                    break
        if "stance" in mc:
            stances["monte_carlo"] = float(mc["stance"])
        if "stance" in regime_data:
            stances["knn"] = float(regime_data["stance"])

        realized_vol = float(mc.get("stdev", 0.0)) / max(float(mc.get("s0", 1.0)), 1e-9)
        trend_short = float(regime_data.get("trend_short", 0.0))
        trend_long = float(regime_data.get("trend_long", 0.0))
        phi = featurize(stances, regime, realized_vol, trend_short, trend_long)

        # Build the GVF context: paired return, funding, vols.
        paired_return = 0.0
        if paired:
            p_regime = await ctx.read(f"regime:{paired}", {}) or {}
            paired_return = float(p_regime.get("trend_short", 0.0))
        funding = float(input_payload.get("funding_rate", 0.0))
        gvf_ctx = {
            "paired_return": paired_return,
            "funding_rate": funding,
            "realized_vol": realized_vol,
            "realized_vol_prev": self._last_realized_vol.get(symbol, realized_vol),
            "z_score": float(regime_data.get("z_score", 0.0)),
        }

        last_phi = self._last_phi.get(symbol)
        td_errs = self.learner.observe(gvf_ctx, last_phi if last_phi is not None else phi, phi)
        preds = self.learner.predict_all(phi)

        await ctx.write(f"gvf:{symbol}", {
            "predictions": preds,
            "td_errors": td_errs,
            "context": gvf_ctx,
        })
        self._last_phi[symbol] = phi
        self._last_realized_vol[symbol] = realized_vol

        cube = self.memcube(
            title=f"horde {symbol} preds",
            body=", ".join(f"{k}={v:+.4f}" for k, v in preds.items()),
            payload={"symbol": symbol, "venue": venue, "predictions": preds,
                     "td_errors": td_errs},
            importance=0.3,
            tags=[symbol, "horde", "gvf"],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=0.0,
                           explain={"predictions": preds})
