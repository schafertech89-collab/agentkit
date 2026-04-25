"""CTA time-series momentum agent.

Vol-targeted multi-lookback momentum, in the spirit of the systematic CTAs
covered on *Top Traders Unplugged* and *Jordi Visser*. The agent computes
return z-scores across three lookbacks (24h, 72h, 240h on the 1-hour bar
cadence the rest of MATS already uses) and emits a stance::

    stance = tanh( w_short * z_24 + w_mid * z_72 + w_long * z_240 )

with optional vol-targeted scaling so the stance shrinks during vol blow-ups.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Agent, AgentContext, AgentOutput


LOOKBACKS = (24, 72, 240)
LOOKBACK_WEIGHTS = (0.5, 0.3, 0.2)
ANNUAL_VOL_TARGET = 0.20


class CTATrendAgent(Agent):
    name = "cta_trend"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        connector = ctx.get_connector(venue)
        if connector is None:
            cube = self.memcube(title=f"CTA {symbol} no connector", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        candles = await connector.candles(symbol, granularity="ONE_HOUR", limit=300)
        if len(candles) < max(LOOKBACKS) + 1:
            cube = self.memcube(title=f"CTA {symbol} insufficient data", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        prices = np.asarray([c.close for c in candles], dtype=float)
        log_rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
        zs: list[float] = []
        for lb in LOOKBACKS:
            window = log_rets[-lb:]
            mu = float(window.mean())
            sd = float(window.std()) or 1e-9
            zs.append(mu / sd * np.sqrt(lb))
        raw = float(sum(w * z for w, z in zip(LOOKBACK_WEIGHTS, zs)))

        # Vol-target scaling: shrink during vol spikes.
        hourly_vol = float(log_rets[-LOOKBACKS[0]:].std())
        annual_vol = hourly_vol * np.sqrt(24 * 365) if hourly_vol > 0 else 1e-9
        vol_scaler = float(np.clip(ANNUAL_VOL_TARGET / annual_vol, 0.1, 2.5))
        stance = float(np.clip(np.tanh(raw) * vol_scaler / 2.5, -1.0, 1.0))

        await ctx.append("cta_trend_stances", {
            "agent": self.name, "symbol": symbol, "venue": venue, "stance": stance,
            "z": zs, "vol_scaler": vol_scaler, "annual_vol": annual_vol,
        })

        cube = self.memcube(
            title=f"CTA {symbol} stance={stance:+.2f}",
            body=(
                f"z24={zs[0]:+.2f} z72={zs[1]:+.2f} z240={zs[2]:+.2f} "
                f"sigma_a={annual_vol:.2%} scaler={vol_scaler:.2f}"
            ),
            payload={
                "symbol": symbol, "venue": venue, "stance": stance,
                "z": zs, "annual_vol": annual_vol, "vol_scaler": vol_scaler,
            },
            importance=0.4 + 0.3 * abs(stance),
            tags=[symbol, venue, "cta"],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=stance,
                           explain={"z": zs, "scaler": vol_scaler})
