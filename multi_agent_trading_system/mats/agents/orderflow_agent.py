"""Order-flow / tape-reading agent (gated).

SMB-Capital style microstructure inputs: imbalance, absorption, sweep
detection, trade-size clustering. None of the existing connectors expose a
real-time tape (`tape_events`) or orderbook-deltas stream — only snapshot
``orderbook()`` and ``candles()``. Without those, the agent operates in a
*degraded* mode that scores the most recent snapshot orderbook for
imbalance only and writes a low-confidence stance. When a future connector
provides ``tape_events()`` we'll branch on its presence.

The agent never raises in production; if data is missing it just emits a
zero-stance MemCube. A ``FeatureUnavailable`` exception is reserved for
explicit testing of the gated path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Agent, AgentContext, AgentOutput


class FeatureUnavailable(RuntimeError):
    """Raised when an order-flow feed is requested but not wired."""


class OrderFlowAgent(Agent):
    name = "orderflow"

    def __init__(self, *, strict: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.strict = strict

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        connector = ctx.get_connector(venue)
        if connector is None:
            return self._zero(symbol, "no connector")

        # Look for a real tape feed first. None of the shipped connectors
        # implement this yet — it stays as a forward-compatible hook.
        if hasattr(connector, "tape_events"):
            try:
                events = await connector.tape_events(symbol, limit=200)
            except Exception as exc:
                if self.strict:
                    raise FeatureUnavailable(f"tape_events failed: {exc}") from exc
                events = None
            if events:
                return await self._from_tape(ctx, symbol, venue, events)

        if self.strict:
            raise FeatureUnavailable(
                f"no tape_events feed on {venue}; order-flow agent gated until L2/L3 wired",
            )

        # Degraded mode: snapshot orderbook imbalance only.
        if not hasattr(connector, "orderbook"):
            return self._zero(symbol, "no orderbook")
        try:
            ob = await connector.orderbook(symbol, depth=10)
        except Exception:
            return self._zero(symbol, "orderbook failed")
        bid_size = sum(getattr(lvl, "size", 0.0) for lvl in (ob.bids or []))
        ask_size = sum(getattr(lvl, "size", 0.0) for lvl in (ob.asks or []))
        denom = bid_size + ask_size
        if denom <= 0:
            return self._zero(symbol, "empty book")
        imbalance = (bid_size - ask_size) / denom
        # Damp to small magnitudes — snapshot imbalance is weak signal.
        stance = float(np.clip(imbalance * 0.4, -0.5, 0.5))

        await ctx.append("orderflow_stances", {
            "agent": self.name, "symbol": symbol, "venue": venue, "stance": stance,
            "imbalance": imbalance, "mode": "snapshot",
        })
        cube = self.memcube(
            title=f"OF {symbol} imb={imbalance:+.2f}",
            body=f"snapshot-mode book imbalance bid={bid_size:.2f} ask={ask_size:.2f}",
            payload={"symbol": symbol, "venue": venue, "stance": stance,
                     "imbalance": imbalance, "mode": "snapshot"},
            importance=0.3 + 0.2 * abs(stance),
            tags=[symbol, venue, "orderflow"],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=stance,
                           explain={"imbalance": imbalance, "mode": "snapshot"})

    # ----------------------------------------------------- helpers
    async def _from_tape(self, ctx: AgentContext, symbol: str, venue: str,
                         events: list[dict]) -> AgentOutput:
        # Aggregate signed prints, sweep detection.
        signed = 0.0
        sweeps = 0
        big = 0
        for ev in events:
            side = (ev.get("side") or "").lower()
            sz = float(ev.get("size", 0.0))
            sign = 1.0 if side == "buy" else -1.0 if side == "sell" else 0.0
            signed += sign * sz
            if ev.get("sweep"):
                sweeps += 1
            if sz > float(ev.get("avg_size", sz)) * 3:
                big += 1
        total = sum(abs(float(ev.get("size", 0.0))) for ev in events) or 1.0
        flow = signed / total
        stance = float(np.clip(flow + 0.05 * np.sign(flow) * (sweeps + big), -1.0, 1.0))
        await ctx.append("orderflow_stances", {
            "agent": self.name, "symbol": symbol, "venue": venue, "stance": stance,
            "flow": flow, "sweeps": sweeps, "big": big, "mode": "tape",
        })
        cube = self.memcube(
            title=f"OF {symbol} flow={flow:+.2f} sweep={sweeps}",
            body=f"tape n={len(events)} signed={signed:+.2f} big={big}",
            payload={"symbol": symbol, "venue": venue, "stance": stance,
                     "flow": flow, "sweeps": sweeps, "big": big, "mode": "tape"},
            importance=0.5 + 0.3 * abs(stance),
            tags=[symbol, venue, "orderflow", "tape"],
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=stance,
                           explain={"flow": flow, "sweeps": sweeps, "big": big})

    def _zero(self, symbol: str, reason: str) -> AgentOutput:
        cube = self.memcube(
            title=f"OF {symbol} ({reason})",
            payload={"symbol": symbol, "stance": 0.0, "reason": reason},
            importance=0.05,
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=0.0,
                           explain={"reason": reason})
