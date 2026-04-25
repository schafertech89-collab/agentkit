"""Cross-venue arbitrage agent.

For each symbol, fan out ``ticker()`` calls to all venues that carry it.
Compute the fee- and slippage-aware edge between venues. Emit a stance and
(when the edge is meaningful) a paired SizedOrder spec that the risk agent
can route through the existing ExecutionAgent.

The agent degrades gracefully:

  * If only one venue carries the symbol, stance is 0.
  * If a connector errors, that venue is skipped.
  * Edges below ``MIN_EDGE_BPS`` round-trip cost are ignored.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Agent, AgentContext, AgentOutput


# Round-trip cost floor — taker + taker + small slippage cushion (bps).
MIN_EDGE_BPS = 8.0
DEFAULT_VENUES = ("coinbase", "wasabi")


class CrossVenueArbAgent(Agent):
    name = "crossvenue_arb"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venues = list(input_payload.get("venues", DEFAULT_VENUES))
        quotes: dict[str, float] = {}
        for v in venues:
            connector = ctx.get_connector(v)
            if connector is None or not hasattr(connector, "ticker"):
                continue
            try:
                t = await connector.ticker(symbol)
                if t is None or not getattr(t, "price", None):
                    continue
                quotes[v] = float(t.price)
            except Exception:
                continue

        if len(quotes) < 2:
            cube = self.memcube(
                title=f"XV-arb {symbol} <2 venues",
                payload={"quotes": quotes, "symbol": symbol},
                importance=0.1,
            )
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        # Find the cheapest and richest venue.
        sorted_q = sorted(quotes.items(), key=lambda kv: kv[1])
        buy_venue, buy_px = sorted_q[0]
        sell_venue, sell_px = sorted_q[-1]
        mid = 0.5 * (buy_px + sell_px) or 1e-9
        edge_bps = (sell_px - buy_px) / mid * 10_000

        actions: list[dict] = []
        stance = 0.0
        if edge_bps > MIN_EDGE_BPS and buy_venue != sell_venue:
            # Stance points to the *buy* side — long the cheap venue.
            stance = float(np.clip((edge_bps - MIN_EDGE_BPS) / 50.0, 0.0, 1.0))
            actions.append({
                "kind": "xv_arb",
                "buy": {"venue": buy_venue, "symbol": symbol, "price": buy_px},
                "sell": {"venue": sell_venue, "symbol": symbol, "price": sell_px},
                "edge_bps": edge_bps,
            })

        await ctx.append("crossvenue_arb_stances", {
            "agent": self.name, "symbol": symbol, "stance": stance,
            "quotes": quotes, "edge_bps": edge_bps,
            "buy_venue": buy_venue, "sell_venue": sell_venue,
        })

        cube = self.memcube(
            title=f"XV-arb {symbol} edge={edge_bps:+.1f}bps",
            body=f"buy {buy_venue}@{buy_px:.4f} / sell {sell_venue}@{sell_px:.4f}",
            payload={
                "symbol": symbol, "stance": stance, "edge_bps": edge_bps,
                "buy_venue": buy_venue, "sell_venue": sell_venue,
                "quotes": quotes,
            },
            importance=0.4 + 0.3 * stance,
            tags=[symbol, "xv_arb"],
        )
        return AgentOutput(
            agent=self.name, memcube=cube, stance=stance,
            actions=actions, explain={"edge_bps": edge_bps, "quotes": quotes},
        )
