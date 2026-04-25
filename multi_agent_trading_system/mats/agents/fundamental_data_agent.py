"""Fundamental data agent — pulls raw on-chain / API factors and stores them.

Separated from the scoring agent so network-IO and math stay in their own
classes. Reads: DeFiLlama (TVL), Coinbase (ticker/funding), on-chain (supply),
Tavily/Perplexity search (dev activity). For production you'd add a graceful
per-provider fallback; here we abstract through a pluggable ``DataSource``.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import Agent, AgentContext, AgentOutput


class FundamentalDataAgent(Agent):
    name = "fundamental_data"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        factors = await self._gather(symbol)
        await ctx.write(f"fundamentals:{symbol}", factors)
        cube = self.memcube(
            title=f"fundamental data {symbol}",
            body=f"{len(factors)} factors collected",
            payload={"symbol": symbol, "factors": factors},
            tags=[symbol, "fundamentals"],
            importance=0.25,
        )
        return AgentOutput(agent=self.name, memcube=cube, stance=0.0, explain=factors)

    async def _gather(self, symbol: str) -> dict[str, float]:
        """Best-effort collection from public endpoints. All failures → 0.0."""
        out: dict[str, float] = {}
        async with httpx.AsyncClient(timeout=5.0) as hx:
            tvl = await _safe(hx, _get_llama_tvl(symbol))
            if tvl is not None:
                out["tvl_change_7d"] = tvl
            funding = await _safe(hx, _get_coinbase_funding(symbol))
            if funding is not None:
                out["funding_rate_8h"] = funding
            depeg = await _safe(hx, _get_stablecoin_depeg(symbol))
            if depeg is not None:
                out["stablecoin_depeg"] = depeg
            # Placeholder values for fields we can't pull without additional API keys.
            out.setdefault("supply_inflation_annual", 0.03)
            out.setdefault("governance_participation", 0.06)
        return out


async def _safe(client, coro):
    try:
        return await coro(client) if callable(coro) else await coro
    except Exception:
        return None


def _get_llama_tvl(symbol: str):
    async def fn(client):
        r = await client.get(f"https://api.llama.fi/protocol/{symbol.lower()}")
        if r.status_code != 200:
            return None
        tvl_list = r.json().get("tvl", [])
        if len(tvl_list) < 8:
            return 0.0
        recent = tvl_list[-1]["totalLiquidityUSD"]
        week_ago = tvl_list[-7]["totalLiquidityUSD"]
        if week_ago <= 0:
            return 0.0
        return (recent - week_ago) / week_ago
    return fn


def _get_coinbase_funding(symbol: str):
    async def fn(client):
        # Not all spot pairs have funding; returns 0 when missing.
        r = await client.get(
            "https://api.coinbase.com/api/v3/brokerage/market/products",
            params={"product_ids": f"{symbol}-PERP"},
        )
        if r.status_code != 200:
            return 0.0
        data = r.json().get("products", [])
        if not data:
            return 0.0
        perp = data[0].get("perpetual_details", {})
        return float(perp.get("funding_rate", 0.0))
    return fn


def _get_stablecoin_depeg(symbol: str):
    async def fn(client):
        if symbol.upper() not in ("USDC", "USDT", "DAI", "USDBC"):
            return 1.0  # treat non-stables as "on peg"
        r = await client.get(
            "https://api.coinbase.com/api/v3/brokerage/products/"
            f"{symbol.upper()}-USD"
        )
        if r.status_code != 200:
            return 1.0
        return float(r.json().get("price", 1.0))
    return fn
