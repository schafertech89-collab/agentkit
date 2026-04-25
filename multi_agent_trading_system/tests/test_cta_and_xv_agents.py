"""Unit tests for the CTA trend, cross-venue arb, and order-flow agents."""

from __future__ import annotations

import numpy as np
import pytest

from mats.agents import (
    CrossVenueArbAgent,
    CTATrendAgent,
    FeatureUnavailable,
    OrderFlowAgent,
)
from mats.agents.base import AgentContext
from mats.connectors._common import Candle, OrderBook, OrderBookLevel, Ticker
from mats.core.scs import build_scs


class _Conn:
    def __init__(self, prices: np.ndarray, ticker_price: float | None = None) -> None:
        self.prices = prices
        self._t = ticker_price

    async def candles(self, symbol, granularity="ONE_HOUR", limit=300):
        n = min(limit, len(self.prices))
        return [
            Candle(ts=i, open=float(self.prices[i]),
                   high=float(self.prices[i] + 0.2),
                   low=float(self.prices[i] - 0.2),
                   close=float(self.prices[i]), volume=1.0)
            for i in range(n)
        ]

    async def ticker(self, symbol):
        return Ticker(symbol=symbol, price=float(self._t or self.prices[-1]),
                      bid=float(self.prices[-1]) - 0.05,
                      ask=float(self.prices[-1]) + 0.05)


class _BookConn:
    def __init__(self, bid_size: float, ask_size: float) -> None:
        self.bid_size = bid_size
        self.ask_size = ask_size

    async def orderbook(self, symbol, depth=10):
        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(100.0, self.bid_size)],
            asks=[OrderBookLevel(100.1, self.ask_size)],
        )


@pytest.mark.asyncio
async def test_cta_stance_positive_on_strong_uptrend():
    rng = np.random.default_rng(0)
    prices = 100.0 + np.cumsum(rng.normal(0.5, 0.4, 300))  # strong drift up
    conn = _Conn(prices)
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": conn})
    agent = CTATrendAgent()
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venue": "coinbase"})
    assert out.stance > 0.0


@pytest.mark.asyncio
async def test_cta_stance_negative_on_strong_downtrend():
    rng = np.random.default_rng(1)
    prices = 100.0 + np.cumsum(rng.normal(-0.5, 0.4, 300))
    conn = _Conn(prices)
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": conn})
    agent = CTATrendAgent()
    out = await agent.step(ctx, {"symbol": "BTC-USD", "venue": "coinbase"})
    assert out.stance < 0.0


@pytest.mark.asyncio
async def test_cta_zero_stance_when_insufficient_data():
    conn = _Conn(np.array([100.0, 101.0, 102.0]))  # too few bars
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": conn})
    agent = CTATrendAgent()
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venue": "coinbase"})
    assert out.stance == 0.0


@pytest.mark.asyncio
async def test_xv_arb_emits_action_when_edge_above_floor():
    cheap = _Conn(np.array([99.0]), ticker_price=99.0)
    rich = _Conn(np.array([100.0]), ticker_price=100.0)  # 100bps edge
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": cheap, "wasabi": rich})
    agent = CrossVenueArbAgent()
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venues": ["coinbase", "wasabi"]})
    assert out.stance > 0.0
    assert any(a.get("kind") == "xv_arb" for a in out.actions)


@pytest.mark.asyncio
async def test_xv_arb_no_action_when_edge_below_floor():
    a = _Conn(np.array([100.00]), ticker_price=100.00)
    b = _Conn(np.array([100.01]), ticker_price=100.01)  # 1bps edge
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": a, "wasabi": b})
    agent = CrossVenueArbAgent()
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venues": ["coinbase", "wasabi"]})
    assert out.stance == 0.0
    assert out.actions == []


@pytest.mark.asyncio
async def test_xv_arb_zero_with_one_venue():
    a = _Conn(np.array([100.0]), ticker_price=100.0)
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": a})
    agent = CrossVenueArbAgent()
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venues": ["coinbase", "wasabi"]})
    assert out.stance == 0.0


@pytest.mark.asyncio
async def test_orderflow_strict_raises_without_tape_feed():
    conn = _BookConn(bid_size=10.0, ask_size=10.0)
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": conn})
    agent = OrderFlowAgent(strict=True)
    with pytest.raises(FeatureUnavailable):
        await agent.step(ctx, {"symbol": "ETH-USD", "venue": "coinbase"})


@pytest.mark.asyncio
async def test_orderflow_falls_back_to_snapshot_imbalance():
    conn = _BookConn(bid_size=30.0, ask_size=10.0)  # heavy bid side
    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": conn})
    agent = OrderFlowAgent(strict=False)
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venue": "coinbase"})
    assert out.stance > 0.0


@pytest.mark.asyncio
async def test_orderflow_consumes_tape_when_available():
    class _TapeConn:
        async def tape_events(self, symbol, limit=200):
            return [
                {"side": "buy", "size": 10.0, "avg_size": 1.0, "sweep": True},
                {"side": "buy", "size": 5.0, "avg_size": 1.0},
                {"side": "sell", "size": 1.0, "avg_size": 1.0},
            ]

        async def orderbook(self, symbol, depth=10):
            return OrderBook(symbol=symbol, bids=[], asks=[])

    ctx = AgentContext(session_id="t", scs=build_scs("memory"),
                       connectors={"coinbase": _TapeConn()})
    agent = OrderFlowAgent(strict=False)
    out = await agent.step(ctx, {"symbol": "ETH-USD", "venue": "coinbase"})
    assert out.stance > 0.0
    assert "flow" in out.explain
