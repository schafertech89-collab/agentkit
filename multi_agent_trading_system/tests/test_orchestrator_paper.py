"""End-to-end paper-mode orchestrator tick using stubbed connectors."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import pytest

from mats.connectors._common import Candle, Order, OrderBook, OrderBookLevel, Receipt, Ticker
from mats.core.orchestrator import Orchestrator, OrchestratorConfig


class StubCoinbase:
    name = "coinbase"

    def __init__(self):
        rng = np.random.default_rng(0)
        self._prices = 100 + np.cumsum(rng.normal(0, 1.0, 240))

    async def ticker(self, symbol: str) -> Ticker:
        p = float(self._prices[-1])
        return Ticker(symbol=symbol, price=p, bid=p - 0.05, ask=p + 0.05)

    async def orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        p = float(self._prices[-1])
        return OrderBook(symbol=symbol,
                         bids=[OrderBookLevel(p - 0.1, 1.0)],
                         asks=[OrderBookLevel(p + 0.1, 1.0)])

    async def candles(self, symbol: str, granularity: str = "ONE_HOUR", limit: int = 200):
        return [
            Candle(ts=i, open=float(self._prices[i]), high=float(self._prices[i] + 0.2),
                   low=float(self._prices[i] - 0.2), close=float(self._prices[i]), volume=1.0)
            for i in range(min(limit, len(self._prices)))
        ]

    async def balances(self):
        return {"USD": 10_000, "ETH": 1.0}

    async def place_order(self, order: Order) -> Receipt:
        return Receipt(venue=self.name, symbol=order.symbol, order_id="stub",
                       status="paper_filled", filled=order.size, avg_price=100.0,
                       paper=True)


class StubChain:
    name = "base"
    CHAIN_ID = 8453
    address = "0xStubTrader"
    reward_wallet = "0xStubReward"
    private_key = ""

    async def eth_balance(self, addr=None): return 0.0
    async def erc20_balance(self, t, h=None): return 0.0
    async def balances(self): return {"ETH": 0.0}
    async def swap(self, **kw):
        return Receipt(venue="base", symbol="stub", order_id="stub", status="paper",
                       filled=0.0, avg_price=0.0, paper=True, raw=kw)
    async def sweep_rewards(self, tokens): return []
    def _web3(self): return None


class StubWasabi:
    name = "wasabi"
    async def ticker(self, s): return Ticker(s, 100.0, 99.9, 100.1)
    async def orderbook(self, s, depth=10): return OrderBook(s, [], [])
    async def open_position(self, **kw):
        return Receipt(venue="wasabi", symbol=kw["symbol"], order_id="stub",
                       status="paper_opened", filled=kw["size_usd"], avg_price=100.0,
                       paper=True)
    async def close_position(self, **kw):
        return Receipt(venue="wasabi", symbol="stub", order_id="stub",
                       status="paper_closed", filled=0.0, avg_price=0.0, paper=True)


class StubPredictBase:
    name = "predictbase"
    async def list_markets(self, **kw): return []
    async def place_bet(self, **kw):
        return Receipt(venue="predictbase", symbol=kw["market_id"], order_id="stub",
                       status="paper_bet", filled=kw["size_usdc"], avg_price=0.5,
                       paper=True)
    async def redeem(self, mid):
        return Receipt(venue="predictbase", symbol=mid, order_id="stub",
                       status="paper_redeem", filled=0.0, avg_price=0.0, paper=True)


class StubAllora:
    name = "allora"
    async def list_topics(self): return []
    async def submit_prediction(self, pred):
        return {"status": "paper_accepted", "topic_id": pred.topic_id}
    async def rewards(self): return []
    async def claim_rewards(self, base_recipient): return []


class StubVenice:
    name = "venice"
    async def price_quote(self, t, a): return {"rate": 1.0, "min_out": a}
    async def swap_to_diem(self, **kw):
        return Receipt(venue="venice", symbol="DIEM", order_id="stub",
                       status="paper", filled=0.0, avg_price=0.0, paper=True)
    async def diem_balance(self, holder=None): return 0.0


@pytest.mark.asyncio
async def test_paper_tick(tmp_path):
    cfg = OrchestratorConfig(
        universe=[("ETH-USD", "coinbase")],
        allora_topics=[(1, "ETH-USD")],
        reward_tokens=[],
        tick_interval_sec=0.01,
        vault_path=str(tmp_path / "vault"),
    )
    orch = Orchestrator(cfg)
    orch.connectors = {
        "coinbase": StubCoinbase(),
        "base": StubChain(),
        "wasabi": StubWasabi(),
        "predictbase": StubPredictBase(),
        "allora": StubAllora(),
        "venice": StubVenice(),
    }
    summary = await orch.tick()
    assert "symbols" in summary
    assert "ETH-USD" in summary["symbols"]
    assert -1.0 <= summary["symbols"]["ETH-USD"]["stance"] <= 1.0
