"""Shared data models and helpers across all connectors."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import httpx


Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit", "stop", "stop_limit", "take_profit"]


@dataclass
class Ticker:
    symbol: str
    price: float
    bid: float
    ask: float
    ts: float = field(default_factory=time.time)


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    ts: float = field(default_factory=time.time)

    def mid(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return 0.5 * (self.bids[0].price + self.asks[0].price)

    def spread_bps(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        m = self.mid()
        return (self.asks[0].price - self.bids[0].price) / max(m, 1e-9) * 10_000.0


@dataclass
class Candle:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    venue: str
    symbol: str
    side: Side
    size: float
    order_type: OrderType = "market"
    price: float | None = None
    stop_price: float | None = None
    client_order_id: str | None = None
    leverage: float | None = None
    reduce_only: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class Receipt:
    venue: str
    symbol: str
    order_id: str
    status: str
    filled: float
    avg_price: float
    ts: float = field(default_factory=time.time)
    paper: bool = False
    raw: dict = field(default_factory=dict)


def live_mode_enabled() -> bool:
    return os.environ.get("MATS_LIVE", "0") == "1"


async def retry(
    fn,
    *,
    retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> Any:
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(min(max_delay, base_delay * (2 ** (attempt - 1))))


class HTTPMixin:
    """Shared async HTTP client. Each connector subclass owns one."""

    def __init__(self, base_url: str, headers: dict[str, str] | None = None,
                 timeout: float = 15.0) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, headers=headers or {}, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()
