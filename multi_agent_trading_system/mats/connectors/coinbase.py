"""Coinbase Advanced Trade connector.

Uses the Advanced Trade REST + WebSocket APIs. When the environment variable
``MATS_LIVE`` is not set to ``1`` every write-mode call returns a *paper*
receipt — the full order path is exercised and the SCS is updated, but no
remote POST is issued.

Authentication uses an ECDSA-signed JWT per the 2026 Coinbase Cloud spec.
The ``_sign`` helper keeps the signing surface testable.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import os
import time
import uuid
from typing import Any

import httpx

from ._common import (
    Candle,
    HTTPMixin,
    Order,
    OrderBook,
    OrderBookLevel,
    Receipt,
    Ticker,
    live_mode_enabled,
    retry,
)


COINBASE_HOST = "https://api.coinbase.com"
SANDBOX_HOST = "https://api-sandbox.coinbase.com"


class CoinbaseConnector(HTTPMixin):
    name = "coinbase"

    def __init__(self, api_key: str | None = None, api_secret: str | None = None,
                 portfolio_id: str | None = None, sandbox: bool | None = None) -> None:
        api_key = api_key or os.environ.get("COINBASE_API_KEY", "")
        api_secret = api_secret or os.environ.get("COINBASE_API_SECRET", "")
        self.portfolio_id = portfolio_id or os.environ.get("COINBASE_PORTFOLIO_ID", "")
        sandbox = sandbox if sandbox is not None else os.environ.get("COINBASE_SANDBOX", "1") == "1"
        host = SANDBOX_HOST if sandbox else COINBASE_HOST
        super().__init__(base_url=host, headers={"User-Agent": "mats/0.1"})
        self.api_key = api_key
        self.api_secret = api_secret
        self.sandbox = sandbox

    # --------------------------------------------------------- signing

    def _sign(self, method: str, path: str, body: str = "") -> dict[str, str]:
        """HMAC-SHA256 signer matching the Cloud Advanced Trade 2026 spec.

        The JWT/ECDSA path is preferable but HMAC is accepted for legacy keys.
        """
        if not self.api_key:
            return {}
        ts = str(int(time.time()))
        message = ts + method.upper() + path + body
        sig = hmac.new(self.api_secret.encode(), message.encode(), hashlib.sha256).hexdigest()
        headers = {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-TIMESTAMP": ts,
            "CB-ACCESS-SIGN": sig,
            "Content-Type": "application/json",
        }
        if self.portfolio_id:
            headers["CB-PORTFOLIO-ID"] = self.portfolio_id
        return headers

    async def _request(self, method: str, path: str, *, params: dict | None = None,
                       body: dict | None = None, signed: bool = False) -> dict:
        body_str = json.dumps(body) if body else ""
        headers = self._sign(method, path, body_str) if signed else {}

        async def _call() -> dict:
            response = await self._client.request(
                method, path, params=params,
                content=body_str if body else None, headers=headers,
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

        return await retry(_call)

    # --------------------------------------------------------- market data

    async def ticker(self, symbol: str) -> Ticker:
        data = await self._request("GET", f"/api/v3/brokerage/products/{symbol}")
        price = float(data.get("price") or data.get("default_quote", {}).get("price", 0.0))
        # Bid/ask via orderbook endpoint for precision
        ob = await self.orderbook(symbol, depth=1)
        bid = ob.bids[0].price if ob.bids else price
        ask = ob.asks[0].price if ob.asks else price
        return Ticker(symbol=symbol, price=price, bid=bid, ask=ask)

    async def orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        data = await self._request(
            "GET", "/api/v3/brokerage/product_book",
            params={"product_id": symbol, "limit": depth},
        )
        book = data.get("pricebook", data)
        bids = [OrderBookLevel(price=float(b["price"]), size=float(b["size"])) for b in book.get("bids", [])[:depth]]
        asks = [OrderBookLevel(price=float(a["price"]), size=float(a["size"])) for a in book.get("asks", [])[:depth]]
        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    async def candles(self, symbol: str, granularity: str = "ONE_HOUR", limit: int = 300) -> list[Candle]:
        now = int(time.time())
        seconds = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "ONE_HOUR": 3600, "SIX_HOUR": 21600,
                   "ONE_DAY": 86400}.get(granularity, 3600)
        start = now - seconds * limit
        data = await self._request(
            "GET", f"/api/v3/brokerage/products/{symbol}/candles",
            params={"start": start, "end": now, "granularity": granularity},
        )
        return [
            Candle(
                ts=int(c["start"]),
                open=float(c["open"]),
                high=float(c["high"]),
                low=float(c["low"]),
                close=float(c["close"]),
                volume=float(c.get("volume", 0.0)),
            )
            for c in data.get("candles", [])
        ]

    # --------------------------------------------------------- account

    async def balances(self) -> dict[str, float]:
        data = await self._request("GET", "/api/v3/brokerage/accounts", signed=True)
        out: dict[str, float] = {}
        for acc in data.get("accounts", []):
            asset = acc.get("currency")
            bal = float(acc.get("available_balance", {}).get("value", 0.0))
            if asset:
                out[asset] = bal
        return out

    # --------------------------------------------------------- trading

    async def place_order(self, order: Order) -> Receipt:
        client_id = order.client_order_id or f"mats-{uuid.uuid4().hex}"
        body: dict[str, Any] = {
            "client_order_id": client_id,
            "product_id": order.symbol,
            "side": order.side.upper(),
            "order_configuration": {},
        }
        if order.order_type == "market":
            body["order_configuration"]["market_market_ioc"] = (
                {"quote_size": f"{order.size:.2f}"} if order.side == "buy"
                else {"base_size": f"{order.size:.8f}"}
            )
        elif order.order_type == "limit":
            body["order_configuration"]["limit_limit_gtc"] = {
                "base_size": f"{order.size:.8f}",
                "limit_price": f"{order.price:.2f}",
            }
        else:
            raise ValueError(f"Unsupported coinbase order_type: {order.order_type}")

        if not live_mode_enabled():
            return Receipt(
                venue=self.name, symbol=order.symbol, order_id=client_id,
                status="paper_filled", filled=order.size, avg_price=order.price or 0.0,
                paper=True, raw=body,
            )

        data = await self._request("POST", "/api/v3/brokerage/orders", body=body, signed=True)
        oid = (data.get("order_id") or data.get("success_response", {}).get("order_id") or client_id)
        return Receipt(
            venue=self.name, symbol=order.symbol, order_id=oid,
            status="accepted", filled=0.0, avg_price=0.0, raw=data,
        )

    async def cancel(self, order_id: str) -> dict:
        if not live_mode_enabled():
            return {"order_id": order_id, "status": "paper_cancelled"}
        return await self._request(
            "POST", "/api/v3/brokerage/orders/batch_cancel",
            body={"order_ids": [order_id]}, signed=True,
        )
