"""Wasabi Protocol connector (perps / leverage on Base).

Wasabi on Base offers leveraged long/short positions via an orderbook at
``WASABI_ORDERBOOK_URL`` and a position-manager contract at
``WASABI_POOL_ADDRESS``. The connector exposes a simple open/close/read API.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from ._common import (
    HTTPMixin,
    Order,
    OrderBook,
    OrderBookLevel,
    Receipt,
    Ticker,
    live_mode_enabled,
    retry,
)
from .base_chain import BaseChainConnector


class WasabiConnector(HTTPMixin):
    name = "wasabi"

    def __init__(self, orderbook_url: str | None = None, pool_address: str | None = None,
                 chain: BaseChainConnector | None = None) -> None:
        self.orderbook_url = orderbook_url or os.environ.get(
            "WASABI_ORDERBOOK_URL", "https://api.wasabi.xyz/orderbook"
        )
        self.pool_address = pool_address or os.environ.get("WASABI_POOL_ADDRESS", "")
        self.chain = chain or BaseChainConnector()
        super().__init__(base_url=self.orderbook_url, timeout=10.0)

    async def ticker(self, symbol: str) -> Ticker:
        async def _call():
            r = await self._client.get(f"/ticker", params={"symbol": symbol})
            r.raise_for_status()
            return r.json()

        data = await retry(_call)
        return Ticker(
            symbol=symbol, price=float(data.get("price", 0.0)),
            bid=float(data.get("bid", 0.0)), ask=float(data.get("ask", 0.0)),
        )

    async def orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        async def _call():
            r = await self._client.get("/book", params={"symbol": symbol, "depth": depth})
            r.raise_for_status()
            return r.json()

        data = await retry(_call)
        bids = [OrderBookLevel(float(b[0]), float(b[1])) for b in data.get("bids", [])][:depth]
        asks = [OrderBookLevel(float(a[0]), float(a[1])) for a in data.get("asks", [])][:depth]
        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    async def positions(self) -> list[dict]:
        addr = self.chain.address
        if not addr:
            return []
        try:
            r = await self._client.get("/positions", params={"address": addr})
            r.raise_for_status()
            return r.json().get("positions", [])
        except Exception:
            return []

    async def open_position(
        self,
        *,
        symbol: str,
        side: str,
        size_usd: float,
        leverage: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Receipt:
        cid = f"wasabi-{uuid.uuid4().hex}"
        intent = {
            "symbol": symbol,
            "side": side,
            "size_usd": size_usd,
            "leverage": leverage,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "pool": self.pool_address,
            "owner": self.chain.address,
            "client_id": cid,
        }
        if not live_mode_enabled():
            return Receipt(
                venue=self.name, symbol=symbol, order_id=cid,
                status="paper_opened", filled=size_usd, avg_price=0.0,
                paper=True, raw=intent,
            )
        # Live path: build calldata via Wasabi API, then sign+send through BaseChainConnector.
        async def _call():
            r = await self._client.post("/intent/open", json=intent)
            r.raise_for_status()
            return r.json()

        quote = await retry(_call)
        tx_receipt = await self._send_calldata(quote)
        return Receipt(
            venue=self.name, symbol=symbol, order_id=tx_receipt.get("tx_hash", cid),
            status="opened", filled=size_usd, avg_price=float(quote.get("entry_price", 0.0)),
            raw=tx_receipt,
        )

    async def close_position(self, *, position_id: str) -> Receipt:
        cid = f"wasabi-close-{uuid.uuid4().hex}"
        intent = {"position_id": position_id, "owner": self.chain.address}
        if not live_mode_enabled():
            return Receipt(
                venue=self.name, symbol=position_id, order_id=cid,
                status="paper_closed", filled=0.0, avg_price=0.0, paper=True,
                raw=intent,
            )
        async def _call():
            r = await self._client.post("/intent/close", json=intent)
            r.raise_for_status()
            return r.json()

        quote = await retry(_call)
        tx_receipt = await self._send_calldata(quote)
        return Receipt(
            venue=self.name, symbol=position_id, order_id=tx_receipt.get("tx_hash", cid),
            status="closed", filled=0.0, avg_price=0.0, raw=tx_receipt,
        )

    async def _send_calldata(self, quote: dict) -> dict:
        w3 = self.chain._web3()
        if w3 is None:
            return {"error": "no web3 provider"}
        from eth_account import Account
        tx = {
            "chainId": self.chain.CHAIN_ID,
            "to": quote["to"],
            "data": quote["data"],
            "value": int(quote.get("value", 0)),
            "gas": int(quote.get("gas", 500_000)),
            "nonce": w3.eth.get_transaction_count(self.chain.address),
            "maxFeePerGas": int(quote.get("maxFeePerGas", w3.eth.gas_price)),
            "maxPriorityFeePerGas": int(quote.get("maxPriorityFeePerGas",
                                                  w3.to_wei(1, "gwei"))),
        }
        signed = Account.sign_transaction(tx, self.chain.private_key)
        h = w3.eth.send_raw_transaction(signed.rawTransaction)
        return {"tx_hash": h.hex()}
