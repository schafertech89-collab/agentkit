"""PredictBase connector — prediction markets on Base.

PredictBase exposes YES/NO binary and scalar markets. Each market has a
factory-deployed AMM address. The connector discovers markets, reads implied
prices, and posts trades via the factory contract.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from ._common import HTTPMixin, Receipt, live_mode_enabled, retry
from .base_chain import BaseChainConnector


@dataclass
class Market:
    market_id: str
    question: str
    outcomes: list[str]
    yes_price: float
    no_price: float
    liquidity_usd: float
    resolution_ts: float
    address: str


class PredictBaseConnector(HTTPMixin):
    name = "predictbase"

    def __init__(self, api_url: str | None = None, factory: str | None = None,
                 chain: BaseChainConnector | None = None) -> None:
        self.api_url = api_url or os.environ.get("PREDICTBASE_API_URL", "https://api.predictbase.xyz")
        self.factory = factory or os.environ.get("PREDICTBASE_FACTORY_ADDRESS", "")
        self.chain = chain or BaseChainConnector()
        super().__init__(base_url=self.api_url, timeout=10.0)

    # --------------------------------------------------------- discovery

    async def list_markets(self, *, tags: list[str] | None = None,
                           min_liquidity: float = 5_000.0) -> list[Market]:
        async def _call():
            r = await self._client.get(
                "/markets",
                params={
                    "min_liquidity_usd": min_liquidity,
                    "tags": ",".join(tags) if tags else None,
                    "status": "open",
                },
            )
            r.raise_for_status()
            return r.json()

        try:
            data = await retry(_call)
        except Exception:
            return []
        out: list[Market] = []
        for m in data.get("markets", []):
            out.append(
                Market(
                    market_id=m.get("id", ""),
                    question=m.get("question", ""),
                    outcomes=m.get("outcomes", ["YES", "NO"]),
                    yes_price=float(m.get("yes_price", 0.5)),
                    no_price=float(m.get("no_price", 0.5)),
                    liquidity_usd=float(m.get("liquidity_usd", 0.0)),
                    resolution_ts=float(m.get("resolution_ts", 0.0)),
                    address=m.get("address", ""),
                )
            )
        return out

    async def market_history(self, market_id: str, limit: int = 240) -> list[dict]:
        async def _call():
            r = await self._client.get(
                f"/markets/{market_id}/history", params={"limit": limit}
            )
            r.raise_for_status()
            return r.json()

        try:
            data = await retry(_call)
        except Exception:
            return []
        return data.get("points", [])

    # --------------------------------------------------------- trading

    async def place_bet(
        self,
        *,
        market_id: str,
        side: str,
        size_usdc: float,
        max_slippage_bps: int = 100,
    ) -> Receipt:
        """Buy YES or NO shares for ``size_usdc`` worth of USDC."""
        cid = f"predictbase-{uuid.uuid4().hex}"
        intent = {
            "market_id": market_id,
            "side": side.upper(),
            "size_usdc": size_usdc,
            "max_slippage_bps": max_slippage_bps,
            "buyer": self.chain.address,
            "client_id": cid,
        }
        if not live_mode_enabled():
            return Receipt(
                venue=self.name, symbol=market_id, order_id=cid,
                status="paper_bet", filled=size_usdc, avg_price=0.0, paper=True,
                raw=intent,
            )
        async def _call():
            r = await self._client.post("/orders", json=intent)
            r.raise_for_status()
            return r.json()

        quote = await retry(_call)
        if quote.get("tx"):
            tx_receipt = await self._send_calldata(quote["tx"])
            quote.update(tx_receipt)
        return Receipt(
            venue=self.name, symbol=market_id,
            order_id=quote.get("tx_hash", cid),
            status=quote.get("status", "accepted"),
            filled=float(quote.get("shares", 0.0)),
            avg_price=float(quote.get("avg_price", 0.0)),
            raw=quote,
        )

    async def redeem(self, market_id: str) -> Receipt:
        cid = f"predictbase-redeem-{uuid.uuid4().hex}"
        if not live_mode_enabled():
            return Receipt(
                venue=self.name, symbol=market_id, order_id=cid,
                status="paper_redeem", filled=0.0, avg_price=0.0, paper=True,
            )
        async def _call():
            r = await self._client.post(f"/markets/{market_id}/redeem",
                                        json={"holder": self.chain.address})
            r.raise_for_status()
            return r.json()

        quote = await retry(_call)
        if quote.get("tx"):
            tx_receipt = await self._send_calldata(quote["tx"])
            quote.update(tx_receipt)
        return Receipt(
            venue=self.name, symbol=market_id, order_id=quote.get("tx_hash", cid),
            status="redeemed", filled=float(quote.get("payout_usdc", 0.0)),
            avg_price=0.0, raw=quote,
        )

    async def _send_calldata(self, quote_tx: dict) -> dict:
        w3 = self.chain._web3()
        if w3 is None:
            return {"error": "no web3"}
        from eth_account import Account
        tx = {
            "chainId": self.chain.CHAIN_ID,
            "to": quote_tx["to"],
            "data": quote_tx["data"],
            "value": int(quote_tx.get("value", 0)),
            "gas": int(quote_tx.get("gas", 300_000)),
            "nonce": w3.eth.get_transaction_count(self.chain.address),
            "maxFeePerGas": int(quote_tx.get("maxFeePerGas", w3.eth.gas_price)),
            "maxPriorityFeePerGas": int(quote_tx.get("maxPriorityFeePerGas",
                                                     w3.to_wei(1, "gwei"))),
        }
        signed = Account.sign_transaction(tx, self.chain.private_key)
        h = w3.eth.send_raw_transaction(signed.rawTransaction)
        return {"tx_hash": h.hex()}
