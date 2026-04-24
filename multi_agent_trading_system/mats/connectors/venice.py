"""Venice.ai DIEM connector.

Venice.ai DIEM is an ERC-20 on Base. The connector performs two actions:

  1. ``price_quote(token_in, amount_in)`` — returns how much DIEM would be
     received for ``amount_in`` of ``token_in`` via the configured swap router.
  2. ``swap_to_diem(token_in, amount_in)`` — executes the swap via
     ``BaseChainConnector.swap`` and tracks the resulting DIEM balance in
     the SCS / governance log.

The DIEM contract address is parameterised by ``VENICE_DIEM_CONTRACT`` so the
system adapts if DIEM migrates to a new deployment.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from ._common import HTTPMixin, Receipt, live_mode_enabled, retry
from .base_chain import BaseChainConnector


class VeniceConnector(HTTPMixin):
    name = "venice"

    def __init__(self, api_key: str | None = None, diem_contract: str | None = None,
                 swap_router: str | None = None, chain: BaseChainConnector | None = None) -> None:
        self.api_key = api_key or os.environ.get("VENICE_API_KEY", "")
        self.diem_contract = diem_contract or os.environ.get("VENICE_DIEM_CONTRACT", "")
        self.swap_router = swap_router or os.environ.get("VENICE_SWAP_ROUTER", "")
        self.chain = chain or BaseChainConnector()
        super().__init__(
            base_url="https://api.venice.ai",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            timeout=10.0,
        )

    async def price_quote(self, token_in: str, amount_in: float) -> dict:
        async def _call():
            r = await self._client.get(
                "/v1/diem/quote",
                params={"token_in": token_in, "amount_in": amount_in,
                        "chain": "base"},
            )
            r.raise_for_status()
            return r.json()

        try:
            return await retry(_call)
        except Exception:
            return {"rate": 0.0, "min_out": 0.0}

    async def swap_to_diem(self, token_in: str, amount_in: float,
                           min_out: float | None = None) -> Receipt:
        quote = await self.price_quote(token_in, amount_in)
        min_amount_out = float(min_out if min_out is not None else quote.get("min_out", 0.0))
        if not self.diem_contract:
            return Receipt(
                venue=self.name, symbol="DIEM",
                order_id=f"venice-{uuid.uuid4().hex}",
                status="missing_diem_contract", filled=0.0, avg_price=0.0,
                paper=True, raw=quote,
            )
        receipt = await self.chain.swap(
            token_in=token_in, token_out=self.diem_contract,
            amount_in=amount_in, min_amount_out=min_amount_out,
            router=self.swap_router or None,
        )
        receipt.metadata = {"quote": quote} if hasattr(receipt, "metadata") else {}
        return receipt

    async def diem_balance(self, holder: str | None = None) -> float:
        if not self.diem_contract:
            return 0.0
        return await self.chain.erc20_balance(self.diem_contract, holder)
