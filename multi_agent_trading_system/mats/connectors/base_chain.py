"""Base (EVM L2) connector.

Thin wrapper over web3.py for balance reads, swap execution, and reward
sweeping. MATS uses this to:
  * read token balances on Base
  * build swaps through an aggregator (0x / ODOS / CDP quote API)
  * sweep Allora/PredictBase rewards to ``BASE_REWARD_WALLET``
  * call Wasabi + PredictBase contracts directly from this single wallet

Every write is wrapped with ``live_mode_enabled`` so signing only occurs in
live mode; otherwise we return a paper receipt with the intended calldata.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from ._common import Order, Receipt, live_mode_enabled


class BaseChainConnector:
    name = "base"
    CHAIN_ID = 8453

    def __init__(self, rpc_url: str | None = None, private_key: str | None = None,
                 reward_wallet: str | None = None) -> None:
        self.rpc_url = rpc_url or os.environ.get("BASE_RPC_URL", "https://mainnet.base.org")
        self.private_key = private_key or os.environ.get("BASE_WALLET_PRIVATE_KEY", "")
        self.reward_wallet = reward_wallet or os.environ.get("BASE_REWARD_WALLET", "")
        self._w3 = None  # lazy init so tests don't need a live RPC

    def _web3(self):
        if self._w3 is not None:
            return self._w3
        try:
            from web3 import Web3
            self._w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        except Exception:
            self._w3 = None
        return self._w3

    @property
    def address(self) -> str:
        if not self.private_key:
            return ""
        try:
            from eth_account import Account
            return Account.from_key(self.private_key).address
        except Exception:
            return ""

    # ---------------------------------------------------------- reads

    async def eth_balance(self, address: str | None = None) -> float:
        w3 = self._web3()
        if w3 is None:
            return 0.0
        addr = address or self.address
        if not addr:
            return 0.0
        wei = w3.eth.get_balance(addr)
        return float(wei) / 1e18

    async def erc20_balance(self, token_address: str, holder: str | None = None) -> float:
        w3 = self._web3()
        if w3 is None:
            return 0.0
        addr = holder or self.address
        if not addr:
            return 0.0
        abi = [
            {"constant": True, "inputs": [{"name": "", "type": "address"}],
             "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}],
             "stateMutability": "view", "type": "function"},
            {"constant": True, "inputs": [],
             "name": "decimals", "outputs": [{"name": "", "type": "uint8"}],
             "stateMutability": "view", "type": "function"},
        ]
        c = w3.eth.contract(address=w3.to_checksum_address(token_address), abi=abi)
        dec = c.functions.decimals().call()
        raw = c.functions.balanceOf(w3.to_checksum_address(addr)).call()
        return float(raw) / (10 ** dec)

    async def balances(self) -> dict[str, float]:
        eth = await self.eth_balance()
        return {"ETH": eth}

    # ---------------------------------------------------------- swap

    async def swap(self, *, token_in: str, token_out: str, amount_in: float,
                   min_amount_out: float = 0.0, router: str | None = None) -> Receipt:
        """Submit a swap via the configured router (0x/ODOS/CDP).

        In paper mode we return a simulated fill at the spot ticker.
        """
        router = router or os.environ.get("VENICE_SWAP_ROUTER", "")
        cid = f"base-swap-{uuid.uuid4().hex}"
        body = {
            "token_in": token_in,
            "token_out": token_out,
            "amount_in": amount_in,
            "min_amount_out": min_amount_out,
            "router": router,
        }
        if not live_mode_enabled() or not self.private_key:
            return Receipt(
                venue=self.name, symbol=f"{token_in}->{token_out}",
                order_id=cid, status="paper_filled",
                filled=amount_in, avg_price=0.0, paper=True, raw=body,
            )

        # Live path: fetch quote from router, sign, broadcast.
        import httpx  # lazy
        from eth_account import Account
        async with httpx.AsyncClient(timeout=10.0) as hx:
            q = await hx.get(router, params={
                "sellToken": token_in, "buyToken": token_out, "sellAmount": int(amount_in),
                "takerAddress": self.address, "chainId": self.CHAIN_ID,
            })
            q.raise_for_status()
            quote = q.json()
        tx = {
            "chainId": self.CHAIN_ID,
            "to": quote["to"],
            "data": quote["data"],
            "value": int(quote.get("value", 0)),
            "gas": int(quote.get("estimatedGas", 500_000)),
        }
        w3 = self._web3()
        tx["nonce"] = w3.eth.get_transaction_count(self.address)
        tx["maxFeePerGas"] = int(quote.get("maxFeePerGas", w3.eth.gas_price))
        tx["maxPriorityFeePerGas"] = int(quote.get("maxPriorityFeePerGas",
                                                    w3.to_wei(1, "gwei")))
        signed = Account.sign_transaction(tx, self.private_key)
        h = w3.eth.send_raw_transaction(signed.rawTransaction)
        return Receipt(
            venue=self.name, symbol=f"{token_in}->{token_out}",
            order_id=h.hex(), status="broadcast",
            filled=amount_in, avg_price=0.0, raw={"tx_hash": h.hex()},
        )

    # ---------------------------------------------------------- sweep

    async def sweep_rewards(self, tokens: list[str]) -> list[Receipt]:
        """Move reward balances to ``reward_wallet``."""
        out: list[Receipt] = []
        if not self.reward_wallet:
            return out
        for token in tokens:
            try:
                balance = await self.erc20_balance(token)
            except Exception:
                continue
            if balance <= 0:
                continue
            out.append(await self._transfer_erc20(token, self.reward_wallet, balance))
        return out

    async def _transfer_erc20(self, token_address: str, to: str, amount: float) -> Receipt:
        cid = f"base-sweep-{uuid.uuid4().hex}"
        if not live_mode_enabled() or not self.private_key:
            return Receipt(
                venue=self.name, symbol=token_address, order_id=cid,
                status="paper_swept", filled=amount, avg_price=0.0, paper=True,
                raw={"to": to, "amount": amount},
            )
        w3 = self._web3()
        from eth_account import Account
        abi = [
            {"constant": False,
             "inputs": [{"name": "_to", "type": "address"},
                        {"name": "_value", "type": "uint256"}],
             "name": "transfer",
             "outputs": [{"name": "", "type": "bool"}],
             "stateMutability": "nonpayable", "type": "function"},
            {"constant": True, "inputs": [], "name": "decimals",
             "outputs": [{"name": "", "type": "uint8"}],
             "stateMutability": "view", "type": "function"},
        ]
        c = w3.eth.contract(address=w3.to_checksum_address(token_address), abi=abi)
        dec = c.functions.decimals().call()
        raw_amount = int(amount * (10 ** dec))
        tx = c.functions.transfer(w3.to_checksum_address(to), raw_amount).build_transaction({
            "from": self.address,
            "nonce": w3.eth.get_transaction_count(self.address),
            "chainId": self.CHAIN_ID,
            "gas": 120_000,
            "maxFeePerGas": int(w3.eth.gas_price * 2),
            "maxPriorityFeePerGas": w3.to_wei(1, "gwei"),
        })
        signed = Account.sign_transaction(tx, self.private_key)
        h = w3.eth.send_raw_transaction(signed.rawTransaction)
        return Receipt(
            venue=self.name, symbol=token_address, order_id=h.hex(),
            status="sweep_broadcast", filled=amount, avg_price=0.0,
            raw={"tx_hash": h.hex(), "to": to},
        )
