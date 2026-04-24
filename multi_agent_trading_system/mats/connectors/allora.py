"""Allora Network connector — prediction submission + reward monitoring.

Allora lets registered workers submit numeric predictions against one of
several *topics* (e.g. ETH-price-1h). Rewards are paid out in $ALLO to a
Cosmos-SDK address derived from the worker key, and bridged to Base via the
built-in burn/mint bridge.

The MCP flow is:
  1. Agent posts a prediction via ``submit_prediction``
  2. Periodically the reward sweeper calls ``claim_rewards`` which triggers
     the bridge to Base
  3. Once rewards land on Base they get swept to the configured reward wallet
     by the BaseChainConnector
  4. The Venice connector exchanges the swept rewards for DIEM
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from ._common import HTTPMixin, live_mode_enabled, retry


@dataclass
class Prediction:
    topic_id: int
    value: float
    confidence: float = 0.5
    extra: dict | None = None


@dataclass
class AlloraReward:
    topic_id: int
    amount_allo: float
    bridged_to_base: bool
    tx_hash: str = ""


class AlloraConnector(HTTPMixin):
    name = "allora"

    def __init__(self, rpc_url: str | None = None, worker_key: str | None = None,
                 topic_ids: list[int] | None = None, chain_id: str | None = None) -> None:
        self.rpc_url = rpc_url or os.environ.get("ALLORA_RPC_URL", "https://rpc.allora.network")
        self.worker_key = worker_key or os.environ.get("ALLORA_WORKER_KEY", "")
        self.chain_id = chain_id or os.environ.get("ALLORA_CHAIN_ID", "allora-testnet-1")
        topics_env = os.environ.get("ALLORA_TOPIC_IDS", "")
        self.topic_ids = topic_ids or [int(x.strip()) for x in topics_env.split(",") if x.strip()]
        super().__init__(base_url=self.rpc_url, timeout=15.0)

    # ------------------------------------------------------------- topics

    async def list_topics(self) -> list[dict]:
        async def _call():
            r = await self._client.get("/topics")
            r.raise_for_status()
            return r.json()

        try:
            data = await retry(_call)
        except Exception:
            return []
        return data.get("topics", [])

    async def topic_detail(self, topic_id: int) -> dict:
        async def _call():
            r = await self._client.get(f"/topics/{topic_id}")
            r.raise_for_status()
            return r.json()

        try:
            return await retry(_call)
        except Exception:
            return {"topic_id": topic_id}

    # ------------------------------------------------------------- submit

    async def submit_prediction(self, pred: Prediction) -> dict:
        nonce = int(time.time() * 1000)
        payload = {
            "topic_id": pred.topic_id,
            "value": pred.value,
            "confidence": pred.confidence,
            "nonce": nonce,
            "extra": pred.extra or {},
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        sig = self._sign(canonical)
        body = {"payload": base64.b64encode(canonical).decode(), "signature": sig,
                "worker_pub": self._pubkey_hint()}

        if not live_mode_enabled():
            return {
                "status": "paper_accepted",
                "topic_id": pred.topic_id,
                "value": pred.value,
                "nonce": nonce,
                "tx_hash": f"paper-{uuid.uuid4().hex}",
            }

        async def _call():
            r = await self._client.post("/inference/submit", json=body)
            r.raise_for_status()
            return r.json()

        return await retry(_call)

    # ----------------------------------------------------------- rewards

    async def rewards(self) -> list[AlloraReward]:
        async def _call():
            r = await self._client.get("/rewards", params={"worker": self._pubkey_hint()})
            r.raise_for_status()
            return r.json()

        try:
            data = await retry(_call)
        except Exception:
            return []
        return [
            AlloraReward(
                topic_id=int(r.get("topic_id", 0)),
                amount_allo=float(r.get("amount", 0.0)),
                bridged_to_base=bool(r.get("bridged_to_base", False)),
                tx_hash=r.get("tx_hash", ""),
            )
            for r in data.get("rewards", [])
        ]

    async def claim_rewards(self, base_recipient: str) -> list[AlloraReward]:
        """Triggers the bridge. ``base_recipient`` is a Base-chain address."""
        body = {"recipient_base": base_recipient, "worker": self._pubkey_hint()}
        if not live_mode_enabled():
            return [AlloraReward(topic_id=t, amount_allo=0.0, bridged_to_base=False,
                                 tx_hash=f"paper-{uuid.uuid4().hex}")
                    for t in self.topic_ids]
        async def _call():
            r = await self._client.post("/rewards/bridge", json=body)
            r.raise_for_status()
            return r.json()

        data = await retry(_call)
        return [
            AlloraReward(
                topic_id=int(r.get("topic_id", 0)),
                amount_allo=float(r.get("amount", 0.0)),
                bridged_to_base=bool(r.get("bridged_to_base", True)),
                tx_hash=r.get("tx_hash", ""),
            )
            for r in data.get("claims", [])
        ]

    # ---------------------------------------------------------- signing

    def _sign(self, message: bytes) -> str:
        if not self.worker_key:
            return ""
        digest = hmac.new(self.worker_key.encode(), message, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _pubkey_hint(self) -> str:
        if not self.worker_key:
            return ""
        return hashlib.sha256(self.worker_key.encode()).hexdigest()[:40]
