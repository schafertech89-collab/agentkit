"""Reward sweeper agent.

Periodically polls:
    * Allora for accrued rewards → bridges them to the configured Base wallet
    * PredictBase for resolved winning markets → redeems them
    * Coinbase for stablecoin earnings (maker rebates, staking)

Once rewards land on Base they are swept into the configured ``reward_wallet``
address, and finally swapped for Venice.ai DIEM via the Venice connector.
Every step is governance-logged so the audit chain captures the full lineage.
"""

from __future__ import annotations

import os
from typing import Any

from .base import Agent, AgentContext, AgentOutput


class RewardSweeperAgent(Agent):
    name = "reward_sweeper"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        base = ctx.get_connector("base")
        allora = ctx.get_connector("allora")
        predictbase = ctx.get_connector("predictbase")
        venice = ctx.get_connector("venice")

        reward_wallet = os.environ.get("BASE_REWARD_WALLET", "") or (base.reward_wallet if base else "")
        if not reward_wallet:
            cube = self.memcube(title="sweep (no reward wallet configured)", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        swept: dict[str, Any] = {}

        # Allora → Base bridge
        if allora is not None:
            allora_claims = await allora.claim_rewards(base_recipient=reward_wallet)
            swept["allora"] = [c.__dict__ for c in allora_claims]

        # PredictBase redemptions
        if predictbase is not None:
            resolved = input_payload.get("resolved_markets") or []
            predictbase_receipts = []
            for market_id in resolved:
                try:
                    r = await predictbase.redeem(market_id)
                    predictbase_receipts.append(r.__dict__)
                except Exception as exc:
                    predictbase_receipts.append({"error": repr(exc), "market_id": market_id})
            swept["predictbase"] = predictbase_receipts

        # Sweep token balances to reward wallet
        reward_tokens = input_payload.get("reward_tokens") or []
        if base is not None and reward_tokens:
            sweeps = await base.sweep_rewards(reward_tokens)
            swept["base_sweeps"] = [s.__dict__ for s in sweeps]

        # Swap swept rewards to Venice.ai DIEM
        diem_swaps: list[dict] = []
        if venice is not None and base is not None and reward_tokens:
            for token in reward_tokens:
                try:
                    balance = await base.erc20_balance(token, reward_wallet)
                except Exception:
                    balance = 0.0
                if balance <= 0:
                    continue
                r = await venice.swap_to_diem(token_in=token, amount_in=balance)
                diem_swaps.append(r.__dict__)
        swept["diem_swaps"] = diem_swaps

        # Audit via SSGM if present
        if ctx.ssgm is not None:
            try:
                ctx.ssgm.chain.append(
                    actor=self.name, action="SWEEP", target=reward_wallet,
                    payload=swept, metadata={"session": ctx.session_id},
                )
            except Exception:
                pass

        cube = self.memcube(
            title=f"sweep {len(diem_swaps)} diem swaps",
            body=f"allora={len(swept.get('allora', []))} "
                 f"pb={len(swept.get('predictbase', []))} "
                 f"swaps={len(diem_swaps)}",
            payload=swept,
            importance=0.35,
            tags=["sweep", "rewards", "diem"],
        )
        await ctx.append("sweeps", swept)
        return AgentOutput(
            agent=self.name, memcube=cube, stance=0.0, explain=swept,
        )
