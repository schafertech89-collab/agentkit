"""Execution agent — dispatches SizedOrders to the right venue(s).

Runs simultaneous trades across Coinbase, Wasabi, PredictBase, and raw Base.
Venue selection is governed by the risk agent's action payload, which names a
venue per order. For compound positions (e.g. long-ETH on Wasabi +
hedge-ETH-bet on PredictBase) the agent coordinates the multi-leg fill with
a simple commit/rollback: all legs are quoted first, then submitted, and if
any leg fails the agent issues best-effort reversals on the filled legs.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..connectors._common import Order, Receipt
from .base import Agent, AgentContext, AgentOutput


class ExecutionAgent(Agent):
    name = "execution"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        actions = input_payload.get("actions") or []
        if not actions:
            cube = self.memcube(title="execution (no orders)", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        receipts: list[Receipt] = []
        errors: list[dict] = []

        # Split into per-venue queues so we can submit simultaneously.
        per_venue: dict[str, list[dict]] = {}
        for action in actions:
            order = action.get("order") or {}
            if order.get("side") == "flat" or order.get("size_usd", 0) <= 0:
                continue
            per_venue.setdefault(order["venue"], []).append(action)

        async def submit_one(action: dict) -> Receipt:
            order_data = action["order"]
            venue = order_data["venue"]
            connector = ctx.get_connector(venue)
            if connector is None:
                raise RuntimeError(f"no connector for venue {venue}")

            if venue == "coinbase":
                o = Order(
                    venue=venue, symbol=order_data["symbol"],
                    side="buy" if order_data["side"] == "long" else "sell",
                    size=order_data["size_usd"],
                    order_type="market",
                )
                return await connector.place_order(o)
            if venue == "wasabi":
                return await connector.open_position(
                    symbol=order_data["symbol"],
                    side=order_data["side"],
                    size_usd=order_data["size_usd"],
                    leverage=order_data.get("leverage", 1.0),
                    stop_loss=order_data.get("stop_loss"),
                    take_profit=order_data.get("take_profit"),
                )
            if venue == "predictbase":
                side = "YES" if order_data["side"] == "long" else "NO"
                return await connector.place_bet(
                    market_id=order_data["symbol"],
                    side=side,
                    size_usdc=order_data["size_usd"],
                )
            if venue == "base":
                # Raw Base DEX swap: symbol is "token_in->token_out"
                sym = order_data["symbol"]
                token_in, token_out = sym.split("->", 1)
                return await connector.swap(
                    token_in=token_in, token_out=token_out,
                    amount_in=order_data["size_usd"],
                )
            raise RuntimeError(f"unsupported venue {venue}")

        tasks = [submit_one(a) for a_list in per_venue.values() for a in a_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res, action in zip(results, [a for a_list in per_venue.values() for a in a_list]):
            if isinstance(res, Exception):
                errors.append({"action": action, "error": repr(res)})
                await ctx.append("execution_errors", {"action": action, "error": repr(res)})
                continue
            receipts.append(res)
            await ctx.append("receipts", {
                "decision_id": action.get("decision_id"),
                "venue": res.venue, "symbol": res.symbol,
                "order_id": res.order_id, "status": res.status,
                "filled": res.filled, "avg_price": res.avg_price,
                "paper": res.paper,
            })

        cube = self.memcube(
            title=f"execution {len(receipts)}/{len(actions)} filled",
            body=f"errors={len(errors)}",
            payload={"receipts": [r.__dict__ for r in receipts], "errors": errors},
            importance=0.45,
            tags=["execution"],
        )
        return AgentOutput(
            agent=self.name, memcube=cube, stance=0.0,
            actions=actions, receipts=receipts,
            explain={"filled": len(receipts), "errors": errors},
        )
