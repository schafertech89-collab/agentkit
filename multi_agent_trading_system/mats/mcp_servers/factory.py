"""MCP server factory — builds individual servers or a composite one."""

from __future__ import annotations

from typing import Any

from ..connectors import (
    AlloraConnector,
    BaseChainConnector,
    CoinbaseConnector,
    PredictBaseConnector,
    VeniceConnector,
    WasabiConnector,
)
from ..core.scs import SCS, build_scs, session_key
from ..learning.ppo_policy import MemoryPolicyController
from ._mcp import MCPServer
from .obsidian_mcp import register_obsidian_tools


def _coinbase(server: MCPServer, connector: CoinbaseConnector | None = None) -> None:
    cb = connector or CoinbaseConnector()

    @server.tool("coinbase.ticker", "Return Coinbase Advanced ticker (bid/ask/price).",
                 {"type": "object", "properties": {"symbol": {"type": "string"}},
                  "required": ["symbol"]})
    async def ticker(symbol: str) -> dict:
        return (await cb.ticker(symbol)).__dict__

    @server.tool("coinbase.orderbook", "Return order book up to depth.",
                 {"type": "object", "properties": {"symbol": {"type": "string"},
                                                   "depth": {"type": "integer"}},
                  "required": ["symbol"]})
    async def orderbook(symbol: str, depth: int = 10) -> dict:
        book = await cb.orderbook(symbol, depth=depth)
        return {"symbol": book.symbol, "bids": [b.__dict__ for b in book.bids],
                "asks": [a.__dict__ for a in book.asks], "spread_bps": book.spread_bps()}

    @server.tool("coinbase.candles", "Return recent candles.",
                 {"type": "object", "properties": {"symbol": {"type": "string"},
                                                   "granularity": {"type": "string"},
                                                   "limit": {"type": "integer"}},
                  "required": ["symbol"]})
    async def candles(symbol: str, granularity: str = "ONE_HOUR", limit: int = 100) -> list[dict]:
        cs = await cb.candles(symbol, granularity=granularity, limit=limit)
        return [c.__dict__ for c in cs]

    @server.tool("coinbase.place_order", "Place a market or limit order.",
                 {"type": "object", "properties": {
                     "symbol": {"type": "string"}, "side": {"type": "string"},
                     "size": {"type": "number"}, "order_type": {"type": "string"},
                     "price": {"type": "number"}}, "required": ["symbol", "side", "size"]})
    async def place_order(symbol: str, side: str, size: float,
                          order_type: str = "market", price: float | None = None) -> dict:
        from ..connectors._common import Order
        r = await cb.place_order(Order(venue="coinbase", symbol=symbol, side=side,
                                        size=size, order_type=order_type, price=price))
        return r.__dict__


def _wasabi(server: MCPServer, connector: WasabiConnector | None = None) -> None:
    w = connector or WasabiConnector()

    @server.tool("wasabi.ticker", "Wasabi perp ticker.")
    async def ticker(symbol: str) -> dict:
        return (await w.ticker(symbol)).__dict__

    @server.tool("wasabi.open", "Open a leveraged position on Wasabi.")
    async def open_pos(symbol: str, side: str, size_usd: float, leverage: float = 1.0,
                       stop_loss: float | None = None, take_profit: float | None = None) -> dict:
        r = await w.open_position(symbol=symbol, side=side, size_usd=size_usd,
                                  leverage=leverage, stop_loss=stop_loss,
                                  take_profit=take_profit)
        return r.__dict__

    @server.tool("wasabi.close", "Close a Wasabi position by id.")
    async def close_pos(position_id: str) -> dict:
        return (await w.close_position(position_id=position_id)).__dict__


def _predictbase(server: MCPServer, connector: PredictBaseConnector | None = None) -> None:
    pb = connector or PredictBaseConnector()

    @server.tool("predictbase.list", "List open prediction markets on Base.")
    async def list_markets(tags: list[str] | None = None,
                           min_liquidity: float = 5_000.0) -> list[dict]:
        markets = await pb.list_markets(tags=tags, min_liquidity=min_liquidity)
        return [m.__dict__ for m in markets]

    @server.tool("predictbase.bet", "Place a YES/NO bet sized in USDC.")
    async def bet(market_id: str, side: str, size_usdc: float,
                  max_slippage_bps: int = 100) -> dict:
        r = await pb.place_bet(market_id=market_id, side=side,
                               size_usdc=size_usdc, max_slippage_bps=max_slippage_bps)
        return r.__dict__

    @server.tool("predictbase.redeem", "Redeem winning shares in a resolved market.")
    async def redeem(market_id: str) -> dict:
        return (await pb.redeem(market_id)).__dict__


def _base(server: MCPServer, connector: BaseChainConnector | None = None) -> None:
    b = connector or BaseChainConnector()

    @server.tool("base.eth_balance", "Return ETH balance for the trader wallet (or override).")
    async def eth_balance(address: str | None = None) -> float:
        return await b.eth_balance(address)

    @server.tool("base.erc20_balance", "Return ERC-20 token balance.")
    async def erc20(token_address: str, holder: str | None = None) -> float:
        return await b.erc20_balance(token_address, holder)

    @server.tool("base.swap", "Swap via the configured router.")
    async def swap(token_in: str, token_out: str, amount_in: float,
                   min_amount_out: float = 0.0) -> dict:
        return (await b.swap(token_in=token_in, token_out=token_out,
                             amount_in=amount_in,
                             min_amount_out=min_amount_out)).__dict__

    @server.tool("base.sweep_rewards", "Sweep ERC-20 balances to the reward wallet.")
    async def sweep(tokens: list[str]) -> list[dict]:
        receipts = await b.sweep_rewards(tokens)
        return [r.__dict__ for r in receipts]


def _allora(server: MCPServer, connector: AlloraConnector | None = None) -> None:
    a = connector or AlloraConnector()

    @server.tool("allora.topics", "List configured Allora topics.")
    async def topics() -> list[dict]:
        return await a.list_topics()

    @server.tool("allora.submit", "Submit a numeric prediction to a topic.")
    async def submit(topic_id: int, value: float, confidence: float = 0.5,
                     extra: dict | None = None) -> dict:
        from ..connectors.allora import Prediction
        return await a.submit_prediction(Prediction(
            topic_id=topic_id, value=value, confidence=confidence, extra=extra,
        ))

    @server.tool("allora.rewards", "List accrued rewards for this worker.")
    async def rewards() -> list[dict]:
        rws = await a.rewards()
        return [r.__dict__ for r in rws]

    @server.tool("allora.claim", "Claim & bridge rewards to a Base recipient.")
    async def claim(base_recipient: str) -> list[dict]:
        rws = await a.claim_rewards(base_recipient)
        return [r.__dict__ for r in rws]


def _venice(server: MCPServer, connector: VeniceConnector | None = None) -> None:
    v = connector or VeniceConnector()

    @server.tool("venice.quote", "Price quote for swapping a token to DIEM.")
    async def quote(token_in: str, amount_in: float) -> dict:
        return await v.price_quote(token_in, amount_in)

    @server.tool("venice.swap", "Swap a token to Venice.ai DIEM on Base.")
    async def swap(token_in: str, amount_in: float, min_out: float | None = None) -> dict:
        r = await v.swap_to_diem(token_in=token_in, amount_in=amount_in, min_out=min_out)
        return r.__dict__

    @server.tool("venice.diem_balance", "Return DIEM balance for holder (or trader wallet).")
    async def diem_balance(holder: str | None = None) -> float:
        return await v.diem_balance(holder)


def _scs(server: MCPServer, scs: SCS) -> None:
    @server.tool("scs.get", "Read a key from the shared context store.")
    async def get(key: str) -> Any:
        return await scs.get(key)

    @server.tool("scs.set", "Write a key to the SCS.")
    async def set_(key: str, value: Any, ttl: int | None = None) -> dict:
        await scs.set(key, value, ttl=ttl)
        return {"ok": True}

    @server.tool("scs.append", "Append an item to a list key in the SCS.")
    async def append(key: str, item: Any) -> dict:
        await scs.append(key, item)
        return {"ok": True}

    @server.tool("scs.keys", "List matching keys in the SCS.")
    async def keys(pattern: str = "scs:*") -> list[str]:
        return await scs.keys(pattern)


def _policy(server: MCPServer, policy: MemoryPolicyController | None = None) -> None:
    p = policy or MemoryPolicyController()

    @server.tool("policy.decide", "Run the RL memory policy against a MemCube snapshot.")
    async def decide(memcube: dict) -> dict:
        from ..core.memcube import MemCube
        cube = MemCube(**memcube)
        d = p.decide(cube)
        return {
            "action": d.action.name, "target_tier": d.target_tier,
            "decay_rate": d.decay_rate, "log_prob": d.log_prob, "value": d.value,
        }

    @server.tool("policy.learn", "Update the policy with a batch of trajectories.")
    async def learn(batch: list) -> dict:
        return p.learn(batch)


# ---------------------------------------------------------- factory


def build_server(name: str, *, scs: SCS | None = None,
                 vault_path: str | None = None,
                 policy: MemoryPolicyController | None = None,
                 connectors: dict[str, Any] | None = None) -> MCPServer:
    s = MCPServer(name=f"mats.{name}")
    connectors = connectors or {}

    if name == "coinbase":
        _coinbase(s, connectors.get("coinbase"))
    elif name == "base":
        _base(s, connectors.get("base"))
    elif name == "wasabi":
        _wasabi(s, connectors.get("wasabi"))
    elif name == "predictbase":
        _predictbase(s, connectors.get("predictbase"))
    elif name == "allora":
        _allora(s, connectors.get("allora"))
    elif name == "venice":
        _venice(s, connectors.get("venice"))
    elif name == "scs":
        if scs is None:
            scs = build_scs("memory")
        _scs(s, scs)
    elif name == "policy":
        _policy(s, policy)
    elif name == "obsidian":
        if vault_path is None:
            raise ValueError("obsidian server requires vault_path")
        register_obsidian_tools(s, vault_path)
    elif name == "composite":
        _coinbase(s, connectors.get("coinbase"))
        _base(s, connectors.get("base"))
        _wasabi(s, connectors.get("wasabi"))
        _predictbase(s, connectors.get("predictbase"))
        _allora(s, connectors.get("allora"))
        _venice(s, connectors.get("venice"))
        _scs(s, scs or build_scs("memory"))
        _policy(s, policy)
        if vault_path:
            register_obsidian_tools(s, vault_path)
    else:
        raise ValueError(f"unknown server name: {name}")
    return s


def build_all(*, scs: SCS | None = None, vault_path: str = "./vault") -> dict[str, MCPServer]:
    return {
        name: build_server(name, scs=scs, vault_path=vault_path)
        for name in ("coinbase", "base", "wasabi", "predictbase",
                     "allora", "venice", "scs", "policy", "obsidian")
    }
