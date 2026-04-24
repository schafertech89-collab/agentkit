"""MCP servers exposing MATS primitives as tools.

Each module defines a ``build_server()`` that returns an MCP-shaped FastAPI
app bound to ``stdio`` or Streamable-HTTP depending on how it's launched. For
stdio-only deployments we use ``fastmcp`` (lazy-imported).

Servers:
    coinbase_mcp      — ticker, orderbook, candles, place_order, cancel
    base_mcp          — balances, erc20, swap, sweep_rewards
    wasabi_mcp        — ticker, orderbook, open_position, close_position
    predictbase_mcp   — list_markets, place_bet, redeem
    allora_mcp        — submit_prediction, rewards, claim_rewards
    venice_mcp        — quote, swap_to_diem, diem_balance
    scs_mcp           — shared context store read/write/pubsub
    policy_mcp        — memory policy decide / learn
    obsidian_mcp      — list_files, get_file, patch, append, search_bm25
    orchestrator_mcp  — tick, status, reflection snapshot
"""

from .factory import build_all, build_server

__all__ = ["build_all", "build_server"]
