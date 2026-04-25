"""CLI: launch one of the MCP servers.

Usage:
    python -m mats.scripts.run_mcp_server --name composite --http --port 9001
    python -m mats.scripts.run_mcp_server --name coinbase --stdio
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from mats.mcp_servers.factory import build_server


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("mats-mcp")
    p.add_argument("--name", required=True,
                   choices=["coinbase", "base", "wasabi", "predictbase",
                            "allora", "venice", "scs", "policy", "obsidian",
                            "composite"])
    p.add_argument("--http", action="store_true")
    p.add_argument("--stdio", action="store_true")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--vault", type=str,
                   default=os.environ.get("OBSIDIAN_VAULT_PATH", "./vault"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    server = build_server(args.name, vault_path=args.vault)
    if args.stdio or not args.http:
        asyncio.run(server.serve_stdio())
        return
    import uvicorn
    default_port = {
        "coinbase": 9011, "base": 9012, "wasabi": 9013, "predictbase": 9014,
        "allora": 9015, "venice": 9016, "scs": 9017, "policy": 9018,
        "obsidian": 9019, "composite": 9010,
    }[args.name]
    port = args.port or default_port
    uvicorn.run(server.app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
