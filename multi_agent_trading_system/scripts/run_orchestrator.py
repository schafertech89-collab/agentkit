"""CLI: run the MATS orchestrator loop.

Usage:
    python -m mats.scripts.run_orchestrator --tier 1 --session mine --interval 60

Respects MATS_LIVE to gate real exchange writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

from mats.core.orchestrator import Orchestrator, OrchestratorConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("mats-orchestrator")
    p.add_argument("--tier", type=int, default=int(os.environ.get("MATS_TIER", "1")))
    p.add_argument("--session", type=str, default=os.environ.get("MATS_SESSION_ID", "default"))
    p.add_argument("--interval", type=float, default=60.0)
    p.add_argument("--universe", type=str, nargs="*", default=None,
                   help='e.g. "ETH-USD:coinbase BTC-USD:coinbase"')
    p.add_argument("--account-equity", type=float, default=10_000.0)
    p.add_argument("--vault", type=str, default=os.environ.get("OBSIDIAN_VAULT_PATH", "./vault"))
    p.add_argument("--scs", type=str, default="memory",
                   choices=["memory", "sqlite", "redis"])
    p.add_argument("--once", action="store_true", help="run a single tick and exit")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> OrchestratorConfig:
    universe = []
    for entry in args.universe or []:
        if ":" in entry:
            sym, venue = entry.split(":", 1)
        else:
            sym, venue = entry, "coinbase"
        universe.append((sym, venue))
    if not universe:
        universe = [("ETH-USD", "coinbase"), ("BTC-USD", "coinbase")]
    scs_kwargs = {}
    if args.scs == "redis":
        scs_kwargs["url"] = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
    elif args.scs == "sqlite":
        scs_kwargs["path"] = "./mats_data/scs.db"
    return OrchestratorConfig(
        tier=args.tier, session_id=args.session,
        universe=universe, tick_interval_sec=args.interval,
        account_equity=args.account_equity, vault_path=args.vault,
        scs_backend=args.scs, scs_kwargs=scs_kwargs,
    )


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cfg = build_config(args)
    orch = Orchestrator(cfg)
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, orch.stop)
    if args.once:
        summary = await orch.tick()
        print(json.dumps(summary, indent=2, default=str))
        return 0
    await orch.run()
    return 0


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
