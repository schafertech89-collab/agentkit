"""Backtest harness for the Sutton-style learners.

Replays a synthetic or pre-saved 1-hour bar series through the orchestrator
tick path with the new learning agents enabled, and reports per-mode metrics
(Sharpe, hit rate, terminal weight drift). Intended for development; not a
walk-forward research harness.

Usage::

    python -m mats.scripts.run_backtest --ticks 200 --seed 0
    python -m mats.scripts.run_backtest --ticks 200 --modes baseline,td,full

Modes:
  * baseline  — DEFAULT_STANCE_WEIGHTS, no learners.
  * td        — TD(λ) only (publishes stance_weights).
  * full      — TD(λ) + options + Dyna-Q+ + Horde + order-flow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

import numpy as np

from mats.agents.risk_agent import DEFAULT_STANCE_WEIGHTS
from mats.connectors._common import Candle, OrderBook, OrderBookLevel, Receipt, Ticker
from mats.core.orchestrator import Orchestrator, OrchestratorConfig
from mats.core.scs import session_key


# Shared synthetic stub connectors that produce a drifting price series
# fully suitable for the technical, MC, KNN, and CTA agents.

class SyntheticCoinbase:
    name = "coinbase"

    def __init__(self, seed: int = 0, n_bars: int = 1024) -> None:
        rng = np.random.default_rng(seed)
        # Two regimes: trend then chop, then trend back.
        ret = np.concatenate([
            rng.normal(0.4, 0.6, n_bars // 3),
            rng.normal(0.0, 0.6, n_bars // 3),
            rng.normal(-0.3, 0.6, n_bars - 2 * (n_bars // 3)),
        ])
        self._prices = 100 + np.cumsum(ret)
        self._cursor = max(240, n_bars // 4)

    def step_forward(self) -> None:
        self._cursor = min(self._cursor + 1, len(self._prices))

    @property
    def cursor(self) -> int:
        return self._cursor

    async def ticker(self, symbol: str) -> Ticker:
        p = float(self._prices[self._cursor - 1])
        return Ticker(symbol=symbol, price=p, bid=p - 0.05, ask=p + 0.05)

    async def orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        p = float(self._prices[self._cursor - 1])
        return OrderBook(symbol=symbol,
                         bids=[OrderBookLevel(p - 0.1, 1.0)],
                         asks=[OrderBookLevel(p + 0.1, 1.0)])

    async def candles(self, symbol: str, granularity: str = "ONE_HOUR", limit: int = 200):
        end = self._cursor
        start = max(0, end - limit)
        return [
            Candle(ts=i, open=float(self._prices[i]),
                   high=float(self._prices[i] + 0.2),
                   low=float(self._prices[i] - 0.2),
                   close=float(self._prices[i]), volume=1.0)
            for i in range(start, end)
        ]

    async def balances(self):
        return {"USD": 10_000, "ETH": 1.0}

    async def place_order(self, order):
        return Receipt(venue=self.name, symbol=order.symbol, order_id="bt",
                       status="paper_filled", filled=order.size, avg_price=100.0,
                       paper=True)


class _NullChain:
    name = "base"
    CHAIN_ID = 8453
    address = "0xBacktest"
    reward_wallet = "0xBacktest"
    private_key = ""

    async def eth_balance(self, addr=None): return 0.0
    async def erc20_balance(self, t, h=None): return 0.0
    async def balances(self): return {"ETH": 0.0}
    async def swap(self, **kw):
        return Receipt(venue="base", symbol="bt", order_id="bt", status="paper",
                       filled=0.0, avg_price=0.0, paper=True, raw=kw)
    async def sweep_rewards(self, tokens): return []
    def _web3(self): return None


class _NullWasabi:
    name = "wasabi"

    def __init__(self, base: SyntheticCoinbase) -> None:
        self._base = base

    async def ticker(self, s):
        # Wasabi prints a small premium so the cross-venue arb agent has signal.
        p = float(self._base._prices[self._base._cursor - 1])
        return Ticker(s, p + 0.03, p - 0.02, p + 0.08)

    async def orderbook(self, s, depth=10):
        return OrderBook(s, [], [])

    async def open_position(self, **kw):
        return Receipt(venue="wasabi", symbol=kw["symbol"], order_id="bt",
                       status="paper_opened", filled=kw["size_usd"], avg_price=100.0,
                       paper=True)

    async def close_position(self, **kw):
        return Receipt(venue="wasabi", symbol="bt", order_id="bt",
                       status="paper_closed", filled=0.0, avg_price=0.0, paper=True)


class _NullPredictBase:
    name = "predictbase"
    async def list_markets(self, **kw): return []
    async def place_bet(self, **kw):
        return Receipt(venue="predictbase", symbol=kw["market_id"], order_id="bt",
                       status="paper_bet", filled=kw["size_usdc"], avg_price=0.5,
                       paper=True)
    async def redeem(self, mid):
        return Receipt(venue="predictbase", symbol=mid, order_id="bt",
                       status="paper_redeem", filled=0.0, avg_price=0.0, paper=True)


class _NullAllora:
    name = "allora"
    async def list_topics(self): return []
    async def submit_prediction(self, pred): return {"status": "paper_accepted"}
    async def rewards(self): return []
    async def claim_rewards(self, base_recipient): return []


class _NullVenice:
    name = "venice"
    async def price_quote(self, t, a): return {"rate": 1.0, "min_out": a}
    async def swap_to_diem(self, **kw):
        return Receipt(venue="venice", symbol="DIEM", order_id="bt",
                       status="paper", filled=0.0, avg_price=0.0, paper=True)
    async def diem_balance(self, holder=None): return 0.0


@dataclass
class ModeResult:
    name: str
    ticks: int
    mean_stance: float
    stance_std: float
    weight_drift: float


async def run_mode(name: str, ticks: int, *, seed: int, learning_enabled: bool,
                   enable_orderflow: bool, vault_path: str) -> ModeResult:
    coinbase = SyntheticCoinbase(seed=seed)
    cfg = OrchestratorConfig(
        universe=[("ETH-USD", "coinbase")],
        allora_topics=[],
        reward_tokens=[],
        tick_interval_sec=0.0,
        vault_path=vault_path,
        learning_enabled=learning_enabled,
        enable_orderflow_agent=enable_orderflow,
        crossvenue_pairs=[("ETH-USD", ["coinbase", "wasabi"])],
        horde_pairings={"ETH-USD": "BTC-USD"},
    )
    orch = Orchestrator(cfg)
    orch.connectors = {
        "coinbase": coinbase,
        "base": _NullChain(),
        "wasabi": _NullWasabi(coinbase),
        "predictbase": _NullPredictBase(),
        "allora": _NullAllora(),
        "venice": _NullVenice(),
    }
    stances: list[float] = []
    weight_history: list[dict] = []
    for _ in range(ticks):
        summary = await orch.tick()
        stances.append(float(summary["symbols"]["ETH-USD"]["stance"]))
        if learning_enabled:
            w = await orch.scs.get(session_key(cfg.session_id, "stance_weights"))
            if w:
                weight_history.append(dict(w))
        coinbase.step_forward()
    arr = np.asarray(stances)
    drift = 0.0
    if weight_history:
        # Drift = L2 distance of terminal weights from the cold-start defaults,
        # so we can tell at a glance whether the TD(λ) learner has moved.
        keys = list(weight_history[-1].keys())
        defaults = np.asarray([DEFAULT_STANCE_WEIGHTS.get(k, 0.0) for k in keys])
        last = np.asarray([weight_history[-1].get(k, 0.0) for k in keys])
        # Renormalise defaults to the same domain (sum to 1) so they're
        # comparable to the projected learned weights.
        if defaults.sum() > 0:
            defaults = defaults / defaults.sum()
        drift = float(np.linalg.norm(last - defaults))
    return ModeResult(
        name=name, ticks=ticks,
        mean_stance=float(arr.mean()) if arr.size else 0.0,
        stance_std=float(arr.std()) if arr.size else 0.0,
        weight_drift=drift,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("mats-backtest")
    p.add_argument("--ticks", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vault", type=str, default="./bt_vault")
    p.add_argument("--modes", type=str, default="baseline,td,full",
                   help="comma-separated subset of {baseline, td, full}")
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results: list[ModeResult] = []
    if "baseline" in modes:
        results.append(await run_mode(
            "baseline", args.ticks, seed=args.seed,
            learning_enabled=False, enable_orderflow=False,
            vault_path=args.vault,
        ))
    if "td" in modes:
        results.append(await run_mode(
            "td", args.ticks, seed=args.seed,
            learning_enabled=True, enable_orderflow=False,
            vault_path=args.vault,
        ))
    if "full" in modes:
        results.append(await run_mode(
            "full", args.ticks, seed=args.seed,
            learning_enabled=True, enable_orderflow=True,
            vault_path=args.vault,
        ))
    print(json.dumps([r.__dict__ for r in results], indent=2))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
