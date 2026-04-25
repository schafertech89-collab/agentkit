"""Venue connectors.

Every connector obeys the ``Connector`` protocol:

    async def ticker(symbol) -> Ticker
    async def orderbook(symbol, depth=10) -> OrderBook
    async def candles(symbol, granularity, limit) -> list[Candle]
    async def place_order(order: Order) -> Receipt
    async def balances() -> dict[str, float]

Paper mode is the default; live mode is gated by ``MATS_LIVE=1`` AND the
governance audit chain being consistent at startup.
"""

from .base_chain import BaseChainConnector
from .coinbase import CoinbaseConnector
from .predictbase import PredictBaseConnector
from .wasabi import WasabiConnector
from .allora import AlloraConnector
from .venice import VeniceConnector

__all__ = [
    "BaseChainConnector",
    "CoinbaseConnector",
    "PredictBaseConnector",
    "WasabiConnector",
    "AlloraConnector",
    "VeniceConnector",
]
