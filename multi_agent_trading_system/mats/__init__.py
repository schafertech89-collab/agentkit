"""Multi-Agent Trading System.

MATS is a provider-agnostic, MCP-first trading stack that coordinates specialized
analysis agents (Monte Carlo, KNN, quant/technical/fundamental, reflection) over
four venues (Coinbase, PredictBase, Wasabi on Base, Base DEXes) and publishes its
predictions to Allora. Rewards are swept to a Base wallet and exchanged for
Venice.ai DIEM.
"""

from __future__ import annotations

__version__ = "0.1.0"
