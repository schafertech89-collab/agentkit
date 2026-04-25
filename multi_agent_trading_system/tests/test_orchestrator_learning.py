"""End-to-end paper-mode orchestrator tick with the Sutton learners enabled."""

from __future__ import annotations

import pytest

from mats.agents.risk_agent import DEFAULT_STANCE_WEIGHTS
from mats.core.orchestrator import Orchestrator, OrchestratorConfig
from mats.core.scs import session_key
from tests.test_orchestrator_paper import (
    StubAllora,
    StubChain,
    StubCoinbase,
    StubPredictBase,
    StubVenice,
    StubWasabi,
)


@pytest.mark.asyncio
async def test_paper_tick_with_learners_enabled(tmp_path):
    cfg = OrchestratorConfig(
        universe=[("ETH-USD", "coinbase")],
        allora_topics=[(1, "ETH-USD")],
        reward_tokens=[],
        tick_interval_sec=0.01,
        vault_path=str(tmp_path / "vault"),
        learning_enabled=True,
        enable_orderflow_agent=True,
        horde_pairings={"ETH-USD": "BTC-USD"},
        crossvenue_pairs=[("ETH-USD", ["coinbase", "wasabi"])],
    )
    orch = Orchestrator(cfg)
    orch.connectors = {
        "coinbase": StubCoinbase(),
        "base": StubChain(),
        "wasabi": StubWasabi(),
        "predictbase": StubPredictBase(),
        "allora": StubAllora(),
        "venice": StubVenice(),
    }
    summary = await orch.tick()
    assert "ETH-USD" in summary["symbols"]

    # The TD(λ) learner should have published stance_weights to SCS.
    weights = await orch.scs.get(session_key("default", "stance_weights"))
    assert weights is not None
    # All Sutton-style agent slots are present in the projected weights.
    assert set(weights.keys()) >= set(DEFAULT_STANCE_WEIGHTS.keys())

    # Options selector should have written an active_option.
    active = await orch.scs.get(session_key("default", "active_option"))
    assert active in {"de_risk", "trend_follow", "mean_revert", "stand_aside"}

    # GVF predictions should be present.
    gvf = await orch.scs.get(session_key("default", "gvf:ETH-USD"))
    assert gvf is not None and "predictions" in gvf

    # Planner state.
    pq = await orch.scs.get(session_key("default", "planner_q:ETH-USD"))
    assert pq is not None and "best_action" in pq


@pytest.mark.asyncio
async def test_paper_tick_with_learners_disabled(tmp_path):
    cfg = OrchestratorConfig(
        universe=[("ETH-USD", "coinbase")],
        allora_topics=[(1, "ETH-USD")],
        reward_tokens=[],
        tick_interval_sec=0.01,
        vault_path=str(tmp_path / "vault"),
        learning_enabled=False,
        enable_orderflow_agent=False,
    )
    orch = Orchestrator(cfg)
    orch.connectors = {
        "coinbase": StubCoinbase(),
        "base": StubChain(),
        "wasabi": StubWasabi(),
        "predictbase": StubPredictBase(),
        "allora": StubAllora(),
        "venice": StubVenice(),
    }
    summary = await orch.tick()
    assert "ETH-USD" in summary["symbols"]
    # No learned weights when learning is disabled.
    weights = await orch.scs.get(session_key("default", "stance_weights"))
    assert weights is None
    # Risk agent falls back to defaults.
    assert -1.0 <= summary["symbols"]["ETH-USD"]["stance"] <= 1.0
