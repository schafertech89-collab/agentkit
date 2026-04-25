"""TD(lambda) stance learner tests."""

from __future__ import annotations

import numpy as np

from mats.learning.td_lambda import (
    DEFAULT_AGENTS,
    REGIMES,
    TDLambdaConfig,
    TDLambdaStanceLearner,
    feature_dim,
    featurize,
    reward_from_outcome,
)


def test_featurize_dim_matches_feature_dim():
    phi = featurize(
        {"technical": 0.3, "knn": -0.1},
        regime="trend_up",
        realized_vol=0.02,
        trend_short=0.5,
        trend_long=0.1,
    )
    assert phi.shape == (feature_dim(),)
    # Bias slot is 1.
    assert phi[0] == 1.0
    # Regime one-hot: trend_up is the first regime.
    regime_offset = 1 + len(DEFAULT_AGENTS)
    assert phi[regime_offset] == 1.0
    # Other regimes are zero.
    for i in range(1, len(REGIMES)):
        assert phi[regime_offset + i] == 0.0


def test_eligibility_trace_decays():
    cfg = TDLambdaConfig(alpha=0.1, gamma=0.95, lam=0.5)
    learner = TDLambdaStanceLearner(cfg)
    phi = np.zeros(learner.dim)
    phi[1] = 1.0
    learner.update(phi, reward=0.0, phi_next=phi)
    # Trace should equal phi after first update.
    assert np.allclose(learner.e[1], 1.0)
    learner.update(phi, reward=0.0, phi_next=phi)
    # After second update, trace = gamma*lam*1 + 1 = 1.475.
    assert np.isclose(learner.e[1], cfg.gamma * cfg.lam * 1.0 + 1.0)


def test_learner_converges_on_toy_bandit():
    """Two stances; only the first correlates with reward. Learner should
    weight it higher."""
    rng = np.random.default_rng(0)
    learner = TDLambdaStanceLearner(TDLambdaConfig(alpha=0.05, gamma=0.0, lam=0.0))
    for _ in range(800):
        s_good = float(rng.normal(0.5, 0.2))
        s_bad = float(rng.normal(0.0, 0.2))
        stances = {"technical": s_good, "fundamental": s_bad}
        phi = featurize(stances, regime="trend_up", realized_vol=0.01,
                        trend_short=0.0, trend_long=0.0)
        reward = float(0.5 * s_good + rng.normal(0.0, 0.05))
        learner.update(phi, reward=reward, phi_next=None)
    weights = learner.stance_weights()
    assert weights["technical"] > weights["fundamental"]


def test_stance_weights_sum_to_one():
    learner = TDLambdaStanceLearner()
    weights = learner.stance_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_reward_from_outcome_subtracts_costs():
    base = reward_from_outcome(0.10, turnover_usd=0.0, inventory_usd=0.0)
    assert base == 0.10
    with_cost = reward_from_outcome(
        0.10, turnover_usd=10_000, inventory_usd=0.0,
        turnover_cost_bps=2.0,
    )
    # 2 bps * 10_000 = $2 cost subtracted from PnL.
    assert with_cost < base
    assert abs(with_cost - (0.10 - 2.0)) < 1e-9


def test_snapshot_roundtrip():
    learner = TDLambdaStanceLearner()
    phi = np.ones(learner.dim) * 0.1
    learner.update(phi, 0.5, phi)
    snap = learner.snapshot()
    other = TDLambdaStanceLearner()
    other.load(snap)
    assert np.allclose(other.w, learner.w)
    assert other.steps == learner.steps
