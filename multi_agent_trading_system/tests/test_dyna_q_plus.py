"""Dyna-Q+ planner tests."""

from __future__ import annotations

import numpy as np

from mats.learning.dyna_q_plus import DynaQPlus, DynaQPlusConfig
from mats.analysis.market_dynamics_model import MarketDynamicsModel, stance_bucket


def test_stance_bucket_monotonic():
    buckets = [stance_bucket(s) for s in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    assert buckets == sorted(buckets)
    assert buckets[0] == 0
    assert buckets[-1] == 4


def test_market_model_records_and_samples():
    rng = np.random.default_rng(0)
    model = MarketDynamicsModel()
    state = ("trend_up", 3)
    next_state = ("trend_up", 4)
    for _ in range(20):
        model.record(state, +1, 0.05, next_state)
    sampled_r, sampled_s = model.sample(state, +1, rng=rng)
    assert sampled_s == next_state
    # mean should be close to 0.05 since variance is essentially zero.
    assert abs(sampled_r - 0.05) < 1e-3
    assert model.visit_count(state, +1) == 20
    # Expected reward is exact under the running mean.
    assert abs(model.expected_reward(state, +1) - 0.05) < 1e-12


def test_dyna_q_plus_learns_positive_action_better():
    """Reward = +action; planner Q(state, +1) should exceed Q(state, -1)."""
    rng = np.random.default_rng(0)
    planner = DynaQPlus(DynaQPlusConfig(alpha=0.2, gamma=0.0, epsilon=0.0,
                                         plan_steps=10))
    state = ("trend_up", 2)
    for _ in range(60):
        # Cycle through actions to populate the model.
        for a in (-1, 0, +1):
            planner.step(state, a, reward=float(a) * 0.1, next_state=state, rng=rng)
    assert planner.q[(state, +1)] > planner.q[(state, -1)]
    assert planner.best_action(state) == +1


def test_novelty_bonus_increases_q_on_stale_pair():
    rng = np.random.default_rng(0)
    planner = DynaQPlus(DynaQPlusConfig(alpha=0.5, gamma=0.0, epsilon=0.0,
                                         kappa=1.0, plan_steps=0))
    state = ("chop", 2)
    # Real-step once to populate the model with a zero-reward pair.
    planner.step(state, +1, reward=0.0, next_state=state, rng=rng)
    q_before = planner.q[(state, +1)]
    # Advance many real steps that don't touch (state, +1).
    for _ in range(100):
        planner.step(("trend_up", 4), -1, reward=0.0, next_state=("trend_up", 4), rng=rng)
    # Force a planning round that will likely sample the stale pair.
    planner.cfg.plan_steps = 200
    planner._plan(rng)
    q_after = planner.q[(state, +1)]
    assert q_after > q_before


def test_choose_falls_back_to_argmax_when_epsilon_zero():
    rng = np.random.default_rng(0)
    planner = DynaQPlus(DynaQPlusConfig(epsilon=0.0))
    state = ("squeeze", 0)
    planner.q[(state, +1)] = 1.0
    planner.q[(state, 0)] = 0.0
    planner.q[(state, -1)] = -1.0
    assert planner.choose(state, rng=rng) == +1
