import numpy as np

from mats.analysis.monte_carlo import MonteCarloEngine


def test_gbm_summary_has_expected_fields():
    rng = np.random.default_rng(42)
    engine = MonteCarloEngine(rng=rng)
    result = engine.simulate_gbm(s0=100.0, mu=0.1, sigma=0.4, horizon_days=1.0, n_paths=1000,
                                 target=101.0)
    assert result.samples.shape == (1000,)
    assert result.mean > 0
    assert result.p05 < result.p95
    assert 0.0 <= (result.prob_target or 0.0) <= 1.0


def test_rollout_returns_distribution():
    engine = MonteCarloEngine()

    def branch(state, rng):
        return [(dict(state, value=state.get("value", 0) + rng.integers(-2, 3)), 0.0)]

    def leaf(state, rng):
        return float(state.get("value", 0))

    result = engine.rollout_tree({"value": 0}, branch, leaf, depth=5, n_rollouts=256)
    assert result.samples.shape == (256,)
    assert result.stdev > 0
