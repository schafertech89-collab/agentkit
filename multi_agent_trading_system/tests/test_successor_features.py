"""Successor-feature estimator tests."""

from __future__ import annotations

import numpy as np

from mats.learning.successor_features import (
    DEFAULT_COMPONENTS,
    SFConfig,
    SuccessorFeatureEstimator,
)


def test_psi_shape():
    sf = SuccessorFeatureEstimator(feature_dim=10)
    assert sf.psi.shape == (len(DEFAULT_COMPONENTS), 10)


def test_value_is_dot_product_of_weights_and_psi_phi():
    rng = np.random.default_rng(0)
    sf = SuccessorFeatureEstimator(feature_dim=4)
    sf.psi = rng.normal(size=sf.psi.shape)
    phi = rng.normal(size=4)
    rw = {c: float(rng.normal()) for c in DEFAULT_COMPONENTS}
    expected = float(
        np.array([rw[c] for c in DEFAULT_COMPONENTS]) @ sf.psi @ phi
    )
    assert abs(sf.value(phi, rw) - expected) < 1e-9


def test_reweighting_changes_value_without_relearn():
    """Re-weighting reward components should change the value but not the
    psi matrix (the whole point of successor features)."""
    rng = np.random.default_rng(1)
    sf = SuccessorFeatureEstimator(feature_dim=5, cfg=SFConfig(alpha=0.1, gamma=0.0))
    phi = rng.normal(size=5)
    components = {"pnl": 0.1, "drawdown": -0.05, "turnover_cost": -0.02, "slippage": -0.01}
    for _ in range(50):
        sf.update(phi, components, phi)
    psi_before = sf.psi.copy()
    v1 = sf.value(phi, {"pnl": 1.0})
    v2 = sf.value(phi, {"pnl": 1.0, "drawdown": 5.0})
    assert v1 != v2
    assert np.allclose(sf.psi, psi_before)


def test_td_update_reduces_error_on_stationary_target():
    sf = SuccessorFeatureEstimator(feature_dim=3, cfg=SFConfig(alpha=0.2, gamma=0.0))
    phi = np.array([1.0, 0.0, 0.0])
    components = {"pnl": 0.5, "drawdown": -0.1, "turnover_cost": 0.0, "slippage": 0.0}
    deltas = []
    for _ in range(60):
        d = sf.update(phi, components, None)
        deltas.append(np.linalg.norm(d))
    assert deltas[-1] < deltas[0]


def test_snapshot_roundtrip():
    sf = SuccessorFeatureEstimator(feature_dim=3)
    sf.psi = np.arange(sf.psi.size, dtype=float).reshape(sf.psi.shape)
    snap = sf.snapshot()
    other = SuccessorFeatureEstimator(feature_dim=3)
    other.load(snap)
    assert np.allclose(other.psi, sf.psi)
