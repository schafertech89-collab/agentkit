"""Horde of GVFs tests."""

from __future__ import annotations

from mats.learning.horde import HordeLearner, default_horde
from mats.learning.td_lambda import feature_dim, featurize


def test_default_horde_has_three_gvfs():
    gvfs = default_horde()
    names = {g.name for g in gvfs}
    assert names == {"intermarket_xcorr", "funding_next_reset", "dealer_gamma_proxy"}


def test_horde_predict_all_returns_per_gvf():
    learner = HordeLearner(default_horde(), feature_dim=feature_dim())
    phi = featurize({"technical": 0.1}, regime="trend_up", realized_vol=0.01,
                    trend_short=0.0, trend_long=0.0)
    preds = learner.predict_all(phi)
    assert set(preds) == {"intermarket_xcorr", "funding_next_reset", "dealer_gamma_proxy"}
    # Untrained predictions are zero.
    for v in preds.values():
        assert v == 0.0


def test_intermarket_xcorr_learns_paired_return():
    """If cumulant = paired_return is constant ~0.5, the GVF prediction
    should grow toward ~0.5 / (1 - gamma) with TD(0)."""
    learner = HordeLearner(default_horde(), feature_dim=feature_dim())
    phi = featurize({"technical": 0.0}, regime="chop", realized_vol=0.01,
                    trend_short=0.0, trend_long=0.0)
    ctx = {"paired_return": 0.5, "funding_rate": 0.0,
           "realized_vol": 0.01, "realized_vol_prev": 0.01, "z_score": 0.0}
    for _ in range(400):
        learner.observe(ctx, phi, phi)
    pred = learner.predict("intermarket_xcorr", phi)
    # gamma=0.5 -> steady state = 0.5 / (1 - 0.5) = 1.0; allow generous tolerance.
    assert pred > 0.5
    assert pred < 1.5


def test_horde_observe_returns_per_gvf_td_errors():
    learner = HordeLearner(default_horde(), feature_dim=feature_dim())
    phi = featurize({}, regime="chop", realized_vol=0.01,
                    trend_short=0.0, trend_long=0.0)
    ctx = {"paired_return": 0.1, "funding_rate": 0.0001,
           "realized_vol": 0.02, "realized_vol_prev": 0.01, "z_score": 1.0}
    errs = learner.observe(ctx, phi, phi)
    assert set(errs) == {"intermarket_xcorr", "funding_next_reset", "dealer_gamma_proxy"}
