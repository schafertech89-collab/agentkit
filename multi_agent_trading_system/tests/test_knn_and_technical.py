import numpy as np

from mats.analysis.knn import KNNRegime, engineer_features
from mats.analysis.technical import TechnicalAnalysis


def test_engineer_features_shape():
    prices = [100.0 + i * 0.5 for i in range(60)]
    features = engineer_features(prices, volumes=[1.0] * 60)
    assert features.shape == (8,)
    assert np.isfinite(features).all()


def test_knn_cold_start_returns_regime():
    model = KNNRegime()
    prices = np.linspace(100, 120, 100).tolist()
    result = model.predict(prices, [1.0] * 100)
    assert "regime" in result and "confidence" in result
    assert result["regime"] in {"trend_up", "trend_down", "chop", "crisis", "squeeze"}


def test_technical_score_range():
    prices = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, size=100)) * 100
    ta = TechnicalAnalysis(prices=prices.tolist())
    score = ta.score()
    assert -1.0 <= score <= 1.0
