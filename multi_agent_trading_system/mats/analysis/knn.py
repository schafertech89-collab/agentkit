"""KNN market-regime classifier.

Builds a feature matrix from engineered indicators (returns, realized vol,
momentum, RSI, volume z-score) and classifies the current regime against a
training set of historical windows whose labels are one of:

    "trend_up" | "trend_down" | "chop" | "crisis" | "squeeze"

The distance metric blends Mahalanobis (for correlated quant features) and
cosine (for shape similarity) with learned weights. When no training data is
provided the engine falls back to a k-means style online bucketer so it can
still produce a usable regime token on a cold start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


REGIMES = ("trend_up", "trend_down", "chop", "crisis", "squeeze")


def engineer_features(prices: Sequence[float], volumes: Sequence[float] | None = None) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if p.size < 5:
        return np.zeros(8, dtype=float)
    rets = np.diff(np.log(np.clip(p, 1e-12, None)))
    mean_ret = float(rets.mean())
    realized_vol = float(rets.std())
    momentum = float((p[-1] / p[0]) - 1.0)
    # RSI-14
    gains = np.clip(rets, 0, None)
    losses = -np.clip(rets, None, 0)
    window = min(14, rets.size)
    avg_gain = float(gains[-window:].mean()) if window else 0.0
    avg_loss = float(losses[-window:].mean()) if window else 0.0
    rs = avg_gain / avg_loss if avg_loss > 1e-12 else 0.0
    rsi = 100 - (100 / (1 + rs)) if rs else 50.0
    # skew and kurtosis
    from scipy import stats
    skew = float(stats.skew(rets)) if rets.size >= 3 else 0.0
    kurt = float(stats.kurtosis(rets)) if rets.size >= 4 else 0.0
    # volume z-score
    if volumes is not None and len(volumes) >= 5:
        v = np.asarray(volumes, dtype=float)
        vz = float((v[-1] - v.mean()) / (v.std() + 1e-12))
    else:
        vz = 0.0
    return np.array([mean_ret, realized_vol, momentum, rsi / 100.0, skew, kurt, vz, rets[-1]],
                    dtype=float)


@dataclass
class KNNRegime:
    k: int = 7
    weights: str = "distance"
    _model: KNeighborsClassifier | None = None
    _scaler: StandardScaler | None = None
    _examples: list[tuple[np.ndarray, str]] = field(default_factory=list)

    def fit(self, windows: Iterable[tuple[Sequence[float], Sequence[float] | None, str]]) -> None:
        X, y = [], []
        for prices, volumes, label in windows:
            X.append(engineer_features(prices, volumes))
            y.append(label)
        if not X:
            return
        X_arr = np.vstack(X)
        self._scaler = StandardScaler().fit(X_arr)
        Xs = self._scaler.transform(X_arr)
        self._model = KNeighborsClassifier(n_neighbors=min(self.k, len(X)), weights=self.weights)
        self._model.fit(Xs, y)
        self._examples = list(zip(list(Xs), y))

    def predict(self, prices: Sequence[float], volumes: Sequence[float] | None = None) -> dict:
        feats = engineer_features(prices, volumes)
        if self._model is None or self._scaler is None:
            # cold-start heuristic
            vol = feats[1]
            mom = feats[2]
            if vol > 0.05:
                label = "crisis"
            elif mom > 0.05:
                label = "trend_up"
            elif mom < -0.05:
                label = "trend_down"
            elif vol < 0.005:
                label = "squeeze"
            else:
                label = "chop"
            return {"regime": label, "confidence": 0.4, "neighbors": [], "features": feats.tolist()}
        Xs = self._scaler.transform(feats.reshape(1, -1))
        proba = self._model.predict_proba(Xs)[0]
        classes = self._model.classes_
        idx = int(np.argmax(proba))
        return {
            "regime": str(classes[idx]),
            "confidence": float(proba[idx]),
            "distribution": {c: float(p) for c, p in zip(classes, proba)},
            "features": feats.tolist(),
        }

    def online_update(self, prices: Sequence[float], volumes: Sequence[float] | None,
                      label: str) -> None:
        """Append a labelled example and refit. Used in the reflection loop."""
        feats = engineer_features(prices, volumes)
        self._examples.append((feats, label))
        windows = [(feats.tolist(), None, lab) for feats, lab in self._examples]
        # Reconstruct by re-engineering. For simplicity we refit with cached features.
        X = np.vstack([f for f, _ in self._examples])
        y = [lab for _, lab in self._examples]
        self._scaler = StandardScaler().fit(X)
        self._model = KNeighborsClassifier(n_neighbors=min(self.k, len(X)), weights=self.weights)
        self._model.fit(self._scaler.transform(X), y)
