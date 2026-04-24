"""Technical analysis — classic indicators, pattern flags, signal scoring.

Every indicator returns either a raw numeric series or a scalar "stance" in
[-1, 1] where negative = bearish, positive = bullish, zero = neutral. A
``score()`` method composes them into a single scalar used by the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _np(series: Sequence[float]) -> np.ndarray:
    return np.asarray(series, dtype=float)


def sma(series: Sequence[float], window: int) -> np.ndarray:
    a = _np(series)
    if window <= 0 or a.size < window:
        return np.full_like(a, np.nan, dtype=float)
    c = np.cumsum(np.insert(a, 0, 0.0))
    out = (c[window:] - c[:-window]) / window
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, out])


def ema(series: Sequence[float], window: int) -> np.ndarray:
    a = _np(series)
    if a.size == 0:
        return a
    alpha = 2.0 / (window + 1.0)
    out = np.empty_like(a)
    out[0] = a[0]
    for i in range(1, a.size):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out


def rsi(series: Sequence[float], window: int = 14) -> float:
    a = _np(series)
    if a.size < window + 1:
        return 50.0
    deltas = np.diff(a[-window - 1 :])
    gains = np.clip(deltas, 0, None).mean()
    losses = -np.clip(deltas, None, 0).mean()
    if losses <= 1e-12:
        return 100.0
    rs = gains / losses
    return float(100 - 100 / (1 + rs))


def macd(series: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    fast_e = ema(series, fast)
    slow_e = ema(series, slow)
    line = fast_e - slow_e
    sig = ema(line, signal)
    hist = line - sig
    return {"macd": float(line[-1]), "signal": float(sig[-1]), "hist": float(hist[-1])}


def bollinger(series: Sequence[float], window: int = 20, k: float = 2.0) -> dict:
    a = _np(series)
    if a.size < window:
        last = float(a[-1]) if a.size else 0.0
        return {"upper": last, "lower": last, "mid": last, "width": 0.0, "z": 0.0}
    m = float(a[-window:].mean())
    s = float(a[-window:].std())
    upper = m + k * s
    lower = m - k * s
    z = (float(a[-1]) - m) / (s + 1e-12)
    return {"upper": upper, "lower": lower, "mid": m, "width": upper - lower, "z": z}


def atr(high: Sequence[float], low: Sequence[float], close: Sequence[float], window: int = 14) -> float:
    h = _np(high); l = _np(low); c = _np(close)
    if min(h.size, l.size, c.size) < window + 1:
        return 0.0
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    return float(tr[-window:].mean())


def stance_from_sma_cross(prices: Sequence[float], fast: int = 20, slow: int = 50) -> float:
    s_fast = sma(prices, fast)[-1]
    s_slow = sma(prices, slow)[-1]
    if np.isnan(s_fast) or np.isnan(s_slow):
        return 0.0
    if s_fast > s_slow * 1.002:
        return 1.0
    if s_fast < s_slow * 0.998:
        return -1.0
    return 0.0


@dataclass
class TechnicalAnalysis:
    prices: Sequence[float]
    highs: Sequence[float] | None = None
    lows: Sequence[float] | None = None
    volumes: Sequence[float] | None = None

    def indicators(self) -> dict:
        m = macd(self.prices)
        bb = bollinger(self.prices)
        out = {
            "rsi_14": rsi(self.prices, 14),
            "macd": m,
            "bollinger": bb,
            "sma_cross_20_50": stance_from_sma_cross(self.prices, 20, 50),
            "sma_20": float(sma(self.prices, 20)[-1]) if len(self.prices) >= 20 else None,
            "sma_50": float(sma(self.prices, 50)[-1]) if len(self.prices) >= 50 else None,
        }
        if self.highs is not None and self.lows is not None:
            out["atr_14"] = atr(self.highs, self.lows, self.prices, 14)
        return out

    def score(self) -> float:
        """Compose indicators into a [-1, 1] bullish/bearish stance."""
        ind = self.indicators()
        r = ind["rsi_14"]
        # RSI stance: 30-70 neutral band, <30 oversold (+), >70 overbought (-)
        rsi_stance = 0.0
        if r < 30:
            rsi_stance = (30 - r) / 30.0
        elif r > 70:
            rsi_stance = -(r - 70) / 30.0
        macd_stance = np.tanh(ind["macd"]["hist"] * 5.0)
        bb_stance = -np.tanh(ind["bollinger"]["z"] / 2.0)  # reversion
        sma_stance = ind["sma_cross_20_50"]
        parts = [rsi_stance, macd_stance, bb_stance, sma_stance]
        return float(np.mean(parts))
