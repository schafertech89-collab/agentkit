"""Quantitative engine.

Factor-model flavoured building blocks used by the quant agent:

    * realized covariance
    * Ledoit-Wolf shrinkage
    * OLS factor regression
    * Sharpe / Sortino / Calmar
    * Black-Litterman blend
    * mean-variance optimal weights (closed form, no short-sale constraint)

These are deliberately lightweight (no dependence on cvxpy) so they work in
constrained container environments. When the optimization problem has
inequality constraints we fall back to projected-gradient descent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats


def log_returns(prices: np.ndarray | Sequence[float]) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(np.clip(p, 1e-12, None)))


def realized_covariance(price_matrix: np.ndarray) -> np.ndarray:
    """columns = assets, rows = observations."""
    rets = np.diff(np.log(np.clip(price_matrix, 1e-12, None)), axis=0)
    return np.cov(rets, rowvar=False)


def ledoit_wolf_shrinkage(cov: np.ndarray, target: np.ndarray | None = None,
                          shrinkage: float = 0.1) -> np.ndarray:
    """Shrink toward a diagonal target (default: trace mean on diagonal)."""
    if target is None:
        target = np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    return (1 - shrinkage) * cov + shrinkage * target


def ols_factor_regression(y: np.ndarray, X: np.ndarray) -> dict:
    """Standard OLS with HAC-less standard errors."""
    X = np.column_stack([np.ones(len(X)), X]) if X.ndim == 1 else np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = resid @ resid / max(1, len(y) - X.shape[1])
    cov_beta = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    t_stats = beta / np.where(se > 0, se, np.nan)
    return {
        "beta": beta.tolist(),
        "se": se.tolist(),
        "t_stats": t_stats.tolist(),
        "r_squared": float(1 - sigma2 * len(y) / (y.var() * len(y) + 1e-12)),
    }


def sharpe(returns: Sequence[float], rf: float = 0.0, periods_per_year: int = 365) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0 or r.std() == 0:
        return 0.0
    excess = r - rf / periods_per_year
    return float((excess.mean() * periods_per_year) / (r.std() * np.sqrt(periods_per_year)))


def sortino(returns: Sequence[float], rf: float = 0.0, periods_per_year: int = 365) -> float:
    r = np.asarray(returns, dtype=float)
    down = r[r < 0]
    if down.size == 0 or down.std() == 0:
        return 0.0
    excess = r - rf / periods_per_year
    return float((excess.mean() * periods_per_year) / (down.std() * np.sqrt(periods_per_year)))


def max_drawdown(returns: Sequence[float]) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return float(dd.min())


def calmar(returns: Sequence[float], periods_per_year: int = 365) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    ann_ret = (1 + r.mean()) ** periods_per_year - 1
    mdd = abs(max_drawdown(r))
    return float(ann_ret / mdd) if mdd > 1e-9 else 0.0


def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 3.0) -> np.ndarray:
    """Closed-form unconstrained MVO with Tikhonov-regularised inverse."""
    reg = cov + np.eye(cov.shape[0]) * 1e-6
    w = np.linalg.solve(reg, mu) / risk_aversion
    # project onto simplex (long-only with full allocation)
    return _simplex_projection(w)


def _simplex_projection(v: np.ndarray) -> np.ndarray:
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        return np.full(n, 1.0 / n)
    rho_max = rho[-1]
    theta = cssv[rho_max] / (rho_max + 1)
    return np.clip(v - theta, 0.0, None)


def black_litterman(
    mu_prior: np.ndarray,
    cov: np.ndarray,
    view_matrix: np.ndarray,
    view_returns: np.ndarray,
    view_uncertainty: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Return posterior expected returns blending priors with subjective views."""
    pi = mu_prior
    P = view_matrix
    Q = view_returns
    Omega = view_uncertainty
    M_inv = np.linalg.inv(tau * cov)
    left = M_inv + P.T @ np.linalg.solve(Omega, P)
    right = M_inv @ pi + P.T @ np.linalg.solve(Omega, Q)
    return np.linalg.solve(left, right)


@dataclass
class QuantitativeEngine:
    """Facade grouping the most-used helpers for agent-side code."""

    def summary(self, prices: Sequence[float]) -> dict:
        r = log_returns(prices)
        return {
            "mean_daily_return": float(r.mean()) if r.size else 0.0,
            "realized_vol": float(r.std()) if r.size else 0.0,
            "skew": float(stats.skew(r)) if r.size >= 3 else 0.0,
            "kurtosis": float(stats.kurtosis(r)) if r.size >= 4 else 0.0,
            "sharpe": sharpe(r),
            "sortino": sortino(r),
            "max_drawdown": max_drawdown(r),
            "calmar": calmar(r),
        }

    def optimal_weights(self, prices_matrix: np.ndarray, mu: np.ndarray | None = None,
                        risk_aversion: float = 3.0) -> np.ndarray:
        cov = ledoit_wolf_shrinkage(realized_covariance(prices_matrix))
        if mu is None:
            rets = np.diff(np.log(np.clip(prices_matrix, 1e-12, None)), axis=0)
            mu = rets.mean(axis=0) * 365
        return mean_variance_weights(mu, cov * 365, risk_aversion)
