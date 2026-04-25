"""Monte Carlo reasoning.

Two modes:
  1. **Price-path simulation** — Geometric Brownian Motion with optional jumps
     (Merton), returning terminal distribution and VaR/ES.
  2. **Decision-tree rollouts** — generic rollouts over an action tree where
     each leaf returns a utility; used by the trading planner when evaluating
     compound multi-venue trade constructions.

Both modes expose a uniform ``MonteCarloResult`` with the summary statistics a
risk engine needs: mean, stdev, median, percentiles, VaR, expected shortfall,
and the probability of reaching a target threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


@dataclass
class MonteCarloResult:
    samples: np.ndarray
    mean: float
    stdev: float
    median: float
    p05: float
    p95: float
    var_95: float
    expected_shortfall_95: float
    prob_target: float | None = None
    paths: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)


class MonteCarloEngine:
    """Monte Carlo simulator for price paths and decision-tree rollouts."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------ price simulation

    def simulate_gbm(
        self,
        s0: float,
        mu: float,
        sigma: float,
        horizon_days: float,
        n_paths: int = 10_000,
        steps_per_day: int = 24,
        jump_intensity: float = 0.0,
        jump_mean: float = 0.0,
        jump_sigma: float = 0.0,
        target: float | None = None,
        return_paths: bool = False,
    ) -> MonteCarloResult:
        """Geometric Brownian Motion with optional Merton jumps."""
        dt = 1.0 / (365.0 * steps_per_day)
        steps = max(1, int(horizon_days * steps_per_day))
        # drift-adjusted for lognormal expectation
        drift = (mu - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        z = self.rng.standard_normal((n_paths, steps))
        increments = drift + vol * z

        if jump_intensity > 0.0:
            jumps = self.rng.poisson(jump_intensity * dt, size=(n_paths, steps))
            # Merton: each jump lognormal with (jump_mean, jump_sigma)
            jump_component = jumps * self.rng.normal(jump_mean, jump_sigma, size=(n_paths, steps))
            increments = increments + jump_component

        log_paths = np.cumsum(increments, axis=1)
        paths = s0 * np.exp(log_paths)
        terminal = paths[:, -1]
        return self._summarize(terminal, s0=s0, target=target, paths=paths if return_paths else None,
                               metadata={"mode": "gbm", "steps": steps, "n_paths": n_paths})

    # -------------------------------------------- portfolio returns (multi-asset)

    def simulate_portfolio(
        self,
        weights: Sequence[float],
        mu: Sequence[float],
        cov: Sequence[Sequence[float]],
        horizon_days: float,
        n_paths: int = 10_000,
    ) -> MonteCarloResult:
        w = np.asarray(weights, dtype=float)
        mu_arr = np.asarray(mu, dtype=float)
        cov_arr = np.asarray(cov, dtype=float)
        dt = horizon_days / 365.0
        mean_return = w @ (mu_arr * dt)
        port_vol = float(np.sqrt(w @ cov_arr @ w * dt))
        samples = self.rng.normal(mean_return, port_vol, size=n_paths)
        return self._summarize(samples, s0=0.0, metadata={"mode": "portfolio"})

    # ----------------------------------------------- decision-tree rollouts

    def rollout_tree(
        self,
        root_state: dict,
        branching: Callable[[dict, np.random.Generator], list[tuple[dict, float]]],
        leaf_utility: Callable[[dict, np.random.Generator], float],
        depth: int = 4,
        n_rollouts: int = 2048,
        discount: float = 0.99,
    ) -> MonteCarloResult:
        """Stochastic rollouts over an arbitrary MDP-shaped tree.

        ``branching`` returns (next_state, log_probability) tuples.
        ``leaf_utility`` returns a scalar reward when depth is exhausted.
        """
        utilities = np.empty(n_rollouts, dtype=float)
        for i in range(n_rollouts):
            state = dict(root_state)
            cumulative = 0.0
            for d in range(depth):
                choices = branching(state, self.rng)
                if not choices:
                    break
                probs = np.array([np.exp(p) for _, p in choices], dtype=float)
                probs = probs / probs.sum()
                idx = int(self.rng.choice(len(choices), p=probs))
                state, _ = choices[idx]
                state = dict(state)
                state.setdefault("depth", 0)
                state["depth"] = d + 1
            cumulative += (discount ** state.get("depth", depth)) * leaf_utility(state, self.rng)
            utilities[i] = cumulative
        return self._summarize(utilities, s0=0.0, metadata={"mode": "rollout"})

    # ---------------------------------------------------------------- core

    @staticmethod
    def _summarize(
        samples: np.ndarray,
        *,
        s0: float = 0.0,
        target: float | None = None,
        paths: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> MonteCarloResult:
        samples = np.asarray(samples, dtype=float).ravel()
        mean = float(samples.mean())
        stdev = float(samples.std())
        median = float(np.median(samples))
        p05 = float(np.percentile(samples, 5))
        p95 = float(np.percentile(samples, 95))
        losses = -(samples - s0) if s0 else -samples
        var_95 = float(np.percentile(losses, 95))
        tail = losses[losses >= var_95]
        es_95 = float(tail.mean()) if tail.size else var_95
        prob_target = None
        if target is not None:
            prob_target = float((samples >= target).mean())
        return MonteCarloResult(
            samples=samples,
            mean=mean,
            stdev=stdev,
            median=median,
            p05=p05,
            p95=p95,
            var_95=var_95,
            expected_shortfall_95=es_95,
            prob_target=prob_target,
            paths=paths,
            metadata=metadata or {},
        )
