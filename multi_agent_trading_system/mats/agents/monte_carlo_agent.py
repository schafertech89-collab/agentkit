"""Monte Carlo agent.

Runs GBM+jumps simulation against the most recent candles and exposes a
risk-aware stance to the planner. Also drops the full terminal distribution
into the SCS so the risk agent can size off the same paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis.monte_carlo import MonteCarloEngine
from .base import Agent, AgentContext, AgentOutput


class MonteCarloAgent(Agent):
    name = "monte_carlo"

    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        symbol = input_payload["symbol"]
        venue = input_payload.get("venue", "coinbase")
        horizon_days = float(input_payload.get("horizon_days", 1.0))
        target_return = float(input_payload.get("target_return", 0.01))
        n_paths = int(input_payload.get("n_paths", 5000))

        connector = ctx.get_connector(venue)
        candles = await connector.candles(symbol, granularity="ONE_HOUR", limit=240)
        if len(candles) < 5:
            cube = self.memcube(title=f"MC {symbol} insufficient data", importance=0.1)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0)

        prices = np.array([c.close for c in candles], dtype=float)
        log_rets = np.diff(np.log(np.clip(prices, 1e-12, None)))
        hourly_mu = float(log_rets.mean())
        hourly_sigma = float(log_rets.std())
        mu_annual = hourly_mu * 24 * 365
        sigma_annual = hourly_sigma * np.sqrt(24 * 365)

        engine = MonteCarloEngine()
        s0 = float(prices[-1])
        target_price = s0 * (1.0 + target_return)
        result = engine.simulate_gbm(
            s0=s0, mu=mu_annual, sigma=sigma_annual,
            horizon_days=horizon_days, n_paths=n_paths,
            steps_per_day=24, jump_intensity=5.0, jump_mean=0.0, jump_sigma=hourly_sigma * 3,
            target=target_price,
        )

        # Stance: probability of reaching target minus probability of breaching -target
        loss_price = s0 * (1.0 - target_return)
        prob_down = float((result.samples <= loss_price).mean())
        prob_up = result.prob_target or 0.0
        stance = float(np.clip((prob_up - prob_down) * 2, -1.0, 1.0))

        cube = self.memcube(
            title=f"MC {symbol} p_up={prob_up:.2%} p_down={prob_down:.2%}",
            body=(
                f"s0={s0:.2f} horizon={horizon_days}d mu={mu_annual:+.2%} "
                f"sigma={sigma_annual:.2%} ES95={result.expected_shortfall_95:.2f}"
            ),
            payload={
                "symbol": symbol, "venue": venue, "s0": s0,
                "horizon_days": horizon_days, "target_return": target_return,
                "mu_annual": mu_annual, "sigma_annual": sigma_annual,
                "prob_up": prob_up, "prob_down": prob_down,
                "var_95": result.var_95, "es_95": result.expected_shortfall_95,
                "mean_terminal": result.mean, "median_terminal": result.median,
            },
            importance=0.5,
            tags=[symbol, "montecarlo"],
        )

        await ctx.write(f"mc:{symbol}", {
            "mean": result.mean, "stdev": result.stdev,
            "p05": result.p05, "p95": result.p95,
            "var_95": result.var_95, "es_95": result.expected_shortfall_95,
            "prob_up": prob_up, "prob_down": prob_down, "stance": stance,
            "s0": s0, "target_return": target_return,
        })

        return AgentOutput(
            agent=self.name, memcube=cube, stance=stance,
            explain={"prob_up": prob_up, "prob_down": prob_down, "var_95": result.var_95},
        )
