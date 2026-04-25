import numpy as np

from mats.analysis.monte_carlo import MonteCarloResult
from mats.analysis.risk_reward import PositionSizer, RiskReward


def _dummy_mc(s0: float = 100.0, sigma_frac: float = 0.03) -> MonteCarloResult:
    sigma = s0 * sigma_frac
    samples = np.random.default_rng(0).normal(s0, sigma, 1000)
    return MonteCarloResult(
        samples=samples, mean=float(samples.mean()), stdev=float(samples.std()),
        median=float(np.median(samples)),
        p05=float(np.percentile(samples, 5)), p95=float(np.percentile(samples, 95)),
        var_95=float(np.percentile(s0 - samples, 95)),
        expected_shortfall_95=float(s0 * 0.04),
    )


def test_kelly_capped():
    rr = RiskReward(expected_return=0.05, win_prob=0.7, loss_if_wrong=0.03,
                    reward_if_right=0.05, sigma=0.02)
    assert 0.0 <= rr.kelly() <= 0.25


def test_sizer_produces_capped_order():
    rr = RiskReward(expected_return=0.05, win_prob=0.6, loss_if_wrong=0.03,
                    reward_if_right=0.06, sigma=0.02)
    sizer = PositionSizer(account_equity=10_000, max_leverage=3.0,
                          max_risk_per_trade=0.01)
    mc = _dummy_mc()
    order = sizer.size(venue="coinbase", symbol="ETH-USD", price=100.0,
                       signal=0.6, mc_result=mc, rr=rr)
    assert order.side == "long"
    assert order.size_usd >= 0
    assert order.leverage <= 3.0


def test_sizer_rejects_weak_signal():
    rr = RiskReward(expected_return=0.01, win_prob=0.5, loss_if_wrong=0.03,
                    reward_if_right=0.03, sigma=0.02)
    sizer = PositionSizer(account_equity=10_000, min_confidence=0.3)
    mc = _dummy_mc()
    order = sizer.size(venue="coinbase", symbol="ETH-USD", price=100.0,
                       signal=0.1, mc_result=mc, rr=rr)
    assert order.side == "flat"
    assert order.size_usd == 0.0
