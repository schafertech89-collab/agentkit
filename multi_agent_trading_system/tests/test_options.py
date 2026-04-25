"""Options framework tests."""

from __future__ import annotations

from mats.learning.options import (
    OptionState,
    OptionsSelector,
    default_macro_options,
)


def _state(**overrides):
    base = dict(regime="chop", fused_stance=0.0, realized_vol=0.01, z_score=0.0)
    base.update(overrides)
    return OptionState(**base)


def test_default_options_have_expected_names():
    names = [o.name for o in default_macro_options()]
    assert set(names) == {"de_risk", "trend_follow", "mean_revert", "stand_aside"}


def test_de_risk_initiates_in_crisis():
    sel = OptionsSelector()
    opt = sel.select(_state(regime="crisis", realized_vol=0.10))
    assert opt.name == "de_risk"
    # And produces a zero bias.
    assert opt.policy(_state(regime="crisis")) == 0.0


def test_trend_follow_initiates_in_trend_with_strong_stance():
    sel = OptionsSelector()
    opt = sel.select(_state(regime="trend_up", fused_stance=0.6))
    assert opt.name == "trend_follow"
    assert opt.policy(_state(regime="trend_up", fused_stance=0.6)) > 0


def test_mean_revert_initiates_on_high_z():
    sel = OptionsSelector()
    opt = sel.select(_state(regime="chop", z_score=2.5))
    assert opt.name == "mean_revert"
    # MR fights the move: positive z -> negative bias.
    assert opt.policy(_state(regime="chop", z_score=2.5)) < 0


def test_stand_aside_when_nothing_else_initiates():
    sel = OptionsSelector()
    opt = sel.select(_state(regime="chop", z_score=0.1))
    assert opt.name == "stand_aside"


def test_intra_option_td_only_updates_active_option():
    sel = OptionsSelector()
    s = _state(regime="trend_up", fused_stance=0.6)
    sel.select(s)
    assert sel.active is not None
    active_before = dict(sel.q)
    next_s = _state(regime="trend_up", fused_stance=0.6)
    sel.step(s, reward=0.5, next_state=next_s)
    # Only trend_follow Q should have moved.
    for name, q in sel.q.items():
        if name == "trend_follow":
            assert q != active_before[name]
        else:
            assert q == active_before[name]


def test_option_terminates_on_regime_flip():
    sel = OptionsSelector()
    s = _state(regime="trend_up", fused_stance=0.6)
    sel.select(s)
    # Flip regime.
    next_s = _state(regime="chop", fused_stance=0.0)
    sel.step(s, reward=0.0, next_state=next_s)
    assert sel.active is None
