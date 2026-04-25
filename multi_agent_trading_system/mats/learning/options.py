"""Options framework (Sutton, Precup, Singh 1999).

An ``Option`` is a triple ``(I, pi, beta)``:

  * ``I``    — initiation set: the predicate ``initiates(state) -> bool``
  * ``pi``   — intra-option policy: maps state to a primitive bias
  * ``beta`` — termination function: ``terminates(state) -> bool`` (or a
    probability in [0, 1])

The ``OptionsSelector`` runs at the macro level: it picks an option when
none is currently active, re-uses the active option until it terminates,
and learns a per-option value via intra-option TD updates.

Macro options provided here:

  * ``trend_follow``  — initiates in trend regimes, terminates on regime
    flip or vol blow-up; bias = +/- aligned with stance sign.
  * ``mean_revert``   — initiates in chop / squeeze, terminates on z-cross.
  * ``de_risk``       — initiates on crisis or vol spike, drives bias to 0.
  * ``stand_aside``   — default; terminates when any other initiates.

Micro options are stubs (``work_passive``, ``cross_spread``, ``iceberg``,
``cancel``) — populated when the order-flow agent has live L2 data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class OptionState:
    regime: str
    fused_stance: float
    realized_vol: float
    z_score: float = 0.0


@dataclass
class Option:
    name: str
    initiates: Callable[[OptionState], bool]
    policy: Callable[[OptionState], float]   # primitive bias in [-1, 1]
    terminates: Callable[[OptionState], bool]
    is_micro: bool = False


# --------------------------------------------------------- macro options ----

def _trend_init(s: OptionState) -> bool:
    return s.regime in ("trend_up", "trend_down") and abs(s.fused_stance) > 0.1


def _trend_pi(s: OptionState) -> float:
    sign = 1.0 if s.fused_stance >= 0 else -1.0
    return float(np.clip(sign * (0.5 + 0.5 * abs(s.fused_stance)), -1.0, 1.0))


def _trend_term(s: OptionState) -> bool:
    return s.regime not in ("trend_up", "trend_down") or s.realized_vol > 0.05


def _mr_init(s: OptionState) -> bool:
    return s.regime in ("chop", "squeeze") and abs(s.z_score) > 1.0


def _mr_pi(s: OptionState) -> float:
    return float(np.clip(-np.sign(s.z_score) * min(1.0, abs(s.z_score) / 3.0), -1.0, 1.0))


def _mr_term(s: OptionState) -> bool:
    return abs(s.z_score) < 0.25 or s.regime in ("trend_up", "trend_down", "crisis")


def _derisk_init(s: OptionState) -> bool:
    return s.regime == "crisis" or s.realized_vol > 0.07


def _derisk_pi(s: OptionState) -> float:
    return 0.0


def _derisk_term(s: OptionState) -> bool:
    return s.regime != "crisis" and s.realized_vol < 0.04


def _stand_init(s: OptionState) -> bool:
    return True


def _stand_pi(s: OptionState) -> float:
    return float(np.clip(s.fused_stance * 0.25, -1.0, 1.0))


def _stand_term(s: OptionState) -> bool:
    # Yields whenever a more specific option fires.
    return _trend_init(s) or _mr_init(s) or _derisk_init(s)


def default_macro_options() -> list[Option]:
    return [
        Option("de_risk", _derisk_init, _derisk_pi, _derisk_term),
        Option("trend_follow", _trend_init, _trend_pi, _trend_term),
        Option("mean_revert", _mr_init, _mr_pi, _mr_term),
        Option("stand_aside", _stand_init, _stand_pi, _stand_term),
    ]


# --------------------------------------------------------- micro stubs ----

def default_micro_options() -> list[Option]:
    """Stubs gated until L2 data is wired."""

    def _never(_: OptionState) -> bool:
        return False

    def _hold(_: OptionState) -> float:
        return 0.0

    return [
        Option("work_passive", _never, _hold, _never, is_micro=True),
        Option("cross_spread", _never, _hold, _never, is_micro=True),
        Option("iceberg", _never, _hold, _never, is_micro=True),
        Option("cancel", _never, _hold, _never, is_micro=True),
    ]


# --------------------------------------------------------- selector -------

@dataclass
class OptionsSelectorConfig:
    alpha: float = 0.05
    gamma: float = 0.95


class OptionsSelector:
    """Picks an option, runs intra-option TD on the option-value function."""

    def __init__(
        self,
        options: Sequence[Option] | None = None,
        cfg: OptionsSelectorConfig | None = None,
    ) -> None:
        self.options = list(options or default_macro_options())
        self.cfg = cfg or OptionsSelectorConfig()
        self.q: dict[str, float] = {o.name: 0.0 for o in self.options}
        self.active: Option | None = None

    def _pick_initial(self, state: OptionState) -> Option:
        # Priority: de_risk > trend_follow > mean_revert > stand_aside (in declared order).
        for o in self.options:
            if not o.is_micro and o.initiates(state):
                return o
        return self.options[-1]

    def select(self, state: OptionState) -> Option:
        if self.active is not None and not self.active.terminates(state):
            return self.active
        self.active = self._pick_initial(state)
        return self.active

    def step(self, state: OptionState, reward: float, next_state: OptionState) -> dict:
        opt = self.select(state)
        # Intra-option TD update for the active option only.
        terminated = opt.terminates(next_state)
        bootstrap = 0.0 if terminated else max(self.q[o.name] for o in self.options)
        target = reward + self.cfg.gamma * bootstrap
        self.q[opt.name] += self.cfg.alpha * (target - self.q[opt.name])
        if terminated:
            self.active = None
        return {
            "active": opt.name,
            "terminated": terminated,
            "primitive_bias": opt.policy(state),
            "q": dict(self.q),
        }

    def primitive_bias(self, state: OptionState) -> float:
        return self.select(state).policy(state)


__all__ = [
    "Option",
    "OptionState",
    "OptionsSelector",
    "OptionsSelectorConfig",
    "default_macro_options",
    "default_micro_options",
]
