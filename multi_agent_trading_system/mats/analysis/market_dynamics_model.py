"""A small learned model of next-step market dynamics.

Given a coarse state ``s = (regime, stance_bucket)`` and an action
``a in {-1, 0, +1}`` (short / flat / long), predict the next reward and the
next state. Used by the Dyna-Q+ planner for sampled rollouts.

The model keeps running mean / variance estimates per ``(s, a)`` cell. It is
intentionally simple — Sutton's Dyna chapter shows that even a tabular
forward model dramatically improves planning, and the goal here is to
provide *some* model, not the best possible one. A torch upgrade path is
available via ``model_kind="mlp"`` once enough data has accumulated.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Tuple

import numpy as np


State = Tuple[str, int]  # (regime, stance_bucket)
Action = int  # -1, 0, +1


def stance_bucket(stance: float, n_buckets: int = 5) -> int:
    """Bucket a [-1, 1] stance into a small integer index."""
    if not math.isfinite(stance):
        return n_buckets // 2
    s = max(-1.0, min(1.0, float(stance)))
    half = n_buckets // 2
    return int(round(s * half)) + half


@dataclass
class _CellStats:
    n: int = 0
    mean_reward: float = 0.0
    m2_reward: float = 0.0  # for online variance via Welford
    next_state_counts: dict[State, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, reward: float, next_state: State) -> None:
        self.n += 1
        delta = reward - self.mean_reward
        self.mean_reward += delta / self.n
        self.m2_reward += delta * (reward - self.mean_reward)
        self.next_state_counts[next_state] += 1

    @property
    def reward_var(self) -> float:
        return self.m2_reward / max(1, self.n - 1)


class MarketDynamicsModel:
    def __init__(self, n_buckets: int = 5) -> None:
        self.n_buckets = n_buckets
        self._cells: dict[Tuple[State, Action], _CellStats] = defaultdict(_CellStats)
        self._global_mean: float = 0.0
        self._global_n: int = 0

    # ----------------------------------------------------- record / sample
    def record(self, state: State, action: Action, reward: float, next_state: State) -> None:
        cell = self._cells[(state, action)]
        cell.update(reward, next_state)
        self._global_n += 1
        self._global_mean += (reward - self._global_mean) / self._global_n

    def known_pairs(self) -> Iterable[Tuple[State, Action]]:
        return tuple(self._cells.keys())

    def sample(self, state: State, action: Action, rng: np.random.Generator | None = None) -> tuple[float, State]:
        rng = rng or np.random.default_rng()
        cell = self._cells.get((state, action))
        if cell is None or cell.n == 0:
            # Unknown cell: best-effort prior.
            return self._global_mean, state
        sigma = math.sqrt(max(1e-9, cell.reward_var))
        reward = float(rng.normal(cell.mean_reward, sigma))
        # Sample next state by empirical frequency.
        nss = list(cell.next_state_counts.items())
        states, counts = zip(*nss)
        probs = np.asarray(counts, dtype=float)
        probs = probs / probs.sum()
        idx = rng.choice(len(states), p=probs)
        return reward, states[idx]

    def expected_reward(self, state: State, action: Action) -> float:
        cell = self._cells.get((state, action))
        if cell is None or cell.n == 0:
            return self._global_mean
        return cell.mean_reward

    def visit_count(self, state: State, action: Action) -> int:
        cell = self._cells.get((state, action))
        return cell.n if cell else 0

    # --------------------------------------------------------- snapshots
    def snapshot(self) -> dict:
        return {
            "n_buckets": self.n_buckets,
            "global_mean": self._global_mean,
            "global_n": self._global_n,
            "cells": [
                {
                    "state": list(state),
                    "action": int(action),
                    "n": cell.n,
                    "mean_reward": cell.mean_reward,
                    "var_reward": cell.reward_var,
                    "next_states": [
                        {"state": list(ns), "count": c}
                        for ns, c in cell.next_state_counts.items()
                    ],
                }
                for (state, action), cell in self._cells.items()
            ],
        }
