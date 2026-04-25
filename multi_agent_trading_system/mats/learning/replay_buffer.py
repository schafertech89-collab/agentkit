"""HEMA-style age-weighted replay buffer.

Stores (observation, action, reward, next_observation) transitions with a
lightweight age-weighted priority. Used by the memory policy when
deciding which MemCubes to consolidate, promote, or decay.

Backend is in-memory with an optional periodic flush to a Redis Stream so
Tier-5 policy-mcp can replay across processes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass
class Transition:
    obs: list[float]
    action: int
    reward: float
    next_obs: list[float]
    done: bool = False
    ts: float = field(default_factory=time.time)
    priority: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    steps: list[Transition] = field(default_factory=list)

    def append(self, t: Transition) -> None:
        self.steps.append(t)

    def total_reward(self) -> float:
        return sum(s.reward for s in self.steps)


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, redis_client: Any | None = None,
                 stream_key: str | None = None) -> None:
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)
        self._lock = asyncio.Lock()
        self._redis = redis_client
        self._stream_key = stream_key or "mats:replay"

    async def add(self, t: Transition) -> None:
        async with self._lock:
            self._buf.append(t)
        if self._redis is not None:
            try:
                await self._redis.xadd(
                    self._stream_key,
                    {"payload": json.dumps(asdict(t), default=str)},
                    maxlen=self.capacity,
                    approximate=True,
                )
            except Exception:
                pass

    async def add_trajectory(self, traj: Trajectory) -> None:
        for t in traj.steps:
            await self.add(t)

    async def sample(self, k: int = 64, *, bias_recent: float = 0.5) -> list[Transition]:
        """Sample with age-weighted priority — newer transitions matter more."""
        import numpy as np
        async with self._lock:
            items = list(self._buf)
        if not items:
            return []
        n = len(items)
        ages = np.arange(n, 0, -1, dtype=float)
        weights = (1.0 / ages) ** bias_recent
        priorities = np.array([max(1e-6, t.priority) for t in items], dtype=float)
        probs = (weights * priorities)
        probs = probs / probs.sum()
        idx = np.random.choice(n, size=min(k, n), replace=False, p=probs)
        return [items[i] for i in idx]

    async def __aiter__(self):
        async with self._lock:
            items = list(self._buf)
        for t in items:
            yield t

    async def size(self) -> int:
        async with self._lock:
            return len(self._buf)
