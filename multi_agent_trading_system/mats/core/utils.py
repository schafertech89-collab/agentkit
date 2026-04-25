"""Shared utilities."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import random
from typing import Any, Awaitable, Callable, TypeVar


log = logging.getLogger(__name__)

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 4,
    base: float = 0.5,
    cap: float = 30.0,
    jitter: bool = True,
    on_error: Callable[[Exception], None] | None = None,
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:
            attempt += 1
            if on_error:
                on_error(exc)
            if attempt > retries:
                raise
            delay = min(cap, base * (2 ** (attempt - 1)))
            if jitter:
                delay *= 0.5 + random.random()
            log.warning("retry %d after %.2fs: %s", attempt, delay, exc)
            await asyncio.sleep(delay)


def require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"missing required env vars: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def coalesce(*values: Any) -> Any:
    for v in values:
        if v is not None and v != "":
            return v
    return None


def snake(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()
