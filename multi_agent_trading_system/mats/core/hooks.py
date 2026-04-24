"""ConduitHooks — the modular seam every tier inherits.

The hooks object is deliberately tiny. Every architecture layers new behaviour
by registering additional callbacks; none of them modify the base contract.

Available hooks:
    beforeSend(payload)       → payload   (mutate outgoing request)
    onStream(token)           → None
    onToolCall(tool, args, result) → None
    onMemCube(cube)           → None
    onAuditEntry(entry)       → None
    onError(exc)              → None
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from typing import Any, Awaitable, Callable

HookFn = Callable[..., Any | Awaitable[Any]]


class ConduitHooks:
    """Synchronous+async dispatch with result-chaining on `beforeSend`."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookFn]] = defaultdict(list)

    def on(self, event: str, fn: HookFn) -> HookFn:
        self._handlers[event].append(fn)
        return fn

    def off(self, event: str, fn: HookFn) -> None:
        if fn in self._handlers.get(event, []):
            self._handlers[event].remove(fn)

    async def emit(self, event: str, *args: Any, **kwargs: Any) -> list[Any]:
        results: list[Any] = []
        for fn in list(self._handlers.get(event, [])):
            try:
                value = fn(*args, **kwargs)
                if inspect.isawaitable(value):
                    value = await value
                results.append(value)
            except Exception as exc:  # surface but never crash the host loop
                results.append(exc)
                await self._best_effort(self._handlers.get("error", []), exc)
        return results

    async def chain(self, event: str, payload: Any) -> Any:
        """Run handlers sequentially where each mutates the payload."""
        for fn in list(self._handlers.get(event, [])):
            try:
                value = fn(payload)
                if inspect.isawaitable(value):
                    value = await value
                if value is not None:
                    payload = value
            except Exception as exc:
                await self._best_effort(self._handlers.get("error", []), exc)
        return payload

    @staticmethod
    async def _best_effort(handlers: list[HookFn], *args: Any) -> None:
        for fn in handlers:
            try:
                result = fn(*args)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                continue


# A global default registry so MCP servers can subscribe without needing
# a reference to the orchestrator. Agents that want isolation create their own.
DEFAULT_HOOKS = ConduitHooks()


def fire_and_forget(coro: Awaitable[Any]) -> None:
    """Helper for callers that want to dispatch without awaiting."""
    loop = asyncio.get_event_loop()
    loop.create_task(coro)
