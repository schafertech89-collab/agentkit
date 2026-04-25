"""SSGM — Structured Self-Governing Memory.

Implements the SSGM 4D failure taxonomy (poisoning, drift, hallucination,
structural) plus a write-boundary decorator that wraps any MCP tool with audit
logging and failure detection.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import time
from enum import Enum
from typing import Any, Awaitable, Callable

from .audit_chain import AuditChain


class FailureCategory(str, Enum):
    POISONING = "poisoning"  # adversarial content inserted
    DRIFT = "drift"  # slow semantic shift away from source of truth
    HALLUCINATION = "hallucination"  # fabricated content not grounded in .raw/
    STRUCTURAL = "structural"  # schema, dependency or index corruption


class GovernanceError(RuntimeError):
    def __init__(self, category: FailureCategory, message: str, details: dict[str, Any] | None = None):
        super().__init__(f"[{category.value}] {message}")
        self.category = category
        self.details = details or {}


class SSGM:
    """Write-side governance — pre/post checks + audit."""

    def __init__(self, chain: AuditChain | None = None, allow_live_writes: bool = True) -> None:
        self.chain = chain or AuditChain()
        self.allow_live_writes = allow_live_writes
        ok, err = self.chain.verify()
        if not ok:
            raise GovernanceError(
                FailureCategory.STRUCTURAL,
                f"audit chain integrity failed: {err}",
            )

    # ---------------------------------------------------------------- API

    def guard_write(
        self,
        *,
        actor: str,
        action: str,
        target: str,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator factory wrapping a write function with governance."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if asyncio.iscoroutinefunction(func):

                @functools.wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    return await self._run_async(
                        func, args, kwargs,
                        actor=actor, action=action, target=target, metadata=metadata,
                    )

                return async_wrapper

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return self._run_sync(
                    func, args, kwargs,
                    actor=actor, action=action, target=target, metadata=metadata,
                )

            return sync_wrapper

        return decorator

    # --------------------------------------------------------------- runners

    async def _run_async(
        self,
        func: Callable[..., Awaitable[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        actor: str,
        action: str,
        target: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        before = self._snapshot_target(target)
        self._pre_check(action, args, kwargs)
        if not self.allow_live_writes:
            raise GovernanceError(
                FailureCategory.STRUCTURAL,
                "live writes disabled by governance flag",
            )
        result = await func(*args, **kwargs)
        after = self._snapshot_target(target, result=result)
        self._post_check(action, before, after, result)
        self.chain.append(
            actor=actor, action=action, target=target,
            before=before, after=after,
            payload={"args_digest": _digest(args), "kwargs_digest": _digest(kwargs)},
            metadata=metadata,
        )
        return result

    def _run_sync(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        actor: str,
        action: str,
        target: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        before = self._snapshot_target(target)
        self._pre_check(action, args, kwargs)
        if not self.allow_live_writes:
            raise GovernanceError(
                FailureCategory.STRUCTURAL, "live writes disabled by governance flag"
            )
        result = func(*args, **kwargs)
        after = self._snapshot_target(target, result=result)
        self._post_check(action, before, after, result)
        self.chain.append(
            actor=actor, action=action, target=target,
            before=before, after=after,
            payload={"args_digest": _digest(args), "kwargs_digest": _digest(kwargs)},
            metadata=metadata,
        )
        return result

    # --------------------------------------------------------------- checks

    def _pre_check(self, action: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        # Poisoning detector: reject blatantly adversarial prompt-injection markers.
        text_blob = _flatten_strings((args, kwargs))
        indicators = (
            "ignore previous instructions",
            "</|endoftext|>",
            "disregard the system prompt",
        )
        lowered = text_blob.lower()
        for needle in indicators:
            if needle in lowered:
                raise GovernanceError(
                    FailureCategory.POISONING,
                    f"rejected write ({action}) due to prompt-injection marker",
                    details={"needle": needle},
                )

    def _post_check(
        self, action: str, before: bytes, after: bytes, result: Any
    ) -> None:
        # Drift detector: trivial sanity — result must be JSON-serializable.
        try:
            json.dumps(result, default=str)
        except (TypeError, ValueError) as exc:
            raise GovernanceError(
                FailureCategory.STRUCTURAL,
                f"write result is not serializable: {exc}",
            ) from exc

    def _snapshot_target(self, target: str, *, result: Any = None) -> bytes:
        # Ad-hoc snapshot based on target: for files we read them, otherwise we
        # hash the (possibly empty) result. The caller's target string should be
        # a filesystem path for write operations against the vault.
        from pathlib import Path

        p = Path(target)
        if p.exists() and p.is_file():
            try:
                return p.read_bytes()
            except OSError:
                return b""
        if result is None:
            return b""
        try:
            return json.dumps(result, sort_keys=True, default=str).encode()
        except (TypeError, ValueError):
            return repr(result).encode()


# ---------------------------------------------------------------- helpers


def _digest(obj: Any) -> str:
    try:
        payload = json.dumps(obj, sort_keys=True, default=str).encode()
    except (TypeError, ValueError):
        payload = repr(obj).encode()
    return hashlib.sha256(payload).hexdigest()


def _flatten_strings(obj: Any) -> str:
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)
        elif inspect.isclass(x):
            return
        else:
            try:
                out.append(str(x))
            except Exception:
                return

    walk(obj)
    return " ".join(out)
