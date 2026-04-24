"""Shared Context Store — CA-MCP blackboard.

A CA-MCP-shaped Shared Context Store implemented with a pluggable backend:

  * InMemorySCS     — for tests and Tier 1
  * SQLiteSCS       — durable, file-based (Tier 1/2 fallback)
  * RedisSCS        — the canonical Tier 3+ backend (Redis 7.4 ACL'd)

All three satisfy the same async protocol. The key schema is:

    scs:{session_id}:goal                 — string
    scs:{session_id}:constraints          — JSON array
    scs:{session_id}:scratch:{step}       — JSON blob per agent step
    scs:{session_id}:events               — pub/sub channel for cross-server
    scs:{session_id}:memcube:{cube_id}    — MemCube snapshots for policy replay

Every MCP tool in MATS reads and writes the SCS directly so that agents
collaborate through the store rather than through the LLM.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import suppress
from typing import Any, AsyncIterator, Protocol


class SCS(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...
    async def append(self, key: str, item: Any) -> None: ...
    async def publish(self, channel: str, event: dict[str, Any]) -> None: ...
    def subscribe(self, channel: str) -> AsyncIterator[dict[str, Any]]: ...
    async def keys(self, pattern: str) -> list[str]: ...
    async def delete(self, key: str) -> None: ...
    async def close(self) -> None: ...


# ---------------------------------------------------------------- in-memory


class InMemorySCS:
    """Process-local SCS. Safe for tests and a single orchestrator."""

    def __init__(self) -> None:
        self._kv: dict[str, tuple[Any, float | None]] = {}
        self._queues: dict[str, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key not in self._kv:
                return None
            value, expires = self._kv[key]
            if expires is not None and time.time() > expires:
                self._kv.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        async with self._lock:
            expires = time.time() + ttl if ttl else None
            self._kv[key] = (value, expires)

    async def append(self, key: str, item: Any) -> None:
        async with self._lock:
            current = self._kv.get(key, ([], None))[0]
            if not isinstance(current, list):
                current = []
            current.append(item)
            self._kv[key] = (current, None)

    async def publish(self, channel: str, event: dict[str, Any]) -> None:
        for q in list(self._queues.get(channel, [])):
            with suppress(asyncio.QueueFull):
                q.put_nowait(event)

    async def subscribe(self, channel: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self._queues.setdefault(channel, []).append(queue)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self._queues.get(channel, []).remove(queue)

    async def keys(self, pattern: str) -> list[str]:
        import fnmatch

        async with self._lock:
            return [k for k in self._kv if fnmatch.fnmatch(k, pattern)]

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._kv.pop(key, None)

    async def close(self) -> None:  # pragma: no cover - no-op
        return None


# ---------------------------------------------------------------- sqlite


class SQLiteSCS:
    """Durable single-file SCS. Good fallback when Redis isn't available."""

    def __init__(self, path: str = "./mats_data/scs.db") -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, expires REAL)"
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS logs (key TEXT, ts REAL, event TEXT)")
        self._conn.commit()
        self._memory = InMemorySCS()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            cur = self._conn.execute("SELECT value, expires FROM kv WHERE key=?", (key,))
            row = cur.fetchone()
        if not row:
            return None
        value, expires = row
        if expires is not None and time.time() > expires:
            await self.delete(key)
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value, default=str)
        expires = time.time() + ttl if ttl else None
        async with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires) VALUES (?, ?, ?)",
                (key, payload, expires),
            )
            self._conn.commit()

    async def append(self, key: str, item: Any) -> None:
        current = await self.get(key)
        if not isinstance(current, list):
            current = []
        current.append(item)
        await self.set(key, current)

    async def publish(self, channel: str, event: dict[str, Any]) -> None:
        async with self._lock:
            self._conn.execute(
                "INSERT INTO logs (key, ts, event) VALUES (?, ?, ?)",
                (channel, time.time(), json.dumps(event, default=str)),
            )
            self._conn.commit()
        await self._memory.publish(channel, event)

    def subscribe(self, channel: str) -> AsyncIterator[dict[str, Any]]:
        return self._memory.subscribe(channel)

    async def keys(self, pattern: str) -> list[str]:
        import fnmatch

        async with self._lock:
            cur = self._conn.execute("SELECT key FROM kv")
            rows = [r[0] for r in cur.fetchall()]
        return [k for k in rows if fnmatch.fnmatch(k, pattern)]

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._conn.execute("DELETE FROM kv WHERE key=?", (key,))
            self._conn.commit()

    async def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------- redis


class RedisSCS:
    """Canonical CA-MCP blackboard. Hash-tagged session keys for Cluster."""

    def __init__(self, url: str = "redis://127.0.0.1:6379/0") -> None:
        import redis.asyncio as aioredis  # lazy import

        self._client = aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        payload = json.dumps(value, default=str)
        if ttl:
            await self._client.setex(key, ttl, payload)
        else:
            await self._client.set(key, payload)

    async def append(self, key: str, item: Any) -> None:
        await self._client.rpush(key, json.dumps(item, default=str))

    async def publish(self, channel: str, event: dict[str, Any]) -> None:
        await self._client.publish(channel, json.dumps(event, default=str))

    async def subscribe(self, channel: str) -> AsyncIterator[dict[str, Any]]:
        pubsub = self._client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                data = message.get("data")
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    yield {"raw": data}
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    async def keys(self, pattern: str) -> list[str]:
        return [k async for k in self._client.scan_iter(match=pattern)]

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------- factory


def build_scs(kind: str = "memory", **kwargs: Any) -> SCS:
    kind = kind.lower()
    if kind == "memory":
        return InMemorySCS()
    if kind == "sqlite":
        return SQLiteSCS(**kwargs)
    if kind == "redis":
        return RedisSCS(**kwargs)
    raise ValueError(f"Unknown SCS backend: {kind}")


def session_key(session_id: str, suffix: str) -> str:
    """Hash-tag the key so Redis Cluster keeps the whole session on one slot."""
    return f"scs:{{{session_id}}}:{suffix}"
