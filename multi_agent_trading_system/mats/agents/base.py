"""Agent base class and shared context object.

Every MATS agent derives from ``Agent`` and implements a single ``step``
coroutine. Agents are orchestrated in round-robin loops by the orchestrator.
Each ``step`` receives an ``AgentContext`` holding shared state (SCS handle,
connectors, memory hooks, reflective feedback) and returns an ``AgentOutput``
with the memcube to persist and whatever side-effects were enqueued.

Agents stay intentionally stateless apart from small "self" scratch in the
context — durable state goes through the SCS so any agent can resume from a
crash without rehydrating mid-decision.
"""

from __future__ import annotations

import abc
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..connectors._common import Receipt
from ..core.hooks import ConduitHooks, DEFAULT_HOOKS
from ..core.memcube import MemCube, MemoryType, Provenance, Tier
from ..core.scs import SCS, session_key
from ..governance.ssgm import SSGM


log = logging.getLogger(__name__)


@dataclass
class AgentContext:
    session_id: str
    scs: SCS
    connectors: dict[str, Any] = field(default_factory=dict)
    hooks: ConduitHooks = field(default_factory=lambda: DEFAULT_HOOKS)
    ssgm: SSGM | None = None
    reflections: list[dict] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)
    clock: Callable[[], float] = time.time

    def get_connector(self, name: str) -> Any:
        return self.connectors.get(name)

    async def read(self, suffix: str, default: Any = None) -> Any:
        value = await self.scs.get(session_key(self.session_id, suffix))
        return value if value is not None else default

    async def write(self, suffix: str, value: Any, ttl: int | None = None) -> None:
        await self.scs.set(session_key(self.session_id, suffix), value, ttl=ttl)

    async def append(self, suffix: str, item: Any) -> None:
        await self.scs.append(session_key(self.session_id, suffix), item)

    async def publish(self, event: dict[str, Any]) -> None:
        await self.scs.publish(session_key(self.session_id, "events"), event)


@dataclass
class AgentOutput:
    agent: str
    memcube: MemCube
    stance: float = 0.0  # [-1, 1] bullish scalar used by the planner
    actions: list[dict] = field(default_factory=list)
    receipts: list[Receipt] = field(default_factory=list)
    explain: dict = field(default_factory=dict)


class Agent(abc.ABC):
    """Base class every specialised agent derives from."""

    name: str = "agent"
    output_tier: Tier = Tier.STM

    def __init__(self, *, hooks: ConduitHooks | None = None) -> None:
        self.hooks = hooks or DEFAULT_HOOKS
        self.id = f"{self.name}-{uuid.uuid4().hex[:8]}"

    @abc.abstractmethod
    async def step(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        raise NotImplementedError

    # ---------------------------------------------------------- helpers

    def memcube(
        self,
        *,
        title: str,
        body: str = "",
        payload: dict | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        memory_type: MemoryType = MemoryType.PLAINTEXT,
    ) -> MemCube:
        cube = MemCube(
            memory_type=memory_type,
            tier=self.output_tier,
            importance=importance,
            tags=[self.name, *(tags or [])],
            title=title,
            body=body,
            payload=payload or {},
            provenance=Provenance.for_payload(
                source=self.name, payload=body or (payload and str(payload)) or title,
            ),
        )
        cube.governance.last_modified_by = self.name
        return cube

    async def run_safely(self, ctx: AgentContext, input_payload: dict[str, Any]) -> AgentOutput:
        try:
            out = await self.step(ctx, input_payload)
            await self.hooks.emit("agent_step", self.name, out)
            return out
        except Exception as exc:  # noqa: BLE001
            log.exception("%s failed: %s", self.name, exc)
            cube = self.memcube(
                title=f"{self.name} error", body=str(exc),
                payload={"error": repr(exc)},
                importance=0.1,
            )
            await self.hooks.emit("error", exc)
            return AgentOutput(agent=self.name, memcube=cube, stance=0.0,
                               explain={"error": str(exc)})
