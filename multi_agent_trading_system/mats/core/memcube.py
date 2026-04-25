"""MemCube — the MemOS-style top-level memory abstraction used across MATS.

A MemCube is the unit of memory that moves between tiers (STM/MTM/LTM), gets
replayed, decayed, consolidated, and audited. Every trading observation,
decision, reflection, or market snapshot is wrapped in a MemCube before it
touches the vault or the shared context store.

Three memory types are distinguished per the MemOS paper:
  * PARAMETRIC   — learned weights (RL policy checkpoints)
  * ACTIVATION   — transient context used for current reasoning
  * PLAINTEXT    — durable markdown notes, orderbook snapshots, decisions

The MemCube is a pydantic model so it serializes cleanly into the vault's
frontmatter-as-YAML and also into Redis / ToolHive over JSON.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MemoryType(str, Enum):
    PARAMETRIC = "parametric"
    ACTIVATION = "activation"
    PLAINTEXT = "plaintext"


class Tier(str, Enum):
    STM = "STM"
    MTM = "MTM"
    LTM = "LTM"
    DECAY = "DECAY"
    RAW = "RAW"


class Provenance(BaseModel):
    source: str
    hash: str
    immutable: bool = False
    source_path: str | None = None

    @classmethod
    def for_payload(cls, source: str, payload: Any, *, immutable: bool = False) -> "Provenance":
        material = payload if isinstance(payload, (str, bytes)) else repr(payload)
        if isinstance(material, str):
            material = material.encode("utf-8")
        return cls(source=source, hash=hashlib.sha256(material).hexdigest(), immutable=immutable)


class Governance(BaseModel):
    version: int = 1
    last_modified_by: str = "system"
    audit_chain: list[str] = Field(default_factory=list)


class PolicySignal(BaseModel):
    utility_score: float = 0.0
    interference: float = 0.0
    regret: float = 0.0


class MemCube(BaseModel):
    """A single addressable memory unit."""

    memcube_id: str = Field(default_factory=lambda: f"mc_{uuid.uuid4().hex}")
    memory_type: MemoryType = MemoryType.PLAINTEXT
    tier: Tier = Tier.STM
    importance: float = 0.5
    activation_strength: float = 1.0
    decay_rate: float = 0.05
    replay_count: int = 0
    consolidated_to: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    tags: list[str] = Field(default_factory=list)
    title: str = ""
    body: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    provenance: Provenance | None = None
    governance: Governance = Field(default_factory=Governance)
    policy_signal: PolicySignal = Field(default_factory=PolicySignal)

    @field_validator("importance", "activation_strength", "decay_rate")
    @classmethod
    def _clamp_unit(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    # ------------------------------------------------------------------ API

    def touch(self, *, by: str = "system") -> None:
        self.updated_at = datetime.now(timezone.utc)
        self.governance.version += 1
        self.governance.last_modified_by = by

    def decay(self, days: float = 1.0) -> None:
        """Exponential decay of activation_strength by `decay_rate` per day."""
        factor = (1.0 - self.decay_rate) ** days
        self.activation_strength = max(0.0, self.activation_strength * factor)

    def replay(self, reward: float = 0.0) -> None:
        self.replay_count += 1
        self.policy_signal.utility_score = 0.9 * self.policy_signal.utility_score + 0.1 * reward
        # Replay strengthens the cube (HEMA-style replay).
        self.activation_strength = min(1.0, self.activation_strength + 0.1)

    def promote(self, target: Tier) -> None:
        order = [Tier.STM, Tier.MTM, Tier.LTM]
        if target not in order:
            self.tier = target
            return
        self.tier = target
        # Promotion increases importance a touch.
        self.importance = min(1.0, self.importance + 0.05)

    def to_frontmatter(self) -> dict[str, Any]:
        """Serializer for markdown vault files."""
        return {
            "memcube_id": self.memcube_id,
            "memory_type": self.memory_type.value,
            "tier": self.tier.value,
            "importance": self.importance,
            "activation_strength": self.activation_strength,
            "decay_rate": self.decay_rate,
            "replay_count": self.replay_count,
            "consolidated_to": self.consolidated_to,
            "tags": list(self.tags),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "provenance": self.provenance.model_dump() if self.provenance else None,
            "governance": self.governance.model_dump(),
            "policy_signal": self.policy_signal.model_dump(),
        }
