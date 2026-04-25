"""Append-only SHA-256 hash chain. Tamper-evident audit log.

Every write MCP tool wraps its operation with ``AuditChain.append`` and records
before/after hashes plus the actor identity. On boot the chain is verified
end-to-end; mismatch halts writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class AuditEntry:
    seq: int
    ts: float
    actor: str
    action: str
    target: str
    before_hash: str
    after_hash: str
    payload_digest: str
    prev_chain_hash: str
    chain_hash: str = ""
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class AuditChain:
    """Line-per-entry JSONL file with a self-linking SHA-256 chain."""

    def __init__(self, path: str | os.PathLike = "./vault/.governance/audit.jsonl",
                 seed: str | None = None) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seed = seed or os.environ.get("GOVERNANCE_HASH_SEED", "mats-genesis")
        self._lock = threading.Lock()
        if not self._path.exists():
            genesis = AuditEntry(
                seq=0,
                ts=time.time(),
                actor="genesis",
                action="GENESIS",
                target="-",
                before_hash="",
                after_hash="",
                payload_digest=hashlib.sha256(self._seed.encode()).hexdigest(),
                prev_chain_hash="",
            )
            genesis.chain_hash = self._hash_entry(genesis)
            self._append_entry(genesis)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _hash_entry(entry: AuditEntry) -> str:
        payload = json.dumps(
            {
                "seq": entry.seq,
                "ts": entry.ts,
                "actor": entry.actor,
                "action": entry.action,
                "target": entry.target,
                "before_hash": entry.before_hash,
                "after_hash": entry.after_hash,
                "payload_digest": entry.payload_digest,
                "prev_chain_hash": entry.prev_chain_hash,
                "metadata": entry.metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _append_entry(self, entry: AuditEntry) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(entry.to_json() + "\n")

    # ---------------------------------------------------------------- API

    def append(
        self,
        *,
        actor: str,
        action: str,
        target: str,
        before: bytes | str = b"",
        after: bytes | str = b"",
        payload: dict | None = None,
        metadata: dict | None = None,
    ) -> AuditEntry:
        before_b = before.encode() if isinstance(before, str) else before
        after_b = after.encode() if isinstance(after, str) else after
        digest = hashlib.sha256(
            json.dumps(payload or {}, sort_keys=True, default=str).encode()
        ).hexdigest()
        with self._lock:
            last = self._last_entry()
            seq = (last.seq + 1) if last else 0
            entry = AuditEntry(
                seq=seq,
                ts=time.time(),
                actor=actor,
                action=action,
                target=target,
                before_hash=hashlib.sha256(before_b).hexdigest(),
                after_hash=hashlib.sha256(after_b).hexdigest(),
                payload_digest=digest,
                prev_chain_hash=last.chain_hash if last else "",
                metadata=metadata or {},
            )
            entry.chain_hash = self._hash_entry(entry)
            self._append_entry(entry)
        return entry

    def _last_entry(self) -> AuditEntry | None:
        last: AuditEntry | None = None
        for entry in self.iter_entries():
            last = entry
        return last

    def iter_entries(self) -> Iterator[AuditEntry]:
        if not self._path.exists():
            return iter(())
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield AuditEntry(**data)

    def verify(self) -> tuple[bool, str | None]:
        prev = ""
        for idx, entry in enumerate(self.iter_entries()):
            expected_prev = prev
            recomputed = self._hash_entry(entry)
            if entry.prev_chain_hash != expected_prev:
                return False, f"seq {entry.seq}: prev_hash mismatch"
            if entry.chain_hash != recomputed:
                return False, f"seq {entry.seq}: chain_hash mismatch"
            if entry.seq != idx:
                return False, f"seq {entry.seq}: non-contiguous (expected {idx})"
            prev = entry.chain_hash
        return True, None
