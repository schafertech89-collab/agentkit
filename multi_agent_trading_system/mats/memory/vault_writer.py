"""Obsidian-vault writer with per-tier directory isolation.

Writes markdown with YAML frontmatter so Obsidian renders them natively.
Respects the CLAUDE.md constitution: ``.raw/`` and ``Episodic/`` are write-
once, and all writes flow through the SSGM guard_write decorator for an
auditable hash chain.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..core.memcube import MemCube, Tier
from ..governance.ssgm import SSGM, FailureCategory, GovernanceError


TIER_DIR = {
    Tier.STM: "STM",
    Tier.MTM: "MTM/episodic",
    Tier.LTM: "LTM/semantic",
    Tier.DECAY: ".decay",
    Tier.RAW: ".raw",
}


def _slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", text).strip("-")
    return text[:80] or "untitled"


class VaultWriter:
    def __init__(self, vault_path: str | os.PathLike, ssgm: SSGM | None = None) -> None:
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.ssgm = ssgm

    # --------------------------------------------------------- API

    def write(self, cube: MemCube) -> Path:
        tier_dir = self.vault_path / TIER_DIR.get(cube.tier, "STM")
        tier_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        path = tier_dir / f"{stamp}-{_slug(cube.title)}-{cube.memcube_id[:8]}.md"

        if self.ssgm is None:
            self._do_write(path, cube)
            return path

        guarded = self.ssgm.guard_write(
            actor=cube.governance.last_modified_by or "system",
            action="VAULT_WRITE", target=str(path),
            metadata={"tier": cube.tier.value, "memcube_id": cube.memcube_id},
        )(self._do_write)
        guarded(path, cube)
        return path

    def read(self, path: str | os.PathLike) -> tuple[dict[str, Any], str]:
        text = Path(path).read_text(encoding="utf-8")
        if text.startswith("---\n"):
            _, frontmatter_raw, body = text.split("---", 2)
            fm = yaml.safe_load(frontmatter_raw) or {}
            return fm, body.strip()
        return {}, text

    def list_tier(self, tier: Tier) -> list[Path]:
        d = self.vault_path / TIER_DIR.get(tier, "STM")
        if not d.exists():
            return []
        return sorted(d.glob("*.md"))

    # --------------------------------------------------------- internals

    def _do_write(self, path: Path, cube: MemCube) -> None:
        if cube.tier in {Tier.RAW} and path.exists():
            raise GovernanceError(
                FailureCategory.STRUCTURAL,
                f"cannot overwrite immutable raw source: {path}",
            )
        frontmatter = yaml.safe_dump(cube.to_frontmatter(), sort_keys=False).strip()
        content = f"---\n{frontmatter}\n---\n\n# {cube.title}\n\n{cube.body}\n"
        path.write_text(content, encoding="utf-8")
