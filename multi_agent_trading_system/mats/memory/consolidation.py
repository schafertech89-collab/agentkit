"""ConsolidateAgent — cron-driven memory consolidation.

Runs on a schedule (systemd timer every 4h, or the ``mats-consolidate`` CLI).
It reads episodic memcubes older than a configurable threshold, summarizes
them into semantic memcubes using the LiteLLM proxy, and updates the
``consolidated_into`` pointer on the source. Forgotten cubes are soft-moved
to ``.decay/`` rather than deleted so strong recall cues can revive them.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml

from ..core.memcube import MemCube, Tier
from ..governance.ssgm import SSGM
from ..learning.ppo_policy import MemoryPolicyController, PolicyAction
from .vault_writer import VaultWriter


class ConsolidateAgent:
    def __init__(
        self,
        *,
        vault_path: str,
        ssgm: SSGM | None = None,
        policy: MemoryPolicyController | None = None,
        litellm_url: str | None = None,
        litellm_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        episodic_age_days: float = 7.0,
    ) -> None:
        self.writer = VaultWriter(vault_path, ssgm=ssgm)
        self.policy = policy or MemoryPolicyController()
        self.ssgm = ssgm
        self.litellm_url = litellm_url or os.environ.get("LITELLM_PROXY_URL", "http://127.0.0.1:4000")
        self.litellm_key = litellm_key or os.environ.get("LITELLM_MASTER_KEY", "")
        self.model = model
        self.episodic_age_days = episodic_age_days

    async def run_once(self) -> dict:
        stats = {"consolidated": 0, "forgotten": 0, "reinforced": 0, "kept": 0}
        episodic_dir = self.writer.vault_path / "MTM" / "episodic"
        if not episodic_dir.exists():
            return stats
        cutoff = time.time() - self.episodic_age_days * 86400
        buckets: list[tuple[Path, MemCube]] = []
        for path in sorted(episodic_dir.glob("*.md")):
            fm, body = self.writer.read(path)
            try:
                cube = MemCube(**_restore_frontmatter(fm))
                cube.body = body
            except Exception:
                continue
            if cube.created_at.timestamp() > cutoff:
                continue
            buckets.append((path, cube))

        summaries: list[str] = []
        for path, cube in buckets:
            decision = self.policy.decide(cube)
            if decision.action == PolicyAction.CONSOLIDATE:
                summaries.append(f"- {cube.title}\n  {cube.body[:400]}")
                stats["consolidated"] += 1
            elif decision.action == PolicyAction.FORGET:
                self._move_to_decay(path)
                stats["forgotten"] += 1
            elif decision.action == PolicyAction.REINFORCE:
                cube.replay(reward=0.1)
                self.writer.write(cube)
                stats["reinforced"] += 1
            else:
                stats["kept"] += 1

        if summaries:
            semantic_body = await self._summarize("\n\n".join(summaries))
            sem_cube = MemCube(
                tier=Tier.LTM, importance=0.6, activation_strength=1.0,
                title=f"consolidation {datetime.now(timezone.utc).date().isoformat()}",
                body=semantic_body, tags=["consolidation", "semantic"],
            )
            self.writer.write(sem_cube)
        return stats

    async def _summarize(self, raw: str) -> str:
        if not self.litellm_key or not self.litellm_url:
            return "[[no LiteLLM configured — raw content preserved]]\n" + raw
        try:
            async with httpx.AsyncClient(timeout=60.0) as hx:
                r = await hx.post(
                    f"{self.litellm_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.litellm_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You summarize episodic trading notes into "
                             "semantic-tier knowledge. Preserve factual claims, mark uncertainty, "
                             "and link to source titles as [[wikilinks]]."},
                            {"role": "user", "content": raw[:8000]},
                        ],
                        "temperature": 0.2,
                    },
                )
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"[[consolidation failed: {exc}]]\n\n{raw}"

    def _move_to_decay(self, path: Path) -> None:
        dst = self.writer.vault_path / ".decay" / path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.ssgm is not None:
            guarded = self.ssgm.guard_write(
                actor="consolidate_agent", action="DECAY_MOVE", target=str(dst),
            )(path.rename)
            guarded(dst)
        else:
            path.rename(dst)


def _restore_frontmatter(fm: dict[str, Any]) -> dict[str, Any]:
    """Pydantic needs ISO strings parsed back to datetimes."""
    out = dict(fm)
    for k in ("created_at", "updated_at"):
        val = out.get(k)
        if isinstance(val, str):
            try:
                out[k] = datetime.fromisoformat(val)
            except ValueError:
                out.pop(k, None)
    return out
