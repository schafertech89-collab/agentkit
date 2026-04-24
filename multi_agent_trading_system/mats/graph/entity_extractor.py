"""Entity-extraction pipeline.

For each freshly written MemCube we run a qwen3:14b (or any OpenAI-compatible
chat model via LiteLLM) extraction prompt producing triples:

    {"subject": "ETH", "predicate": "CORRELATED_WITH", "object": "BTC",
     "confidence": 0.82, "valid_from": "2026-04-24T12:00Z"}

Triples below a 0.6 confidence threshold are dropped. The extractor is
idempotent: repeat calls with the same content hash return cached results.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


EXTRACTION_PROMPT = """You extract (subject, predicate, object) triples from trading notes.
Return JSON: {"triples": [{"subject": str, "predicate": str, "object": str,
"confidence": float in [0, 1], "valid_from": ISO8601}]}
Predicates are one of: MENTIONS, REFERENCES, CONTRADICTS, DERIVED_FROM, CAUSED,
CORRELATED_WITH, LISTED_ON, PLACED_ON.
Drop anything you are not >= 0.6 confident about.
Input:\n"""


@dataclass
class Triple:
    subject: str
    predicate: str
    object: str
    confidence: float
    valid_from: str | None = None


class EntityExtractor:
    def __init__(self, litellm_url: str | None = None, api_key: str | None = None,
                 model: str = "ollama/qwen3:14b", cache_dir: str = "./mats_data/extractor_cache") -> None:
        self.url = litellm_url or os.environ.get("LITELLM_PROXY_URL", "http://127.0.0.1:4000")
        self.key = api_key or os.environ.get("LITELLM_MASTER_KEY", "")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def extract(self, text: str) -> list[Triple]:
        h = hashlib.sha256(text.encode()).hexdigest()[:32]
        cache_file = self.cache_dir / f"{h}.json"
        if cache_file.exists():
            return [Triple(**t) for t in json.loads(cache_file.read_text())]
        try:
            async with httpx.AsyncClient(timeout=60.0) as hx:
                r = await hx.post(
                    f"{self.url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": EXTRACTION_PROMPT},
                            {"role": "user", "content": text[:4000]},
                        ],
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
                r.raise_for_status()
                data = r.json()
                payload = json.loads(data["choices"][0]["message"]["content"])
                raw_triples = payload.get("triples", [])
        except Exception:
            raw_triples = []
        triples = [
            Triple(**{k: t.get(k) for k in ("subject", "predicate", "object",
                                             "confidence", "valid_from")})
            for t in raw_triples
            if t.get("subject") and t.get("object") and t.get("confidence", 0) >= 0.6
        ]
        cache_file.write_text(json.dumps([t.__dict__ for t in triples]))
        return triples
