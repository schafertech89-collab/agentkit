"""Obsidian-vault MCP tools.

Exposes vault reads (BM25 search + file ops) and governance-audited writes.
For simplicity we bundle a bm25s index over all markdown files and rebuild it
on startup. Large vaults can swap this for the upstream ``conduit-search``
FastMCP stdio server.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ._mcp import MCPServer


def _iter_markdown(vault_path: Path):
    for p in sorted(vault_path.rglob("*.md")):
        if ".obsidian" in p.parts:
            continue
        yield p


def register_obsidian_tools(server: MCPServer, vault_path: str) -> None:
    root = Path(vault_path)
    root.mkdir(parents=True, exist_ok=True)

    index: dict[str, Any] = {"docs": [], "paths": []}

    def _rebuild_index() -> None:
        try:
            import bm25s
        except Exception:
            return
        docs: list[str] = []
        paths: list[str] = []
        for p in _iter_markdown(root):
            try:
                docs.append(p.read_text(encoding="utf-8"))
                paths.append(str(p))
            except OSError:
                continue
        if not docs:
            index["retriever"] = None
            index["paths"] = []
            return
        retriever = bm25s.BM25()
        tokenised = bm25s.tokenize(docs)
        retriever.index(tokenised)
        index["retriever"] = retriever
        index["paths"] = paths
        index["docs"] = docs

    _rebuild_index()

    @server.tool("obsidian.list_files", "List markdown files in the vault.")
    async def list_files(prefix: str = "") -> list[str]:
        files = []
        for p in _iter_markdown(root):
            rel = str(p.relative_to(root))
            if not prefix or rel.startswith(prefix):
                files.append(rel)
        return files

    @server.tool("obsidian.get_file", "Read a markdown file from the vault.")
    async def get_file(path: str) -> dict:
        target = (root / path).resolve()
        if not str(target).startswith(str(root.resolve())):
            return {"error": "path escapes vault"}
        if not target.exists():
            return {"error": "not found"}
        return {"path": str(target.relative_to(root)),
                "content": target.read_text(encoding="utf-8")}

    @server.tool("obsidian.search_bm25", "BM25+ search across the vault.")
    async def search_bm25(query: str, k: int = 5) -> list[dict]:
        try:
            import bm25s
        except Exception:
            return [{"error": "bm25s not installed"}]
        retriever = index.get("retriever")
        if retriever is None:
            _rebuild_index()
            retriever = index.get("retriever")
        if retriever is None:
            return []
        tokenised = bm25s.tokenize([query])
        docs, scores = retriever.retrieve(tokenised, k=min(k, len(index["paths"])))
        results = []
        for doc_idx, score in zip(docs[0], scores[0]):
            path = index["paths"][int(doc_idx)]
            snippet = index["docs"][int(doc_idx)][:500]
            results.append({"path": path, "score": float(score), "snippet": snippet})
        return results

    @server.tool("obsidian.append", "Append content to a markdown file (creates if missing).")
    async def append(path: str, content: str) -> dict:
        target = (root / path).resolve()
        if not str(target).startswith(str(root.resolve())):
            return {"error": "path escapes vault"}
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return {"ok": True, "path": str(target.relative_to(root))}

    @server.tool("obsidian.patch", "Replace a regex match in a file.")
    async def patch(path: str, pattern: str, replacement: str) -> dict:
        target = (root / path).resolve()
        if not str(target).startswith(str(root.resolve())):
            return {"error": "path escapes vault"}
        if not target.exists():
            return {"error": "not found"}
        text = target.read_text(encoding="utf-8")
        new_text, count = re.subn(pattern, replacement, text)
        if count:
            target.write_text(new_text, encoding="utf-8")
        return {"ok": True, "replacements": count}

    @server.tool("obsidian.rebuild_index", "Rebuild the BM25 search index.")
    async def rebuild() -> dict:
        _rebuild_index()
        return {"ok": True, "docs": len(index.get("paths", []))}
