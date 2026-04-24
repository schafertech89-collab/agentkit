"""WebSocket relay — Approach C (Tier 3+).

Full-duplex WSS for live agent traces, SCS updates, and user interrupts.
Used by the Perchance approach C generator.

Client → server messages:
    {"type": "chat", "prompt": "...", "domain": "..."}
    {"type": "tool", "name": "...", "args": {...}}
    {"type": "tick"}
    {"type": "subscribe"}   (default)

Server → client messages:
    {"type": "delta", ...}       — LLM token stream
    {"type": "agent_trace", ...} — SCS events
    {"type": "tool_result", ...}
    {"type": "error", ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..core.orchestrator import Orchestrator


log = logging.getLogger("mats.ws_relay")


def build_ws_app(orchestrator: Orchestrator | None = None) -> FastAPI:
    orch = orchestrator or Orchestrator()
    app = FastAPI(title="mats-ws-relay")
    app.state.orch = orch

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        secret = os.environ.get("CONDUIT_SHARED_SECRET", "")
        token = websocket.query_params.get("token") or websocket.headers.get("X-Conduit-Secret")
        if not secret or token != secret:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        session = orch.cfg.session_id
        stop_event = asyncio.Event()

        async def pump() -> None:
            async for event in orch.scs.subscribe(f"scs:{{{session}}}:events"):
                if stop_event.is_set():
                    return
                await websocket.send_text(json.dumps(event, default=str))

        pumper = asyncio.create_task(pump())
        try:
            while True:
                text = await websocket.receive_text()
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"type": "error",
                                                          "error": "invalid json"}))
                    continue
                t = msg.get("type")
                if t == "chat":
                    await _handle_chat(websocket, msg)
                elif t == "tool":
                    await _handle_tool(websocket, orch, msg)
                elif t == "tick":
                    summary = await orch.tick()
                    await websocket.send_text(json.dumps({"type": "tick", "summary": summary}))
                elif t == "stop":
                    break
                else:
                    await websocket.send_text(json.dumps({"type": "ack", "echo": msg}))
        except WebSocketDisconnect:
            pass
        finally:
            stop_event.set()
            pumper.cancel()

    return app


async def _handle_chat(ws: WebSocket, msg: dict) -> None:
    import httpx
    url = os.environ.get("LITELLM_PROXY_URL", "http://127.0.0.1:4000")
    key = os.environ.get("LITELLM_MASTER_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    payload = {
        "model": msg.get("model", "claude-sonnet-4-6"),
        "messages": msg.get("messages") or [{"role": "user", "content": msg.get("prompt", "")}],
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=None) as hx:
            async with hx.stream("POST", f"{url}/v1/chat/completions",
                                 headers=headers, json=payload) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if line == "[DONE]":
                        await ws.send_text(json.dumps({"type": "done"}))
                        return
                    try:
                        delta = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    await ws.send_text(json.dumps({"type": "delta", "delta": delta}))
    except Exception as exc:  # noqa: BLE001
        await ws.send_text(json.dumps({"type": "error", "error": repr(exc)}))


async def _handle_tool(ws: WebSocket, orch: Orchestrator, msg: dict) -> None:
    from ..mcp_servers.factory import build_server
    composite = build_server(
        "composite", scs=orch.scs, vault_path=orch.cfg.vault_path,
        policy=orch.policy, connectors=orch.connectors,
    )
    name = msg.get("name")
    args = msg.get("args") or {}
    tool = composite.tools.get(name)
    if tool is None:
        await ws.send_text(json.dumps({"type": "error", "error": f"unknown tool {name}"}))
        return
    try:
        result = await tool.handler(**args)
    except Exception as exc:
        await ws.send_text(json.dumps({"type": "tool_error", "name": name, "error": repr(exc)}))
        return
    await ws.send_text(json.dumps({"type": "tool_result", "name": name,
                                   "result": result}, default=str))
