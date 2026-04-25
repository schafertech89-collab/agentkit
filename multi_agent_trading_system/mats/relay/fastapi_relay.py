"""FastAPI relay — Approach A (dumb Cloudflare Tunnel relay) + SSE streaming.

This is the single HTTPS front-door for the Perchance client. It validates
a shared secret, proxies chat completions through LiteLLM, and forwards
MCP tool calls to the composite MCP server.

Endpoints:
    POST /chat                — body: {prompt, model?, domain?}, streams SSE
    POST /tool/:name           — calls a named MCP tool
    GET  /status               — orchestrator tick counter + mode
    POST /ingest/outcome       — posts realized P&L for a decision
    GET  /events               — SSE stream of agent traces
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..core.hooks import ConduitHooks, DEFAULT_HOOKS
from ..core.orchestrator import Orchestrator, OrchestratorConfig
from ..mcp_servers.factory import build_server


log = logging.getLogger("mats.relay")


def _verify_secret(req: Request) -> None:
    expected = os.environ.get("CONDUIT_SHARED_SECRET", "")
    got = req.headers.get("X-Conduit-Secret", "")
    if not expected or got != expected:
        raise HTTPException(401, "invalid conduit secret")


def build_app(orchestrator: Orchestrator | None = None,
              hooks: ConduitHooks | None = None) -> FastAPI:
    hooks = hooks or DEFAULT_HOOKS
    orchestrator = orchestrator or Orchestrator()
    composite = build_server(
        "composite",
        scs=orchestrator.scs,
        vault_path=orchestrator.cfg.vault_path,
        policy=orchestrator.policy,
        connectors=orchestrator.connectors,
    )

    app = FastAPI(title="mats-relay", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://user-content.perchance.org", "*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    app.state.orchestrator = orchestrator
    app.state.mcp = composite
    app.state.hooks = hooks
    app.state.event_subscribers: list[asyncio.Queue] = []

    # Republish SCS events to the event stream
    session_channel = f"scs:{{{orchestrator.cfg.session_id}}}:events"

    async def _event_pump() -> None:
        async for event in orchestrator.scs.subscribe(session_channel):
            for q in list(app.state.event_subscribers):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.event_pump = asyncio.create_task(_event_pump())

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        app.state.event_pump.cancel()
        await orchestrator.scs.close()

    # ----------------------------------------------------------- /status
    @app.get("/status")
    async def status() -> dict:
        live = os.environ.get("MATS_LIVE", "0") == "1"
        return {
            "mode": "live" if live else "paper",
            "tier": orchestrator.cfg.tier,
            "session_id": orchestrator.cfg.session_id,
            "tick": orchestrator.tick_count,
            "universe": orchestrator.cfg.universe,
        }

    # ----------------------------------------------------------- /chat
    @app.post("/chat")
    async def chat(req: Request, _: None = Depends(_verify_secret)):
        body = await req.json()
        prompt = body.get("prompt", "")
        model = body.get("model") or os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6")
        domain = body.get("domain", "projects")
        return EventSourceResponse(_chat_stream(prompt, model, domain, body))

    async def _chat_stream(prompt: str, model: str, domain: str, body: dict) -> AsyncIterator[dict]:
        url = os.environ.get("LITELLM_PROXY_URL", "http://127.0.0.1:4000")
        key = os.environ.get("LITELLM_MASTER_KEY", "")
        messages = body.get("messages") or [
            {"role": "system",
             "content": f"You are MATS, a multi-agent trading system. "
                        f"Use tools when asked to look up markets or submit trades. "
                        f"Current domain: {domain}."},
            {"role": "user", "content": prompt},
        ]
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        payload = {"model": model, "messages": messages, "stream": True}
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
                            yield {"event": "done", "data": "[DONE]"}
                            return
                        try:
                            delta = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield {"event": "delta", "data": json.dumps(delta)}
        except Exception as exc:  # noqa: BLE001
            yield {"event": "error", "data": json.dumps({"error": repr(exc)})}

    # ----------------------------------------------------------- /tool
    @app.post("/tool/{name}")
    async def call_tool(name: str, req: Request, _: None = Depends(_verify_secret)):
        tool = composite.tools.get(name)
        if tool is None:
            raise HTTPException(404, f"unknown tool {name}")
        args = await req.json()
        result = await tool.handler(**(args or {}))
        return JSONResponse({"tool": name, "result": result})

    # ----------------------------------------------------------- /ingest
    @app.post("/ingest/outcome")
    async def ingest_outcome(req: Request, _: None = Depends(_verify_secret)):
        body = await req.json()
        decision_id = body.get("decision_id")
        pnl = float(body.get("pnl", 0.0))
        if not decision_id:
            raise HTTPException(400, "decision_id required")
        outcomes = await orchestrator.scs.get(
            f"scs:{{{orchestrator.cfg.session_id}}}:outcomes"
        ) or {}
        outcomes[decision_id] = {"pnl": pnl, "ts": time.time(), **body}
        await orchestrator.scs.set(
            f"scs:{{{orchestrator.cfg.session_id}}}:outcomes", outcomes,
        )
        return {"ok": True, "decision_id": decision_id}

    # ----------------------------------------------------------- /events
    @app.get("/events")
    async def events(_: None = Depends(_verify_secret)):
        queue: asyncio.Queue = asyncio.Queue(maxsize=1024)
        app.state.event_subscribers.append(queue)

        async def gen() -> AsyncIterator[dict]:
            try:
                while True:
                    event = await queue.get()
                    yield {"event": event.get("type", "event"),
                           "data": json.dumps(event, default=str)}
            finally:
                if queue in app.state.event_subscribers:
                    app.state.event_subscribers.remove(queue)

        return EventSourceResponse(gen())

    # ----------------------------------------------------------- /tick
    @app.post("/tick")
    async def tick(_: None = Depends(_verify_secret)):
        summary = await orchestrator.tick()
        return summary

    # ----------------------------------------------------------- /mcp proxy
    app.mount("/mcp", composite.app)

    return app


def main() -> None:
    import uvicorn
    port = int(os.environ.get("RELAY_PORT", "8787"))
    uvicorn.run(build_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
