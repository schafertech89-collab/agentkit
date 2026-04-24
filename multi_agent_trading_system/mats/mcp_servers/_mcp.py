"""Lightweight JSON-RPC-ish MCP server adapter.

We don't take a hard dependency on the ``mcp`` Python SDK because MATS needs
to run in stripped-down environments. Instead we implement the minimum
Streamable-HTTP shape the 2025-11-25 spec requires:

    POST /mcp { "jsonrpc": "2.0", "id": ..., "method": "tools/call",
                "params": { "name": "...", "arguments": {...} } }

Tools are registered via ``@server.tool("name")``. Each tool is an async
coroutine returning a JSON-serializable value. Errors propagate as standard
JSON-RPC error objects.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


log = logging.getLogger(__name__)


@dataclass
class MCPTool:
    name: str
    description: str
    handler: Callable[..., Awaitable[Any]]
    schema: dict = field(default_factory=dict)


class MCPServer:
    """Minimal Streamable-HTTP MCP server."""

    def __init__(self, name: str, version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self.tools: dict[str, MCPTool] = {}
        self.app = FastAPI(title=f"mats/{name}")
        self._install_routes()

    # ----------------------------------------------------------- API

    def tool(self, name: str, description: str = "", schema: dict | None = None):
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            handler = fn if asyncio.iscoroutinefunction(fn) else self._asyncify(fn)
            self.tools[name] = MCPTool(name=name, description=description,
                                       handler=handler, schema=schema or {})
            return fn

        return decorator

    # ----------------------------------------------------------- routes

    def _install_routes(self) -> None:
        @self.app.get("/.well-known/mcp-server")
        async def well_known() -> dict:
            return {
                "name": self.name, "version": self.version,
                "protocol": "mcp/2025-11-25", "transport": "streamable-http",
                "tools": [{"name": t.name, "description": t.description,
                           "input_schema": t.schema}
                          for t in self.tools.values()],
            }

        @self.app.post("/mcp")
        async def mcp(req: Request) -> JSONResponse:
            try:
                data = await req.json()
            except Exception:
                raise HTTPException(400, "invalid json")
            method = data.get("method")
            rpc_id = data.get("id")
            if method == "tools/list":
                result = {"tools": [
                    {"name": t.name, "description": t.description,
                     "inputSchema": t.schema}
                    for t in self.tools.values()
                ]}
                return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result})
            if method == "tools/call":
                params = data.get("params") or {}
                name = params.get("name")
                args = params.get("arguments", {}) or {}
                tool = self.tools.get(name)
                if not tool:
                    return JSONResponse({
                        "jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32601, "message": f"tool not found: {name}"},
                    })
                try:
                    result = await tool.handler(**args)
                except Exception as exc:
                    log.exception("tool %s failed", name)
                    return JSONResponse({
                        "jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32000, "message": str(exc),
                                  "data": {"trace": traceback.format_exc()}},
                    })
                return JSONResponse({
                    "jsonrpc": "2.0", "id": rpc_id,
                    "result": {"content": [{"type": "json", "json": result}]},
                })
            return JSONResponse({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            })

    # ----------------------------------------------------- stdio entry

    async def serve_stdio(self) -> None:
        """Serve the MCP server over stdio for Claude Desktop."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader(loop=loop)
        protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        writer_transport, writer_protocol = await loop.connect_write_pipe(
            lambda: asyncio.Protocol(), sys.stdout,
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, loop)
        while True:
            line = await reader.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            resp = await self._dispatch(msg)
            writer.write((json.dumps(resp) + "\n").encode())
            await writer.drain()

    async def _dispatch(self, msg: dict) -> dict:
        method = msg.get("method")
        rpc_id = msg.get("id")
        params = msg.get("params") or {}
        if method == "tools/list":
            return {
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {"tools": [
                    {"name": t.name, "description": t.description, "inputSchema": t.schema}
                    for t in self.tools.values()
                ]},
            }
        if method == "tools/call":
            tool = self.tools.get(params.get("name"))
            if not tool:
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32601, "message": "tool not found"}}
            try:
                result = await tool.handler(**(params.get("arguments") or {}))
            except Exception as exc:
                return {"jsonrpc": "2.0", "id": rpc_id,
                        "error": {"code": -32000, "message": str(exc)}}
            return {"jsonrpc": "2.0", "id": rpc_id,
                    "result": {"content": [{"type": "json", "json": result}]}}
        return {"jsonrpc": "2.0", "id": rpc_id,
                "error": {"code": -32601, "message": f"unknown method {method}"}}

    @staticmethod
    def _asyncify(fn: Callable[..., Any]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper
