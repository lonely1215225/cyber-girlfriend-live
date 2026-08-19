"""Small Streamable-HTTP MCP gateway for realtime LLM function tools."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx


logger = logging.getLogger("s2s.mcp")
PROTOCOL_VERSION = "2025-03-26"
MAX_TOOL_OUTPUT_CHARS = max(1000, int(os.environ.get("MCP_MAX_OUTPUT_CHARS", "6000")))
DEFAULT_TOOL_ALLOWLIST = {
    "coingecko:execute",
    "coingecko:search_docs",
    "exa:web_search_exa",
    "exa:web_fetch_exa",
    "gdelt:gdelt_search_articles",
}
_configured_allowlist = os.environ.get("MCP_TOOL_ALLOWLIST", "").strip()
TOOL_ALLOWLIST = (
    {item.strip() for item in _configured_allowlist.split(",") if item.strip()}
    if _configured_allowlist
    else DEFAULT_TOOL_ALLOWLIST
)
_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _parse_rpc_response(response: httpx.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" not in content_type:
        data = response.json()
        return data if isinstance(data, dict) else {}
    result: dict[str, Any] = {}
    for line in response.text.splitlines():
        if not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        candidate = json.loads(raw)
        if isinstance(candidate, dict):
            result = candidate
    return result


def _tool_output(result: dict[str, Any]) -> str:
    pieces: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            pieces.append(str(block["text"]))
        elif block.get("type") in {"resource", "resource_link"}:
            pieces.append(json.dumps(block, ensure_ascii=False))
    if not pieces and result.get("structuredContent") is not None:
        pieces.append(json.dumps(result["structuredContent"], ensure_ascii=False))
    text = "\n".join(pieces).strip() or "MCP tool returned no content."
    if result.get("isError"):
        text = f"MCP tool error: {text}"
    if len(text) > MAX_TOOL_OUTPUT_CHARS:
        text = f"{text[:MAX_TOOL_OUTPUT_CHARS]}\n[结果过长，已截断]"
    return text


@dataclass(frozen=True)
class McpServerConfig:
    key: str
    label: str
    url: str


class McpHttpClient:
    def __init__(self, config: McpServerConfig) -> None:
        self.config = config
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=12.0), follow_redirects=True)
        self._lock = asyncio.Lock()
        self._session_id = ""
        self._initialized = False
        self._next_id = 0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._http.post(self.config.url, headers=self._headers(), json=payload)
        response.raise_for_status()
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        data = _parse_rpc_response(response)
        if data.get("error"):
            error = data["error"]
            raise RuntimeError(str(error.get("message") if isinstance(error, dict) else error))
        return data

    async def _initialize(self) -> None:
        if self._initialized:
            return
        self._next_id += 1
        await self._post(
            {
                "jsonrpc": "2.0",
                "id": self._next_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "cyber-girlfriend-live", "version": "1.0.0"},
                },
            }
        )
        await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._initialized = True

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            for attempt in range(2):
                try:
                    await self._initialize()
                    self._next_id += 1
                    data = await self._post(
                        {
                            "jsonrpc": "2.0",
                            "id": self._next_id,
                            "method": method,
                            "params": params,
                        }
                    )
                    result = data.get("result")
                    return result if isinstance(result, dict) else {}
                except (httpx.HTTPError, RuntimeError, json.JSONDecodeError):
                    self._initialized = False
                    self._session_id = ""
                    if attempt:
                        raise
            return {}

    async def close(self) -> None:
        await self._http.aclose()


class McpGateway:
    def __init__(self) -> None:
        enabled = os.environ.get("MCP_ENABLED", "1").strip().lower() in {"1", "true", "yes", "on"}
        configs = []
        if enabled:
            configs = [
                McpServerConfig(
                    "coingecko",
                    "CoinGecko",
                    os.environ.get("MCP_COINGECKO_URL", "https://mcp.api.coingecko.com/mcp").strip(),
                ),
                McpServerConfig(
                    "exa",
                    "Exa",
                    os.environ.get("MCP_EXA_URL", "https://mcp.exa.ai/mcp").strip(),
                ),
                McpServerConfig(
                    "gdelt",
                    "GDELT",
                    os.environ.get("MCP_GDELT_URL", "https://gdelt.caseyjhand.com/mcp").strip(),
                ),
            ]
        self.clients = {config.key: McpHttpClient(config) for config in configs if config.url}
        self._tools: list[dict[str, Any]] = []
        self._mapping: dict[str, tuple[McpHttpClient, str]] = {}
        self._tools_lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.clients)

    async def list_tools(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        async with self._tools_lock:
            if self._tools and not refresh:
                return [dict(tool) for tool in self._tools]
            tools: list[dict[str, Any]] = []
            mapping: dict[str, tuple[McpHttpClient, str]] = {}
            results = await asyncio.gather(
                *(client.request("tools/list", {}) for client in self.clients.values()),
                return_exceptions=True,
            )
            for client, result in zip(self.clients.values(), results):
                if isinstance(result, Exception):
                    logger.warning("MCP %s tools/list failed: %s", client.config.label, result)
                    continue
                for raw in result.get("tools") or []:
                    if not isinstance(raw, dict) or not raw.get("name"):
                        continue
                    original_name = str(raw["name"])
                    if f"{client.config.key}:{original_name}" not in TOOL_ALLOWLIST:
                        continue
                    public_name = _SAFE_NAME_RE.sub("_", f"mcp_{client.config.key}_{original_name}")[:64]
                    schema = raw.get("inputSchema") if isinstance(raw.get("inputSchema"), dict) else {
                        "type": "object",
                        "properties": {},
                    }
                    schema = dict(schema)
                    schema.pop("$schema", None)
                    tools.append(
                        {
                            "type": "function",
                            "name": public_name,
                            "description": f"[{client.config.label} MCP] {raw.get('description') or original_name}",
                            "parameters": schema,
                            "source": client.config.key,
                        }
                    )
                    mapping[public_name] = (client, original_name)
            # CoinGecko's official server exposes a powerful generic `execute`
            # tool, but small local models frequently omit its required
            # TypeScript `run(client)` wrapper. Give the model a narrow,
            # deterministic price operation while still executing it through
            # the official CoinGecko MCP server underneath.
            coingecko = self.clients.get("coingecko")
            if coingecko and "mcp_coingecko_execute" in mapping:
                price_name = "mcp_coingecko_price"
                tools.append(
                    {
                        "type": "function",
                        "name": price_name,
                        "description": (
                            "[CoinGecko MCP] Get the current cryptocurrency price. "
                            "Use this immediately for current Bitcoin, Ethereum, or other coin prices."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "coin_id": {
                                    "type": "string",
                                    "description": "CoinGecko coin id, for example bitcoin or ethereum.",
                                },
                                "vs_currencies": {
                                    "type": "string",
                                    "description": "Comma-separated quote currencies, for example usd,cny.",
                                    "default": "usd,cny",
                                },
                            },
                            "required": ["coin_id"],
                        },
                        "source": "coingecko",
                    }
                )
                mapping[price_name] = (coingecko, "__price__")
            self._tools = tools
            self._mapping = mapping
            logger.info("Loaded %d tools from %d MCP servers", len(tools), len(self.clients))
            return [dict(tool) for tool in tools]

    async def call(self, public_name: str, arguments: dict[str, Any]) -> str:
        if not self._mapping:
            await self.list_tools()
        target = self._mapping.get(public_name)
        if target is None:
            raise KeyError(public_name)
        client, original_name = target
        if original_name == "__price__":
            coin_id = str(arguments.get("coin_id") or "bitcoin").strip().lower()
            currencies = str(arguments.get("vs_currencies") or "usd,cny").strip().lower()
            if not re.fullmatch(r"[a-z0-9-]{1,80}", coin_id):
                raise ValueError("invalid CoinGecko coin id")
            if not re.fullmatch(r"[a-z0-9,-]{1,80}", currencies):
                raise ValueError("invalid quote currencies")
            code = (
                "async function run(client) { return await client.simple.price.get({ "
                f"ids: {json.dumps(coin_id)}, vs_currencies: {json.dumps(currencies)}"
                " }); }"
            )
            result = await client.request(
                "tools/call",
                {
                    "name": "execute",
                    "arguments": {"code": code, "intent": "Get current cryptocurrency price"},
                },
            )
            return _tool_output(result)
        result = await client.request(
            "tools/call", {"name": original_name, "arguments": arguments}
        )
        return _tool_output(result)

    async def warmup(self) -> None:
        if not self.enabled:
            return
        try:
            await self.list_tools(refresh=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP warmup failed: %s", exc)

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self.clients.values()), return_exceptions=True)
