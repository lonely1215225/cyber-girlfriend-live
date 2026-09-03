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

from dialogue_intent import is_news_request, news_search_query
from rss_news import RssNewsAggregator
from smart_search import SmartSearchGateway, format_hits_for_speech


logger = logging.getLogger("s2s.mcp")
PROTOCOL_VERSION = "2025-03-26"
MAX_TOOL_OUTPUT_CHARS = max(1000, int(os.environ.get("MCP_MAX_OUTPUT_CHARS", "6000")))
DEFAULT_TOOL_ALLOWLIST = {
    "coingecko:execute",
    "coingecko:search_docs",
    "exa:web_search_exa",
    "exa:web_fetch_exa",
    "gdelt:gdelt_search_articles",
    "tavily:tavily-search",
    "tavily:tavily-extract",
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
    DISCOVERY_TOOL_NAME = "request_external_capabilities"
    DIALOGUE_WEB_TOOL_NAME = "smart_web_search"

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
            tavily_url = os.environ.get("MCP_TAVILY_URL", "").strip()
            if tavily_url:
                configs.append(McpServerConfig("tavily", "Tavily", tavily_url))
        self.clients = {config.key: McpHttpClient(config) for config in configs if config.url}
        self._tools: list[dict[str, Any]] = []
        self._mapping: dict[str, tuple[McpHttpClient, str]] = {}
        self._tools_lock = asyncio.Lock()
        self._price_http = httpx.AsyncClient(
            base_url="https://api.coingecko.com/api/v3",
            timeout=httpx.Timeout(5.0, connect=1.5),
            headers={"accept": "application/json", "user-agent": "cyber-girlfriend-live/1.0"},
        )
        self.rss_news = RssNewsAggregator()
        self.smart_search = SmartSearchGateway()

    @property
    def enabled(self) -> bool:
        return bool(self.clients) or self.rss_news.enabled or self.smart_search.enabled

    @classmethod
    def discovery_tool(cls) -> dict[str, Any]:
        """Expose one small capability request instead of every remote schema.

        The model still decides whether external data is necessary.  This is
        progressive tool disclosure: ordinary conversation does not pay for,
        or get distracted by, the full remote tool registry.
        """
        return {
            "type": "function",
            "name": cls.DISCOVERY_TOOL_NAME,
            "description": (
                "Call this only when the user needs current, external, private-document, page, market, "
                "news, or vision information. Never call it for greetings, casual conversation, "
                "emotional support, roleplay, opinions, or questions answerable from conversation context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capabilities": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["web", "news", "market", "page", "knowledge", "vision"],
                        },
                        "minItems": 1,
                        "maxItems": 3,
                        "description": (
                            "web=current products, vehicles, people, facts or general internet lookup; "
                            "news=recent events/articles; market=cryptocurrency price only; "
                            "page=read a known URL; knowledge=private MCP/documents; vision=webcam."
                        ),
                    }
                },
                "required": ["capabilities"],
            },
            "source": "tool-discovery",
        }

    def dialogue_web_tool(self) -> dict[str, Any] | None:
        """The only tool chat and live voice may see: live web search.

        Do not call ``list_tools()`` here. That path talks to every configured
        MCP server (prices, news, fetch, custom docs) and is what made casual
        ``@小麻`` replies stall or look like a busy channel.
        """
        if not self.smart_search.search_enabled:
            return None
        return {
            "type": "function",
            "name": self.DIALOGUE_WEB_TOOL_NAME,
            "description": (
                "查询当前互联网并返回经过清洗的结构化结果。"
                "仅在用户明确要求查网，或问题必须依赖最新网上事实时调用。"
                "普通闲聊、情绪、吐槽、承接上下文不要调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "具体、完整的查询问题。"},
                    "topic": {
                        "type": "string", "enum": ["general", "news"],
                        "default": "general",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
                },
                "required": ["query"],
            },
            "source": "smart-search",
            "progress_text": "我去看一眼，马上回你。",
        }

    async def tools_for_capabilities(self, capabilities: list[str]) -> list[dict[str, Any]]:
        """Return a compact, non-duplicated tool set for model-selected capabilities."""
        requested = {str(item).strip().lower() for item in capabilities}
        tools = await self.list_tools()
        by_name = {str(tool.get("name") or ""): tool for tool in tools}
        selected: list[dict[str, Any]] = []

        def add(name: str) -> None:
            tool = by_name.get(name)
            if tool is not None and tool not in selected:
                selected.append(tool)

        if requested & {"web", "news"}:
            add("smart_web_search")
        if "news" in requested:
            add("local_rss_news")
            for tool in tools:
                if str(tool.get("source") or "") == "gdelt" and tool not in selected:
                    selected.append(tool)
        if "market" in requested:
            add("mcp_coingecko_price")
        if "page" in requested:
            add("smart_web_fetch")
        if "knowledge" in requested:
            # Unknown/custom MCP servers are document or domain capabilities.
            # Exa and CoinGecko raw tools are intentionally omitted because
            # their compact adapters above provide the same operation.
            for tool in tools:
                source = str(tool.get("source") or "")
                if source not in {"coingecko", "exa", "gdelt", "rss", "smart-search", "jina-reader"}:
                    if tool not in selected:
                        selected.append(tool)
        return [dict(tool) for tool in selected]

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
                            "progress_text": "我去看一眼，马上回你。",
                        }
                    )
                    mapping[public_name] = (client, original_name)
            if self.rss_news.enabled:
                tools.append(
                    {
                        "type": "function",
                        "name": "local_rss_news",
                        "description": (
                            "查询直播间的实时中文 RSS 资讯池。用户询问最新新闻、科技资讯、"
                            "知识话题，或指定 iDaily、中新网、澎湃、人民日报、极客公园、"
                            "cnBeta、IT之家、知乎日报时必须调用。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": ["新闻", "科技", "知识", "全部"],
                                    "description": "用户指定的资讯类别。",
                                },
                                "source": {
                                    "type": "string",
                                    "description": "用户指定的来源名称；未指定则留空。",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "用户原始问题或希望筛选的关键词。",
                                },
                                "limit": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 8,
                                    "default": 5,
                                },
                            },
                            "required": ["query"],
                        },
                        "source": "rss",
                        "progress_text": "我翻一下今天的，马上说。",
                    }
                )
            if self.smart_search.search_enabled:
                tools.append(
                    {
                        "type": "function",
                        "name": "smart_web_search",
                        "description": (
                            "查询当前互联网并返回经过清洗的结构化结果。适用于实时事实、网页资料、"
                            "非 RSS 新闻和需要联网核实的问题；不要用它查询币价。"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "具体、完整的查询问题。"},
                                "topic": {
                                    "type": "string", "enum": ["general", "news"],
                                    "default": "general",
                                },
                                "limit": {"type": "integer", "minimum": 1, "maximum": 8, "default": 5},
                            },
                            "required": ["query"],
                        },
                        "source": "smart-search",
                        "progress_text": "我去看一眼，马上回你。",
                    }
                )
            if self.smart_search.fetch_enabled:
                tools.append(
                    {
                        "type": "function",
                        "name": "smart_web_fetch",
                        "description": "读取一个公开网页的正文。仅在搜索摘要不足以回答时使用。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string", "description": "搜索结果给出的公开 HTTP(S) URL。"},
                                "max_chars": {
                                    "type": "integer", "minimum": 1000, "maximum": 12000, "default": 5000,
                                },
                            },
                            "required": ["url"],
                        },
                        "source": "jina-reader",
                        "progress_text": "我去打开看一眼。",
                    }
                )
            # CoinGecko's official server exposes a generic `execute` tool.
            # This capability adapter supplies its required TypeScript wrapper
            # while leaving the decision to use it entirely to the model.
            coingecko = self.clients.get("coingecko")
            if coingecko:
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
                        "progress_text": "我去对一下最新的数。",
                    }
                )
                mapping[price_name] = (coingecko, "__price__")
            self._tools = tools
            self._mapping = mapping
            logger.info("Loaded %d tools from %d MCP servers", len(tools), len(self.clients))
            return [dict(tool) for tool in tools]

    @staticmethod
    def _spoken_search_output(raw: str) -> str:
        try:
            return format_hits_for_speech(raw)
        except (json.JSONDecodeError, RuntimeError, TypeError, ValueError, KeyError):
            return raw

    async def prefetch_spoken_evidence(self, query: str) -> str:
        """Fetch Chinese headlines or search hits before the host starts talking."""
        query = str(query or "").strip()
        if not query:
            raise ValueError("search query is empty")
        news = is_news_request(query) or bool(re.search(r"新闻|热搜|资讯|头条", query))
        lookup_query = news_search_query(query) or query
        errors: list[str] = []
        if self.smart_search.search_enabled:
            try:
                topic = "news" if news else "general"
                raw = await self.smart_search.search(
                    lookup_query, topic=topic, limit=5, ignore_circuit=True
                )
                return self._spoken_search_output(raw)
            except Exception as exc:
                logger.warning("smart search prefetch failed: %s", exc)
                errors.append(str(exc))
        if news and self.rss_news.enabled and not self.smart_search.search_enabled:
            try:
                return await self.rss_news.spoken_brief(query, limit=4)
            except Exception as exc:
                logger.warning("RSS spoken brief failed: %s", exc)
                errors.append(str(exc))
        raise RuntimeError("; ".join(errors) or "没有可用的查询来源")

    async def call(self, public_name: str, arguments: dict[str, Any]) -> str:
        if public_name == "smart_web_search":
            query = str(arguments.get("query") or "")
            topic = str(arguments.get("topic") or "general")
            if topic != "news" and re.search(r"新闻|热搜|资讯", query):
                topic = "news"
            limit = int(arguments.get("limit") or 5)
            try:
                raw = await self.smart_search.search(query, topic=topic, limit=limit)
                return self._spoken_search_output(raw)
            except Exception as exc:
                if self.rss_news.enabled:
                    logger.warning("smart_web_search falling back to RSS: %s", exc)
                    spoken = getattr(self.rss_news, "spoken_brief", None)
                    if callable(spoken) and (
                        is_news_request(query) or re.search(r"新闻|热搜|资讯", query)
                    ):
                        try:
                            return await spoken(query, limit=limit)
                        except Exception:
                            pass
                    return await self.rss_news.query_topics(query=query, limit=limit)
                raise
        if public_name == "smart_web_fetch":
            return await self.smart_search.fetch(
                str(arguments.get("url") or ""),
                max_chars=int(arguments.get("max_chars") or 5000),
            )
        if public_name == "local_rss_news":
            category = str(arguments.get("category") or "").strip()
            if category == "全部":
                category = ""
            return await self.rss_news.query_topics(
                category=category,
                source=str(arguments.get("source") or ""),
                query=str(arguments.get("query") or ""),
                limit=int(arguments.get("limit") or 5),
            )
        if not self._mapping:
            await self.list_tools()
        target = self._mapping.get(public_name)
        if target is None:
            raise KeyError(public_name)
        client, original_name = target
        if original_name == "__price__":
            coin_id = str(arguments.get("coin_id") or "").strip().lower()
            if not coin_id:
                return json.dumps(
                    {
                        "ok": False,
                        "error": "missing_required_argument",
                        "message": "coin_id is required; infer it from the user's actual subject before retrying",
                    },
                    ensure_ascii=False,
                )
            currencies = str(arguments.get("vs_currencies") or "usd,cny").strip().lower()
            if not re.fullmatch(r"[a-z0-9-]{1,80}", coin_id):
                raise ValueError("invalid CoinGecko coin id")
            if not re.fullmatch(r"[a-z0-9,-]{1,80}", currencies):
                raise ValueError("invalid quote currencies")
            async def fetch_rest() -> str:
                response = await self._price_http.get(
                    "/simple/price",
                    params={"ids": coin_id, "vs_currencies": currencies},
                )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict) and isinstance(payload.get(coin_id), dict):
                    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                raise RuntimeError("CoinGecko REST returned no requested price")

            code = (
                "async function run(client) { return await client.simple.price.get({ "
                f"ids: {json.dumps(coin_id)}, vs_currencies: {json.dumps(currencies)}"
                " }); }"
            )

            async def fetch_mcp() -> str:
                result = await client.request(
                    "tools/call",
                    {
                        "name": "execute",
                        "arguments": {"code": code, "intent": "Get current cryptocurrency price"},
                    },
                )
                output = _tool_output(result)
                if not output or output.lower().startswith("mcp tool error:"):
                    raise RuntimeError(output or "CoinGecko MCP returned no price")
                return output

            # Race CoinGecko's narrow REST endpoint and its official MCP
            # endpoint. Network conditions vary; waiting for one to fail before
            # starting the other made their latency add up.
            pending = {asyncio.create_task(fetch_rest()), asyncio.create_task(fetch_mcp())}
            errors: list[BaseException] = []
            try:
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    winner = ""
                    for task in done:
                        try:
                            winner = winner or task.result()
                        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
                            errors.append(exc)
                    if winner:
                        return winner
                raise RuntimeError("; ".join(str(error) or type(error).__name__ for error in errors))
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
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
        await asyncio.gather(
            *(client.close() for client in self.clients.values()),
            self._price_http.aclose(),
            self.smart_search.close(),
            return_exceptions=True,
        )
