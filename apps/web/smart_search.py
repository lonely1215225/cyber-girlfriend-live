"""Key-optional, normalized web search for the live-room agent.

The LLM never sees provider-specific response objects.  Providers are tried in
priority order to preserve free quota, with cache and a small circuit breaker
so an unhealthy endpoint does not delay every viewer question.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import socket
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import quote, urlparse

import httpx
from httpcore._backends.anyio import AnyIOBackend


logger = logging.getLogger("s2s.smart_search")
MAX_QUERY_CHARS = 500
MAX_SNIPPET_CHARS = 900
_SPACE_RE = re.compile(r"\s+")
_CGNAT = ipaddress.ip_network("100.64.0.0/10")
_SHARED_100 = ipaddress.ip_network("100.0.0.0/8")
_SEARCH_HOSTS = frozenset({"api.tavily.com", "api.exa.ai", "r.jina.ai"})
_orig_getaddrinfo = socket.getaddrinfo


def _usable_public_ipv4(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except (TypeError, ValueError):
        return False
    return ip.version == 4 and ip.is_global and ip not in _CGNAT and ip not in _SHARED_100


def _public_ipv4_infos(host: str, port: int, socktype: int, proto: int, flags: int):
    infos = _orig_getaddrinfo(host, port, socket.AF_INET, socktype, proto, flags)
    filtered = []
    seen: set[str] = set()
    for item in infos:
        text = str(item[4][0])
        if not _usable_public_ipv4(text) or text in seen:
            continue
        seen.add(text)
        filtered.append(item)
    return filtered


def first_public_ipv4(host: str) -> str:
    hostname = str(host or "").split("%", 1)[0].lower().rstrip(".")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        infos = _public_ipv4_infos(host, 443, socket.SOCK_STREAM, 0, 0)
        return str(infos[0][4][0]) if infos else ""
    return hostname if _usable_public_ipv4(hostname) else ""


def _search_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    hostname = str(host or "").split("%", 1)[0].lower().rstrip(".")
    socktype = type or socket.SOCK_STREAM
    if hostname in _SEARCH_HOSTS:
        filtered = _public_ipv4_infos(host, port, socktype, proto, flags)
        if filtered:
            return filtered
        raise socket.gaierror(socket.EAI_NONAME, f"no public IPv4 for {hostname}")
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


class PublicIPv4Backend(AnyIOBackend):
    """Connect search APIs to a reachable public IPv4. SNI stays on the hostname."""

    async def connect_tcp(
        self,
        host,
        port,
        timeout=None,
        local_address=None,
        socket_options=None,
    ):
        hostname = str(host or "").split("%", 1)[0].lower().rstrip(".")
        if hostname in _SEARCH_HOSTS:
            targets = [
                str(item[4][0])
                for item in _public_ipv4_infos(host, port, socket.SOCK_STREAM, 0, 0)
            ]
            if not targets:
                raise OSError(f"no public IPv4 for {host}")
        elif _usable_public_ipv4(hostname):
            targets = [hostname]
        else:
            return await super().connect_tcp(
                host, port, timeout, local_address, socket_options
            )

        last_exc: BaseException | None = None
        per_try = timeout
        if timeout is not None and len(targets) > 1:
            per_try = max(2.0, float(timeout) / len(targets))
        for target in targets:
            try:
                return await super().connect_tcp(
                    target, port, per_try, local_address, socket_options
                )
            except Exception as exc:
                last_exc = exc
                logger.warning("search IPv4 %s:%s failed: %s", target, port, exc)
        assert last_exc is not None
        raise last_exc


def _search_http_client(timeout_seconds: float) -> httpx.AsyncClient:
    transport = httpx.AsyncHTTPTransport(
        verify=True,
        trust_env=False,
        retries=1,
        limits=httpx.Limits(max_keepalive_connections=0, max_connections=8),
    )
    transport._pool._network_backend = PublicIPv4Backend()
    return httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(timeout_seconds, connect=min(8.0, timeout_seconds)),
        follow_redirects=True,
        trust_env=False,
        headers={"user-agent": "cyber-girlfriend-live/1.0", "accept": "application/json"},
    )


def install_public_ipv4_resolver() -> None:
    """Prefer reachable IPv4 for paid search APIs. CGNAT/IPv6 used to eat the budget."""
    socket.getaddrinfo = _search_getaddrinfo


install_public_ipv4_resolver()


@dataclass(frozen=True, slots=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str
    published_at: str = ""
    score: float = 0.0


class SmartSearchGateway:
    def __init__(self) -> None:
        self.tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
        self.exa_key = os.environ.get("EXA_API_KEY", "").strip()
        self.jina_key = os.environ.get("JINA_API_KEY", "").strip()
        self.searxng_url = os.environ.get("SEARXNG_URL", "").strip().rstrip("/")
        self.jina_reader_enabled = os.environ.get("JINA_READER_ENABLED", "1").lower() in {
            "1", "true", "yes", "on",
        }
        self.cache_seconds = max(15.0, float(os.environ.get("SMART_SEARCH_CACHE_SECONDS", "180")))
        self.timeout_seconds = max(6.0, float(os.environ.get("SMART_SEARCH_TIMEOUT_SECONDS", "12")))
        self.evidence_budget_seconds = max(
            2.0,
            min(self.timeout_seconds, float(os.environ.get("SMART_SEARCH_EVIDENCE_BUDGET_SECONDS", "8"))),
        )
        self.cooldown_seconds = max(10.0, float(os.environ.get("SMART_SEARCH_COOLDOWN_SECONDS", "20")))
        self._http = _search_http_client(self.timeout_seconds)
        self._cache: dict[str, tuple[float, str]] = {}
        self._health: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def _reset_http(self) -> None:
        old = self._http
        self._http = _search_http_client(self.timeout_seconds)
        try:
            await old.aclose()
        except Exception:
            pass

    @property
    def search_enabled(self) -> bool:
        return bool(self.tavily_key or self.exa_key or self.searxng_url)

    @property
    def fetch_enabled(self) -> bool:
        return self.jina_reader_enabled

    @property
    def enabled(self) -> bool:
        return self.search_enabled or self.fetch_enabled

    @property
    def provider_names(self) -> list[str]:
        names = []
        if self.tavily_key:
            names.append("tavily")
        if self.exa_key:
            names.append("exa")
        if self.searxng_url:
            names.append("searxng")
        return names

    @staticmethod
    def _clean(value: Any, limit: int) -> str:
        return _SPACE_RE.sub(" ", str(value or "")).strip()[:limit]

    @staticmethod
    def _valid_public_url(value: Any) -> str:
        url = str(value or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            return ""
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname == "localhost" or hostname.endswith(".local"):
            return ""
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return parsed._replace(fragment="").geturl()
        return "" if not address.is_global else parsed._replace(fragment="").geturl()

    def _available(self, provider: str, *, ignore_circuit: bool = False) -> bool:
        if ignore_circuit:
            return True
        failures, retry_at = self._health.get(provider, (0, 0.0))
        return failures < 2 or time.monotonic() >= retry_at

    def _succeeded(self, provider: str) -> None:
        self._health.pop(provider, None)

    def _failed(self, provider: str) -> None:
        failures, _ = self._health.get(provider, (0, 0.0))
        failures += 1
        retry_at = time.monotonic() + self.cooldown_seconds if failures >= 2 else 0.0
        self._health[provider] = (failures, retry_at)

    async def _tavily(self, query: str, topic: str, limit: int) -> list[SearchHit]:
        response = await self._http.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.tavily_key}"},
            json={
                "query": query,
                "topic": "news" if topic == "news" else "general",
                "search_depth": "basic",
                "max_results": limit,
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        response.raise_for_status()
        payload = response.json()
        return [
            SearchHit(
                title=self._clean(item.get("title"), 240),
                url=self._valid_public_url(item.get("url")),
                snippet=self._clean(item.get("content"), MAX_SNIPPET_CHARS),
                source="Tavily",
                published_at=self._clean(item.get("published_date"), 80),
                score=float(item.get("score") or 0.0),
            )
            for item in payload.get("results", []) if isinstance(item, dict)
        ]

    async def _exa(self, query: str, topic: str, limit: int) -> list[SearchHit]:
        payload: dict[str, Any] = {
            "query": query,
            "type": "auto",
            "numResults": limit,
            "contents": {"highlights": {"maxCharacters": MAX_SNIPPET_CHARS}},
        }
        if topic == "news":
            payload["category"] = "news"
        response = await self._http.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": self.exa_key},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        hits: list[SearchHit] = []
        for item in data.get("results", []):
            if not isinstance(item, dict):
                continue
            highlights = item.get("highlights") or []
            snippet = " ".join(str(part) for part in highlights if part)
            hits.append(SearchHit(
                title=self._clean(item.get("title"), 240),
                url=self._valid_public_url(item.get("url")),
                snippet=self._clean(snippet or item.get("text"), MAX_SNIPPET_CHARS),
                source="Exa",
                published_at=self._clean(item.get("publishedDate"), 80),
                score=float(item.get("score") or 0.0),
            ))
        return hits

    async def _searxng(self, query: str, topic: str, limit: int) -> list[SearchHit]:
        response = await self._http.get(
            f"{self.searxng_url}/search",
            params={
                "q": query, "format": "json", "language": "zh-CN",
                "categories": "news" if topic == "news" else "general",
                "safesearch": 1,
            },
        )
        response.raise_for_status()
        data = response.json()
        return [
            SearchHit(
                title=self._clean(item.get("title"), 240),
                url=self._valid_public_url(item.get("url")),
                snippet=self._clean(item.get("content"), MAX_SNIPPET_CHARS),
                source=f"SearXNG/{self._clean(item.get('engine'), 40) or 'web'}",
                published_at=self._clean(item.get("publishedDate"), 80),
                score=float(item.get("score") or 0.0),
            )
            for item in data.get("results", [])[:limit] if isinstance(item, dict)
        ]

    @staticmethod
    def _dedupe(hits: list[SearchHit], limit: int) -> list[SearchHit]:
        output: list[SearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            if not hit.url or not hit.title or not hit.snippet:
                continue
            key = hit.url.split("#", 1)[0].rstrip("/").lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(hit)
            if len(output) >= limit:
                break
        return output

    async def search(
        self,
        query: str,
        *,
        topic: str = "general",
        limit: int = 5,
        ignore_circuit: bool = False,
    ) -> str:
        query = self._clean(query, MAX_QUERY_CHARS)
        if not query:
            raise ValueError("search query is empty")
        topic = "news" if topic == "news" else "general"
        limit = max(1, min(8, int(limit)))
        cache_key = json.dumps([query.lower(), topic, limit], ensure_ascii=False)
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] <= self.cache_seconds:
            return cached[1]

        providers: list[tuple[str, Callable[[], Awaitable[list[SearchHit]]]]] = []
        if self.tavily_key:
            providers.append(("tavily", lambda: self._tavily(query, topic, limit)))
        if self.exa_key:
            providers.append(("exa", lambda: self._exa(query, topic, limit)))
        if self.searxng_url:
            providers.append(("searxng", lambda: self._searxng(query, topic, limit)))
        if not providers:
            raise RuntimeError("智能搜索尚未配置 Tavily、Exa 或 SearXNG")

        errors: list[str] = []
        hits: list[SearchHit] = []
        used: list[str] = []
        for name, provider in providers:
            if not self._available(name, ignore_circuit=ignore_circuit):
                errors.append(f"{name}: circuit open")
                continue
            try:
                result = self._dedupe(await provider(), limit)
                if result:
                    self._succeeded(name)
                    hits.extend(result)
                    used.append(name)
                    logger.info("smart search provider=%s results=%d topic=%s", name, len(result), topic)
                    # One healthy provider is enough. This avoids spending two
                    # free quotas for every ordinary viewer question.
                    break
                raise RuntimeError("no relevant results")
            except (httpx.HTTPError, ValueError, RuntimeError, KeyError, OSError) as exc:
                self._failed(name)
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                logger.warning("smart search provider=%s failed: %s: %s", name, type(exc).__name__, exc)
                if isinstance(exc, (httpx.ConnectTimeout, httpx.ConnectError, httpx.NetworkError, OSError)):
                    await self._reset_http()

        hits = self._dedupe(hits, limit)
        if not hits:
            raise RuntimeError("; ".join(errors) or "all search providers failed")
        output = json.dumps({
            "query": query,
            "topic": topic,
            "sources": used,
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results": [asdict(hit) for hit in hits],
        }, ensure_ascii=False, separators=(",", ":"))
        async with self._lock:
            self._cache[cache_key] = (now, output)
            if len(self._cache) > 128:
                oldest = min(self._cache, key=lambda key: self._cache[key][0])
                self._cache.pop(oldest, None)
        return output

    async def search_all(self, query: str, *, topic: str = "news", limit: int = 6) -> str:
        """Query independent providers concurrently for evidence-sensitive Agent tasks."""
        query = self._clean(query, MAX_QUERY_CHARS)
        if not query:
            raise ValueError("search query is empty")
        topic = "news" if topic == "news" else "general"
        limit = max(2, min(8, int(limit)))
        cache_key = json.dumps(["all", query.lower(), topic, limit], ensure_ascii=False)
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] <= self.cache_seconds:
            return cached[1]

        providers: list[tuple[str, Callable[[], Awaitable[list[SearchHit]]]]] = []
        if self.tavily_key and self._available("tavily"):
            providers.append(("tavily", lambda: self._tavily(query, topic, limit)))
        if self.exa_key and self._available("exa"):
            providers.append(("exa", lambda: self._exa(query, topic, limit)))
        if self.searxng_url and self._available("searxng"):
            providers.append(("searxng", lambda: self._searxng(query, topic, limit)))
        if not providers:
            return await self.search(query, topic=topic, limit=limit)

        tasks = [asyncio.create_task(provider(), name=name) for name, provider in providers]
        done, pending = await asyncio.wait(tasks, timeout=self.evidence_budget_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        outcomes_by_name: dict[str, Any] = {}
        for task in done:
            try:
                outcomes_by_name[task.get_name()] = task.result()
            except Exception as exc:
                outcomes_by_name[task.get_name()] = exc
        for task in pending:
            outcomes_by_name[task.get_name()] = TimeoutError("provider exceeded evidence budget")
        hits: list[SearchHit] = []
        used: list[str] = []
        provider_hits: list[tuple[str, list[SearchHit]]] = []
        errors: list[str] = []
        for name, _ in providers:
            outcome = outcomes_by_name.get(name, RuntimeError("provider did not complete"))
            if isinstance(outcome, Exception):
                self._failed(name)
                errors.append(f"{name}: {type(outcome).__name__}: {outcome}")
                continue
            result = self._dedupe(outcome, limit)
            if result:
                self._succeeded(name)
                provider_hits.append((name, result))
            else:
                self._failed(name)
                errors.append(f"{name}: no relevant results")
        # Round-robin keeps one prolific provider from pushing every result
        # from the other independent sources past the global limit.
        for index in range(limit):
            for name, bucket in provider_hits:
                if index < len(bucket):
                    hits.append(bucket[index])
        hits = self._dedupe(hits, limit)
        if not hits:
            raise RuntimeError("; ".join(errors) or "all search providers failed")
        used = list(dict.fromkeys(
            name for name, bucket in provider_hits
            if any(hit in hits for hit in bucket)
        ))
        output = json.dumps({
            "query": query, "topic": topic, "sources": used,
            "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results": [asdict(hit) for hit in hits],
        }, ensure_ascii=False, separators=(",", ":"))
        async with self._lock:
            self._cache[cache_key] = (now, output)
        logger.info("smart search parallel providers=%s results=%d", ",".join(used), len(hits))
        return output

    async def fetch(self, url: str, *, max_chars: int = 5000) -> str:
        if not self.fetch_enabled:
            raise RuntimeError("Jina Reader is disabled")
        safe_url = self._valid_public_url(url)
        if not safe_url:
            raise ValueError("only public HTTP(S) URLs can be fetched")
        # Resolve locally as a second SSRF guard before giving the URL to a
        # remote reader. Resolution failure is allowed because Jina may still
        # reach a temporarily unavailable-to-us public hostname.
        try:
            addresses = await asyncio.to_thread(socket.getaddrinfo, urlparse(safe_url).hostname, 443)
            for item in addresses:
                if not ipaddress.ip_address(item[4][0]).is_global:
                    raise ValueError("private network URLs are not allowed")
        except socket.gaierror:
            pass
        headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
        if self.jina_key:
            headers["Authorization"] = f"Bearer {self.jina_key}"
        response = await self._http.get(f"https://r.jina.ai/{quote(safe_url, safe=':/?&=%#')}", headers=headers)
        response.raise_for_status()
        clean = response.text.strip()
        if not clean:
            raise RuntimeError("Jina Reader returned no content")
        title_match = re.search(r"^Title:\s*(.+)$", clean, re.M)
        published_match = re.search(r"^Published Time:\s*(.+)$", clean, re.M)
        body_match = re.search(r"^Markdown Content:\s*(.*)$", clean, re.M | re.S)
        body = body_match.group(1) if body_match else clean
        body = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", body)
        body = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", body)
        body = re.sub(r"https?://\S+", "", body)
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        limit = max(1000, min(12000, int(max_chars)))
        return json.dumps({
            "source": urlparse(safe_url).hostname or "web",
            "title": self._clean(title_match.group(1) if title_match else "", 240),
            "published_at": self._clean(published_match.group(1) if published_match else "", 80),
            "content": body[:limit],
        }, ensure_ascii=False, separators=(",", ":"))

    async def close(self) -> None:
        await self._http.aclose()


def format_hits_for_speech(raw: str) -> str:
    """Turn provider JSON into short Chinese evidence the host can read."""
    payload = json.loads(raw)
    lines = ["刚才查到的资料："]
    for index, item in enumerate(payload.get("results") or [], start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        source = str(item.get("source") or "网页")[:40]
        snippet = str(item.get("snippet") or "").strip()[:90]
        lines.append(f"{index}. {title}（{source}）")
        if snippet:
            lines.append(f"   {snippet}")
    if len(lines) < 2:
        raise RuntimeError("搜索没有返回可说的结果")
    return "\n".join(lines)
