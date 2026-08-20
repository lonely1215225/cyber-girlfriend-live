"""Key-free RSS/Atom aggregation for current-news grounding."""

from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus, urlparse

import httpx


logger = logging.getLogger("s2s.rss_news")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}|[\u4e00-\u9fff]{2,}")
DEFAULT_FEEDS = (
    ("UN News", "https://news.un.org/feed/subscribe/en/news/all/rss.xml"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("NPR World", "https://feeds.npr.org/1004/rss.xml"),
    ("DW World", "https://rss.dw.com/rdf/rss-en-world"),
)
ALLOWED_FEED_HOSTS = {
    "news.google.com",
    "news.un.org",
    "www.aljazeera.com",
    "feeds.npr.org",
    "rss.dw.com",
}
ALLOWED_FEED_HOSTS.update(
    host.strip().lower()
    for host in os.environ.get("NEWS_RSS_ALLOWED_HOSTS", "").split(",")
    if host.strip()
)


def _clean(value: str, limit: int) -> str:
    text = html.unescape(_TAG_RE.sub(" ", value or ""))
    return _SPACE_RE.sub(" ", text).strip()[:limit]


def _child_text(element: ET.Element, names: tuple[str, ...]) -> str:
    for child in list(element):
        local_name = child.tag.rsplit("}", 1)[-1].lower()
        if local_name in names and child.text:
            return child.text.strip()
    return ""


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class NewsItem:
    title: str
    link: str
    source: str
    published: datetime | None
    summary: str = ""
    query_match: bool = False


def formatted_news_blocks(output: str) -> list[str]:
    """Extract compact attributed story blocks from formatted RSS output."""
    starts = list(re.finditer(r"(?m)^\d+\.\s+", output))
    blocks: list[str] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(output)
        block = output[match.end() : end].strip()
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        lines = [line for line in lines if not line.startswith("原文：")]
        if lines:
            blocks.append("\n".join(lines)[:1200])
    return blocks


class IdleNewsRotator:
    """Choose a different recent headline for each active caller."""

    def __init__(self, history_size: int = 12, max_audiences: int = 256) -> None:
        self.history_size = max(2, history_size)
        self.max_audiences = max(8, max_audiences)
        self._recent: OrderedDict[str, deque[str]] = OrderedDict()

    @staticmethod
    def _identity(block: str) -> str:
        headline = block.splitlines()[0] if block else ""
        return re.sub(r"\W+", "", headline).lower()

    def choose(self, audience: str, output: str) -> str:
        blocks = formatted_news_blocks(output)
        if not blocks:
            raise ValueError("RSS output contains no news stories")
        recent = self._recent.setdefault(audience, deque(maxlen=self.history_size))
        self._recent.move_to_end(audience)
        if len(self._recent) > self.max_audiences:
            self._recent.popitem(last=False)
        selected = next((block for block in blocks if self._identity(block) not in recent), None)
        if selected is None:
            recent.clear()
            selected = blocks[0]
        recent.append(self._identity(selected))
        return selected


def parse_feed(payload: bytes, fallback_source: str, *, query_match: bool = False) -> list[NewsItem]:
    if len(payload) > 1_500_000:
        raise ValueError("RSS response is too large")
    root = ET.fromstring(payload)
    items: list[NewsItem] = []
    for element in root.iter():
        local_name = element.tag.rsplit("}", 1)[-1].lower()
        if local_name not in {"item", "entry"}:
            continue
        title = _clean(_child_text(element, ("title",)), 240)
        if not title:
            continue
        link = _child_text(element, ("link",))
        if not link:
            for child in list(element):
                if child.tag.rsplit("}", 1)[-1].lower() == "link" and child.attrib.get("href"):
                    link = child.attrib["href"]
                    break
        source = _clean(_child_text(element, ("source",)), 80) or fallback_source
        summary = _clean(_child_text(element, ("description", "summary", "content")), 420)
        published = _parse_time(
            _child_text(element, ("pubdate", "published", "updated", "date"))
        )
        items.append(
            NewsItem(
                title=title,
                link=link[:600],
                source=source,
                published=published,
                summary=summary,
                query_match=query_match,
            )
        )
    return items


def _configured_feeds() -> tuple[tuple[str, str], ...]:
    raw = os.environ.get("NEWS_RSS_FEEDS", "").strip()
    if not raw:
        return DEFAULT_FEEDS
    feeds: list[tuple[str, str]] = []
    for index, entry in enumerate(raw.split(";"), start=1):
        label, separator, url = entry.strip().partition("=")
        if not separator:
            url, label = label, f"RSS {index}"
        parsed = urlparse(url.strip())
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_FEED_HOSTS:
            logger.warning("Ignoring unapproved NEWS_RSS_FEEDS URL: %s", url)
            continue
        feeds.append((_clean(label, 50), url.strip()))
    return tuple(feeds) or DEFAULT_FEEDS


class RssNewsAggregator:
    def __init__(self) -> None:
        self.enabled = os.environ.get("NEWS_RSS_ENABLED", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.google_enabled = os.environ.get("NEWS_GOOGLE_RSS_ENABLED", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.timeout = max(3.0, float(os.environ.get("NEWS_RSS_TIMEOUT", "8")))
        self.cache_seconds = max(30.0, float(os.environ.get("NEWS_RSS_CACHE_SECONDS", "120")))
        self.max_items = max(3, min(20, int(os.environ.get("NEWS_RSS_MAX_ITEMS", "10"))))
        self.max_age_hours = max(1, int(os.environ.get("NEWS_RSS_MAX_AGE_HOURS", "168")))
        self.feeds = _configured_feeds()
        self._cache: dict[str, tuple[float, str]] = {}

    async def _fetch(
        self, client: httpx.AsyncClient, label: str, url: str, query_match: bool
    ) -> list[NewsItem]:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 1_500_000:
                    raise ValueError(f"{label} RSS response is too large")
                chunks.append(chunk)
        return parse_feed(b"".join(chunks), label, query_match=query_match)

    async def search(self, query: str) -> str:
        if not self.enabled:
            raise RuntimeError("RSS news is disabled")
        cache_key = _SPACE_RE.sub(" ", query).strip().lower()
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and now - cached[0] <= self.cache_seconds:
            return cached[1]

        generic_headlines_request = bool(re.search(r"(?:今日|今天|最新)?(?:国际|全球|世界)?新闻", query))
        # A topic query is best served by the query feed alone. Fixed headline
        # feeds add latency and unrelated stories; they are useful for a broad
        # "today's world news" request.
        sources = (
            [(label, url, False) for label, url in self.feeds]
            if generic_headlines_request or not self.google_enabled
            else []
        )
        if self.google_enabled:
            google_url = (
                "https://news.google.com/rss/search?q="
                f"{quote_plus(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            )
            sources.insert(0, ("Google News", google_url, True))
        headers = {"User-Agent": "CyberGirlfriendLive/1.0 RSS news reader"}
        timeout = httpx.Timeout(self.timeout, connect=min(5.0, self.timeout))
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            results = await asyncio.gather(
                *(self._fetch(client, label, url, query_match) for label, url, query_match in sources),
                return_exceptions=True,
            )

        items: list[NewsItem] = []
        for (label, _, _), result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning("RSS source %s failed: %s", label, result)
            else:
                items.extend(result)
        if not items:
            raise RuntimeError("所有 RSS 新闻源均不可用")
        has_query_results = any(item.query_match for item in items)
        if self.google_enabled and not has_query_results and not generic_headlines_request:
            raise RuntimeError("查询型 RSS 没有返回相关结果")

        query_tokens = {token.lower() for token in _TOKEN_RE.findall(query)}
        oldest_timestamp = datetime.now(timezone.utc).timestamp() - self.max_age_hours * 3600
        timely_items = [
            item
            for item in items
            if item.published is None or item.published.timestamp() >= oldest_timestamp
        ]
        if timely_items:
            items = timely_items

        def score(item: NewsItem) -> tuple[int, float]:
            text = f"{item.title} {item.summary}".lower()
            overlap = sum(1 for token in query_tokens if token in text)
            timestamp = item.published.timestamp() if item.published else 0.0
            return (100 if item.query_match else 0) + overlap * 10, timestamp

        unique: list[NewsItem] = []
        seen: set[str] = set()
        source_counts: dict[str, int] = {}
        for item in sorted(items, key=score, reverse=True):
            identity = re.sub(r"\W+", "", item.title).lower()
            source_key = item.source.strip().lower()
            if not identity or identity in seen or source_counts.get(source_key, 0) >= 2:
                continue
            seen.add(identity)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            unique.append(item)
            if len(unique) >= self.max_items:
                break

        lines = ["RSS 聚合新闻（仅含标题、摘要与原文链接，按相关性和时间排序）："]
        for index, item in enumerate(unique, start=1):
            published = item.published.strftime("%Y-%m-%d %H:%M UTC") if item.published else "时间未提供"
            lines.append(f"{index}. [{item.source}] {published} — {item.title}")
            title_key = re.sub(r"\W+", "", item.title).lower()
            summary_key = re.sub(r"\W+", "", item.summary).lower()
            if item.summary and summary_key not in title_key and title_key not in summary_key:
                lines.append(f"   摘要：{item.summary}")
            if item.link:
                lines.append(f"   原文：{item.link}")
        output = "\n".join(lines)[:6000]
        self._cache[cache_key] = (now, output)
        if len(self._cache) > 64:
            oldest = min(self._cache, key=lambda item: self._cache[item][0])
            self._cache.pop(oldest, None)
        return output
