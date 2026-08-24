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
_CATEGORY_KEYWORDS = {
    "科技": ("科技", "数码", "互联网", "人工智能", "ai新闻", "it新闻"),
    "知识": ("知识", "科普", "知乎", "知乎日报"),
    "新闻": (
        "国际新闻", "国内新闻", "全球新闻", "世界新闻", "时事", "热点新闻",
        "体育新闻", "财经新闻", "金融新闻", "社会新闻", "娱乐新闻", "军事新闻",
    ),
}
_SOURCE_ALIASES = {
    "iDaily 每日环球视野": ("idaily", "每日环球", "每日环球视野"),
    "中国新闻网·国际新闻": ("中新网", "中国新闻网"),
    "澎湃新闻": ("澎湃", "澎湃新闻"),
    "人民日报": ("人民日报", "人民网"),
    "极客公园": ("极客公园",),
    "cnBeta": ("cnbeta",),
    "IT之家": ("it之家",),
    "知乎日报": ("知乎", "知乎日报"),
}
DEFAULT_FEEDS = (
    ("新闻", "iDaily 每日环球视野", "https://plink.anyfeeder.com/idaily/today"),
    ("新闻", "中国新闻网·国际新闻", "https://plink.anyfeeder.com/newscn/whxw"),
    ("新闻", "澎湃新闻", "https://plink.anyfeeder.com/thepaper"),
    ("新闻", "人民日报", "https://plink.anyfeeder.com/people-daily"),
    ("科技", "极客公园", "https://plink.anyfeeder.com/geekpark"),
    ("科技", "cnBeta", "https://plink.anyfeeder.com/cnbeta"),
    ("科技", "IT之家", "https://plink.anyfeeder.com/ithome/it"),
    ("知识", "知乎日报", "https://plink.anyfeeder.com/zhihu/daily"),
)
ALLOWED_FEED_HOSTS = {
    "news.google.com",
    "plink.anyfeeder.com",
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
    category: str = "新闻"


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


def infer_topic_filters(text: str) -> tuple[str, str]:
    """Infer an explicitly requested category/source without treating every story as generic news."""
    lowered = text.lower()
    source = next(
        (
            label
            for label, aliases in _SOURCE_ALIASES.items()
            if any(alias.lower() in lowered for alias in aliases)
        ),
        "",
    )
    category = next(
        (
            name
            for name, keywords in _CATEGORY_KEYWORDS.items()
            if any(keyword.lower() in lowered for keyword in keywords)
        ),
        "",
    )
    if not category and source:
        category = next(
            (feed_category for feed_category, label, _ in DEFAULT_FEEDS if label == source),
            "",
        )
    return category, source


def _relevance_terms(query: str) -> set[str]:
    """Extract useful Chinese/Latin terms without requiring a tokenizer dependency."""
    lowered = query.lower()
    for phrase in (
        "帮我", "给我", "查一下", "查询", "看看", "讲讲", "说说", "有没有",
        "有什么", "我想听", "我想看", "今天", "今日", "最近", "当前", "现在",
        "最新", "相关", "方面", "内容", "新闻", "资讯", "热点", "日报",
    ):
        lowered = lowered.replace(phrase, " ")
    for aliases in _SOURCE_ALIASES.values():
        for alias in aliases:
            lowered = lowered.replace(alias.lower(), " ")
    terms: set[str] = set()
    for token in _TOKEN_RE.findall(lowered):
        token = token.strip().lower()
        if not token:
            continue
        terms.add(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{3,}", token):
            terms.update(token[index : index + 2] for index in range(len(token) - 1))
    return terms


class IdleNewsRotator:
    """Rotate categories and avoid recently spoken stories per audience."""

    CATEGORY_ORDER = ("新闻", "科技", "知识")

    def __init__(self, history_size: int = 96, max_audiences: int = 256) -> None:
        self.history_size = max(2, history_size)
        self.max_audiences = max(8, max_audiences)
        self._recent: OrderedDict[str, deque[str]] = OrderedDict()
        self._category_index: dict[str, int] = {}

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
            evicted, _ = self._recent.popitem(last=False)
            self._category_index.pop(evicted, None)

        start = self._category_index.get(audience, 0)
        categories = self.CATEGORY_ORDER[start:] + self.CATEGORY_ORDER[:start]
        selected = None
        selected_category = ""
        for category in categories:
            marker = f"[{category}｜"
            selected = next(
                (
                    block
                    for block in blocks
                    if marker in block and self._identity(block) not in recent
                ),
                None,
            )
            if selected is not None:
                selected_category = category
                break
        if selected is None:
            selected = next((block for block in blocks if self._identity(block) not in recent), None)
        if selected is None:
            recent.clear()
            selected = blocks[0]
        recent.append(self._identity(selected))
        if selected_category:
            category_index = self.CATEGORY_ORDER.index(selected_category)
            self._category_index[audience] = (category_index + 1) % len(self.CATEGORY_ORDER)
        return selected


def parse_feed(
    payload: bytes,
    fallback_source: str,
    *,
    query_match: bool = False,
    category: str = "新闻",
) -> list[NewsItem]:
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
                category=category,
            )
        )
    return items


def _configured_feeds() -> tuple[tuple[str, str, str], ...]:
    raw = os.environ.get("NEWS_RSS_FEEDS", "").strip()
    if not raw:
        return DEFAULT_FEEDS
    feeds: list[tuple[str, str, str]] = []
    for index, entry in enumerate(raw.split(";"), start=1):
        label, separator, url = entry.strip().partition("=")
        if not separator:
            url, label = label, f"RSS {index}"
        parsed = urlparse(url.strip())
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_FEED_HOSTS:
            logger.warning("Ignoring unapproved NEWS_RSS_FEEDS URL: %s", url)
            continue
        feeds.append(("自定义", _clean(label, 50), url.strip()))
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
        self.max_age_hours = max(1, int(os.environ.get("NEWS_RSS_MAX_AGE_HOURS", "72")))
        self.feeds = _configured_feeds()
        self._cache: dict[str, tuple[float, str]] = {}

    async def _fetch(
        self,
        client: httpx.AsyncClient,
        category: str,
        label: str,
        url: str,
        query_match: bool,
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
        return parse_feed(
            b"".join(chunks), label, query_match=query_match, category=category
        )

    async def _fetch_many(
        self, sources: list[tuple[str, str, str, bool]]
    ) -> list[NewsItem]:
        """Fetch with bounded concurrency so one host/network limit cannot sink the batch."""
        headers = {"User-Agent": "CyberGirlfriendLive/1.0 RSS news reader"}
        timeout = httpx.Timeout(self.timeout, connect=min(5.0, self.timeout))
        semaphore = asyncio.Semaphore(2)
        limits = httpx.Limits(max_connections=3, max_keepalive_connections=2)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers, limits=limits
        ) as client:
            async def guarded(source: tuple[str, str, str, bool]):
                category, label, url, query_match = source
                async with semaphore:
                    return await self._fetch(
                        client, category, label, url, query_match
                    )

            results = await asyncio.gather(
                *(guarded(source) for source in sources), return_exceptions=True
            )

        items: list[NewsItem] = []
        for (_, label, _, _), result in zip(sources, results):
            if isinstance(result, Exception):
                logger.warning(
                    "RSS source %s failed: %s: %s",
                    label,
                    type(result).__name__,
                    result,
                )
            else:
                items.extend(result)
        if not items:
            raise RuntimeError("所有 RSS 新闻源均不可用")
        return items

    @staticmethod
    def _format(items: list[NewsItem], *, output_limit: int = 6000) -> str:
        lines = ["RSS 聚合资讯（仅含标题、摘要与原文链接，按相关性和时间排序）："]
        for index, item in enumerate(items, start=1):
            published = (
                item.published.strftime("%Y-%m-%d %H:%M UTC")
                if item.published
                else "时间未提供"
            )
            lines.append(
                f"{index}. [{item.category}｜{item.source}] {published} — {item.title}"
            )
            title_key = re.sub(r"\W+", "", item.title).lower()
            summary_key = re.sub(r"\W+", "", item.summary).lower()
            if (
                item.summary
                and summary_key not in title_key
                and title_key not in summary_key
            ):
                lines.append(f"   摘要：{item.summary}")
            if item.link:
                lines.append(f"   原文：{item.link}")
        return "\n".join(lines)[:output_limit]

    async def latest_topics(self) -> str:
        """Return a large classified pool used only by proactive broadcasts."""
        if not self.enabled:
            raise RuntimeError("RSS news is disabled")
        cache_key = "__latest_topics__"
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and now - cached[0] <= self.cache_seconds:
            return cached[1]

        sources = [
            (category, label, url, False)
            for category, label, url in self.feeds
        ]
        items = await self._fetch_many(sources)
        oldest_timestamp = datetime.now(timezone.utc).timestamp() - self.max_age_hours * 3600
        timely = [
            item
            for item in items
            if item.published is None or item.published.timestamp() >= oldest_timestamp
        ]
        if timely:
            items = timely

        unique: list[NewsItem] = []
        seen: set[str] = set()
        source_counts: dict[str, int] = {}
        for item in sorted(
            items,
            key=lambda value: value.published.timestamp() if value.published else 0.0,
            reverse=True,
        ):
            identity = re.sub(r"\W+", "", item.title).lower()
            source_key = item.source.strip().lower()
            if not identity or identity in seen or source_counts.get(source_key, 0) >= 12:
                continue
            seen.add(identity)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            unique.append(item)

        output = self._format(unique, output_limit=48_000)
        self._cache[cache_key] = (now, output)
        return output

    async def query_topics(
        self,
        *,
        category: str = "",
        source: str = "",
        query: str = "",
        limit: int = 5,
    ) -> str:
        """Filter the same classified pool used by proactive broadcasts."""
        inferred_category, inferred_source = infer_topic_filters(
            " ".join(part for part in (category, source, query) if part)
        )
        requested_category = category.strip()
        if requested_category not in IdleNewsRotator.CATEGORY_ORDER:
            requested_category = inferred_category
        requested_source = inferred_source or source.strip()
        blocks = formatted_news_blocks(await self.latest_topics())
        if requested_category:
            marker = f"[{requested_category}｜"
            blocks = [block for block in blocks if marker in block]
        if requested_source:
            source_key = requested_source.lower()
            blocks = [
                block
                for block in blocks
                if source_key in block.splitlines()[0].lower()
            ]
        if not blocks:
            detail = "、".join(part for part in (requested_category, requested_source) if part)
            raise RuntimeError(f"RSS 资讯池中暂时没有符合“{detail or query}”的内容")

        query_tokens = _relevance_terms(query)
        if query_tokens:
            blocks.sort(
                key=lambda block: sum(token in block.lower() for token in query_tokens),
                reverse=True,
            )
        selected = blocks[: max(1, min(8, int(limit)))]
        label = " / ".join(part for part in (requested_category, requested_source) if part) or "综合"
        lines = [f"RSS 最新资讯（筛选：{label}；后端实时获取）："]
        for index, block in enumerate(selected, start=1):
            lines.append(f"{index}. {block}")
        return "\n".join(lines)[:6000]

    async def search(self, query: str) -> str:
        if not self.enabled:
            raise RuntimeError("RSS news is disabled")
        cache_key = _SPACE_RE.sub(" ", query).strip().lower()
        cached = self._cache.get(cache_key)
        now = time.monotonic()
        if cached and now - cached[0] <= self.cache_seconds:
            return cached[1]

        generic_headlines_request = bool(
            re.search(
                r"(?:今日|今天|最新|国际|全球|世界|热点).{0,8}新闻|"
                r"新闻.{0,8}(?:今日|今天|最新|国际|全球|世界|热点)",
                query,
            )
        )
        # A topic query is best served by the query feed alone. Fixed headline
        # feeds add latency and unrelated stories; they are useful for a broad
        # "today's world news" request.
        sources = (
            [(category, label, url, False) for category, label, url in self.feeds]
            if generic_headlines_request or not self.google_enabled
            else []
        )
        if self.google_enabled:
            google_url = (
                "https://news.google.com/rss/search?q="
                f"{quote_plus(query)}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            )
            sources.insert(0, ("搜索", "Google News", google_url, True))
        items = await self._fetch_many(sources)
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

        output = self._format(unique)
        self._cache[cache_key] = (now, output)
        if len(self._cache) > 64:
            oldest = min(self._cache, key=lambda item: self._cache[item][0])
            self._cache.pop(oldest, None)
        return output
