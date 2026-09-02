"""Key-free RSS/Atom aggregation for current-news grounding."""

from __future__ import annotations

import asyncio
import hashlib
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
from difflib import SequenceMatcher
from urllib.parse import parse_qsl, quote_plus, urlencode, urlparse, urlunparse

import httpx


logger = logging.getLogger("s2s.rss_news")
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}|[\u4e00-\u9fff]{2,}")
_TITLE_NOISE_RE = re.compile(
    r"(?:最新|刚刚|突发|重磅|快讯|独家|现场|视频|图集|组图|媒体|消息称|"
    r"人民日报|澎湃新闻|中新网|中国新闻网|cnBeta|IT之家|极客公园|知乎日报)",
    re.IGNORECASE,
)
_TRACKING_QUERY_PREFIXES = ("utm_", "spm", "from", "source", "ref", "share")
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
_GOOGLE_HEADLINE_FEEDS = (
    ("新闻", "Google News 要闻", "https://news.google.com/rss?hl=zh-CN&gl=CN&ceid=CN:zh-Hans"),
    ("科技", "Google News 科技", "https://news.google.com/rss/search?q=%E7%A7%91%E6%8A%80&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"),
    ("知识", "Google News 知识", "https://news.google.com/rss/search?q=%E7%A7%91%E5%AD%A6+OR+%E7%9F%A5%E8%AF%86&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"),
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


def normalize_news_title(title: str) -> str:
    """Stable, cheap title form used by both pool and persisted broadcast history."""
    text = html.unescape(title or "").lower()
    text = re.sub(r"(?<=\d)[,.，](?=\d)", "", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*万", lambda m: str(int(float(m.group(1)) * 10_000)), text)
    text = _TITLE_NOISE_RE.sub("", text)
    return re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", text)


def canonical_news_url(url: str) -> str:
    parsed = urlparse(url or "")
    if not parsed.scheme or not parsed.netloc:
        return ""
    query = [
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith(_TRACKING_QUERY_PREFIXES)
    ]
    path = re.sub(r"/{2,}", "/", parsed.path or "/").rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", urlencode(query), ""))


def news_event_fingerprint(title: str, url: str = "") -> str:
    """Small persistent ID; fuzzy comparisons still use the stored normalized title."""
    canonical = canonical_news_url(url)
    basis = canonical or normalize_news_title(title)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def news_titles_similar(left: str, right: str) -> bool:
    """Detect syndicated versions without an embedding model or LLM latency."""
    a, b = normalize_news_title(left), normalize_news_title(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = sorted((a, b), key=len)
    if len(shorter) >= 14 and shorter in longer:
        return True
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    # CJK bigram overlap catches reordered syndicated headlines while requiring
    # enough shared substance to avoid collapsing merely related stories.
    def grams(value: str) -> set[str]:
        return {value[i:i + 2] for i in range(max(0, len(value) - 1))}
    ga, gb = grams(a), grams(b)
    overlap = len(ga & gb) / max(1, min(len(ga), len(gb)))
    numbers_a, numbers_b = set(re.findall(r"\d+(?:\.\d+)?", a)), set(re.findall(r"\d+(?:\.\d+)?", b))
    same_numbered_event = bool(numbers_a & numbers_b) and ratio >= 0.65 and overlap >= 0.55
    return ratio >= 0.82 or same_numbered_event or (
        len(shorter) >= 14 and ratio >= 0.68 and overlap >= 0.72
    )


def news_block_title(block: str) -> str:
    headline = block.splitlines()[0] if block else ""
    return headline.split(" — ", 1)[-1].strip()


def news_block_metadata(block: str) -> dict[str, str]:
    first = block.splitlines()[0] if block else ""
    match = re.match(r"\[(?P<category>[^｜\]]+)｜(?P<source>[^\]]+)\]\s+(?P<published>.*?)\s+—\s+(?P<title>.+)", first)
    summary = ""
    source_url = ""
    for line in block.splitlines()[1:]:
        if line.startswith("摘要："):
            summary = line.removeprefix("摘要：").strip()
        elif line.startswith("原文："):
            source_url = line.removeprefix("原文：").strip()
    values = match.groupdict() if match else {}
    return {
        "category": values.get("category", "新闻"),
        "source": values.get("source", ""),
        "published_at": values.get("published", ""),
        "title": values.get("title", news_block_title(block)),
        "summary": summary,
        "source_url": source_url,
        "evidence": block[:2400],
    }


def formatted_news_blocks(output: str, *, include_links: bool = False) -> list[str]:
    """Extract compact attributed story blocks from formatted RSS output."""
    starts = list(re.finditer(r"(?m)^\d+\.\s+", output))
    blocks: list[str] = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(output)
        block = output[match.end() : end].strip()
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not include_links:
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

    def choose(self, audience: str, output: str, *, persisted_titles: list[str] | None = None) -> str:
        blocks = formatted_news_blocks(output, include_links=True)
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
        persisted_titles = persisted_titles or []

        def unused(block: str) -> bool:
            title = news_block_title(block)
            return (
                self._identity(block) not in recent
                and not any(news_titles_similar(title, old) for old in persisted_titles)
                and not any(news_titles_similar(title, old) for old in recent)
            )

        for category in categories:
            marker = f"[{category}｜"
            selected = next(
                (
                    block
                    for block in blocks
                    if marker in block and unused(block)
                ),
                None,
            )
            if selected is not None:
                selected_category = category
                break
        if selected is None:
            selected = next((block for block in blocks if unused(block)), None)
        if selected is None:
            raise ValueError("RSS pool has no unbroadcast news events")
        recent.append(news_block_title(selected))
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
        self,
        sources: list[tuple[str, str, str, bool]],
        *,
        overall_timeout: float | None = None,
    ) -> list[NewsItem]:
        """Fetch same-host feeds one at a time. AnyFeeder sits on one CDN, and
        two parallel streams there regularly die with ConnectTimeout/SSLError."""
        headers = {"User-Agent": "CyberGirlfriendLive/1.0 RSS news reader"}
        timeout = httpx.Timeout(self.timeout, connect=min(8.0, self.timeout))
        host_gates: dict[str, asyncio.Semaphore] = {}
        limits = httpx.Limits(max_connections=4, max_keepalive_connections=2)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers, limits=limits
        ) as client:
            async def guarded(index_source: tuple[int, tuple[str, str, str, bool]]):
                index, source = index_source
                category, label, url, query_match = source
                host = (urlparse(url).hostname or label).lower()
                gate = host_gates.setdefault(host, asyncio.Semaphore(1))
                async with gate:
                    items = await asyncio.wait_for(
                        self._fetch(client, category, label, url, query_match),
                        timeout=self.timeout + 2.0,
                    )
                    return index, items

            tasks = [
                asyncio.create_task(guarded((index, source)))
                for index, source in enumerate(sources)
            ]
            done, pending = await asyncio.wait(
                tasks,
                timeout=overall_timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            indexed: dict[int, list[NewsItem] | Exception] = {}
            for task in done:
                try:
                    result_index, items = task.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("RSS fetch task failed: %s: %s", type(exc).__name__, exc)
                    continue
                indexed[result_index] = items
            results = [indexed.get(index, RuntimeError("cancelled")) for index in range(len(sources))]

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

        items: list[NewsItem] = []
        if self.google_enabled:
            google_sources = [
                (category, label, url, False)
                for category, label, url in _GOOGLE_HEADLINE_FEEDS
            ]
            try:
                items.extend(await self._fetch_many(google_sources, overall_timeout=14.0))
            except RuntimeError:
                items = []
        try:
            items.extend(
                await self._fetch_many(
                    [
                        (category, label, url, False)
                        for category, label, url in self.feeds
                    ],
                    overall_timeout=22.0,
                )
            )
        except RuntimeError:
            pass
        if not items:
            raise RuntimeError("所有 RSS 新闻源均不可用")
        oldest_timestamp = datetime.now(timezone.utc).timestamp() - self.max_age_hours * 3600
        timely = [
            item
            for item in items
            if item.published is None or item.published.timestamp() >= oldest_timestamp
        ]
        if timely:
            items = timely

        unique: list[NewsItem] = []
        seen_urls: set[str] = set()
        source_counts: dict[str, int] = {}
        for item in sorted(
            items,
            key=lambda value: value.published.timestamp() if value.published else 0.0,
            reverse=True,
        ):
            identity = normalize_news_title(item.title)
            canonical_url = canonical_news_url(item.link)
            source_key = item.source.strip().lower()
            if (
                not identity
                or (canonical_url and canonical_url in seen_urls)
                or any(news_titles_similar(item.title, old.title) for old in unique)
                or source_counts.get(source_key, 0) >= 12
            ):
                continue
            if canonical_url:
                seen_urls.add(canonical_url)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            unique.append(item)

        output = self._format(unique, output_limit=48_000)
        self._cache[cache_key] = (now, output)
        return output

    async def spoken_brief(self, query: str = "", limit: int = 4) -> str:
        """Return a few Chinese headlines the host can speak immediately."""
        if not self.enabled:
            raise RuntimeError("RSS news is disabled")
        cached = self._cache.get("__latest_topics__")
        now = time.monotonic()
        if cached and now - cached[0] <= self.cache_seconds:
            raw = cached[1]
        else:
            sources: list[tuple[str, str, str, bool]] = []
            if self.google_enabled:
                sources.extend(
                    (category, label, url, False)
                    for category, label, url in _GOOGLE_HEADLINE_FEEDS[:2]
                )
            sources.extend(
                (category, label, url, False)
                for category, label, url in self.feeds[:3]
            )
            try:
                items = await self._fetch_many(sources, overall_timeout=8.0)
            except Exception as exc:
                raise RuntimeError("新闻源这会儿连不上") from exc
            if not items:
                raise RuntimeError("新闻源这会儿连不上")
            raw = self._format(items[:16])
        blocks = formatted_news_blocks(raw)
        tokens = _relevance_terms(query)
        if tokens:
            blocks.sort(
                key=lambda block: sum(token in block.lower() for token in tokens),
                reverse=True,
            )
        selected = blocks[: max(1, min(6, int(limit)))]
        if not selected:
            raise RuntimeError("暂时没有可播的新闻")
        lines = ["刚才查到的最新资讯："]
        for index, block in enumerate(selected, start=1):
            rows = [line.strip() for line in block.splitlines() if line.strip()]
            if not rows:
                continue
            lines.append(f"{index}. {rows[0]}")
            for row in rows[1:]:
                if row.startswith("摘要："):
                    lines.append(f"   {row[:90]}")
                    break
        if len(lines) < 2:
            raise RuntimeError("暂时没有可播的新闻")
        return "\n".join(lines)

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
        seen_urls: set[str] = set()
        source_counts: dict[str, int] = {}
        for item in sorted(items, key=score, reverse=True):
            identity = normalize_news_title(item.title)
            canonical_url = canonical_news_url(item.link)
            source_key = item.source.strip().lower()
            if (
                not identity
                or (canonical_url and canonical_url in seen_urls)
                or any(news_titles_similar(item.title, old.title) for old in unique)
                or source_counts.get(source_key, 0) >= 2
            ):
                continue
            if canonical_url:
                seen_urls.add(canonical_url)
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
