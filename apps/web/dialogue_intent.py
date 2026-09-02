"""Server-owned chat vs live-lookup intent. The model does not decide this."""

from __future__ import annotations

import re

WEB_SEARCH_INTENT_RE = re.compile(
    r"(?:查(?:一下|下|查|证)?|搜索|搜一下|联网|核实|验证|热搜|行情|报价|汇率|多少钱|什么价|现价|"
    r"现在.{0,12}(?:价格|多少钱|报价|行情|天气|汇率)|"
    r"今天.{0,10}(?:新闻|天气|价格|发生)|"
    r"(?:最新|实时).{0,10}(?:新闻|消息|价格|行情|进展)|"
    r"最近.{0,14}(?:上涨|下跌|涨|跌|原因)|"
    r"为什么.{0,14}(?:上涨|下跌|涨|跌)|"
    r"(?:网上|互联网).{0,6}(?:查|搜)|"
    r"\b(?:current|latest|today'?s?)\s+(?:price|news|weather|market)\b)",
    re.IGNORECASE,
)
NEWS_FOLLOWUP_RE = re.compile(
    r"这个|这条|刚才(?:那|这)?|刚刚(?:那|这)?|那条|此事|这件事|是真的吗|后来呢|后来怎么|进展"
)
_COMMENT_MARKERS = (
    "【当前评论，这是唯一需要回答的问题】",
    "【当前评论】",
)
_SPEAKER_SAID_RE = re.compile(r"直播间观众“.+?”说：")

SIMPLE_CHAT_POLICY = (
    "这是简单闲聊：只回答对方这一句。用中文口语一两句，总共不超过四十个字。"
    "可以顺着这个人的记忆接话，但不要复述、续写旧答案，也不要接着讲新闻或瓜。"
    "禁止分点、标题、第一块瓜、第二块瓜、长故事和书面腔。能一句说完就一句。"
)
SPOKEN_CHINESE_POLICY = (
    "观众在用中文。你必须只用中文口语回答，两三句说完。"
    "禁止英文句子、禁止Markdown、禁止井号标题和列表。"
    "英文资料只可用来理解，说出口必须是中文。"
)
_HAN_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def viewer_utterance(text: str) -> str:
    """Return the viewer's actual line, ignoring packed memory/news wrappers."""
    value = str(text or "").strip()
    for marker in _COMMENT_MARKERS:
        if marker in value:
            value = value.split(marker, 1)[1].strip()
            break
    spoken = _SPEAKER_SAID_RE.search(value)
    if spoken:
        value = value[spoken.end():].strip()
    return value


def needs_web_search(text: str) -> bool:
    """True only when the current utterance asks for a live internet fact."""
    return bool(WEB_SEARCH_INTENT_RE.search(viewer_utterance(text)))


def wants_news_context(text: str) -> bool:
    """Attach the room's current headline only for lookup or a clear follow-up."""
    utterance = viewer_utterance(text)
    return needs_web_search(utterance) or bool(NEWS_FOLLOWUP_RE.search(utterance))


def viewer_is_chinese(text: str) -> bool:
    return bool(_HAN_RE.search(viewer_utterance(text)))


def looks_like_english_answer(text: str) -> bool:
    """True when a spoken reply leaked into English prose."""
    value = str(text or "").strip()
    if not value:
        return False
    han = len(_HAN_RE.findall(value))
    latin = len(_LATIN_RE.findall(value))
    return latin >= 20 and latin > han * 2
