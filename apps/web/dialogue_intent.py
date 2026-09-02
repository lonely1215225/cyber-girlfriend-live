"""Server-owned chat vs live-lookup intent. The model does not decide this."""

from __future__ import annotations

import re

WEB_SEARCH_INTENT_RE = re.compile(
    r"(?:查(?:一下|下|查|证)?|搜索|搜一下|联网|核实|验证|热搜|行情|报价|汇率|多少钱|什么价|现价|"
    r"现在.{0,12}(?:价格|多少钱|报价|行情|天气|汇率)|"
    r"今天.{0,10}(?:新闻|天气|价格|发生)|"
    r"(?:最新|实时).{0,10}(?:新闻|消息|价格|行情|进展)|"
    r"看看.{0,16}(?:新闻|资讯|热搜|头条)|"
    r"(?:有啥|有什么|来[条个]|讲[讲个]|播)(?:点|个|条)?(?:新闻|资讯|热搜|头条)|"
    r"(?:新闻|资讯|热搜|头条).{0,10}(?:说来听听|讲讲|说说)|"
    r"最近.{0,14}(?:上涨|下跌|涨|跌|原因)|"
    r"为什么.{0,14}(?:上涨|下跌|涨|跌)|"
    r"(?:网上|互联网).{0,6}(?:查|搜)|"
    r"\b(?:current|latest|today'?s?)\s+(?:price|news|weather|market)\b)",
    re.IGNORECASE,
)
NEWS_REQUEST_RE = re.compile(
    r"(?:看看|讲讲|说说|播|来|有啥|有什么|最新|今天|今日|热搜).{0,16}(?:新闻|资讯|热搜|头条)|"
    r"(?:新闻|资讯|热搜|头条).{0,10}(?:看看|讲讲|说说|说来听听)",
    re.IGNORECASE,
)
SEARCH_FILLER_RE = re.compile(
    r"正在联网|正在查询|查找相关资料|核对一下来源|核对完就告诉|"
    r"先帮你查|我先帮你查|这就去查|再核对|核对清楚|同时查几个来源",
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
    "这是简单闲聊：先给对方要的东西，再可补半句。一两句中文口语，总共不超过八十字。"
    "要名字、答案、选择时，第一句就必须点出来，禁止只预告、卖关子或反问。"
    "可以顺着这个人的记忆接话，但不要复述、续写旧答案，也不要接着讲新闻或瓜。"
    "禁止标题、长故事和书面腔。英文专有名词可以原样说。"
)
SPOKEN_CHINESE_POLICY = (
    "观众在用中文。你必须只用中文口语回答，两三句说完。"
    "禁止英文句子、禁止Markdown、禁止井号标题和列表。"
    "英文资料只可用来理解，说出口必须是中文。"
)
LOOKED_UP_EVIDENCE_POLICY = (
    "网上资料已经查到了。口头应承已经对观众说过了，不要再重复那一句。"
    "只根据【已查到的资料】用两三句中文口语讲最值得听的一两件事。"
    "每句不超过二十八个字，先说完一句再补下一句，不要把整段新闻塞进第一句。"
    "不要再说正在查、正在联网、稍等或核对来源，不要编造资料里没有的事，不要念网址。"
)
_PRICE_WAIT_RE = re.compile(r"价格|多少钱|行情|报价|汇率|现价")
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
    utterance = viewer_utterance(text)
    return bool(WEB_SEARCH_INTENT_RE.search(utterance) or NEWS_REQUEST_RE.search(utterance))


def is_news_request(text: str) -> bool:
    """True when the viewer asked for headlines rather than a priced fact."""
    return bool(NEWS_REQUEST_RE.search(viewer_utterance(text)))


def looks_like_search_filler(text: str) -> bool:
    """True when the model spoke a lookup promise instead of a fact."""
    return bool(SEARCH_FILLER_RE.search(str(text or "").strip()))


def lookup_wait_line(text: str) -> str:
    """One in-character beat while a live lookup is still running."""
    utterance = viewer_utterance(text)
    if is_news_request(utterance):
        return "我翻一下今天的，马上说。"
    if _PRICE_WAIT_RE.search(utterance):
        return "我去对一下最新的数。"
    return "我去看一眼，马上回你。"


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
