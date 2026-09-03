"""Server-owned chat vs live-lookup intent. The model does not decide this."""

from __future__ import annotations

import re

# WeChat / DuerOS-style skill patterns: verb lexicon × noun lexicon, plus a
# reject list so a news word inside "别说新闻了" does not start a lookup.


def _choice(*parts: str) -> str:
    return "(?:" + "|".join(parts) + ")"


_FILL = r".{0,14}"
_ASK = _choice(
    "看看", "看下", "看一看", "看一下", "瞅瞅", "瞧瞧", "刷刷", "翻翻",
    "讲讲", "讲个", "讲点", "讲条", "讲一下",
    "说说", "说个", "说点", "说条", "说一下",
    "聊聊", "聊点", "聊个",
    "来点", "来条", "来个", "来则",
    "播报", "播一下",
    "听听", "说来听听", "讲来听听",
    "有没有", "有没有啥", "有啥", "有什么", "还有啥", "还有什么", "还有",
    "想听", "想看", "想知道", "我想知道",
    "告诉我", "跟我说", "给我讲", "给我说",
    "查", "查一下", "查下", "查查", "查询", "查证",
    "搜", "搜索", "搜一下", "搜搜", "搜下",
    "找找", "找一下", "打听",
    "帮我查", "给我查", "帮我看看", "帮我搜", "帮我找",
    "百度一下", "谷歌一下", "上网查", "网上搜", "网上查",
    "联网", "核实", "验证", "确认一下",
)
_TIME = _choice(
    "今天", "今日", "最新", "现在", "实时", "最近", "刚才", "刚刚", "这会儿", "眼下",
)
_WHERE = _choice("网上", "外面", "圈里", "微博", "抖音", "推特", "推上", "全网")
_NEWS_NOUN = _choice(
    "新闻", "资讯", "热搜", "热榜", "头条", "头版", "要闻",
    "时事", "热点", "简报", "早报", "晚报", "八卦", "爆料",
)
_GOSSIP = _choice(
    "吃瓜", "爆瓜", "吃个瓜", "来点瓜", "来个瓜",
    "有啥瓜", "有什么瓜", "啥瓜", "什么瓜",
    r"(?<![西傻金南东生])瓜(?:呢|吗|不|啊)",
)
_EVENT = _choice(
    "发生了什么", "发生啥了", "发生什么了",
    "出什么事", "出啥事", "出大事",
    "有啥大事", "有什么大事", "有什么新鲜",
    "新鲜事", "大新闻", "啥事了",
    "都在说啥", "都在传啥", "都在聊啥", "都在讨论",
)
_WEATHER = _choice("天气", "天气预报", "气温", "温度", "下雨", "下雪", "几度")
_PRICE = _choice(
    "价格", "多少钱", "行情", "报价", "汇率", "现价", "股价", "币价",
    "金价", "油价", "什么价", "啥价", "涨了没", "跌了没", "涨没涨", "跌没跌",
)
_SEARCH_VERB = _choice(
    "查一下", "查下", "查查", "查证", "查询",
    "搜索", "搜一下", "搜搜", "搜下",
    "找找", "打听",
    "联网", "核实", "验证",
    "百度一下", "谷歌一下",
    r"(?:网上|互联网|上网).{0,6}(?:查|搜|找)",
)
_ASK_TAIL = _choice("看看", "讲讲", "说说", "说来听听", "吗", "呢", "不", "啊")

REJECT_LOOKUP_RE = re.compile(
    r"(?:别|不要|不想)(?:再)?(?:说|讲|听|播)(?:了)?(?:这些)?(?:新闻|资讯|热搜|头条|瓜)|"
    r"(?:不想听|不要)(?:新闻|资讯|热搜)|"
    r"没有新闻就算|"
    r"别说别的",
    re.IGNORECASE,
)
NEWS_REQUEST_RE = re.compile(
    rf"{_ASK}{_FILL}{_NEWS_NOUN}|"
    rf"(?:{_TIME}|{_WHERE}){_FILL}{_NEWS_NOUN}|"
    rf"{_NEWS_NOUN}{_FILL}{_ASK_TAIL}|"
    rf"{_GOSSIP}|"
    rf"(?:{_TIME}|{_WHERE}){_FILL}{_EVENT}|"
    rf"(?<!没)有{_NEWS_NOUN}(?:吗|不|呢|啊)?|"
    rf"^{_NEWS_NOUN}(?:吗|呢|不|啊)?$",
    re.IGNORECASE,
)
WEB_SEARCH_INTENT_RE = re.compile(
    rf"{NEWS_REQUEST_RE.pattern}|"
    rf"(?:{_ASK}|{_TIME}){_FILL}{_WEATHER}|"
    rf"{_WEATHER}{_FILL}(?:怎么样|咋样|如何|吗|呢)|"
    rf"(?:{_ASK}|{_TIME}){_FILL}{_PRICE}|"
    rf"{_PRICE}|"
    rf"{_SEARCH_VERB}|"
    r"最近.{0,14}(?:上涨|下跌|涨|跌|原因)|"
    r"为什么.{0,14}(?:上涨|下跌|涨|跌)|"
    r"\b(?:current|latest|today'?s?)\s+(?:price|news|weather|market)\b",
    re.IGNORECASE,
)
_NEWS_ASK_NOISE_RE = re.compile(
    r"@小麻|[？?！!。,.，、~～]|帮我|给我|你|再|先|"
    r"看看|看一看|看一下|看下|瞅瞅|瞧瞧|刷刷|翻翻|"
    r"讲讲|讲个|讲点|讲条|讲一下|说说|说个|说点|说条|说一下|"
    r"聊聊|聊点|聊个|来点|来条|来个|来则|播报|播一下|"
    r"听听|说来听听|讲来听听|告诉我|跟我说|给我讲|给我说|"
    r"想听|想看|想知道|我想知道|"
    r"有没有啥|有没有|有啥|有什么|还有啥|还有什么|还有|"
    r"查一下|查下|查查|查询|查证|查|搜索|搜一下|搜搜|搜下|搜|"
    r"找找|找一下|打听|帮我看看|帮我搜|帮我找|帮我查|给我查|"
    r"百度一下|谷歌一下|上网查|网上搜|网上查|联网|核实|验证|确认一下|"
    r"一下|一点|一条|一个|最新|今天|今日|现在|实时|最近|刚才|刚刚|这会儿|眼下|"
    r"网上|外面|圈里|微博|抖音|全网|的|点|个|条|则|下|"
    r"新闻|资讯|热搜|热榜|头条|头版|要闻|时事|热点|简报|早报|晚报|八卦|爆料|"
    r"吃瓜|爆瓜|吃个瓜|来点瓜|来个瓜|有啥瓜|啥瓜|什么瓜|瓜|"
    r"不|吗|呢|啊|呀|吧|哇"
)
SEARCH_FILLER_RE = re.compile(
    r"正在联网|正在查询|查找相关资料|核对一下来源|核对完就告诉|"
    r"先帮你查|我先帮你查|这就去查|再核对|核对清楚|同时查几个来源",
    re.IGNORECASE,
)
NEWS_FOLLOWUP_RE = re.compile(
    r"(?:这条|那条)(?:新闻|资讯|消息|报道|瓜)?|"
    r"(?:刚才|刚刚)(?:的)?(?:那|这)(?:条|个|则|件事)?|"
    r"此事|这件事|是真的吗|是假的吗|假新闻|辟谣|"
    r"后来呢|后来怎么|后来如何|还有下文|后续呢|最新进展|"
    r"为什么.{0,8}(?:涨|跌|上涨|下跌)",
    re.IGNORECASE,
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
_PRICE_WAIT_RE = re.compile(
    r"价格|多少钱|行情|报价|汇率|现价|股价|币价|金价|油价|什么价|啥价|涨了没|跌了没"
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


def _rejected_lookup(text: str) -> bool:
    return bool(REJECT_LOOKUP_RE.search(viewer_utterance(text)))


def needs_web_search(text: str) -> bool:
    """True only when the current utterance asks for a live internet fact."""
    utterance = viewer_utterance(text)
    if not utterance or REJECT_LOOKUP_RE.search(utterance):
        return False
    return bool(WEB_SEARCH_INTENT_RE.search(utterance) or NEWS_REQUEST_RE.search(utterance))


def is_news_request(text: str) -> bool:
    """True when the viewer asked for headlines rather than a priced fact."""
    utterance = viewer_utterance(text)
    if not utterance or REJECT_LOOKUP_RE.search(utterance):
        return False
    return bool(NEWS_REQUEST_RE.search(utterance))


def news_search_query(text: str) -> str:
    """Turn a spoken news ask into a query a search API can actually use."""
    utterance = viewer_utterance(text)
    if not utterance:
        return ""
    if not is_news_request(utterance):
        return utterance
    leftover = _NEWS_ASK_NOISE_RE.sub("", utterance)
    leftover = re.sub(r"\s+", "", leftover)
    if leftover in {"", "瓜", "八卦"} or len(leftover) < 2:
        return "今天国内外热点新闻"
    return utterance


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
    if not utterance or REJECT_LOOKUP_RE.search(utterance):
        return False
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
