"""
Tiny server for the speech-to-speech demo.

The demo used to ship as a `sdk: static` Space, but the web-search tool needs a
search key the browser must NOT see. A static Space has no runtime process, so it
can't hold a secret the front-end uses. This server fixes that: it serves the
unchanged front-end AND exposes a same-origin `/api/search` proxy that holds the
Serper key server-side (see docs/adr/0001).

Everything lives in one container; the speech-to-speech backend stays a separate,
load-balanced service the browser talks to over WebSocket as before. The load
balancer's address is a secret too (like the Serper key): the browser never sees
it. `/api/session` proxies the session handshake server-side so only the
per-session compute URL the LB hands back (which the browser must dial) is exposed.

On the deployed Space the server also meters conversation time by HF login tier
(anonymous / signed-in / PRO) — see `limiter.py` and `auth.py`. That whole feature
is off unless BOTH `LOAD_BALANCER_URL` and `SPACE_ID` are set, so it runs only on
the live Space, never locally (even with the LB exported for testing).

`SPEECH_TO_SPEECH_URL` overrides everything: when set, the LB logic above is
disabled entirely (no session proxy, no queue, no metering, no sign-in) and the
browser connects directly to that URL, shown read-only in Settings.

Endpoints:
  GET  /api/config           -> { search, lb, allowDirect, s2sUrl, auth }
  GET  /api/me               -> login + tier + remaining budget (LB mode only)
  POST /api/search           -> { results, answer }  Google via Serper.dev
  POST /api/session          -> proxies <LB>/session: a grant, or a queue ticket
  GET  /api/queue/{id}       -> proxies <LB>/queue/{id}: position, or a grant on claim
  DELETE /api/queue/{id}     -> leave the queue (explicit "Leave queue" button)
  POST /api/queue/end        -> leave the queue (sendBeacon on teardown)
  POST /api/session/heartbeat-> extend the reservation; { expired }
  POST /api/session/end      -> reconcile + refund (sendBeacon on teardown)
  /*                         -> static files (index.html, main.js, ...)

When every compute slot is busy the load balancer hands back a queue ticket
instead of a grant; the browser polls /api/queue/{id} until it reaches the front
and a slot frees. Waiting reserves nothing — the daily budget is only reserved at
the moment a slot is actually claimed (a grant), never while queued.
"""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from urllib.parse import quote

import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import auth
import limiter
from mcp_gateway import McpGateway
from mention_reply import MentionReplyWorker
from room_manager import LiveRoom, RoomError

logger = logging.getLogger("s2s.search")

SERPER_KEY = os.environ.get("SERPER_API_KEY", "").strip()
# Speech-to-speech load balancer URL. When set, the browser POSTs /api/session
# (which proxies <lb>/session here, server-side) and connects to the URL the LB
# returns (the original flow). The LB address itself is never sent to the browser.
# When empty, the user may instead set a direct s2s server URL in Settings and the
# browser connects to it straight (no load balancer).
LOAD_BALANCER_URL = os.environ.get("LOAD_BALANCER_URL", "").strip()
# Direct s2s server URL pinned by the deploy. Takes priority over the load
# balancer: when set, ALL LB logic is disabled (no /api/session proxy, no queue,
# no limiter, no sign-in) and the browser connects to this URL directly. Unlike
# the LB address it is NOT a secret — /api/config sends it to the client, which
# shows it read-only in Settings.
SPEECH_TO_SPEECH_URL = os.environ.get("SPEECH_TO_SPEECH_URL", "").strip()
if SPEECH_TO_SPEECH_URL:
    LOAD_BALANCER_URL = ""
LIVE_ROOM_ENABLED = os.environ.get("LIVE_ROOM_ENABLED", "1").strip().lower() in {
    "1", "true", "yes", "on",
}
S2S_INTERNAL_WS_URL = os.environ.get(
    "S2S_INTERNAL_WS_URL", "ws://127.0.0.1:8765/v1/realtime"
).strip()
ROOM_COOKIE = "cg_live_room"
ADMIN_COOKIE = "cg_admin_settings"
ADMIN_SETTINGS_PASSWORD = os.environ.get("ADMIN_SETTINGS_PASSWORD", "123456")
ADMIN_SESSION_TTL_SECONDS = max(300, int(os.environ.get("ADMIN_SESSION_TTL_SECONDS", "1800")))
_admin_secret = (os.environ.get("ADMIN_SESSION_SECRET", "").encode() or secrets.token_bytes(32))
_admin_attempts: dict[str, list[float]] = {}
live_room = LiveRoom(
    queue_limit=int(os.environ.get("LIVE_ROOM_QUEUE_LIMIT", "100")),
    pending_timeout_s=int(os.environ.get("LIVE_ROOM_JOIN_TIMEOUT", "60")),
    max_call_s=int(os.environ.get("LIVE_ROOM_MAX_CALL_SECONDS", "600")),
)
# HF injects SPACE_ID ("owner/space") into every Space runtime; it's absent
# locally and on a plain `docker run`. We meter conversation time ONLY on the
# deployed Space — i.e. when BOTH the LB is configured AND we're on a Space.
# Off-Space (local dev, even with the LB exported) the app still proxies the LB,
# but nothing is metered: no budget, no reservations, no sign-in gating.
SPACE_ID = os.environ.get("SPACE_ID", "").strip()
LIMITER_ENABLED = bool(LOAD_BALANCER_URL) and bool(SPACE_ID)
# Optional deployment gate. When enabled, session allocation fails closed unless
# Hugging Face OAuth is available and the requester is signed in.
REQUIRE_LOGIN = os.environ.get("REQUIRE_LOGIN", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_STARTUP_GREETING = (
    "Start the conversation now with a brief, spontaneous greeting in character. "
    "Keep it to one sentence, invite the user in naturally, and vary the wording each time."
)
STARTUP_GREETING = os.environ.get("STARTUP_GREETING", DEFAULT_STARTUP_GREETING).strip()
IDLE_PROMPT = os.environ.get(
    "IDLE_PROMPT",
    "对方安静了一会儿。请主动关心对方还在不在，并自然挑一个轻松的新话题聊，"
    "例如最近在做什么、喜欢的动漫、动物、音乐、电影、游戏或想去的地方。"
    "不要说这是系统要求，不要重复刚聊过的话题，用一到两句口语表达。",
).strip()
IDLE_PROMPT_MIN_SECONDS = max(15, int(os.environ.get("IDLE_PROMPT_MIN_SECONDS", "35")))
IDLE_PROMPT_MAX_SECONDS = max(
    IDLE_PROMPT_MIN_SECONDS, int(os.environ.get("IDLE_PROMPT_MAX_SECONDS", "55"))
)
AVATAR_LISTEN_URL = os.environ.get("AVTR1_LOCAL_TEE_URL", "").rstrip("/")
SERPER_URL = "https://google.serper.dev/search"
# Cap results so the tool output stays small enough to feed back to the model.
MAX_RESULTS = 5
HERE = os.path.dirname(os.path.abspath(__file__))
LB_USER_AGENT = "hf-realtime-voice-space"

app = FastAPI(title="s2s-demo")
mcp_gateway = McpGateway()
mention_replies = MentionReplyWorker(
    live_room,
    mcp_gateway,
    S2S_INTERNAL_WS_URL,
    AVATAR_LISTEN_URL,
    max_queue=int(os.environ.get("MENTION_REPLY_QUEUE_LIMIT", "30")),
)

# Wire HF OAuth before the app serves (no-op unless the OAuth env is present).
# Sign-in only matters when we're metering (prod Space), so gate it on that.
AUTH_ENABLED = LIMITER_ENABLED and auth.attach(app)


@app.on_event("startup")
async def _startup():
    """Stand up the usage DB and a periodic sweeper — metered (prod Space) only."""
    if LIVE_ROOM_ENABLED:
        live_room.start()
        mention_replies.start()
    if mcp_gateway.enabled:
        asyncio.create_task(mcp_gateway.warmup())
    if not LIMITER_ENABLED:
        return
    limiter.init()
    asyncio.create_task(_sweeper())


@app.on_event("shutdown")
async def _shutdown():
    if LIVE_ROOM_ENABLED:
        await mention_replies.stop()
        await live_room.stop()
    await mcp_gateway.close()


async def _sweeper():
    while True:
        await asyncio.sleep(limiter.REAP_AFTER_SEC)
        try:
            await asyncio.to_thread(limiter.sweep)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("usage sweep failed: %r", exc)


class SearchRequest(BaseModel):
    query: str
    # Optional user-supplied key (fallback when the deploy has no server key).
    # Used for this request only; never stored.
    key: str | None = None


class AdminUnlockRequest(BaseModel):
    password: str


class McpCallRequest(BaseModel):
    name: str
    arguments: dict


def _public_ws_url(request: Request) -> str:
    """Same-origin realtime URL so remote browsers do not dial localhost."""
    pinned = os.environ.get("PUBLIC_WS_URL", "").strip()
    if pinned:
        return pinned
    forwarded = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    proto = "wss" if forwarded == "https" else "ws"
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or ""
    ).split(",")[0].strip()
    if not host or host.startswith("127.0.0.1") or host.startswith("localhost"):
        public_ip = os.environ.get("PUBLIC_IP", "").strip()
        public_port = os.environ.get("PUBLIC_HTTP_PORT", "").strip()
        if public_ip and public_port:
            host = f"{public_ip}:{public_port}"
        elif public_ip:
            host = public_ip
        else:
            host = "127.0.0.1"
    return f"{proto}://{host}/v1/realtime"


def _room_ws_url(request: Request, ticket: str) -> str:
    forwarded = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    proto = "wss" if forwarded == "https" else "ws"
    host = (request.headers.get("x-forwarded-host") or request.headers.get("host") or "127.0.0.1").split(",")[0].strip()
    return f"{proto}://{host}/api/realtime?ticket={quote(ticket, safe='')}"


def _set_room_cookie(response, participant_token: str, request: Request) -> None:
    forwarded = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    response.set_cookie(
        ROOM_COOKIE,
        participant_token,
        max_age=365 * 24 * 60 * 60,
        httponly=True,
        secure=forwarded == "https",
        samesite="lax",
        path="/",
    )


def _request_is_https(request: Request) -> bool:
    forwarded = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    return forwarded == "https"


def _admin_token(expires: int) -> str:
    value = str(expires)
    signature = hmac.new(_admin_secret, value.encode(), hashlib.sha256).hexdigest()
    return f"{value}.{signature}"


def _admin_unlocked(request: Request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE, "")
    try:
        raw_expiry, signature = token.split(".", 1)
        expires = int(raw_expiry)
    except (TypeError, ValueError):
        return False
    expected = hmac.new(_admin_secret, raw_expiry.encode(), hashlib.sha256).hexdigest()
    return expires >= int(time.time()) and hmac.compare_digest(signature, expected)


def _admin_client_key(request: Request) -> str:
    real_ip = request.headers.get("x-real-ip", "").strip()
    return real_ip or (request.client.host if request.client else "unknown")


def _set_admin_cookie(response: JSONResponse, request: Request) -> None:
    expires = int(time.time()) + ADMIN_SESSION_TTL_SECONDS
    response.set_cookie(
        ADMIN_COOKIE,
        _admin_token(expires),
        max_age=ADMIN_SESSION_TTL_SECONDS,
        httponly=True,
        secure=_request_is_https(request),
        samesite="strict",
        path="/",
    )


async def _room_identity(request: Request, *, create: bool = True):
    return await live_room.identify(request.cookies.get(ROOM_COOKIE), create=create)


def _room_error(exc: RoomError) -> JSONResponse:
    body = {"detail": str(exc), "reason": exc.code}
    if exc.code == "at_capacity":
        body = {"state": "at_capacity", "detail": str(exc)}
    return JSONResponse(body, status_code=exc.status)


def _room_grant_payload(request: Request, data: dict) -> dict:
    if data.get("state") == "queued":
        return data
    ticket = str(data.get("session_token") or "")
    connect_url = _room_ws_url(request, ticket)
    return {**data, "connect_url": connect_url, "websocket_url": connect_url}


@app.get("/avatar-sync.js", include_in_schema=False)
async def avatar_sync_js():
    return FileResponse(os.path.join(HERE, "avatar-sync.js"), media_type="application/javascript")


@app.get("/api/avatar-config")
def avatar_config():
    return {
        "transport": "http-flv",
        "sampleRate": int(os.environ.get("AVATAR_SAMPLE_RATE", "16000")),
    }


@app.get("/api/config")
def config(request: Request):
    """Client bootstrap: whether web search is available, whether the deploy runs
    behind a load balancer (so the browser uses the /api/session proxy + limiter),
    whether HF sign-in is available, and whether the user may instead set a direct
    s2s server URL. The LB address itself is intentionally NOT included."""
    s2s_url = SPEECH_TO_SPEECH_URL
    if not s2s_url or s2s_url.lower() in {"auto", "same-origin", "same_origin"}:
        s2s_url = _public_ws_url(request)
    return {
        "search": bool(SERPER_KEY),
        "lb": LIVE_ROOM_ENABLED or bool(LOAD_BALANCER_URL),
        "liveRoom": LIVE_ROOM_ENABLED,
        "allowDirect": not LIVE_ROOM_ENABLED and not LOAD_BALANCER_URL,
        # Deploy-pinned direct s2s URL (empty when unset). Not a secret: the
        # browser dials it itself, and Settings shows it locked.
        "s2sUrl": "" if LIVE_ROOM_ENABLED else s2s_url,
        "startupGreeting": STARTUP_GREETING,
        "idlePrompt": IDLE_PROMPT,
        "idlePromptMinSeconds": IDLE_PROMPT_MIN_SECONDS,
        "idlePromptMaxSeconds": IDLE_PROMPT_MAX_SECONDS,
        "mcp": mcp_gateway.enabled,
        "auth": AUTH_ENABLED,
        "requireLogin": REQUIRE_LOGIN,
    }


@app.get("/api/admin/status")
async def admin_status(request: Request):
    return {"unlocked": _admin_unlocked(request), "expiresIn": ADMIN_SESSION_TTL_SECONDS}


@app.post("/api/admin/unlock")
async def admin_unlock(body: AdminUnlockRequest, request: Request):
    key = _admin_client_key(request)
    now = time.monotonic()
    attempts = [stamp for stamp in _admin_attempts.get(key, []) if now - stamp < 60]
    if len(attempts) >= 5:
        raise HTTPException(status_code=429, detail="尝试次数过多，请一分钟后再试")
    if not hmac.compare_digest(str(body.password), ADMIN_SETTINGS_PASSWORD):
        attempts.append(now)
        _admin_attempts[key] = attempts
        raise HTTPException(status_code=401, detail="管理密码不正确")
    _admin_attempts.pop(key, None)
    response = JSONResponse({"unlocked": True, "expiresIn": ADMIN_SESSION_TTL_SECONDS})
    _set_admin_cookie(response, request)
    return response


@app.post("/api/admin/lock")
async def admin_lock():
    response = JSONResponse({"unlocked": False})
    response.delete_cookie(ADMIN_COOKIE, path="/")
    return response


@app.get("/api/me")
async def me(request: Request):
    """Login state, tier, and remaining daily budget. Only meaningful in LB mode;
    sets the anonymous tracking cookie when first seen."""
    if not LIMITER_ENABLED:
        return {"enabled": False}
    view = auth.user_view(request)
    tier, keys, set_cookie = auth.resolve_identity(request)
    unlimited = limiter.budget_for(tier) is None
    rem = None if unlimited else await asyncio.to_thread(limiter.remaining, keys, tier)
    out = {
        "enabled": True,
        "auth": AUTH_ENABLED,
        "loginRequired": REQUIRE_LOGIN,
        **view,
        "remainingSec": rem,
        "limitSec": limiter.budget_for(tier),
        "loginUrl": auth.OAUTH_LOGIN_PATH if AUTH_ENABLED else None,
        "logoutUrl": auth.OAUTH_LOGOUT_PATH if AUTH_ENABLED else None,
    }
    resp = JSONResponse(out)
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


def _login_gate(request: Request) -> JSONResponse | None:
    """Reject anonymous session use when this deployment requires HF login."""
    if not REQUIRE_LOGIN:
        return None
    if not AUTH_ENABLED:
        return JSONResponse(
            {"reason": "auth_unavailable"},
            status_code=503,
        )
    if auth.current_user(request) is None:
        return _login_required_response("login_required")
    return None


def _login_required_response(reason: str, set_cookie=None) -> JSONResponse:
    """Actionable 401 understood by the browser's login-required flow."""
    resp = JSONResponse(
        {
            "reason": reason,
            "loginUrl": auth.OAUTH_LOGIN_PATH if AUTH_ENABLED else None,
        },
        status_code=401,
    )
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


@app.post("/api/search")
async def search(req: SearchRequest):
    """Proxy a Google search via Serper.dev. The key stays on the server unless
    the user brought their own (then theirs is used for this request only)."""
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query.")

    key = (req.key or "").strip() or SERPER_KEY
    if not key:
        # No server key and the user didn't supply one — search is unavailable.
        raise HTTPException(status_code=503, detail="Search is not configured.")

    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    payload = {"q": query, "num": MAX_RESULTS}
    try:
        async with httpx.AsyncClient(timeout=12.0) as http:
            resp = await http.post(SERPER_URL, headers=headers, json=payload)
    except httpx.RequestError as exc:
        logger.warning("Serper unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Search provider unreachable.")

    if resp.status_code != 200:
        # Serper's error body carries the real reason (e.g. "Not enough
        # credits") and contains no key, so it's safe to log and relay.
        body = resp.text[:300]
        logger.warning("Serper error %s: %s", resp.status_code, body)
        msg = None
        try:
            msg = resp.json().get("message")
        except Exception:
            pass
        detail = f"Search provider error ({resp.status_code})"
        if msg:
            detail += f": {msg}"
        raise HTTPException(status_code=502, detail=detail)

    data = resp.json()
    results = []
    for item in (data.get("organic") or [])[:MAX_RESULTS]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
            }
        )

    # A direct answer when Google has one — saves the model a hop.
    box = data.get("answerBox") or {}
    answer = box.get("answer") or box.get("snippet") or None
    if not answer:
        kg = data.get("knowledgeGraph") or {}
        answer = kg.get("description") or None

    return JSONResponse({"query": query, "answer": answer, "results": results})


@app.get("/api/mcp/tools")
async def mcp_tools():
    if not mcp_gateway.enabled:
        return {"enabled": False, "tools": [], "sources": []}
    try:
        tools = await mcp_gateway.list_tools()
    except Exception as exc:  # noqa: BLE001
        logger.warning("MCP tools unavailable: %s", exc)
        raise HTTPException(status_code=502, detail="MCP 服务暂时不可用")
    sources = sorted({str(tool.get("source")) for tool in tools if tool.get("source")})
    return {"enabled": True, "tools": tools, "sources": sources}


@app.post("/api/mcp/call")
async def mcp_call(body: McpCallRequest, request: Request):
    if not mcp_gateway.enabled:
        raise HTTPException(status_code=503, detail="MCP 未启用")
    if LIVE_ROOM_ENABLED:
        try:
            participant, _ = await _room_identity(request, create=False)
            state = await live_room.snapshot(participant.token)
        except RoomError as exc:
            return _room_error(exc)
        if state.get("me", {}).get("status") != "calling":
            raise HTTPException(status_code=403, detail="只有当前连线者可以调用 MCP 工具")
    try:
        output = await mcp_gateway.call(body.name, body.arguments)
    except KeyError:
        raise HTTPException(status_code=404, detail="未知的 MCP 工具")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (httpx.HTTPError, RuntimeError, json.JSONDecodeError) as exc:
        logger.warning("MCP call %s failed: %s", body.name, exc)
        raise HTTPException(status_code=502, detail="MCP 工具调用失败")
    return {"name": body.name, "output": output}


@app.post("/api/avatar/interrupt")
async def avatar_interrupt(request: Request):
    """Let only the active caller cut queued avatar audio for manual barge-in."""
    try:
        participant, _ = await _room_identity(request, create=False)
        state = await live_room.snapshot(participant.token)
    except RoomError as exc:
        return _room_error(exc)
    if state.get("me", {}).get("status") != "calling":
        raise HTTPException(status_code=403, detail="只有当前连线者可以打断数字人")
    if AVATAR_LISTEN_URL:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.post(f"{AVATAR_LISTEN_URL}/interrupt")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("manual avatar interrupt failed: %s", exc)
            raise HTTPException(status_code=502, detail="数字人打断失败")
    return {"ok": True}


class RenameRequest(BaseModel):
    name: str


class RoomChatRequest(BaseModel):
    text: str


@app.post("/api/room/join")
async def room_join(request: Request):
    if not LIVE_ROOM_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    participant, is_new = await _room_identity(request)
    data = await live_room.snapshot(participant.token)
    response = JSONResponse(data)
    if is_new:
        _set_room_cookie(response, participant.token, request)
    return response


@app.get("/api/room/state")
async def room_state(request: Request):
    if not LIVE_ROOM_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        participant, _ = await _room_identity(request, create=False)
        return await live_room.snapshot(participant.token)
    except RoomError as exc:
        return _room_error(exc)


@app.patch("/api/room/name")
async def room_rename(body: RenameRequest, request: Request):
    if not LIVE_ROOM_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        participant, _ = await _room_identity(request, create=False)
        return {"me": await live_room.rename(participant.token, body.name)}
    except RoomError as exc:
        return _room_error(exc)


@app.post("/api/room/chat")
async def room_chat(body: RoomChatRequest, request: Request):
    if not LIVE_ROOM_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        participant, _ = await _room_identity(request, create=False)
        message = await live_room.publish_chat(participant.token, body.text)
        mention_position = mention_replies.enqueue(message)
        return {"message": message, "mentionQueuePosition": mention_position}
    except RoomError as exc:
        return _room_error(exc)


@app.get("/api/room/events")
async def room_events(request: Request):
    if not LIVE_ROOM_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    try:
        participant, _ = await _room_identity(request, create=False)
    except RoomError as exc:
        return _room_error(exc)
    channel = await live_room.subscribe(participant.token)

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(channel.get(), timeout=15)
                    yield f"event: room\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            await live_room.unsubscribe(channel)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/api/session")
async def session(request: Request):
    """Proxy the session handshake to the load balancer, keeping its URL secret,
    and meter conversation time by tier.

    The browser POSTs here (same-origin); we resolve the caller's tier, refuse if
    today's budget is already spent (402), otherwise POST <LOAD_BALANCER_URL>/session
    and relay the JSON back. The LB body carries a per-session `connect_url`
    (compute host + short-lived token) the browser must dial directly — that one
    URL is unavoidably exposed, but the stable load-balancer address is not. On a
    successful grant we reserve the first time chunk against the day's budget."""
    if LIVE_ROOM_ENABLED:
        try:
            participant, is_new = await _room_identity(request)
            data = await live_room.request_session(participant.token)
            # A newly granted/queued live interaction always outranks an
            # audience mention response. This is a no-op unless the room bot is
            # currently using the speech pipeline.
            await mention_replies.interrupt()
            response = JSONResponse(_room_grant_payload(request, data))
            if is_new:
                _set_room_cookie(response, participant.token, request)
            return response
        except RoomError as exc:
            return _room_error(exc)

    if not LOAD_BALANCER_URL:
        # No LB configured — this deploy is direct-mode only; the browser should
        # never call this. 404 so it's indistinguishable from a missing route.
        raise HTTPException(status_code=404, detail="Not found.")

    login_error = _login_gate(request)
    if login_error is not None:
        return login_error

    tier, keys, set_cookie = auth.resolve_identity(request)
    # Metering runs only on the deployed Space; off-Space the LB still proxies but
    # nothing is tracked. Within metering, unlimited tiers (pro, org) aren't either.
    tracked = LIMITER_ENABLED and limiter.budget_for(tier) is not None

    # Refuse before troubling the LB if the day's budget is already gone. Done
    # here (at enqueue) so we never put a user who can't talk into the queue.
    if tracked:
        rem = await asyncio.to_thread(limiter.remaining, keys, tier)
        if rem is not None and rem <= 0:
            resp = JSONResponse(
                {"tier": tier, "reason": "limit", "remainingSec": 0}, status_code=402
            )
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    url = f"{LOAD_BALANCER_URL.rstrip('/')}/session"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            lb = await http.post(
                url,
                headers=_load_balancer_headers(request),
                content="{}",
            )
    except httpx.RequestError as exc:
        logger.warning("Load balancer unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    # The queue is full: the LB replies 503 {state:"at_capacity"}. Relay it as-is
    # so the client shows a soft "try again shortly", not a hard error.
    if lb.status_code == 503:
        body = _safe_json(lb)
        if body.get("state") == "at_capacity":
            resp = JSONResponse({"state": "at_capacity"}, status_code=503)
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    if lb.status_code == 401:
        body = _safe_json(lb)
        reason = body.get("reason") or "login_required"
        if reason == "token_invalid":
            session = getattr(request, "scope", {}).get("session")
            if isinstance(session, dict):
                session.pop("oauth_info", None)
        logger.info("Session authentication rejected: %s", reason)
        return _login_required_response(reason, set_cookie)

    if lb.status_code != 200:
        # The LB's error body may name the reason (e.g. capacity); it carries no
        # secret, so relay a trimmed copy.
        logger.warning("Session handshake failed %s: %s", lb.status_code, lb.text[:300])
        raise HTTPException(status_code=502, detail=f"Session handshake failed ({lb.status_code}).")

    data = lb.json()

    # Busy pool: the LB queued us. Relay the ticket untouched — crucially with NO
    # reservation, so waiting in line never costs the day's budget.
    if data.get("state") == "queued":
        data["tier"] = tier
        resp = JSONResponse(data)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    # A slot was free: reserve the first chunk now and return the grant.
    return await _finalize_grant(data, keys, tier, tracked, set_cookie)


def _load_balancer_headers(request: Request) -> dict[str, str]:
    """Headers for the server-to-server session allocation request.

    Reachy Mini uses this dedicated header for an optional HF user token. The
    load balancer fingerprints it immediately and resolves the account through
    whoami asynchronously, so allocation remains fast. Anonymous visitors send
    no credential header.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": LB_USER_AGENT,
    }
    token = auth.current_access_token(request)
    if token:
        headers["X-Reachy-Mini-Authorization"] = f"Bearer {token}"
    return headers


@app.get("/api/queue/{queue_id}")
async def queue_status(queue_id: str, request: Request):
    """Poll a waiting ticket: relay the position, or — when the head of the line
    claims a freed slot — reserve the budget now and return the grant. Re-checks the
    daily budget at claim, since a multi-minute wait could have spent it elsewhere."""
    if LIVE_ROOM_ENABLED:
        try:
            participant, _ = await _room_identity(request, create=False)
            data = await live_room.poll_queue(participant.token, queue_id)
            return JSONResponse(_room_grant_payload(request, data))
        except RoomError as exc:
            return _room_error(exc)

    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")

    login_error = _login_gate(request)
    if login_error is not None:
        return login_error

    tier, keys, set_cookie = auth.resolve_identity(request)
    tracked = LIMITER_ENABLED and limiter.budget_for(tier) is not None

    url = f"{LOAD_BALANCER_URL.rstrip('/')}/queue/{queue_id}"
    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            lb = await http.get(url)
    except httpx.RequestError as exc:
        logger.warning("Load balancer unreachable: %r", exc)
        raise HTTPException(status_code=502, detail="Speech service unreachable.")

    if lb.status_code == 404:
        # Ticket unknown/expired (reaped after we stopped polling). Tell the client
        # to start over rather than spin.
        resp = JSONResponse({"state": "expired"}, status_code=404)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    if lb.status_code != 200:
        logger.warning("Queue poll failed %s: %s", lb.status_code, lb.text[:300])
        raise HTTPException(status_code=502, detail=f"Queue poll failed ({lb.status_code}).")

    data = lb.json()

    if data.get("state") == "queued":
        data["tier"] = tier
        resp = JSONResponse(data)
        if set_cookie:
            auth.set_anon_cookie(resp, set_cookie)
        return resp

    # Claimed a slot. Re-check the budget: it may have been spent in another tab
    # during the wait. If so, refuse — the just-claimed slot is now a pending
    # session on the LB and its pending-timeout reaper reclaims it shortly.
    if tracked:
        rem = await asyncio.to_thread(limiter.remaining, keys, tier)
        if rem is not None and rem <= 0:
            resp = JSONResponse(
                {"tier": tier, "reason": "limit", "remainingSec": 0}, status_code=402
            )
            if set_cookie:
                auth.set_anon_cookie(resp, set_cookie)
            return resp

    return await _finalize_grant(data, keys, tier, tracked, set_cookie)


@app.delete("/api/queue/{queue_id}")
async def queue_leave(queue_id: str, request: Request):
    """Leave the queue from the explicit 'Leave queue' button (a real fetch)."""
    if LIVE_ROOM_ENABLED:
        try:
            participant, _ = await _room_identity(request, create=False)
            await live_room.leave_queue(participant.token, queue_id)
            return {"ok": True}
        except RoomError as exc:
            return _room_error(exc)
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")
    await _lb_leave(queue_id)
    return {"ok": True}


@app.post("/api/queue/end")
async def queue_end(request: Request):
    """Leave the queue on teardown/tab-close (navigator.sendBeacon, which can only
    POST). Body: { queueId }. Best-effort; the LB reaps the ticket on TTL anyway."""
    if LIVE_ROOM_ENABLED:
        try:
            participant, _ = await _room_identity(request, create=False)
            qid = await _queue_id(request)
            await live_room.leave_queue(participant.token, qid or None)
            return {"ok": True}
        except RoomError as exc:
            return _room_error(exc)
    if not LOAD_BALANCER_URL:
        raise HTTPException(status_code=404, detail="Not found.")
    qid = await _queue_id(request)
    if qid:
        await _lb_leave(qid)
    return {"ok": True}


async def _finalize_grant(data, keys, tier, tracked, set_cookie):
    """Shared grant tail (fast path or queue claim): reserve the first chunk, attach
    the metering fields the client needs, and set the anon cookie."""
    remaining = None
    if tracked and data.get("session_id"):
        await asyncio.to_thread(limiter.begin, data["session_id"], keys, tier)
        remaining = await asyncio.to_thread(limiter.remaining, keys, tier)

    data.update({
        "tier": tier,
        "limited": tracked,
        "remainingSec": remaining,
        "heartbeatSec": limiter.HEARTBEAT_SEC,
    })
    resp = JSONResponse(data)
    if set_cookie:
        auth.set_anon_cookie(resp, set_cookie)
    return resp


async def _lb_leave(queue_id: str) -> None:
    """Best-effort: tell the LB to drop a waiting ticket."""
    url = f"{LOAD_BALANCER_URL.rstrip('/')}/queue/{queue_id}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as http:
            await http.delete(url)
    except httpx.RequestError as exc:
        logger.warning("Queue leave failed: %r", exc)


def _safe_json(response) -> dict:
    try:
        body = response.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


async def _queue_id(request: Request) -> str:
    """Pull `queueId` from a JSON body, tolerating sendBeacon's blob posts."""
    try:
        data = await request.json()
    except Exception:
        return ""
    return (data or {}).get("queueId", "") if isinstance(data, dict) else ""


async def _session_id(request: Request) -> str:
    """Pull `sessionId` from a JSON body, tolerating sendBeacon's blob posts."""
    try:
        data = await request.json()
    except Exception:
        return ""
    return (data or {}).get("sessionId", "") if isinstance(data, dict) else ""


@app.post("/api/session/heartbeat")
async def session_heartbeat(request: Request):
    """Extend the live reservation one chunk at a time. `expired` once the day's
    budget is spent — the client then tears down."""
    if LIVE_ROOM_ENABLED:
        try:
            participant, _ = await _room_identity(request, create=False)
            snapshot = await live_room.snapshot(participant.token)
            return {"expired": snapshot["me"].get("status") not in {"ready", "calling"}}
        except RoomError:
            return {"expired": True}
    if not LIMITER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    sid = await _session_id(request)
    alive = bool(sid) and await asyncio.to_thread(limiter.heartbeat, sid)
    return {"expired": not alive}


@app.post("/api/session/end")
async def session_end(request: Request):
    """Clean teardown: reconcile to real elapsed time and refund the unused
    chunk. Sent via navigator.sendBeacon, so it must succeed without a response."""
    if LIVE_ROOM_ENABLED:
        try:
            participant, _ = await _room_identity(request, create=False)
            sid = await _session_id(request)
            await live_room.end_session(participant.token, sid or None)
            mention_replies.notify()
            return {"ok": True}
        except RoomError as exc:
            return _room_error(exc)
    if not LIMITER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found.")
    sid = await _session_id(request)
    if sid:
        await asyncio.to_thread(limiter.end, sid)
    return {"ok": True}


def _add_caller_identity(message: str, display_name: str) -> str:
    """Attach server-owned identity/tool policy to a full instruction update."""
    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return message
    if payload.get("type") != "session.update" or not isinstance(payload.get("session"), dict):
        return message
    session_data = payload["session"]
    # A later tools-only update is a patch. Adding an `instructions` key to it
    # would replace the complete personality prompt with only the caller name.
    if "instructions" not in session_data:
        return message
    identity = f"当前正在与你连线的观众名字是“{display_name}”。请自然地用这个名字称呼对方。"
    tool_policy = (
        "如果用户询问实时价格、最新新闻、近期公告或其他会变化的信息，并且会话提供了相关工具，"
        "必须在当前轮立即调用最合适的工具，禁止只说‘我去查’、‘稍等’或把调用拖到下一轮。"
        "收到工具结果后直接用中文回答用户；如果工具报错，要明确说明查询失败原因，不能假装已经查到。"
        "查询币价优先使用 mcp_coingecko_price；查询国际新闻优先使用 Exa 或 GDELT。"
    )
    instructions = str(session_data.get("instructions") or "").strip()
    additions = [item for item in (identity, tool_policy) if item not in instructions]
    if additions:
        session_data["instructions"] = "\n".join([instructions, *additions]).strip()
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


@app.websocket("/api/realtime")
async def room_realtime(websocket: WebSocket):
    """Authenticated one-at-a-time proxy to the internal S2S WebSocket."""
    if not LIVE_ROOM_ENABLED:
        await websocket.close(code=4404)
        return
    token = websocket.cookies.get(ROOM_COOKIE, "")
    ticket = websocket.query_params.get("ticket", "")
    try:
        session_id, display_name = await live_room.claim_websocket(token, ticket)
    except RoomError:
        await websocket.close(code=4403)
        return

    try:
        async with websockets.connect(
            S2S_INTERNAL_WS_URL,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as upstream:
            await websocket.accept()
            assistant_text: dict[str, str] = {}
            listen_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)

            async def listener_to_avatar() -> None:
                """Batch caller PCM so AVTR-1 can render active listening."""
                if not AVATAR_LISTEN_URL:
                    return
                async with httpx.AsyncClient(timeout=2.0) as client:
                    try:
                        while True:
                            chunk = bytearray(await listen_queue.get())
                            deadline = asyncio.get_running_loop().time() + 0.16
                            while len(chunk) < 6400:
                                remaining = deadline - asyncio.get_running_loop().time()
                                if remaining <= 0:
                                    break
                                try:
                                    chunk.extend(await asyncio.wait_for(listen_queue.get(), remaining))
                                except asyncio.TimeoutError:
                                    break
                            try:
                                await client.post(
                                    f"{AVATAR_LISTEN_URL}/listen-chunk",
                                    content=bytes(chunk),
                                    headers={"Content-Type": "application/octet-stream"},
                                )
                            except httpx.HTTPError as exc:
                                # Listening motion is cosmetic. A renderer
                                # hiccup must never tear down the voice call.
                                logger.debug("avatar listener tee failed: %r", exc)
                    finally:
                        with contextlib.suppress(Exception):
                            await client.post(f"{AVATAR_LISTEN_URL}/listen-reset")

            async def publish_room_transcript(message: str) -> None:
                try:
                    event = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    return
                event_type = str(event.get("type") or "")
                if event_type == "conversation.item.input_audio_transcription.completed":
                    await live_room.publish_transcript(
                        session_id=session_id,
                        event_id=str(event.get("item_id") or "user"),
                        role="user",
                        speaker=display_name,
                        text=str(event.get("transcript") or ""),
                    )
                    return
                if event_type in {
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                }:
                    response_id = str(event.get("response_id") or "assistant")
                    assistant_text[response_id] = assistant_text.get(response_id, "") + str(event.get("delta") or "")
                    await live_room.publish_transcript(
                        session_id=session_id,
                        event_id=response_id,
                        role="assistant",
                        speaker="小雅",
                        text=assistant_text[response_id],
                        partial=True,
                    )
                    return
                if event_type in {
                    "response.audio_transcript.done",
                    "response.output_audio_transcript.done",
                }:
                    response_id = str(event.get("response_id") or "assistant")
                    transcript = str(event.get("transcript") or assistant_text.get(response_id) or "")
                    assistant_text[response_id] = transcript
                    await live_room.publish_transcript(
                        session_id=session_id,
                        event_id=response_id,
                        role="assistant",
                        speaker="小雅",
                        text=transcript,
                    )
                    return
                response = event.get("response") if isinstance(event.get("response"), dict) else {}
                if event_type == "response.done" and response.get("status") == "cancelled":
                    response_id = str(response.get("id") or "")
                    if response_id and assistant_text.get(response_id):
                        await live_room.publish_transcript(
                            session_id=session_id,
                            event_id=response_id,
                            role="assistant",
                            speaker="小雅",
                            text=assistant_text[response_id],
                            interrupted=True,
                        )

            async def browser_to_upstream():
                while True:
                    event = await websocket.receive()
                    if event["type"] == "websocket.disconnect":
                        return
                    if event.get("text") is not None:
                        message = event["text"]
                        await upstream.send(_add_caller_identity(message, display_name))
                        if AVATAR_LISTEN_URL:
                            try:
                                payload = json.loads(message)
                                if payload.get("type") == "input_audio_buffer.append":
                                    pcm = base64.b64decode(payload.get("audio") or "", validate=True)
                                    if pcm:
                                        if listen_queue.full():
                                            listen_queue.get_nowait()
                                        listen_queue.put_nowait(pcm)
                            except (ValueError, TypeError, json.JSONDecodeError, asyncio.QueueFull):
                                pass
                    elif event.get("bytes") is not None:
                        await upstream.send(event["bytes"])

            async def upstream_to_browser():
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)
                        await publish_room_transcript(message)

            async def bridge():
                tasks = {
                    asyncio.create_task(browser_to_upstream()),
                    asyncio.create_task(upstream_to_browser()),
                }
                if AVATAR_LISTEN_URL:
                    tasks.add(asyncio.create_task(listener_to_avatar()))
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    task.result()

            await asyncio.wait_for(bridge(), timeout=live_room.max_call_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=1000, reason="本次连线时间已结束")
    except (WebSocketDisconnect, websockets.ConnectionClosed):
        pass
    except Exception as exc:
        logger.warning("room realtime proxy ended: %r", exc)
        with contextlib.suppress(RuntimeError):
            await websocket.close(code=1011, reason="语音服务连接中断")
    finally:
        await live_room.end_session(token, session_id)
        mention_replies.notify()


# Static front-end. Registered last so the /api routes win. `html=True` serves
# index.html at "/". The repo is public anyway, so serving the dir is fine.
app.mount("/", StaticFiles(directory=HERE, html=True), name="static")
