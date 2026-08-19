# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Shared engine builders used by local CLI entry points.

Returns `(engine_config, engine_run)` for each supported engine kind, where
`engine_run` is the worklet-shaped callable `(EventBus, StreamClocks) -> Coroutine`.

Credentials are supplied inline via `EngineOptions` so the local UI can pass
them per-session instead of relying on process env vars.
"""

from collections.abc import Callable, Coroutine
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Discriminator

from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.conversation_engines.cartesia_client import CartesiaApiClient
from avaturn_live_streamer.conversation_engines.configs import (
    CartesiaConversationEngineConfig,
    ConversationEngineConfig,
    OpenAIRealtimeAPIConversationEngineConfig,
    OpenaiRealtimeApiVoice,
)
from avaturn_live_streamer.conversation_engines.realtime_api_client import RealtimeApiClient
from avaturn_live_streamer.event_bus import EventBus

EngineKind = Literal["openai", "cartesia"]
ENGINE_KINDS: tuple[EngineKind, ...] = ("openai", "cartesia")

type EngineRun = Callable[[EventBus, StreamClocks], Coroutine[None, None, None]]
type BuiltEngine = tuple[ConversationEngineConfig, EngineRun]

_CARTESIA_TOKEN_URL = "https://api.cartesia.ai/access-token"
_CARTESIA_VERSION = "2025-04-16"

DEFAULT_OPENAI_PROMPT = (
    "You are a friendly, concise voice assistant. Speak naturally and keep "
    "answers under 50 words. Avoid emojis or unreadable symbols."
)
DEFAULT_OPENAI_VOICE: OpenaiRealtimeApiVoice = "shimmer"
DEFAULT_OPENAI_MODEL = "gpt-realtime-2"


class OpenAIEngineOptions(BaseModel):
    type: Literal["openai"] = "openai"
    api_key: str
    model: str = DEFAULT_OPENAI_MODEL
    prompt: str = DEFAULT_OPENAI_PROMPT
    voice: OpenaiRealtimeApiVoice = DEFAULT_OPENAI_VOICE


class CartesiaEngineOptions(BaseModel):
    type: Literal["cartesia"] = "cartesia"
    api_key: str
    agent_id: str


EngineOptions = Annotated[
    OpenAIEngineOptions | CartesiaEngineOptions,
    Discriminator("type"),
]


async def _mint_cartesia_token(api_key: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.post(
            _CARTESIA_TOKEN_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Cartesia-Version": _CARTESIA_VERSION,
            },
            json={"grants": {"agent": True}, "expires_in": 300},
        )
        r.raise_for_status()
        token = r.json().get("token")
        if not token:
            raise RuntimeError("Cartesia access-token response missing 'token' field")
        return token


async def mint_openai_realtime_secret(
    *,
    api_key: str,
    model: str = DEFAULT_OPENAI_MODEL,
    prompt: str = DEFAULT_OPENAI_PROMPT,
    voice: OpenaiRealtimeApiVoice = DEFAULT_OPENAI_VOICE,
    tracing: dict[str, object] | str | None = "auto",
) -> str:
    from openai import AsyncClient

    oai = AsyncClient(api_key=api_key)
    session: dict[str, object] = {
        "type": "realtime",
        "model": model,
        "audio": {
            "input": {
                "turn_detection": {"type": "semantic_vad", "eagerness": "high"},
            },
            "output": {"voice": voice},
        },
    }
    if prompt.strip():
        session["instructions"] = prompt
    if tracing is not None:
        session["tracing"] = tracing
    secret = await oai.realtime.client_secrets.create(
        expires_after={"seconds": 7200, "anchor": "created_at"},
        session=session,  # pyright: ignore [reportArgumentType]
    )
    return secret.value


async def build_cartesia(
    *,
    stream_id: str,
    options: CartesiaEngineOptions,
) -> BuiltEngine:
    token = await _mint_cartesia_token(options.api_key)
    cfg = CartesiaConversationEngineConfig(access_token=token, agent_id=options.agent_id)
    return cfg, CartesiaApiClient(cfg, stream_id=stream_id).run


async def build_openai(
    *,
    stream_id: str,
    options: OpenAIEngineOptions,
) -> BuiltEngine:
    tracing: dict[str, object] = {
        "workflow_name": "avaturn-live-local",
        "group_id": stream_id,
        "metadata": {"engine": "openai-realtime", "stream_id": stream_id},
    }
    secret = await mint_openai_realtime_secret(
        api_key=options.api_key,
        model=options.model,
        prompt=options.prompt,
        voice=options.voice,
        tracing=tracing,
    )
    cfg = OpenAIRealtimeAPIConversationEngineConfig(client_secret=secret)
    return cfg, RealtimeApiClient(cfg).run


async def build_engine(options: EngineOptions, *, stream_id: str) -> BuiltEngine:
    match options:
        case OpenAIEngineOptions():
            return await build_openai(stream_id=stream_id, options=options)
        case CartesiaEngineOptions():
            return await build_cartesia(stream_id=stream_id, options=options)
