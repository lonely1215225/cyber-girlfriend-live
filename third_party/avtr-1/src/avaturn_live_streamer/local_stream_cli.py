# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Local-dev CLI: serves a tiny FastAPI app on localhost that lets a browser
peer connect via WebRTC, configure a conversation engine in the UI (OpenAI
Realtime / Cartesia, with credentials kept in the browser's localStorage), and
runs the same stream pipeline as `run-session` but with a `LocalRTC` peer
(aiortc-backed) in place of Daily.

No Daily API key or Daily SDK frontend required, and no OPENAI__API_KEY /
CARTESIA_* env vars: credentials are POSTed per-session and the server mints
short-lived tokens for each.

Sessions are one-at-a-time but reusable: disconnect and click Start again for
a new one.

Invoke via:
    uv run python -m avaturn_live_streamer.local_stream_cli [--host 0.0.0.0] [--port 7860]
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

import httpx
import typer
import uvicorn
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from avaturn_live_streamer.core.logs import get_logger, setup_logging
from avaturn_live_streamer.settings import get_config
from avaturn_live_streamer.types import BackgroundId, RendererAvatarId
from avaturn_live_streamer.clocks import StreamClocks
from avaturn_live_streamer.config import RendererConfig
from avaturn_live_streamer.conversation_engines.builders import (
    BuiltEngine,
    EngineOptions,
    build_engine,
)
from avaturn_live_streamer.event_bus import EventBus
from avaturn_live_streamer.events import Shutdown
from avaturn_live_streamer.localrtc import (
    LocalRTC,
    LocalRTCWorklet,
    has_turn,
    resolve_ice_servers,
    serialize_ice_servers,
)


def _filter_sdp_to_relay_only(sdp: str) -> str:
    """Strip ``typ host`` and ``typ srflx`` candidate lines from an SDP answer.

    aiortc's ``RTCConfiguration`` in this version doesn't accept
    ``iceTransportPolicy``, so it always gathers all candidate types. When the
    server is on AWS behind an SG that blocks inbound UDP, the host/srflx
    candidates (private/docker IPs and the public IP) are useless to the
    browser; worse, Cloudflare TURN refuses ``CreatePermission`` for the RFC1918
    addresses and Firefox handles that by tearing down the whole TURN
    allocation -- killing the relay-relay pair too. By dropping the unusable
    candidates from the SDP we keep only the relay candidate, which the browser
    can reach via Cloudflare.
    """
    out: list[str] = []
    for line in sdp.splitlines():
        if line.startswith("a=candidate:") and (" typ host" in line or " typ srflx" in line):
            continue
        out.append(line)
    # Preserve trailing newline behavior of the original.
    return "\r\n".join(out) + ("\r\n" if sdp.endswith(("\n", "\r\n")) else "")
from avaturn_live_streamer.renderer import create_renderer_client_registry
from avaturn_live_streamer.runner import run_stream
from avaturn_live_streamer.types import PixelFormat
from avaturn_live_streamer.worklets.delayed_event import run_delayed_event_worklet
from avaturn_live_streamer.worklets.rendering import RenderingWorklet
from avaturn_live_streamer.worklets.timeout import TimeoutWorklet

_LOGGER = get_logger()

_UI_HTML_PATH = Path(__file__).parent / "local_stream_ui.html"


class _LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"


class _OfferBody(BaseModel):
    engine: EngineOptions
    sdp: str
    type: Literal["offer"]
    avatar_id: str
    background_id: str


_RENDERER_MODEL = "avtrn-1"


def _renderer_base_url() -> str | None:
    cfg = get_config()
    rc = cfg.renderers.renderers.get(_RENDERER_MODEL)
    if rc is None or not rc.lb_or_instance_url:
        return None
    # For `single` mode this is the renderer instance itself; for `load-balanced`
    # mode it's the LB. Both serve /avatars from the renderer registry.
    return rc.lb_or_instance_url.rstrip("/")


class _SessionSlot:
    """One concurrent session at a time. Slot is released when the session task ends."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def claim(self) -> None:
        async with self._lock:
            if self._task is not None and not self._task.done():
                raise HTTPException(status_code=409, detail="Another session is active")
            self._task = None

    def attach(self, task: asyncio.Task[None]) -> None:
        self._task = task
        task.add_done_callback(self._on_done)

    def _on_done(self, task: asyncio.Task[None]) -> None:
        if task is self._task:
            self._task = None
        if task.cancelled():
            _LOGGER.info("local-stream session cancelled; slot free")
            return
        if (exc := task.exception()) is not None:
            _LOGGER.error("local-stream session failed; slot free", error=str(exc), exc_info=exc)
            return
        _LOGGER.info("local-stream session ended; slot free")


async def _run_session(
    *,
    built_engine: BuiltEngine,
    peer: LocalRTC,
    avatar: str,
    background: str,
    idle_timeout: int,
    max_duration: int,
) -> None:
    cfg = get_config()
    renderer_registry = create_renderer_client_registry(cfg.renderers)
    # Build the renderer config directly — the session/avatar DB is not vendored
    # for the localrtc-only slice; `avatar` here is the renderer-side avatar id.
    renderer_config = RendererConfig(
        avatar_id=RendererAvatarId(avatar),
        background_id=BackgroundId(background),
        pixel_format=PixelFormat.YUV_I420,
        model="avtrn-1",
    )
    _engine_config, engine_run = built_engine
    pixel_format = renderer_config.pixel_format

    rendering = RenderingWorklet(renderer_registry, renderer_config)
    peer_worklet = LocalRTCWorklet(peer, pixel_format)
    timeout_worklet = TimeoutWorklet(idle_timeout, max_duration)

    needs_stop = asyncio.Event()

    async def _emit_stop(bus: EventBus, _clock: StreamClocks):
        _ = _clock

        async def _set_stop():
            async with bus.subscribe(Shutdown) as sub:
                bus.ready()
                await sub.get_next()
                needs_stop.set()

        asyncio.create_task(_set_stop())
        await needs_stop.wait()
        await bus.publish(Shutdown())

    try:
        await run_stream(
            rendering.run,
            peer_worklet.run,
            run_delayed_event_worklet,
            engine_run,
            timeout_worklet.run,
            _emit_stop,
        )
    finally:
        await peer.close()


def _make_app(
    *,
    idle_timeout: int,
    max_duration: int,
) -> FastAPI:
    app = FastAPI()
    slot = _SessionSlot()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _UI_HTML_PATH.read_text()

    @app.get("/avatars")
    async def avatars_route() -> dict[str, object]:
        """Proxy of the renderer's ``/avatars`` endpoint.

        The renderer auto-discovers loaded avatars and backgrounds at startup;
        we surface both lists to the browser so the user can pick known-good
        ids. The browser UI is solely responsible for what's selected per
        session.
        """
        base = _renderer_base_url()
        result: dict[str, object] = {"avatars": [], "backgrounds": []}
        if not base:
            result["error"] = "renderer URL not configured"
            return result
        try:
            async with httpx.AsyncClient(timeout=5.0) as http:
                r = await http.get(f"{base}/avatars")
                r.raise_for_status()
                data = r.json()
        except Exception as exc:
            _LOGGER.warning("renderer /avatars proxy failed", error=str(exc))
            result["error"] = f"renderer /avatars failed: {exc}"
            return result
        if isinstance(data, dict):
            avatars_list = data.get("avatars")
            if isinstance(avatars_list, list):
                result["avatars"] = [str(a) for a in avatars_list]
            backgrounds_list = data.get("backgrounds")
            if isinstance(backgrounds_list, list):
                result["backgrounds"] = [str(b) for b in backgrounds_list]
        return result

    @app.get("/ice-servers")
    async def ice_servers_route() -> dict[str, object]:
        servers = await resolve_ice_servers()
        # Keep iceTransportPolicy as "all" even when TURN is configured. The
        # probe established that the working pair is server-TURN-relay <->
        # browser-srflx (the browser's home public IP via STUN, reached because
        # the home router is cone NAT). Forcing relay drops srflx from the
        # browser's offer and -- combined with whatever Cloudflare TURN does
        # for relay-to-relay -- aioice never actually starts ICE checks. Letting
        # the browser advertise host + srflx + relay lets ICE pick the path
        # that's known to work.
        return {
            "iceServers": serialize_ice_servers(servers),
            "iceTransportPolicy": "all",
        }

    @app.post("/probe-offer")
    async def probe_offer(body: dict[str, str]) -> dict[str, str]:
        """ICE-only probe: negotiate a PC with policy=all and close it shortly.

        Used by the browser's connectivity check to discover whether the
        selected candidate pair is direct (host/srflx) or relayed (relay), i.e.
        whether the server is actually reachable on UDP from the browser.
        Does NOT touch the session slot or start any pipeline.
        """
        sdp = body.get("sdp")
        type_ = body.get("type")
        if not sdp or type_ != "offer":
            raise HTTPException(status_code=400, detail="missing sdp/type=offer")
        servers = await resolve_ice_servers()
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=servers),  # policy=all (default)
        )
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=type_))
        answer = await pc.createAnswer()
        assert answer is not None
        await pc.setLocalDescription(answer)

        async def _close_later() -> None:
            await asyncio.sleep(15.0)
            try:
                await pc.close()
            except Exception:
                pass

        asyncio.create_task(_close_later())
        local = pc.localDescription
        sdp_out = local.sdp
        if has_turn(servers):
            sdp_out = _filter_sdp_to_relay_only(sdp_out)
        return {"sdp": sdp_out, "type": local.type}

    @app.post("/offer")
    async def offer(body: _OfferBody) -> dict[str, str]:
        await slot.claim()

        # Build the conversation engine up front (mints any tokens) so credential
        # / network errors surface as HTTP 4xx/5xx here instead of disappearing
        # into the fire-and-forget session task.
        try:
            built_engine = await build_engine(body.engine, stream_id="local")
        except HTTPException:
            raise
        except Exception as exc:
            _LOGGER.warning("engine build failed", error=str(exc))
            raise HTTPException(status_code=400, detail=f"engine build failed: {exc}") from exc

        ice_servers = await resolve_ice_servers()
        pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=ice_servers))
        peer = LocalRTC(pc)

        # Diagnostic: log the candidate types the browser actually sent us. If
        # the offer doesn't include relay candidates, the wait-for-gathering
        # change on the JS side didn't take effect.
        remote_candidate_lines = [
            line for line in body.sdp.splitlines() if line.startswith("a=candidate:")
        ]
        _LOGGER.info(
            "/offer remote candidates",
            count=len(remote_candidate_lines),
            candidates=remote_candidate_lines,
        )

        await pc.setRemoteDescription(RTCSessionDescription(sdp=body.sdp, type=body.type))
        answer = await pc.createAnswer()
        assert answer is not None
        await pc.setLocalDescription(answer)

        # Diagnostic: log the candidates the server gathered. If our SDP filter
        # is overly aggressive (or aiortc didn't gather any relay), we'll see it.
        local_candidate_lines = [
            line for line in pc.localDescription.sdp.splitlines() if line.startswith("a=candidate:")
        ]
        _LOGGER.info(
            "/offer local candidates",
            count=len(local_candidate_lines),
            candidates=local_candidate_lines,
        )

        chosen_avatar = body.avatar_id.strip()
        if not chosen_avatar:
            raise HTTPException(status_code=400, detail="avatar_id is required")
        chosen_background = body.background_id.strip()
        if not chosen_background:
            raise HTTPException(status_code=400, detail="background_id is required")
        task = asyncio.create_task(
            _run_session(
                built_engine=built_engine,
                peer=peer,
                avatar=chosen_avatar,
                background=chosen_background,
                idle_timeout=idle_timeout,
                max_duration=max_duration,
            )
        )
        slot.attach(task)

        local = pc.localDescription
        sdp_out = local.sdp
        if has_turn(ice_servers):
            sdp_out = _filter_sdp_to_relay_only(sdp_out)
        return {"sdp": sdp_out, "type": local.type}

    return app


def run_local_stream(
    host: Annotated[
        str, typer.Option(help="Bind address (use 0.0.0.0 for remote/Docker)")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Local server port")] = 8081,
    idle_timeout: Annotated[int, typer.Option(help="Idle timeout (s)")] = 30,
    max_duration: Annotated[int, typer.Option(help="Max session duration (s)")] = 3600,
    log_level: Annotated[_LogLevel, typer.Option()] = _LogLevel.INFO,
) -> None:
    """Start a local WebRTC streaming session server (aiortc-backed).

    Opens a localhost UI; pick an engine in the browser and click Start.
    Requires aiortc/av (dev dependencies). Does not use Daily.
    """
    cfg = get_config()
    cfg.logging.level = log_level.value
    setup_logging(cfg.logging)

    # httpx logs every renderer request at INFO; demote to WARNING.
    import logging as _logging

    _logging.getLogger("httpx").setLevel(_logging.WARNING)
    _logging.getLogger("httpcore").setLevel(_logging.WARNING)

    if not _UI_HTML_PATH.exists():
        raise RuntimeError(f"UI HTML missing at {_UI_HTML_PATH}")

    app = _make_app(
        idle_timeout=idle_timeout,
        max_duration=max_duration,
    )
    _LOGGER.info("local-stream server starting", host=host, port=port)
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Open http://{display_host}:{port}/ in your browser (or the host's reachable address).")

    uvicorn.run(app, host=host, port=port, log_level=log_level.value.lower())


if __name__ == "__main__":
    typer.run(run_local_stream)
