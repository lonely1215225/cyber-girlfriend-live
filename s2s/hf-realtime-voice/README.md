---
title: HF Realtime Voice
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Voice chat over WebSocket against a HF speech-to-speech
hf_oauth: true
hf_oauth_expiration_minutes: 10080
---

# Minimal Conversation App (S2S backend, **WebSocket** transport)

> 本项目的默认部署已在此基础上启用多人直播间模式：`room_manager.py`
> 管理随机身份、在线状态和 FIFO 连线队列，`room.js` 通过 SSE 实时渲染观众数、
> 当前连线人和排队列表。浏览器只能使用 `/api/session` 取得一次性票据，再通过
> `/api/realtime` 的同源代理连接内部 S2S，不会暴露可绕过队列的内部地址。

Drop-in alternative to [`amir-tfrere/minimal-conversation-app-s2s-backend`](https://huggingface.co/spaces/amir-tfrere/minimal-conversation-app-s2s-backend)
that uses the **WebSocket** route of the Hugging Face speech-to-speech
backend instead of the WebRTC SDP proxy. Same load balancer, same
`/session` handshake, same UI, same orb. Just a different wire.

## How it works

1. App POSTs `<lb_url>/session` (empty JSON body).
2. The LB picks a ready compute (round-robin) and returns:
   ```json
   {
     "session_id": "...",
     "websocket_url": "wss://<compute>/v1/realtime",
     "connect_url": "wss://<compute>/v1/realtime?session_token=<JWT>",
     "session_token": "<JWT>",
     "pending_timeout_s": 60
   }
   ```
3. App opens a WebSocket **directly** on `connect_url` (no rewrite to
   `https://`; unlike the WebRTC client which POSTs an SDP offer).
4. Server pushes `session.created` on connect. Client replies with
   `session.update` (OpenAI Realtime **GA** schema: `session.audio.input`,
   `session.audio.output`, `session.output_modalities`).
5. Client streams mic audio as PCM16 16 kHz mono base64 chunks
   (`input_audio_buffer.append`, one frame every ~40 ms).
6. Server pushes `response.output_audio.delta` (PCM16 24 kHz mono base64)
   and transcript deltas.

The backend exposes one concurrent session per compute (same as WebRTC
mode); the LB pins the session via a signed `session_token`.

## Why WebSocket instead of WebRTC

| | WebRTC (original) | WebSocket (this) |
|---|---|---|
| Transport | UDP + Opus 48 kHz + ICE/STUN | TCP + raw PCM16 |
| NAT traversal | needs STUN, can fail on corporate / cellular | none, works everywhere TCP is allowed |
| Audio quality | excellent (Opus, jitter buffer, FEC) | good (raw PCM, simple ring buffer) |
| Latency | lowest (~50-150 ms) | low (~150-300 ms typical) |
| Echo cancellation | browser AEC active on the WebRTC track | browser AEC active via `getUserMedia` constraints |
| Debuggability | needs `chrome://webrtc-internals` | `wscat` / DevTools network tab |
| Mobile data | sometimes blocked (UDP) | always works (HTTPS+WSS) |

## Backend requirement

This app talks to the WebSocket route `@app.websocket("/v1/realtime")`
defined in
[`websocket_router.py`](https://github.com/huggingface/speech-to-speech/blob/feat/webrtc-transport/src/speech_to_speech/api/openai_realtime/websocket_router.py)
on the **`feat/webrtc-transport`** branch. The same compute serves both
the WebRTC POST and the WebSocket upgrade on the same path; no backend
change required.

Smoke-test from the shell:

```bash
LB="${LOAD_BALANCER_URL:?Set LOAD_BALANCER_URL in your shell}"
curl -X POST "$LB/session" -H "Content-Type: application/json" -d '{}'
# -> { "connect_url": "wss://<compute>/v1/realtime?session_token=..." }
# Feed connect_url into a wscat / websocat and you should get a
# session.created event back immediately.
```

## Tools

The assistant can call two tools mid-conversation (toggle them from the **Tools**
button, top-right):

- **Web search** — Google results via Serper.dev, proxied server-side so the key
  never reaches the browser. Set `SERPER_API_KEY` as a Space secret. Without it,
  the tool is disabled unless the user pastes their own key in the Tools panel.
- **Camera** — while enabled, a live self-view shows bottom-left; when the model
  calls the tool, the current frame is sent to the vision-language model so it can
  see what you're showing it.

## Connecting to a backend

Three modes, picked by env (`/api/config` tells the client which one is active):

- **`SPEECH_TO_SPEECH_URL` env** — highest priority. The browser connects
  **directly** to this realtime WebSocket URL; it's shown read-only in Settings.
  Setting it disables the load-balancer logic entirely (no `/api/session` proxy,
  no queue, no metering, no sign-in). Unlike the LB address it is not a secret.
- **`LOAD_BALANCER_URL` env** — the original flow: the browser POSTs the
  same-origin `/api/session` proxy, the server forwards to the LB, and the
  browser dials the per-session compute URL the LB hands back. The LB address
  never reaches the browser; the Settings URL field is hidden. When a visitor
  signs in with Hugging Face, the proxy forwards their OAuth access token to the
  allocator through `X-Reachy-Mini-Authorization`; the token remains
  server-side. Set `REQUIRE_LOGIN=true` to reject anonymous allocation and ask
  visitors to sign in before starting.
- **Neither** — **Settings → Speech-to-speech server URL**: paste a full
  `connect_url` (`wss://host/v1/realtime?...`) or a bare host like `localhost:8080`
  (the app adds `/v1/realtime`), and the browser connects to it directly.

| `SPEECH_TO_SPEECH_URL` | `LOAD_BALANCER_URL` | `SPACE_ID` | Connection | URL field | Metering |
|:---:|:---:|:---:|---|---|---|
| ✅ | any | any | direct → pinned URL | visible, locked | off |
| – | ✅ | ✅ | LB proxy | hidden | **on** |
| – | ✅ | – | LB proxy | hidden | off |
| – | – | any | direct → user URL | editable | off |

**Settings → Restart** reconnects with the current voice, instructions and URL.

## Usage limits

Conversation time is metered per UTC day by sign-in tier (see `limiter.py` /
`auth.py`), but **only on the deployed Space** — metering turns on only when BOTH
`LOAD_BALANCER_URL` and `SPACE_ID` (injected automatically by the HF Space
runtime) are present. Running locally — even with `LOAD_BALANCER_URL` exported —
leaves the app unmetered. Tunable via env:

| Env | Default | What |
|-----|---------|------|
| `LIMIT_ANON_SEC` | `300` | Daily seconds for anonymous visitors (5 min) |
| `LIMIT_FREE_SEC` | `600` | Daily seconds for signed-in non-PRO users (10 min) |
| `REQUIRE_LOGIN` | unset | Set to `true` to require HF sign-in before allocating or claiming a session |
| `STARTUP_GREETING` | one-sentence greeting prompt | Hidden prompt that opens the conversation and warms the model; set empty to disable |
| `IDLE_PROMPT` | proactive topic prompt | Hidden prompt used after the active caller stays quiet; set empty to disable |
| `IDLE_PROMPT_MIN_SECONDS` / `IDLE_PROMPT_MAX_SECONDS` | `35` / `55` | Random quiet interval before the avatar starts a new topic |
| `UNLIMITED_ORGS` | _(adds to defaults)_ | Extra HF org names whose members get **unlimited** usage, like PRO |
| `USAGE_HASH_SECRET` | _(random)_ | HMAC secret for hashing identity keys + signing the anon cookie |

PRO members are always unlimited. Members of `cerebras`, `HuggingFaceM4`,
`smolagents`, and `pollen-robotics` are unlimited out of the box (shown as
"Team", not "PRO"); set `UNLIMITED_ORGS=my-team` to add more. Matched
case-insensitively against the user's organisations from HF OAuth.

## Run locally

The app is now a small FastAPI server (it serves the front-end *and* the search
proxy from one container).

```bash
pip install -r requirements.txt
export SERPER_API_KEY=...          # optional; web search is disabled without it
export SPEECH_TO_SPEECH_URL=...    # optional; pin a direct s2s server URL (overrides the LB)
export LOAD_BALANCER_URL=...       # optional; session-proxy flow (set a URL in Settings otherwise)
uvicorn server:app --reload --port 7860
# or, matching production: docker build -t s2s . && docker run -p 7860:7860 -e SERPER_API_KEY=... -e LOAD_BALANCER_URL=... s2s
```

Then open <http://localhost:7860/>, click the orb, allow the mic, talk.

> Browsers require **HTTPS or `localhost`** for `getUserMedia()` (mic + camera).
> `127.0.0.1` and `localhost` both work; plain `http://192.168.x.y` does NOT.

## Settings (stored in `localStorage`)

| Key | What |
|-----|------|
| Load balancer URL | Base URL of your S2S deployment. App POSTs `<lb>/session`. |
| Microphone | Input device for capture. Applies on the next conversation or Restart. |
| Speakers | Assistant-audio output. Chrome/Edge can switch live; other browsers use the system default. |
| Voice | Qwen3-TTS speaker name (Aiden, Ryan, Dylan, Eric, Ono_Anna, Serena, Sohee, Uncle_Fu, Vivian) |
| Instructions | System prompt sent in `session.update` once the WS opens |

LocalStorage keys are namespaced `s2s.ws.*` so this app's settings do
NOT collide with the WebRTC variant.

## Files

| File | Role |
|------|------|
| `index.html` | Single page, orb + settings modal (identical UI to the WebRTC app) |
| `main.js` | State machine, settings, tools, camera, noise-gate UI wiring |
| `ui/chat.js` | `ChatView`: history panel, ephemeral bubbles, transcript/tool streaming |
| `ui/account.js` | `Account`: HF login chip + popover, daily-limit modal |
| `ui/dom.js` | Shared helpers: `$`, `escHtml`, `truncateError`, `DEBUG` |
| `auth.py` | HF OAuth + per-request identity (tier, hashed keys) |
| `limiter.py` | SQLite per-day talk-time budget (chunked server-clock reservation) |
| `ws/s2s-ws-client.js` | WebSocket handshake + OpenAI Realtime GA protocol |
| `ws/user-audio-recorder.js` | Bounded browser-local PCM capture and WAV replay for user turns |
| `ws/codec.js` | base64 <-> PCM helpers + transcript extraction (pure) |
| `ws/orb-visualizer.js` | `OrbVisualiser`: FFT bands -> orb CSS custom properties |
| `worklets/mic-capture.js` | AudioWorklet: 48 kHz Float32 -> 16 kHz Int16 PCM, posts ~40 ms chunks |
| `worklets/audio-playback.js` | AudioWorklet: 24 kHz Float32 ring buffer -> 48 kHz, linear interp, fade in/out |
| `style.css` | Orb animations, layout, dark theme (verbatim from the WebRTC app) |

The chat history keeps the exact post-gate audio sent over WebSocket and exposes
it through a local replay control. Backend VAD also drives an immediate
“Listening…” → “Sending voice…” bubble; when STT is enabled, the same bubble and
history row are updated with the transcript.

## Audio pipeline notes

- **Input**: `getUserMedia({ echoCancellation, noiseSuppression, autoGainControl })`
  feeds the `mic-capture` worklet at the `AudioContext` rate. The worklet
  resamples to 16 kHz (boxcar lowpass + decimation on the 48 -> 16 fast
  path, linear interpolation fallback for odd rates) and packs Int16 LE.
- **Output**: `response.output_audio.delta` decodes to Int16 -> Float32
  and is posted to the `audio-playback` worklet. The worklet maintains a
  per-context ring buffer, linearly interpolates 24 -> 48, and applies
  short 32-frame fades on entry/exit to suppress clicks.
- **Barge-in**: when the server VAD detects user speech mid-response
  (`input_audio_buffer.speech_started` while `ai-speaking`), the client
  posts `{ kind: "clear" }` to the playback worklet to wipe the queue
  immediately. The server itself cancels the in-flight response.

## Credits

- Backend: [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) on `feat/webrtc-transport`
- UI verbatim from `amir-tfrere/minimal-conversation-app-s2s-backend` (Pollen Robotics × Hugging Face)
