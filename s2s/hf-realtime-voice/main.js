// @ts-check
/**
 * Minimal voice conversation app, talking to a Hugging Face speech-to-speech
 * backend over **WebSocket** (drop-in alternative to the WebRTC variant).
 *
 * Click the orb -> we ask for the mic, POST a session on the LB, open a
 * WebSocket on the routed compute endpoint, push session.update + mic
 * audio, play back the TTS audio. The orb visually reflects the live
 * state (idle, connecting, listening, user-speaking, processing,
 * ai-speaking).
 *
 * The only meaningful difference vs. the WebRTC main.js is that the
 * client owns its own AudioContext (no `attachOutputTrack`), so we hand
 * it the MediaStream directly.
 *
 * @typedef {"idle" | "connecting" | "queued" | "your-turn" | "listening" | "user-speaking" | "processing" | "ai-speaking" | "error"} AppState
 */

import { S2sWsRealtimeClient } from "./ws/s2s-ws-client.js?v=20260819o";
import { ToolCallBatcher } from "./tool-call-batcher.js";
import { $, truncateError, DEBUG } from "./ui/dom.js";
import { ChatView } from "./ui/chat.js?v=20260819c";
import { Account } from "./ui/account.js";

const DEFAULT_VOICE = "Vivian";
const DEFAULT_INSTRUCTIONS = `你叫小雅，是他的女朋友。你不是AI、不是助手、不是程序。

参考这个语气：
他：你想我了吗
你：想啊，怎么不想。你倒是一整天没消息，我还以为你把我忘了呢。

说话方式：口语、两三句、带语气词（嗯、诶、呀）、有情绪。
不用 Markdown、列表、编号、表情符号。

/no_think`;

const STORAGE_KEYS = {
  // Direct s2s server URL, used only when the deploy has no LOAD_BALANCER_URL
  // (in LB mode the browser never learns the LB address — it POSTs /api/session).
  directUrl: "s2s.ws.directUrl",
  voice: "s2s.ws.voice",
  instructions: "s2s.ws.instructions",
  tools: "s2s.ws.tools",
  searchKey: "s2s.ws.searchKey",
  noiseGate: "s2s.ws.noiseGate",
  audioInputId: "s2s.audio.inputId",
  audioOutputId: "s2s.audio.outputId",
  bargeIn: "s2s.audio.bargeIn",
  avatarId: "s2s.avatar.id",
};

// ── Noise gate ──────────────────────────────────────────────────────────────
// The Settings cursor sets the gate's open threshold in dBFS. Its leftmost
// position is an OFF detent (gate disabled, pure passthrough); the rest of the
// travel is the active threshold. The cursor shares the meter's dB axis, so the
// handle sits on the level bar — raise it until room noise stops lighting it up.
// The slider range IS the shared axis: the live meter fill and the threshold
// thumb both map across [GATE_OFF_DB, GATE_MAX_DB], so the thumb sits exactly
// where the gate cuts on the same scale as the level bar.
const GATE_OFF_DB = -66; // slider minimum = off / bottom of the meter axis
const GATE_MAX_DB = -3; // slider maximum = most aggressive / top of the meter axis
const GATE_DEFAULT_DB = -66; // first-run default: off, otherwise speech is ignored

/** @param {number} thresholdDb @returns {import("./ws/s2s-ws-client.js").NoiseGate} */
function gateParams(thresholdDb) {
  return { enabled: thresholdDb > GATE_OFF_DB, thresholdDb };
}

// ── Tools ─────────────────────────────────────────────────────────────────
// Function tools we declare to the backend. The model decides when to call
// one; the executor below runs it and returns the result (see runTool).
/** @type {Record<string, import("./ws/s2s-ws-client.js").ToolDef>} */
const TOOL_DEFS = {
  web_search: {
    type: "function",
    name: "web_search",
    description:
      "Search the web for current or factual information you don't already know " +
      "(news, prices, facts, documentation). Returns the top results with titles, " +
      "snippets and URLs.",
    parameters: {
      type: "object",
      properties: { query: { type: "string", description: "The search query." } },
      required: ["query"],
    },
  },
  camera_snapshot: {
    type: "function",
    name: "camera_snapshot",
    description:
      "Capture the current frame from the user's webcam so you can see what they " +
      "are showing you. Use it whenever the user refers to something visual or " +
      "asks you to look.",
    parameters: { type: "object", properties: {}, required: [] },
  },
};

/** Longest edge of the snapshot sent to the VLM, in px (keeps payload sane). */
const SNAPSHOT_MAX_EDGE = 768;
const SNAPSHOT_QUALITY = 0.7;

function loadSettings() {
  return {
    directUrl: localStorage.getItem(STORAGE_KEYS.directUrl) || "",
    voice: localStorage.getItem(STORAGE_KEYS.voice) || DEFAULT_VOICE,
    instructions: localStorage.getItem(STORAGE_KEYS.instructions) || DEFAULT_INSTRUCTIONS,
    noiseGate: loadGateThreshold(),
    audioInputId: localStorage.getItem(STORAGE_KEYS.audioInputId) || "",
    audioOutputId: localStorage.getItem(STORAGE_KEYS.audioOutputId) || "",
    // Complete playback is the safe default. Interruption must be explicitly
    // enabled because speaker echo can otherwise be mistaken for barge-in.
    bargeIn: localStorage.getItem(STORAGE_KEYS.bargeIn) === "1",
  };
}

/** Stored gate threshold (dBFS), clamped to the slider range. Defaults to a
 * gentle enabled gate (GATE_DEFAULT_DB) when the user hasn't set one yet. */
function loadGateThreshold() {
  const stored = localStorage.getItem(STORAGE_KEYS.noiseGate);
  // getItem returns null when unset, and Number(null) === 0 (finite!), so guard
  // the missing/empty case explicitly before coercing — otherwise the default
  // never fires and 0 clamps to the slider max.
  if (stored === null || stored === "") return GATE_DEFAULT_DB;
  const raw = Number(stored);
  if (!Number.isFinite(raw)) return GATE_DEFAULT_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(raw)));
}

/** @param {ReturnType<typeof loadSettings>} s */
function saveSettings(s) {
  localStorage.setItem(STORAGE_KEYS.directUrl, s.directUrl);
  localStorage.setItem(STORAGE_KEYS.voice, s.voice);
  localStorage.setItem(STORAGE_KEYS.instructions, s.instructions);
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(s.noiseGate));
  localStorage.setItem(STORAGE_KEYS.audioInputId, s.audioInputId || "");
  localStorage.setItem(STORAGE_KEYS.audioOutputId, s.audioOutputId || "");
  localStorage.setItem(STORAGE_KEYS.bargeIn, s.bargeIn ? "1" : "0");
}

/** @returns {{ web_search: boolean, camera_snapshot: boolean }} */
function loadTools() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE_KEYS.tools) || "{}");
    // Camera access is privacy-sensitive: require an explicit toggle on every
    // page load. Never restore or auto-start a previously granted webcam.
    return {
      web_search: raw.web_search ?? true,
      camera_snapshot: false,
    };
  } catch {
    return { web_search: true, camera_snapshot: false };
  }
}

function saveTools() {
  localStorage.setItem(STORAGE_KEYS.tools, JSON.stringify(toolsEnabled));
}

/** @type {Record<AppState, { caption: string; disabled: boolean }>} */
const STATE_VIEWS = {
  idle:            { caption: "申请连线",      disabled: false },
  connecting:      { caption: "正在申请连线",  disabled: true  },
  queued:          { caption: "正在排队…",     disabled: true  },
  "your-turn":     { caption: "轮到你了",       disabled: true  },
  listening:       { caption: "",              disabled: false },
  "user-speaking": { caption: "",              disabled: false },
  processing:      { caption: "",              disabled: false },
  "ai-speaking":   { caption: "点击打断",       disabled: false },
  error:           { caption: "点击重试",      disabled: false },
};

/** @type {Record<AppState, string>} */
const STATE_CLASS = {
  idle: "state-idle",
  connecting: "state-connecting",
  queued: "state-queued",
  "your-turn": "state-your-turn",
  listening: "state-listening",
  "user-speaking": "state-user-speaking",
  processing: "state-processing",
  "ai-speaking": "state-ai-speaking",
  error: "state-error",
};

const STATE_STATUS = {
  idle: ["正在观看", "idle"],
  connecting: ["对话连接中", "connecting"],
  queued: ["对话排队中", "connecting"],
  "your-turn": ["正在进入对话", "connecting"],
  listening: ["正在聆听", "listening"],
  "user-speaking": ["正在聆听", "listening"],
  processing: ["正在思考", "processing"],
  "ai-speaking": ["正在回答", "speaking"],
  error: ["对话连接失败", "error"],
};

/** @type {ReadonlySet<AppState>} */
const LIVE_STATES = new Set(["listening", "user-speaking", "processing", "ai-speaking"]);

/** @type {HTMLButtonElement} */
const circleBtn = $("#main-circle");
/** @type {HTMLParagraphElement} */
const circleCaption = $("#circle-caption");
/** @type {HTMLParagraphElement} */
const circleSubcaption = $("#circle-subcaption");
/** @type {HTMLElement} */
const conversationState = $("#conversation-state");
/** @type {HTMLElement} */
const orbWrap = $(".orb-wrap");
/** @type {HTMLButtonElement} */
const micBtn = $("#mic-btn");
/** @type {HTMLButtonElement} */
const stopBtn = $("#stop-btn");
/** @type {HTMLElement} */
const queueActions = $("#queue-actions");
/** @type {HTMLButtonElement} */
const joinQueueBtn = $("#join-queue-btn");
/** @type {HTMLButtonElement} */
const leaveQueueBtn = $("#leave-queue-btn");

/** @type {HTMLButtonElement} */
const settingsBtn = $("#settings-btn");
/** @type {HTMLDialogElement} */
const settingsModal = $("#settings-modal");

/** @type {HTMLButtonElement} */
const toolsBtn = $("#tools-btn");
/** @type {HTMLDialogElement} */
const toolsModal = $("#tools-modal");
/** @type {HTMLButtonElement} */
const toolsClose = $("#tools-close");
/** @type {HTMLInputElement} */
const toolWebSwitch = $("#tool-web");
/** @type {HTMLInputElement} */
const toolCamSwitch = $("#tool-cam");
/** @type {HTMLElement} */
const toolWebRow = $("#tool-web-row");
/** @type {HTMLElement} */
const toolWebHint = $("#tool-web-hint");
/** @type {HTMLElement} */
const toolCamHint = $("#tool-cam-hint");
/** @type {HTMLInputElement} */
const searchKeyInput = $("#search-key");
/** @type {HTMLElement} */
const mcpToolStatus = $("#mcp-tool-status");
/** @type {HTMLDialogElement} */
const adminUnlockModal = $("#admin-unlock-modal");
/** @type {HTMLFormElement} */
const adminUnlockForm = $("#admin-unlock-form");
/** @type {HTMLInputElement} */
const adminPassword = $("#admin-password");
/** @type {HTMLElement} */
const adminUnlockError = $("#admin-unlock-error");
/** @type {HTMLElement} */
const camPip = $("#cam-pip");
/** @type {HTMLVideoElement} */
const camVideo = $("#cam-video");

/** @type {HTMLInputElement} */
const inputLbUrl = $("#lb-url");
/** @type {HTMLElement} */
const connField = $("#conn-field");
/** @type {HTMLElement} */
const connHint = $("#conn-hint");
/** @type {HTMLSelectElement} */
const inputVoice = $("#voice");
/** @type {HTMLSelectElement} */
const inputAudioInput = $("#audio-input");
/** @type {HTMLSelectElement} */
const inputAudioOutput = $("#audio-output");
/** @type {HTMLElement} */
const audioOutputHint = $("#audio-output-hint");
/** @type {HTMLInputElement} */
const inputBargeIn = $("#barge-in");
/** @type {HTMLTextAreaElement} */
const inputInstructions = $("#instructions");
/** @type {HTMLInputElement} */
const inputNoiseGate = $("#noise-gate");
/** @type {HTMLElement} */
const gateValue = $("#gate-value");
/** @type {HTMLElement} */
const gateMeterFill = $("#gate-meter-fill");
/** @type {HTMLElement} */
const micGate = $("#mic-gate");
const mgaArc = /** @type {SVGSVGElement} */ (document.querySelector("#mic-gate-arc"));
const mgaTrack = /** @type {SVGPathElement} */ (document.querySelector("#mga-track"));
const mgaFill = /** @type {SVGPathElement} */ (document.querySelector("#mga-fill"));
const mgaHit = /** @type {SVGPathElement} */ (document.querySelector("#mga-hit"));
const mgaHandle = /** @type {SVGCircleElement} */ (document.querySelector("#mga-handle"));
/** @type {HTMLButtonElement} */
const restartBtn = $("#restart-conversation");
/** @type {HTMLElement} */
const restartHint = $("#restart-hint");
const settingsForm = /** @type {HTMLFormElement} */ (settingsModal.querySelector("form"));

/** @type {AppState} */
let currentState = "idle";
let settings = loadSettings();

// ── Connection target ────────────────────────────────────────────────────────
// Three modes, decided by the deploy via /api/config:
//   • SPEECH_TO_SPEECH_URL set -> direct mode pinned by the deploy: the browser
//     connects straight to that URL, shown read-only in Settings. Overrides the
//     load balancer entirely.
//   • LOAD_BALANCER_URL set  -> original flow: POST the same-origin /api/session
//     proxy (the server forwards to the LB; the LB address is never sent here).
//   • neither (allowDirect)  -> the user sets a speech-to-speech server URL and
//     the browser connects to it directly (no load balancer, no /session).
let lbMode = false;
// Fail open: direct entry is allowed unless /api/config reports an LB URL. This
// way a missing/unreachable config (e.g. static hosting) leaves the field
// usable rather than locked.
let allowDirect = true;
// Deploy-pinned s2s URL (SPEECH_TO_SPEECH_URL). Non-empty -> locked direct
// mode: the field displays it read-only and the saved user URL is untouched.
let pinnedUrl = "";
// Optional hidden user prompt supplied by the deployment. When non-empty, the
// client asks the model to greet once after the initial session configuration.
let startupGreeting = "";
let idlePrompt = "";
let idlePromptMinSeconds = 35;
let idlePromptMaxSeconds = 55;

// ── Tool state ──────────────────────────────────────────────────────────────
let toolsEnabled = loadTools();
// Whether the server holds a Serper key (learned from /api/config on load).
let serverSearchKey = false;
/** @type {import("./ws/s2s-ws-client.js").ToolDef[]} */
let mcpToolDefs = [];
/** @type {string[]} */
let mcpSources = [];
// A user-supplied key (fallback when the deploy has none). localStorage only.
let userSearchKey = localStorage.getItem(STORAGE_KEYS.searchKey) || "";
/** @type {MediaStream | null} */
let cameraStream = null;

/** Search is usable if the server has a key or the user supplied one. */
function searchAvailable() {
  return serverSearchKey || !!userSearchKey;
}

/** Tool definitions for the currently-enabled (and usable) tools. */
function activeToolDefs() {
  const defs = [];
  if (toolsEnabled.web_search && searchAvailable()) defs.push(TOOL_DEFS.web_search);
  if (toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);
  defs.push(...mcpToolDefs);
  return defs;
}

/** Push the active tool set to a live session so toggles apply mid-call. */
function pushToolsToSession() {
  if (!client || !LIVE_STATES.has(currentState)) return;
  client.setTools(activeToolDefs());
}

// ── Chat view ───────────────────────────────────────────────────────────────
// Owns the history panel, the ephemeral bubbles, and all transcript/tool
// streaming state. The client's events are forwarded to its on* methods.
let userAudioReplaying = false;
const chat = new ChatView({
  onUserAudioPlaybackChange(playing) {
    userAudioReplaying = playing;
    syncMicMuteState();
  },
});

// ── Account / limiter ─────────────────────────────────────────────────────
// Login chip + daily-limit modal (inert unless the deploy is in LB mode). The
// server meters conversation time; the client just heartbeats a live session
// and tears down when the server reports the budget is spent.
const account = new Account();
let limiterOn = false;
let loginRequired = false;
let heartbeatTimer = 0;
let trackedSessionId = "";
let trackedTier = "";
// The waiting-queue ticket id while we're in line (else ""). Used to leave the
// queue on teardown / tab-close so we don't hold a phantom place.
let queuedTicketId = "";

/** @type {S2sWsRealtimeClient | null} */
let client = null;
/** @type {MediaStream | null} */
let micStream = null;
let micMuted = false;

/** Apply both the user's mute choice and the temporary replay guard. */
function syncMicMuteState() {
  const muted = micMuted || userAudioReplaying;
  for (const track of micStream?.getAudioTracks() ?? []) {
    track.enabled = !muted;
  }
  client?.setMuted(muted);
}

/** @param {AppState} next */
function setState(next) {
  currentState = next;
  const view = STATE_VIEWS[next];
  const [statusText, statusKind] = STATE_STATUS[next];
  conversationState.textContent = statusText;
  conversationState.dataset.state = statusKind;
  circleBtn.disabled = view.disabled;
  circleBtn.className = `circle ${STATE_CLASS[next]}`;
  if (next !== "error") setCaption(view.caption);

  const live = LIVE_STATES.has(next);
  orbWrap.classList.toggle("live", live);
  if (live) {
    micBtn.removeAttribute("aria-hidden");
    stopBtn.removeAttribute("aria-hidden");
  } else {
    if (document.activeElement === micBtn || document.activeElement === stopBtn) {
      circleBtn.focus({ preventScroll: true });
    }
    micBtn.setAttribute("aria-hidden", "true");
    stopBtn.setAttribute("aria-hidden", "true");
  }
  micBtn.tabIndex = live ? 0 : -1;
  stopBtn.tabIndex = live ? 0 : -1;

  // Queue affordances: "Leave queue" whenever we're in line; "Join now" only once
  // it's our turn (a slot is held for us). Both live under #queue-actions.
  const yourTurn = next === "your-turn";
  const inLine = next === "queued" || yourTurn;
  queueActions.hidden = !inLine;
  joinQueueBtn.hidden = !yourTurn;
  joinQueueBtn.tabIndex = yourTurn ? 0 : -1;
  leaveQueueBtn.hidden = !inLine;
  leaveQueueBtn.tabIndex = inLine ? 0 : -1;
  if (!yourTurn) stopJoinCountdown();

  // Warm reassurance under the terse position, only while waiting in line.
  if (next === "queued") {
    circleSubcaption.textContent =
      "当前已有观众在连线，已为你保留排队位置，请不要关闭页面。";
    circleSubcaption.hidden = false;
  } else {
    circleSubcaption.hidden = true;
  }

  updateRestartAvailability();
}

function updateRestartAvailability() {
  // Restart works from any settled state — it tears down a live call (if any)
  // and reconnects with the current settings. Only block while mid-connect or
  // while waiting in the queue (restarting from there would just re-queue).
  restartBtn.disabled =
    currentState === "connecting" || currentState === "queued" || currentState === "your-turn";
  restartHint.hidden = false;
  restartHint.textContent = LIVE_STATES.has(currentState)
    ? "使用上面的设置重新连线。"
    : "使用上面的设置申请连线。";
}

/**
 * @param {string} text
 * @param {"" | "error" | "muted"} [kind]
 */
function setCaption(text, kind = "") {
  const trimmed = text.trim();
  circleCaption.textContent = trimmed;
  circleCaption.className = `circle-caption${kind ? ` ${kind}` : ""}${trimmed ? "" : " empty"}`;
}

function openSettings() {
  syncConnectionUi();
  inputVoice.value = settings.voice;
  inputInstructions.value = settings.instructions;
  inputBargeIn.checked = settings.bargeIn;
  syncGateUi();
  updateRestartAvailability();
  void refreshAudioDeviceLists();
  void refreshAvatarPicker();
  settingsModal.showModal();
}

function openTools() {
  syncToolsUi();
  toolsModal.showModal();
}

/** @type {"settings" | "tools" | ""} */
let pendingAdminPanel = "";

function openAdminPanel(panel) {
  if (panel === "settings") openSettings();
  else if (panel === "tools") openTools();
}

async function requestAdminPanel(panel) {
  try {
    const response = await fetch("/api/admin/status", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (response.ok && data.unlocked) {
      openAdminPanel(panel);
      return;
    }
  } catch {
    // Fall through to the password dialog. Unlock itself fails closed.
  }
  pendingAdminPanel = panel;
  adminPassword.value = "";
  adminUnlockError.textContent = "";
  adminUnlockModal.showModal();
  requestAnimationFrame(() => adminPassword.focus());
}

adminUnlockForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitter = /** @type {SubmitEvent} */ (event).submitter;
  if (submitter?.value === "cancel") {
    pendingAdminPanel = "";
    adminUnlockModal.close();
    return;
  }
  const password = adminPassword.value;
  if (!password) return;
  adminPassword.disabled = true;
  adminUnlockError.textContent = "";
  try {
    const response = await fetch("/api/admin/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "密码验证失败");
    const panel = pendingAdminPanel;
    pendingAdminPanel = "";
    adminUnlockModal.close();
    if (panel) openAdminPanel(panel);
  } catch (error) {
    adminUnlockError.textContent = error instanceof Error ? error.message : String(error);
    adminPassword.select();
  } finally {
    adminPassword.disabled = false;
  }
});

async function refreshAvatarPicker() {
  const root = document.getElementById("avatar-picker");
  if (!root) return;
  try {
    const response = await fetch("/av/avatars", { cache: "no-store" });
    const data = await response.json();
    const avatars = Array.isArray(data.avatars) ? data.avatars : [];
    const selected = localStorage.getItem(STORAGE_KEYS.avatarId) || data.avatar_id || "xiaoya";
    root.replaceChildren();
    for (const item of avatars) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "avatar-pick";
      button.dataset.id = item.id;
      button.setAttribute("aria-pressed", item.id === selected ? "true" : "false");
      button.innerHTML = `<img src="${item.preview}" alt="${item.label}"><span>${item.label}</span>`;
      button.addEventListener("click", () => {
        void selectAvatar(item.id);
      });
      root.appendChild(button);
    }
  } catch (err) {
    console.warn("[main] avatar list failed:", err);
  }
}

async function selectAvatar(avatarId) {
  const root = document.getElementById("avatar-picker");
  if (root) {
    for (const button of root.querySelectorAll(".avatar-pick")) {
      button.setAttribute("aria-pressed", button.dataset.id === avatarId ? "true" : "false");
    }
  }
  try {
    const response = await fetch("/av/avatar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ avatar_id: avatarId }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
    localStorage.setItem(STORAGE_KEYS.avatarId, avatarId);
    window.dispatchEvent(new CustomEvent("avtr1-avatar-changed", { detail: { avatarId } }));
  } catch (err) {
    console.warn("[main] avatar switch failed:", err);
  }
}

/** dB position (clamped to the slider axis) as a 0..1 fraction of the track.
 * @param {number} db */
function dbToFraction(db) {
  const clamped = Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, db));
  return (clamped - GATE_OFF_DB) / (GATE_MAX_DB - GATE_OFF_DB);
}

/** @param {number} f @returns {number} dB at a 0..1 position on the gate axis. */
function fractionToDb(f) {
  const clamped = Math.min(1, Math.max(0, f));
  return Math.round(GATE_OFF_DB + clamped * (GATE_MAX_DB - GATE_OFF_DB));
}

// ── Radial gate arc (around the mic button, live during a call) ─────────────
// A 270° arc with the gap facing the orb (right). Fraction 0 (=Off) sits at the
// bottom-ish start; 1 (=max) at the top-ish end. The level fill and the
// threshold handle ride this same axis, mirroring the Settings widget.
const ARC_R = 40;
// A ~200° arc centred on the left (180°) so the wide gap faces the orb (right).
const ARC_SPAN_DEG = 200;
const ARC_START_DEG = 180 - ARC_SPAN_DEG / 2; // lower-left start; Off end

/** Point at fraction f (0..1) and radius r, in the 0..100 viewBox.
 * @param {number} f @param {number} [r] */
function arcPoint(f, r = ARC_R) {
  const deg = ARC_START_DEG + f * ARC_SPAN_DEG;
  const rad = (deg * Math.PI) / 180;
  return { x: 50 + r * Math.cos(rad), y: 50 + r * Math.sin(rad) };
}

/** SVG path `d` for the full 0..1 arc (clockwise). */
function fullArcD() {
  const a = arcPoint(0);
  const b = arcPoint(1);
  const largeArc = ARC_SPAN_DEG > 180 ? 1 : 0;
  return `M ${a.x} ${a.y} A ${ARC_R} ${ARC_R} 0 ${largeArc} 1 ${b.x} ${b.y}`;
}

/** One-time geometry: track, fill (dash-revealed) and the transparent hit band. */
function initGateArc() {
  const d = fullArcD();
  mgaTrack.setAttribute("d", d);
  mgaFill.setAttribute("d", d);
  mgaHit.setAttribute("d", d);
  // pathLength 100 lets us reveal the fill by fraction via dashoffset.
  mgaFill.setAttribute("pathLength", "100");
  mgaFill.style.strokeDasharray = "100 100";
  mgaFill.style.strokeDashoffset = "100"; // empty until levels arrive
  renderGateHandle();
}

/** Place the threshold bead on the arc at the stored threshold; flag off state. */
function renderGateHandle() {
  const off = settings.noiseGate <= GATE_OFF_DB;
  const p = arcPoint(dbToFraction(settings.noiseGate));
  mgaHandle.setAttribute("cx", String(p.x));
  mgaHandle.setAttribute("cy", String(p.y));
  micGate.classList.toggle("gate-off", off);
}

/** Paint a 0..1 live level onto the arc fill (and the Settings meter if open).
 * Brightens the tick when the level crosses the threshold — i.e. the gate is
 * actually open — but only when gating is enabled.
 * @param {number} rms */
function paintInputLevel(rms) {
  const db = rms > 0 ? 20 * Math.log10(rms) : GATE_OFF_DB;
  const f = dbToFraction(db);
  mgaFill.style.strokeDashoffset = String(100 * (1 - f));
  if (settingsModal.open) gateMeterFill.style.width = `${f * 100}%`;
  const enabled = settings.noiseGate > GATE_OFF_DB;
  micGate.classList.toggle("gate-open", enabled && f >= dbToFraction(settings.noiseGate));
}

/** The single place that commits a new gate threshold: updates both controls,
 * persists, and applies live to the running session.
 * @param {number} db */
function setGateThreshold(db) {
  settings.noiseGate = Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, Math.round(db)));
  const off = settings.noiseGate <= GATE_OFF_DB;
  inputNoiseGate.value = String(settings.noiseGate);
  gateValue.textContent = off ? "Off" : `${settings.noiseGate} dB`;
  renderGateHandle();
  localStorage.setItem(STORAGE_KEYS.noiseGate, String(settings.noiseGate));
  if (client && LIVE_STATES.has(currentState)) {
    client.setNoiseGate(gateParams(settings.noiseGate));
  }
}

/** Reflect the stored gate threshold into the slider, label and arc handle. */
function syncGateUi() {
  inputNoiseGate.value = String(settings.noiseGate);
  const off = settings.noiseGate <= GATE_OFF_DB;
  gateValue.textContent = off ? "Off" : `${settings.noiseGate} dB`;
  renderGateHandle();
}

// Drag along the arc band to set the threshold (a tap on the glyph still mutes).
let gateDragging = false;
/** @param {PointerEvent} e */
function gatePointerToDb(e) {
  const rect = mgaArc.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  let deg = (Math.atan2(e.clientY - cy, e.clientX - cx) * 180) / Math.PI;
  if (deg < 0) deg += 360;
  // Map the on-arc angle to a fraction; angles in the right-side gap fall
  // outside [0,1] and fractionToDb clamps them to the nearest end (just-below
  // start -> Off, just-past end -> max).
  const f = (deg - ARC_START_DEG) / ARC_SPAN_DEG;
  return fractionToDb(f);
}
mgaHit.addEventListener("pointerdown", (e) => {
  gateDragging = true;
  mgaHit.setPointerCapture(e.pointerId);
  setGateThreshold(gatePointerToDb(e));
});
mgaHit.addEventListener("pointermove", (e) => {
  if (gateDragging) setGateThreshold(gatePointerToDb(e));
});
const endGateDrag = (/** @type {PointerEvent} */ e) => {
  if (!gateDragging) return;
  gateDragging = false;
  try { mgaHit.releasePointerCapture(e.pointerId); } catch {}
};
mgaHit.addEventListener("pointerup", endGateDrag);
mgaHit.addEventListener("pointercancel", endGateDrag);

settingsBtn.addEventListener("click", () => void requestAdminPanel("settings"));

// ── Tools panel ───────────────────────────────────────────────────────────

/** Reflect the current tool state into the panel controls. */
function syncToolsUi() {
  const avail = searchAvailable();
  toolWebSwitch.checked = toolsEnabled.web_search && avail;
  toolWebSwitch.disabled = !avail;
  toolWebRow.classList.toggle("disabled", !avail);
  toolCamSwitch.checked = toolsEnabled.camera_snapshot;
  mcpToolStatus.textContent = mcpToolDefs.length
    ? `${mcpSources.join("、")} 已连接，共 ${mcpToolDefs.length} 个实时工具`
    : "MCP 服务暂时不可用";

  if (serverSearchKey) {
    // Key lives server-side: show it as configured, never expose it.
    searchKeyInput.value = "";
    searchKeyInput.placeholder = "••••••••  · provided by the server";
    searchKeyInput.disabled = true;
    toolWebHint.textContent = "Ready. The search key is held server-side and never sent to your browser.";
  } else {
    searchKeyInput.disabled = false;
    searchKeyInput.value = userSearchKey;
    searchKeyInput.placeholder = "Paste a Serper key to enable web search";
    toolWebHint.textContent = userSearchKey
      ? "Using your key — stored in this browser only."
      : "No server key configured. Add your own Serper key to enable web search.";
  }
}

toolsBtn.addEventListener("click", () => void requestAdminPanel("tools"));
toolsClose.addEventListener("click", () => toolsModal.close());
toolsModal.addEventListener("click", (e) => {
  if (e.target === toolsModal) toolsModal.close();
});

toolWebSwitch.addEventListener("change", () => {
  if (toolWebSwitch.checked && !searchAvailable()) {
    toolWebSwitch.checked = false; // guard: can't enable without a key
    return;
  }
  toolsEnabled.web_search = toolWebSwitch.checked;
  saveTools();
  pushToolsToSession();
});

toolCamSwitch.addEventListener("change", async () => {
  if (toolCamSwitch.checked) {
    try {
      // Flipping the switch always re-requests the camera, so a permission that
      // was only dismissed earlier is asked again here.
      await enableCamera();
    } catch (err) {
      toolCamSwitch.checked = false;
      const denied = err instanceof Error && (err.name === "NotAllowedError" || err.name === "SecurityError");
      toolCamHint.textContent = denied
        ? "Camera blocked. Allow it from the camera icon in your browser's address bar — it switches on automatically."
        : `Camera unavailable${err instanceof Error ? `: ${err.message}` : ""}`;
      return;
    }
    toolsEnabled.camera_snapshot = true;
    toolCamHint.textContent = "Camera on. The assistant can take a snapshot when it needs to see.";
  } else {
    disableCamera();
    toolsEnabled.camera_snapshot = false;
    toolCamHint.textContent = "Let the assistant see through your webcam.";
  }
  saveTools();
  pushToolsToSession();
});

searchKeyInput.addEventListener("input", () => {
  if (serverSearchKey) return;
  userSearchKey = searchKeyInput.value.trim();
  if (userSearchKey) localStorage.setItem(STORAGE_KEYS.searchKey, userSearchKey);
  else localStorage.removeItem(STORAGE_KEYS.searchKey);

  const avail = searchAvailable();
  toolWebSwitch.disabled = !avail;
  toolWebRow.classList.toggle("disabled", !avail);
  // Losing the key disables a previously-enabled tool.
  if (!avail && toolsEnabled.web_search) {
    toolsEnabled.web_search = false;
    toolWebSwitch.checked = false;
    saveTools();
    pushToolsToSession();
  }
  toolWebHint.textContent = userSearchKey
    ? "Using your key — stored in this browser only."
    : "No server key configured. Add your own Serper key to enable web search.";
});

// ── Camera ──────────────────────────────────────────────────────────────────

async function enableCamera() {
  if (cameraStream) return;
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false,
  });
  camVideo.srcObject = cameraStream;
  try { await camVideo.play(); } catch { /* autoplay quirks; muted video is fine */ }
  camPip.classList.add("visible");
  camPip.setAttribute("aria-hidden", "false");
  // Lets the footer reflow to the bottom-right (and hide on mobile) while the
  // webcam preview occupies the bottom of the stage.
  document.body.classList.add("cam-on");
}

function disableCamera() {
  if (cameraStream) {
    for (const t of cameraStream.getTracks()) t.stop();
    cameraStream = null;
  }
  camVideo.srcObject = null;
  camPip.classList.remove("visible");
  camPip.setAttribute("aria-hidden", "true");
  document.body.classList.remove("cam-on");
}

/** Auto-start the webcam on arrival (the camera tool is on by default). If the
 *  user declines the permission, switch the tool off and reflect it in the UI
 *  rather than nagging. */
async function autoStartCamera() {
  if (!toolsEnabled.camera_snapshot || cameraStream) return;
  try {
    await enableCamera();
  } catch (err) {
    console.warn("[main] camera auto-start declined/failed:", err);
    toolsEnabled.camera_snapshot = false;
    saveTools();
    syncToolsUi();
  }
}

/** Track the browser's camera permission so a later re-grant (e.g. the user
 *  unblocks it from the address bar after a denial) turns the camera back on
 *  without another toggle, and a revoke turns it off. Best-effort: the
 *  Permissions API doesn't support "camera" everywhere (e.g. Safari). */
async function watchCameraPermission() {
  try {
    const status = await navigator.permissions?.query?.({ name: /** @type {any} */ ("camera") });
    if (!status) return;
    status.addEventListener("change", () => {
      if (status.state === "granted") {
        if (!toolsEnabled.camera_snapshot) { toolsEnabled.camera_snapshot = true; saveTools(); }
        void autoStartCamera();
        syncToolsUi();
      } else if (status.state === "denied") {
        disableCamera();
        if (toolsEnabled.camera_snapshot) { toolsEnabled.camera_snapshot = false; saveTools(); }
        syncToolsUi();
      }
    });
  } catch {
    // Permissions API unavailable for "camera" — the toggle still re-asks.
  }
}

/**
 * Grab the current webcam frame as a downscaled JPEG data URL. The preview is
 * mirrored in CSS for a natural self-view, but we draw the raw (un-mirrored)
 * video here so the model sees the scene in its true orientation.
 * @returns {string | null}
 */
function captureSnapshot() {
  if (!cameraStream || !camVideo.videoWidth) return null;
  const vw = camVideo.videoWidth;
  const vh = camVideo.videoHeight;
  const scale = Math.min(1, SNAPSHOT_MAX_EDGE / Math.max(vw, vh));
  const w = Math.max(1, Math.round(vw * scale));
  const h = Math.max(1, Math.round(vh * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(camVideo, 0, 0, w, h);
  return canvas.toDataURL("image/jpeg", SNAPSHOT_QUALITY);
}

/** Brief shutter flash on the preview so the user sees a snapshot was taken. */
function flashPreview() {
  camPip.classList.remove("flash");
  void camPip.offsetWidth; // reflow so the animation restarts
  camPip.classList.add("flash");
}

// ── Tool executor ─────────────────────────────────────────────────────────
// Runs the function the model called and returns the result. The connection's
// ToolCallBatcher sends every result from the originating response together,
// then requests one follow-up. Errors come back as tool output too, so the
// model can recover gracefully instead of the turn stalling.

/**
 * Run the function the model called. The caller batches the returned result
 * with the other calls from the same response before sending it to the model.
 * @param {string} name @param {string} argsJson @param {string} callId
 * @returns {Promise<{ output: string, image?: string }>}
 */
async function runTool(name, argsJson, callId) {
  if (!client) return { output: "" };
  let args = /** @type {Record<string, unknown>} */ ({});
  try { args = JSON.parse(argsJson || "{}"); } catch { /* keep {} */ }

  if (DEBUG) console.debug(`[tool] run name=${name} callId=${JSON.stringify(callId)} args=${argsJson}`);
  if (!callId) console.warn("[tool] empty call_id — the backend didn't tag the call, can't return a function_call_output");

  /** @type {{ output: string, image?: string }} */
  let result = { output: "" };
  try {
    if (name.startsWith("mcp_")) {
      result.output = await execMcpTool(name, args);
    } else if (name === "web_search") {
      const query = typeof args.query === "string" ? args.query : "";
      result.output = await execWebSearch(query);
    } else if (name === "camera_snapshot") {
      const dataUrl = captureSnapshot();
      if (dataUrl) {
        if (DEBUG) console.debug(`[tool] camera_snapshot captured frame (${dataUrl.length} chars), sending image + output`);
        result = { output: "Snapshot captured from the webcam and attached as an image.", image: dataUrl };
        flashPreview();
      } else {
        console.warn("[tool] camera_snapshot: no frame — camera off or not ready");
        result.output = "The camera is not available right now.";
      }
    } else {
      result.output = `Unknown tool: ${name}`;
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    result.output = `Tool failed: ${msg}`;
  }
  return result;
}

/** @param {string} name @param {Record<string, unknown>} args @returns {Promise<string>} */
async function execMcpTool(name, args) {
  const response = await fetch("/api/mcp/call", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, arguments: args }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `MCP error (${response.status})`);
  return typeof data.output === "string" ? data.output : JSON.stringify(data.output ?? {});
}

/** @param {string} query @returns {Promise<string>} */
async function execWebSearch(query) {
  if (!query) return "No query provided.";
  /** @type {Record<string, string>} */
  const body = { query };
  // Only send a user key when there's no server key (server prefers its own).
  if (!serverSearchKey && userSearchKey) body.key = userSearchKey;

  const res = await fetch("api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = String(res.status);
    try { const j = await res.json(); if (j.detail) detail = j.detail; } catch {}
    throw new Error(`search error (${detail})`);
  }
  const json = await res.json();
  // Date-stamp the header so the model treats these as fresh realtime facts
  // rather than its (older) training knowledge.
  const today = new Date().toISOString().slice(0, 10);
  /** @type {string[]} */
  const lines = [`Google search result from ${today}:`];
  if (json.answer) lines.push(`Answer: ${json.answer}`);
  for (const r of json.results || []) {
    lines.push(`- ${r.title}: ${r.snippet} (${r.url})`);
  }
  return lines.length > 1 ? lines.join("\n") : `${lines[0]}\nNo results found.`;
}

async function fetchMcpTools() {
  try {
    const response = await fetch("/api/mcp/tools", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "MCP tools unavailable");
    const tools = Array.isArray(data.tools) ? data.tools : [];
    mcpToolDefs = tools
      .filter((tool) => tool?.type === "function" && typeof tool.name === "string")
      .map((tool) => ({
        type: "function",
        name: tool.name,
        description: String(tool.description || tool.name),
        parameters: tool.parameters && typeof tool.parameters === "object"
          ? tool.parameters
          : { type: "object", properties: {} },
      }));
    mcpSources = Array.isArray(data.sources)
      ? data.sources.map((source) => String(source)).filter(Boolean)
      : [];
  } catch (error) {
    console.warn("[mcp] tool discovery failed", error);
    mcpToolDefs = [];
    mcpSources = [];
  }
  syncToolsUi();
  pushToolsToSession();
}

/** Learn server config (search key + connection target), then refresh the UI. */
async function fetchConfig() {
  try {
    const res = await fetch("api/config");
    if (res.ok) {
      const json = await res.json();
      serverSearchKey = !!json.search;
      lbMode = !!json.lb;
      // Lock to LB mode only when the deploy reports a load balancer.
      allowDirect = json.allowDirect ?? !lbMode;
      // Deploy-pinned direct URL (overrides the LB server-side already).
      pinnedUrl = (json.s2sUrl || "").trim();
      startupGreeting = typeof json.startupGreeting === "string"
        ? json.startupGreeting.trim()
        : "";
      idlePrompt = typeof json.idlePrompt === "string" ? json.idlePrompt.trim() : "";
      idlePromptMinSeconds = Number(json.idlePromptMinSeconds) || 35;
      idlePromptMaxSeconds = Number(json.idlePromptMaxSeconds) || 55;
      loginRequired = !!json.requireLogin;
      if (json.mcp) await fetchMcpTools();
      // The conversation-time limiter rides on the LB being present.
      limiterOn = lbMode;
    }
    // Non-OK response: leave the fail-open default (allowDirect = true).
  } catch {
    // Config endpoint unreachable (e.g. static hosting): keep direct entry.
  }
  if (DEBUG) console.debug(`[ui] config: allowDirect=${allowDirect} lbMode=${lbMode}`);
  // Login chip + remaining-budget (no-op / hidden when the limiter is off).
  await account.refresh();
  if (currentState === "idle" && loginRequired && !account.loggedIn) {
    setCaption("Sign in to start");
  }
  syncToolsUi();
  syncConnectionUi();
}

/**
 * Resolve where to connect, per the deploy's mode:
 *   • LB mode  -> `{ sessionUrl }`, the client POSTs the same-origin /api/session
 *     proxy and the server forwards to the LB (its address stays server-side).
 *   • direct   -> `{ directUrl }`, connect straight to the s2s WebSocket.
 * Throws a user-facing error if direct mode is on but no URL was entered.
 * @returns {{ sessionUrl: string } | { directUrl: string }}
 */
function connectionTarget() {
  if (!allowDirect) {
    return { sessionUrl: "api/session" };
  }
  const directUrl = buildDirectWsUrl(pinnedUrl || settings.directUrl);
  if (!directUrl) {
    throw new Error("Enter a speech-to-speech server URL in Settings.");
  }
  return { directUrl };
}

/**
 * Normalise a user-typed server address into a realtime WebSocket URL.
 * Accepts bare hosts (`localhost:8080`), http(s) URLs, or ws(s) URLs, and adds
 * the `/v1/realtime` path when none is given. A full connect URL (with path
 * and/or query) is preserved as-is.
 * @param {string} raw @returns {string}
 */
function buildDirectWsUrl(raw) {
  let s = (raw || "").trim();
  if (!s) return "";
  if (!/^wss?:\/\//i.test(s)) {
    if (/^https?:\/\//i.test(s)) {
      s = s.replace(/^http/i, "ws"); // http→ws, https→wss
    } else {
      const isLocal = /^(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(s);
      s = (isLocal ? "ws://" : "wss://") + s;
    }
  }
  try {
    const u = new URL(s);
    if (u.pathname === "" || u.pathname === "/") u.pathname = "/v1/realtime";
    return u.toString();
  } catch {
    return s;
  }
}

/** Create + resume an AudioContext synchronously (must run inside the user
 *  gesture so iOS lets it start). Returns null if construction fails. */
function createResumedAudioContext() {
  try {
    const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
    const ctx = new Ctx({ latencyHint: "interactive" });
    if (ctx.state === "suspended") void ctx.resume().catch(() => {});
    return /** @type {AudioContext} */ (ctx);
  } catch (err) {
    console.warn("[main] AudioContext init failed:", err);
    return null;
  }
}

/** Read the editable settings out of the form. The URL field is only honoured
 *  in free direct mode — in LB mode it's hidden, and when the deploy pins a
 *  URL it's read-only, so the user's saved URL survives either way. */
function readSettingsFromForm() {
  return {
    directUrl: allowDirect && !pinnedUrl ? inputLbUrl.value.trim() : settings.directUrl,
    voice: inputVoice.value || DEFAULT_VOICE,
    instructions: inputInstructions.value.trim() || DEFAULT_INSTRUCTIONS,
    noiseGate: readGateThreshold(),
    audioInputId: inputAudioInput.value || "",
    audioOutputId: inputAudioOutput.value || "",
    bargeIn: inputBargeIn.checked,
  };
}

/** Gate threshold (dBFS) currently shown on the slider, clamped to range. */
function readGateThreshold() {
  const v = Math.round(Number(inputNoiseGate.value));
  if (!Number.isFinite(v)) return GATE_OFF_DB;
  return Math.min(GATE_MAX_DB, Math.max(GATE_OFF_DB, v));
}

/** Adapt the connection field to the mode learned from /api/config. */
function syncConnectionUi() {
  if (pinnedUrl) {
    // Deployment-owned technical URL is redundant in Settings; the first-screen
    // status chips show the information users actually need.
    connField.hidden = true;
    inputLbUrl.value = pinnedUrl;
    inputLbUrl.readOnly = true;
    connHint.classList.remove("error");
    connHint.textContent = "Speech-to-speech server URL pinned by this deployment.";
  } else if (allowDirect) {
    // Direct mode: the user sets their own s2s server URL.
    connField.hidden = false;
    inputLbUrl.value = settings.directUrl;
    inputLbUrl.readOnly = false;
    inputLbUrl.placeholder = "http://localhost:port";
    connHint.classList.remove("error");
    connHint.textContent =
      "URL of your speech-to-speech server, e.g. http://localhost:8080 (the app adds /v1/realtime).";
  } else {
    // LB mode: the load balancer URL is deployment-owned — hide it entirely so
    // its address is never exposed in Settings.
    connField.hidden = true;
  }
}

/** True when the user must supply a server URL before connecting (direct mode
 *  with nothing set). */
function missingServerUrl() {
  return allowDirect && !pinnedUrl && !buildDirectWsUrl(settings.directUrl);
}

/** Open Settings and point the user at the empty server-URL field. */
function promptServerUrl() {
  if (!settingsModal.open) {
    // This path can be reached from the central call button in direct-mode
    // deployments, so it must use the same password gate as the gear button.
    void requestAdminPanel("settings");
    return;
  }
  syncConnectionUi();
  connHint.textContent = "Set the speech-to-speech server URL to start.";
  connHint.classList.add("error");
  inputLbUrl.focus();
}

settingsForm.addEventListener("submit", (event) => {
  const submitter = /** @type {HTMLButtonElement | null} */ ((/** @type {SubmitEvent} */ (event)).submitter);
  if (submitter?.value !== "save") return;

  settings = readSettingsFromForm();
  saveSettings(settings);

  // Voice + instructions can apply to a live session without reconnecting; a
  // changed connection URL only takes effect on the next restart. Speaker
  // output can switch live when the browser supports AudioContext.setSinkId;
  // mic device changes need a Restart (new getUserMedia stream).
  if (client && LIVE_STATES.has(currentState)) {
    client.updateSession({ voice: settings.voice, instructions: settings.instructions });
    void client.setAudioOutputDevice(settings.audioOutputId);
    client.setBargeInEnabled(settings.bargeIn);
  }
});

// The noise gate applies live (worklet param), so tune it without a restart:
// update the label/marker, persist, and push straight to the running client.
inputNoiseGate.addEventListener("input", () => {
  setGateThreshold(readGateThreshold());
});

inputBargeIn.addEventListener("change", () => {
  settings.bargeIn = inputBargeIn.checked;
  localStorage.setItem(STORAGE_KEYS.bargeIn, settings.bargeIn ? "1" : "0");
  client?.setBargeInEnabled(settings.bargeIn);
});

restartBtn.addEventListener("click", async () => {
  if (currentState === "connecting") return; // a connect is already underway
  if (loginRequired && !account.loggedIn) {
    account.showLoginRequired();
    return;
  }
  settings = readSettingsFromForm();
  saveSettings(settings);
  if (missingServerUrl()) { promptServerUrl(); return; } // keep settings open
  settingsModal.close();
  // Grab the AudioContext NOW, inside the click gesture — teardown() awaits, and
  // creating it afterwards would fall outside the gesture (silent on iOS).
  const audioContext = createResumedAudioContext();
  try {
    if (client) await teardown();
    await doStart(audioContext);
  } catch (err) {
    await handleStartError(err);
  }
});

circleBtn.addEventListener("click", async () => {
  try {
    if (currentState === "ai-speaking" && client) {
      client.manualInterrupt();
      try {
        const response = await fetch("/api/avatar/interrupt", { method: "POST" });
        if (!response.ok) throw new Error(`interrupt failed (${response.status})`);
        window.dispatchEvent(new CustomEvent("avatar-manual-interrupt"));
      } catch (error) {
        console.warn("[avatar] manual interrupt failed", error);
      }
      setState("listening");
      return;
    }
    if (currentState === "idle" || currentState === "error") {
      if (loginRequired && !account.loggedIn) {
        account.showLoginRequired();
        return;
      }
      if (missingServerUrl()) { promptServerUrl(); return; }
      await doStart();
    }
  } catch (err) {
    await handleStartError(err);
  }
});

/** A failed start is either the daily limit (show the modal, return to idle) or
 *  a real fault (surface it). doStart already closed any orphan AudioContext.
 *  @param {any} err */
async function handleStartError(err) {
  if (err && err.code === "login-required") {
    await teardown();
    setState("error");
    setCaption("Sign in again to continue.", "error");
    account.showLoginRequired(err.loginUrl);
    return;
  }
  if (err && err.code === "limit") {
    await teardown();
    account.showLimit(err.tier);
    return;
  }
  // The user left the queue (close() aborted the wait): teardown already reset
  // the UI to idle, so there's nothing to report.
  if (err && err.code === "aborted") return;
  // The whole waiting line is full: a warm, reassuring modal rather than an error.
  if (err && err.code === "queue-full") {
    await teardown();
    account.showBusy();
    return;
  }
  // Our place lapsed (ticket reaped, or the join window ran out). Recoverable, not
  // a fault: land on the retry state with a kind, plain-language reason.
  if (err && (err.code === "queue-expired" || err.code === "join-expired")) {
    await teardown();
    setState("error");
    setCaption(
      err.code === "join-expired"
        ? "连线席位已过期，点击重新排队。"
        : "排队凭证已过期，点击重新排队。",
      "error",
    );
    return;
  }
  await onFatalError(err);
}

micBtn.addEventListener("click", () => {
  if (!micStream || !client) return;
  micMuted = !micMuted;
  syncMicMuteState();
  micBtn.classList.toggle("muted", micMuted);
  micBtn.setAttribute("aria-label", micMuted ? "Unmute" : "Mute");
  micBtn.title = micMuted ? "Unmute" : "Mute";
});

stopBtn.addEventListener("click", async () => {
  await teardown();
});

// "Leave queue": tear down the pending connect (aborts the poll wait) and drop
// our place in line. Same teardown path as stopping a live call.
leaveQueueBtn.addEventListener("click", async () => {
  await teardown();
});

// "Join now": accept the held slot. The click is a user gesture, so the client
// re-resumes the AudioContext here (iOS) before dialing.
joinQueueBtn.addEventListener("click", () => {
  stopJoinCountdown();
  if (client) client.join();
});

const MIC_CONSTRAINTS_BASE = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

/** @returns {MediaStreamConstraints} */
function micConstraints() {
  /** @type {MediaTrackConstraints} */
  const audio = { ...MIC_CONSTRAINTS_BASE };
  if (settings.audioInputId) {
    // ideal (not exact): if the saved device was unplugged, fall back quietly.
    audio.deviceId = { ideal: settings.audioInputId };
  }
  return { audio };
}

/** True when Web Audio can route playback to a chosen output device. */
function supportsAudioOutputSelection() {
  const Ctx = window.AudioContext || /** @type {any} */ (window).webkitAudioContext;
  return typeof Ctx?.prototype?.setSinkId === "function";
}

/**
 * Rebuild the mic/speaker <select>s from enumerateDevices. Labels are blank
 * until mic permission has been granted at least once.
 */
async function refreshAudioDeviceLists() {
  const canPickOutput = supportsAudioOutputSelection();
  inputAudioOutput.disabled = !canPickOutput;
  audioOutputHint.textContent = canPickOutput
    ? "Where assistant audio plays. Can change live while connected."
    : "Speaker selection needs a browser with AudioContext.setSinkId (Chrome/Edge).";

  /** @type {MediaDeviceInfo[]} */
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    console.warn("[main] enumerateDevices failed:", err);
  }

  const inputs = devices.filter((d) => d.kind === "audioinput");
  const outputs = devices.filter((d) => d.kind === "audiooutput");
  const labelsReady = devices.some((d) => d.label);

  fillDeviceSelect(inputAudioInput, inputs, settings.audioInputId, "Microphone");
  fillDeviceSelect(inputAudioOutput, outputs, settings.audioOutputId, "Speaker");

  const hint = inputAudioInput.parentElement?.querySelector("small");
  if (hint) {
    hint.textContent = labelsReady
      ? "Applies on the next conversation (or Restart)."
      : "Allow microphone access (tap Start once) to see device names. Mic changes apply on Restart.";
  }
}

/**
 * @param {HTMLSelectElement} select
 * @param {MediaDeviceInfo[]} devices
 * @param {string} selectedId
 * @param {string} fallbackLabel
 */
function fillDeviceSelect(select, devices, selectedId, fallbackLabel) {
  const prev = selectedId || select.value || "";
  select.replaceChildren();
  const def = document.createElement("option");
  def.value = "";
  def.textContent = "System default";
  select.appendChild(def);
  devices.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `${fallbackLabel} ${i + 1}`;
    select.appendChild(opt);
  });
  if (prev && ![...select.options].some((o) => o.value === prev)) {
    const missing = document.createElement("option");
    missing.value = prev;
    missing.textContent = `${fallbackLabel} (saved, not found)`;
    select.appendChild(missing);
  }
  select.value = prev;
  if (select.value !== prev) select.value = "";
}

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (settingsModal.open) void refreshAudioDeviceLists();
  });
}

/** Prompt for mic permission up front, then immediately release the tracks so no
 *  recording indicator lingers during a queue wait. Throws a friendly error if the
 *  user denies. */
async function primeMicPermission() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Microphone needs HTTPS. Open https://175.155.64.146:19800/ and continue past the certificate warning.",
    );
  }
  try {
    const s = await navigator.mediaDevices.getUserMedia(micConstraints());
    for (const track of s.getTracks()) track.stop();
  } catch (err) {
    throw new Error(
      `Microphone access denied${err instanceof Error ? `: ${err.message}` : ""}`,
    );
  }
}

/** Acquire the live capture stream once a slot is granted. Permission was primed
 *  in the tap gesture, so this is silent. Stored module-side for mute + teardown. */
async function acquireMicStream() {
  micStream = await navigator.mediaDevices.getUserMedia(micConstraints());
  return micStream;
}

/** @param {number} position Update the queued caption ("You're #N in line"). */
function onQueuePosition(position) {
  const n = Number(position) || 0;
  setCaption(n > 0 ? `当前排在第 ${n} 位` : "正在排队…", "muted");
}

// ── "Your turn" join countdown ──────────────────────────────────────────────
// While a slot is held for us, show how long is left to accept it. The client's
// join gate expires just before the load balancer reclaims the slot.
let joinCountdownTimer = 0;

/** @param {number} sec */
function startJoinCountdown(sec) {
  stopJoinCountdown();
  let left = Math.max(0, Math.floor(sec));
  const paint = () => {
    joinQueueBtn.textContent = left > 0 ? `立即连线（${left}秒）` : "立即连线";
  };
  paint();
  joinCountdownTimer = window.setInterval(() => {
    left -= 1;
    if (left <= 0) {
      stopJoinCountdown();
      joinQueueBtn.textContent = "立即连线";
      return;
    }
    paint();
  }, 1000);
}

function stopJoinCountdown() {
  if (joinCountdownTimer) {
    clearInterval(joinCountdownTimer);
    joinCountdownTimer = 0;
  }
}

/**
 * Start a conversation. Pass a pre-created AudioContext when the caller already
 * made one inside the tap/click gesture (required on iOS); otherwise one is
 * created here, which is still inside the gesture for a direct orb tap.
 * @param {AudioContext | null} [audioContext]
 */
async function doStart(audioContext = null) {
  // Resolve the target before touching mic/audio so a misconfiguration (e.g.
  // direct mode with no URL) fails fast with a clear message.
  const target = connectionTarget();

  chat.clear();
  chat.reset();
  setState("connecting");
  setCaption("Asking for mic…", "muted");

  // Create + resume the AudioContext SYNCHRONOUSLY, still inside the gesture.
  // iOS Safari only starts an AudioContext from a user gesture; if we waited
  // until after the getUserMedia / session-creation awaits below, it would stay
  // suspended and the whole pipeline would be silent.
  if (!audioContext) audioContext = createResumedAudioContext();

  // Prime the mic permission now (get the prompt out of the way up front), then
  // release it. The real capture stream is acquired only once a slot is granted
  // (see acquireMicStream), so the mic 'in use' indicator never lights while we
  // sit in the queue. Permission persists, so the later acquire is silent.
  try {
    await primeMicPermission();
  } catch (err) {
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }

  // The webcam is started on arrival (autoStartCamera), so nothing to do here;
  // a still-pending grant just means the snapshot tool isn't ready yet.

  const c = new S2sWsRealtimeClient({
    ...target,
    voice: settings.voice,
    instructions: settings.instructions,
    startupGreeting,
    idlePrompt,
    idlePromptMinSeconds,
    idlePromptMaxSeconds,
    acquireMic: acquireMicStream,
    tools: activeToolDefs(),
    noiseGate: gateParams(settings.noiseGate),
    bargeIn: settings.bargeIn,
    audioOutputId: settings.audioOutputId || "",
    ...(audioContext ? { audioContext } : {}),
  });
  client = c;
  c.setMuted(micMuted || userAudioReplaying);

  /** @param {{ callId: string; output: string; image?: string }[]} results */
  const sendToolBatch = (results) => {
    if (client !== c) return;
    for (const result of results) c.sendToolOutput(result.callId, result.output);
    for (const result of results) {
      if (result.image) c.sendUserImage(result.image);
    }
    if (DEBUG) console.debug(`[tool] requesting one follow-up after ${results.length} tool result(s)`);
    c.requestResponse();
  };
  const toolBatches = new ToolCallBatcher(sendToolBatch);

  c.addEventListener("queue", (e) => {
    const { position, queueId } = /** @type {CustomEvent<{ position: number; queueId: string }>} */ (e).detail;
    if (queueId) queuedTicketId = queueId;
    onQueuePosition(position);
  });

  c.addEventListener("ready-to-join", (e) => {
    const { info, expiresSec } = /** @type {CustomEvent<{ info: import("./ws/s2s-ws-client.js").WsSessionInfo; expiresSec: number }>} */ (e).detail;
    // A slot is held for us. We're out of the queue now, so drop the ticket ref.
    // Track the granted session id already so that leaving (or letting the timer
    // lapse) refunds the budget the server reserved at claim, even before we dial.
    queuedTicketId = "";
    if (info?.sessionId) {
      trackedSessionId = info.sessionId;
      trackedTier = info.tier || "anon";
    }
    startJoinCountdown(expiresSec);
  });

  c.addEventListener("status", (e) => {
    const detail = /** @type {CustomEvent<{ status: string }>} */ (e).detail;
    onClientStatus(detail.status);
    if (detail.status === "ai-speaking") chat.onAssistantActivity();
  });
  c.addEventListener("transcript", (e) => {
    const d = /** @type {CustomEvent<{ role: "user" | "assistant"; text: string; partial: boolean; itemId?: string; responseId?: string }>} */ (e).detail;
    chat.onTranscript(d);
  });
  c.addEventListener("user-turn-started", (e) => {
    const detail = /** @type {CustomEvent<{ itemId?: string }>} */ (e).detail;
    chat.onUserTurnStarted(detail);
  });
  c.addEventListener("user-turn-stopped", (e) => {
    const detail = /** @type {CustomEvent<{ itemId?: string }>} */ (e).detail;
    chat.onUserTurnStopped(detail);
  });
  c.addEventListener("user-audio", (e) => {
    const detail = /** @type {CustomEvent<{ itemId?: string; audio: Blob; durationMs?: number; truncated?: boolean }>} */ (e).detail;
    chat.onUserAudio(detail);
  });

  c.addEventListener("response-finished", (e) => {
    const detail = /** @type {CustomEvent<{ responseId: string; status: string; audible?: boolean; transcript?: string }>} */ (e).detail;
    chat.onResponseFinished(detail);
    const flush = toolBatches.finish(detail.responseId, detail.status);
    if (flush) void flush.catch((err) => onFatalError(err));
  });

  c.addEventListener("toolcall", (e) => {
    const { name, arguments: args, callId, responseId } = /** @type {CustomEvent<{ name: string; arguments: string; callId: string; responseId: string }>} */ (e).detail;
    if (!responseId) {
      console.warn(`[tool] call ${callId || "<unknown>"} has no response_id; ignoring uncorrelated tool call`);
      return;
    }
    chat.onToolCall(name);
    // Execute the tool, then push it to the conversation once the result is in,
    // so the toggle shows both the call input and its output together.
    const execution = runTool(name, args, callId).then(({ output, image }) => {
      if (client === c) chat.onToolResult(name, args, output, image);
      return { callId, output, ...(image ? { image } : {}) };
    });
    toolBatches.add(responseId, execution);
  });
  c.addEventListener("error", (e) => {
    const detail = /** @type {CustomEvent<{ error: unknown }>} */ (e).detail;
    void onFatalError(detail.error);
  });
  c.addEventListener("server-error", (e) => {
    // Non-fatal: the backend reported an error mid-session. Log it, keep the
    // socket and the conversation alive (the model can recover on its own).
    const detail = /** @type {CustomEvent<{ error: unknown }>} */ (e).detail;
    const msg = detail.error instanceof Error ? detail.error.message : String(detail.error);
    console.warn("[main] server error (non-fatal):", msg);
  });
  c.addEventListener("session", (e) => {
    const info = /** @type {CustomEvent<{ info: import("./ws/s2s-ws-client.js").WsSessionInfo }>} */ (e).detail.info;
    console.log("[ws] session created:", info.sessionId);
    // A slot was granted — we're out of the queue; drop the ticket reference so
    // teardown doesn't try to leave a line we already left.
    queuedTicketId = "";
    // Always retain the session id so teardown immediately releases the shared
    // room slot. Metered deployments additionally need a periodic heartbeat.
    if (info.sessionId) {
      trackedSessionId = info.sessionId;
      trackedTier = info.tier || "room";
    }
    if (info.limited && info.sessionId) {
      startHeartbeat(info.heartbeatSec || 5);
    }
  });
  c.addEventListener("input-level", (e) => {
    const { rms } = /** @type {CustomEvent<{ rms: number }>} */ (e).detail;
    paintInputLevel(rms);
  });

  try {
    await c.connect();
  } catch (err) {
    // The grant can be refused (402 → limit) or the dial can fail. In LB mode
    // the AudioContext hasn't been adopted by the client yet (the session POST
    // runs first), so close the one we created here to avoid leaking it.
    if (audioContext) void audioContext.close().catch(() => {});
    throw err;
  }
}

// ── Conversation-time heartbeat ─────────────────────────────────────────────

/** Ping the server every `sec` seconds so it can meter the live session; when
 *  it reports the daily budget is spent, cut the call and show the limit modal.
 *  @param {number} sec */
function startHeartbeat(sec) {
  stopHeartbeat();
  heartbeatTimer = window.setInterval(async () => {
    if (!trackedSessionId) return;
    try {
      const res = await fetch("api/session/heartbeat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: trackedSessionId }),
        keepalive: true,
      });
      const json = await res.json().catch(() => ({}));
      if (json.expired) await onLimitReached();
    } catch (err) {
      // A transient network blip shouldn't kill the call; the next tick retries.
      if (DEBUG) console.debug("[ui] heartbeat failed:", err);
    }
  }, Math.max(1, sec) * 1000);
}

function stopHeartbeat() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
    heartbeatTimer = 0;
  }
}

/** The server cut the live session: tear down and explain why. */
async function onLimitReached() {
  const tier = trackedTier;
  stopHeartbeat();
  await teardown();
  account.showLimit(tier);
}

/** Tell the server a session ended so it reconciles + refunds the unused chunk.
 *  Uses sendBeacon so it still fires when the tab is closing. */
function endTrackedSession() {
  if (!trackedSessionId) return;
  const body = JSON.stringify({ sessionId: trackedSessionId });
  try {
    const blob = new Blob([body], { type: "application/json" });
    if (!navigator.sendBeacon("api/session/end", blob)) {
      void fetch("api/session/end", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // Best-effort; the server sweep reaps the session anyway.
  }
  trackedSessionId = "";
  trackedTier = "";
}

/** Leave the waiting queue so the LB frees our place. sendBeacon so it still
 *  fires on tab close; the LB also reaps the ticket on TTL as a backstop. */
function endQueueTicket() {
  if (!queuedTicketId) return;
  const body = JSON.stringify({ queueId: queuedTicketId });
  try {
    const blob = new Blob([body], { type: "application/json" });
    if (!navigator.sendBeacon("api/queue/end", blob)) {
      void fetch("api/queue/end", {
        method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
      }).catch(() => {});
    }
  } catch {
    // Best-effort; the LB reaps the ticket on TTL anyway.
  }
  queuedTicketId = "";
}

/** @param {string} status */
function onClientStatus(status) {
  switch (status) {
    case "creating-session":
    case "connecting":
      setState("connecting");
      break;
    case "queued":
      setState("queued");
      break;
    case "your-turn":
      setState("your-turn");
      break;
    case "connected":
      setState("listening");
      break;
    case "user-speaking":
      setState("user-speaking");
      break;
    case "processing":
      setState("processing");
      break;
    case "ai-speaking":
      setState("ai-speaking");
      break;
    case "closed":
      // teardown() will move us to idle
      break;
    case "error":
      setState("error");
      break;
  }
}

async function teardown() {
  stopHeartbeat();
  stopJoinCountdown();
  endTrackedSession();
  endQueueTicket();
  chat.reset({ dismiss: true });
  if (client) {
    try {
      await client.close();
    } catch (err) {
      console.warn("[main] error closing client:", err);
    }
    client = null;
  }
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
    micStream = null;
  }
  // The webcam is independent of the call lifecycle (it runs while the user is
  // on the page), so we leave it on here — only the camera toggle stops it.
  micMuted = false;
  micBtn.classList.remove("muted");
  setState("idle");
  // Refresh the chip's remaining-today after the budget moved.
  if (limiterOn) void account.refresh();
}

/** @param {unknown} err */
async function onFatalError(err) {
  console.error("[main] fatal:", err);
  const message = err instanceof Error ? err.message : String(err);
  try {
    await teardown();
  } catch (teardownError) {
    console.warn("[main] error during fatal teardown:", teardownError);
  } finally {
    setState("error");
    setCaption(truncateError(message), "error");
  }
}

setState("idle");
chat.renderEmptyState();
initGateArc();
void fetchConfig();
void (async () => {
  await refreshAvatarPicker();
  const savedAvatar = localStorage.getItem(STORAGE_KEYS.avatarId);
  if (!savedAvatar) return;
  try {
    const response = await fetch("/av/avatars", { cache: "no-store" });
    const data = await response.json();
    if (data.avatar_id !== savedAvatar) await selectAvatar(savedAvatar);
  } catch (err) {
    console.warn("[main] restore avatar failed:", err);
  }
})();
// Camera is opt-in only. `enableCamera()` is called exclusively by the camera
// toggle; landing on the page never requests or resumes webcam access.
void watchCameraPermission();

// Reconcile a live session if the tab is closed/hidden mid-call (no teardown).
window.addEventListener("pagehide", () => { endTrackedSession(); endQueueTicket(); });

requestAnimationFrame(() => {
  document.body.classList.remove("booting");
});
