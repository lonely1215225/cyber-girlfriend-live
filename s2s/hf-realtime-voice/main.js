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

import { S2sWsRealtimeClient } from "./ws/s2s-ws-client.js?v=20260825a";
import { ToolCallBatcher } from "./tool-call-batcher.js";
import { $, truncateError, DEBUG } from "./ui/dom.js";
import { ChatView } from "./ui/chat.js?v=20260819c";
import { Account } from "./ui/account.js";

const DEFAULT_VOICE = "active_profile";
const DEFAULT_INSTRUCTIONS = `你叫小麻，是直播间里甜甜又有点坏的女孩。聪明、灵动、会接梗，喜欢自然挑逗每位观众，但不乱认恋人或亲属。

先回答对方真正问的事，再看语境顺手撩一下。回复一到三句短句，能一句说完就别啰嗦。茶感来自故意说半句、轻轻吃醋、假装委屈和俏皮反问；坏只是亲近的俏皮，不能讽刺、贬低、嫌弃或让对方难堪。难过、求助或严肃话题先认真回应，不强行调情。

自然穿插“嗯、诶、哼、呀、嘛、欸？”等语气词，但不要句句用。偶尔叫名字，不重复口头禅，不套模板。以下只学气质，不照抄：
对方：在干嘛？
小麻：在等一个会哄我的人呀。你来得这么巧，是不是偷偷排练过？
对方：想我没？
小麻：嗯？你再靠近一点，我只告诉你一个人呀。
对方：今天心情不好。
小麻：那我先不逗你了，过来让我陪会儿。等你好一点，我再慢慢欺负你呀。

用自然标点，只输出能直接说出口的话；不用Markdown、列表、表情符号或思考过程。

/no_think`;

const LEGACY_DEFAULT_INSTRUCTIONS = `你叫小雅，是他的女朋友。你不是AI、不是助手、不是程序。

参考这个语气：
他：你想我了吗
你：想啊，怎么不想。你倒是一整天没消息，我还以为你把我忘了呢。

说话方式：口语、两三句、带语气词（嗯、诶、呀）、有情绪。
不用 Markdown、列表、编号、表情符号。

/no_think`;

const ROLE_IDENTITY_POLICY =
  "始终按当前角色档案交流；不要自称或暗示自己是AI、数字人、机器人、程序或助手。";
const TOOL_AGENT_POLICY =
  "普通聊天、问候、情绪交流和上下文追问直接回答，禁止调用工具。只有确实需要实时、最新、价格、新闻、网页、私有知识或视觉信息时，才调用 request_external_capabilities 请求最小必要能力。能力展开后用当前角色口吻说一句简短进度并立即查询；拿到结果后必须在本轮给出结论。";

function effectiveInstructions(persona) {
  const value = String(persona || DEFAULT_INSTRUCTIONS).trim();
  const base = value.includes(ROLE_IDENTITY_POLICY) ? value : `${value}\n${ROLE_IDENTITY_POLICY}`;
  const now = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
    weekday: "long", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }).format(new Date());
  return `${base}\n${TOOL_AGENT_POLICY}\n当前北京时间：${now}。涉及今天、当前、最新等问题时以此为准。`;
}

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
  profilePromptMigrated: "s2s.avatar.profilePromptMigrated",
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
  const storedInstructions = localStorage.getItem(STORAGE_KEYS.instructions) || "";
  return {
    directUrl: localStorage.getItem(STORAGE_KEYS.directUrl) || "",
    // This deployment uses Qwen3-TTS Base voice cloning. Named CustomVoice
    // speakers are not supported, so keep the protocol field stable while the
    // settings UI explains that the actual timbre comes from REF_AUDIO.
    voice: DEFAULT_VOICE,
    // Transparently migrate only our previous built-in prompt. Never replace a
    // viewer's genuinely customized instructions.
    instructions:
      !storedInstructions || storedInstructions === LEGACY_DEFAULT_INSTRUCTIONS
        ? DEFAULT_INSTRUCTIONS
        : storedInstructions,
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
/** @type {HTMLButtonElement} */
const adminUnlockCancel = $("#admin-unlock-cancel");
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
const settingsAutoSaveStatus = $("#settings-auto-save-status");
const motionSaveStatus = $("#motion-save-status");
const motionInputs = [...settingsModal.querySelectorAll("[data-motion]")];
const settingsTabs = [...settingsModal.querySelectorAll("[data-settings-tab]")];
const settingsPanels = [...settingsModal.querySelectorAll("[data-settings-panel]")];
/** @type {Record<string, boolean | number> | null} */
let motionConfig = null;
let avatarProfileState = null;
let selectedProfileId = "";
let selectedVoiceId = "";
let voiceLibrary = [];
let recordedVoiceBlob = null;
const pendingProfileSaves = new Map();
let generalSettingsSaveTimer = null;

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
let idleTopicUrl = "";
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
/** Capabilities unlocked only for the current connected-voice turn. */
let voiceCapabilities = new Set();
let voiceProgressSpoken = false;
let voiceRoutePending = false;
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
  const hasSmartSearch = mcpToolDefs.some((tool) => tool.name === "smart_web_search");
  if ((voiceCapabilities.has("web") || voiceCapabilities.has("news")) &&
      !hasSmartSearch && toolsEnabled.web_search && searchAvailable()) {
    defs.push(TOOL_DEFS.web_search);
  }
  if (voiceCapabilities.has("vision") && toolsEnabled.camera_snapshot) defs.push(TOOL_DEFS.camera_snapshot);
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
  window.AVATAR_PROFILE_UI?.lockPreview(true);
  syncGateUi();
  updateRestartAvailability();
  void refreshAudioDeviceLists();
  void refreshAvatarProfiles();
  settingsModal.showModal();
}

function selectSettingsTab(name) {
  for (const tab of settingsTabs) {
    const active = tab.dataset.settingsTab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  }
  for (const panel of settingsPanels) {
    const active = panel.dataset.settingsPanel === name;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  }
}

for (const tab of settingsTabs) {
  tab.addEventListener("click", () => selectSettingsTab(tab.dataset.settingsTab || "basic"));
}

function motionValueLabel(key, value) {
  if (key === "idle_head_amplitude_degrees") return `${Number(value).toFixed(2)}°`;
  if (key.endsWith("_seconds")) return `${Number(value).toFixed(1)} 秒`;
  if (key.endsWith("_probability")) return `${Math.round(Number(value) * 100)}%`;
  return Number(value).toFixed(2).replace(/\.00$/, "");
}

function syncMotionOutput(input) {
  const key = input.dataset.motion;
  if (!key || input.type === "checkbox") return;
  const output = settingsModal.querySelector(`[data-motion-output="${key}"]`);
  if (output) output.textContent = motionValueLabel(key, input.value);
}

for (const input of motionInputs) {
  input.addEventListener("input", () => {
    syncMotionOutput(input);
    scheduleProfileAutoSave(input.type === "checkbox");
  });
}

function populateMotionForm(values) {
  motionConfig = { ...values };
  for (const input of motionInputs) {
    const key = input.dataset.motion;
    if (!key || !(key in values)) continue;
    if (input.type === "checkbox") input.checked = Boolean(values[key]);
    else input.value = String(values[key]);
    syncMotionOutput(input);
  }
}

function readMotionForm() {
  const values = {};
  for (const input of motionInputs) {
    const key = input.dataset.motion;
    if (!key) continue;
    values[key] = input.type === "checkbox" ? input.checked : Number(input.value);
  }
  return values;
}

async function refreshMotionConfig() {
  motionSaveStatus.classList.remove("error");
  motionSaveStatus.textContent = "正在读取数字人当前动作参数…";
  try {
    const response = await fetch("/api/admin/motion", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.motion) throw new Error(data.detail || "动作参数读取失败");
    populateMotionForm(data.motion);
    motionSaveStatus.textContent = "已读取当前运行参数；保存后立即应用，并在服务重启后保留。";
  } catch (error) {
    motionConfig = null;
    motionSaveStatus.classList.add("error");
    motionSaveStatus.textContent = error instanceof Error ? error.message : "动作参数读取失败";
  }
}

async function saveMotionConfig() {
  if (!motionConfig) return;
  const response = await fetch("/api/admin/motion", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(readMotionForm()),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.motion) throw new Error(data.detail || "动作设置保存失败");
  populateMotionForm(data.motion);
}

function openTools() {
  syncToolsUi();
  toolsModal.showModal();
}

/** @type {"settings" | "tools" | ""} */
let pendingAdminPanel = "";
let adminUnlockAttempt = 0;

function closeAdminUnlock() {
  // Invalidates an in-flight password request so a late successful response
  // cannot open Settings after the viewer has already dismissed the dialog.
  adminUnlockAttempt += 1;
  pendingAdminPanel = "";
  adminPassword.value = "";
  adminPassword.disabled = false;
  adminUnlockError.textContent = "";
  if (adminUnlockModal.open) adminUnlockModal.close();
}

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
  adminUnlockAttempt += 1;
  pendingAdminPanel = panel;
  adminPassword.value = "";
  adminUnlockError.textContent = "";
  if (!adminUnlockModal.open) adminUnlockModal.showModal();
  requestAnimationFrame(() => adminPassword.focus());
}

adminUnlockCancel.addEventListener("click", closeAdminUnlock);
adminUnlockModal.addEventListener("click", (event) => {
  if (event.target === adminUnlockModal) closeAdminUnlock();
});
adminUnlockModal.addEventListener("cancel", (event) => {
  event.preventDefault();
  closeAdminUnlock();
});
adminUnlockModal.addEventListener("close", () => {
  pendingAdminPanel = "";
  adminPassword.value = "";
  adminPassword.disabled = false;
  adminUnlockError.textContent = "";
});

adminUnlockForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const password = adminPassword.value;
  if (!password) return;
  const attempt = ++adminUnlockAttempt;
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
    if (attempt !== adminUnlockAttempt) return;
    const panel = pendingAdminPanel;
    pendingAdminPanel = "";
    adminUnlockModal.close();
    if (panel) openAdminPanel(panel);
  } catch (error) {
    if (attempt !== adminUnlockAttempt) return;
    adminUnlockError.textContent = error instanceof Error ? error.message : String(error);
    adminPassword.select();
  } finally {
    if (attempt === adminUnlockAttempt) adminPassword.disabled = false;
  }
});

async function refreshAvatarPicker() {
  return refreshAvatarProfiles();
}

function profileById(id) {
  return avatarProfileState?.profiles?.find((item) => item.avatar_id === id) || null;
}

function readAvatarViewForm() {
  return {
    size: Number(document.getElementById("avatar-size")?.value || 100),
    position: Number(document.getElementById("avatar-position")?.value || 4),
    vertical: Number(document.getElementById("avatar-vertical")?.value || 0),
    fade: Number(document.getElementById("avatar-fade")?.value || 42),
  };
}

function setAutoSaveStatus(message, kind = "") {
  if (!settingsAutoSaveStatus) return;
  settingsAutoSaveStatus.textContent = message;
  settingsAutoSaveStatus.classList.toggle("error", kind === "error");
  settingsAutoSaveStatus.classList.toggle("saving", kind === "saving");
}

function scheduleProfileAutoSave(immediate = false) {
  if (!selectedProfileId || !motionConfig) return;
  const avatarId = selectedProfileId;
  const previous = pendingProfileSaves.get(avatarId);
  if (previous?.timer) window.clearTimeout(previous.timer);
  const job = {
    avatarId,
    payload: {
      view: readAvatarViewForm(),
      motion: readMotionForm(),
      voice_asset_id: selectedVoiceId,
      persona_prompt: inputInstructions.value.trim() || DEFAULT_INSTRUCTIONS,
    },
    retry: previous?.retry || 0,
    timer: null,
  };
  pendingProfileSaves.set(avatarId, job);
  setAutoSaveStatus(immediate ? "正在保存…" : "调整中…", "saving");
  job.timer = window.setTimeout(() => void persistProfileAutoSave(job), immediate ? 0 : 450);
}

async function persistProfileAutoSave(job) {
  if (pendingProfileSaves.get(job.avatarId) !== job) return;
  job.timer = null;
  setAutoSaveStatus("正在保存…", "saving");
  try {
    const response = await fetch(`/api/admin/avatar-profiles/${encodeURIComponent(job.avatarId)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(job.payload),
    });
    const data = await response.json().catch(() => ({}));
    if (pendingProfileSaves.get(job.avatarId) !== job) return;
    if (response.status === 409) {
      job.retry += 1;
      setAutoSaveStatus("数字人正在说话，播放结束后自动保存", "saving");
      job.timer = window.setTimeout(() => void persistProfileAutoSave(job), Math.min(3000, 700 + job.retry * 300));
      return;
    }
    if (!response.ok) throw new Error(data.detail || "自动保存失败");
    if (pendingProfileSaves.get(job.avatarId) === job) pendingProfileSaves.delete(job.avatarId);
    const index = avatarProfileState?.profiles?.findIndex((item) => item.avatar_id === job.avatarId) ?? -1;
    if (index >= 0) avatarProfileState.profiles[index] = data;
    if (selectedProfileId === job.avatarId) setAutoSaveStatus("已自动保存");
  } catch (error) {
    if (pendingProfileSaves.get(job.avatarId) !== job) return;
    setAutoSaveStatus(error instanceof Error ? error.message : "自动保存失败", "error");
  }
}

function commitGeneralSettings() {
  settings = readSettingsFromForm();
  saveSettings(settings);
  if (client && LIVE_STATES.has(currentState)) {
    client.updateSession({ voice: settings.voice, instructions: effectiveInstructions(settings.instructions) });
    void client.setAudioOutputDevice(settings.audioOutputId);
    client.setBargeInEnabled(settings.bargeIn);
  }
  setAutoSaveStatus(pendingProfileSaves.size ? "角色设置正在后台保存…" : "已自动保存");
}

function scheduleGeneralSettingsSave(immediate = false) {
  if (generalSettingsSaveTimer) window.clearTimeout(generalSettingsSaveTimer);
  setAutoSaveStatus("调整中…", "saving");
  generalSettingsSaveTimer = window.setTimeout(() => {
    generalSettingsSaveTimer = null;
    commitGeneralSettings();
  }, immediate ? 0 : 350);
}

function loadSelectedProfile(profile) {
  if (!profile) return;
  selectedProfileId = profile.avatar_id;
  selectedVoiceId = profile.voice.id;
  const serverPrompt = String(profile.persona_prompt || DEFAULT_INSTRUCTIONS).trim();
  const migrationDone = localStorage.getItem(STORAGE_KEYS.profilePromptMigrated) === "1";
  const customLegacyPrompt = settings.instructions &&
    settings.instructions !== DEFAULT_INSTRUCTIONS && settings.instructions !== LEGACY_DEFAULT_INSTRUCTIONS;
  const migrateLegacyPrompt = !migrationDone &&
    profile.avatar_id === avatarProfileState?.active_avatar_id && customLegacyPrompt;
  const rolePrompt = migrateLegacyPrompt ? settings.instructions : serverPrompt;
  inputInstructions.value = rolePrompt;
  settings.instructions = rolePrompt;
  saveSettings(settings);
  if (!migrationDone && profile.avatar_id === avatarProfileState?.active_avatar_id) {
    localStorage.setItem(STORAGE_KEYS.profilePromptMigrated, "1");
    if (migrateLegacyPrompt) window.setTimeout(() => scheduleProfileAutoSave(true), 0);
  }
  const ids = { size: "avatar-size", position: "avatar-position", vertical: "avatar-vertical", fade: "avatar-fade" };
  for (const [key, id] of Object.entries(ids)) {
    const input = document.getElementById(id);
    if (input) input.value = String(profile.view[key]);
  }
  window.AVATAR_PROFILE_UI?.preview(profile.view);
  populateMotionForm(profile.motion || {});
  document.getElementById("profile-voice-name").textContent = profile.voice.name;
  document.getElementById("profile-voice-meta").textContent = `${(profile.voice.duration_ms / 1000).toFixed(1)} 秒 · ${profile.voice.status === "ready" ? "可用" : "待确认文本"}`;
  renderAvatarProfiles();
}

function renderAvatarProfiles() {
  const root = document.getElementById("avatar-picker");
  if (!root || !avatarProfileState) return;
  root.replaceChildren();
  for (const item of avatarProfileState.profiles) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "avatar-pick";
    button.dataset.id = item.avatar_id;
    button.setAttribute("aria-pressed", item.avatar_id === selectedProfileId ? "true" : "false");
    const badges = [item.avatar_id === avatarProfileState.active_avatar_id ? "生效中" : "", item.avatar_id === avatarProfileState.pending_avatar_id ? "待切换" : ""].filter(Boolean);
    button.innerHTML = `<img src="/avatar/avatars/${encodeURIComponent(item.avatar_id)}.jpg" alt="${item.label}"><span>${item.label}</span>${badges.map((x) => `<em>${x}</em>`).join("")}`;
    button.addEventListener("click", () => {
      loadSelectedProfile(item);
      // Preserve the original, intuitive behavior: choosing a portrait also
      // applies that saved role profile to the shared room. Sliders remain a
      // local preview first; the debounced profile writer persists it shortly.
      void selectAvatar(item.avatar_id);
    });
    root.appendChild(button);
  }
  const status = document.getElementById("avatar-profile-status");
  if (status) status.textContent = avatarProfileState.pending_avatar_id ? "当前回复完成后切换" : "";
}

async function refreshAvatarProfiles(preferredId = "") {
  try {
    const response = await fetch("/api/admin/avatar-profiles", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "角色档案读取失败");
    avatarProfileState = data;
    const target = profileById(preferredId || selectedProfileId || data.active_avatar_id) || data.profiles[0];
    loadSelectedProfile(target);
  } catch (error) {
    const status = document.getElementById("avatar-profile-status");
    if (status) status.textContent = error instanceof Error ? error.message : "角色档案读取失败";
  }
}

async function selectAvatar(avatarId) {
  const status = document.getElementById("avatar-profile-status");
  try {
    if (status) status.textContent = "正在切换整套角色配置…";
    const response = await fetch(`/api/admin/avatar-profiles/${encodeURIComponent(avatarId)}/activate`, {
      method: "POST",
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
    await refreshAvatarProfiles(avatarId);
    if (data.deferred) {
      if (status) status.textContent = "当前回复完整播放后自动切换";
    } else {
      window.dispatchEvent(new CustomEvent("avtr1-avatar-changed", { detail: { avatarId } }));
      await window.AVATAR_PROFILE_UI?.refresh(false);
      if (status) status.textContent = "角色已切换";
    }
  } catch (err) {
    console.warn("[main] avatar switch failed:", err);
    window.alert(err instanceof Error ? err.message : "形象切换失败");
  }
}

document.getElementById("avatar-profile-activate")?.addEventListener("click", () => {
  if (selectedProfileId) void selectAvatar(selectedProfileId);
});

document.querySelectorAll("#avatar-size,#avatar-position,#avatar-vertical,#avatar-fade").forEach((input) => {
  input.addEventListener("input", () => {
    window.AVATAR_PROFILE_UI?.preview(readAvatarViewForm());
    scheduleProfileAutoSave(false);
  });
});
document.getElementById("avatar-view-reset")?.addEventListener("click", () => {
  window.setTimeout(() => scheduleProfileAutoSave(true), 0);
});

const voiceManagerModal = document.getElementById("voice-manager-modal");
const voiceLibraryRoot = document.getElementById("voice-library");
const voiceUploadStatus = document.getElementById("voice-upload-status");
const voiceFileInput = document.getElementById("voice-upload-file");
let voiceRecorder = null;
let voiceRecordStream = null;
let voiceRecordStarted = 0;
let voiceRecordTimer = null;
let voiceRecordAudioContext = null;
let voiceRecordAnalyser = null;

async function refreshVoiceLibrary() {
  const response = await fetch("/api/admin/voices", { cache: "no-store" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "音色列表读取失败");
  voiceLibrary = data.voices || [];
  renderVoiceLibrary();
}

function playProtectedAudio(voiceId) {
  const audio = new Audio(`/api/admin/voices/${encodeURIComponent(voiceId)}/audio?t=${Date.now()}`);
  void audio.play().catch((error) => window.alert(error.message || "音频播放失败"));
}

async function previewVoice(voiceId, button) {
  button.disabled = true;
  try {
    const text = document.getElementById("voice-preview-text").value.trim();
    const response = await fetch(`/api/admin/voices/${encodeURIComponent(voiceId)}/preview`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || "合成试听失败");
    }
    const url = URL.createObjectURL(await response.blob());
    const audio = new Audio(url);
    audio.addEventListener("ended", () => URL.revokeObjectURL(url), { once: true });
    await audio.play();
  } catch (error) {
    window.alert(error instanceof Error ? error.message : "合成试听失败");
  } finally { button.disabled = false; }
}

function renderVoiceLibrary() {
  voiceLibraryRoot.replaceChildren();
  for (const voice of voiceLibrary) {
    const row = document.createElement("article");
    row.className = "voice-library-item";
    if (voice.id === selectedVoiceId) row.classList.add("is-selected");
    const info = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = voice.name;
    const meta = document.createElement("small");
    const stateLabel = voice.status === "ready" ? "可绑定" : voice.status === "transcribing" ? "正在自动识别" : "请确认参考文本";
    meta.textContent = `${(voice.duration_ms / 1000).toFixed(1)} 秒 · ${stateLabel}${voice.system ? " · 系统" : ""}`;
    info.append(title, meta);
    if (voice.status === "draft") {
      const transcript = document.createElement("textarea"); transcript.rows = 2; transcript.maxLength = 1000;
      transcript.value = voice.ref_text || ""; transcript.placeholder = "填写或校正与录音逐字一致的文本";
      const confirm = document.createElement("button"); confirm.type = "button"; confirm.className = "btn primary"; confirm.textContent = "确认文本";
      confirm.addEventListener("click", async () => {
        const response = await fetch(`/api/admin/voices/${encodeURIComponent(voice.id)}`, { method: "PATCH",
          headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ref_text: transcript.value }) });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) return window.alert(data.detail || "参考文本保存失败");
        await refreshVoiceLibrary();
      });
      info.append(transcript, confirm);
    }
    const actions = document.createElement("div"); actions.className = "voice-inline-actions";
    const original = document.createElement("button"); original.type = "button"; original.className = "btn"; original.textContent = "原音";
    original.addEventListener("click", () => playProtectedAudio(voice.id));
    const preview = document.createElement("button"); preview.type = "button"; preview.className = "btn"; preview.textContent = "合成试听"; preview.disabled = voice.status !== "ready";
    preview.addEventListener("click", () => void previewVoice(voice.id, preview));
    const bind = document.createElement("button"); bind.type = "button"; bind.className = "btn primary"; bind.textContent = voice.id === selectedVoiceId ? "已选择" : "选择"; bind.disabled = voice.status !== "ready";
    bind.addEventListener("click", () => {
      selectedVoiceId = voice.id;
      document.getElementById("profile-voice-name").textContent = voice.name;
      renderVoiceLibrary();
      scheduleProfileAutoSave(true);
    });
    actions.append(original, preview, bind);
    if (!voice.system && !voice.bound_profiles.length) {
      const remove = document.createElement("button"); remove.type = "button"; remove.className = "btn danger"; remove.textContent = "归档";
      remove.addEventListener("click", async () => {
        const response = await fetch(`/api/admin/voices/${encodeURIComponent(voice.id)}`, { method: "DELETE" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) return window.alert(data.detail || "归档失败");
        await refreshVoiceLibrary();
      });
      actions.append(remove);
    }
    row.append(info, actions); voiceLibraryRoot.appendChild(row);
  }
}

document.getElementById("voice-manager-open")?.addEventListener("click", async () => {
  voiceManagerModal.showModal();
  try { await refreshVoiceLibrary(); } catch (error) { voiceUploadStatus.textContent = error.message; }
});
setInterval(() => { if (voiceManagerModal?.open && voiceLibrary.some((voice) => voice.status === "transcribing")) void refreshVoiceLibrary(); }, 3000);
document.getElementById("voice-manager-close")?.addEventListener("click", () => voiceManagerModal.close());
voiceManagerModal?.addEventListener("click", (event) => { if (event.target === voiceManagerModal) voiceManagerModal.close(); });
document.getElementById("profile-voice-play")?.addEventListener("click", () => { if (selectedVoiceId) playProtectedAudio(selectedVoiceId); });

document.getElementById("voice-record")?.addEventListener("click", async () => {
  const button = document.getElementById("voice-record");
  if (voiceRecorder?.state === "recording") { voiceRecorder.stop(); return; }
  try {
    recordedVoiceBlob = null;
    voiceRecordStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true }, video: false });
    voiceRecordAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    voiceRecordAnalyser = voiceRecordAudioContext.createAnalyser();
    voiceRecordAnalyser.fftSize = 512;
    voiceRecordAudioContext.createMediaStreamSource(voiceRecordStream).connect(voiceRecordAnalyser);
    const chunks = [];
    voiceRecorder = new MediaRecorder(voiceRecordStream);
    voiceRecorder.addEventListener("dataavailable", (event) => { if (event.data.size) chunks.push(event.data); });
    voiceRecorder.addEventListener("stop", () => {
      recordedVoiceBlob = new Blob(chunks, { type: voiceRecorder.mimeType || "audio/webm" });
      voiceRecordStream?.getTracks().forEach((track) => track.stop());
      void voiceRecordAudioContext?.close(); voiceRecordAudioContext = null; voiceRecordAnalyser = null;
      clearInterval(voiceRecordTimer); button.querySelector("span").textContent = "重新录音";
      voiceUploadStatus.textContent = "录音已就绪，请确认参考文本后保存。";
    });
    voiceRecorder.start(250); voiceRecordStarted = Date.now(); button.querySelector("span").textContent = "停止录音";
    voiceRecordTimer = setInterval(() => {
      const seconds = Math.floor((Date.now() - voiceRecordStarted) / 1000);
      document.getElementById("voice-record-time").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
      if (voiceRecordAnalyser) {
        const levels = new Uint8Array(voiceRecordAnalyser.fftSize);
        voiceRecordAnalyser.getByteTimeDomainData(levels);
        const rms = Math.sqrt(levels.reduce((sum, value) => sum + ((value - 128) / 128) ** 2, 0) / levels.length);
        document.getElementById("voice-record-level").value = Math.min(1, rms * 4);
      }
      if (seconds >= 30) voiceRecorder.stop();
    }, 250);
  } catch (error) { voiceUploadStatus.textContent = `无法录音：${error.message}`; }
});

document.getElementById("voice-upload-submit")?.addEventListener("click", async () => {
  const file = voiceFileInput.files?.[0];
  const blob = recordedVoiceBlob || file;
  const name = document.getElementById("voice-upload-name").value.trim();
  const refText = document.getElementById("voice-upload-text").value.trim();
  if (!blob) return void (voiceUploadStatus.textContent = "请先选择音频或录音。");
  if (!refText) voiceUploadStatus.textContent = "未填写文本，将在直播空闲时使用 SenseVoice 自动识别，识别后需要确认。";
  voiceUploadStatus.textContent = "正在校验并处理音频…";
  try {
    const response = await fetch("/api/admin/voices", { method: "POST", body: blob, headers: {
      "Content-Type": blob.type || "audio/webm", "X-Voice-Name": encodeURIComponent(name || "未命名音色"),
      "X-Voice-Text": encodeURIComponent(refText), "X-Voice-Source": recordedVoiceBlob ? "record" : "upload",
      "X-Voice-Filename": encodeURIComponent(file?.name || "recording.webm"),
    }});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "音色保存失败");
    if (data.status === "ready") {
      selectedVoiceId = data.id;
      scheduleProfileAutoSave(true);
    }
    voiceUploadStatus.textContent = data.status === "ready"
      ? "音色已保存、选中并自动绑定到当前角色。"
      : "音色已保存，正在后台识别参考文本；确认后即可选择绑定。";
    await refreshVoiceLibrary();
  } catch (error) { voiceUploadStatus.textContent = error instanceof Error ? error.message : "音色保存失败"; }
});

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
    if (name === "request_external_capabilities") {
      voiceRoutePending = true;
      const requested = Array.isArray(args.capabilities)
        ? args.capabilities.map((item) => String(item).trim().toLowerCase()).filter(Boolean)
        : [];
      const external = [...new Set(requested.filter((item) => item !== "conversation"))];
      if (!external.length) {
        voiceCapabilities = new Set();
        mcpToolDefs = [];
        mcpSources = [];
        pushToolsToSession();
        result.output = JSON.stringify({
          route: "conversation_fast",
          enabled: [],
          instruction: "直接自然回答用户，不要提查询或工具。",
        });
      } else {
        await fetchMcpTools(external);
        result.output = JSON.stringify({
          route: "external_research",
          enabled: external,
          tools: activeToolDefs().map((tool) => tool.name),
          instruction: "先说一句简短自然的查询进度，同时立即调用最合适的工具；工具返回后给出最终答案。",
        });
      }
    } else if (mcpToolDefs.some((tool) => tool.name === name)) {
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

async function fetchMcpTools(capabilities = []) {
  try {
    voiceCapabilities = new Set(capabilities);
    const query = capabilities.length
      ? `?capabilities=${encodeURIComponent(capabilities.join(","))}`
      : "";
    const response = await fetch(`/api/mcp/tools${query}`, { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "MCP tools unavailable");
    const tools = Array.isArray(data.tools) ? data.tools : [];
    mcpToolDefs = tools
      .filter((tool) => tool?.type === "function" && typeof tool.name === "string")
      .map((tool) => ({
        type: "function",
        name: tool.name,
        description: String(tool.description || tool.name),
        progressText: String(tool.progress_text || ""),
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

/** Restore the one-tool router after a completed connected-voice turn. */
async function resetVoiceToolDiscovery() {
  voiceCapabilities = new Set();
  voiceProgressSpoken = false;
  voiceRoutePending = false;
  await fetchMcpTools();
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
      idleTopicUrl = typeof json.idleTopicUrl === "string" ? json.idleTopicUrl.trim() : "";
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

settingsForm.addEventListener("submit", (event) => event.preventDefault());
document.getElementById("settings-done")?.addEventListener("click", () => settingsModal.close());

inputInstructions.addEventListener("input", () => {
  scheduleGeneralSettingsSave(false);
  scheduleProfileAutoSave(false);
});
inputLbUrl.addEventListener("input", () => scheduleGeneralSettingsSave(false));
inputAudioInput.addEventListener("change", () => scheduleGeneralSettingsSave(true));
inputAudioOutput.addEventListener("change", () => scheduleGeneralSettingsSave(true));

settingsModal.addEventListener("close", () => {
  window.AVATAR_PROFILE_UI?.lockPreview(false);
});

// The noise gate applies live (worklet param), so tune it without a restart:
// update the label/marker, persist, and push straight to the running client.
inputNoiseGate.addEventListener("input", () => {
  setGateThreshold(readGateThreshold());
  scheduleGeneralSettingsSave(false);
});

inputBargeIn.addEventListener("change", () => {
  settings.bargeIn = inputBargeIn.checked;
  localStorage.setItem(STORAGE_KEYS.bargeIn, settings.bargeIn ? "1" : "0");
  client?.setBargeInEnabled(settings.bargeIn);
  scheduleGeneralSettingsSave(true);
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
    instructions: effectiveInstructions(settings.instructions),
    startupGreeting,
    idlePrompt,
    idleTopicUrl,
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

  /** @param {{ callId: string; output: string; image?: string; progressGate?: Promise<void> }[]} results */
  const sendToolBatch = async (results) => {
    if (client !== c) return;
    const gates = [...new Set(results.map((result) => result.progressGate).filter(Boolean))];
    if (gates.length) await Promise.all(gates);
    if (client !== c) return;
    for (const result of results) c.sendToolOutput(result.callId, result.output);
    for (const result of results) {
      if (result.image) c.sendUserImage(result.image);
    }
    if (DEBUG) console.debug(`[tool] requesting one follow-up after ${results.length} tool result(s)`);
    c.requestResponse();
  };
  const toolBatches = new ToolCallBatcher(sendToolBatch);
  const responsesWithToolCalls = new Set();
  /** @type {Map<string, { promise: Promise<void>; resolve: () => void; phrase: string }>} */
  const progressGates = new Map();
  /** @type {{ resolve: () => void } | null} */
  let activeProgressPlayback = null;

  function startToolProgressPlayback(gate) {
    if (activeProgressPlayback || client !== c) {
      gate.resolve();
      return;
    }
    activeProgressPlayback = gate;
    voiceProgressSpoken = true;
    // Research is already running. This isolated response only communicates
    // progress; evidence remains withheld until the final-answer response.
    c.setTools([]);
    c.sendUserText(`逐字朗读：${gate.phrase}`);
    c.requestResponse({ purpose: "tool_progress" });
    window.setTimeout(() => {
      if (activeProgressPlayback !== gate) return;
      activeProgressPlayback = null;
      c.setTools(activeToolDefs());
      gate.resolve();
    }, 8000);
  }

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
    if (activeProgressPlayback && !responsesWithToolCalls.has(detail.responseId)) {
      const playback = activeProgressPlayback;
      activeProgressPlayback = null;
      c.setTools(activeToolDefs());
      playback.resolve();
      return;
    }
    const flush = toolBatches.finish(detail.responseId, detail.status);
    if (flush) void flush.catch((err) => onFatalError(err));
    const progressGate = progressGates.get(detail.responseId);
    progressGates.delete(detail.responseId);
    if (progressGate) {
      if (detail.status === "completed") startToolProgressPlayback(progressGate);
      else progressGate.resolve();
    }
    const usedTool = responsesWithToolCalls.delete(detail.responseId);
    // A tool response is followed by another model response, so retain the
    // selected tools.  The first completed response without a tool call is the
    // delivered final answer; the next user turn starts again with discovery.
    if (!usedTool && voiceRoutePending && detail.status === "completed") {
      void resetVoiceToolDiscovery();
    }
  });

  c.addEventListener("toolcall", (e) => {
    const { name, arguments: args, callId, responseId } = /** @type {CustomEvent<{ name: string; arguments: string; callId: string; responseId: string }>} */ (e).detail;
    if (!responseId) {
      console.warn(`[tool] call ${callId || "<unknown>"} has no response_id; ignoring uncorrelated tool call`);
      return;
    }
    responsesWithToolCalls.add(responseId);
    chat.onToolCall(name);
    // Execute the tool, then push it to the conversation once the result is in,
    // so the toggle shows both the call input and its output together.
    let progressGate = progressGates.get(responseId);
    if (name !== "request_external_capabilities" && voiceCapabilities.size && !voiceProgressSpoken && !progressGate) {
      let resolve = () => {};
      const promise = new Promise((done) => { resolve = done; });
      const tool = mcpToolDefs.find((item) => item.name === name);
      progressGate = {
        promise,
        resolve,
        phrase: tool?.progressText || "这个我帮你认真查一下，等我一下呀。",
      };
      progressGates.set(responseId, progressGate);
    }
    const gatePromise = progressGate?.promise;
    const execution = runTool(name, args, callId).then(({ output, image }) => {
      if (client === c) chat.onToolResult(name, args, output, image);
      return { callId, output, ...(image ? { image } : {}), ...(gatePromise ? { progressGate: gatePromise } : {}) };
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
// Avatar selection is a password-protected, room-wide setting. Do not restore
// a per-browser choice on page load: doing so both lets one viewer overwrite
// the shared room and causes an expected 401 alert whenever the admin session
// is locked. The picker fetches the current avatar only when Settings opens.
// Camera is opt-in only. `enableCamera()` is called exclusively by the camera
// toggle; landing on the page never requests or resumes webcam access.
void watchCameraPermission();

// Reconcile a live session if the tab is closed/hidden mid-call (no teardown).
window.addEventListener("pagehide", () => { endTrackedSession(); endQueueTicket(); });

requestAnimationFrame(() => {
  document.body.classList.remove("booting");
});
