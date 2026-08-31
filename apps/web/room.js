// @ts-check

const $ = (selector) => document.querySelector(selector);
const viewerCount = $("#room-viewer-count");
const activeCaller = $("#room-active-caller");
const queueCount = $("#room-queue-count");
const queueList = $("#room-queue-list");
const myName = $("#room-my-name");
const myNames = $("#room-generated-names");
const myStatus = $("#room-my-status");
const nameButton = $("#room-name-btn");
const nameDialog = /** @type {HTMLDialogElement | null} */ ($("#room-name-dialog"));
const nameForm = /** @type {HTMLFormElement | null} */ ($("#room-name-form"));
const nameInput = /** @type {HTMLInputElement | null} */ ($("#room-name-input"));
const nameError = $("#room-name-error");
const liveTranscript = $("#room-live-transcript");
const chatEmpty = $("#room-chat-empty");
const chatForm = /** @type {HTMLFormElement | null} */ ($("#room-chat-form"));
const chatInput = /** @type {HTMLInputElement | null} */ ($("#room-chat-input"));
const chatSend = /** @type {HTMLButtonElement | null} */ ($("#room-chat-send"));
const chatError = $("#room-chat-error");
const mentionMenu = $("#room-mention-menu");
const mentionOption = /** @type {HTMLButtonElement | null} */ ($("#room-mention-xiaoma"));
const liveChat = $("#room-live-chat");
const chatToggle = /** @type {HTMLButtonElement | null} */ ($("#room-chat-toggle"));
const chatToggleLabel = $("#room-chat-toggle-label");
const viewersButton = /** @type {HTMLButtonElement | null} */ ($("#room-viewers-btn"));
const viewersPopover = $("#room-viewers-popover");
const viewersClose = /** @type {HTMLButtonElement | null} */ ($("#room-viewers-close"));
const viewersList = $("#room-viewers-list");
const viewersEmpty = $("#room-viewers-empty");
const viewersSummary = $("#room-viewers-summary");

let source = null;
let reconnectTimer = 0;
let current = null;
let chatExpanded = false;
const transcriptNodes = new Map();

function text(node, value) {
  if (node) node.textContent = value;
}

function setChatExpanded(expanded) {
  chatExpanded = Boolean(expanded);
  liveChat?.classList.toggle("is-collapsed", !chatExpanded);
  liveChat?.classList.toggle("is-expanded", chatExpanded);
  chatToggle?.setAttribute("aria-expanded", String(chatExpanded));
  text(chatToggleLabel, chatExpanded ? "收起" : "展开");
  if (chatExpanded && liveTranscript) {
    requestAnimationFrame(() => { liveTranscript.scrollTop = liveTranscript.scrollHeight; });
  }
}

function mentionRange() {
  if (!chatInput) return null;
  const cursor = chatInput.selectionStart ?? chatInput.value.length;
  const beforeCursor = chatInput.value.slice(0, cursor);
  const at = beforeCursor.lastIndexOf("@");
  if (at < 0) return null;
  const query = beforeCursor.slice(at + 1);
  if (/\s/.test(query) || !"小麻".startsWith(query)) return null;
  return { start: at, end: cursor };
}

function setMentionOpen(open) {
  const visible = Boolean(open && mentionMenu && mentionOption && !chatInput?.disabled);
  if (mentionMenu) mentionMenu.hidden = !visible;
  mentionOption?.setAttribute("aria-selected", String(visible));
  chatInput?.setAttribute("aria-expanded", String(visible));
}

function updateMentionMenu() {
  setMentionOpen(Boolean(mentionRange()));
}

function insertXiaomaMention() {
  if (!chatInput) return;
  const range = mentionRange();
  if (!range) return;
  const suffix = chatInput.value.slice(range.end);
  const spacer = suffix.startsWith(" ") ? "" : " ";
  chatInput.value = `${chatInput.value.slice(0, range.start)}@小麻${spacer}${suffix}`;
  const cursor = range.start + 3 + spacer.length;
  chatInput.setSelectionRange(cursor, cursor);
  setMentionOpen(false);
  chatInput.focus();
}

function setViewersOpen(open) {
  if (!viewersPopover || !viewersButton) return;
  viewersPopover.hidden = !open;
  viewersButton.setAttribute("aria-expanded", String(open));
  viewersButton.classList.toggle("is-open", open);
}

function renderViewers(state) {
  if (!viewersList) return;
  const viewers = Array.isArray(state.viewers) ? state.viewers : [];
  viewersList.replaceChildren();
  text(viewersSummary, `${state.viewer_count ?? viewers.length} 人在线`);
  if (viewersEmpty) viewersEmpty.hidden = viewers.length > 0;
  const statusLabels = { calling: "连线中", ready: "待连线", queued: "排队中", watching: "观看中" };
  for (const viewer of viewers) {
    const item = document.createElement("li");
    const isMe = Boolean(state.me?.id && viewer.id === state.me.id);
    if (isMe) item.classList.add("is-me");
    const avatar = document.createElement("span");
    avatar.className = "room-viewer-avatar";
    avatar.textContent = String(viewer.name || "观").trim().slice(0, 1) || "观";
    const info = document.createElement("span");
    info.className = "room-viewer-info";
    const name = document.createElement("strong");
    name.textContent = viewer.name || "匿名观众";
    const status = document.createElement("small");
    status.textContent = `${statusLabels[viewer.status] || "观看中"}${isMe ? " · 我" : ""}`;
    info.append(name, status);
    const dot = document.createElement("i");
    dot.setAttribute("aria-label", "在线");
    item.append(avatar, info, dot);
    viewersList.appendChild(item);
  }
}

function render(state) {
  if (!state || !state.me) return;
  current = state;
  text(viewerCount, String(state.viewer_count ?? 0));
  text(queueCount, String(state.queue?.length ?? 0));
  text(activeCaller, state.active?.name || "暂无连线");
  text(myName, state.me.name || "观众");
  text(myNames, [state.me.name_zh, state.me.name_en].filter(Boolean).join(""));
  renderViewers(state);

  const labels = {
    watching: "正在观看",
    queued: `排队第 ${state.me.queue_position || "-"} 位`,
    ready: "轮到你连线",
    calling: "正在连线",
  };
  text(myStatus, labels[state.me.status] || "正在观看");
  document.body.dataset.roomStatus = state.me.status || "watching";
  const jobs = Array.isArray(state.agent_jobs) ? state.agent_jobs : [];
  const jobsByMessage = new Map(jobs.map((job) => [job.message_id, job]));
  const messages = (Array.isArray(state.messages) ? state.messages : []).filter((item) => {
    // Lifecycle rows are rendered from agent_jobs below. Ignore copies left by
    // older servers, and remove an obsolete partial progress line as soon as
    // the corresponding task reaches a terminal result.
    if (item.kind === "agent_status") return false;
    const job = jobsByMessage.get(item.reply_to?.id);
    return !(job?.terminal && item.partial && item.role === "assistant");
  });
  const messageById = new Map(messages.map((item) => [item.id, item]));
  for (const job of jobs) {
    const origin = messageById.get(job.message_id);
    if (job.terminal || !job.id || !origin) continue;
    const alreadyHasSpokenProgress = messages.some((item) => (
      item.partial && item.role === "assistant" && item.reply_to?.id === job.message_id
    ));
    if (alreadyHasSpokenProgress) continue;
    messages.push({
      id: `agent-status:${job.id}`,
      kind: "agent_status",
      role: "assistant",
      speaker: "小麻",
      text: job.status_text || "已进入回复队列",
      partial: true,
      interrupted: false,
      // Keep the status attached to its source comment. Using updated_at made
      // the same row jump to the bottom on every phase update.
      created_at: Number(origin.created_at || job.created_at || Date.now() / 1000) + 0.000001,
      agent_job_id: job.id,
      agent_phase: job.phase || "queued",
      reply_to: {
        id: job.message_id,
        speaker: job.speaker || "观众",
        text: `@小麻 ${job.prompt || ""}`.trim(),
      },
    });
  }
  messages.sort((left, right) => Number(left.created_at || 0) - Number(right.created_at || 0));
  renderTranscript(messages);

  if (queueList) {
    queueList.replaceChildren();
    const items = Array.isArray(state.queue) ? state.queue : [];
    if (!items.length) {
      const empty = document.createElement("li");
      empty.className = "room-queue-empty";
      empty.textContent = "当前无人排队";
      queueList.appendChild(empty);
    } else {
      for (const item of items.slice(0, 8)) {
        const li = document.createElement("li");
        li.className = item.id === state.me.id ? "is-me" : "";
        const pos = document.createElement("span");
        pos.className = "room-queue-position";
        pos.textContent = String(item.position).padStart(2, "0");
        const label = document.createElement("span");
        label.className = "room-queue-name";
        label.textContent = item.name;
        li.append(pos, label);
        queueList.appendChild(li);
      }
      if (items.length > 8) {
        const more = document.createElement("li");
        more.className = "room-queue-empty";
        more.textContent = `还有 ${items.length - 8} 位…`;
        queueList.appendChild(more);
      }
    }
  }
  window.dispatchEvent(new CustomEvent("live-room-state", { detail: state }));
}

chatToggle?.addEventListener("click", () => setChatExpanded(!chatExpanded));

viewersButton?.addEventListener("click", (event) => {
  event.stopPropagation();
  setViewersOpen(viewersPopover?.hidden ?? true);
});
viewersClose?.addEventListener("click", () => setViewersOpen(false));
viewersPopover?.addEventListener("click", (event) => event.stopPropagation());
document.addEventListener("click", () => setViewersOpen(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setViewersOpen(false);
    setMentionOpen(false);
  }
});

chatInput?.addEventListener("input", updateMentionMenu);
chatInput?.addEventListener("click", updateMentionMenu);
chatInput?.addEventListener("keydown", (event) => {
  if (mentionMenu?.hidden) return;
  if (["Enter", "Tab", "ArrowDown", "ArrowUp"].includes(event.key)) {
    event.preventDefault();
    insertXiaomaMention();
  }
});
chatInput?.addEventListener("blur", () => {
  window.setTimeout(() => setMentionOpen(false), 120);
});
mentionOption?.addEventListener("pointerdown", (event) => event.preventDefault());
mentionOption?.addEventListener("click", insertXiaomaMention);

function renderTranscript(messages) {
  if (!liveTranscript) return;
  const recent = messages.slice(-80);
  if (chatEmpty) chatEmpty.hidden = recent.length > 0;
  const distanceFromBottom = liveTranscript.scrollHeight - liveTranscript.scrollTop - liveTranscript.clientHeight;
  const shouldFollow = distanceFromBottom < 72;
  const retained = new Set(recent.map((item) => item.id));
  for (const [id, node] of transcriptNodes) {
    if (!retained.has(id)) {
      node.classList.add("leaving");
      window.setTimeout(() => node.remove(), 450);
      transcriptNodes.delete(id);
    }
  }
  for (const item of recent) {
    let node = transcriptNodes.get(item.id);
    if (!node) {
      node = document.createElement("article");
      const role = ["assistant", "user", "viewer"].includes(item.role) ? item.role : "viewer";
      node.className = `room-live-line ${role}`;
      const speaker = document.createElement("span");
      speaker.className = "room-live-speaker";
      const content = document.createElement("p");
      content.className = "room-live-content";
      const body = document.createElement("div");
      body.className = "room-live-body";
      const quote = document.createElement("div");
      quote.className = "room-live-quote";
      body.append(quote, content);
      node.append(speaker, body);
      liveTranscript.appendChild(node);
      transcriptNodes.set(item.id, node);
      requestAnimationFrame(() => node.classList.add("visible"));
    }
    node.querySelector(".room-live-speaker").textContent = item.speaker || (item.role === "user" ? "观众" : "小雅");
    const quote = node.querySelector(".room-live-quote");
    if (item.reply_to?.text) {
      quote.textContent = `回复 ${item.reply_to.speaker || "观众"}：${item.reply_to.text}`;
      quote.hidden = false;
    } else {
      quote.textContent = "";
      quote.hidden = true;
    }
    node.querySelector(".room-live-content").textContent = `${item.text || ""}${item.interrupted ? "  · 已打断" : ""}`;
    node.classList.toggle("partial", !!item.partial);
    node.classList.toggle("agent-status", item.kind === "agent_status" || !!item.agent_job_id);
    node.classList.toggle("is-me", !!current?.me?.id && item.participant_id === current.me.id);
    node.classList.toggle("voice", item.kind === "voice");
    if (item.agent_phase) node.dataset.agentPhase = item.agent_phase;
    else delete node.dataset.agentPhase;
    // Sorting an array does not move existing elements in the DOM. Re-appending
    // each retained node is cheap at this bounded size and guarantees the
    // visual order matches the canonical transcript order after every update.
    liveTranscript.appendChild(node);
  }
  for (const [index, item] of recent.entries()) {
    transcriptNodes.get(item.id)?.style.setProperty("--line-age", String(recent.length - index - 1));
  }
  if (shouldFollow) requestAnimationFrame(() => { liveTranscript.scrollTop = liveTranscript.scrollHeight; });
}

chatForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput?.value.trim() || "";
  if (!message || !chatInput || !chatSend) return;
  chatInput.disabled = true;
  chatSend.disabled = true;
  text(chatError, "");
  try {
    const response = await fetch("/api/room/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: message }),
      credentials: "same-origin",
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "发送失败");
    chatInput.value = "";
    setMentionOpen(false);
  } catch (error) {
    text(chatError, error instanceof Error ? error.message : String(error));
  } finally {
    chatInput.disabled = false;
    chatSend.disabled = false;
    chatInput.focus();
  }
});

async function joinRoom() {
  const response = await fetch("/api/room/join", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    credentials: "same-origin",
  });
  if (!response.ok) throw new Error(`进入直播间失败 (${response.status})`);
  render(await response.json());
}

function connectEvents() {
  if (source) source.close();
  source = new EventSource("/api/room/events");
  source.addEventListener("room", (event) => {
    try {
      render(JSON.parse(event.data || "{}"));
    } catch (error) {
      console.warn("[room] invalid state", error);
    }
  });
  source.onerror = () => {
    source?.close();
    source = null;
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(() => void boot(), 2000);
  };
}

async function boot() {
  try {
    await joinRoom();
    connectEvents();
  } catch (error) {
    console.warn("[room] join failed", error);
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(() => void boot(), 2500);
  }
}

nameButton?.addEventListener("click", () => {
  if (!nameDialog || !nameInput) return;
  nameInput.value = current?.me?.name || "";
  text(nameError, "");
  nameDialog.showModal();
  nameInput.select();
});

nameForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const submitter = /** @type {SubmitEvent} */ (event).submitter;
  if (submitter?.value === "cancel") {
    nameDialog?.close();
    return;
  }
  const name = nameInput?.value.trim() || "";
  try {
    const response = await fetch("/api/room/name", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || "修改名字失败");
    nameDialog?.close();
  } catch (error) {
    text(nameError, error instanceof Error ? error.message : String(error));
  }
});

window.addEventListener("pagehide", () => source?.close());
setChatExpanded(false);
void boot();
