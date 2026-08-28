/* ============================================================================
 * avatar-sync.js  —  AVTR-1 WebRTC/WHEP with HTTP-FLV fallback
 *
 * TTS PCM is routed to AVTR-1 on the server. The browser only plays the
 * muxed live stream and never sends generated audio back upstream.
 * ========================================================================== */
(function () {
  'use strict';

  const DEFAULTS = {
    gatewayBase: '/av',
    idleSrc: '/avatar/idle.mp4?v=pretty1',
    fit: 'right',
    rightOffset: '-4%',
    objectPosition: '78% 50%',
    maskFade: 0.42,
    showStatus: true,
  };

  const CFG = Object.assign({}, DEFAULTS, window.AVATAR_CONFIG || {});
  const MUSIC_STORAGE_KEY = 'avtr1.backgroundMusic.v1';
  const LEGACY_TRANSPORT_STORAGE_KEY = 'avtr1.transport.v1';
  const VIEW_DEFAULTS = {
    size: 100,
    position: Math.round(-parseFloat(CFG.rightOffset || '-4')),
    vertical: 0,
    fade: Math.round(Number(CFG.maskFade || 0) * 100),
  };

  function clamp(value, min, max, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
  }

  function loadAvatarView() {
    return { ...VIEW_DEFAULTS };
  }

  let avatarView = loadAvatarView();
  let profilePreviewLocked = false;
  let musicEnabled = localStorage.getItem(MUSIC_STORAGE_KEY) !== '0';
  let transportPreference = '';
  let activeTransport = '';
  window.AVATAR_MUTE_TTS = true;
  const gw = () => CFG.gatewayBase.replace(/\/+$/, '');

  function requestedTransport(config = streamConfig) {
    if (transportPreference) return transportPreference;
    return config?.transport === 'webrtc' ? 'webrtc' : 'http-flv';
  }

  function renderTransportControls(detail = '') {
    const requested = requestedTransport();
    document.querySelectorAll('[data-avatar-transport]').forEach((button) => {
      const selected = button.dataset.avatarTransport === requested;
      button.setAttribute('aria-checked', selected ? 'true' : 'false');
      button.classList.toggle('active', selected);
    });
    const status = document.getElementById('avatar-transport-status');
    if (!status) return;
    if (detail) {
      status.textContent = detail;
    } else if (activeTransport) {
      status.textContent = activeTransport === 'webrtc'
        ? '当前正在使用 WebRTC'
        : '当前正在使用 HTTP-FLV';
    } else {
      status.textContent = requested === 'webrtc'
        ? '将优先使用 WebRTC，协商失败时临时回退 HTTP-FLV。'
        : '固定使用 HTTP-FLV，不会自动切换到 WebRTC。';
    }
  }

  function applyGlobalTransport(value, revision, reconnect = true) {
    if (value !== 'webrtc' && value !== 'http-flv') return false;
    const changed = transportPreference !== value;
    transportPreference = value;
    streamConfig = {
      ...(streamConfig || {}),
      transport: value,
      transport_revision: Number(revision || streamConfig?.transport_revision || 0),
    };
    try { localStorage.removeItem(LEGACY_TRANSPORT_STORAGE_KEY); } catch (_) { /* ignore */ }
    if (!changed) {
      renderTransportControls();
      return false;
    }
    activeTransport = '';
    webrtcRetryAfter = 0;
    webrtcFallbackReason = '';
    renderTransportControls(changed ? '正在切换全局播放模式…' : '');
    if (changed && reconnect) reconnectLive();
    return changed;
  }

  async function setTransportPreference(value) {
    if (value !== 'webrtc' && value !== 'http-flv') return;
    renderTransportControls('正在保存全局播放模式…');
    try {
      const response = await fetch('/api/admin/avatar-transport', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ transport: value }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.ok) throw new Error(data.detail || '播放模式保存失败');
      applyGlobalTransport(data.transport, data.transport_revision, true);
    } catch (error) {
      renderTransportControls(error instanceof Error ? error.message : String(error));
    }
  }

  function bindTransportControls() {
    document.querySelectorAll('[data-avatar-transport]').forEach((button) => {
      button.addEventListener('click', () => {
        void setTransportPreference(button.dataset.avatarTransport || '');
      });
    });
    renderTransportControls();
  }

  const maskGradient = `linear-gradient(90deg, rgba(0,0,0,0) 0%,
    rgba(0,0,0,.30) var(--av-fade-soft),
    rgba(0,0,0,.85) var(--av-fade-mid),
    #000 var(--av-fade-end))`;
  const maskCss = `-webkit-mask-image:${maskGradient}; mask-image:${maskGradient};
    -webkit-mask-size:100% 100%; mask-size:100% 100%;`;
  const fitCss = CFG.fit === 'cover'
    ? `inset:0; width:100%; height:100%;
       object-fit:cover; object-position:${CFG.objectPosition};`
    : `top:calc(50% + var(--av-vertical)); right:var(--av-right); height:var(--av-height); width:auto;
       max-width:none; object-fit:cover; transform:translateY(-50%);`;

  const style = document.createElement('style');
  style.textContent = `
    html, body { background:#000 !important; }
    #av-layer {
      position:fixed; inset:0; z-index:0;
      background:#000; overflow:hidden; pointer-events:none;
      --av-height:100%; --av-right:${CFG.rightOffset}; --av-vertical:0%;
      --av-fade-soft:18.9%; --av-fade-mid:32.8%; --av-fade-end:42%;
    }
    #av-idle, #av-live-video {
      position:absolute; ${fitCss} ${maskCss} display:block;
      transition:opacity .35s ease;
    }
    #av-idle { opacity:1; }
    #av-live-video { opacity:0; }
    #av-layer.live #av-idle { opacity:0; }
    #av-layer.live #av-live-video { opacity:1; }
    #av-looks {
      position:fixed; right:max(12px, env(safe-area-inset-right));
      bottom:max(18px, env(safe-area-inset-bottom)); z-index:9998;
      display:flex; gap:8px; pointer-events:auto;
    }
    #av-looks button {
      width:58px; padding:0; border:2px solid rgba(255,255,255,.25);
      border-radius:10px; overflow:hidden; cursor:pointer; background:#111;
    }
    #av-looks button[aria-pressed="true"] { border-color:#fff; }
    #av-looks img { display:block; width:58px; height:78px; object-fit:cover; }
    #av-layer.av-no-mask #av-idle,
    #av-layer.av-no-mask #av-live-video {
      -webkit-mask-image:none; mask-image:none;
    }
    /* Keep the app above the video without rewriting every body child's
       positioning. The old broad selector changed fixed overlays (notably the
       room chat) to position:relative and pushed them below the clipped page. */
    #app { position:relative; z-index:1; }
    #av-status {
      position:fixed; left:max(14px, env(safe-area-inset-left));
      top:max(16px, env(safe-area-inset-top)); bottom:auto; z-index:9999;
      font:11px/1.6 ui-monospace, monospace; color:rgba(245,246,250,.72);
      background:rgba(10,11,16,.68); border:1px solid rgba(255,255,255,.10);
      border-radius:999px; padding:4px 10px; pointer-events:none;
      backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
    }
  `;
  (document.head || document.documentElement).appendChild(style);

  let layer, idle, liveVideo, statusEl;
  let flvPlayer = null;
  let peerConnection = null;
  let whepSessionUrl = '';
  let webrtcStatsHandle = null;
  let streamConfig = null;
  let webrtcRetryAfter = 0;
  let audioUnlocked = false;
  let connecting = false;
  let reconnectRequested = false;
  let retryHandle = null;
  let webrtcUpgradeHandle = null;
  let watchdogHandle = null;
  let lastProgressAt = 0;
  let lastMediaTime = -1;
  let webrtcFallbackReason = '';

  function getPeerConnectionConstructor() {
    // Current Chromium exposes RTCPeerConnection. The prefixed names keep the
    // viewer usable in older Chromium/WebView shells still found on TVs and
    // embedded Android browsers.
    return window.RTCPeerConnection
      || window.webkitRTCPeerConnection
      || window.mozRTCPeerConnection
      || null;
  }

  function describeWebRTCEnvironment() {
    const diagnostics = {
      supported: Boolean(getPeerConnectionConstructor()),
      secureContext: Boolean(window.isSecureContext),
      protocol: window.location.protocol,
      topLevel: window.top === window.self,
    };
    window.AVATAR_WEBRTC_DIAGNOSTICS = diagnostics;
    return diagnostics;
  }

  function syncAudioRoute() {
    if (liveVideo) liveVideo.muted = !audioUnlocked;
    // Audio and video must share the FLV muxer's timestamp. Never play the
    // lower-latency WebSocket copy in parallel: it leads the rendered mouth
    // and creates an audible echo once the FLV audio catches up.
    window.AVATAR_MUTE_TTS = true;
  }

  function setStatus(s) {
    if (!statusEl) return;
    let label = s;
    let state = 'connecting';
    if (s === 'idle') {
      label = '未连接';
      state = 'idle';
    } else if (s === 'speaking') {
      label = '说话中';
      state = 'speaking';
    } else if (s.includes('弱网')) {
      label = '弱网保护中';
      state = 'connected';
    } else if (s.includes('WebRTC') || s.includes('FLV') || s === 'connected') {
      label = s.includes('FLV') ? '已连接 · 兼容模式' : '已连接';
      state = 'connected';
    } else if (s.includes('失败')) {
      label = '连接失败';
      state = 'error';
    } else if (s.includes('connecting') || s.includes('重连')) {
      label = '连接中';
    }
    statusEl.textContent = '数字人 ' + label;
    statusEl.dataset.state = state;
  }

  function positionLabel(value) {
    if (!value) return '居中';
    return `${Math.abs(value)}% 向${value > 0 ? '右' : '左'}`;
  }

  function verticalLabel(value) {
    if (!value) return '居中';
    return `${Math.abs(value)}% 向${value > 0 ? '下' : '上'}`;
  }

  function applyAvatarView(persist = false) {
    if (!layer) return;
    const fade = avatarView.fade;
    layer.style.setProperty('--av-height', `${avatarView.size}%`);
    layer.style.setProperty('--av-right', `${-avatarView.position}%`);
    layer.style.setProperty('--av-vertical', `${avatarView.vertical}%`);
    layer.style.setProperty('--av-fade-soft', `${(fade * 0.45).toFixed(1)}%`);
    layer.style.setProperty('--av-fade-mid', `${(fade * 0.78).toFixed(1)}%`);
    layer.style.setProperty('--av-fade-end', `${fade}%`);
    layer.classList.toggle('av-no-mask', fade === 0);

    const sizeValue = document.getElementById('avatar-size-value');
    const positionValue = document.getElementById('avatar-position-value');
    const verticalValue = document.getElementById('avatar-vertical-value');
    const fadeValue = document.getElementById('avatar-fade-value');
    if (sizeValue) sizeValue.textContent = `${avatarView.size}%`;
    if (positionValue) positionValue.textContent = positionLabel(avatarView.position);
    if (verticalValue) verticalValue.textContent = verticalLabel(avatarView.vertical);
    if (fadeValue) fadeValue.textContent = fade ? `${fade}%` : '关闭';
  }

  function bindAvatarControls() {
    const size = document.getElementById('avatar-size');
    const position = document.getElementById('avatar-position');
    const vertical = document.getElementById('avatar-vertical');
    const fade = document.getElementById('avatar-fade');
    const reset = document.getElementById('avatar-view-reset');
    if (!size || !position || !vertical || !fade) return;
    size.value = String(avatarView.size);
    position.value = String(avatarView.position);
    vertical.value = String(avatarView.vertical);
    fade.value = String(avatarView.fade);
    const update = () => {
      avatarView = {
        size: clamp(size.value, 70, 135, VIEW_DEFAULTS.size),
        position: clamp(position.value, -20, 20, VIEW_DEFAULTS.position),
        vertical: clamp(vertical.value, -20, 20, VIEW_DEFAULTS.vertical),
        fade: clamp(fade.value, 0, 70, VIEW_DEFAULTS.fade),
      };
      applyAvatarView(false);
    };
    size.addEventListener('input', update);
    position.addEventListener('input', update);
    vertical.addEventListener('input', update);
    fade.addEventListener('input', update);
    reset?.addEventListener('click', () => {
      avatarView = { ...VIEW_DEFAULTS };
      size.value = String(avatarView.size);
      position.value = String(avatarView.position);
      vertical.value = String(avatarView.vertical);
      fade.value = String(avatarView.fade);
      applyAvatarView(false);
    });
    applyAvatarView();
  }

  function quietMpegts() {
    const control = window.mpegts?.LoggingControl;
    if (!control) return;
    try {
      control.applyConfig({ enableCallback: false });
    } catch (_) { /* ignore */ }
    ['enableVerbose', 'enableInfo', 'enableDebug', 'enableWarn'].forEach((key) => {
      try { control[key] = false; } catch (_) { /* ignore */ }
    });
  }

  function destroyPlayer() {
    if (webrtcUpgradeHandle) {
      clearTimeout(webrtcUpgradeHandle);
      webrtcUpgradeHandle = null;
    }
    if (webrtcStatsHandle) {
      clearInterval(webrtcStatsHandle);
      webrtcStatsHandle = null;
    }
    const pc = peerConnection;
    peerConnection = null;
    if (pc) {
      try { pc.ontrack = null; pc.onconnectionstatechange = null; pc.close(); } catch (_) { /* ignore */ }
    }
    const sessionUrl = whepSessionUrl;
    whepSessionUrl = '';
    if (sessionUrl) {
      fetch(sessionUrl, { method: 'DELETE', keepalive: true }).catch(() => {});
    }
    const player = flvPlayer;
    flvPlayer = null;
    if (player) {
      try { player.pause(); } catch (_) { /* ignore */ }
      try { player.unload(); } catch (_) { /* ignore */ }
      try { player.detachMediaElement(); } catch (_) { /* ignore */ }
      try { player.destroy(); } catch (_) { /* ignore */ }
    }
    if (liveVideo) {
      try { liveVideo.pause(); } catch (_) { /* ignore */ }
      liveVideo.removeAttribute('src');
      liveVideo.srcObject = null;
      try { liveVideo.load(); } catch (_) { /* ignore */ }
    }
  }

  function scheduleWebRTCUpgrade() {
    if (
      requestedTransport() !== 'webrtc'
      || webrtcUpgradeHandle
      || !flvPlayer
      || document.hidden
    ) return;
    const delay = Math.max(5000, webrtcRetryAfter - Date.now());
    webrtcUpgradeHandle = setTimeout(async () => {
      webrtcUpgradeHandle = null;
      if (!flvPlayer || connecting || document.hidden) return;
      try {
        const response = await fetch('/api/avatar-stream-health', { cache: 'no-store' });
        const health = response.ok ? await response.json() : {};
        const selectedReady = musicEnabled ? health?.paths?.music : health?.paths?.voice;
        if (selectedReady) {
          webrtcRetryAfter = 0;
          reconnectLive();
          return;
        }
      } catch (_) { /* keep the working FLV stream and probe again later */ }
      scheduleWebRTCUpgrade();
    }, delay);
  }

  function scheduleReconnect(delayMs = 1200) {
    if (retryHandle || document.hidden) return;
    setStatus('重连中');
    syncAudioRoute();
    layer?.classList.remove('live');
    retryHandle = setTimeout(() => {
      retryHandle = null;
      void connect();
    }, delayMs);
  }

  function markProgress() {
    const currentTime = Number(liveVideo?.currentTime || 0);
    if (currentTime > lastMediaTime + 0.02) {
      lastMediaTime = currentTime;
      lastProgressAt = Date.now();
    }
  }

  function startWatchdog() {
    if (watchdogHandle || !liveVideo) return;
    liveVideo.addEventListener('timeupdate', markProgress);
    liveVideo.addEventListener('playing', markProgress);
    liveVideo.addEventListener('error', () => scheduleReconnect(400));
    watchdogHandle = setInterval(() => {
      if (document.hidden || connecting || retryHandle) return;
      markProgress();
      if (Date.now() - lastProgressAt >= 4000) scheduleReconnect(400);
    }, 1500);
  }

  async function avatarStreamConfig(force = false) {
    if (streamConfig && !force) return streamConfig;
    const response = await fetch('/api/avatar-config', { cache: 'no-store' });
    if (!response.ok) throw new Error(`直播配置不可用 (${response.status})`);
    streamConfig = await response.json();
    transportPreference = streamConfig.transport === 'webrtc' ? 'webrtc' : 'http-flv';
    try { localStorage.removeItem(LEGACY_TRANSPORT_STORAGE_KEY); } catch (_) { /* ignore */ }
    return streamConfig;
  }

  async function refreshGlobalTransport() {
    const previousTransport = transportPreference;
    const previousRevision = streamConfig?.transport_revision;
    try {
      const config = await avatarStreamConfig(true);
      const changed = previousRevision !== undefined && (
        previousTransport !== config.transport
        || Number(previousRevision) !== Number(config.transport_revision)
      );
      renderTransportControls();
      if (changed && previousTransport !== config.transport) {
        activeTransport = '';
        webrtcRetryAfter = 0;
        webrtcFallbackReason = '';
        reconnectLive();
      }
    } catch (_) { /* retain the last server-owned value */ }
  }

  function waitForIceGathering(pc, timeoutMs = 2500) {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise((resolve) => {
      const timeout = setTimeout(done, timeoutMs);
      function done() {
        clearTimeout(timeout);
        pc.removeEventListener('icegatheringstatechange', changed);
        resolve();
      }
      function changed() {
        if (pc.iceGatheringState === 'complete') done();
      }
      pc.addEventListener('icegatheringstatechange', changed);
    });
  }

  function waitForPlayback(timeoutMs = 8000) {
    return new Promise((resolve, reject) => {
      let timeout = null;
      const done = () => {
        if (timeout) clearTimeout(timeout);
        liveVideo.removeEventListener('playing', done);
        resolve();
      };
      liveVideo.addEventListener('playing', done, { once: true });
      timeout = setTimeout(() => {
        liveVideo.removeEventListener('playing', done);
        reject(new Error('直播首帧超时'));
      }, timeoutMs);
      liveVideo.play().catch(() => {
        // A later user gesture will unlock audio; muted video normally starts
        // immediately, so keep waiting for the real playing event here.
      });
      if (!liveVideo.paused && liveVideo.readyState >= 2) done();
    });
  }

  function startWebRTCStats(pc) {
    let previous = null;
    if (webrtcStatsHandle) clearInterval(webrtcStatsHandle);
    webrtcStatsHandle = setInterval(async () => {
      if (peerConnection !== pc || pc.connectionState === 'closed') return;
      try {
        const reports = await pc.getStats();
        const current = {
          received: 0, lost: 0, bytes: 0, jitter: 0, rtt: 0,
          decodedFrames: 0, droppedFrames: 0, freezes: 0,
          concealedSamples: 0, at: Date.now(), transport: 'webrtc',
        };
        reports.forEach((report) => {
          if (report.type === 'inbound-rtp' && !report.isRemote) {
            current.received += Number(report.packetsReceived || 0);
            current.lost += Math.max(0, Number(report.packetsLost || 0));
            current.bytes += Number(report.bytesReceived || 0);
            current.jitter = Math.max(current.jitter, Number(report.jitter || 0));
            current.decodedFrames += Number(report.framesDecoded || 0);
            current.droppedFrames += Number(report.framesDropped || 0);
            current.freezes += Number(report.freezeCount || 0);
            current.concealedSamples += Number(report.concealedSamples || 0);
          } else if (report.type === 'candidate-pair' && report.state === 'succeeded' && report.nominated) {
            current.rtt = Number(report.currentRoundTripTime || 0);
          }
        });
        const packetDelta = previous
          ? Math.max(0, current.received - previous.received) + Math.max(0, current.lost - previous.lost)
          : 0;
        current.lossRate = packetDelta
          ? Math.max(0, current.lost - previous.lost) / packetDelta
          : 0;
        current.bitrate = previous
          ? Math.max(0, current.bytes - previous.bytes) * 8 * 1000 / Math.max(1, current.at - previous.at)
          : 0;
        previous = current;
        window.AVATAR_STREAM_STATS = current;
        if (current.lossRate >= 0.08 || current.jitter >= 0.12) {
          setStatus('AVTR-1 WebRTC · 弱网');
        } else {
          setStatus('AVTR-1 WebRTC');
        }
      } catch (_) { /* stats are diagnostic only */ }
    }, 2000);
  }

  async function connectWebRTC(config) {
    const PeerConnection = getPeerConnectionConstructor();
    if (!PeerConnection) {
      const diagnostics = describeWebRTCEnvironment();
      const reason = diagnostics.secureContext
        ? 'WebRTC 被浏览器设置、扩展或管理策略禁用'
        : '当前页面未被浏览器视为安全连接，WebRTC 不可用';
      throw new Error(reason);
    }
    if (!config?.whep) throw new Error('WHEP 地址未配置');
    const endpoint = musicEnabled ? config.whep.music : config.whep.voice;
    if (!endpoint) throw new Error('WHEP 地址未配置');
    const pc = new PeerConnection({ bundlePolicy: 'max-bundle' });
    peerConnection = pc;
    const media = new MediaStream();
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    pc.ontrack = (event) => {
      if (peerConnection !== pc) return;
      media.addTrack(event.track);
      liveVideo.srcObject = media;
    };
    pc.onconnectionstatechange = () => {
      if (peerConnection !== pc) return;
      if (pc.connectionState === 'failed') {
        webrtcRetryAfter = Date.now() + 30000;
        scheduleReconnect(200);
      } else if (pc.connectionState === 'disconnected') {
        setTimeout(() => {
          if (peerConnection === pc && pc.connectionState === 'disconnected') scheduleReconnect(200);
        }, 2500);
      }
    };
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIceGathering(pc);
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/sdp', Accept: 'application/sdp' },
      body: pc.localDescription.sdp,
    });
    if (response.status !== 201) {
      throw new Error(`WHEP 协商失败 (${response.status})`);
    }
    const location = response.headers.get('Location');
    if (location) whepSessionUrl = new URL(location, new URL(endpoint, window.location.href)).href;
    await pc.setRemoteDescription({ type: 'answer', sdp: await response.text() });
    liveVideo.muted = true;
    await waitForPlayback(6000);
    startWebRTCStats(pc);
    return 'AVTR-1 WebRTC';
  }

  async function connectFLV() {
    const mpegts = window.mpegts;
    if (!mpegts?.isSupported?.() || !mpegts.createPlayer) throw new Error('当前浏览器不支持直播解码');
    quietMpegts();
    const player = mpegts.createPlayer({
      type: 'flv', isLive: true,
      url: `${gw()}/livestream.flv?music=${musicEnabled ? '1' : '0'}&t=${Date.now()}`,
    }, {
      enableStashBuffer: true,
      stashInitialSize: 128 * 1024,
      liveBufferLatencyChasing: true,
      liveBufferLatencyMaxLatency: 2.4,
      liveBufferLatencyMinRemain: 0.8,
      lazyLoad: false,
      autoCleanupSourceBuffer: true,
      autoCleanupMaxBackwardDuration: 4,
      autoCleanupMinBackwardDuration: 2,
    });
    flvPlayer = player;
    liveVideo.muted = true;
    player.attachMediaElement(liveVideo);
    player.load();
    const statsEvent = mpegts.Events?.STATISTICS_INFO;
    if (statsEvent) player.on(statsEvent, (info) => {
      window.AVATAR_STREAM_STATS = {
        transport: 'http-flv', speed: Number(info?.speed || 0),
        decodedFrames: Number(info?.decodedFrames || 0),
        droppedFrames: Number(info?.droppedFrames || 0), at: Date.now(),
      };
    });
    await waitForPlayback(8000);
    return 'AVTR-1 HTTP-FLV';
  }

  async function connect() {
    if (connecting) return;
    connecting = true;
    if (retryHandle) { clearTimeout(retryHandle); retryHandle = null; }
    setStatus('connecting…');
    destroyPlayer();
    try {
      const config = await avatarStreamConfig();
      const requested = requestedTransport(config);
      renderTransportControls();
      let label = '';
      if (requested === 'webrtc' && Date.now() >= webrtcRetryAfter) {
        try {
          label = await connectWebRTC(config);
        } catch (err) {
          console.warn('[avatar] WebRTC unavailable, falling back to FLV:', err);
          webrtcFallbackReason = err?.message || String(err);
          webrtcRetryAfter = Date.now() + 30000;
          destroyPlayer();
        }
      }
      if (!label) label = await connectFLV();
      activeTransport = label.includes('WebRTC') ? 'webrtc' : 'http-flv';
      lastMediaTime = -1;
      lastProgressAt = Date.now();
      layer.classList.add('live');
      syncAudioRoute();
      setStatus(label);
      if (statusEl && label.includes('FLV') && webrtcFallbackReason) {
        statusEl.title = webrtcFallbackReason;
        statusEl.setAttribute('aria-label', `数字人已使用兼容模式：${webrtcFallbackReason}`);
      } else if (statusEl) {
        statusEl.title = '';
        statusEl.removeAttribute('aria-label');
      }
      if (label.includes('FLV') && requested === 'webrtc') {
        renderTransportControls(`WebRTC 暂不可用，当前临时使用 HTTP-FLV：${webrtcFallbackReason}`);
        scheduleWebRTCUpgrade();
      } else {
        renderTransportControls();
      }
      startWatchdog();
    } catch (err) {
      destroyPlayer();
      syncAudioRoute();
      layer.classList.remove('live');
      setStatus('连接失败: ' + (err?.message || err));
      scheduleReconnect(3000);
    } finally {
      connecting = false;
      if (reconnectRequested) {
        reconnectRequested = false;
        reconnectLive();
      }
    }
  }

  const AVATAR_STORAGE_KEY = 's2s.avatar.id';

  function currentAvatarId() {
    try {
      return localStorage.getItem(AVATAR_STORAGE_KEY) || 'xiaoya';
    } catch (_) {
      return 'xiaoya';
    }
  }

  function stillUrl(id) {
    return `/avatar/avatars/${encodeURIComponent(id)}.jpg?v=look2`;
  }

  function setStill(id) {
    if (idle) idle.src = stillUrl(id);
  }

  function applyProfile(profile, reconnect = false) {
    if (!profile || !profile.avatar_id) return;
    const view = profile.view || {};
    avatarView = {
      size: clamp(view.size, 70, 135, VIEW_DEFAULTS.size),
      position: clamp(view.position, -20, 20, VIEW_DEFAULTS.position),
      vertical: clamp(view.vertical, -20, 20, VIEW_DEFAULTS.vertical),
      fade: clamp(view.fade, 0, 70, VIEW_DEFAULTS.fade),
    };
    applyAvatarView();
    setStill(profile.avatar_id);
    markLookPressed(profile.avatar_id);
    try { localStorage.setItem(AVATAR_STORAGE_KEY, profile.avatar_id); } catch (_) { /* cache only */ }
    if (reconnect) reconnectLive();
  }

  async function refreshActiveProfile(reconnect = false) {
    try {
      const response = await fetch('/api/avatar-profile/active', { cache: 'no-store' });
      if (!response.ok) return;
      const profile = await response.json();
      const prior = window.AVTR_ACTIVE_PROFILE?.state_revision;
      window.AVTR_ACTIVE_PROFILE = profile;
      if (!profilePreviewLocked) {
        applyProfile(profile, reconnect && prior !== undefined && prior !== profile.state_revision);
      }
      window.dispatchEvent(new CustomEvent('avtr1-profile-updated', { detail: profile }));
    } catch (_) { /* retain last rendered profile */ }
  }

  window.AVATAR_PROFILE_UI = {
    preview(view) { avatarView = { ...avatarView, ...(view || {}) }; applyAvatarView(); },
    currentView() { return { ...avatarView }; },
    apply: applyProfile,
    refresh: refreshActiveProfile,
    lockPreview(locked) {
      profilePreviewLocked = Boolean(locked);
      if (!profilePreviewLocked) void refreshActiveProfile(false);
    },
    reconnect: reconnectLive,
  };

  window.AVATAR_STREAM_UI = {
    current() {
      return { requested: requestedTransport(), active: activeTransport };
    },
    setTransport: setTransportPreference,
  };

  async function switchAvatar(avatarId) {
    try {
      const response = await fetch('/api/admin/avatar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ avatar_id: avatarId }),
      });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.detail || data.error || 'switch failed');
      setStill(avatarId);
      try {
        localStorage.setItem(AVATAR_STORAGE_KEY, avatarId);
      } catch (_) { /* ignore */ }
      markLookPressed(avatarId);
      layer?.classList.remove('live');
      reconnectLive();
    } catch (err) {
      console.warn('[avatar] switch failed:', err);
    }
  }

  function markLookPressed(avatarId) {
    document.querySelectorAll('#av-looks button, #avatar-picker .avatar-pick').forEach((button) => {
      button.setAttribute('aria-pressed', button.dataset.id === avatarId ? 'true' : 'false');
    });
  }

  async function buildLooks() {
    let avatars = [
      { id: 'xiaoya_locket', label: '白背心' },
      { id: 'xiaoya', label: '小雅' },
      { id: 'xiaoya_idle', label: '暖光正脸' },
      { id: 'xiaoya_beach_close', label: '海边近景' },
      { id: 'xiaoya_beach', label: '海边' },
      { id: 'sauna_portrait', label: '桑拿正脸' },
    ];
    let selected = currentAvatarId();
    try {
      const response = await fetch(gw() + '/avatars', { cache: 'no-store' });
      const data = await response.json();
      if (Array.isArray(data.avatars) && data.avatars.length) avatars = data.avatars;
      if (avatars.some((item) => item.id === data.avatar_id)) selected = data.avatar_id;
    } catch (_) { /* keep defaults */ }
    // The gateway owns this room-wide setting. Local storage is only a cache
    // for the fallback still shown before the public status request finishes.
    try { localStorage.setItem(AVATAR_STORAGE_KEY, selected); } catch (_) { /* ignore */ }
    const strip = document.createElement('div');
    strip.id = 'av-looks';
    for (const item of avatars) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.id = item.id;
      button.title = item.label;
      button.setAttribute('aria-pressed', item.id === selected ? 'true' : 'false');
      button.innerHTML = `<img src="${stillUrl(item.id)}" alt="${item.label}">`;
      button.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        void switchAvatar(item.id);
      });
      strip.appendChild(button);
    }
    document.body.appendChild(strip);
    setStill(selected);
  }

  function build() {
    layer = document.createElement('div');
    layer.id = 'av-layer';

    idle = document.createElement('img');
    idle.id = 'av-idle';
    idle.alt = '';
    idle.draggable = false;
    idle.src = stillUrl(currentAvatarId());
    layer.appendChild(idle);

    liveVideo = document.createElement('video');
    liveVideo.id = 'av-live-video';
    liveVideo.autoplay = true;
    // Muted autoplay gets the picture moving immediately. The first user
    // gesture unlocks the muxed FLV audio for the rest of the page lifetime.
    liveVideo.muted = true;
    liveVideo.preload = 'auto';
    liveVideo.playsInline = true;
    liveVideo.setAttribute('playsinline', '');
    layer.appendChild(liveVideo);

    document.body.insertBefore(layer, document.body.firstChild);
    applyAvatarView();
    bindAvatarControls();
    void refreshActiveProfile(false);

    statusEl = document.getElementById('avatar-connection-state');
    if (!statusEl && CFG.showStatus) {
      statusEl = document.createElement('div');
      statusEl.id = 'av-status';
      document.body.appendChild(statusEl);
    }
    setStatus('idle');
    bindMusicToggle();
    bindTransportControls();
    document.addEventListener('pointerdown', () => {
      audioUnlocked = true;
      syncAudioRoute();
      liveVideo?.play().then(() => {
        syncAudioRoute();
      }).catch(() => {});
    }, { once: true, capture: true });
  }

  function bindMusicToggle() {
    const button = document.getElementById('background-music-toggle');
    if (!button) return;
    const render = () => {
      button.setAttribute('aria-pressed', musicEnabled ? 'true' : 'false');
      button.setAttribute('aria-label', musicEnabled ? '关闭背景音乐' : '开启背景音乐');
      button.title = musicEnabled ? '关闭我的背景音乐' : '开启我的背景音乐';
      button.classList.toggle('is-muted', !musicEnabled);
    };
    render();
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      musicEnabled = !musicEnabled;
      try { localStorage.setItem(MUSIC_STORAGE_KEY, musicEnabled ? '1' : '0'); } catch (_) { /* ignore */ }
      render();
      reconnectLive();
    });
  }

  function reconnectLive() {
    if (retryHandle) {
      clearTimeout(retryHandle);
      retryHandle = null;
    }
    if (connecting) {
      // Serialize mode/profile/music changes with the current negotiation.
      // Starting a second WHEP/MSE player before the first promise settles can
      // let its late completion overwrite the newly selected transport.
      reconnectRequested = true;
      destroyPlayer();
      layer?.classList.remove('live');
      return;
    }
    destroyPlayer();
    layer?.classList.remove('live');
    syncAudioRoute();
    void connect();
  }

  function boot() {
    build();
    window.addEventListener('avtr1-avatar-changed', (event) => {
      const avatarId = event.detail?.avatarId || currentAvatarId();
      setStill(avatarId);
      markLookPressed(avatarId);
      reconnectLive();
    });
    window.addEventListener('avatar-manual-interrupt', () => {
      // Recreate MSE playback to discard audio already buffered in the browser;
      // the server endpoint has simultaneously cleared the renderer queue.
      reconnectLive();
    });
    if (window.EventSource) {
      const events = new EventSource('/api/avatar-profile/events');
      events.addEventListener('profile', () => void refreshActiveProfile(true));
      events.addEventListener('transport', (event) => {
        try {
          const data = JSON.parse(event.data || '{}');
          applyGlobalTransport(data.transport, data.revision, true);
        } catch (_) { /* periodic refresh remains the fallback */ }
      });
    }
    setInterval(() => {
      void refreshActiveProfile(true);
      void refreshGlobalTransport();
    }, 2000);
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && flvPlayer && requestedTransport() === 'webrtc') {
        scheduleWebRTCUpgrade();
      }
    });
    void connect();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
})();
