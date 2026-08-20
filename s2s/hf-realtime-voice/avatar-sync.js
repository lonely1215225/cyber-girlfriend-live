/* ============================================================================
 * avatar-sync.js  —  AVTR-1 HTTP-FLV into a native <video> element
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
  const VIEW_STORAGE_KEY = 'avtr1.avatarView.v1';
  const MUSIC_STORAGE_KEY = 'avtr1.backgroundMusic.v1';
  const VIEW_DEFAULTS = {
    size: 100,
    position: Math.round(-parseFloat(CFG.rightOffset || '-4')),
    fade: Math.round(Number(CFG.maskFade || 0) * 100),
  };

  function clamp(value, min, max, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
  }

  function loadAvatarView() {
    try {
      const saved = JSON.parse(localStorage.getItem(VIEW_STORAGE_KEY) || '{}');
      return {
        size: clamp(saved.size, 70, 135, VIEW_DEFAULTS.size),
        position: clamp(saved.position, -20, 20, VIEW_DEFAULTS.position),
        fade: clamp(saved.fade, 0, 70, VIEW_DEFAULTS.fade),
      };
    } catch (_) {
      return { ...VIEW_DEFAULTS };
    }
  }

  let avatarView = loadAvatarView();
  let musicEnabled = localStorage.getItem(MUSIC_STORAGE_KEY) !== '0';
  window.AVATAR_MUTE_TTS = true;
  const gw = () => CFG.gatewayBase.replace(/\/+$/, '');

  const maskGradient = `linear-gradient(90deg, rgba(0,0,0,0) 0%,
    rgba(0,0,0,.30) var(--av-fade-soft),
    rgba(0,0,0,.85) var(--av-fade-mid),
    #000 var(--av-fade-end))`;
  const maskCss = `-webkit-mask-image:${maskGradient}; mask-image:${maskGradient};
    -webkit-mask-size:100% 100%; mask-size:100% 100%;`;
  const fitCss = CFG.fit === 'cover'
    ? `inset:0; width:100%; height:100%;
       object-fit:cover; object-position:${CFG.objectPosition};`
    : `top:50%; right:var(--av-right); height:var(--av-height); width:auto;
       max-width:none; object-fit:cover; transform:translateY(-50%);`;

  const style = document.createElement('style');
  style.textContent = `
    html, body { background:#000 !important; }
    #av-layer {
      position:fixed; inset:0; z-index:0;
      background:#000; overflow:hidden; pointer-events:none;
      --av-height:100%; --av-right:${CFG.rightOffset};
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
  let audioUnlocked = false;
  let connecting = false;
  let retryHandle = null;
  let watchdogHandle = null;
  let lastProgressAt = 0;
  let lastMediaTime = -1;

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
    } else if (s.includes('FLV') || s === 'connected') {
      label = '已连接';
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

  function applyAvatarView(persist = false) {
    if (!layer) return;
    const fade = avatarView.fade;
    layer.style.setProperty('--av-height', `${avatarView.size}%`);
    layer.style.setProperty('--av-right', `${-avatarView.position}%`);
    layer.style.setProperty('--av-fade-soft', `${(fade * 0.45).toFixed(1)}%`);
    layer.style.setProperty('--av-fade-mid', `${(fade * 0.78).toFixed(1)}%`);
    layer.style.setProperty('--av-fade-end', `${fade}%`);
    layer.classList.toggle('av-no-mask', fade === 0);
    if (persist) localStorage.setItem(VIEW_STORAGE_KEY, JSON.stringify(avatarView));

    const sizeValue = document.getElementById('avatar-size-value');
    const positionValue = document.getElementById('avatar-position-value');
    const fadeValue = document.getElementById('avatar-fade-value');
    if (sizeValue) sizeValue.textContent = `${avatarView.size}%`;
    if (positionValue) positionValue.textContent = positionLabel(avatarView.position);
    if (fadeValue) fadeValue.textContent = fade ? `${fade}%` : '关闭';
  }

  function bindAvatarControls() {
    const size = document.getElementById('avatar-size');
    const position = document.getElementById('avatar-position');
    const fade = document.getElementById('avatar-fade');
    const reset = document.getElementById('avatar-view-reset');
    if (!size || !position || !fade) return;
    size.value = String(avatarView.size);
    position.value = String(avatarView.position);
    fade.value = String(avatarView.fade);
    const update = () => {
      avatarView = {
        size: clamp(size.value, 70, 135, VIEW_DEFAULTS.size),
        position: clamp(position.value, -20, 20, VIEW_DEFAULTS.position),
        fade: clamp(fade.value, 0, 70, VIEW_DEFAULTS.fade),
      };
      applyAvatarView(true);
    };
    size.addEventListener('input', update);
    position.addEventListener('input', update);
    fade.addEventListener('input', update);
    reset?.addEventListener('click', () => {
      avatarView = { ...VIEW_DEFAULTS };
      size.value = String(avatarView.size);
      position.value = String(avatarView.position);
      fade.value = String(avatarView.fade);
      applyAvatarView(true);
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

  async function connect() {
    if (connecting) return;
    connecting = true;
    if (retryHandle) {
      clearTimeout(retryHandle);
      retryHandle = null;
    }
    setStatus('connecting…');
    destroyPlayer();
    try {
      const mpegts = window.mpegts;
      if (!mpegts?.isSupported?.() || !mpegts.createPlayer) {
        throw new Error('当前浏览器不支持直播解码');
      }
      quietMpegts();
      const player = mpegts.createPlayer({
        type: 'flv',
        isLive: true,
        url: `${gw()}/livestream.flv?music=${musicEnabled ? '1' : '0'}&t=${Date.now()}`,
      }, {
        // A small bounded stash is smoother on real-world mobile/public
        // networks than zero-buffer chasing. Audio and video stay muxed, so
        // the added latency never changes lip-sync.
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
      // Always establish playback muted first so reconnects remain eligible
      // for autoplay even when they happen outside a user gesture. We restore
      // audio immediately below when this page has already been unlocked.
      liveVideo.muted = true;
      player.attachMediaElement(liveVideo);
      player.load();
      const statsEvent = mpegts.Events?.STATISTICS_INFO;
      if (statsEvent) {
        player.on(statsEvent, (info) => {
          window.AVATAR_STREAM_STATS = {
            speed: Number(info?.speed || 0),
            decodedFrames: Number(info?.decodedFrames || 0),
            droppedFrames: Number(info?.droppedFrames || 0),
            at: Date.now(),
          };
        });
      }
      lastMediaTime = -1;
      lastProgressAt = Date.now();
      await Promise.race([
        player.play().catch(() => {}),
        new Promise((resolve) => liveVideo.addEventListener('playing', resolve, { once: true })),
        new Promise((_, reject) => setTimeout(() => reject(new Error('直播首帧超时')), 8000)),
      ]);
      layer.classList.add('live');
      syncAudioRoute();
      setStatus('AVTR-1 HTTP-FLV');
      startWatchdog();
    } catch (err) {
      destroyPlayer();
      syncAudioRoute();
      layer.classList.remove('live');
      setStatus('连接失败: ' + (err?.message || err));
      scheduleReconnect(3000);
    } finally {
      connecting = false;
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
      { id: 'xiaoya', label: '小雅' },
      { id: 'xiaoya_idle', label: '暖光正脸' },
      { id: 'xiaoya_beach_close', label: '海边近景' },
      { id: 'xiaoya_beach', label: '海边' },
      { id: 'xiaoya_locket', label: '白背心' },
    ];
    try {
      const response = await fetch(gw() + '/avatars', { cache: 'no-store' });
      const data = await response.json();
      if (Array.isArray(data.avatars) && data.avatars.length) avatars = data.avatars;
    } catch (_) { /* keep defaults */ }
    const selected = currentAvatarId();
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
    void buildLooks();

    statusEl = document.getElementById('avatar-connection-state');
    if (!statusEl && CFG.showStatus) {
      statusEl = document.createElement('div');
      statusEl.id = 'av-status';
      document.body.appendChild(statusEl);
    }
    setStatus('idle');
    bindMusicToggle();
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
    connecting = false;
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
    void connect();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
})();
