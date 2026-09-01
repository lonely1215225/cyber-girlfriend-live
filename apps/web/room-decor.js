const SCENES = new Set(["sun", "rain", "night", "rainbow", "overcast"]);
const MODES = new Set(["auto", "sun", "rain", "night", "rainbow"]);
const SCENE_LABEL = {
  sun: "晴",
  rain: "雨",
  night: "夜",
  rainbow: "虹",
  overcast: "阴",
};

const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
const mobile = window.matchMedia?.("(max-width: 900px)")?.matches;

function sceneOf(decor) {
  const scene = decor?.scene;
  return SCENES.has(scene) ? scene : "sun";
}

function modeOf(decor) {
  const mode = decor?.mode;
  return MODES.has(mode) ? mode : "auto";
}

function syncBadge(scene) {
  const badge = document.getElementById("room-weather-badge");
  const label = document.getElementById("room-weather-badge-label");
  if (badge) badge.dataset.scene = scene;
  if (label) label.textContent = SCENE_LABEL[scene] || "晴";
}

function resizeCanvas(canvas) {
  if (!canvas) return { width: 0, height: 0, ctx: null };
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth || window.innerWidth;
  const height = canvas.clientHeight || window.innerHeight;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height, ctx };
}

function makeStars(count) {
  return Array.from({ length: count }, () => ({
    x: Math.random(),
    y: Math.random(),
    r: 0.4 + Math.random() * 1.3,
    phase: Math.random() * Math.PI * 2,
    speed: 0.4 + Math.random() * 1.4,
  }));
}

function makeStreaks(count, width, height) {
  return Array.from({ length: count }, () => resetStreak({
    x: 0, y: 0, len: 0, speed: 0, width: 1, opacity: 0.4, wind: 0,
  }, width, height, true));
}

function resetStreak(item, width, height, scatter) {
  item.x = Math.random() * (width + 80) - 40;
  item.y = scatter ? Math.random() * height : -20 - Math.random() * 80;
  item.len = 10 + Math.random() * 18;
  item.speed = 14 + Math.random() * 18;
  item.width = 0.7 + Math.random() * 1.1;
  item.opacity = 0.22 + Math.random() * 0.38;
  item.wind = 1.2 + Math.random() * 1.6;
  return item;
}

function spawnDrop(x, y) {
  return {
    x,
    y,
    r: 2.2 + Math.random() * 3.6,
    vx: (Math.random() - 0.5) * 0.15,
    vy: 0,
    stuck: 50 + Math.random() * 90,
    life: 220 + Math.random() * 160,
    merge: true,
  };
}

const skyCanvas = document.getElementById("room-stars");
const glassCanvas = document.getElementById("room-glass");
let starState = { width: 0, height: 0, ctx: null, stars: makeStars(mobile ? 46 : 90) };
let rainState = {
  width: 0,
  height: 0,
  ctx: null,
  streaks: [],
  drops: [],
  splash: 0,
  meteors: [],
  nextMeteor: 1800,
};
let scene = "sun";
let raf = 0;
let last = 0;

function layout() {
  starState = { ...starState, ...resizeCanvas(skyCanvas) };
  rainState = { ...rainState, ...resizeCanvas(glassCanvas) };
  const count = reducedMotion ? 0 : (mobile ? 70 : 150);
  if (rainState.width) {
    rainState.streaks = makeStreaks(count, rainState.width, rainState.height);
  }
}

function drawStars(now) {
  const { ctx, width, height, stars } = starState;
  if (!ctx || scene !== "night") {
    ctx?.clearRect(0, 0, width, height);
    return;
  }
  ctx.clearRect(0, 0, width, height);
  for (const star of stars) {
    const twinkle = 0.35 + 0.65 * Math.abs(Math.sin(now * 0.001 * star.speed + star.phase));
    ctx.fillStyle = `rgba(245, 240, 220, ${twinkle})`;
    ctx.beginPath();
    ctx.arc(star.x * width, star.y * height, star.r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function spawnMeteor(width, height) {
  const fromTop = Math.random() < 0.65;
  const angle = 0.28 + Math.random() * 0.55;
  const speed = 18 + Math.random() * 22;
  return {
    x: fromTop ? Math.random() * width * 0.85 : -30,
    y: fromTop ? -16 - Math.random() * 40 : Math.random() * height * 0.45,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    len: 90 + Math.random() * 160,
    life: 1,
    decay: 0.012 + Math.random() * 0.01,
    width: 1.4 + Math.random() * 1.6,
  };
}

function drawMeteors(dt) {
  const { ctx, width, height } = rainState;
  if (!ctx || scene !== "night" || reducedMotion) {
    rainState.meteors = [];
    return;
  }
  rainState.nextMeteor -= dt;
  if (rainState.nextMeteor <= 0) {
    rainState.meteors.push(spawnMeteor(width, height));
    if (Math.random() < 0.22) rainState.meteors.push(spawnMeteor(width, height));
    rainState.nextMeteor = 2200 + Math.random() * 7000;
  }
  const next = [];
  for (const meteor of rainState.meteors) {
    meteor.x += meteor.vx * dt * 0.06;
    meteor.y += meteor.vy * dt * 0.06;
    meteor.life -= meteor.decay * dt * 0.06;
    if (meteor.life <= 0 || meteor.x > width + 80 || meteor.y > height + 80) continue;
    const tx = meteor.x - meteor.vx * meteor.len * 0.04;
    const ty = meteor.y - meteor.vy * meteor.len * 0.04;
    const grad = ctx.createLinearGradient(tx, ty, meteor.x, meteor.y);
    grad.addColorStop(0, "rgba(255,255,255,0)");
    grad.addColorStop(0.55, `rgba(220, 230, 255, ${0.35 * meteor.life})`);
    grad.addColorStop(1, `rgba(255, 252, 240, ${0.95 * meteor.life})`);
    ctx.strokeStyle = grad;
    ctx.lineWidth = meteor.width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(meteor.x, meteor.y);
    ctx.stroke();
    ctx.fillStyle = `rgba(255, 255, 255, ${meteor.life})`;
    ctx.beginPath();
    ctx.arc(meteor.x, meteor.y, meteor.width * 1.1, 0, Math.PI * 2);
    ctx.fill();
    next.push(meteor);
  }
  rainState.meteors = next;
}

function drawRain(dt) {
  const { ctx, width, height } = rainState;
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  const raining = scene === "rain";
  const leftover = scene === "rainbow";
  if (!raining && !leftover && !rainState.drops.length && scene !== "night") return;

  if (raining && !reducedMotion) {
    for (const streak of rainState.streaks) {
      streak.x += streak.wind * dt * 0.06;
      streak.y += streak.speed * dt * 0.06;
      ctx.strokeStyle = `rgba(190, 214, 232, ${streak.opacity})`;
      ctx.lineWidth = streak.width;
      ctx.beginPath();
      ctx.moveTo(streak.x, streak.y);
      ctx.lineTo(streak.x + streak.wind * 0.8, streak.y + streak.len);
      ctx.stroke();
      if (streak.y > height * (0.18 + Math.random() * 0.7) && Math.random() < 0.045) {
        rainState.drops.push(spawnDrop(streak.x, streak.y));
      }
      if (streak.y > height + 20) resetStreak(streak, width, height, false);
    }
  }

  const next = [];
  for (const drop of rainState.drops) {
    drop.life -= dt * 0.06;
    if (drop.stuck > 0) drop.stuck -= dt * 0.06;
    else drop.vy += 0.012 * dt;
    drop.x += drop.vx * dt;
    drop.y += drop.vy * dt * 0.04;
    if (drop.life <= 0 || drop.y > height + 8) continue;
    const alpha = Math.max(0, Math.min(0.72, drop.life / 180));
    ctx.fillStyle = `rgba(210, 228, 240, ${alpha})`;
    ctx.beginPath();
    ctx.ellipse(drop.x, drop.y, drop.r * 0.72, drop.r, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.7})`;
    ctx.beginPath();
    ctx.ellipse(drop.x - drop.r * 0.22, drop.y - drop.r * 0.28, drop.r * 0.2, drop.r * 0.14, 0, 0, Math.PI * 2);
    ctx.fill();
    next.push(drop);
  }
  rainState.drops = next.slice(-80);
  drawMeteors(dt);
}

function tick(now) {
  const dt = Math.min(48, now - last || 16);
  last = now;
  drawStars(now);
  drawRain(dt);
  raf = window.requestAnimationFrame(tick);
}

function startLoop() {
  if (raf) return;
  last = performance.now();
  raf = window.requestAnimationFrame(tick);
}

function stopLoop() {
  if (!raf) return;
  window.cancelAnimationFrame(raf);
  raf = 0;
}

export function applyRoomDecor(decor) {
  const next = sceneOf(decor);
  const mode = modeOf(decor);
  scene = next;
  document.body.dataset.roomScene = next;
  document.body.dataset.roomMode = mode;
  syncBadge(next);
  if (next === "rainbow") {
    rainState.drops = rainState.drops.slice(0, 18);
    if (!rainState.drops.length && rainState.width) {
      for (let i = 0; i < 12; i += 1) {
        rainState.drops.push(spawnDrop(
          24 + Math.random() * (rainState.width - 48),
          40 + Math.random() * (rainState.height * 0.7),
        ));
      }
    }
  } else if (next !== "rain") {
    rainState.drops = [];
  }
}

window.addEventListener("live-room-state", (event) => {
  applyRoomDecor(event.detail?.decor);
});

window.addEventListener("resize", () => {
  layout();
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopLoop();
  else startLoop();
});

function bootScene() {
  const hour = new Date().getHours();
  const scene = hour < 6 || hour >= 20 ? "night" : "sun";
  applyRoomDecor({ scene, mode: "auto" });
}

layout();
bootScene();
if (!document.hidden) startLoop();
