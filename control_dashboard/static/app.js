const LABEL = {
  online: "Online",
  starting: "Starting",
  degraded: "Degraded",
  offline: "Offline",
  not_configured: "Not configured",
};

const ROUTE_TITLES = {
  "/": "Overview",
  "/physical": "Physical",
  "/simulator": "Simulator",
  "/vision": "Vision",
  "/people": "People",
  "/ai-stack": "AI Stack",
  "/tools": "Tools",
  "/logs": "Logs",
  "/settings": "Settings",
  "/config": "Settings",
  "/tests": "System checks",
};

let lastEventId = 0;
let paused = false;
let lastStatus = null;

const view = document.getElementById("view");
const eventList = document.getElementById("event-list");
const eventFilter = document.getElementById("event-filter");
const progressEl = document.getElementById("progress");
const readyBanner = document.getElementById("ready-banner");
const readyLabel = document.getElementById("ready-label");
const readyReason = document.getElementById("ready-reason");
const healthStrip = document.getElementById("health-strip");
const currentSection = document.getElementById("current-section");
const eventsPanel = document.getElementById("events-panel");

function route() {
  return location.hash.replace(/^#/, "") || "/";
}

function statusClass(status) {
  return `dot dot--${status || "offline"}`;
}

function statusLabel(status, label) {
  const safeStatus = status || "unknown";
  return `<span class="status-label ${statusClass(safeStatus)}"><span class="status-dot status-dot--${escapeHtml(
    safeStatus,
  )}" aria-hidden="true"></span>${escapeHtml(label || LABEL[safeStatus] || safeStatus)}</span>`;
}

function pageHeading(eyebrow, title, description, aside = "") {
  return `<header class="page-heading"><div><p class="eyebrow">${escapeHtml(eyebrow)}</p><h2>${escapeHtml(
    title,
  )}</h2><p>${escapeHtml(description)}</p></div>${aside ? `<div class="page-heading__aside">${aside}</div>` : ""}</header>`;
}

function cameraRouteActive() {
  return route() === "/physical" || route() === "/vision";
}

let physicalCameraTimer = null;
let physicalCameraBackoffMs = 1000;
let physicalCameraOn = true;
let physicalCameraAbort = null;
let physicalCameraGeneration = 0;
let physicalCameraUrl = null;
const audioMeters = { microphone: { target: 0, value: 0, received: 0 }, speaker: { target: 0, value: 0, received: 0 } };
let audioAnimation = null;
let audioPolling = false;

function stopPhysicalCamera() {
  physicalCameraGeneration += 1;
  if (physicalCameraAbort) physicalCameraAbort.abort();
  physicalCameraAbort = null;
  if (physicalCameraUrl) URL.revokeObjectURL(physicalCameraUrl);
  physicalCameraUrl = null;
  if (physicalCameraTimer) {
    clearTimeout(physicalCameraTimer);
    physicalCameraTimer = null;
  }
}

function levelWave(kind, value) {
  updateLevelWave(kind, value);
  const bars = Array.from({ length: 36 }, () => "<span></span>").join("");
  return `<div class="audio-wave" id="${kind}-wave" role="meter" aria-label="${kind} RMS level" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">${bars}</div><div class="meter-caption"><span id="${kind}-signal">Idle</span><span>REAL RMS${kind === "speaker" ? " · TTS OUTPUT" : " · INPUT"}</span></div>`;
}

function updateLevelWave(kind, value) {
  const meter = audioMeters[kind];
  meter.target = Math.max(0, Math.min(1, Number(value) || 0));
  meter.received = Number.isFinite(value) ? performance.now() : 0;
  if (!audioAnimation) audioAnimation = requestAnimationFrame(animateAudioMeters);
}

function animateAudioMeters(now) {
  audioAnimation = null;
  if (route() !== "/physical" || document.hidden) return;
  const reducedMotion = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  for (const [kind, meter] of Object.entries(audioMeters)) {
    const connected = meter.received > 0 && now - meter.received < 1800;
    const target = connected ? meter.target : 0;
    const elapsed = Math.min(100, Math.max(0, now - (meter.lastFrame || now)));
    meter.lastFrame = now;
    meter.value += (target - meter.value) * (1 - Math.exp(-elapsed / (target > meter.value ? 65 : 240)));
    const wave = document.getElementById(`${kind}-wave`);
    if (!wave) continue;
    [...wave.children].forEach((bar, index) => {
      const distance = Math.abs(index - 17.5) / 17.5;
      const envelope = Math.max(.2, 1 - distance * .7);
      const carrier = .45 + Math.abs(Math.sin(index * .68 + (reducedMotion ? 0 : now / 210))) * .55;
      const idle = connected ? .045 : .025 + carrier * .018;
      const amplitude = Math.min(1, idle + meter.value * envelope * carrier);
      bar.style.transform = `scaleY(${amplitude})`;
    });
    const percent = Math.round(meter.value * 100);
    if (wave.getAttribute("aria-valuenow") !== String(percent)) wave.setAttribute("aria-valuenow", String(percent));
    const signal = document.getElementById(`${kind}-signal`);
    const label = connected ? (percent > 2 ? `${percent}% · Active` : "Idle") : "Telemetry disconnected";
    if (signal && signal.textContent !== label) signal.textContent = label;
  }
  audioAnimation = requestAnimationFrame(animateAudioMeters);
}

async function updatePhysicalTelemetry() {
  if (route() !== "/physical" || !view.querySelector(".physical")) return;
  if (audioPolling || document.hidden) return;
  audioPolling = true;
  let media;
  try { media = await api("/api/physical/audio", { signal: AbortSignal.timeout(1500) }); }
  catch (error) { media = {}; }
  finally { audioPolling = false; }
  updateLevelWave("microphone", media.microphone_level);
  updateLevelWave("speaker", media.speaker_level);
  const micStatus = document.getElementById("microphone-status");
  const speakerStatus = document.getElementById("speaker-status");
  if (micStatus) micStatus.textContent = (media.microphone_status || "offline").toUpperCase();
  if (speakerStatus) speakerStatus.textContent = (media.speaker_status || "offline").toUpperCase();
}

async function renderPhysical() {
  stopPhysicalCamera();
  view.innerHTML = `<section class="physical">${pageHeading("PHYSICAL ROBOT", "Physical", "Reading the active Reachy target and media state.")}<div class="empty-state"><p>Checking physical systems…</p></div></section>`;
  const status = await api("/api/physical/status");
  if (route() !== "/physical") return;
  const target = status.target || {};
  const banners = status.banners || {};
  const media = status.media || {};
  const robot = status.robot || {};
  const safeStop = status.safe_stop || {};
  physicalCameraOn = status.camera_preview_enabled !== false;
  const isPhysical = target.kind === "physical";
  const stack = (status.stack || [])
    .map(
      (row) =>
        `<li><span class="name">${escapeHtml(row.name)}</span><span class="${statusClass(row.status)}">${escapeHtml(
          row.label || row.status,
        )}</span></li>`,
    )
    .join("");
  view.innerHTML = `
    <section class="physical">
      <div class="physical__head">
        <div>
          <h2 class="physical__title">${escapeHtml(target.assistant_name || "Reachy Mini")} — Physical AI Stack</h2>
          <p class="physical__target">TARGET · ${escapeHtml(target.label || "UNKNOWN")} · ${escapeHtml(
            target.host || "—",
          )}:${escapeHtml(String(target.port || ""))}</p>
          <p class="physical__note">${escapeHtml(target.summary || "")}</p>
        </div>
        <div class="physical__badges">
          <span class="badge ${banners.connected ? "is-on" : "is-off"}">${banners.connected ? "● Connected" : "● Offline"}</span>
          <span class="badge ${banners.ai_online ? "is-on" : "is-off"}">${banners.ai_online ? "● AI Online" : "● AI Offline"}</span>
          <span class="badge ${banners.audio_online ? "is-on" : "is-off"}">${banners.audio_online ? "● Audio Online" : "● Audio Offline"}</span>
        </div>
      </div>

      ${
        isPhysical
          ? `<p class="physical__note">Controlling <strong>PHYSICAL REACHY MINI</strong> — not the simulator.</p>`
          : `<p class="card__reason">Physical controls are locked while the target is ${escapeHtml(
              target.label || target.kind || "unknown",
            )}. Simulator overview remains on the main page.</p>`
      }

      <div class="physical__panels">
        <article class="panel">
          <h3>Camera Preview · LIVE PHYSICAL CAMERA</h3>
          <div class="camera-frame" id="camera-frame">
            <img id="camera-img" alt="Physical Reachy Mini camera preview" hidden />
            <div class="camera-frame__msg" id="camera-msg">${
              physicalCameraOn ? "Waiting for camera preview…" : "PREVIEW OFF — dashboard not requesting frames"
            }</div>
          </div>
          <div class="actions" style="margin-top:12px">
            <button type="button" class="btn" id="camera-toggle">${
              physicalCameraOn ? "Preview Off" : "Preview On"
            }</button>
          </div>
          <p class="card__meta" id="camera-meta">${escapeHtml(
            (status.camera_preview && status.camera_preview.summary) ||
              media.camera_summary ||
              "Camera Preview On/Off only controls dashboard frame requests.",
          )}</p>

          <h3 style="margin-top:20px">Microphone Input</h3>
          <p id="microphone-status" class="${statusClass(media.microphone_status === "error" ? "offline" : "online")}">${escapeHtml(
            (media.microphone_status || "offline").toUpperCase(),
          )}</p>
          ${levelWave("microphone", media.microphone_level)}
          <dl class="kv">
            ${kv("Sample rate", media.input_sample_rate ? `${media.input_sample_rate} Hz` : "")}
            ${kv("Muted", media.microphone_muted ? "yes" : "no")}
          </dl>
          <div class="actions">
            <button type="button" class="btn" data-physical="mic" data-muted="${media.microphone_muted ? "0" : "1"}" ${
              isPhysical ? "" : "disabled"
            }>${media.microphone_muted ? "Unmute" : "Mute"}</button>
            <button type="button" class="btn" data-physical="mic-test" ${isPhysical ? "" : "disabled"}>Watch Levels</button>
          </div>
          <p class="card__meta">Uses Reachy's existing mic pipeline — no second capture stream.</p>

          <h3 style="margin-top:20px">Speaker Output</h3>
          <p id="speaker-status" class="${statusClass(media.speaker_status === "error" ? "offline" : "online")}">${escapeHtml(
            (media.speaker_status || "offline").toUpperCase(),
          )}</p>
          ${levelWave("speaker", media.speaker_level)}
          <dl class="kv">
            ${kv("Sample rate", media.output_sample_rate ? `${media.output_sample_rate} Hz` : "")}
            ${kv("Volume control", "unavailable")}
            ${kv("Muted", media.speaker_muted ? "yes" : "no")}
          </dl>
          <div class="actions">
            <button type="button" class="btn" data-physical="speaker" data-muted="${media.speaker_muted ? "0" : "1"}" ${
              isPhysical ? "" : "disabled"
            }>${media.speaker_muted ? "Unmute" : "Mute"}</button>
            <button type="button" class="btn" data-physical="speaker-test" ${isPhysical ? "" : "disabled"}>Test Speaker</button>
          </div>
          <p class="card__meta">${escapeHtml(
            media.volume_control_summary || "Volume control unavailable — no Reachy client volume API.",
          )}</p>
        </article>

        <article class="panel">
          <h3>AI Stack</h3>
          <ul class="stack-list">${stack}</ul>
        </article>
      </div>

      <div class="physical__panels">
        <article class="panel">
          <h3>PHYSICAL REACHY MINI</h3>
          <dl class="kv">
            ${kv("Connection", robot.connection)}
            ${kv("SDK", robot.sdk)}
            ${kv("Motors", robot.motors)}
            ${kv("Camera", robot.camera)}
            ${kv("Microphone", robot.microphone)}
            ${kv("Speaker", robot.speaker)}
            ${kv("Robot state", robot.state)}
            ${kv("Daemon host", robot.wlan_ip)}
            ${kv("Daemon state", robot.daemon_state)}
          </dl>
          <div class="actions">
            <button type="button" class="btn btn--danger" data-physical="safe-stop" ${
              isPhysical && safeStop.available ? "" : "disabled"
            }>SAFE STOP</button>
          </div>
          <p class="card__meta">${escapeHtml(
            safeStop.summary ||
              "Stops active motion and disables motors. Not goto_sleep. Does not stop Reachy/Hermes/dashboard.",
          )}</p>
        </article>
        <article class="panel">
          <h3>Hermes / Local AI</h3>
          <dl class="kv">
            ${kv("Hermes", status.hermes?.label)}
            ${kv("Hermes detail", status.hermes?.summary)}
            ${kv("Local AI", status.local_ai?.label)}
            ${kv("Model", status.local_ai?.model)}
            ${kv("GPU", status.local_ai?.gpu)}
          </dl>
        </article>
      </div>
    </section>`;

  const cameraToggle = view.querySelector("#camera-toggle");
  if (cameraToggle) {
    cameraToggle.addEventListener("click", async () => {
      const next = !physicalCameraOn;
      await api("/api/physical/camera", { method: "POST", body: JSON.stringify({ enabled: next }) });
      await renderPhysical();
    });
  }
  view.querySelectorAll("[data-physical]").forEach((button) => {
    button.addEventListener("click", async () => {
      const action = button.getAttribute("data-physical");
      try {
        if (action === "mic") {
          const muted = button.getAttribute("data-muted") === "1";
          showProgress(muted ? "Muting microphone…" : "Unmuting microphone…");
          await api("/api/physical/mic", { method: "POST", body: JSON.stringify({ muted }) });
        } else if (action === "mic-test") {
          showProgress("Unmute and speak — watching Reachy mic levels.");
          await api("/api/physical/mic", { method: "POST", body: JSON.stringify({ muted: false }) });
        } else if (action === "speaker") {
          const muted = button.getAttribute("data-muted") === "1";
          showProgress(muted ? "Muting speaker…" : "Unmuting speaker…");
          await api("/api/physical/speaker", { method: "POST", body: JSON.stringify({ muted }) });
        } else if (action === "speaker-test") {
          showProgress("Playing short speaker test…");
          const result = await api("/api/physical/speaker/test", { method: "POST", body: "{}" });
          showProgress(result.result?.ok === false ? result.result.error : "Speaker test sent.");
          return;
        } else if (action === "safe-stop") {
          if (
            !window.confirm(
              "SAFE STOP: stop moves and disable motors on the PHYSICAL robot?\n\nThis is not goto_sleep. Reachy, Hermes, and the dashboard stay running. Motors stay disabled until re-enabled elsewhere.",
            )
          )
            return;
          showProgress("Safe stop (motor / torque disable)…");
          const result = await api("/api/physical/safe-stop", { method: "POST", body: "{}" });
          showProgress(JSON.stringify(result.result || result));
        }
        await renderPhysical();
      } catch (error) {
        showProgress(String(error));
      }
    });
  });

  if (isPhysical && physicalCameraOn) {
    schedulePhysicalCamera();
  }
}

function schedulePhysicalCamera() {
  stopPhysicalCamera();
  const generation = physicalCameraGeneration;
  const img = document.getElementById("camera-img");
  const msg = document.getElementById("camera-msg");
  const meta = document.getElementById("camera-meta");
  if (!img || !msg) return;
  const tick = async () => {
    if (generation !== physicalCameraGeneration || !cameraRouteActive() || !physicalCameraOn || document.hidden) return;
    const started = performance.now();
    try {
      const controller = new AbortController();
      physicalCameraAbort = controller;
      const timeout = setTimeout(() => controller.abort(), 5000);
      let response;
      try {
        response = await fetch(`/api/physical/camera.jpg?t=${Date.now()}`, { cache: "no-store", signal: controller.signal });
      } finally { clearTimeout(timeout); }
      if (generation !== physicalCameraGeneration) return;
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        img.hidden = true;
        msg.hidden = false;
        msg.textContent = payload.error || "CAMERA OFFLINE";
        if (meta) meta.textContent = payload.meta?.summary || payload.error || "";
        physicalCameraBackoffMs = Math.min(8000, Math.max(1000, physicalCameraBackoffMs * 2));
      } else {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const decoded = new Image();
        decoded.src = url;
        try { await decoded.decode(); }
        catch (error) { URL.revokeObjectURL(url); throw error; }
        if (generation !== physicalCameraGeneration) { URL.revokeObjectURL(url); return; }
        const previous = physicalCameraUrl;
        physicalCameraUrl = url;
        img.src = url;
        if (previous) URL.revokeObjectURL(previous);
        img.hidden = false;
        msg.hidden = true;
        if (meta) meta.textContent = `Live · ${Math.round(performance.now() - started)} ms request + decode · ${Math.round(blob.size / 1024)} KiB`;
        physicalCameraBackoffMs = Math.max(0, 200 - (performance.now() - started));
      }
    } catch (error) {
      if (generation !== physicalCameraGeneration) return;
      img.hidden = true;
      msg.hidden = false;
      msg.textContent = "CAMERA PREVIEW OFFLINE";
      physicalCameraBackoffMs = Math.min(8000, Math.max(1000, physicalCameraBackoffMs * 2));
    }
    if (generation === physicalCameraGeneration) physicalCameraTimer = setTimeout(tick, physicalCameraBackoffMs);
  };
  physicalCameraTimer = setTimeout(tick, 200);
}

async function api(path, options) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function showProgress(text) {
  progressEl.hidden = !text;
  progressEl.textContent = text || "";
}

function renderReady(readiness) {
  readyLabel.textContent = readiness.label;
  readyReason.textContent = readiness.reason || "";
  readyBanner.classList.toggle("is-ready", readiness.ready);
  readyBanner.classList.toggle("is-not", !readiness.ready);
}

function renderGlobalStatus(status) {
  const wanted = [
    ["reachy_daemon", "Reachy"],
    ["conversation", "Conversation"],
    ["speech", "Speech"],
    ["llama", "LLM"],
    ["home_assistant", "Home"],
    ["apex", "Apex"],
    ["bus", "Bus"],
  ];
  healthStrip.innerHTML = wanted.map(([id, label]) => {
    const service = status.services.find((item) => item.id === id);
    const state = service?.status || "unknown";
    const detail = service ? `${service.name}: ${service.summary || LABEL[state] || state}` : `${label}: unavailable`;
    return `<span class="health-chip" data-status="${escapeHtml(state)}" title="${escapeHtml(detail)}"><span class="health-chip__dot" aria-hidden="true"></span>${escapeHtml(label)}</span>`;
  }).join("");
}

function serviceCard(service) {
  const reason = service.reason && service.status !== "online" ? `<p class="card__reason">${escapeHtml(service.reason)}</p>` : "";
  const meta = [
    service.port ? `Port ${service.port}` : null,
    service.managed && service.external ? "External process" : null,
    service.details?.environment ? service.details.environment : null,
    service.details?.model ? `Model ${service.details.model}` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return `<button type="button" class="card" data-open="${service.id}">
    <div class="card__row">
      <span class="card__name">${escapeHtml(service.name)}</span>
      ${statusLabel(service.status)}
    </div>
    <p class="card__meta">${escapeHtml(meta || service.description)}</p>
    ${reason}
  </button>`;
}

function renderOverview(status) {
  const online = status.services.filter((service) => service.status === "online").length;
  const attention = status.services.filter((service) => ["offline", "degraded"].includes(service.status)).length;
  const configured = status.services.filter((service) => service.status !== "not_configured").length;
  const reachy = status.services.find((service) => service.id === "reachy_daemon");
  const conversation = status.services.find((service) => service.id === "conversation");
  const priorityIds = ["reachy_daemon", "conversation", "speech", "llama", "home_assistant", "apex", "bus"];
  const priority = priorityIds.map((id) => status.services.find((service) => service.id === id)).filter(Boolean);
  const summary = status.readiness.ready
    ? "Required systems are responding. Reachy is ready for an explicit operator action."
    : status.readiness.reason || "System health requires attention.";
  view.innerHTML = `
    ${pageHeading("COMMAND CENTRE", "System overview", "A real-time view of Reachy's operational readiness.")}
    <section class="overview-hero">
      <article class="hero-card ${status.readiness.ready ? "" : "is-offline"}">
        <div class="hero-card__status"><span class="hero-card__beacon" aria-hidden="true"></span>
          ${statusLabel(status.readiness.ready ? "online" : "offline", status.readiness.label)}</div>
        <div><h2>${status.readiness.ready ? "Reachy is healthy and ready." : "Reachy needs attention."}</h2>
          <p class="hero-card__copy">${escapeHtml(summary)}</p></div>
        <div class="hero-card__meta"><span>ROBOT <strong>${escapeHtml(reachy ? LABEL[reachy.status] || reachy.status : "Unavailable")}</strong></span>
          <span>CONVERSATION <strong>${escapeHtml(conversation ? LABEL[conversation.status] || conversation.status : "Unavailable")}</strong></span>
          <span>LAST CHECK <strong>${escapeHtml(reachy?.last_health_at || "Unavailable")}</strong></span></div>
      </article>
      <div class="metric-stack">
        <article class="metric-card"><p class="metric-card__label">Services online</p><strong class="metric-card__value">${online}/${status.services.length}</strong><span class="metric-card__note">Current health checks</span></article>
        <article class="metric-card"><p class="metric-card__label">Need attention</p><strong class="metric-card__value">${attention}</strong><span class="metric-card__note">Offline or degraded</span></article>
        <article class="metric-card"><p class="metric-card__label">Configured</p><strong class="metric-card__value">${configured}</strong><span class="metric-card__note">Available integrations</span></article>
        <article class="metric-card"><p class="metric-card__label">Recovery</p><strong class="metric-card__value">${status.auto_restart ? "On" : "Off"}</strong><span class="metric-card__note">Managed services only</span></article>
      </div>
    </section>
    <div class="section-heading"><div><h3>Operational systems</h3><p>Open a service for controls, diagnostics, and recent logs.</p></div></div>
    <div class="system-grid">${priority.map(serviceCard).join("")}</div>`;
  view.querySelectorAll("[data-open]").forEach((button) => {
    button.addEventListener("click", () => {
      location.hash = `#/service/${button.getAttribute("data-open")}`;
    });
  });
}

function kv(label, value) {
  if (value === undefined || value === null || value === "") return "";
  return `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(String(value))}</dd>`;
}

async function renderDetail(serviceId) {
  const [service, logs] = await Promise.all([
    api(`/api/services/${serviceId}`),
    api(`/api/services/${serviceId}/logs`),
  ]);
  const blocked = (service.blocked_by || [])
    .map(
      (item) =>
        `<p>${escapeHtml(item.name)} is ${escapeHtml(item.status)}. ${escapeHtml(item.summary || "")}</p>
         <button type="button" class="btn btn--primary" data-action="start" data-id="${item.id}">Start ${escapeHtml(item.name)}</button>`,
    )
    .join("");
  const managedButtons = service.managed
    ? `<button type="button" class="btn btn--primary" data-action="start" data-id="${service.id}">Start</button>
       <button type="button" class="btn" data-action="restart" data-id="${service.id}">Restart</button>
       <button type="button" class="btn btn--danger" data-action="stop" data-id="${service.id}">Stop</button>`
    : `<p class="card__meta">This service is monitored only. The dashboard does not start or stop it.</p>`;
  const dev = lastStatus?.development_mode
    ? kv("PID", service.pid) + kv("Command", service.command) + kv("Working directory", service.cwd)
    : "";
  view.innerHTML = `
    <article class="detail">
      <p><a href="#/">← Overview</a></p>
      <h2>${escapeHtml(service.name)}</h2>
      ${statusLabel(service.status)}
      <p>${escapeHtml(service.summary || "")}</p>
      ${service.reason && service.status !== "online" ? `<p class="card__reason">${escapeHtml(service.reason)}</p>` : ""}
      ${service.suggested_action ? `<p><strong>Suggested:</strong> ${escapeHtml(service.suggested_action)}</p>` : ""}
      ${blocked ? `<div class="blocked">${blocked}</div>` : ""}
      <dl class="kv">
        ${kv("Port", service.port)}
        ${kv("Host", service.host)}
        ${kv("Ownership", service.managed ? (service.external ? "External process" : service.owned ? "Started by dashboard" : "Not running") : "Monitored only")}
        ${kv("Environment", service.details?.environment)}
        ${kv("Model", service.details?.model)}
        ${kv("GPU", service.details?.gpu)}
        ${kv("Last response", service.latency_ms != null ? `${service.latency_ms} ms` : "")}
        ${kv("Started", service.started_at)}
        ${kv("Restart attempts", `${service.restart_attempts} / ${service.restart_limit}`)}
        ${dev}
      </dl>
      <div class="actions">
        ${managedButtons}
        <button type="button" class="btn" data-action="health" data-id="${service.id}">Health check</button>
      </div>
      ${
        service.technical
          ? `<details class="tech"><summary>Technical error</summary><pre>${escapeHtml(service.technical)}</pre></details>`
          : ""
      }
      <div class="logs-head">
        <h3>Recent logs</h3>
        <button type="button" class="btn btn--ghost" id="copy-logs">Copy logs</button>
      </div>
      <pre class="logs" id="service-logs">${escapeHtml((logs.lines || []).join("\n") || "No captured logs yet.")}</pre>
    </article>`;
  bindActions(view);
  const copyLogs = view.querySelector("#copy-logs");
  const logEl = view.querySelector("#service-logs");
  if (copyLogs && logEl) {
    copyLogs.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(logEl.textContent || "");
        copyLogs.textContent = "Copied";
      } catch (error) {
        showProgress(`Could not copy logs: ${error}`);
      }
    });
  }
}

function renderTests(status) {
  const buttons = status.services
    .map(
      (service) =>
        `<button type="button" class="btn" data-action="test" data-id="${service.id}">Test ${escapeHtml(service.name)}</button>`,
    )
    .join("");
  view.innerHTML = `<article class="detail"><h2>System test</h2><p>Read-only checks. Home Assistant and Apex tests do not control devices.</p><div class="tests">${buttons}</div><pre class="logs" id="test-out"></pre></article>`;
  bindActions(view);
}

function renderSimulator(status) {
  const simulator = status.services.find((service) => service.id === "reachy_daemon");
  const conversation = status.services.find((service) => service.id === "conversation");
  const environment = simulator?.details?.environment;
  const simulatorActive = environment === "simulator";
  view.innerHTML = `
    ${pageHeading("VIRTUAL ROBOT", "Simulator", "Inspect and manage the existing local Reachy Mini simulator.", statusLabel(
      simulator?.status || "unknown",
      simulatorActive ? "Simulator active" : environment ? `Target: ${environment}` : "Target unknown",
    ))}
    <div class="settings-layout">
      <article class="panel"><div class="panel__header"><h3>Simulator daemon</h3>${statusLabel(simulator?.status || "unknown")}</div>
        <p class="physical__note">${escapeHtml(simulator?.summary || "Simulator status is unavailable.")}</p>
        <dl class="kv">${kv("Environment", environment || "Unavailable")}${kv("Host", simulator?.host)}${kv("Port", simulator?.port)}
          ${kv("Ownership", simulator?.ownership)}${kv("Last response", simulator?.latency_ms != null ? `${simulator.latency_ms} ms` : "Unavailable")}</dl>
        <div class="actions">
          ${simulator?.managed ? `<button class="btn btn--primary" data-action="start" data-id="reachy_daemon">Start simulator</button>
          <button class="btn" data-action="restart" data-id="reachy_daemon">Restart</button>
          <button class="btn btn--danger" data-action="stop" data-id="reachy_daemon">Stop</button>` : '<span class="physical__note">Monitored only</span>'}
        </div>
      </article>
      <article class="panel"><div class="panel__header"><h3>Conversation binding</h3>${statusLabel(conversation?.status || "unknown")}</div>
        <p class="physical__note">${escapeHtml(conversation?.summary || "Conversation status is unavailable.")}</p>
        <dl class="kv">${kv("Target", simulatorActive ? "Virtual Reachy Mini" : "Not confirmed as simulator")}
          ${kv("Dependency", "Reachy Mini daemon")}${kv("Port", conversation?.port)}</dl>
        <details class="diagnostic"><summary>Safety and diagnostics</summary><p>Viewing this page performs health checks only. Simulator commands run only after an explicit button press.</p></details>
      </article>
    </div>`;
  bindActions(view);
}

async function renderVision() {
  stopPhysicalCamera();
  view.innerHTML = `<section class="vision-page">${pageHeading("VISUAL SYSTEM", "Vision", "Reading existing camera telemetry.")}<div class="empty-state"><p>Checking camera state…</p></div></section>`;
  const status = await api("/api/physical/status");
  if (route() !== "/vision") return;
  const media = status.media || {};
  const target = status.target || {};
  physicalCameraOn = status.camera_preview_enabled !== false;
  view.innerHTML = `<section class="vision-page">
    ${pageHeading("VISUAL SYSTEM", "Vision", "Live camera presentation and existing camera diagnostics.", statusLabel(
      media.camera_status === "error" ? "offline" : media.camera_status === "offline" ? "offline" : "online",
      media.camera_status || "Unknown",
    ))}
    <div class="physical__panels">
      <article class="panel"><div class="panel__header"><h3>Camera viewport</h3><span class="quiet-badge">${escapeHtml(
        target.label || "Unknown target",
      )}</span></div>
        <div class="camera-frame" id="camera-frame"><img id="camera-img" alt="Reachy Mini camera preview" hidden />
          <div class="camera-frame__msg" id="camera-msg">${physicalCameraOn ? "Waiting for the existing camera preview…" : "Preview is off"}</div></div>
        <div class="camera-meta"><span id="camera-meta">${escapeHtml(media.camera_summary || "Camera telemetry unavailable.")}</span><span>READ ONLY</span></div>
      </article>
      <article class="panel"><div class="panel__header"><h3>Vision diagnostics</h3>${statusLabel(
        media.camera_status === "error" ? "offline" : media.camera_status === "offline" ? "offline" : "online",
      )}</div>
        <dl class="kv">${kv("Camera state", media.camera_status || "Unavailable")}${kv("Target", target.label || "Unavailable")}
          ${kv("Detector", "Unavailable in dashboard telemetry")}${kv("Tracking", "Unavailable in dashboard telemetry")}
          ${kv("Recognition", "Not exposed on this page")}</dl>
        <p class="physical__note">This view does not enable detection, tracking, recognition, or enrolment.</p>
        <details class="diagnostic"><summary>Technical details</summary><p>${escapeHtml(target.summary || "No target diagnostic is available.")}</p></details>
      </article>
    </div></section>`;
  if (target.kind === "physical" && physicalCameraOn) schedulePhysicalCamera();
}

function serviceNode(service) {
  const dependencies = service.depends_on?.length ? `Depends on ${service.depends_on.join(", ")}` : "No declared dependencies";
  return `<article class="service-node"><div class="service-node__head"><h4>${escapeHtml(service.name)}</h4>${statusLabel(
    service.status,
  )}</div><p>${escapeHtml(service.summary || service.description)}</p><div class="service-node__meta">
    <span>${service.port ? `:${escapeHtml(service.port)}` : "NO PORT"}</span><span>${escapeHtml(dependencies)}</span></div></article>`;
}

function renderAIStack(status) {
  const groups = [
    ["system", "Robot runtime", "Core Reachy services"],
    ["ai", "Intelligence", "Language and speech services"],
    ["tools", "Connected systems", "Optional tools and integrations"],
  ];
  view.innerHTML = `${pageHeading("SERVICE TOPOLOGY", "AI stack", "Health, ownership, and declared dependencies from the current service registry.")}
    <div class="topology">${groups.map(([id, title, description]) => {
      const services = status.services.filter((service) => service.group === id);
      return `<section class="topology-group"><div class="topology-group__label"><h3>${title}</h3><p>${description}</p></div>
        <div class="topology-group__services">${services.map(serviceNode).join("") || '<p class="physical__note">No registered services.</p>'}</div></section>`;
    }).join("")}</div>
    <div class="section-heading"><div><h3>Service controls</h3><p>Open a service to inspect it or perform an explicit action.</p></div></div>
    <div class="system-grid">${status.services.map(serviceCard).join("")}</div>`;
  view.querySelectorAll("[data-open]").forEach((button) => button.addEventListener("click", () => {
    location.hash = `#/service/${button.getAttribute("data-open")}`;
  }));
}

function renderTools(status) {
  const toolServices = status.services.filter((service) => service.group === "tools");
  const grouped = [
    ["Home", ["home_assistant"]],
    ["Reef", ["apex"]],
    ["Transport", ["bus"]],
    ["AI delegation", ["hermes"]],
  ];
  view.innerHTML = `${pageHeading("CAPABILITIES", "Tools & integrations", "Availability reflects registered services only; opening this page never calls a tool.")}
    <div class="tool-groups">${grouped.map(([title, ids]) => {
      const services = ids.map((id) => toolServices.find((service) => service.id === id)).filter(Boolean);
      return `<section class="tool-group"><div class="tool-group__head"><h3>${title}</h3><span class="quiet-badge">${services.length ? "Registered" : "Unavailable"}</span></div>
        ${services.length ? services.map((service) => `<div class="tool-service"><span class="status-dot status-dot--${escapeHtml(service.status)}"></span>
          <div class="tool-service__copy"><strong>${escapeHtml(service.name)}</strong><span>${escapeHtml(service.summary || service.description)}</span></div>
          <button class="btn btn--ghost" data-open="${escapeHtml(service.id)}">Inspect</button></div>`).join("") :
          '<div class="empty-state"><p>No service is registered for this category.</p></div>'}</section>`;
    }).join("")}</div>
    <details class="diagnostic"><summary>Scope of this view</summary><p>Individual LLM-callable tools are not exposed by the current dashboard API, so they are not fabricated here.</p></details>`;
  view.querySelectorAll("[data-open]").forEach((button) => button.addEventListener("click", () => {
    location.hash = `#/service/${button.getAttribute("data-open")}`;
  }));
}

function renderLogs() {
  view.innerHTML = `${pageHeading("OPERATIONS CONSOLE", "Logs & events", "Filter the existing local event feed without deleting server log files.")}
    <div class="empty-state"><h3>Live event console</h3><p>Events appear below as the dashboard receives them. Open a service from AI Stack for its captured log tail.</p></div>`;
  eventsPanel.hidden = false;
}

async function renderSettings(status) {
  const config = await api("/api/config");
  if (route() !== "/settings") return;
  const rows = Object.entries(config.env).map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`).join("");
  view.innerHTML = `${pageHeading("CONTROL PLANE", "Settings", "Dashboard preferences and redacted runtime configuration.")}
    <div class="settings-layout">
      <article class="panel"><div class="panel__header"><h3>Dashboard behavior</h3></div>
        <div class="setting-row"><div><h3>Developer mode</h3><p>Show process and command details in service diagnostics.</p></div>
          <label class="switch"><input type="checkbox" data-setting="development_mode" ${status.development_mode ? "checked" : ""} aria-label="Developer mode"><span></span></label></div>
        <div class="setting-row"><div><h3>Automatic recovery</h3><p>Allow the existing controller to recover managed services.</p></div>
          <label class="switch"><input type="checkbox" data-setting="auto_restart" ${status.auto_restart ? "checked" : ""} aria-label="Automatic recovery"><span></span></label></div>
        <div class="setting-row"><div><h3>Read-only tests</h3><p>Run registered service probes without controlling connected devices.</p></div>
          <button class="btn" id="open-tests">Open checks</button></div>
      </article>
      <article class="panel"><div class="panel__header"><h3>Runtime configuration</h3><span class="quiet-badge">Secrets masked</span></div>
        <dl class="kv">${kv("Conversation app", config.conversation_root)}${kv("AI stack", config.ai_stack_root)}</dl>
        <details class="diagnostic"><summary>Environment values</summary><table class="config-table">${rows}</table></details>
      </article>
    </div>`;
  view.querySelectorAll("[data-setting]").forEach((input) => input.addEventListener("change", async () => {
    input.disabled = true;
    const key = input.getAttribute("data-setting");
    try {
      await api("/api/settings", { method: "POST", body: JSON.stringify({ [key]: input.checked }) });
      showProgress(`${key === "auto_restart" ? "Automatic recovery" : "Developer mode"} updated.`);
      await refresh();
    } catch (error) {
      input.checked = !input.checked;
      showProgress(String(error));
    } finally {
      input.disabled = false;
    }
  }));
  document.getElementById("open-tests").addEventListener("click", () => {
    renderTests(lastStatus);
    currentSection.textContent = "System checks";
  });
}

function bindActions(root) {
  root.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", async () => {
      const id = button.getAttribute("data-id");
      const action = button.getAttribute("data-action");
      showProgress(`${action} ${id}…`);
      try {
        const result = await api(`/api/services/${id}/${action}`, { method: "POST", body: "{}" });
        const out = document.getElementById("test-out");
        if (out) out.textContent = JSON.stringify(result, null, 2);
        showProgress(result.error || result.summary || JSON.stringify(result.steps ? result : { ok: result.ok }, null, 2));
        await refresh();
      } catch (error) {
        showProgress(String(error));
      }
    });
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function appendEvents(items) {
  for (const event of items) {
    lastEventId = Math.max(lastEventId, event.id);
    const item = document.createElement("li");
    item.className = event.level;
    item.textContent = `${event.ts}  ${event.message}`;
    eventList.prepend(item);
  }
}

async function pollEvents() {
  if (paused) return;
  const params = new URLSearchParams({ after_id: String(lastEventId) });
  if (eventFilter.value) params.set("service", eventFilter.value);
  if (document.getElementById("errors-only").checked) params.set("errors_only", "1");
  const payload = await api(`/api/events?${params}`);
  appendEvents(payload.events || []);
}

function fillFilter(status) {
  const current = eventFilter.value;
  eventFilter.innerHTML = `<option value="">All services</option>` + status.services.map((service) => `<option value="${service.id}">${escapeHtml(service.name)}</option>`).join("");
  eventFilter.value = current;
}

function updateNavigation(path) {
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${path}` || (path === "/" && link.getAttribute("href") === "#/"));
  });
  currentSection.textContent = path.startsWith("/service/") ? "Service detail" : ROUTE_TITLES[path] || "Overview";
}

async function renderCurrentRoute(path) {
  eventsPanel.hidden = true;
  if (!cameraRouteActive()) {
    stopPhysicalCamera();
  }
  if (path.startsWith("/service/")) {
    await renderDetail(path.slice("/service/".length));
  } else if (path === "/physical") {
    // Keep the camera timer alive; only rebuild the page when navigating here.
    if (!view.querySelector(".physical")) {
      await renderPhysical();
    } else {
      await updatePhysicalTelemetry();
    }
  } else if (path === "/simulator") {
    renderSimulator(lastStatus);
  } else if (path === "/vision") {
    if (!view.querySelector(".vision-page")) await renderVision();
  } else if (path === "/people") {
    if (!view.querySelector(".people")) await renderPeople();
  } else if (path === "/ai-stack") {
    renderAIStack(lastStatus);
  } else if (path === "/tools") {
    renderTools(lastStatus);
  } else if (path === "/logs") {
    renderLogs();
  } else if (path === "/settings" || path === "/config") {
    await renderSettings(lastStatus);
  } else if (path === "/tests") {
    renderTests(lastStatus);
  } else {
    renderOverview(lastStatus);
  }
}

async function refresh() {
  const path = route();
  if (path === "/people" && !view.querySelector(".people")) await renderPeople();
  lastStatus = await api("/api/status");
  if (route() !== path) return;
  renderReady(lastStatus.readiness);
  renderGlobalStatus(lastStatus);
  fillFilter(lastStatus);
  updateNavigation(path);
  await renderCurrentRoute(path);
}

document.getElementById("start-all").addEventListener("click", async () => {
  showProgress("Starting Reachy AI stack…");
  const result = await api("/api/stack/start", { method: "POST", body: "{}" });
  const lines = (result.steps || []).map((step) => `${step.ok ? "✓" : "✗"} ${step.name}: ${step.error || (step.already_running ? (step.external ? "already running — external process" : "already running") : "ok")}`);
  showProgress(lines.join("\n") + (result.ok ? "\n\nSYSTEM READY" : "\n\nStartup finished with errors"));
  await refresh();
});

document.getElementById("stop-all").addEventListener("click", async () => {
  showProgress("Stopping managed services…");
  const result = await api("/api/stack/stop", { method: "POST", body: "{}" });
  const lines = (result.steps || []).map((step) => {
    if (step.already_stopped) return `✓ ${step.name}: already stopped`;
    if (step.stopped_pids && step.stopped_pids.length) return `✓ ${step.name}: stopped`;
    return `${step.ok ? "✓" : "✗"} ${step.name}${step.error ? `: ${step.error}` : ""}`;
  });
  showProgress(lines.join("\n"));
  await refresh();
});

document.getElementById("clear-events").addEventListener("click", () => {
  eventList.innerHTML = "";
});

document.getElementById("pause-events").addEventListener("click", (event) => {
  paused = !paused;
  event.currentTarget.textContent = paused ? "Resume" : "Pause";
});

window.addEventListener("hashchange", () => {
  if (peopleUI.busy) { location.hash = "#/people"; return; }
  const path = route();
  if (path !== "/people") clearPeoplePhotos();
  stopPhysicalCamera();
  updateNavigation(path);
  if (lastStatus) renderCurrentRoute(path).catch(console.error);
  else refresh().catch(console.error);
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopPhysicalCamera();
  else if (cameraRouteActive()) { schedulePhysicalCamera(); updatePhysicalTelemetry(); }
});
window.addEventListener("beforeunload", event => {
  if (peopleUI.busy) { event.preventDefault(); event.returnValue = ""; }
  else { clearPeoplePhotos(); stopPhysicalCamera(); }
});
setInterval(() => updatePhysicalTelemetry(), 150);
function updateClock() {
  const now = new Date();
  document.getElementById("local-time").textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  document.getElementById("local-date").textContent = now.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
}
updateClock();
setInterval(updateClock, 30000);
refresh().catch((error) => {
  readyLabel.textContent = "SYSTEM NOT READY";
  readyReason.textContent = String(error);
});
setInterval(() => {
  refresh().catch(console.error);
    pollEvents().catch(console.error);
}, 4000);
