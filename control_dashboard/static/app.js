const GROUPS = [
  { id: "system", title: "System status" },
  { id: "ai", title: "AI services" },
  { id: "tools", title: "Agents / tools" },
];

const DOT = {
  online: "🟢",
  starting: "🟡",
  degraded: "🟡",
  offline: "🔴",
  not_configured: "⚪",
};

const LABEL = {
  online: "Online",
  starting: "Starting",
  degraded: "Degraded",
  offline: "Offline",
  not_configured: "Not configured",
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
const devMode = document.getElementById("dev-mode");
const autoRestart = document.getElementById("auto-restart");

function route() {
  return location.hash.replace(/^#/, "") || "/";
}

function statusClass(status) {
  return `dot dot--${status || "offline"}`;
}

let physicalCameraTimer = null;
let physicalCameraBackoffMs = 1000;
let physicalCameraOn = true;
const physicalLevelHistory = { microphone: Array(32).fill(0), speaker: Array(32).fill(0) };

function stopPhysicalCamera() {
  if (physicalCameraTimer) {
    clearTimeout(physicalCameraTimer);
    physicalCameraTimer = null;
  }
}

function levelWave(kind, value) {
  const level = Math.max(0, Math.min(1, Number(value) || 0));
  physicalLevelHistory[kind] = [...physicalLevelHistory[kind].slice(1), level];
  const bars = physicalLevelHistory[kind]
    .map((sample) => `<span style="height:${Math.max(8, Math.round(sample * 100))}%"></span>`)
    .join("");
  return `<div class="level-wave" id="${kind}-wave" role="meter" aria-label="${kind} level" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${Math.round(level * 100)}">${bars}</div>`;
}

function updateLevelWave(kind, value) {
  const level = Math.max(0, Math.min(1, Number(value) || 0));
  physicalLevelHistory[kind] = [...physicalLevelHistory[kind].slice(1), level];
  const wave = document.getElementById(`${kind}-wave`);
  if (!wave) return;
  wave.setAttribute("aria-valuenow", String(Math.round(level * 100)));
  [...wave.children].forEach((bar, index) => {
    bar.style.height = `${Math.max(8, Math.round(physicalLevelHistory[kind][index] * 100))}%`;
  });
}

async function updatePhysicalTelemetry() {
  if (route() !== "/physical" || !view.querySelector(".physical")) return;
  const status = await api("/api/physical/status");
  const media = status.media || {};
  updateLevelWave("microphone", media.microphone_level);
  updateLevelWave("speaker", media.speaker_level);
  const micStatus = document.getElementById("microphone-status");
  const speakerStatus = document.getElementById("speaker-status");
  if (micStatus) micStatus.textContent = (media.microphone_status || "offline").toUpperCase();
  if (speakerStatus) speakerStatus.textContent = (media.speaker_status || "offline").toUpperCase();
}

async function renderPhysical() {
  stopPhysicalCamera();
  const status = await api("/api/physical/status");
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
  const img = document.getElementById("camera-img");
  const msg = document.getElementById("camera-msg");
  const meta = document.getElementById("camera-meta");
  if (!img || !msg) return;
  const tick = async () => {
    if (route() !== "/physical" || !physicalCameraOn) return;
    try {
      const response = await fetch(`/api/physical/camera.jpg?t=${Date.now()}`, { cache: "no-store" });
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
        const previous = img.src;
        img.onload = () => {
          if (previous && previous.startsWith("blob:")) URL.revokeObjectURL(previous);
        };
        img.src = url;
        img.hidden = false;
        msg.hidden = true;
        if (meta) meta.textContent = "LIVE PHYSICAL CAMERA — dashboard preview";
        physicalCameraBackoffMs = 1000;
      }
    } catch (error) {
      img.hidden = true;
      msg.hidden = false;
      msg.textContent = "CAMERA PREVIEW OFFLINE";
      physicalCameraBackoffMs = Math.min(8000, Math.max(1000, physicalCameraBackoffMs * 2));
    }
    physicalCameraTimer = setTimeout(tick, physicalCameraBackoffMs);
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
      <span class="${statusClass(service.status)}">${DOT[service.status] || "⚪"} ${LABEL[service.status] || service.status}</span>
    </div>
    <p class="card__meta">${escapeHtml(meta || service.description)}</p>
    ${reason}
  </button>`;
}

function renderOverview(status) {
  const sections = GROUPS.map((group) => {
    const cards = status.services.filter((service) => service.group === group.id).map(serviceCard).join("");
    return `<h2 class="group-title">${group.title}</h2><div class="grid">${cards}</div>`;
  }).join("");
  view.innerHTML = sections;
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
      <p class="${statusClass(service.status)}">${DOT[service.status] || ""} ${LABEL[service.status] || service.status}</p>
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

async function renderConfig() {
  const config = await api("/api/config");
  const rows = Object.entries(config.env)
    .map(([key, value]) => `<tr><th>${escapeHtml(key)}</th><td>${escapeHtml(value)}</td></tr>`)
    .join("");
  view.innerHTML = `
    <article class="detail">
      <h2>System configuration</h2>
      <p>Discovered from the conversation app <code>.env</code> and <code>control_dashboard/services.json</code>. Secrets are masked.</p>
      <dl class="kv">
        ${kv("Conversation app", config.conversation_root)}
        ${kv("AI stack", config.ai_stack_root)}
      </dl>
      <table class="config-table">${rows}</table>
    </article>`;
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

async function refresh() {
  lastStatus = await api("/api/status");
  renderReady(lastStatus.readiness);
  devMode.checked = !!lastStatus.development_mode;
  autoRestart.checked = !!lastStatus.auto_restart;
  fillFilter(lastStatus);
  const path = route();
  document.querySelectorAll(".nav a").forEach((link) => {
    link.classList.toggle("is-active", link.getAttribute("href") === `#${path}` || (path === "/" && link.getAttribute("href") === "#/"));
  });
  if (path !== "/physical") {
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
  } else if (path === "/config") {
    await renderConfig();
  } else if (path === "/tests") {
    renderTests(lastStatus);
  } else {
    renderOverview(lastStatus);
  }
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

document.getElementById("clear-events").addEventListener("click", async () => {
  await api("/api/events/clear", { method: "POST", body: "{}" });
  eventList.innerHTML = "";
  lastEventId = 0;
});

document.getElementById("pause-events").addEventListener("click", (event) => {
  paused = !paused;
  event.currentTarget.textContent = paused ? "Resume" : "Pause";
});

devMode.addEventListener("change", async () => {
  await api("/api/settings", { method: "POST", body: JSON.stringify({ development_mode: devMode.checked }) });
  await refresh();
});
autoRestart.addEventListener("change", async () => {
  await api("/api/settings", { method: "POST", body: JSON.stringify({ auto_restart: autoRestart.checked }) });
});

window.addEventListener("hashchange", () => refresh().catch(console.error));
refresh().catch((error) => {
  readyLabel.textContent = "SYSTEM NOT READY";
  readyReason.textContent = String(error);
});
setInterval(() => {
  refresh().catch(console.error);
    pollEvents().catch(console.error);
}, 4000);
