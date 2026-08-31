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
  const meta = [service.port ? `Port ${service.port}` : null, service.details?.model ? `Model ${service.details.model}` : null]
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
  if (path.startsWith("/service/")) {
    await renderDetail(path.slice("/service/".length));
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
  const lines = (result.steps || []).map((step) => `${step.ok ? "✓" : "✗"} ${step.name}: ${step.error || (step.already_running ? "already running" : "ok")}`);
  showProgress(lines.join("\n") + (result.ok ? "\n\nSYSTEM READY" : "\n\nStartup finished with errors"));
  await refresh();
});

document.getElementById("stop-all").addEventListener("click", async () => {
  showProgress("Stopping managed services…");
  const result = await api("/api/stack/stop", { method: "POST", body: "{}" });
  const lines = (result.steps || []).map((step) => `${step.ok ? "✓" : "✗"} ${step.name}`);
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
