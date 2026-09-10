const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

function dashboard() {
  const timers = new Map();
  const elements = new Map();
  let clock = 100;
  let sequence = 0;
  const revoked = [];
  function element(id) {
    if (!elements.has(id)) elements.set(id, {
      textContent: "", hidden: false, src: "", innerHTML: "", value: "", children: [],
      firstElementChild: { style: {} }, attributes: {},
      addEventListener() {}, querySelector() { return {}; }, querySelectorAll() { return []; },
      classList: { toggle() {} },
      setAttribute(key, value) { this.attributes[key] = value; },
      getAttribute(key) { return this.attributes[key]; },
    });
    return elements.get(id);
  }
  const context = vm.createContext({
    console, Number, Math, Date, Object, Array, String, Promise, AbortController, AbortSignal,
    location: { hash: "#/physical" }, performance: { now: () => clock },
    URL: { createObjectURL: () => `blob:${++sequence}`, revokeObjectURL: url => revoked.push(url) },
    Image: class { async decode() {} },
    document: { hidden: false, getElementById: element, querySelectorAll: () => [], addEventListener() {} },
    window: { addEventListener() {} },
    requestAnimationFrame: () => 1,
    setInterval() {}, clearTimeout: id => timers.delete(id),
    setTimeout: (callback, delay) => { const id = ++sequence; timers.set(id, { callback, delay }); return id; },
    fetch: () => new Promise(() => {}),
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../control_dashboard/static/people.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../control_dashboard/static/app.js"), "utf8"), context);
  return {
    context, timers, elements, revoked, element,
    run: expression => vm.runInContext(expression, context),
    advance: milliseconds => { clock += milliseconds; },
    tick: async () => {
      const [id, timer] = timers.entries().next().value;
      timers.delete(id);
      await timer.callback();
    },
  };
}

test("both meters smooth real input, decay on idle, and label disconnects", () => {
  const app = dashboard();
  for (const kind of ["microphone", "speaker"]) {
    app.run(`updateLevelWave('${kind}', 0.8); animateAudioMeters(performance.now())`);
    app.advance(65);
    app.run("animateAudioMeters(performance.now())");
    const value = app.run(`audioMeters.${kind}.value`);
    assert.ok(value > 0 && value < 0.8);
    app.run(`updateLevelWave('${kind}', 0)`);
    app.advance(100);
    app.run("animateAudioMeters(performance.now())");
    assert.ok(app.run(`audioMeters.${kind}.value`) < value);
    app.advance(2000);
    app.run("animateAudioMeters(performance.now())");
    assert.equal(app.element(`${kind}-signal`).textContent, "Telemetry disconnected");
  }
});

test("audio failures cannot escape into speech or queue overlapping requests", async () => {
  const app = dashboard();
  let calls = 0;
  let reject;
  app.context.fetch = () => { calls++; return new Promise((_, failure) => { reject = failure; }); };
  const pending = app.run("updatePhysicalTelemetry()");
  await app.run("updatePhysicalTelemetry()");
  assert.equal(calls, 1);
  reject(new Error("offline"));
  await pending;
  assert.equal(app.run("audioPolling"), false);
  assert.equal(app.run("audioMeters.microphone.received"), 0);
});

test("preview keeps a single current request and decodes before display", async () => {
  const app = dashboard();
  let resolve;
  let calls = 0;
  app.context.fetch = () => { calls++; return new Promise(done => { resolve = done; }); };
  app.run("schedulePhysicalCamera()");
  const pending = app.tick();
  assert.equal(calls, 1);
  assert.equal(app.element("camera-img").src, "");
  resolve({ ok: true, blob: async () => ({ size: 10000 }) });
  await pending;
  assert.match(app.element("camera-img").src, /^blob:/);
  assert.equal(app.timers.size, 1);
  assert.equal([...app.timers.values()][0].delay, 200);
});

test("old preview responses cannot display or restart after navigation", async () => {
  const app = dashboard();
  let resolve;
  let signal;
  app.context.fetch = (_, options) => { signal = options.signal; return new Promise(done => { resolve = done; }); };
  app.run("schedulePhysicalCamera()");
  const pending = app.tick();
  app.context.location.hash = "#/people";
  app.run("stopPhysicalCamera()");
  assert.equal(signal.aborted, true);
  resolve({ ok: true, blob: async () => ({ size: 10000 }) });
  await pending;
  assert.equal(app.element("camera-img").src, "");
  assert.equal(app.timers.size, 0);
});

test("camera failure backs off and then reconnects to the latest frame", async () => {
  const app = dashboard();
  app.context.fetch = async () => ({ ok: false, json: async () => ({ error: "Camera unavailable" }) });
  app.run("schedulePhysicalCamera()");
  await app.tick();
  assert.equal(app.element("camera-img").hidden, true);
  assert.ok([...app.timers.values()][0].delay >= 1000);
  app.context.fetch = async () => ({ ok: true, blob: async () => ({ size: 10000 }) });
  await app.tick();
  assert.equal(app.element("camera-img").hidden, false);
  assert.equal([...app.timers.values()][0].delay, 200);
  app.run("stopPhysicalCamera()");
  assert.equal(app.revoked.length, 1);
});

test("hidden pages do not fetch camera frames", async () => {
  const app = dashboard();
  let calls = 0;
  app.context.fetch = () => { calls++; };
  app.context.document.hidden = true;
  app.run("schedulePhysicalCamera()");
  await app.tick();
  assert.equal(calls, 0);
  assert.equal(app.timers.size, 0);
});

test("photo selections clear references and release every preview URL", () => {
  const app = dashboard();
  app.run("peopleUI.files = [{name:'one.jpg'}, {name:'two.png'}]; peopleUI.urls = ['blob:1','blob:2']; clearPeoplePhotos()");
  assert.equal(app.run("peopleUI.files.length"), 0);
  assert.equal(app.run("peopleUI.urls.length"), 0);
  assert.deepEqual(app.revoked, ["blob:1", "blob:2"]);
});

test("upload guidance rejects excess files before reading them", () => {
  const app = dashboard();
  app.run("selectPeoplePhotos(Array(11).fill({name:'photo.png',size:100,type:'image/png'}))");
  assert.equal(app.run("peopleUI.files.length"), 0);
  assert.match(app.element("people-message").textContent, /up to 10/);
});

test("upload guidance accounts for base64 body expansion", () => {
  const app = dashboard();
  // 36 MiB binary would pass a naive 32 MiB check if split... use one file under 8 MiB that encodes over budget with helpers.
  // Four 8 MiB files = 32 MiB binary ≈ 42.7 MiB base64; still under 48. Use sizes that push encoded estimate over 48 MiB.
  const almostMax = Math.floor(8 * 1024 * 1024);
  app.run(`selectPeoplePhotos([
    {name:'a.jpg',size:${almostMax},type:'image/jpeg'},
    {name:'b.jpg',size:${almostMax},type:'image/jpeg'},
    {name:'c.jpg',size:${almostMax},type:'image/jpeg'},
    {name:'d.jpg',size:${almostMax},type:'image/jpeg'},
    {name:'e.jpg',size:${Math.floor(7.5 * 1024 * 1024)},type:'image/jpeg'}
  ])`);
  assert.equal(app.run("peopleUI.files.length"), 0);
  assert.match(app.element("people-message").textContent, /encoded upload limit|32 MiB combined/);
});

test("rejected upload controls are re-enabled for correction", async () => {
  const app = dashboard();
  const removeButton = { disabled: true };
  app.context.document.querySelectorAll = selector => selector === "[data-remove-photo]" ? [removeButton] : [];
  app.context.FormData = class { *[Symbol.iterator]() { yield ["name", "Test"]; } };
  app.context.fetch = async () => ({ ok: false, json: async () => ({ persisted: false, error: "Rejected" }) });
  app.run("peopleUI.mode = 'edit'");
  await app.run("savePerson({preventDefault(){},currentTarget:{}})");
  assert.equal(removeButton.disabled, false);
  assert.equal(app.element("people-fields").disabled, false);
  assert.equal(app.element("people-message").textContent, "Rejected");
});

test("network failure shows a concrete People API message instead of raw Failed to fetch", async () => {
  const app = dashboard();
  app.context.document.querySelectorAll = () => [];
  app.context.FormData = class { *[Symbol.iterator]() { yield ["name", "Probe"]; } };
  app.context.fetch = async () => { throw new TypeError("Failed to fetch"); };
  app.run("peopleUI.mode = 'edit'");
  await app.run("savePerson({preventDefault(){},currentTarget:{}})");
  assert.match(app.element("people-message").textContent, /People API|Restart the local dashboard/);
});

test("missing People route returns a restart hint instead of a JSON parse crash", async () => {
  const app = dashboard();
  app.context.document.querySelectorAll = () => [];
  app.context.FormData = class { *[Symbol.iterator]() { yield ["name", "Probe"]; } };
  app.context.fetch = async () => ({ ok: false, status: 404, json: async () => { throw new SyntaxError("Unexpected end"); } });
  app.run("peopleUI.mode = 'edit'");
  await app.run("savePerson({preventDefault(){},currentTarget:{}})");
  assert.match(app.element("people-message").textContent, /People API is missing|Restart the local dashboard/);
});

test("zero microphone level still counts as connected telemetry", () => {
  const app = dashboard();
  app.run("updateLevelWave('microphone', 0); animateAudioMeters(performance.now())");
  assert.ok(app.run("audioMeters.microphone.received") > 0);
  assert.equal(app.element("microphone-signal").textContent, "Idle");
});

test("audio polling resumes after leaving and returning to Physical", async () => {
  const app = dashboard();
  let calls = 0;
  app.context.fetch = async () => {
    calls += 1;
    return {
      ok: true,
      json: async () => ({
        microphone_level: 0.4,
        speaker_level: 0.1,
        microphone_status: "listening",
        speaker_status: "ready",
      }),
    };
  };
  app.context.location.hash = "#/physical";
  await app.run("updatePhysicalTelemetry()");
  assert.equal(calls, 1);
  app.context.location.hash = "#/people";
  await app.run("updatePhysicalTelemetry()");
  assert.equal(calls, 1);
  app.context.location.hash = "#/physical";
  await app.run("updatePhysicalTelemetry()");
  assert.equal(calls, 2);
  assert.equal(app.run("audioMeters.microphone.target"), 0.4);
});

test("commercial dashboard views render from registered service status", () => {
  const app = dashboard();
  app.run(`lastStatus = {
    readiness: {ready: false, label: "SYSTEM NOT READY", reason: "LLM offline"},
    auto_restart: true,
    services: [
      {id:"reachy_daemon",name:"Reachy Mini",group:"system",status:"offline",summary:"Not connected",port:8000,depends_on:[],details:{environment:"physical"},last_health_at:"12:00"},
      {id:"conversation",name:"Conversation",group:"system",status:"offline",summary:"Stopped",port:7860,depends_on:["reachy_daemon"]},
      {id:"speech",name:"Speech",group:"ai",status:"offline",summary:"Stopped",port:8765,depends_on:[]},
      {id:"llama",name:"LLM",group:"ai",status:"offline",summary:"Stopped",port:8080,depends_on:[]},
      {id:"home_assistant",name:"Home Assistant",group:"tools",status:"not_configured",summary:"Not configured",depends_on:[]},
      {id:"apex",name:"Apex",group:"tools",status:"online",summary:"Healthy",depends_on:[]},
      {id:"bus",name:"Bus",group:"tools",status:"online",summary:"Healthy",depends_on:["home_assistant"]}
    ]
  }`);
  app.run("renderOverview(lastStatus)");
  assert.match(app.element("view").innerHTML, /System overview/);
  assert.match(app.element("view").innerHTML, /Reachy needs attention/);
  app.run("renderAIStack(lastStatus)");
  assert.match(app.element("view").innerHTML, /SERVICE TOPOLOGY/);
  assert.match(app.element("view").innerHTML, /Depends on reachy_daemon/);
  app.run("renderTools(lastStatus)");
  assert.match(app.element("view").innerHTML, /Tools &amp; integrations/);
  assert.match(app.element("view").innerHTML, /Not configured/);
});

test("logs clear is presentation-only", () => {
  const source = fs.readFileSync(path.join(__dirname, "../control_dashboard/static/app.js"), "utf8");
  const handler = source.match(/getElementById\("clear-events"\)\.addEventListener\("click",[\s\S]*?\n}\);/);
  assert.ok(handler);
  assert.doesNotMatch(handler[0], /api\/events\/clear/);
});
