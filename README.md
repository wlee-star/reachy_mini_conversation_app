---
title: Reachy Mini Conversation App
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: static
pinned: false
short_description: Talk with Reachy Mini!
suggested_storage: large
tags:
 - reachy_mini
 - reachy_mini_python_app
---

# Reachy Mini conversation app

Conversational app for the Reachy Mini robot combining realtime voice, vision, personality-aware tools, and choreographed motion.

![Reachy Mini Dance](docs/assets/reachy_mini_dance.gif)

## Table of contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the app](#running-the-app)
- [Local AI control dashboard](#local-ai-control-dashboard)
- [LLM tools](#llm-tools-exposed-to-the-assistant)
- [Creating and adding tools](#creating-and-adding-tools)
- [Advanced features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Overview

- Low-latency audio conversation through the Hugging Face realtime backend, using the built-in server or a local endpoint.
- Vision is handled by the realtime backend when the `camera` tool is used.
- Layered motion system queues primary moves (dances, emotions, goto poses, breathing) while blending speech-reactive wobble.
- Async tools integrate motion, camera capture, and MCP Tool Spaces. The optional web UI (`--ui`) manages conversations, personalities, tools, and settings.

## Architecture

The app connects the user, AI services, and robot hardware:

<p align="center">
  <img src="docs/assets/conversation_app_arch.svg" alt="Architecture Diagram" width="600"/>
</p>

## Installation

> [!IMPORTANT]
> Install [Reachy Mini's SDK](https://github.com/pollen-robotics/reachy_mini/) before using this app.<br>
> Windows support is currently experimental and has not been extensively tested. Use with caution.

<details open>
<summary>Using uv (recommended)</summary>

Set up with [uv](https://docs.astral.sh/uv/):

```bash
# macOS (Homebrew)
uv venv --python /opt/homebrew/bin/python3.12 .venv

# Linux / Windows (Python in PATH)
uv venv --python python3.12 .venv

source .venv/bin/activate
uv sync
```

Include dev dependencies:
```bash
uv sync --group dev
```

</details>

> [!NOTE]
> Run `uv sync --frozen` to install the exact dependency set from `uv.lock` without re-resolving versions.

<details>
<summary>Using pip</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install dev dependencies:
```bash
pip install -e .[dev]                   # Development tools
```

</details>

## Configuration

This checkout is intended to run conversation against a **local** Hugging Face [speech-to-speech](https://github.com/huggingface/speech-to-speech) server on your AI PC. `deployed` mode still exists as a rollback, but it sends audio and transcripts to Hugging Face cloud inference.

Copy `.env.example` to `.env`. The example file selects local mode and `ws://127.0.0.1:8765/v1/realtime`.

| Variable | Description |
|----------|-------------|
| `REALTIME_TRANSCRIPTION_LANGUAGE` | Optional input transcription language for the realtime backend. Defaults to `en`; set to a backend-supported code such as `zh` for Chinese. |
| `HF_REALTIME_CONNECTION_MODE` | `local` uses `HF_REALTIME_WS_URL` on your AI PC. `deployed` uses the built-in Hugging Face cloud server. Application default remains `deployed` if unset; this project's `.env.example` sets `local`. |
| `HF_REALTIME_WS_URL` | Direct websocket endpoint for speech-to-speech. Accepts `ws://127.0.0.1:8765/v1` or `ws://127.0.0.1:8765/v1/realtime`. Used when `HF_REALTIME_CONNECTION_MODE=local`. |
| `HF_TOKEN` | Optional token for Hugging Face Hub downloads and private Space tools. Local realtime endpoints receive only this explicitly configured token. Not used for cloud conversation when mode is `local`. |
| `HERMES_CONFIG_PATH` | Optional path to a Hermes Agent `config.yaml` / `mcp_servers.yaml`. HTTP MCP servers listed there can be imported into `installed_local_mcp.json`. |
| `HERMES_GATEWAY_URL` | Optional Hermes Agent API base for advanced delegated tasks. For a local Hermes gateway this is `http://127.0.0.1:8642/v1/chat/completions`. A host:port value such as `http://127.0.0.1:8642` is normalized to that chat-completions path. When set with `HERMES_API_KEY`, the `ask_hermes` tool POSTs OpenAI chat-completions (`{"model","messages"}`) and reads `choices[0].message.content`. Reef trend/history queries call Hermes live first and attach the latest `~/reef-monitor/reef_thread.jsonl` report as context. If Hermes is unavailable or exceeds `HERMES_REEF_REQUEST_TIMEOUT_SECONDS`, the cached report is still returned with `stale=true` and `source=cache`. Session continuity uses header `X-Hermes-Session-Id`. A second `ask_hermes` while one is in flight returns the cache immediately (or a controlled error if no cache), instead of queueing behind the in-flight request. |
| `HERMES_API_KEY` | Bearer token sent as `Authorization: Bearer …`. Must match Hermes `API_SERVER_KEY`. Do not commit a real key. |
| `HERMES_REQUEST_TIMEOUT_SECONDS` | Live-wait timeout for non-Reef Hermes delegated tasks. Defaults to `180`. |
| `HERMES_REEF_REQUEST_TIMEOUT_SECONDS` | Live-wait timeout for interactive Reef trend/history checks. Defaults to `15`. If Hermes exceeds this, the validated `reef_thread.jsonl` cache is returned rather than waiting approximately two minutes. |
| `HERMES_CIRCUIT_FAILURE_THRESHOLD` | Consecutive Hermes failures before the client stops calling the gateway for a cooldown. Defaults to `2`. Does not disable Home Assistant, bus, reef cache, or robot movement. |
| `HERMES_CIRCUIT_COOLDOWN_SECONDS` | How long an open Hermes circuit stays fail-fast before one probe request. Defaults to `60`. |
| `HA_URL` | Optional local Home Assistant base URL, for example `http://homeassistant.local:8123`. Used by the local `home_assistant` tool for simple state reads, light on/off, and scene activation without Hermes. |
| `HA_TOKEN` | Home Assistant long-lived access token for the local `home_assistant` tool. Do not commit a real token. |
| `HA_BUS_ENTITY_ID` | Optional Home Assistant entity for bus arrivals. Defaults to `sensor.route_311_at_rockwall_cres`. |
| `APEX_STATUS_URL` | Optional Neptune Apex Fusion status URL, for example `http://192.168.0.143:8080/status`. Used by the local `apex` and `reef_status` tools. Live Apex data is `source=live` and `stale=false`. If unset or unreachable, they fall back to `~/reef-monitor/reef_cache.json` with `stale=true` and `source=cache`. |
| `REEF_CACHE_MAX_AGE_SECONDS` | Optional freshness threshold for Reef/Hermes cache fallbacks. Defaults to `3600`. Cache younger than this is `status=degraded`; older cache is `status=stale`. Cached data is never returned as `status=success`. |
| `REACHY_MINI_APP_TIMEOUT_MINUTES` | Minutes of inactivity before Reachy goes to sleep and the app stops. Defaults to `1440` (one day); set to `0` to disable. |

### Hugging Face Connection Modes

Use the built-in Hugging Face server through the app-managed Space proxy only as a rollback. It is cloud inference:

```env
HF_REALTIME_CONNECTION_MODE=deployed
```

Deployed session allocation falls back to cached `hf auth login` credentials and reports the daemon-provided hardware ID when available. Cached credentials and the hardware ID are not sent to local endpoints.

### Local AI PC (RTX 3090 / later Mac mini)

Run inference **outside** this app's virtualenv. On Windows, prefer WSL2 Ubuntu or Docker so Qwen3-TTS CUDA wheels match the speech-to-speech docs. Keep the Reachy daemon and this conversation app on native Windows.

Start llama.cpp on port `8080`, then speech-to-speech on port `8765`, pointing the LLM at loopback only:

```bash
llama-server -hf ggml-org/gemma-4-E4B-it-GGUF -np 2 -c 65536 -fa on --swa-full --host 127.0.0.1 --port 8080
speech-to-speech --mode realtime --stt parakeet-tdt --tts qwen3 --llm_backend responses-api --model_name "ggml-org/gemma-4-E4B-it-GGUF" --responses_api_base_url "http://127.0.0.1:8080/v1" --responses_api_api_key "" --ws_host 127.0.0.1 --ws_port 8765 --qwen3_tts_backend torch --qwen3_tts_device cuda
```

Never leave speech-to-speech on its default hosted OpenAI model. For Wireless Reachy, bind speech-to-speech with `--host 0.0.0.0` and allow inbound TCP 8765 on the LAN firewall only. Do not publish port 8080.

Speech-to-speech accepts **one** realtime websocket by default (`--num_pipelines` is 1). A second conversation app, or a reconnect while the previous session is still draining, is rejected with `All session slots are in use`. Stop extra clients and wait a few seconds, or restart speech-to-speech if `http://127.0.0.1:8765/v1/pool` shows a stuck unit.

Companion start/stop scripts live in the sibling `reachy-mini-local-ai` folder next to this checkout. Apple Silicon later uses the same `HF_REALTIME_WS_URL` and `speech-to-speech serve --mac-optimal-settings`.

`docs/scheme.mmd` now mentions local MCP. Regenerate `docs/assets/conversation_app_arch.svg` from that source before publishing docs.

Run your own realtime voice backend using [speech-to-speech](https://github.com/huggingface/speech-to-speech) on the same machine as the conversation app:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

Run your own Hugging Face backend on your laptop and connect to it from Reachy Mini Wireless over the same Wi-Fi network:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://<your-laptop-lan-ip>:8765/v1/realtime
```

For that LAN setup, make sure the backend listens on an address reachable from the robot, not only on `127.0.0.1`.

If the backend stays bound to loopback on your laptop, you can forward it into the robot over SSH instead:

```bash
ssh -N -R 8765:127.0.0.1:8765 <robot-user>@<robot-host>
```

Then set this on the robot:

```env
HF_REALTIME_CONNECTION_MODE=local
HF_REALTIME_WS_URL=ws://127.0.0.1:8765/v1/realtime
```

In the web UI's Settings view, the Connection section lets you choose either the built-in server or a local `host:port` target. The UI writes `HF_REALTIME_CONNECTION_MODE` for you, and the local path writes `HF_REALTIME_WS_URL` with a default of `localhost:8765`.

## Running the app

Activate your virtual environment, then launch:

```bash
reachy-mini-conversation-app
```

> [!TIP]
> Starting the app launches `reachy-mini-daemon --sim` only when nothing is already answering on `localhost:8000`. If you start from the control dashboard, leave the Reachy Mini card running and start only the conversation app — do not start a second simulator. Pass `--no-sim` to skip launch, or set `REACHY_DAEMON_HOST` to a physical robot's Wi-Fi IP.

The app runs in console mode. Add `--ui` to serve the web interface at http://127.0.0.1:7860/.

On each application boot, Reachy greets Walter using `Australia/Sydney` local time, announces startup diagnostics, checks the AI stack and the connected Reachy Mini daemon (simulator or physical robot), then speaks a short readiness report. The spoken sequence runs once per process and does not repeat on realtime reconnects. Optional integrations that are unset are reported as not configured, not as failures.

### Local AI control dashboard

To see whether llama.cpp, speech-to-speech, Hermes, Home Assistant, Apex, and this app are actually healthy — and to start or stop the **managed** pieces without asking Cursor each time — run:

```bash
python -m control_dashboard
```

Then open http://127.0.0.1:8788/. The dashboard is a separate orchestration layer. It does not rewrite this conversation app, move motors, or send device-control commands during health checks.

| Service | Purpose | Startup | Port | Dependencies | Health check | Stop |
|---------|---------|---------|------|--------------|--------------|------|
| llama.cpp | Local LLM for speech-to-speech | `llama-server` (model from `control_dashboard/services.json`, default Gemma GGUF) | 8080 | — | TCP + `/health` + `/v1/models` | Only a whitelisted `llama-server` on that port |
| Speech-to-speech | Realtime STT/TTS websocket | Companion `reachy-mini-local-ai` venv (`python -m speech_to_speech.s2s_pipeline`) | 8765 | llama.cpp | TCP + process match + `/v1/pool` | Whitelisted `speech-to-speech` / `speech_to_speech` |
| Qwen TTS | Bundled `--tts qwen3` | Not a separate process | — | speech | Speech running with `--tts qwen3` | Stop speech |
| Hermes | Delegated agent API | `hermes gateway start` | 8642 | `HERMES_GATEWAY_URL` + `HERMES_API_KEY` | TCP + `/v1/models` | `hermes gateway stop` |
| Conversation app | This application | `python -m reachy_mini_conversation_app.main --no-camera --no-sim --ui` | 7860 | llama, speech, Reachy daemon | HTTP `/` | Whitelisted conversation process |
| Reachy Mini daemon | Virtual SDK daemon for local testing | `reachy-mini-daemon --sim` (MuJoCo window) | 8000 | — | GET `/api/daemon/status` on 127.0.0.1:8000 with `simulation_enabled` (simulator only unless `REACHY_DAEMON_HOST` is set) | Whitelisted `reachy-mini-daemon --sim` |
| Home Assistant | Lights, scenes, bus sensor | External | from `HA_URL` | `HA_URL` + `HA_TOKEN` | GET `/api/` only | Not stopped here |
| Neptune Apex | Reef status | External | from `APEX_STATUS_URL` | URL | GET status JSON | Not stopped here |
| Bus API | Bus arrivals | HA entity from `HA_BUS_ENTITY_ID` or `sensor.route_311_at_rockwall_cres` | via HA | Home Assistant | GET entity state | Not stopped here |

**Start all** starts only missing managed services, in dependency order, and skips anything already healthy. Already-running whitelist matches that this dashboard did not start are reported as **external process** and are not claimed. It waits until a service is up, not merely starting, before launching dependents. Auto-restart gives up after three failed recoveries, does not restart a service the user stopped, and does not keep retrying or flooding the event log. **Stop all** stops managed whitelist matches, including a leftover local `reachy-mini-daemon --sim`. It does not stop Home Assistant or Apex, and it will not kill an unrelated process on port 8000 (for example the Reachy Mini desktop app or a physical robot on the LAN). Reachy Mini is started as the SDK **simulator** (`--sim`, with the MuJoCo window and media). A physical robot answering on `reachy-mini.local` or a LAN IP does not count as this card being healthy.

On Windows, `reachy-mini.local` often does not resolve. If you want the physical robot instead of the simulator, set `REACHY_DAEMON_HOST` in `.env` to its Wi-Fi IP and leave the sim stopped.

Service definitions: `control_dashboard/services.json`. Machine-specific overrides: gitignored `control_dashboard/services.local.json`. Add a service by appending a registry object and, if needed, a probe in `control_dashboard/checks.py`.

Do not enable Windows logon auto-start yet. The start-all path is reusable for a later Scheduled Task after wake. For a future Mac mini, keep bind addresses in the registry instead of hard-coding `C:\` or a GPU name.

Assumptions: the default llama command matches this README (`ggml-org/gemma-4-E4B-it-GGUF`); companion `start-local-ai.ps1` still uses a local Qwen3 GGUF if you override `start.args`. Bus data is the HA sensor, not a direct Transport for NSW client. MCP stubs on 8760/8751/8752/8740 are unused by the current local tools and are not shown.

### CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--no-camera` | `False` | Run without camera capture. |
| `--ui` | `False` | Serve the web UI at http://127.0.0.1:7860/, in addition to console mode. |
| `--no-sim` | `False` | Do not launch the MuJoCo simulator. Use this to connect to an existing daemon or a physical robot. |
| `--robot-name` | `None` | Optional. Connect to a specific robot by name when running multiple daemons on the same subnet. See [Multiple robots on the same subnet](#advanced-features). |
| `--debug` | `False` | Enable verbose logging for troubleshooting. |

### Examples

```bash
# Audio-only conversation (no camera)
reachy-mini-conversation-app --no-camera

# Launch with the minimal web UI for personality/mic/settings control
reachy-mini-conversation-app --ui

# Connect to a physical robot (do not start the MuJoCo simulator)
reachy-mini-conversation-app --no-sim
```

## LLM tools exposed to the assistant

The default profile exposes these tools. Use Tools → Tool access to customize any profile.
Every bundled profile enables `head_tracking` by default; users can still disable it per personality.

| Tool | Action | Dependencies |
|------|--------|--------------|
| `dance` | Queue a dance from `reachy_mini_dances_library`. | Core install only. |
| `stop_dance` | Clear queued dances. | Core install only. |
| `play_emotion` | Play a recorded emotion clip via Hugging Face datasets. | Core install only. Uses the default open emotions dataset: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library). |
| `stop_emotion` | Clear queued emotions. | Core install only. |
| `camera` | Capture the latest camera frame and analyze it with the selected realtime backend. | Core install only. Requires the camera (disable with `--no-camera`). |
| `idle_do_nothing` | Explicitly remain idle during an idle turn. Not intended for normal conversation turns. | Core install only. |
| `move_head` | Queue a head pose change (left/right/up/down/front). | Core install only. |
| `head_tracking` | Follow the user's face with the head, or stop following. | Core install only. Requires a daemon with the `vision` extra and a camera. |
| `go_to_sleep` | Run Reachy's sleep movement and stop the current app after an explicit user request. | Core install only. |
| `sweep_look` | Sweep Reachy's head left, right, and back to center. | Shared tool, enabled by default in the default profile. |
| `remember` | Save one short, stable fact about the user for future sessions. | Core install only. Stored in the app instance data directory. |
| `forget` | Remove a saved memory fact by matching a short query. | Core install only. |
| `home_assistant` | Read Home Assistant entity state, turn lights on/off, activate scenes, and read bus arrivals directly over the local LAN. A confirmed control result plays the existing `success` emotion and a short spoken confirmation. | Set `HA_URL` and `HA_TOKEN`; optionally set `HA_BUS_ENTITY_ID`. |
| `monitor_bus` | Query the live Home Assistant Route 311 sensor (current and following services), then optionally watch that specific service in the background for 15/10/7/5-minute and arrival alerts. The 10-minute alert also queues Reachy's official `helpful1` emotion once per watched service. Switching or continuous watches require an explicit user request. Arrival times are interpreted in `Australia/Sydney`. Does not add a second bus API. Active watches persist in `bus_monitors.v1.json`. | Same Home Assistant config as `home_assistant`. |
| `apex` | Read current Neptune Apex / reef status from `APEX_STATUS_URL` (`/status` JSON) for water parameters, equipment, alarms, and alerts. Current reef stats, status, readings, pH, temperature, and ATO questions call this tool immediately and speak the raw probe values. Report, trend, and analysis questions do not use this path. Live reads are `source=live`; cache fallback is `stale=true`. | Set `APEX_STATUS_URL`. Falls back to `~/reef-monitor/reef_cache.json`. |
| `reef_status` | Legacy fast-path reef status reader; same live `/status` URL or cache as `apex`. | Set `APEX_STATUS_URL`, or keep the reef cache producer. |
| `ask_hermes` | Forward advanced delegated tasks to the Hermes Gateway, such as other buses/trains (not live Route 311), research, multi-step household tasks, or reef report/trend/analysis. Live Hermes results are `source=live`. If Hermes is unavailable or exceeds the Reef live-wait timeout, the Reefy `reef_thread.jsonl` cache is returned with `stale=true` and `source=cache`. Direct `apex__*` / `home_assistant__*` MCP tools are not registered while this tool is on. | Set `HERMES_GATEWAY_URL` and `HERMES_API_KEY`. Optional `HERMES_REEF_REQUEST_TIMEOUT_SECONDS` (default 15). |

Weather, web search, and time are no longer enabled on the default profile. The bundled Hugging Face Tool Spaces remain installable from Tools if you accept cloud MCP calls. For other local HTTP MCP servers (time, weather), register them in `external_content/installed_local_mcp.json` and enable the `{alias}__{tool}` IDs per personality. Simple Apex and Home Assistant operations use local Python tools; Hermes remains available through `ask_hermes` for advanced delegation.

> [!NOTE]
> `remember`/`forget` facts are stored in `memory.v1.json` inside the app's instance data directory (`~/.local/share/reachy_mini_conversation_app/` by default, or the instance path used by the desktop launcher). `forget` only removes facts matched by query. To reset all remembered facts, delete this file. Active Route 311 watches persist in `bus_monitors.v1.json` in the same directory.

## Creating and adding tools

Tools can run locally as Python code, as a LAN/local HTTP MCP server, or remotely in an MCP-compatible Hugging Face Space. Keep robot, camera, and deterministic local-data operations in local tools. Simple Apex and Home Assistant requests use the bundled local Python tools; reserve `ask_hermes` for advanced reasoning, research, buses/trains, and multi-step tasks. A Space is a fallback for shareable cloud services and is not used for normal local conversation.

### Local tools

Create one Python module per tool, with the file name matching the tool's unique `name`. See [`idle_do_nothing.py`](src/reachy_mini_conversation_app/tools/idle_do_nothing.py) for a minimal implementation.

Each tool subclasses `Tool` and defines `name`, a model-facing `description`, an object-shaped JSON Schema in `parameters_schema`, and an async `__call__` method. Use `ToolDependencies` for runtime services, and set `needs_response = False` for actions that should not trigger a spoken follow-up. Catch expected operational failures, log them with the module logger, and return `{"error": "..."}` so the conversation can continue.

Restart the app after adding the module. Use Tools → Tool access to enable it for a personality, or add its name to that profile's `default_tools` in `profile.md`. See [External profiles and tools](#external-profiles-and-tools) for external directories and autoload behavior.

### Hugging Face Space tools

To publish a remote tool, create a Gradio Space, expose its API as MCP with `mcp_server=True`, and give each function clear type hints and docstrings. Verify that `https://<space-subdomain>.hf.space/gradio_api/mcp/schema` lists the expected tools before installing the Space.

Use the maintained [weather](https://huggingface.co/spaces/pollen-robotics/reachy-mini-weather-tool), [time](https://huggingface.co/spaces/pollen-robotics/reachy-mini-time-tool), and [search](https://huggingface.co/spaces/pollen-robotics/reachy-mini-search-tool) Spaces as examples. See Gradio's [MCP server guide](https://www.gradio.app/guides/building-mcp-server-with-gradio) for additional publishing guidance and [Installing Hugging Face Space tools](#installing-hugging-face-space-tools) for this app's installation steps.

## Advanced features

Built-in motion content is published as open Hugging Face datasets:

- Emotions: [`pollen-robotics/reachy-mini-emotions-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-emotions-library)
- Dances: [`pollen-robotics/reachy-mini-dances-library`](https://huggingface.co/datasets/pollen-robotics/reachy-mini-dances-library)

Spoken requests for Reachy to dance queue the official `dance1` emotion from the emotions library immediately; conversation continues as usual.

<details>
<summary>Custom profiles</summary>

Create custom profiles with dedicated instructions and per-profile tool access.

Select and save a startup profile in the UI. The choice is stored in `startup_settings.json`. Before one is saved, `REACHY_MINI_CUSTOM_PROFILE=<name>` can select `profiles/<name>/`; otherwise the app uses `default`.

Every profile directory contains one strict schema-version-1 `profile.md`. TOML metadata is enclosed by `+++`; the remaining Markdown body is the realtime assistant prompt:

```markdown
+++
schema_version = 1
voice = "Aiden"
greeting = "Greet me warmly in one sentence, in character, and vary the wording each time."
hidden = false
default_tools = [
  "dance",
  "camera",
  "sweep_look",
]
+++

## Identity

You are a concise, friendly robot guide.
```

`schema_version`, `default_tools`, and a non-empty Markdown body are required. `voice`, `greeting`, and `hidden` are optional. Set `hidden = true` to omit a profile from the UI. An empty `default_tools` list is valid and inherits nothing.

`default_tools` is the authored baseline. Tools → Tool access stores overrides in instance-local `profile_toolsets.json` without changing bundled profiles. Restoring defaults removes the override. Active-profile changes reconnect the conversation; other changes apply when selected.

Profile directories are data-only. Python tool implementations belong in `src/reachy_mini_conversation_app/tools/`, or in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY` for external tools. Each enabled tool ID must resolve to a shared tool, an external tool, or a tool from an installed Hugging Face Space.

See [Creating and adding tools](#creating-and-adding-tools) for the local tool interface and a maintained example.

To manage personalities in the UI:

With `--ui`, Home lists the available profiles and the built-in default:

- Tap a card to apply that personality and start talking.
- Tap "Manage tools" on a saved personality to open its tool access directly.
- Tap "Custom" to create a personality with a name, instructions, and optional greeting. It inherits the default tools, which can be changed under "Manage tools". Managed instances store it at `user_personalities/<name>/profile.md`; standalone runs use `external_content/user_personalities/<name>/profile.md`.

Switching a personality reloads its prompt and effective tools through a quick backend reconnect. Editing `profile.md` directly requires re-selecting the profile or restarting the app.

</details>

<details>
<summary>Locked profile mode</summary>

To create a locked variant of the app that cannot switch profiles, edit `src/reachy_mini_conversation_app/config.py` and set the `LOCKED_PROFILE` constant to the desired profile name:
```python
LOCKED_PROFILE: str | None = "mars_rover"  # Lock to this profile
```
When set, the app ignores saved startup settings, `REACHY_MINI_CUSTOM_PROFILE`, and UI selection. The UI marks the profile as locked and disables editing.

</details>

<a id="external-profiles-and-tools"></a>

<details>
<summary>External profiles and tools</summary>

You can extend the app with profiles/tools stored outside the repository defaults.

- Core profiles are under `profiles/`.
- Core tools are under `src/reachy_mini_conversation_app/tools/`.

Recommended layout:

```text
external_content/
├── external_profiles/
│   └── my_profile/
│       └── profile.md
├── external_tools/
│   └── my_custom_tool.py
├── user_personalities/
│   └── my_custom_profile/
│       └── profile.md
├── installed_tool_spaces.json
└── profile_toolsets.json
```

Environment variables:

Set these values in your `.env` when you want env-driven external profile/tool selection:

```env
# Optional fallback/manual profile selector:
REACHY_MINI_CUSTOM_PROFILE=my_profile
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=./external_content/external_profiles
REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=./external_content/external_tools
# Optional convenience mode:
# AUTOLOAD_EXTERNAL_TOOLS=1
```

Loading rules:

- Profiles: each directory requires a schema-version-1 `profile.md` with explicit `default_tools`; there is no cross-profile fallback.
- Default mode: enabled IDs must resolve to a shared, external, or installed Tool Space tool.
- Autoload: `AUTOLOAD_EXTERNAL_TOOLS=1` adds every valid `*.py` module from `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- Web UI: Tools → Tool access enables external modules per profile; it does not upload or edit Python.
- Separation: profile directories contain data only; external Python belongs in `REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY`.
- Tool names: every loaded class needs a unique `Tool.name`; duplicates fail fast.

</details>

<a id="installing-hugging-face-space-tools"></a>

<details>
<summary>Installing Hugging Face Space tools</summary>

You can install MCP-compatible Hugging Face Spaces as remote tool sources for this app. Private Spaces work too, as long as `HF_TOKEN` is set (or you have run `hf auth login`) for an account that can access them. To publish a new Space, follow [Creating and adding tools](#hugging-face-space-tools).

Tools → Tool Spaces installs or refreshes a global source. Its tools then appear under Tools → Tool access for per-profile selection. Removing a Space removes its tools from every profile. Active-profile changes reconnect the conversation; other changes apply when selected.

The app accepts Hugging Face Spaces exposing the standard `/gradio_api/mcp/` endpoint, not arbitrary MCP URLs. Installation discovers the Space's tools and assigns namespaced local IDs, so do not guess or hard-code those IDs beforehand.

```bash
# install + enable in active profile
reachy-mini-conversation-app tool-spaces add <owner/space-name>

# enable in a specific profile
reachy-mini-conversation-app tool-spaces add <owner/space-name> --profile NAME

# install without enabling
reachy-mini-conversation-app tool-spaces add <owner/space-name> --install-only

# list installed spaces
reachy-mini-conversation-app tool-spaces list

# remove an installed space
reachy-mini-conversation-app tool-spaces remove owner/space-name
```

Bundled Pollen Spaces use static specs and are enabled by the default profile. Custom Spaces are validated through the Hugging Face Hub; HF tokens are sent only to private Spaces. Tool metadata is cached in:

- `installed_tool_spaces.json` in the managed app instance directory
- `external_content/installed_tool_spaces.json` in terminal mode

Startup and profile switching read this cache without discovery or MCP probing. Network access occurs only during install, refresh, or remote tool calls. Per-profile access is stored in `profile_toolsets.json` beside the manifest, or under `external_content/` in terminal mode.

Recommended tags for discoverability on Hugging Face:

- `reachy-mini-tool`
- `mcp`

Tags are advisory; installation still requires successful MCP validation.

> [!NOTE]
> Preinstalled Pollen Spaces can be removed like any other (`tool-spaces remove pollen-robotics/reachy-mini-weather-tool`). To restore access, reinstall the Space and restore or update the relevant profile under "Tool access".

</details>

<details>
<summary>Multiple robots on the same subnet</summary>

If you run multiple Reachy Mini daemons on the same network, use:

```bash
reachy-mini-conversation-app --robot-name <name>
```

`<name>` must match the daemon's `--robot-name` value so the app connects to the correct robot.

</details>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [`AGENTS.md`](AGENTS.md) for coding-agent standards.

## License

Apache 2.0
