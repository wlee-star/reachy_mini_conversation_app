"""Real health checks for each registered service."""

from __future__ import annotations
import time
import shutil
import logging
import subprocess
from typing import Any, Callable
from dataclasses import field, dataclass
from urllib.parse import quote, urlsplit

from control_dashboard import net, proc
from control_dashboard.registry import ServiceSpec, DashboardConfig


logger = logging.getLogger(__name__)

STATUS_ONLINE = "online"
STATUS_STARTING = "starting"
STATUS_DEGRADED = "degraded"
STATUS_OFFLINE = "offline"
STATUS_NOT_CONFIGURED = "not_configured"
DEFAULT_BUS_ENTITY_ID = "sensor.route_311_at_rockwall_cres"

CheckFn = Callable[[ServiceSpec, DashboardConfig, dict[str, Any]], "HealthResult"]


def speech_pool_has_idle_slot(payload: dict[str, Any]) -> bool:
    """Return whether speech-to-speech `/v1/pool` has a claimable unit."""
    units = payload.get("units")
    if isinstance(units, list) and units:
        return any(isinstance(unit, dict) and unit.get("state") == "idle" for unit in units)
    in_use = payload.get("in_use")
    size = payload.get("size")
    if isinstance(in_use, int) and isinstance(size, int):
        return in_use < size
    return False


def speech_pool_is_stuck(payload: dict[str, Any]) -> bool:
    """Return whether any pipeline unit is quarantined and unclaimable."""
    units = payload.get("units")
    if not isinstance(units, list):
        return False
    return any(isinstance(unit, dict) and unit.get("state") == "stuck" for unit in units)


@dataclass
class HealthResult:
    """Plain-English health result for one service."""

    status: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    technical: str | None = None
    suggested_action: str | None = None


_GPU_CACHE: tuple[float, str | None] | None = None


def gpu_name() -> str | None:
    """Return the first NVIDIA GPU name when nvidia-smi is available."""
    global _GPU_CACHE
    now = time.time()
    if _GPU_CACHE is not None and now - _GPU_CACHE[0] < 30:
        return _GPU_CACHE[1]
    if shutil.which("nvidia-smi") is None:
        _GPU_CACHE = (now, None)
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        _GPU_CACHE = (now, None)
        return None
    name = output.strip().splitlines()[0].strip() if output.strip() else None
    _GPU_CACHE = (now, name)
    return name


def _process_snapshot(spec: ServiceSpec, *, include_command: bool = False) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"pids": [], "command": None, "name": None}
    if spec.port is None:
        return snapshot
    pids = proc.listening_pids(spec.port)
    snapshot["pids"] = pids
    if not pids or not include_command:
        return snapshot
    command = proc.process_command_line(pids[0])
    snapshot["command"] = command
    snapshot["name"] = proc.process_name(pids[0])
    if spec.process_match and command and not proc.command_matches(command, spec.process_match):
        snapshot["match"] = False
    else:
        snapshot["match"] = True if command else None
    return snapshot


def check_llama(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check llama.cpp process, port, API, and loaded model."""
    host = spec.host or "127.0.0.1"
    port = spec.port or 8080
    if not net.port_open(host, port):
        return HealthResult(
            STATUS_OFFLINE,
            f"llama.cpp is not listening on {host}:{port}.",
            details={"port": port, "gpu": gpu_name()},
            suggested_action="Start llama.cpp",
            technical="TCP connection refused",
        )
    health = net.http_request(f"http://{host}:{port}/health", timeout_s=5.0)
    models = net.http_request(f"http://{host}:{port}/v1/models", timeout_s=5.0)
    model_id = None
    payload = net.json_payload(models)
    if payload is not None:
        data = payload.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            model_id = data[0].get("id")
    process = _process_snapshot(spec)
    if extra.get("probe"):
        probe = net.http_request(
            f"http://{host}:{port}/v1/chat/completions",
            method="POST",
            json_body={
                "model": model_id or "local",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout_s=20.0,
        )
        extra["probe_result"] = {"ok": probe.ok, "latency_ms": round(probe.latency_ms, 1), "error": probe.error}
    if health.ok and models.ok:
        return HealthResult(
            STATUS_ONLINE,
            "llama.cpp API is responding and a model is loaded.",
            details={
                "port": port,
                "model": model_id,
                "api": True,
                "model_loaded": True,
                "gpu": gpu_name(),
                "process": process,
            },
            latency_ms=round(models.latency_ms, 1),
        )
    if health.status_code == 503:
        return HealthResult(
            STATUS_STARTING,
            "llama.cpp is loading a model.",
            details={"port": port, "gpu": gpu_name(), "process": process},
            technical=health.body or health.error,
        )
    return HealthResult(
        STATUS_DEGRADED,
        f"Something is listening on port {port}, but the llama.cpp API is not healthy.",
        details={"port": port, "model": model_id, "gpu": gpu_name(), "process": process},
        technical=health.error or models.error,
        suggested_action="Restart llama.cpp",
        latency_ms=round(health.latency_ms, 1),
    )


def check_speech(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check the speech-to-speech realtime port and process."""
    host = spec.host or "127.0.0.1"
    port = spec.port or 8765
    env_url = (config.env.get("HF_REALTIME_WS_URL") or "").strip()
    if env_url:
        parsed = urlsplit(env_url)
        host = parsed.hostname or host
        port = parsed.port or port
    if not net.port_open(host, port):
        return HealthResult(
            STATUS_OFFLINE,
            f"Speech-to-speech is not listening on {host}:{port}.",
            details={"port": port, "ws_url": env_url or f"ws://{host}:{port}/v1/realtime"},
            suggested_action="Start speech-to-speech",
            technical="TCP connection refused",
        )
    process = _process_snapshot(spec, include_command=True)
    http = net.http_request(f"http://{host}:{port}/v1/models", timeout_s=3.0)
    tts = "qwen3" in (process.get("command") or "").lower()
    details = {
        "port": port,
        "ws_url": env_url or f"ws://{host}:{port}/v1/realtime",
        "stt": "parakeet-tdt" if "parakeet" in (process.get("command") or "").lower() else None,
        "tts": "qwen3" if tts else None,
        "process": process,
        "http_api": http.ok,
    }
    if process.get("match") is False:
        return HealthResult(
            STATUS_DEGRADED,
            f"Port {port} is open, but the listener does not look like speech-to-speech.",
            details=details,
            technical=process.get("command"),
        )
    pool_result = net.http_request(f"http://{host}:{port}/v1/pool", timeout_s=3.0)
    pool = net.json_payload(pool_result)
    if pool is not None:
        details["pool"] = {"size": pool.get("size"), "in_use": pool.get("in_use"), "units": pool.get("units")}
        if speech_pool_is_stuck(pool):
            return HealthResult(
                STATUS_DEGRADED,
                "Speech-to-speech is running, but its realtime session slot is stuck.",
                details=details,
                suggested_action="Restart speech-to-speech",
                technical="GET /v1/pool state=stuck",
                latency_ms=round(pool_result.latency_ms, 1),
            )
    return HealthResult(
        STATUS_ONLINE,
        "Speech-to-speech realtime port is accepting connections.",
        details=details,
        latency_ms=round(http.latency_ms, 1) if http.ok else None,
    )


def check_qwen_tts(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Treat Qwen TTS as healthy when speech-to-speech is running with --tts qwen3."""
    speech = config.service("speech")
    if speech is None:
        return HealthResult(STATUS_NOT_CONFIGURED, "Speech-to-speech is not in the registry.")
    speech_health = check_speech(speech, config, extra)
    command = str((speech_health.details.get("process") or {}).get("command") or "")
    if speech_health.status == STATUS_OFFLINE:
        return HealthResult(
            STATUS_OFFLINE,
            "Qwen TTS is bundled with speech-to-speech, which is offline.",
            details={"bundled_with": "speech"},
            suggested_action="Start speech-to-speech",
        )
    if "--tts" in command.lower() and "qwen3" not in command.lower():
        return HealthResult(
            STATUS_DEGRADED,
            "Speech-to-speech is running, but not with --tts qwen3.",
            details={"bundled_with": "speech", "command_tts": True},
            technical=command,
        )
    return HealthResult(
        STATUS_ONLINE,
        "Qwen TTS is enabled inside speech-to-speech.",
        details={"bundled_with": "speech", "backend": "torch/cuda or mlx", "standalone": False},
    )


def check_hermes(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check Hermes API reachability without sending a chat completion."""
    missing = spec.missing_needs(config.env)
    gateway = (config.env.get("HERMES_GATEWAY_URL") or "").strip()
    api_key = (config.env.get("HERMES_API_KEY") or "").strip()
    if missing:
        return HealthResult(
            STATUS_NOT_CONFIGURED,
            "Hermes is not configured in the conversation app .env.",
            details={"missing": missing, "provider": None},
        )
    parsed = urlsplit(gateway)
    host = parsed.hostname or spec.host or "127.0.0.1"
    port = parsed.port or spec.port or 8642
    if not net.port_open(host, port):
        return HealthResult(
            STATUS_OFFLINE,
            f"Hermes API is not listening on {host}:{port}.",
            details={"port": port, "provider": "hermes-agent", "configured": True},
            suggested_action="Start Hermes",
            technical="TCP connection refused",
        )
    models_url = gateway.replace("/v1/chat/completions", "/v1/models")
    if models_url.endswith("/v1/chat/completions"):
        models_url = gateway
    result = net.http_request(
        models_url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout_s=5.0,
    )
    details = {
        "port": port,
        "provider": "hermes-agent",
        "api": result.ok or result.status_code in {401, 403, 404},
        "authentication": result.ok or result.status_code not in {401, 403},
        "home_assistant": bool((config.env.get("HA_URL") or "").strip()),
        "apex": bool((config.env.get("APEX_STATUS_URL") or "").strip()),
    }
    if result.status_code in {401, 403}:
        return HealthResult(
            STATUS_DEGRADED,
            "Hermes is reachable, but the API key was rejected.",
            details=details,
            technical=f"HTTP {result.status_code}",
            latency_ms=round(result.latency_ms, 1),
        )
    if result.ok or result.status_code == 404:
        return HealthResult(
            STATUS_ONLINE,
            "Hermes API is reachable.",
            details=details,
            latency_ms=round(result.latency_ms, 1),
        )
    return HealthResult(
        STATUS_DEGRADED,
        "Hermes port is open, but the API did not respond as expected.",
        details=details,
        technical=result.error,
        latency_ms=round(result.latency_ms, 1),
    )


def check_conversation(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check the conversation app UI port and process."""
    host = spec.host or "127.0.0.1"
    port = spec.port or 7860
    daemon_host = _configured_daemon_host(config)
    if daemon_host and not _is_loopback_host(host) and host == daemon_host:
        daemon_port = int(config.env.get("REACHY_DAEMON_PORT") or 8000)
        current = net.http_request(
            f"http://{daemon_host}:{daemon_port}/api/apps/current-app-status",
            timeout_s=4.0,
        )
        payload = net.json_payload(current) or {}
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        running = payload.get("state") == "running" and info.get("name") == "reachy_mini_conversation_app"
        if running:
            return HealthResult(
                STATUS_ONLINE,
                "Conversation app is running on the physical Reachy Mini.",
                details={"host": host, "managed_by_daemon": True, "state": "running"},
                latency_ms=round(current.latency_ms, 1),
            )
        if current.ok:
            return HealthResult(
                STATUS_STARTING if payload.get("state") == "starting" else STATUS_OFFLINE,
                f"Conversation app on physical Reachy Mini is {payload.get('state') or 'stopped'}.",
                details={"host": host, "managed_by_daemon": True, "state": payload.get("state")},
                suggested_action="Start conversation app",
                technical=str(payload.get("error") or "") or None,
            )
    process = _process_snapshot(spec, include_command=True)
    ui_up = net.port_open(host, port)
    if ui_up:
        if process.get("match") is False:
            occupant = process.get("command") or process.get("name") or f"pid {process.get('pids')}"
            return HealthResult(
                STATUS_DEGRADED,
                f"Port {port} is open, but the listener is not the conversation app.",
                details={"port": port, "ui": False, "process": process, "occupant": occupant},
                technical=str(occupant),
                suggested_action="Stop the process occupying port 7860, then start the conversation app",
            )
        page = net.http_request(f"http://{host}:{port}/", timeout_s=4.0)
        return HealthResult(
            STATUS_ONLINE if page.ok or page.status_code else STATUS_DEGRADED,
            "Conversation app web UI is responding."
            if page.ok or page.status_code
            else "UI port is open but did not return HTTP.",
            details={"port": port, "ui": True, "process": process},
            latency_ms=round(page.latency_ms, 1),
            technical=None if page.ok or page.status_code else page.error,
        )
    return HealthResult(
        STATUS_OFFLINE,
        f"Conversation app UI is not listening on {host}:{port}.",
        details={"port": port, "ui": False, "process": process},
        suggested_action="Start conversation app",
        technical="TCP connection refused",
    )


def _is_loopback_host(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    return name in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or name.startswith("127.")


def _configured_daemon_host(config: DashboardConfig) -> str | None:
    raw = (config.env.get("REACHY_DAEMON_HOST") or config.env.get("REACHY_DAEMON_URL") or "").strip()
    if not raw:
        return None
    if "://" in raw:
        parsed = urlsplit(raw)
        return parsed.hostname
    return raw.split(":", 1)[0] or None


def _daemon_environment(payload: dict[str, Any]) -> str:
    if payload.get("simulation_enabled") is True:
        return "simulator"
    if payload.get("wireless_version") is True or payload.get("desktop_app_daemon") is True:
        return "physical"
    if payload.get("simulation_enabled") is False:
        return "physical"
    return "unknown"


def _daemon_state_value(payload: dict[str, Any]) -> str | None:
    state = payload.get("state") if payload.get("state") is not None else payload.get("status")
    if state is None:
        return None
    return str(state)


def _local_sim_listener(spec: ServiceSpec) -> int | None:
    if spec.port is None or not spec.process_match:
        return None
    for pid in proc.listening_pids(spec.port):
        if proc.pid_matches_pattern(pid, spec.process_match):
            return pid
    return None


def _probe_daemon_http(host: str, port: int) -> HealthResult:
    timeout_s = 0.4 if _is_loopback_host(host) else 1.2
    if not net.host_resolves(host):
        return HealthResult(
            STATUS_OFFLINE,
            f"{host} did not resolve",
            details={"unresolved": True, "host": host, "port": port},
            technical=f"{host} did not resolve",
        )
    if not net.port_open(host, port, timeout_s=timeout_s):
        return HealthResult(
            STATUS_OFFLINE,
            f"{host}:{port} refused TCP",
            details={"refused": True, "host": host, "port": port},
            technical=f"{host}:{port} refused TCP",
        )
    result = net.http_request(f"http://{host}:{port}/api/daemon/status", timeout_s=4.0)
    payload = net.json_payload(result) or {}
    environment = _daemon_environment(payload) if payload else "unknown"
    details = {
        "host": host,
        "port": port,
        "sdk_daemon": bool(result.ok),
        "environment": environment,
        "simulation_enabled": payload.get("simulation_enabled"),
        "wireless_version": payload.get("wireless_version"),
        "desktop_app_daemon": payload.get("desktop_app_daemon"),
        "state": _daemon_state_value(payload),
    }
    if result.ok:
        return HealthResult(
            STATUS_ONLINE,
            f"Reachy Mini daemon is reachable at {host}:{port}.",
            details={**details, "robot_reachable": True},
            latency_ms=round(result.latency_ms, 1),
        )
    if result.status_code:
        return HealthResult(
            STATUS_DEGRADED,
            f"Daemon HTTP at {host}:{port} responded with {result.status_code}.",
            details={**details, "sdk_daemon": True, "robot_reachable": False},
            technical=result.error or f"HTTP {result.status_code}",
            latency_ms=round(result.latency_ms, 1),
        )
    return HealthResult(
        STATUS_OFFLINE,
        f"{host}:{port} refused TCP",
        details={"refused": True, "host": host, "port": port},
        technical=result.error or f"{host}:{port} refused TCP",
    )


def _accept_local_simulator(spec: ServiceSpec, probed: HealthResult) -> HealthResult | None:
    """Return a local-sim health result, or None if this answer is not the simulator."""
    if probed.status not in {STATUS_ONLINE, STATUS_DEGRADED}:
        return None
    environment = str(probed.details.get("environment") or "unknown")
    if environment == "simulator" or (environment == "unknown" and _local_sim_listener(spec) is not None):
        details = {
            **probed.details,
            "environment": "simulator",
            "sdk_daemon": True,
            "robot_reachable": True,
            "process": _process_snapshot(spec, include_command=True),
        }
        state = probed.details.get("state")
        if state is not None and str(state).lower() in {"stopped", "stopping", "error"}:
            return HealthResult(
                STATUS_DEGRADED,
                f"Local Reachy Mini simulator is answering, but daemon state is {state}.",
                details=details,
                technical=f"state={state}",
                latency_ms=probed.latency_ms,
                suggested_action="Restart Reachy Mini",
            )
        return HealthResult(
            STATUS_ONLINE,
            "Local Reachy Mini simulator is answering GET /api/daemon/status.",
            details=details,
            latency_ms=probed.latency_ms,
        )
    if environment == "physical":
        return HealthResult(
            STATUS_OFFLINE,
            "Port 8000 is a physical Reachy Mini daemon, not the local MuJoCo simulator.",
            details={
                **probed.details,
                "sdk_daemon": True,
                "robot_reachable": False,
                "process": _process_snapshot(spec, include_command=True),
            },
            technical="simulation_enabled=false",
            suggested_action="Close the Reachy Mini desktop app, then start the simulator from this dashboard.",
            latency_ms=probed.latency_ms,
        )
    if probed.status in {STATUS_ONLINE, STATUS_DEGRADED}:
        return HealthResult(
            STATUS_OFFLINE,
            "Something answered /api/daemon/status on 127.0.0.1:8000, but it is not the MuJoCo simulator.",
            details={
                **probed.details,
                "sdk_daemon": False,
                "robot_reachable": False,
                "process": _process_snapshot(spec, include_command=True),
            },
            technical=probed.technical,
            suggested_action="Close the process on port 8000 if it is not reachy-mini-daemon --sim.",
            latency_ms=probed.latency_ms,
        )
    return None


def check_reachy_daemon(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check the local Reachy Mini simulator, or an explicit REACHY_DAEMON_HOST."""
    port = spec.port or 8000
    configured = _configured_daemon_host(config)
    if configured and not _is_loopback_host(configured):
        probed = _probe_daemon_http(configured, port)
        if probed.status in {STATUS_ONLINE, STATUS_DEGRADED}:
            details = {**probed.details, "robot_reachable": probed.status == STATUS_ONLINE}
            return HealthResult(
                probed.status,
                probed.summary,
                details=details,
                latency_ms=probed.latency_ms,
                technical=probed.technical,
            )
        return HealthResult(
            STATUS_OFFLINE,
            (
                f"No Reachy Mini daemon is answering at {configured}:{port}. "
                "Check REACHY_DAEMON_HOST, or clear it to use the virtual simulator."
            ),
            details={
                "port": port,
                "sdk_daemon": False,
                "robot_reachable": False,
                "tried_hosts": [configured],
                "environment": "physical",
            },
            technical=probed.technical or probed.summary,
            suggested_action="Check REACHY_DAEMON_HOST, or clear it to use the virtual simulator",
        )

    probed = _probe_daemon_http("127.0.0.1", port)
    accepted = _accept_local_simulator(spec, probed)
    if accepted is not None:
        return accepted

    return HealthResult(
        STATUS_OFFLINE,
        (
            "The local Reachy Mini simulator is not listening on 127.0.0.1:8000. "
            "Use Start on this card (or Start all) to launch the virtual Reachy Mini simulator."
        ),
        details={
            "port": port,
            "sdk_daemon": False,
            "robot_reachable": False,
            "tried_hosts": ["127.0.0.1"],
            "environment": None,
            "process": _process_snapshot(spec, include_command=True),
        },
        technical=probed.technical or probed.summary,
        suggested_action="Start the virtual Reachy Mini simulator",
    )


def check_home_assistant(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check Home Assistant /api/ without controlling devices."""
    missing = spec.missing_needs(config.env)
    if missing:
        return HealthResult(
            STATUS_NOT_CONFIGURED,
            "Home Assistant is not configured in .env.",
            details={"missing": missing},
        )
    base = (config.env.get("HA_URL") or "").strip().rstrip("/")
    token = (config.env.get("HA_TOKEN") or "").strip()
    result = net.http_request(
        f"{base}/api/",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout_s=5.0,
    )
    if result.status_code in {401, 403}:
        return HealthResult(
            STATUS_DEGRADED,
            "Home Assistant is reachable, but authentication failed.",
            details={"api": True, "authentication": False},
            technical=f"HTTP {result.status_code}",
            latency_ms=round(result.latency_ms, 1),
        )
    if result.ok:
        return HealthResult(
            STATUS_ONLINE,
            "Home Assistant API accepted the configured token.",
            details={"api": True, "authentication": True, "url_host": urlsplit(base).hostname},
            latency_ms=round(result.latency_ms, 1),
        )
    return HealthResult(
        STATUS_OFFLINE,
        "Home Assistant API is not responding.",
        details={"api": False, "authentication": False, "url_host": urlsplit(base).hostname},
        technical=result.error,
        suggested_action="Check HA_URL and that Home Assistant is on the LAN",
        latency_ms=round(result.latency_ms, 1),
    )


def check_apex(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Read-only GET of the Apex /status URL."""
    missing = spec.missing_needs(config.env)
    url = (config.env.get("APEX_STATUS_URL") or "").strip()
    if missing or not url:
        return HealthResult(
            STATUS_NOT_CONFIGURED,
            "APEX_STATUS_URL is not set.",
            details={"missing": missing or ["APEX_STATUS_URL"]},
        )
    result = net.http_request(url, timeout_s=5.0)
    payload = net.json_payload(result)
    if result.ok and payload is not None:
        return HealthResult(
            STATUS_ONLINE,
            "Apex status endpoint returned valid JSON.",
            details={
                "url_host": urlsplit(url).hostname,
                "has_probes": "probes" in payload,
                "controller": payload.get("controller")
                if not isinstance(payload.get("controller"), dict)
                else payload["controller"].get("hostname"),
            },
            latency_ms=round(result.latency_ms, 1),
        )
    if result.ok:
        return HealthResult(
            STATUS_DEGRADED,
            "Apex responded, but the body was not JSON.",
            details={"url_host": urlsplit(url).hostname},
            technical=result.body[:300],
            latency_ms=round(result.latency_ms, 1),
        )
    return HealthResult(
        STATUS_OFFLINE,
        "Apex status endpoint is not reachable.",
        details={"url_host": urlsplit(url).hostname},
        technical=result.error,
        suggested_action="Check APEX_STATUS_URL",
        latency_ms=round(result.latency_ms, 1),
    )


def check_bus(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Read the Home Assistant bus arrival sensor without writing state."""
    entity = (config.env.get("HA_BUS_ENTITY_ID") or "").strip() or DEFAULT_BUS_ENTITY_ID
    ha = check_home_assistant(
        config.service("home_assistant") or spec,
        config,
        extra,
    )
    if ha.status in {STATUS_NOT_CONFIGURED, STATUS_OFFLINE}:
        return HealthResult(
            ha.status,
            "Bus arrivals are read from Home Assistant, which is not available.",
            details={"entity_id": entity, "via": "home_assistant"},
            technical=ha.technical,
            suggested_action=ha.suggested_action,
        )
    base = (config.env.get("HA_URL") or "").strip().rstrip("/")
    token = (config.env.get("HA_TOKEN") or "").strip()
    result = net.http_request(
        f"{base}/api/states/{quote(entity, safe='')}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout_s=5.0,
    )
    payload = net.json_payload(result)
    if result.ok and payload is not None:
        return HealthResult(
            STATUS_ONLINE,
            "Bus arrival sensor is readable from Home Assistant.",
            details={"entity_id": entity, "state": payload.get("state"), "via": "home_assistant"},
            latency_ms=round(result.latency_ms, 1),
        )
    if result.status_code == 404:
        return HealthResult(
            STATUS_DEGRADED,
            "Home Assistant is up, but the Route 311 sensor was not found.",
            details={"entity_id": entity, "via": "home_assistant"},
            technical="HTTP 404",
        )
    return HealthResult(
        STATUS_OFFLINE,
        "Could not read the bus arrival sensor.",
        details={"entity_id": entity, "via": "home_assistant"},
        technical=result.error or f"HTTP {result.status_code}",
        latency_ms=round(result.latency_ms, 1),
    )


_CHECKS: dict[str, CheckFn] = {
    "llama": check_llama,
    "speech": check_speech,
    "qwen_tts": check_qwen_tts,
    "hermes": check_hermes,
    "conversation": check_conversation,
    "reachy_daemon": check_reachy_daemon,
    "home_assistant": check_home_assistant,
    "apex": check_apex,
    "bus": check_bus,
}


def check_service(spec: ServiceSpec, config: DashboardConfig, *, probe: bool = False) -> HealthResult:
    """Run the registered health check for one service."""
    checker = _CHECKS.get(spec.health)
    extra: dict[str, Any] = {"probe": probe}
    if checker is None:
        if spec.port and spec.host and net.port_open(spec.host, spec.port):
            return HealthResult(STATUS_ONLINE, f"{spec.name} port is open.", details={"port": spec.port})
        return HealthResult(STATUS_OFFLINE, f"{spec.name} is offline.")
    try:
        result = checker(spec, config, extra)
    except Exception as exc:
        logger.warning("Health check failed for %s: %s", spec.id, exc)
        return HealthResult(
            STATUS_OFFLINE,
            f"{spec.name} health check failed.",
            technical=str(exc),
        )
    if extra.get("probe_result"):
        result.details["probe"] = extra["probe_result"]
    return result


def diagnose(spec: ServiceSpec, result: HealthResult, config: DashboardConfig) -> str:
    """Return a short plain-English explanation for a failed service."""
    if result.status == STATUS_ONLINE:
        return result.summary
    if result.status == STATUS_NOT_CONFIGURED:
        return result.summary
    if spec.id == "conversation" and any(dep == "speech" for dep in spec.depends_on):
        speech = config.service("speech")
        if speech is not None:
            speech_result = check_speech(speech, config, {})
            if speech_result.status == STATUS_OFFLINE:
                return (
                    f"The conversation app expects speech-to-speech on port {speech.port}, "
                    "but nothing is currently responding."
                )
    return result.summary
