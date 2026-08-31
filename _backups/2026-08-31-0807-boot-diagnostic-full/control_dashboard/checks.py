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
    process = _process_snapshot(spec)
    ui_up = net.port_open(host, port)
    if ui_up:
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
    # Console-only: look for a matching process without the UI port.
    if spec.process_match:
        # Port snapshot is empty; we cannot cheaply scan all PIDs. Treat as offline.
        pass
    return HealthResult(
        STATUS_OFFLINE,
        f"Conversation app UI is not listening on {host}:{port}.",
        details={"port": port, "ui": False, "process": process},
        suggested_action="Start conversation app",
        technical="TCP connection refused",
    )


_MDNS_CACHE: tuple[float, list[tuple[str, int]]] | None = None
_MDNS_TTL_S = 20.0


def _configured_daemon_host(config: DashboardConfig) -> str | None:
    raw = (config.env.get("REACHY_DAEMON_HOST") or config.env.get("REACHY_DAEMON_URL") or "").strip()
    if not raw:
        return None
    if "://" in raw:
        parsed = urlsplit(raw)
        return parsed.hostname
    return raw.split(":", 1)[0] or None


def _mdns_daemon_targets() -> list[tuple[str, int]]:
    """Discover wireless/LAN daemons via the Reachy Mini SDK mDNS helper."""
    global _MDNS_CACHE
    now = time.time()
    if _MDNS_CACHE is not None and now - _MDNS_CACHE[0] < _MDNS_TTL_S:
        return _MDNS_CACHE[1]
    targets: list[tuple[str, int]] = []
    try:
        from reachy_mini.utils.discovery import find_robots
    except ImportError:
        _MDNS_CACHE = (now, targets)
        return targets
    try:
        robots = find_robots(timeout=2.0)
    except Exception as exc:
        logger.warning("Reachy mDNS discovery failed: %s", exc)
        _MDNS_CACHE = (now, targets)
        return targets
    seen: set[tuple[str, int]] = set()
    for robot in robots:
        port = int(getattr(robot, "port", None) or 8000)
        candidates = [getattr(robot, "host", None), *list(getattr(robot, "addresses", None) or [])]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            item = (candidate.strip(), port)
            if item not in seen:
                seen.add(item)
                targets.append(item)
    _MDNS_CACHE = (now, targets)
    return targets


def _probe_daemon_http(host: str, port: int) -> HealthResult | None:
    timeout_s = 0.4 if host in {"127.0.0.1", "localhost"} else 1.2
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
    if result.ok:
        payload = net.json_payload(result) or {}
        return HealthResult(
            STATUS_ONLINE,
            f"Reachy Mini daemon is reachable at {host}:{port}.",
            details={
                "host": host,
                "port": port,
                "sdk_daemon": True,
                "robot_reachable": True,
                "state": payload.get("state") or payload.get("status") or payload,
            },
            latency_ms=round(result.latency_ms, 1),
        )
    if result.status_code:
        return HealthResult(
            STATUS_DEGRADED,
            f"Daemon HTTP at {host}:{port} responded with {result.status_code}.",
            details={"host": host, "port": port, "sdk_daemon": True, "robot_reachable": False},
            technical=result.error or f"HTTP {result.status_code}",
            latency_ms=round(result.latency_ms, 1),
        )
    return HealthResult(
        STATUS_OFFLINE,
        f"{host}:{port} refused TCP",
        details={"refused": True, "host": host, "port": port},
        technical=result.error or f"{host}:{port} refused TCP",
    )


def check_reachy_daemon(spec: ServiceSpec, config: DashboardConfig, extra: dict[str, Any]) -> HealthResult:
    """Check the Reachy Mini daemon status endpoint without moving motors."""
    port = spec.port or 8000
    hosts: list[str] = []
    for candidate in (spec.host, "127.0.0.1", "localhost", _configured_daemon_host(config), *spec.fallback_hosts):
        if candidate and candidate not in hosts:
            hosts.append(candidate)

    last_error = "no hosts tried"
    unresolved: list[str] = []
    refused: list[str] = []
    tried = list(hosts)

    def consider(host: str, probe_port: int) -> HealthResult | None:
        nonlocal last_error
        probed = _probe_daemon_http(host, probe_port)
        if probed.status in {STATUS_ONLINE, STATUS_DEGRADED}:
            return probed
        last_error = probed.technical or probed.summary
        if probed.details.get("unresolved"):
            unresolved.append(host)
        else:
            refused.append(f"{host}:{probe_port}")
        return None

    for host in hosts:
        matched = consider(host, port)
        if matched is not None:
            return matched

    for mdns_host, mdns_port in _mdns_daemon_targets():
        if mdns_host not in tried:
            tried.append(mdns_host)
        matched = consider(mdns_host, mdns_port)
        if matched is not None:
            return matched

    bits = ["No Reachy Mini daemon is answering on this PC or the LAN."]
    if "127.0.0.1:8000" in refused or "localhost:8000" in refused:
        bits.append("Nothing is listening on localhost:8000 (Reachy Mini Control / USB daemon is not running).")
    if "reachy-mini.local" in unresolved:
        bits.append("Windows could not resolve reachy-mini.local; that is common without working mDNS.")
    if not _mdns_daemon_targets():
        bits.append("No robot advertised itself on the LAN via mDNS.")
    bits.append(
        "Use Start on this card (or Start all) to launch the virtual Reachy Mini simulator, "
        "or set REACHY_DAEMON_HOST to a physical robot's Wi-Fi IP."
    )
    return HealthResult(
        STATUS_OFFLINE,
        " ".join(bits),
        details={
            "port": port,
            "sdk_daemon": False,
            "robot_reachable": False,
            "tried_hosts": tried,
            "unresolved": unresolved,
            "refused": refused,
        },
        technical=last_error,
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
