"""Physical Reachy Mini adapter for the control dashboard.

Isolates physical-robot observation and control from the simulator stack.
All motor/audio/camera commands go through the conversation app's existing
media pipeline (HTTP on :7860) — this module never opens a second SDK client.
"""

from __future__ import annotations
import time
import logging
import threading
from typing import Any
from dataclasses import dataclass
from urllib.parse import urlsplit

from control_dashboard import net, checks, events
from control_dashboard.checks import HealthResult
from control_dashboard.registry import ServiceSpec, DashboardConfig


logger = logging.getLogger(__name__)

TARGET_PHYSICAL = "physical"
TARGET_SIMULATOR = "simulator"
TARGET_UNKNOWN = "unknown"

DAEMON_PORT = 8000
CONVERSATION_PORT = 7860

_CAMERA_PREVIEW_ENABLED = True
_CAMERA_LOCK = threading.Lock()
_CAMERA_LAST_ATTEMPT = 0.0
_CAMERA_BACKOFF_S = 1.0
_CAMERA_MIN_INTERVAL_S = 0.2
_CAMERA_MAX_BACKOFF_S = 8.0


class CommandBlocked(Exception):
    """Raised when a physical command fails a safety guard."""

    def __init__(self, reason: str) -> None:
        """Store a stable blocked-command reason for API responses."""
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RobotTarget:
    """Resolved robot target for dashboard commands."""

    kind: str
    host: str | None
    port: int
    simulation_enabled: bool | None
    summary: str


def _is_loopback_host(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    return name in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or name.startswith("127.")


def configured_daemon_host(config: DashboardConfig) -> str | None:
    """Return REACHY_DAEMON_HOST hostname when set."""
    raw = (config.env.get("REACHY_DAEMON_HOST") or config.env.get("REACHY_DAEMON_URL") or "").strip()
    if not raw:
        return None
    if "://" in raw:
        parsed = urlsplit(raw)
        return parsed.hostname
    return raw.split(":", 1)[0] or None


def _kind_from_daemon_probe(probed: HealthResult) -> tuple[str, bool | None]:
    """Map a daemon probe to a target kind; unknown when identity is unclear."""
    environment = str(probed.details.get("environment") or "unknown")
    sim = probed.details.get("simulation_enabled")
    sim_flag = sim if isinstance(sim, bool) else None
    if environment == "simulator" or sim is True:
        return TARGET_SIMULATOR, True if sim is True else sim_flag
    if environment == "physical" or sim is False:
        return TARGET_PHYSICAL, False if sim is False else sim_flag
    return TARGET_UNKNOWN, sim_flag


def resolve_target(config: DashboardConfig) -> RobotTarget:
    """Resolve physical vs simulator from config and a live daemon probe.

    Fail closed: when identity cannot be determined confidently, return UNKNOWN
    and leave physical commands blocked.
    """
    configured = configured_daemon_host(config)
    if configured and not _is_loopback_host(configured):
        probed = checks._probe_daemon_http(configured, DAEMON_PORT)
        if probed.status in {checks.STATUS_ONLINE, checks.STATUS_DEGRADED}:
            kind, sim_flag = _kind_from_daemon_probe(probed)
            summary = probed.summary
            if kind == TARGET_UNKNOWN:
                summary = f"Daemon at {configured} answered, but physical/simulator identity is unclear."
            return RobotTarget(
                kind=kind,
                host=configured,
                port=DAEMON_PORT,
                simulation_enabled=sim_flag,
                summary=summary,
            )
        return RobotTarget(
            kind=TARGET_UNKNOWN,
            host=configured,
            port=DAEMON_PORT,
            simulation_enabled=None,
            summary=f"Configured host {configured} did not answer; target unresolved ({probed.summary}).",
        )

    host = "127.0.0.1"
    probed = checks._probe_daemon_http(host, DAEMON_PORT)
    if probed.status in {checks.STATUS_ONLINE, checks.STATUS_DEGRADED}:
        kind, sim_flag = _kind_from_daemon_probe(probed)
        summary = probed.summary
        if kind == TARGET_UNKNOWN:
            summary = "Local daemon answered, but physical/simulator identity is unclear."
        return RobotTarget(
            kind=kind,
            host=host,
            port=DAEMON_PORT,
            simulation_enabled=sim_flag,
            summary=summary,
        )
    if configured and _is_loopback_host(configured):
        return RobotTarget(
            kind=TARGET_UNKNOWN,
            host=host,
            port=DAEMON_PORT,
            simulation_enabled=None,
            summary="Configured for local daemon; waiting for a clear physical or simulator answer.",
        )
    return RobotTarget(
        kind=TARGET_UNKNOWN,
        host=host,
        port=DAEMON_PORT,
        simulation_enabled=None,
        summary="No Reachy Mini daemon detected on localhost:8000.",
    )


def assert_physical_command(target: RobotTarget, *, require_connected: bool = True) -> None:
    """Block commands that must only run against a physical Reachy Mini."""
    if target.kind != TARGET_PHYSICAL:
        raise CommandBlocked(f"COMMAND BLOCKED: target is {target.kind.upper()}, not PHYSICAL REACHY MINI.")
    if require_connected and target.host is None:
        raise CommandBlocked("COMMAND BLOCKED: physical Reachy Mini host is unknown.")


def conversation_base_url(config: DashboardConfig) -> str:
    """Return the conversation app HTTP base URL used for dashboard media control."""
    spec = config.service("conversation")
    host = (spec.host if spec and spec.host else None) or "127.0.0.1"
    port = (spec.port if spec and spec.port else None) or CONVERSATION_PORT
    return f"http://{host}:{port}"


def _conversation_json(
    config: DashboardConfig,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> tuple[HealthResult, dict[str, Any] | None]:
    url = f"{conversation_base_url(config)}{path}"
    result = net.http_request(url, method=method, json_body=body, timeout_s=timeout_s)
    payload = net.json_payload(result)
    if result.ok and payload is not None:
        return (
            HealthResult(
                checks.STATUS_ONLINE,
                "Conversation dashboard API responded.",
                details=payload,
                latency_ms=round(result.latency_ms, 1),
            ),
            payload,
        )
    if result.status_code == 404:
        return (
            HealthResult(
                checks.STATUS_DEGRADED,
                "Conversation app is up, but the physical dashboard API is missing.",
                technical="HTTP 404",
                suggested_action="Restart the conversation app to load dashboard routes",
                latency_ms=round(result.latency_ms, 1),
            ),
            payload,
        )
    return (
        HealthResult(
            checks.STATUS_OFFLINE,
            "Conversation dashboard API is not available.",
            technical=result.error or f"HTTP {result.status_code}",
            suggested_action="Start the conversation app with --ui",
            latency_ms=round(result.latency_ms, 1),
        ),
        payload,
    )


def _status_label(status: str) -> str:
    mapping = {
        checks.STATUS_ONLINE: "ONLINE",
        checks.STATUS_STARTING: "STARTING",
        checks.STATUS_DEGRADED: "DEGRADED",
        checks.STATUS_OFFLINE: "OFFLINE",
        checks.STATUS_NOT_CONFIGURED: "NOT CONFIGURED",
        "connected": "CONNECTED",
        "error": "ERROR",
        "unknown": "UNKNOWN",
    }
    return mapping.get(status, status.upper())


def _service_row(spec: ServiceSpec | None, result: HealthResult | None, *, fallback_name: str) -> dict[str, Any]:
    if spec is None or result is None:
        return {
            "id": fallback_name,
            "name": fallback_name,
            "status": "unknown",
            "label": "UNKNOWN",
            "summary": "Service not registered.",
        }
    return {
        "id": spec.id,
        "name": spec.name,
        "status": result.status,
        "label": _status_label(result.status),
        "summary": result.summary,
        "latency_ms": result.latency_ms,
        "details": {
            key: value
            for key, value in result.details.items()
            if key not in {"process", "probe"}
            and not str(key).lower().endswith(("token", "key", "password", "secret"))
        },
        "suggested_action": result.suggested_action,
        "technical": result.technical,
    }


def build_physical_status(config: DashboardConfig, health_fn: Any) -> dict[str, Any]:
    """Assemble the physical AI-stack status payload from real health checks."""
    target = resolve_target(config)
    stack_ids = [
        "conversation",
        "speech",
        "hermes",
        "llama",
        "home_assistant",
        "apex",
        "reachy_daemon",
    ]
    rows: list[dict[str, Any]] = []
    results: dict[str, HealthResult] = {}
    for service_id in stack_ids:
        spec = config.service(service_id)
        if spec is None:
            continue
        result = health_fn(spec)
        results[service_id] = result
        display_name = {
            "conversation": "Reachy Conversation",
            "speech": "Speech-to-Speech",
            "hermes": "Hermes",
            "llama": "Local LLM",
            "home_assistant": "Home Assistant",
            "apex": "Reef / Apex",
            "reachy_daemon": "Reachy SDK Daemon",
        }.get(service_id, spec.name)
        row = _service_row(spec, result, fallback_name=display_name)
        row["name"] = display_name
        if service_id == "reachy_daemon":
            env = result.details.get("environment")
            if target.kind == TARGET_PHYSICAL and env == "simulator":
                row["status"] = checks.STATUS_OFFLINE
                row["label"] = "OFFLINE"
                row["summary"] = "Local daemon is the simulator; physical Reachy Mini is not the active target."
            elif target.kind == TARGET_PHYSICAL and result.status == checks.STATUS_ONLINE:
                row["label"] = "CONNECTED"
        rows.append(row)

    media_health, media_payload = _conversation_json(config, "/api/dashboard/status")
    camera_status = "offline"
    mic_status = "offline"
    speaker_status = "offline"
    media: dict[str, Any] = {}
    if media_payload is not None:
        media = media_payload
        camera_status = str(media.get("camera_status") or "unknown")
        mic_status = str(media.get("microphone_status") or "unknown")
        speaker_status = str(media.get("speaker_status") or "unknown")
        rows.extend(
            [
                {
                    "id": "camera",
                    "name": "Camera",
                    "status": "online"
                    if camera_status in {"live", "ready", "connected"}
                    else ("error" if camera_status == "error" else "offline"),
                    "label": "CONNECTED" if camera_status in {"live", "ready", "connected"} else camera_status.upper(),
                    "summary": media.get("camera_summary") or f"Camera is {camera_status}.",
                },
                {
                    "id": "microphone",
                    "name": "Microphone",
                    "status": "online"
                    if mic_status in {"listening", "idle", "muted", "connected"}
                    else ("error" if mic_status == "error" else "offline"),
                    "label": "CONNECTED"
                    if mic_status in {"listening", "idle", "muted", "connected"}
                    else mic_status.upper(),
                    "summary": media.get("microphone_summary") or f"Microphone is {mic_status}.",
                },
                {
                    "id": "speaker",
                    "name": "Speaker",
                    "status": "online"
                    if speaker_status in {"ready", "playing", "muted", "connected"}
                    else ("error" if speaker_status == "error" else "offline"),
                    "label": "CONNECTED"
                    if speaker_status in {"ready", "playing", "muted", "connected"}
                    else speaker_status.upper(),
                    "summary": media.get("speaker_summary") or f"Speaker is {speaker_status}.",
                },
            ]
        )
    else:
        rows.extend(
            [
                {
                    "id": "camera",
                    "name": "Camera",
                    "status": "offline",
                    "label": "OFFLINE",
                    "summary": media_health.summary,
                },
                {
                    "id": "microphone",
                    "name": "Microphone",
                    "status": "offline",
                    "label": "OFFLINE",
                    "summary": media_health.summary,
                },
                {
                    "id": "speaker",
                    "name": "Speaker",
                    "status": "offline",
                    "label": "OFFLINE",
                    "summary": media_health.summary,
                },
            ]
        )

    daemon = results.get("reachy_daemon")
    conversation = results.get("conversation")
    speech = results.get("speech")
    physical_connected = (
        target.kind == TARGET_PHYSICAL and daemon is not None and daemon.status == checks.STATUS_ONLINE
    )
    ai_online = (
        conversation is not None
        and conversation.status == checks.STATUS_ONLINE
        and speech is not None
        and speech.status
        in {
            checks.STATUS_ONLINE,
            checks.STATUS_DEGRADED,
        }
    )
    audio_online = mic_status not in {"offline", "error", "unknown"} and speaker_status not in {
        "offline",
        "error",
        "unknown",
    }

    hermes = results.get("hermes")
    llama = results.get("llama")
    return {
        "target": {
            "kind": target.kind,
            "label": {
                TARGET_PHYSICAL: "PHYSICAL REACHY MINI",
                TARGET_SIMULATOR: "VIRTUAL REACHY MINI",
                TARGET_UNKNOWN: "UNKNOWN TARGET",
            }.get(target.kind, target.kind.upper()),
            "host": target.host,
            "port": target.port,
            "simulation_enabled": target.simulation_enabled,
            "summary": target.summary,
            "assistant_name": (config.env.get("ASSISTANT_NAME") or config.env.get("WAKE_NAME") or "Reachy Mini").strip()
            or "Reachy Mini",
        },
        "banners": {
            "connected": physical_connected,
            "ai_online": ai_online,
            "audio_online": bool(audio_online and media_payload is not None),
        },
        "stack": rows,
        "hermes": {
            "status": hermes.status if hermes else "unknown",
            "label": _status_label(hermes.status) if hermes else "UNKNOWN",
            "summary": hermes.summary if hermes else "Hermes is not registered.",
            "latency_ms": hermes.latency_ms if hermes else None,
            "details": (hermes.details if hermes else {}),
        },
        "local_ai": {
            "status": llama.status if llama else "unknown",
            "label": _status_label(llama.status) if llama else "UNKNOWN",
            "summary": llama.summary if llama else "Local LLM is not registered.",
            "model": (llama.details.get("model") if llama else None),
            "gpu": (llama.details.get("gpu") if llama else None),
            "latency_ms": llama.latency_ms if llama else None,
        },
        "media": media,
        "media_api": {
            "status": media_health.status,
            "summary": media_health.summary,
            "technical": media_health.technical,
        },
        "camera_preview_enabled": camera_preview_enabled(),
        "robot": {
            "connection": "CONNECTED" if physical_connected else "OFFLINE",
            "sdk": "CONNECTED" if physical_connected else "OFFLINE",
            "motors": media.get("motors_status") or ("UNKNOWN" if media_payload else "OFFLINE"),
            "camera": camera_status.upper(),
            "microphone": mic_status.upper(),
            "speaker": speaker_status.upper(),
            "state": media.get("robot_state") or ("IDLE" if physical_connected else "OFFLINE"),
            "wlan_ip": (daemon.details.get("host") if daemon else None) or target.host,
            "daemon_state": (daemon.details.get("state") if daemon else None),
        },
        "safe_stop": {
            "available": bool(media.get("safe_stop_available")),
            "label": "SAFE STOP (motor / torque disable)",
            "summary": media.get("safe_stop_summary")
            or (
                "Stops active moves and disables motors. Does not sleep the robot, "
                "and does not stop Reachy, Hermes, or this dashboard."
                if media.get("safe_stop_available")
                else "Safe stop unavailable until the conversation app dashboard API is online."
            ),
        },
        "camera_preview": {
            "enabled": camera_preview_enabled(),
            "summary": (
                "ON — dashboard is displaying the camera preview."
                if camera_preview_enabled()
                else "OFF — dashboard preview stopped (robot camera not powered down)."
            ),
        },
    }


def camera_preview_enabled() -> bool:
    """Return whether the dashboard is allowed to fetch camera frames."""
    return _CAMERA_PREVIEW_ENABLED


def set_camera_preview_enabled(enabled: bool) -> bool:
    """Enable or disable dashboard camera preview (does not release robot media)."""
    global _CAMERA_PREVIEW_ENABLED, _CAMERA_BACKOFF_S
    with _CAMERA_LOCK:
        _CAMERA_PREVIEW_ENABLED = bool(enabled)
        if enabled:
            _CAMERA_BACKOFF_S = 1.0
        logger.info("Physical camera preview %s", "enabled" if enabled else "disabled")
        events.emit("info", "physical", f"Camera preview {'ON' if enabled else 'OFF'}")
        return _CAMERA_PREVIEW_ENABLED


def fetch_camera_jpeg(config: DashboardConfig) -> tuple[bytes | None, dict[str, Any]]:
    """Fetch one JPEG from the conversation app with backoff; never opens a second camera."""
    global _CAMERA_LAST_ATTEMPT, _CAMERA_BACKOFF_S
    target = resolve_target(config)
    meta: dict[str, Any] = {"target": target.kind, "preview_enabled": camera_preview_enabled()}
    if not camera_preview_enabled():
        meta.update(
            {
                "status": "preview_off",
                "summary": "Camera preview OFF — dashboard stopped requesting frames (robot camera unchanged).",
            }
        )
        return None, meta
    try:
        assert_physical_command(target, require_connected=False)
    except CommandBlocked as exc:
        meta.update({"status": "blocked", "summary": str(exc.reason)})
        return None, meta

    now = time.monotonic()
    with _CAMERA_LOCK:
        wait = _CAMERA_BACKOFF_S if _CAMERA_BACKOFF_S > _CAMERA_MIN_INTERVAL_S else _CAMERA_MIN_INTERVAL_S
        if now - _CAMERA_LAST_ATTEMPT < wait and _CAMERA_BACKOFF_S > _CAMERA_MIN_INTERVAL_S:
            meta.update({"status": "backoff", "summary": "Waiting before camera retry.", "backoff_s": round(wait, 1)})
            return None, meta
        _CAMERA_LAST_ATTEMPT = now

    url = f"{conversation_base_url(config)}/api/dashboard/camera.jpg"
    jpeg = _http_bytes(url, timeout_s=4.0)
    if jpeg:
        with _CAMERA_LOCK:
            _CAMERA_BACKOFF_S = _CAMERA_MIN_INTERVAL_S
        meta.update({"status": "live", "summary": "LIVE", "bytes": len(jpeg)})
        return jpeg, meta

    with _CAMERA_LOCK:
        _CAMERA_BACKOFF_S = min(_CAMERA_MAX_BACKOFF_S, max(_CAMERA_MIN_INTERVAL_S * 2, _CAMERA_BACKOFF_S * 2))
        backoff = _CAMERA_BACKOFF_S
    meta.update(
        {
            "status": "offline",
            "summary": "CAMERA OFFLINE",
            "error": "no jpeg frame",
            "backoff_s": round(backoff, 1),
        }
    )
    logger.info("Physical camera fetch failed (backoff %.1fs)", backoff)
    return None, meta


def _http_bytes(url: str, *, timeout_s: float) -> bytes | None:
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
            if "image/jpeg" in content_type or (data[:2] == b"\xff\xd8"):
                return bytes(data)
            return None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.debug("Binary HTTP fetch failed for %s: %s", url, exc)
        return None


def run_physical_action(config: DashboardConfig, action: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a guarded physical dashboard action via the conversation app."""
    target = resolve_target(config)
    assert_physical_command(target, require_connected=False)
    path_map = {
        "mic": ("/api/dashboard/mic", "POST"),
        "speaker": ("/api/dashboard/speaker", "POST"),
        "speaker_test": ("/api/dashboard/speaker/test", "POST"),
        "safe_stop": ("/api/dashboard/safe-stop", "POST"),
    }
    if action not in path_map:
        raise CommandBlocked(f"COMMAND BLOCKED: unknown action {action}")
    path, method = path_map[action]
    logger.info("Physical action %s target=%s host=%s", action, target.kind, target.host)
    events.emit("info", "physical", f"Physical action: {action}")
    health, payload = _conversation_json(config, path, method=method, body=body or {}, timeout_s=8.0)
    if payload is None:
        raise CommandBlocked(health.technical or health.summary)
    if payload.get("error"):
        raise CommandBlocked(str(payload["error"]))
    return {"ok": True, "target": target.kind, "result": payload}
