"""Local HTTP API and static UI for the control dashboard."""

from __future__ import annotations
import json
import socket
import logging
import threading
from typing import Any
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from control_dashboard import paths, stack, events, physical
from control_dashboard.redact import redact_text
from control_dashboard.physical import CommandBlocked
from control_dashboard.registry import DashboardConfig, load_config


logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = frozenset({"start", "stop", "restart", "health", "test", "ack_stop"})
_controller: stack.StackController | None = None
_config: DashboardConfig | None = None
_settings: dict[str, Any] = {"development_mode": True, "auto_restart": True}


def _load_settings() -> None:
    if not paths.SETTINGS_PATH.is_file():
        return
    try:
        payload: object = json.loads(paths.SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read dashboard settings: %s", exc)
        return
    if isinstance(payload, dict):
        _settings.update(payload)


def _save_settings() -> None:
    paths.ensure_runtime()
    try:
        paths.SETTINGS_PATH.write_text(json.dumps(_settings, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write dashboard settings: %s", exc)


def _require() -> tuple[DashboardConfig, stack.StackController]:
    if _config is None or _controller is None:
        raise RuntimeError("Dashboard is not initialized.")
    return _config, _controller


def _json(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    if length > 1_000_000:
        return {}
    raw = handler.rfile.read(length)
    try:
        payload: object = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail_log(service_id: str, lines: int = 80) -> list[str]:
    path = paths.LOG_DIR / f"{service_id}.log"
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Failed to read log %s: %s", path, exc)
        return []
    return [redact_text(line) for line in text.splitlines()[-lines:]]


def _status_payload() -> dict[str, Any]:
    config, controller = _require()
    results: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_map = {pool.submit(controller.health, spec): spec.id for spec in config.services}
        for future in as_completed(future_map):
            results[future_map[future]] = future.result()
    snapshots = [controller.snapshot(spec, results[spec.id]) for spec in config.services]
    readiness = controller.system_readiness(snapshots)
    return {
        "readiness": readiness,
        "services": snapshots,
        "development_mode": bool(_settings.get("development_mode", True)),
        "auto_restart": bool(_settings.get("auto_restart", True)),
        "dashboard": {"host": config.host, "port": config.port},
    }


def _handle_action(service_id: str, action: str) -> tuple[int, dict[str, Any]]:
    config, controller = _require()
    if action not in ALLOWED_ACTIONS:
        return 400, {"error": "Unknown action."}
    spec = config.service(service_id)
    if spec is None:
        return 404, {"error": "Unknown service."}
    if action == "health":
        result = controller.health(spec)
        return 200, controller.snapshot(spec, result)
    if action == "test":
        result = controller.health(spec, probe=True)
        events.emit("info", spec.id, f"{spec.name} test: {result.summary}")
        return 200, controller.snapshot(spec, result)
    if action == "start":
        return 200, controller.start(spec)
    if action == "stop":
        return 200, controller.stop(spec)
    if action == "ack_stop":
        return 200, controller.acknowledge_intentional_stop(spec, reason="sleep")
    return 200, controller.restart(spec)


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    """Refuse to share the dashboard port with a second controller."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        """Bind the port exclusively so a second dashboard cannot share it."""
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve the dashboard UI and a small JSON control API."""

    server_version = "ReachyControlDashboard/0.1"

    def log_message(self, format: str, *args: object) -> None:
        """Write access logs through the module logger."""
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET routes."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path in {"/", "/index.html"}:
            return self._send_file(paths.STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            return self._send_static(path[len("/static/") :])
        if path == "/api/status":
            return _json(self, _status_payload())
        if path == "/api/config":
            config, _controller = _require()
            return _json(
                self,
                {
                    "env": config.public_env,
                    "conversation_root": str(config.conversation_root),
                    "ai_stack_root": str(config.ai_stack_root),
                    "development_mode": bool(_settings.get("development_mode", True)),
                    "auto_restart": bool(_settings.get("auto_restart", True)),
                    "services": [
                        {
                            "id": spec.id,
                            "name": spec.name,
                            "managed": spec.managed,
                            "required": spec.required,
                            "port": spec.port,
                            "depends_on": spec.depends_on,
                            "description": spec.description,
                        }
                        for spec in config.services
                    ],
                },
            )
        if path == "/api/events":
            after = int((query.get("after_id") or ["0"])[0] or 0)
            service = (query.get("service") or [None])[0]
            errors_only = (query.get("errors_only") or ["0"])[0] in {"1", "true"}
            return _json(
                self, {"events": events.list_events(after_id=after, service=service, errors_only=errors_only)}
            )
        if path == "/api/physical/status":
            config, controller = _require()
            return _json(self, physical.build_physical_status(config, controller.health))
        if path == "/api/physical/target":
            config, _controller = _require()
            target = physical.resolve_target(config)
            return _json(
                self,
                {
                    "kind": target.kind,
                    "label": {
                        physical.TARGET_PHYSICAL: "PHYSICAL REACHY MINI",
                        physical.TARGET_SIMULATOR: "VIRTUAL REACHY MINI",
                        physical.TARGET_UNKNOWN: "UNKNOWN TARGET",
                    }.get(target.kind, target.kind.upper()),
                    "host": target.host,
                    "port": target.port,
                    "simulation_enabled": target.simulation_enabled,
                    "summary": target.summary,
                },
            )
        if path == "/api/physical/camera.jpg":
            config, _controller = _require()
            jpeg, meta = physical.fetch_camera_jpeg(config)
            if jpeg is None:
                status_code = 409 if meta.get("status") == "preview_off" else 503
                return _json(
                    self,
                    {
                        "error": meta.get("summary") or "CAMERA OFFLINE",
                        "meta": meta,
                    },
                    status_code,
                )
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Camera-Status", str(meta.get("status") or "live"))
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return None
        if path.startswith("/api/services/") and path.endswith("/logs"):
            service_id = path[len("/api/services/") : -len("/logs")].strip("/")
            return _json(self, {"lines": _tail_log(service_id)})
        if path.startswith("/api/services/") and path.count("/") == 3:
            service_id = path.split("/")[3]
            config, controller = _require()
            spec = config.service(service_id)
            if spec is None:
                return _json(self, {"error": "Unknown service."}, 404)
            return _json(self, controller.snapshot(spec, controller.health(spec)))
        return _json(self, {"error": "Not found."}, 404)

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST control routes."""
        parsed = urlparse(self.path)
        path = parsed.path
        body = _read_json(self)
        if path == "/api/events/clear":
            events.clear_events()
            return _json(self, {"ok": True})
        if path == "/api/settings":
            if "development_mode" in body:
                _settings["development_mode"] = bool(body["development_mode"])
            if "auto_restart" in body:
                _settings["auto_restart"] = bool(body["auto_restart"])
            _save_settings()
            return _json(self, dict(_settings))
        if path == "/api/physical/camera":
            enabled = bool(body.get("enabled")) if "enabled" in body else True
            return _json(self, {"ok": True, "enabled": physical.set_camera_preview_enabled(enabled)})
        if path in {
            "/api/physical/mic",
            "/api/physical/speaker",
            "/api/physical/speaker/test",
            "/api/physical/safe-stop",
        }:
            config, _controller = _require()
            action = {
                "/api/physical/mic": "mic",
                "/api/physical/speaker": "speaker",
                "/api/physical/speaker/test": "speaker_test",
                "/api/physical/safe-stop": "safe_stop",
            }[path]
            try:
                return _json(self, physical.run_physical_action(config, action, body))
            except CommandBlocked as exc:
                events.emit("error", "physical", exc.reason)
                return _json(self, {"ok": False, "error": exc.reason, "blocked": True}, 409)
            except Exception as exc:
                logger.warning("Physical action %s failed: %s", action, exc)
                events.emit("error", "physical", f"{action} failed: {exc}")
                return _json(self, {"ok": False, "error": str(exc)}, 500)
        if path == "/api/stack/start":
            _, controller = _require()
            return _json(self, controller.start_all())
        if path == "/api/stack/stop":
            _, controller = _require()
            return _json(self, controller.stop_all())
        if path.startswith("/api/services/") and path.count("/") == 4:
            _, _, _, service_id, action = path.split("/")
            status, payload = _handle_action(service_id, action)
            return _json(self, payload, status)
        return _json(self, {"error": "Not found."}, 404)

    def _send_static(self, relative: str) -> None:
        candidate = (paths.STATIC_DIR / relative).resolve()
        if paths.STATIC_DIR.resolve() not in candidate.parents and candidate != paths.STATIC_DIR.resolve():
            return _json(self, {"error": "Not found."}, 404)
        if not candidate.is_file():
            return _json(self, {"error": "Not found."}, 404)
        content_type = "text/plain"
        if candidate.suffix == ".css":
            content_type = "text/css"
        elif candidate.suffix == ".js":
            content_type = "application/javascript"
        elif candidate.suffix == ".svg":
            content_type = "image/svg+xml"
        self._send_file(candidate, content_type)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            return _json(self, {"error": "Not found."}, 404)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _recovery_loop(controller: stack.StackController, stop_event: threading.Event) -> None:
    while not stop_event.wait(controller.config.poll_interval_s):
        if not _settings.get("auto_restart", True):
            continue
        try:
            controller.recover_once()
        except Exception as exc:
            logger.warning("Recovery pass failed: %s", exc)


def serve(host: str | None = None, port: int | None = None) -> None:
    """Start the dashboard HTTP server."""
    global _config, _controller
    paths.ensure_runtime()
    _load_settings()
    _config = load_config()
    if _settings.get("development_mode") is None:
        _settings["development_mode"] = _config.development_mode
    _controller = stack.StackController(_config)
    bind_host = host or _config.host
    bind_port = port or _config.port
    httpd = ExclusiveThreadingHTTPServer((bind_host, bind_port), DashboardHandler)
    stop_event = threading.Event()
    recovery = threading.Thread(target=_recovery_loop, args=(_controller, stop_event), daemon=True)
    recovery.start()
    events.emit("info", "dashboard", f"Dashboard listening on http://{bind_host}:{bind_port}")
    logger.info("Reachy Mini control dashboard at http://%s:%s", bind_host, bind_port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Dashboard shutting down")
    finally:
        stop_event.set()
        httpd.server_close()
