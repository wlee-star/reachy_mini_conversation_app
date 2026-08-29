"""Start, stop, restart, and recover whitelisted services."""

from __future__ import annotations
import json
import time
import logging
import threading
from typing import Any, Callable
from pathlib import Path
from datetime import datetime, timezone

from control_dashboard import net, proc, paths, events
from control_dashboard.checks import (
    STATUS_ONLINE,
    STATUS_DEGRADED,
    STATUS_STARTING,
    HealthResult,
    diagnose,
    check_service,
    speech_pool_is_stuck,
    speech_pool_has_idle_slot,
)
from control_dashboard.redact import redact_text
from control_dashboard.registry import ServiceSpec, DashboardConfig


logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]
_SPEECH_SLOT_WAIT_S = 20.0
_SPEECH_SLOT_POLL_S = 0.5


class StackController:
    """Owns process lifecycle for registry services."""

    def __init__(self, config: DashboardConfig) -> None:
        """Create a controller for the loaded registry."""
        self.config = config
        self._lock = threading.RLock()
        self._owned: dict[str, dict[str, Any]] = {}
        self._user_stopped: set[str] = set()
        self._starting: set[str] = set()
        self._restarts: dict[str, int] = {}
        self._last_health: dict[str, HealthResult] = {}
        self._op_log: list[dict[str, Any]] = []
        paths.ensure_runtime()
        self._load_owned()

    def _load_owned(self) -> None:
        if not paths.OWNED_PATH.is_file():
            return
        try:
            payload: object = json.loads(paths.OWNED_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read owned process file: %s", exc)
            return
        if isinstance(payload, dict):
            self._owned = {str(key): value for key, value in payload.items() if isinstance(value, dict)}

    def _save_owned(self) -> None:
        paths.ensure_runtime()
        try:
            paths.OWNED_PATH.write_text(json.dumps(self._owned, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write owned process file: %s", exc)

    def _note(self, level: str, service: str, message: str, technical: str | None = None) -> None:
        events.emit(level, service, message, technical=redact_text(technical) if technical else None)
        self._op_log.append(
            {
                "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "level": level,
                "service": service,
                "message": message,
            }
        )
        self._op_log = self._op_log[-80:]

    def refresh_config(self, config: DashboardConfig) -> None:
        """Replace the loaded registry after a reload."""
        self.config = config

    def last_health(self, service_id: str) -> HealthResult | None:
        """Return the most recent health result."""
        return self._last_health.get(service_id)

    def health(self, spec: ServiceSpec, *, probe: bool = False) -> HealthResult:
        """Run and cache a health check."""
        result = check_service(spec, self.config, probe=probe)
        with self._lock:
            self._last_health[spec.id] = result
        return result

    def snapshot(self, spec: ServiceSpec, result: HealthResult) -> dict[str, Any]:
        """Build the API payload for one service."""
        owned = self._owned.get(spec.id) or {}
        process = result.details.get("process") if isinstance(result.details.get("process"), dict) else {}
        pids = process.get("pids") if isinstance(process.get("pids"), list) else []
        pid = pids[0] if pids else owned.get("pid")
        command = process.get("command")
        blocked_by = self._blocked_by(spec)
        return {
            "id": spec.id,
            "name": spec.name,
            "group": spec.group,
            "description": spec.description,
            "managed": spec.managed,
            "required": spec.required,
            "status": result.status,
            "summary": result.summary,
            "reason": diagnose(spec, result, self.config) if result.status != STATUS_ONLINE else None,
            "details": result.details,
            "latency_ms": result.latency_ms,
            "technical": redact_text(result.technical) if result.technical else None,
            "suggested_action": result.suggested_action,
            "host": spec.host,
            "port": spec.port,
            "depends_on": spec.depends_on,
            "blocked_by": blocked_by,
            "pid": pid,
            "command": redact_text(command) if command else None,
            "cwd": (spec.start or {}).get("cwd") if spec.start else None,
            "owned": spec.id in self._owned,
            "user_stopped": spec.id in self._user_stopped,
            "restart_attempts": self._restarts.get(spec.id, 0),
            "restart_limit": self.config.auto_restart_max,
            "started_at": owned.get("started_at"),
            "last_health_at": datetime.now().strftime("%H:%M:%S"),
        }

    def _blocked_by(self, spec: ServiceSpec, *, fresh: bool = False) -> list[dict[str, str]]:
        blocked: list[dict[str, str]] = []
        for dep_id in spec.depends_on:
            dep = self.config.service(dep_id)
            if dep is None:
                continue
            current = None if fresh else self._last_health.get(dep_id)
            if current is None:
                current = self.health(dep)
            if current.status not in {STATUS_ONLINE, STATUS_DEGRADED}:
                blocked.append({"id": dep.id, "name": dep.name, "status": current.status, "summary": current.summary})
        return blocked

    def _can_stop(self, spec: ServiceSpec, pid: int) -> bool:
        if spec.id in self._owned and self._owned[spec.id].get("pid") == pid:
            return True
        command = proc.process_command_line(pid)
        if spec.process_match and proc.command_matches(command, spec.process_match):
            return True
        return False

    def start(self, spec: ServiceSpec, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Start one managed service if it is not already healthy."""

        def report(message: str) -> None:
            if progress:
                progress(message)
            self._note("info", spec.id, message)

        if not spec.managed:
            return {"ok": False, "error": f"{spec.name} is monitored only and cannot be started from this dashboard."}
        if spec.start is None:
            return {"ok": False, "error": f"{spec.name} has no start command."}

        with self._lock:
            already_starting = spec.id in self._starting
            if not already_starting:
                self._starting.add(spec.id)
        try:
            current = self.health(spec)
            if current.status == STATUS_ONLINE:
                report(f"{spec.name} already running")
                report(f"{spec.name} health check passed")
                self._restarts[spec.id] = 0
                self._adopt_listening_pid(spec, self._owned.get(spec.id) or {})
                return {"ok": True, "status": current.status, "already_running": True}

            if already_starting or current.status == STATUS_STARTING:
                report(f"{spec.name} already running")
                owned_pid = (self._owned.get(spec.id) or {}).get("pid")
                pid = owned_pid if isinstance(owned_pid, int) else None
                return self._wait_until_ready(spec, pid, report)

            blocked = self._blocked_by(spec, fresh=True)
            if blocked:
                names = ", ".join(item["name"] for item in blocked)
                return {
                    "ok": False,
                    "error": f"{spec.name} cannot start until {names} is healthy.",
                    "blocked_by": blocked,
                }

            live = self._existing_instance_pid(spec)
            if live is not None:
                report(f"{spec.name} already running")
                if spec.id not in self._owned:
                    self._owned[spec.id] = {
                        "pid": live,
                        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                    }
                else:
                    self._owned[spec.id]["pid"] = live
                self._save_owned()
                self._user_stopped.discard(spec.id)
                return self._wait_until_ready(spec, live, report)

            executable = str(spec.start.get("executable") or "")
            args = [str(arg) for arg in spec.start.get("args") or []]
            cwd = Path(str(spec.start.get("cwd") or paths.REPO_ROOT))
            if not executable or not Path(executable).exists() and shutil_which(executable) is None:
                return {
                    "ok": False,
                    "error": f"Cannot find {spec.name} executable: {executable}",
                    "technical": executable,
                }
            resolved = shutil_which(executable) or executable
            log_path = paths.LOG_DIR / f"{spec.id}.log"
            self._stop_leftover_matches(spec, report)
            if spec.id == "conversation":
                self._wait_for_speech_slot(report)
            report(f"Starting {spec.name}...")
            try:
                pid = proc.start_process(
                    spec.id,
                    resolved,
                    args,
                    cwd,
                    log_path,
                    gui=bool(spec.start.get("gui")),
                )
            except OSError as exc:
                logger.warning("Failed to start %s: %s", spec.id, exc)
                self._note("error", spec.id, f"Failed to start {spec.name}", str(exc))
                return {"ok": False, "error": f"Failed to start {spec.name}.", "technical": str(exc)}

            self._owned[spec.id] = {
                "pid": pid,
                "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            }
            self._save_owned()
            self._user_stopped.discard(spec.id)
            return self._wait_until_ready(spec, pid, report)
        finally:
            if not already_starting:
                with self._lock:
                    self._starting.discard(spec.id)

    def stop(self, spec: ServiceSpec, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Stop a managed service the dashboard is allowed to control."""

        def report(message: str) -> None:
            if progress:
                progress(message)
            self._note("info", spec.id, message)

        if not spec.managed:
            return {"ok": False, "error": f"{spec.name} is not started or stopped by this dashboard."}

        if spec.stop_args and spec.start:
            executable = shutil_which(str(spec.start.get("executable") or "")) or str(
                spec.start.get("executable") or ""
            )
            try:
                proc.start_process(
                    f"{spec.id}-stop",
                    executable,
                    [str(arg) for arg in spec.stop_args],
                    Path(str(spec.start.get("cwd") or paths.REPO_ROOT)),
                    paths.LOG_DIR / f"{spec.id}.log",
                )
            except OSError as exc:
                logger.warning("Stop command failed for %s: %s", spec.id, exc)

        pids: list[int] = []
        owned_pid = (self._owned.get(spec.id) or {}).get("pid")
        if isinstance(owned_pid, int):
            pids.append(owned_pid)
        if spec.port:
            pids.extend(proc.listening_pids(spec.port))
        unique: list[int] = []
        for pid in pids:
            if pid not in unique:
                unique.append(pid)

        stopped: list[int] = []
        refused: list[int] = []
        for pid in unique:
            if self._can_stop(spec, pid):
                report(f"Stopping {spec.name} pid {pid}")
                proc.stop_pid(pid)
                stopped.append(pid)
            else:
                refused.append(pid)
                logger.warning("Refusing to stop pid %s for %s; command line did not match", pid, spec.id)

        self._owned.pop(spec.id, None)
        self._save_owned()
        self._user_stopped.add(spec.id)
        if not stopped and not unique:
            report(f"{spec.name} was not running")
            return {"ok": True, "already_stopped": True}
        if not stopped and refused:
            return {
                "ok": False,
                "error": f"Found a process on the {spec.name} port, but it did not match the whitelist so it was left running.",
            }
        time.sleep(1.0)
        return {"ok": True, "stopped_pids": stopped}

    def restart(self, spec: ServiceSpec, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Stop then start a managed service."""
        stopped = self.stop(spec, progress=progress)
        if not stopped.get("ok"):
            return stopped
        time.sleep(1.0)
        return self.start(spec, progress=progress)

    def start_all(self, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Start managed services in dependency order, skipping those already up."""
        steps: list[dict[str, Any]] = []
        ordered = topological(self.config.services)
        for spec in ordered:
            if not spec.managed:
                continue
            result = self.start(spec, progress=progress)
            steps.append({"id": spec.id, "name": spec.name, **result})
            if not result.get("ok") and spec.required:
                detail = str(result.get("error") or "").strip()
                suffix = f": {detail}" if detail else ""
                self._note("error", spec.id, f"Required service {spec.name} failed to start{suffix}")
        return {
            "ok": all(
                step.get("ok")
                for step in steps
                if self.config.service(step["id"]) and self.config.service(step["id"]).required
            ),
            "steps": steps,
        }

    def stop_all(self, progress: ProgressFn | None = None) -> dict[str, Any]:
        """Stop managed services in reverse dependency order."""
        steps: list[dict[str, Any]] = []
        ordered = list(reversed(topological(self.config.services)))
        for spec in ordered:
            if not spec.managed:
                continue
            steps.append({"id": spec.id, "name": spec.name, **self.stop(spec, progress=progress)})
        return {"ok": all(step.get("ok") for step in steps), "steps": steps}

    def _stop_leftover_matches(self, spec: ServiceSpec, report: ProgressFn) -> None:
        """Kill whitelist matches that are not the owned PID so they cannot hold a slot."""
        if not spec.process_match:
            return
        owned_pid = (self._owned.get(spec.id) or {}).get("pid")
        killed: list[int] = []
        for pid in proc.pids_matching(spec.process_match):
            if pid == owned_pid:
                continue
            if not self._can_stop(spec, pid):
                continue
            report(f"Stopping leftover {spec.name} pid {pid}")
            proc.stop_pid(pid)
            killed.append(pid)
        if killed:
            self._wait_for_port_free(spec)

    def _wait_for_port_free(self, spec: ServiceSpec) -> None:
        """Wait until a just-killed daemon has released its listen port."""
        if spec.port is None:
            return
        deadline = time.time() + 8.0
        while time.time() < deadline:
            if not proc.listening_pids(spec.port):
                return
            time.sleep(0.25)

    def _wait_for_speech_slot(self, report: ProgressFn) -> None:
        """Wait until speech-to-speech has a free realtime session, if it exposes /v1/pool."""
        speech = self.config.service("speech")
        if speech is None or not speech.host or not speech.port:
            return
        pool_url = f"http://{speech.host}:{speech.port}/v1/pool"
        deadline = time.time() + _SPEECH_SLOT_WAIT_S
        logged = False
        while time.time() < deadline:
            result = net.http_request(pool_url, timeout_s=2.0)
            payload = net.json_payload(result)
            if payload is None:
                return
            if speech_pool_is_stuck(payload):
                report("Speech-to-speech realtime slot is stuck; start conversation anyway")
                return
            if speech_pool_has_idle_slot(payload):
                return
            if not logged:
                report("Waiting for speech-to-speech to free a realtime session slot")
                logged = True
            time.sleep(_SPEECH_SLOT_POLL_S)

    def _wait_until_ready(self, spec: ServiceSpec, pid: int | None, report: ProgressFn) -> dict[str, Any]:
        """Poll health until the service is up or ``ready_timeout_s`` elapses."""
        deadline = time.time() + spec.ready_timeout_s
        while time.time() < deadline:
            result = self.health(spec)
            if result.status in {STATUS_ONLINE, STATUS_DEGRADED}:
                report(f"{spec.name} running")
                self._restarts[spec.id] = 0
                owned = self._owned.get(spec.id) or {}
                self._adopt_listening_pid(spec, owned)
                return {"ok": True, "status": result.status, "pid": owned.get("pid", pid)}
            if result.status == STATUS_STARTING:
                report(f"{spec.name} still starting...")
            time.sleep(2.0)
        result = self.health(spec)
        self._note("error", spec.id, f"{spec.name} did not become healthy in time", result.technical)
        return {
            "ok": False,
            "error": f"{spec.name} started but did not become healthy in time.",
            "technical": result.technical,
            "status": result.status,
            "pid": pid,
        }

    def recover_once(self) -> None:
        """Restart owned auto-restart services that dropped offline."""
        for spec in self.config.services:
            if not spec.managed or not spec.auto_restart:
                continue
            if spec.id in self._user_stopped:
                continue
            with self._lock:
                if spec.id in self._starting or spec.id not in self._owned:
                    continue
                owned_pid = self._owned[spec.id].get("pid")
            if isinstance(owned_pid, int) and proc.pid_is_running(owned_pid):
                continue
            live = self._live_service_pid(spec)
            if live is not None:
                self._set_owned_pid(spec, live)
                continue
            result = self.health(spec)
            with self._lock:
                if spec.id in self._starting or spec.id not in self._owned:
                    continue
                owned = self._owned[spec.id]
            if result.status in {STATUS_ONLINE, STATUS_STARTING, STATUS_DEGRADED}:
                self._adopt_listening_pid(spec, owned)
                continue
            if started_within(owned.get("started_at"), spec.ready_timeout_s):
                continue
            if self._blocked_by(spec, fresh=True):
                continue
            attempts = self._restarts.get(spec.id, 0)
            if attempts >= self.config.auto_restart_max:
                self._note(
                    "error",
                    spec.id,
                    f"{spec.name} crashed repeatedly; stopping retries ({attempts}/{self.config.auto_restart_max})",
                )
                with self._lock:
                    self._owned.pop(spec.id, None)
                self._save_owned()
                continue
            self._restarts[spec.id] = attempts + 1
            self._note(
                "warning",
                spec.id,
                f"{spec.name} stopped unexpectedly. Attempting restart {attempts + 1}/{self.config.auto_restart_max}",
            )
            started = self.start(spec)
            if started.get("ok"):
                self._note("info", spec.id, f"{spec.name} restarted")
            else:
                self._note("error", spec.id, f"{spec.name} restart failed", str(started.get("error")))

    def _existing_instance_pid(self, spec: ServiceSpec) -> int | None:
        # A second Reachy --sim fails on port 8443; reuse the live child instead.
        if spec.port is not None:
            listeners = proc.listening_pids(spec.port)
            for pid in listeners:
                if self._can_stop(spec, pid):
                    return pid
            if listeners:
                return listeners[0]
        if spec.id == "reachy_daemon" and spec.process_match:
            for pid in proc.pids_matching(spec.process_match):
                if proc.pid_is_running(pid):
                    return pid
        return None

    def _live_service_pid(self, spec: ServiceSpec) -> int | None:
        """Return a still-running PID that this dashboard is allowed to own."""
        existing = self._existing_instance_pid(spec)
        if existing is not None:
            return existing
        if spec.process_match:
            for pid in proc.pids_matching(spec.process_match):
                if proc.pid_is_running(pid):
                    return pid
        return None

    def _set_owned_pid(self, spec: ServiceSpec, pid: int) -> None:
        with self._lock:
            owned = self._owned.get(spec.id)
            if owned is None:
                self._owned[spec.id] = {
                    "pid": pid,
                    "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                }
            elif owned.get("pid") == pid:
                return
            else:
                owned["pid"] = pid
        self._save_owned()

    def _adopt_listening_pid(self, spec: ServiceSpec, owned: dict[str, Any]) -> None:
        """Replace a dead wrapper PID with the process actually serving the port."""
        live = self._live_service_pid(spec)
        if live is None:
            return
        self._set_owned_pid(spec, live)
        owned["pid"] = live

    def system_readiness(self, snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute overall SYSTEM READY from required services."""
        blockers = [
            item
            for item in snapshots
            if item.get("required") and item.get("status") not in {STATUS_ONLINE, STATUS_DEGRADED}
        ]
        degraded = [item for item in snapshots if item.get("status") == STATUS_DEGRADED]
        if blockers:
            first = blockers[0]
            return {
                "ready": False,
                "label": "SYSTEM NOT READY",
                "reason": first.get("reason") or first.get("summary"),
                "blockers": [{"id": item["id"], "name": item["name"], "status": item["status"]} for item in blockers],
                "suggested_action": first.get("suggested_action"),
            }
        if degraded:
            return {
                "ready": True,
                "label": "SYSTEM READY",
                "reason": "Core services are up, with warnings.",
                "blockers": [],
                "warnings": [{"id": item["id"], "name": item["name"]} for item in degraded],
            }
        return {"ready": True, "label": "SYSTEM READY", "reason": "Required services are healthy.", "blockers": []}


def started_within(started_at: object, timeout_s: float) -> bool:
    """Return whether ``started_at`` is still inside the ready-timeout window."""
    if not isinstance(started_at, str) or timeout_s <= 0:
        return False
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return False
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
    return 0 <= age < timeout_s


def shutil_which(executable: str) -> str | None:
    """Resolve an executable on PATH when it is not an absolute path."""
    import shutil
    from pathlib import Path as PathLib

    path = PathLib(executable)
    if path.exists():
        return str(path)
    return shutil.which(executable)


def topological(services: list[ServiceSpec]) -> list[ServiceSpec]:
    """Return services in dependency-first order."""
    by_id = {spec.id: spec for spec in services}
    seen: set[str] = set()
    ordered: list[ServiceSpec] = []

    def visit(spec: ServiceSpec) -> None:
        if spec.id in seen:
            return
        seen.add(spec.id)
        for dep_id in spec.depends_on:
            dep = by_id.get(dep_id)
            if dep is not None:
                visit(dep)
        ordered.append(spec)

    for spec in services:
        visit(spec)
    return ordered
