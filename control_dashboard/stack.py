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
        self._lifecycle_lock = threading.RLock()
        self._owned: dict[str, dict[str, Any]] = {}
        self._user_stopped: set[str] = set()
        self._starting: set[str] = set()
        self._restarts: dict[str, int] = {}
        self._stop_intent: dict[str, str] = {}
        self._last_health: dict[str, HealthResult] = {}
        self._op_log: list[dict[str, Any]] = []
        paths.ensure_runtime()
        self._load_owned()
        self._load_stopped()

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

    def _load_stopped(self) -> None:
        if not paths.STOPPED_PATH.is_file():
            return
        try:
            payload: object = json.loads(paths.STOPPED_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read stopped-service file: %s", exc)
            return
        if isinstance(payload, list):
            self._user_stopped = {str(item) for item in payload}
        elif isinstance(payload, dict) and isinstance(payload.get("ids"), list):
            self._user_stopped = {str(item) for item in payload["ids"]}

    def _save_stopped(self) -> None:
        paths.ensure_runtime()
        try:
            paths.STOPPED_PATH.write_text(json.dumps(sorted(self._user_stopped), indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write stopped-service file: %s", exc)

    def _mark_user_stopped(self, spec: ServiceSpec) -> None:
        self._user_stopped.add(spec.id)
        self._save_stopped()

    def _clear_user_stopped(self, spec: ServiceSpec) -> None:
        if spec.id not in self._user_stopped:
            return
        self._user_stopped.discard(spec.id)
        self._save_stopped()

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
        command = process.get("command") or owned.get("command")
        blocked_by = self._blocked_by(spec)
        dashboard_owned = self._is_dashboard_owned(spec)
        external = self._is_external_instance(spec, result)
        ownership = "dashboard" if dashboard_owned else "external" if external else "none"
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
            "command": redact_text(str(command)) if command else None,
            "cwd": (spec.start or {}).get("cwd") if spec.start else None,
            "owned": dashboard_owned,
            "external": external,
            "started_by_dashboard": dashboard_owned,
            "ownership": ownership,
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

    def _is_dashboard_owned(self, spec: ServiceSpec) -> bool:
        owned = self._owned.get(spec.id) or {}
        if not owned:
            return False
        return bool(owned.get("started_by_dashboard", True))

    def _is_external_instance(self, spec: ServiceSpec, result: HealthResult) -> bool:
        if self._is_dashboard_owned(spec):
            return False
        if result.status in {STATUS_ONLINE, STATUS_DEGRADED, STATUS_STARTING}:
            return True
        return self._existing_instance_pid(spec) is not None

    def _can_stop(self, spec: ServiceSpec, pid: int) -> bool:
        return self._pid_belongs_to_service(spec, pid)

    def _pid_belongs_to_service(self, spec: ServiceSpec, pid: int) -> bool:
        """Return whether a live PID is the intended service, never a reused PID alone."""
        if pid <= 0 or not proc.pid_is_running(pid):
            return False
        if spec.process_match:
            return proc.pid_matches_pattern(pid, spec.process_match)
        if spec.id == "reachy_daemon" and spec.port is not None:
            return pid in proc.listening_pids(spec.port)
        return False

    def _protected_pids(self, spec: ServiceSpec) -> set[int]:
        protected: set[int] = set()
        owned = self._owned.get(spec.id) or {}
        for key in ("pid", "wrapper_pid"):
            candidate = owned.get(key)
            if isinstance(candidate, int) and self._pid_belongs_to_service(spec, candidate):
                protected.add(candidate)
        return protected

    def _is_protected_pid(self, spec: ServiceSpec, pid: int, protected: set[int]) -> bool:
        if pid in protected:
            return True
        return any(proc.pid_in_same_tree(pid, owned) for owned in protected)

    def _hermes_trace(self, spec: ServiceSpec, report: ProgressFn | None, *lines: str) -> None:
        if spec.id != "hermes":
            return
        for line in lines:
            message = line if line.startswith("[HERMES]") else f"[HERMES] {line}"
            logger.info("%s", message)
            if report:
                report(message)
            else:
                self._note("info", spec.id, message)

    def _mark_stop_intent(self, spec: ServiceSpec, reason: str) -> None:
        self._stop_intent[spec.id] = reason

    def _clear_stop_intent(self, spec: ServiceSpec) -> None:
        self._stop_intent.pop(spec.id, None)

    def _service_start_env(self, spec: ServiceSpec) -> dict[str, str] | None:
        """Pin the conversation app to the local simulator unless a physical host is set."""
        if spec.id != "conversation":
            return None
        host = (self.config.env.get("REACHY_DAEMON_HOST") or self.config.env.get("REACHY_DAEMON_URL") or "").strip()
        if host:
            return None
        return {"REACHY_DAEMON_HOST": "127.0.0.1"}

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

        acquired = False
        wait_pid: int | None = None
        finished: dict[str, Any] | None = None
        claim_ownership = True
        with self._lifecycle_lock:
            with self._lock:
                already_starting = spec.id in self._starting
                if not already_starting:
                    self._starting.add(spec.id)
                    acquired = True
            try:
                current = self.health(spec)
                if spec.id == "hermes":
                    owned_pid = (self._owned.get(spec.id) or {}).get("pid")
                    command = proc.process_command_line(owned_pid) if isinstance(owned_pid, int) else None
                    self._hermes_trace(
                        spec,
                        report,
                        f"Health check PID={owned_pid if isinstance(owned_pid, int) else 'none'}",
                        f"Process exists={isinstance(owned_pid, int) and proc.pid_is_running(owned_pid)}",
                        f"Command matches Hermes={bool(spec.process_match and proc.command_matches(command, spec.process_match))}",
                        f"Health={'healthy' if current.status in {STATUS_ONLINE, STATUS_DEGRADED} else 'unhealthy'}",
                    )
                if current.status == STATUS_ONLINE:
                    external = not self._is_dashboard_owned(spec)
                    label = (
                        f"{spec.name} already running — external process"
                        if external
                        else f"{spec.name} already running"
                    )
                    report(label)
                    report(f"{spec.name} health check passed")
                    self._restarts[spec.id] = 0
                    self._clear_stop_intent(spec)
                    self._clear_user_stopped(spec)
                    if not external:
                        self._adopt_listening_pid(spec, self._owned.get(spec.id) or {})
                    finished = {
                        "ok": True,
                        "status": current.status,
                        "already_running": True,
                        "external": external,
                    }
                elif already_starting or current.status == STATUS_STARTING:
                    report(f"{spec.name} already running")
                    owned_pid = (self._owned.get(spec.id) or {}).get("pid")
                    wait_pid = owned_pid if isinstance(owned_pid, int) else None
                    claim_ownership = self._is_dashboard_owned(spec)
                else:
                    blocked = self._blocked_by(spec, fresh=True)
                    if blocked:
                        names = ", ".join(item["name"] for item in blocked)
                        finished = {
                            "ok": False,
                            "error": f"{spec.name} cannot start until {names} is healthy.",
                            "blocked_by": blocked,
                        }
                    else:
                        live = self._existing_instance_pid(spec)
                        if live is not None:
                            external = not self._is_dashboard_owned(spec)
                            report(
                                f"{spec.name} already running — external process"
                                if external
                                else f"{spec.name} already running"
                            )
                            self._clear_stop_intent(spec)
                            if self._is_dashboard_owned(spec):
                                self._set_owned_pid(spec, live)
                            else:
                                claim_ownership = False
                            self._clear_user_stopped(spec)
                            wait_pid = live
                        else:
                            executable = str(spec.start.get("executable") or "")
                            args = [str(arg) for arg in spec.start.get("args") or []]
                            cwd = Path(str(spec.start.get("cwd") or paths.REPO_ROOT))
                            if not executable or not Path(executable).exists() and shutil_which(executable) is None:
                                finished = {
                                    "ok": False,
                                    "error": f"Cannot find {spec.name} executable: {executable}",
                                    "technical": executable,
                                }
                            else:
                                resolved = shutil_which(executable) or executable
                                log_path = paths.LOG_DIR / f"{spec.id}.log"
                                if spec.id == "reachy_daemon" and spec.port is not None:
                                    occupied = [
                                        pid
                                        for pid in proc.listening_pids(spec.port)
                                        if proc.pid_is_running(pid) and not self._pid_belongs_to_service(spec, pid)
                                    ]
                                    if occupied:
                                        finished = {
                                            "ok": False,
                                            "error": (
                                                "Port 8000 is already in use by another program, not the MuJoCo simulator. "
                                                "Close the Reachy Mini desktop app if it is open, then start Reachy Mini again."
                                            ),
                                        }
                                if finished is None:
                                    self._stop_leftover_matches(spec, report)
                                    if spec.id == "conversation":
                                        self._wait_for_speech_slot(report)
                                    report(f"Starting {spec.name}...")
                                    self._hermes_trace(
                                        spec,
                                        report,
                                        "Starting Hermes",
                                        f"Command={resolved} {' '.join(args)}".rstrip(),
                                    )
                                    try:
                                        pid = proc.start_process(
                                            spec.id,
                                            resolved,
                                            args,
                                            cwd,
                                            log_path,
                                            env=self._service_start_env(spec),
                                            gui=bool(spec.start.get("gui")),
                                        )
                                    except OSError as exc:
                                        logger.warning("Failed to start %s: %s", spec.id, exc)
                                        self._note("error", spec.id, f"Failed to start {spec.name}", str(exc))
                                        finished = {
                                            "ok": False,
                                            "error": f"Failed to start {spec.name}.",
                                            "technical": str(exc),
                                        }
                                    else:
                                        self._owned[spec.id] = {
                                            "pid": pid,
                                            "wrapper_pid": pid,
                                            "command": f"{resolved} {' '.join(args)}".strip(),
                                            "port": spec.port,
                                            "started_by_dashboard": True,
                                            "started_at": datetime.now(timezone.utc)
                                            .astimezone()
                                            .isoformat(timespec="seconds"),
                                        }
                                        self._save_owned()
                                        self._clear_user_stopped(spec)
                                        self._clear_stop_intent(spec)
                                        self._hermes_trace(spec, report, f"PID={pid}", "Process group=new")
                                        wait_pid = pid
            except Exception:
                if acquired:
                    with self._lock:
                        self._starting.discard(spec.id)
                raise
            if finished is not None and acquired:
                with self._lock:
                    self._starting.discard(spec.id)

        if finished is not None:
            return finished
        try:
            return self._wait_until_ready(spec, wait_pid, report, claim=claim_ownership)
        finally:
            if acquired:
                with self._lock:
                    self._starting.discard(spec.id)

    def stop(
        self, spec: ServiceSpec, progress: ProgressFn | None = None, *, owned_only: bool = False
    ) -> dict[str, Any]:
        """Stop a managed service the dashboard is allowed to control."""

        def report(message: str) -> None:
            if progress:
                progress(message)
            self._note("info", spec.id, message)

        if not spec.managed:
            return {"ok": False, "error": f"{spec.name} is not started or stopped by this dashboard."}

        with self._lifecycle_lock:
            return self._stop_locked(spec, report, owned_only=owned_only)

    def _stop_locked(self, spec: ServiceSpec, report: ProgressFn, *, owned_only: bool = False) -> dict[str, Any]:
        caller = "shutdown"
        self._mark_stop_intent(spec, caller)
        owned_pid = (self._owned.get(spec.id) or {}).get("pid")
        dashboard_owned = self._is_dashboard_owned(spec)
        self._hermes_trace(
            spec,
            report,
            "STOP REQUEST",
            f"PID={owned_pid if isinstance(owned_pid, int) else 'none'}",
            "Reason=intentional shutdown",
            f"Caller={caller}",
        )
        if owned_only and not dashboard_owned:
            live = self._existing_instance_pid(spec)
            if live is not None:
                report(f"{spec.name} is an external process; left running")
                self._clear_stop_intent(spec)
                return {"ok": True, "external": True, "left_running": True, "pid": live}
            report(f"{spec.name} was not running")
            self._owned.pop(spec.id, None)
            self._save_owned()
            self._mark_user_stopped(spec)
            return {"ok": True, "already_stopped": True}

        if dashboard_owned:
            self._run_stop_command(spec)

        pids: list[int] = []
        if dashboard_owned and isinstance(owned_pid, int):
            pids.append(owned_pid)
        wrapper_pid = (self._owned.get(spec.id) or {}).get("wrapper_pid")
        if dashboard_owned and isinstance(wrapper_pid, int):
            pids.append(wrapper_pid)
        if not owned_only:
            if spec.port:
                pids.extend(proc.listening_pids(spec.port))
            if spec.process_match:
                pids.extend(proc.pids_matching(spec.process_match))
        unique: list[int] = []
        for pid in pids:
            if pid not in unique:
                unique.append(pid)

        stopped: list[int] = []
        refused: list[int] = []
        for pid in unique:
            if self._pid_belongs_to_service(spec, pid):
                report(f"Stopping {spec.name} pid {pid}")
                proc.stop_pid(pid)
                gone = proc.wait_until_gone(pid)
                self._hermes_trace(spec, report, "STOP RESULT", f"PID={pid}", f"Terminated={gone}")
                stopped.append(pid)
            elif proc.pid_is_running(pid):
                refused.append(pid)
                logger.warning("Refusing to stop pid %s for %s; command line did not match", pid, spec.id)

        self._owned.pop(spec.id, None)
        self._save_owned()
        self._mark_user_stopped(spec)
        if stopped:
            self._wait_for_port_free(spec)
        if not stopped and not unique:
            report(f"{spec.name} was not running")
            return {"ok": True, "already_stopped": True}
        if not stopped and refused:
            return {
                "ok": False,
                "error": f"Found a process on the {spec.name} port, but it did not match the whitelist so it was left running.",
            }
        return {"ok": True, "stopped_pids": stopped}

    def _run_stop_command(self, spec: ServiceSpec) -> None:
        """Run a service stop CLI synchronously so it cannot race a later start."""
        if not spec.stop_args or not spec.start:
            return
        executable = shutil_which(str(spec.start.get("executable") or "")) or str(spec.start.get("executable") or "")
        if not executable:
            return
        try:
            proc.run_logged(
                executable,
                [str(arg) for arg in spec.stop_args],
                Path(str(spec.start.get("cwd") or paths.REPO_ROOT)),
                paths.LOG_DIR / f"{spec.id}.log",
                timeout_s=30.0,
            )
        except OSError as exc:
            logger.warning("Stop command failed for %s: %s", spec.id, exc)
        self._wait_for_port_free(spec)

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
        """Kill whitelist matches that are not the live managed instance."""
        if not spec.process_match:
            return
        protected = self._protected_pids(spec)
        matches = [pid for pid in proc.pids_matching(spec.process_match) if self._pid_belongs_to_service(spec, pid)]
        stale = [pid for pid in matches if not self._is_protected_pid(spec, pid, protected)]
        if not stale:
            return
        if spec.stop_args:
            self._mark_stop_intent(spec, "cleanup")
            for pid in stale:
                self._hermes_trace(
                    spec,
                    report,
                    "STOP REQUEST",
                    f"PID={pid}",
                    "Reason=unmanaged or stale instance",
                    "Caller=startup",
                )
                report(f"Stopping leftover {spec.name} pid {pid}")
            self._run_stop_command(spec)
            remaining = [pid for pid in stale if proc.pid_is_running(pid) and self._pid_belongs_to_service(spec, pid)]
        else:
            remaining = stale
        killed: list[int] = []
        for pid in remaining:
            if self._is_protected_pid(spec, pid, protected):
                continue
            if not self._pid_belongs_to_service(spec, pid):
                continue
            self._mark_stop_intent(spec, "cleanup")
            self._hermes_trace(
                spec,
                report,
                "STOP REQUEST",
                f"PID={pid}",
                "Reason=unmanaged or stale instance",
                "Caller=startup",
            )
            report(f"Stopping leftover {spec.name} pid {pid}")
            proc.stop_pid(pid)
            gone = proc.wait_until_gone(pid)
            self._hermes_trace(spec, report, "STOP RESULT", f"PID={pid}", f"Terminated={gone}")
            killed.append(pid)
        if killed or spec.stop_args:
            self._wait_for_port_free(spec)

    def _wait_for_port_free(self, spec: ServiceSpec) -> None:
        """Wait until a just-killed daemon has released its listen port."""
        if spec.port is None:
            return
        deadline = time.time() + 8.0
        while time.time() < deadline:
            proc.invalidate_listen_cache()
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

    def _recent_log_excerpt(self, spec: ServiceSpec, lines: int = 40) -> str | None:
        path = paths.LOG_DIR / f"{spec.id}.log"
        if not path.is_file():
            return None
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed to read %s log: %s", spec.id, exc)
            return None
        excerpt = "\n".join(text.splitlines()[-lines:]).strip()
        return excerpt or None

    def _startup_failure_detail(self, spec: ServiceSpec, result: HealthResult) -> str:
        parts: list[str] = []
        if result.technical:
            parts.append(result.technical)
        if result.summary:
            parts.append(result.summary)
        excerpt = self._recent_log_excerpt(spec)
        if excerpt:
            for line in reversed(excerpt.splitlines()):
                lowered = line.lower()
                if "error" in lowered or "connection failed" in lowered or "traceback" in lowered:
                    parts.append(line.strip())
                    break
            parts.append(excerpt)
        return "\n".join(parts) if parts else "no captured output"

    def _foreign_port_occupant(self, spec: ServiceSpec, owned_pid: int | None) -> str | None:
        if spec.port is None:
            return None
        for pid in proc.listening_pids(spec.port):
            if owned_pid is not None and (pid == owned_pid or proc.pid_in_same_tree(pid, owned_pid)):
                continue
            if self._pid_belongs_to_service(spec, pid):
                continue
            if not proc.pid_is_running(pid):
                continue
            command = proc.process_command_line(pid) or proc.process_name(pid) or f"pid {pid}"
            return f"pid {pid}: {command}"
        return None

    def _wait_until_ready(
        self, spec: ServiceSpec, pid: int | None, report: ProgressFn, *, claim: bool = True
    ) -> dict[str, Any]:
        """Poll health until the service is up or ``ready_timeout_s`` elapses."""
        deadline = time.time() + spec.ready_timeout_s
        while time.time() < deadline:
            result = self.health(spec)
            if result.status in {STATUS_ONLINE, STATUS_DEGRADED}:
                report(f"{spec.name} running")
                self._restarts[spec.id] = 0
                owned = self._owned.get(spec.id) or {}
                if claim and spec.id in self._owned:
                    self._adopt_listening_pid(spec, owned)
                live_pid = owned.get("pid", pid)
                command = proc.process_command_line(live_pid) if isinstance(live_pid, int) else None
                self._hermes_trace(
                    spec,
                    report,
                    f"Health check PID={live_pid if isinstance(live_pid, int) else 'none'}",
                    f"Process exists={isinstance(live_pid, int) and proc.pid_is_running(live_pid)}",
                    f"Command matches Hermes={bool(spec.process_match and proc.command_matches(command, spec.process_match))}",
                    "Health=healthy",
                )
                return {
                    "ok": True,
                    "status": result.status,
                    "pid": live_pid,
                    "already_running": not claim,
                    "external": not claim,
                }
            if spec.id in self._user_stopped or self._stop_intent.get(spec.id):
                return {
                    "ok": False,
                    "error": f"{spec.name} was stopped before it became healthy.",
                    "status": result.status,
                    "pid": pid,
                }
            if pid is not None and not proc.pid_is_running(pid):
                live = self._live_service_pid(spec)
                if live is not None:
                    if claim and spec.id in self._owned:
                        self._set_owned_pid(spec, live)
                    pid = live
                elif spec.id == "hermes" and self._stop_intent.get(spec.id) is None:
                    self._hermes_trace(
                        spec,
                        report,
                        "UNEXPECTED EXIT",
                        f"PID={pid}",
                        "Exit code=unknown",
                        "Signal=unknown",
                    )
                elif spec.id != "hermes":
                    report(f"{spec.name} exited before becoming healthy")
                    detail = self._startup_failure_detail(spec, result)
                    self._note("error", spec.id, f"{spec.name} exited before becoming healthy", detail)
                    if claim:
                        self._owned.pop(spec.id, None)
                        self._save_owned()
                    return {
                        "ok": False,
                        "error": f"{spec.name} exited before becoming healthy.",
                        "technical": detail,
                        "status": result.status,
                        "pid": pid,
                    }
            occupant = self._foreign_port_occupant(spec, pid)
            if occupant is not None:
                report(f"{spec.name} port {spec.port} is occupied by another process")
                self._note("error", spec.id, f"{spec.name} port is occupied", occupant)
                return {
                    "ok": False,
                    "error": f"{spec.name} cannot start because port {spec.port} is in use by another process.",
                    "technical": occupant,
                    "status": result.status,
                    "pid": pid,
                }
            if result.status == STATUS_STARTING:
                report(f"{spec.name} still starting...")
            time.sleep(2.0)
        result = self.health(spec)
        detail = self._startup_failure_detail(spec, result)
        alive = pid is not None and proc.pid_is_running(pid)
        message = (
            f"{spec.name} is still initializing after {spec.ready_timeout_s:.0f}s"
            if alive
            else f"{spec.name} did not become healthy in time"
        )
        self._note("error", spec.id, message, detail)
        return {
            "ok": False,
            "error": (
                f"{spec.name} process is still running but did not become healthy in {spec.ready_timeout_s:.0f}s."
                if alive
                else f"{spec.name} started but did not become healthy in time."
            ),
            "technical": detail,
            "status": result.status,
            "pid": pid,
        }

    def recover_once(self) -> None:
        """Restart owned auto-restart services that dropped offline."""
        if not self._lifecycle_lock.acquire(blocking=False):
            return
        try:
            self._recover_locked()
        finally:
            self._lifecycle_lock.release()

    def _recover_locked(self) -> None:
        for spec in self.config.services:
            if not spec.managed or not spec.auto_restart:
                continue
            if spec.id in self._user_stopped:
                continue
            with self._lock:
                if spec.id in self._starting or spec.id not in self._owned:
                    continue
                if not bool(self._owned[spec.id].get("started_by_dashboard", True)):
                    continue
                owned_pid = self._owned[spec.id].get("pid")
                wrapper_pid = self._owned[spec.id].get("wrapper_pid")
            owned_alive = isinstance(owned_pid, int) and proc.pid_is_running(owned_pid)
            wrapper_alive = isinstance(wrapper_pid, int) and proc.pid_is_running(wrapper_pid)
            if owned_alive:
                continue
            live = self._live_service_pid(spec)
            if live is not None:
                if spec.id == "hermes" and isinstance(owned_pid, int):
                    self._hermes_trace(
                        spec,
                        None,
                        f"Health check PID={owned_pid}",
                        "Process exists=false",
                        "Command matches Hermes=false",
                        "Health=healthy",
                    )
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
            if spec.id == "reachy_daemon" and started_within(owned.get("started_at"), spec.ready_timeout_s):
                continue
            if self._blocked_by(spec, fresh=True):
                continue
            intent = self._stop_intent.get(spec.id)
            if intent:
                continue
            if spec.id == "hermes":
                self._hermes_trace(
                    spec,
                    None,
                    "UNEXPECTED EXIT",
                    f"PID={owned_pid if isinstance(owned_pid, int) else 'none'}",
                    "Exit code=unknown",
                    "Signal=unknown",
                    f"Wrapper alive={wrapper_alive}",
                )
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
        # Hermes daemonizes out of the CLI wrapper; Reachy --sim fails if a second copy is spawned.
        if spec.port is not None:
            listeners = proc.listening_pids(spec.port)
            for pid in listeners:
                if self._pid_belongs_to_service(spec, pid):
                    return pid
        if spec.id in {"reachy_daemon", "hermes"} and spec.process_match:
            for pid in proc.pids_matching(spec.process_match):
                if proc.pid_is_running(pid) and self._pid_belongs_to_service(spec, pid):
                    return pid
        return None

    def _live_service_pid(self, spec: ServiceSpec) -> int | None:
        """Return a still-running PID that this dashboard is allowed to own."""
        existing = self._existing_instance_pid(spec)
        if existing is not None:
            return existing
        if spec.process_match:
            for pid in proc.pids_matching(spec.process_match):
                if self._pid_belongs_to_service(spec, pid):
                    return pid
        return None

    def _set_owned_pid(self, spec: ServiceSpec, pid: int) -> None:
        with self._lock:
            owned = self._owned.get(spec.id)
            if owned is None:
                return
            if owned.get("pid") == pid:
                return
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
