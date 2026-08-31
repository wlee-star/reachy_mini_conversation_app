"""Load the service registry from JSON and the conversation app .env."""

from __future__ import annotations
import json
import shutil
import logging
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

from control_dashboard import paths
from control_dashboard.redact import parse_env_file, public_env_map


logger = logging.getLogger(__name__)


@dataclass
class ServiceSpec:
    """One dashboard-managed or monitored service."""

    id: str
    name: str
    group: str
    description: str
    managed: bool
    required: bool
    auto_restart: bool
    health: str
    depends_on: list[str] = field(default_factory=list)
    host: str | None = None
    port: int | None = None
    fallback_hosts: list[str] = field(default_factory=list)
    process_match: str | None = None
    ready_timeout_s: float = 30.0
    needs: list[str] = field(default_factory=list)
    start: dict[str, Any] | None = None
    stop_args: list[str] | None = None

    def missing_needs(self, env: dict[str, str]) -> list[str]:
        """Return required env keys that are empty."""
        missing: list[str] = []
        for key in self.needs:
            if not (env.get(key) or "").strip():
                missing.append(key)
        return missing


@dataclass
class DashboardConfig:
    """Resolved dashboard and service registry."""

    host: str
    port: int
    poll_interval_s: float
    auto_restart_max: int
    development_mode: bool
    conversation_root: Path
    ai_stack_root: Path
    env: dict[str, str]
    public_env: dict[str, str]
    services: list[ServiceSpec]

    def service(self, service_id: str) -> ServiceSpec | None:
        """Return a service spec by id."""
        for spec in self.services:
            if spec.id == service_id:
                return spec
        return None

    def service_ids(self) -> set[str]:
        """Return the whitelist of known service ids."""
        return {spec.id for spec in self.services}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _fill(value: str, mapping: dict[str, str]) -> str:
    filled = value
    for key, replacement in mapping.items():
        filled = filled.replace("{" + key + "}", replacement)
    return filled


def load_conversation_env(env_path: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from the conversation app .env when present."""
    if not env_path.is_file():
        example = env_path.with_name(".env.example")
        if example.is_file():
            try:
                return parse_env_file(example.read_text(encoding="utf-8"))
            except OSError as exc:
                logger.warning("Failed to read %s: %s", example, exc)
        return {}
    try:
        return parse_env_file(env_path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("Failed to read %s: %s", env_path, exc)
        return {}


def _hermes_exe() -> str:
    found = shutil.which("hermes")
    if found:
        return found
    candidate = paths.hermes_home() / "bin" / ("hermes.exe" if os_is_windows() else "hermes")
    return str(candidate)


def os_is_windows() -> bool:
    """Return whether the dashboard is running on Windows."""
    import sys

    return sys.platform == "win32"


def load_config() -> DashboardConfig:
    """Load services.json, optional local overlay, and conversation .env."""
    raw = _read_json(paths.SERVICES_PATH)
    overlay = _read_json(paths.LOCAL_SERVICES_PATH)
    if overlay:
        raw = _deep_merge(raw, overlay)

    dashboard = raw.get("dashboard") if isinstance(raw.get("dashboard"), dict) else {}
    path_cfg = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}

    conversation_root = Path(str(path_cfg.get("conversation_root") or paths.REPO_ROOT))
    if "{repo_root}" in str(conversation_root):
        conversation_root = paths.REPO_ROOT
    ai_stack_root = Path(str(path_cfg.get("ai_stack_root") or paths.default_ai_stack_root()))
    if "{ai_stack_root}" in str(path_cfg.get("ai_stack_root") or ""):
        ai_stack_root = paths.default_ai_stack_root()

    env_path = Path(str(path_cfg.get("conversation_env") or (paths.REPO_ROOT / ".env")))
    if "{repo_root}" in str(env_path):
        env_path = paths.REPO_ROOT / ".env"
    env = load_conversation_env(env_path)

    speech_exe = paths.venv_script(ai_stack_root, "speech-to-speech")
    mapping = {
        "repo_root": str(paths.REPO_ROOT),
        "ai_stack_root": str(ai_stack_root),
        "conversation_root": str(conversation_root),
        "conversation_python": str(paths.venv_python(conversation_root)),
        "speech_python": str(paths.venv_python(ai_stack_root)),
        "speech_exe": str(speech_exe),
        "daemon_exe": str(paths.venv_script(conversation_root, "reachy-mini-daemon")),
        "hermes_exe": _hermes_exe(),
        "hermes_home": str(paths.hermes_home()),
    }

    services: list[ServiceSpec] = []
    raw_services = raw.get("services") if isinstance(raw.get("services"), list) else []
    for item in raw_services:
        if not isinstance(item, dict):
            continue
        start = item.get("start")
        if isinstance(start, dict):
            start = dict(start)
            if isinstance(start.get("executable"), str):
                start["executable"] = _fill(start["executable"], mapping)
            if isinstance(start.get("cwd"), str):
                start["cwd"] = _fill(start["cwd"], mapping)
            args = start.get("args")
            if isinstance(args, list):
                start["args"] = [_fill(str(arg), mapping) for arg in args]
        else:
            start = None
        services.append(
            ServiceSpec(
                id=str(item.get("id") or ""),
                name=str(item.get("name") or item.get("id") or ""),
                group=str(item.get("group") or "other"),
                description=str(item.get("description") or ""),
                managed=bool(item.get("managed")),
                required=bool(item.get("required")),
                auto_restart=bool(item.get("auto_restart")),
                health=str(item.get("health") or "port"),
                depends_on=[str(dep) for dep in item.get("depends_on") or [] if isinstance(dep, str)],
                host=str(item["host"]) if item.get("host") else None,
                port=int(item["port"]) if item.get("port") is not None else None,
                fallback_hosts=[str(host) for host in item.get("fallback_hosts") or [] if isinstance(host, str)],
                process_match=str(item["process_match"]) if item.get("process_match") else None,
                ready_timeout_s=float(item.get("ready_timeout_s") or 30.0),
                needs=[str(key) for key in item.get("needs") or [] if isinstance(key, str)],
                start=start,
                stop_args=[str(arg) for arg in item.get("stop_args") or []] or None,
            )
        )

    return DashboardConfig(
        host=str(dashboard.get("host") or "127.0.0.1"),
        port=int(dashboard.get("port") or 8788),
        poll_interval_s=float(dashboard.get("poll_interval_s") or 3),
        auto_restart_max=int(dashboard.get("auto_restart_max") or 3),
        development_mode=bool(dashboard.get("development_mode", True)),
        conversation_root=conversation_root,
        ai_stack_root=ai_stack_root,
        env=env,
        public_env=public_env_map(env),
        services=services,
    )
