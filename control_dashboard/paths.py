"""Filesystem locations used by the control dashboard."""

from __future__ import annotations
import os
import sys
from pathlib import Path


DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT = DASHBOARD_DIR.parent
RUNTIME_DIR = DASHBOARD_DIR / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"
STATIC_DIR = DASHBOARD_DIR / "static"
SERVICES_PATH = DASHBOARD_DIR / "services.json"
LOCAL_SERVICES_PATH = DASHBOARD_DIR / "services.local.json"
OWNED_PATH = RUNTIME_DIR / "owned.json"
STOPPED_PATH = RUNTIME_DIR / "stopped.json"
SETTINGS_PATH = RUNTIME_DIR / "settings.json"


def venv_python(root: Path) -> Path:
    """Return the venv interpreter for a project root."""
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def venv_script(root: Path, name: str) -> Path:
    """Return a venv console script path."""
    if sys.platform == "win32":
        return root / ".venv" / "Scripts" / f"{name}.exe"
    return root / ".venv" / "bin" / name


def default_ai_stack_root() -> Path:
    """Return the sibling local-AI folder when it exists."""
    candidate = REPO_ROOT.parent / "reachy-mini-local-ai"
    return candidate if candidate.is_dir() else REPO_ROOT.parent / "reachy-mini-local-ai"


def hermes_home() -> Path:
    """Return the Hermes Agent home directory."""
    local_app = os.environ.get("LOCALAPPDATA")
    if local_app:
        return Path(local_app) / "hermes"
    return Path.home() / ".hermes"


def ensure_runtime() -> None:
    """Create runtime directories used for logs and owned-process state."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
