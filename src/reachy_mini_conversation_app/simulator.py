"""Launch the local Reachy Mini MuJoCo simulator when the conversation app starts."""

import os
import sys
import json
import time
import ctypes
import logging
import subprocess
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen
from collections.abc import Callable

from reachy_mini_conversation_app.config import PROJECT_ROOT


logger = logging.getLogger(__name__)

SIM_DAEMON_HOST = "127.0.0.1"
SIM_DAEMON_PORT = 8000
SIM_READY_TIMEOUT_S = 90.0
_READY_POLL_S = 1.0
_STATUS_URL = f"http://{SIM_DAEMON_HOST}:{SIM_DAEMON_PORT}/api/daemon/status"
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def should_launch_simulator(no_sim: bool) -> bool:
    """Return whether this process should start the local MuJoCo simulator."""
    if no_sim:
        return False
    host = _configured_daemon_host()
    if host is None:
        return True
    return _is_loopback_host(host)


def _remote_physical_host() -> bool:
    host = _configured_daemon_host()
    return host is not None and not _is_loopback_host(host)


def ensure_simulator_running(app_logger: logging.Logger, *, no_sim: bool) -> None:
    """Start reachy-mini-daemon --sim only when no local daemon is already answering."""
    if not should_launch_simulator(no_sim):
        if no_sim and not _remote_physical_host() and not _daemon_ready():
            raise ConnectionError(
                "Reachy Mini simulator is not listening on 127.0.0.1:8000. "
                "Start Reachy Mini from the control dashboard, or omit --no-sim."
            )
        app_logger.info("Skipping Reachy Mini simulator launch")
        return
    if _daemon_ready() or _daemon_image_running():
        _bring_mujoco_window_to_front()
        app_logger.info("Reachy Mini daemon already running on %s:%s", SIM_DAEMON_HOST, SIM_DAEMON_PORT)
        return

    app_logger.info("Starting Reachy Mini simulator (reachy-mini-daemon --sim)")
    try:
        _spawn_simulator()
    except FileNotFoundError as exc:
        raise ConnectionError(
            "reachy-mini-daemon was not found. Activate the conversation app venv and retry."
        ) from exc
    except OSError as exc:
        raise ConnectionError(f"Failed to start reachy-mini-daemon: {exc}") from exc

    if not _wait_until(_daemon_ready, SIM_READY_TIMEOUT_S):
        raise ConnectionError(
            f"Reachy Mini simulator did not become ready at {_STATUS_URL} within {SIM_READY_TIMEOUT_S:.0f}s."
        )
    app_logger.info("Reachy Mini simulator is ready")


def _configured_daemon_host() -> str | None:
    raw = (os.getenv("REACHY_DAEMON_HOST") or "").strip()
    if not raw:
        return None
    if "://" in raw:
        return urlsplit(raw).hostname
    host = raw.split("/", 1)[0]
    if host.startswith("[") and "]" in host:
        return host[1 : host.index("]")]
    if host.count(":") == 1:
        return host.split(":", 1)[0] or None
    return host or None


def _is_loopback_host(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    return name in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or name.startswith("127.")


def _daemon_status() -> dict[str, object] | None:
    try:
        with urlopen(_STATUS_URL, timeout=0.5) as response:
            if not (200 <= int(response.status) < 300):
                return None
            payload: object = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _daemon_ready() -> bool:
    return _daemon_status() is not None


def _bring_mujoco_window_to_front() -> None:
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    for hwnd in _mujoco_window_hwnds():
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)


def _mujoco_window_hwnds() -> list[int]:
    if sys.platform != "win32":
        return []
    user32 = ctypes.windll.user32
    found: list[int] = []
    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t)

    def _enum(hwnd: int, _lparam: int) -> int:
        if hwnd == 0 or not user32.IsWindowVisible(hwnd):
            return 1
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return 1
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if "mujoco" in buf.value.lower():
            found.append(hwnd)
        return 1

    user32.EnumWindows(enum_proc(_enum), 0)
    return found


def _wait_until(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(_READY_POLL_S)
    return predicate()


def _daemon_executable() -> str:
    if sys.platform == "win32":
        candidate = PROJECT_ROOT / ".venv" / "Scripts" / "reachy-mini-daemon.exe"
    else:
        candidate = PROJECT_ROOT / ".venv" / "bin" / "reachy-mini-daemon"
    if candidate.is_file():
        return str(candidate)
    return "reachy-mini-daemon"


def _simulator_command() -> list[str]:
    return [
        _daemon_executable(),
        "--sim",
        "--fastapi-host",
        SIM_DAEMON_HOST,
        "--fastapi-port",
        str(SIM_DAEMON_PORT),
    ]


def _daemon_image_running() -> bool:
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", "IMAGENAME eq reachy-mini-daemon.exe", "/FO", "CSV", "/NH"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        lowered = output.lower()
        return "reachy-mini-daemon.exe" in lowered and "no tasks" not in lowered
    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "command="],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return False
    return any("reachy-mini-daemon" in line for line in output.splitlines())


def _spawn_simulator() -> None:
    command = _simulator_command()
    if sys.platform == "win32":
        # Keep a real child PID. `cmd /c start` exits immediately and a second
        # launch then binds port 8443 and opens another MuJoCo window.
        subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            creationflags=_CREATE_NEW_PROCESS_GROUP | _CREATE_NEW_CONSOLE,
        )
    else:
        subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    logger.info("Spawned %s", " ".join(command))
