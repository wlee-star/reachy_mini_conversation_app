"""Start, inspect, and stop whitelisted local processes."""

from __future__ import annotations
import os
import re
import sys
import time
import signal
import logging
import subprocess
from typing import IO, Any
from pathlib import Path


logger = logging.getLogger(__name__)

_LISTEN_CACHE: tuple[float, dict[int, list[int]]] | None = None
_LISTEN_TTL_S = 1.5
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
_LOG_HANDLES: dict[str, IO[str]] = {}


def listening_table() -> dict[int, list[int]]:
    """Return a port-to-PID map of TCP listeners, cached briefly."""
    global _LISTEN_CACHE
    now = time.time()
    if _LISTEN_CACHE is not None and now - _LISTEN_CACHE[0] < _LISTEN_TTL_S:
        return _LISTEN_CACHE[1]
    table = _listening_table_windows() if sys.platform == "win32" else _listening_table_posix()
    _LISTEN_CACHE = (now, table)
    return table


def listening_pids(port: int) -> list[int]:
    """Return PIDs listening on a TCP port."""
    return list(listening_table().get(port, []))


def _listening_table_windows() -> dict[int, list[int]]:
    try:
        output = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("netstat failed: %s", exc)
        return {}
    table: dict[int, list[int]] = {}
    pattern = re.compile(r":(\d+)\s+\S+\s+LISTENING\s+(\d+)", re.IGNORECASE)
    for line in output.splitlines():
        match = pattern.search(line)
        if match is None:
            continue
        port = int(match.group(1))
        pid = int(match.group(2))
        if pid and pid not in table.setdefault(port, []):
            table[port].append(pid)
    return table


def _listening_table_posix() -> dict[int, list[int]]:
    table: dict[int, list[int]] = {}
    commands = (
        ["ss", "-ltnp"],
        ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
    )
    for command in commands:
        try:
            output = subprocess.check_output(command, text=True, timeout=5, stderr=subprocess.DEVNULL)
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
        for match in re.finditer(r":(\d+)\b", output):
            port = int(match.group(1))
            pids = [int(pid) for pid in re.findall(r"pid=(\d+)", output[match.start() : match.start() + 120])]
            for pid in pids:
                if pid and pid not in table.setdefault(port, []):
                    table[port].append(pid)
        if table:
            return table
    return table


def process_command_line(pid: int) -> str | None:
    """Return the command line for a PID, if it can be read."""
    if pid <= 0:
        return None
    if sys.platform == "win32":
        return _windows_command_line(pid)
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    line = output.strip()
    return line or None


def _windows_command_line(pid: int) -> str | None:
    script = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("Could not read command line for PID %s: %s", pid, exc)
        return None
    line = output.strip()
    return line or None


def process_name(pid: int) -> str | None:
    """Return the executable name for a PID."""
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return None
        if "PID" in output and "eq" in output:
            return None
        parts = [part.strip('"') for part in output.strip().split(",")]
        return parts[0] if parts and parts[0] else None
    try:
        output = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "comm="],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return output.strip() or None


def command_matches(command_line: str | None, pattern: str) -> bool:
    """Return whether a process command line matches a service whitelist pattern."""
    if not command_line or not pattern:
        return False
    return re.search(pattern, command_line, re.IGNORECASE) is not None


_VENV_LEAK_KEYS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "UV_PROJECT",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON",
    # Dashboard health checks import reachy_mini, which mutates these on the parent.
    # GStreamer then prepends them again in the child and gi cannot be imported.
    "GST_PYTHONPATH_1_0",
    "GI_TYPELIB_PATH",
    "GST_PLUGIN_PATH_1_0",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "GST_REGISTRY_1_0",
    "GST_PLUGIN_SCANNER_1_0",
    "PYGI_DLL_DIRS",
)


def child_env(executable: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a child environment that does not inherit the dashboard venv."""
    popen_env = os.environ.copy()
    for key in _VENV_LEAK_KEYS:
        popen_env.pop(key, None)
    if extra:
        popen_env.update(extra)
    exe_dir = Path(executable).resolve().parent
    popen_env["PATH"] = str(exe_dir) + os.pathsep + popen_env.get("PATH", "")
    venv_root = exe_dir.parent
    if (venv_root / "pyvenv.cfg").is_file():
        popen_env["VIRTUAL_ENV"] = str(venv_root)
    popen_env["PYTHONNOUSERSITE"] = "1"
    popen_env["PYTHONUNBUFFERED"] = "1"
    popen_env["PYTHONUTF8"] = "1"
    return popen_env


def start_process(
    service_id: str,
    executable: str,
    args: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    *,
    gui: bool = False,
) -> int:
    """Start a detached process and return its PID."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    previous = _LOG_HANDLES.pop(service_id, None)
    if previous is not None:
        previous.close()
    log_file = log_path.open("a", encoding="utf-8")
    _LOG_HANDLES[service_id] = log_file
    popen_env = child_env(executable, extra=env)
    kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": popen_env,
    }
    if sys.platform == "win32":
        flags = _CREATE_NEW_PROCESS_GROUP
        # A new console lets the MuJoCo viewer surface. Do not use `cmd /c start`:
        # that wrapper exits immediately and recover_once kills the live simulator.
        flags |= _CREATE_NEW_CONSOLE if gui else _CREATE_NO_WINDOW
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    started = subprocess.Popen([executable, *args], **kwargs)
    logger.info("Started %s pid=%s", service_id, started.pid)
    return started.pid


def pids_matching(pattern: str) -> list[int]:
    """Return PIDs whose command line matches the whitelist regex."""
    if not pattern:
        return []
    if sys.platform == "win32":
        return _windows_pids_matching(pattern)
    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,command="],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("ps failed while matching processes: %s", exc)
        return []
    pids: list[int] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if pid_text.isdigit() and command_matches(command, pattern):
            pids.append(int(pid_text))
    return pids


def _windows_pids_matching(pattern: str) -> list[int]:
    encoded = pattern.replace("'", "''")
    script = (
        f"$pat = '{encoded}'; "
        "Get-CimInstance Win32_Process | ForEach-Object { "
        "if ($_.CommandLine -and ($_.CommandLine -match $pat)) { $_.ProcessId } }"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            text=True,
            timeout=20,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        logger.warning("Could not list processes matching %s: %s", pattern, exc)
        return []
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]


def pid_is_running(pid: int) -> bool:
    """Return whether a PID is still alive."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            output = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                text=True,
                timeout=5,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
        return str(pid) in output and "No tasks" not in output
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stop_pid(pid: int) -> None:
    """Stop a process tree by PID."""
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        logger.warning("SIGTERM failed for pid %s: %s", pid, exc)
