import logging
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import simulator
from reachy_mini_conversation_app.utils import parse_args


def test_parse_args_accepts_no_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-sim is a recognized conversation-app flag."""
    monkeypatch.setattr("sys.argv", ["reachy-mini-conversation-app", "--no-sim"])

    args, _unknown = parse_args()

    assert args.no_sim is True


@pytest.mark.parametrize(
    ("no_sim", "daemon_host", "expected"),
    [
        (True, None, False),
        (False, None, True),
        (False, "", True),
        (False, "127.0.0.1", True),
        (False, "localhost", True),
        (False, "192.168.0.50", False),
        (False, "http://192.168.0.50:8000", False),
    ],
)
def test_should_launch_simulator(
    monkeypatch: pytest.MonkeyPatch, no_sim: bool, daemon_host: str | None, expected: bool
) -> None:
    """The sim starts locally unless --no-sim or a remote REACHY_DAEMON_HOST is set."""
    if daemon_host is None:
        monkeypatch.delenv("REACHY_DAEMON_HOST", raising=False)
    else:
        monkeypatch.setenv("REACHY_DAEMON_HOST", daemon_host)

    assert simulator.should_launch_simulator(no_sim) is expected


def test_ensure_simulator_running_reuses_existing_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing local daemon is reused instead of spawning a second simulator."""
    popen = MagicMock()
    bring = MagicMock()
    monkeypatch.setattr(simulator, "_daemon_ready", lambda: True)
    monkeypatch.setattr(simulator, "_bring_mujoco_window_to_front", bring)
    monkeypatch.setattr(simulator.subprocess, "Popen", popen)

    simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=False)

    popen.assert_not_called()
    bring.assert_called_once()


def test_ensure_simulator_running_spawns_until_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing local daemon is started and the app waits until /status answers."""
    ready_calls = {"count": 0}

    def fake_ready() -> bool:
        ready_calls["count"] += 1
        return ready_calls["count"] >= 2

    popen = MagicMock()
    monkeypatch.setattr(simulator, "_daemon_ready", fake_ready)
    monkeypatch.setattr(simulator, "_daemon_image_running", lambda: False)
    monkeypatch.setattr(simulator.subprocess, "Popen", popen)
    monkeypatch.setattr(simulator.time, "sleep", lambda _seconds: None)

    simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=False)

    popen.assert_called_once()
    command = popen.call_args.args[0]
    assert "--sim" in command
    assert "cmd.exe" not in command


def test_ensure_simulator_running_reports_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing reachy-mini-daemon executable is a connection error, not a crash."""
    monkeypatch.setattr(simulator, "_daemon_ready", lambda: False)
    monkeypatch.setattr(simulator, "_daemon_image_running", lambda: False)
    monkeypatch.setattr(simulator.subprocess, "Popen", MagicMock(side_effect=FileNotFoundError("missing")))

    with pytest.raises(ConnectionError, match="reachy-mini-daemon was not found"):
        simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=False)


def test_ensure_simulator_running_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup fails clearly if the simulator never answers."""
    monkeypatch.setattr(simulator, "SIM_READY_TIMEOUT_S", 0.0)
    monkeypatch.setattr(simulator, "_daemon_ready", lambda: False)
    monkeypatch.setattr(simulator, "_daemon_image_running", lambda: False)
    monkeypatch.setattr(simulator.subprocess, "Popen", MagicMock())
    monkeypatch.setattr(simulator.time, "sleep", lambda _seconds: None)
    monotonic_values = iter([0.0, 0.0, 1.0])
    monkeypatch.setattr(simulator.time, "monotonic", lambda: next(monotonic_values))

    with pytest.raises(ConnectionError, match="did not become ready"):
        simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=False)


def test_ensure_simulator_running_honors_no_sim_when_local_daemon_is_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-sim reuses a local simulator and does not spawn another."""
    popen = MagicMock()
    monkeypatch.delenv("REACHY_DAEMON_HOST", raising=False)
    monkeypatch.setattr(simulator, "_daemon_ready", lambda: True)
    monkeypatch.setattr(simulator.subprocess, "Popen", popen)

    simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=True)

    popen.assert_not_called()


def test_no_sim_without_local_daemon_does_not_fall_through_to_physical(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-sim must not let the SDK hunt for a physical robot when localhost is empty."""
    popen = MagicMock()
    monkeypatch.delenv("REACHY_DAEMON_HOST", raising=False)
    monkeypatch.setattr(simulator, "_daemon_ready", lambda: False)
    monkeypatch.setattr(simulator.subprocess, "Popen", popen)

    with pytest.raises(ConnectionError, match="127.0.0.1:8000"):
        simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=True)

    popen.assert_not_called()


def test_ensure_simulator_running_reuses_booting_daemon_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A MuJoCo daemon that has not answered HTTP yet must not be launched twice."""
    popen = MagicMock()
    monkeypatch.setattr(simulator, "_daemon_ready", lambda: False)
    monkeypatch.setattr(simulator, "_daemon_image_running", lambda: True)
    monkeypatch.setattr(simulator, "_bring_mujoco_window_to_front", MagicMock())
    monkeypatch.setattr(simulator.subprocess, "Popen", popen)

    simulator.ensure_simulator_running(logging.getLogger("test"), no_sim=False)

    popen.assert_not_called()
