"""Behavior tests for the local control dashboard."""

import os
from unittest.mock import MagicMock

import pytest

from control_dashboard.net import HttpResult
from control_dashboard.proc import child_env
from control_dashboard.stack import StackController, topological
from control_dashboard.checks import (
    STATUS_ONLINE,
    STATUS_OFFLINE,
    STATUS_DEGRADED,
    STATUS_STARTING,
    HealthResult,
    diagnose,
    check_speech,
    check_reachy_daemon,
)
from control_dashboard.redact import (
    mask_secret,
    redact_text,
    is_secret_key,
    parse_env_file,
    public_env_map,
)
from control_dashboard.registry import load_config


def test_secrets_are_masked_and_never_echoed() -> None:
    """Dashboard output must not include raw tokens or bearer values."""
    parsed = parse_env_file("HF_TOKEN=hf_super_secret\nHA_TOKEN=ha_secret\nHA_URL=http://homeassistant.local:8123\n")
    public = public_env_map(parsed)
    assert is_secret_key("HA_TOKEN")
    assert public["HF_TOKEN"] == "configured"
    assert public["HA_TOKEN"] == "configured"
    assert public["HA_URL"] == "http://homeassistant.local:8123"
    assert "ha_secret" not in str(public)
    assert mask_secret("abc") == "********"
    assert "sekrit" not in redact_text("Authorization: Bearer sekrit")
    assert "sekrit" not in redact_text("HERMES_API_KEY=sekrit")


def test_registry_discovers_real_project_services() -> None:
    """The registry must describe the services that actually exist in this project."""
    config = load_config()
    ids = {spec.id for spec in config.services}
    assert ids >= {
        "llama",
        "speech",
        "qwen_tts",
        "hermes",
        "conversation",
        "reachy_daemon",
        "home_assistant",
        "apex",
        "bus",
    }
    llama = config.service("llama")
    speech = config.service("speech")
    conversation = config.service("conversation")
    assert llama is not None and llama.port == 8080 and llama.managed
    assert speech is not None and speech.depends_on == ["llama"]
    assert conversation is not None and "speech" in conversation.depends_on
    assert conversation is not None and "reachy_daemon" in conversation.depends_on
    assert conversation.start is not None
    assert "--no-sim" in conversation.start.get("args", [])
    bus = config.service("bus")
    assert bus is not None and bus.managed is False
    reachy = config.service("reachy_daemon")
    assert reachy is not None and reachy.managed is True
    assert reachy.start is not None
    assert "--sim" in reachy.start.get("args", [])
    assert reachy.process_match is not None
    assert "--sim" in reachy.process_match
    assert "--headless" not in reachy.start.get("args", [])
    assert "--no-media" not in reachy.start.get("args", [])
    assert reachy.start.get("gui") is True
    assert "HA_TOKEN" not in str(config.public_env.values()) or all(
        value in {"configured", "not configured"} or "TOKEN" not in key for key, value in config.public_env.items()
    )


def test_start_skips_healthy_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start must not launch a duplicate when the health check already passes."""
    config = load_config()
    llama = config.service("llama")
    assert llama is not None
    controller = StackController(config)
    controller._owned = {}
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_ONLINE, "already up"),
    )
    started = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [321])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [321])
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: "llama-server --host 127.0.0.1 --port 8080",
    )
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    result = controller.start(llama)
    assert result["ok"] is True
    assert result["already_running"] is True
    assert result["external"] is True
    started.assert_not_called()
    assert "llama" not in controller._owned


def test_stop_refuses_unrelated_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop must not kill a process whose command line is outside the whitelist."""
    config = load_config()
    llama = config.service("llama")
    assert llama is not None
    controller = StackController(config)
    controller._owned = {}
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda port: [4242])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.proc.process_command_line", lambda pid: "notepad.exe")
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    killed = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.stop_pid", killed)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    result = controller.stop(llama)
    assert result["ok"] is False
    killed.assert_not_called()


def test_start_all_is_dependency_ordered() -> None:
    """Speech must come after llama; conversation after speech."""
    config = load_config()
    order = [spec.id for spec in topological(config.services) if spec.managed]
    assert order.index("llama") < order.index("speech")
    assert order.index("speech") < order.index("hermes")
    assert order.index("hermes") < order.index("reachy_daemon")
    assert order.index("speech") < order.index("conversation")
    assert order.index("reachy_daemon") < order.index("conversation")


def test_system_not_ready_names_offline_required_service() -> None:
    """Overall readiness must explain which required service is down."""
    config = load_config()
    controller = StackController(config)
    snapshots = [
        {
            "id": "llama",
            "name": "llama.cpp",
            "required": True,
            "status": STATUS_OFFLINE,
            "reason": "llama.cpp is not listening on 127.0.0.1:8080.",
            "suggested_action": "Start llama.cpp",
        },
        {
            "id": "speech",
            "name": "Speech-to-speech",
            "required": True,
            "status": STATUS_ONLINE,
            "reason": None,
            "suggested_action": None,
        },
    ]
    readiness = controller.system_readiness(snapshots)
    assert readiness["ready"] is False
    assert readiness["label"] == "SYSTEM NOT READY"
    assert readiness["blockers"][0]["id"] == "llama"


def test_reachy_offline_explains_missing_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reachy offline copy must name the local simulator."""
    config = load_config()
    spec = config.service("reachy_daemon")
    assert spec is not None
    monkeypatch.setattr("control_dashboard.checks._configured_daemon_host", lambda _config: None)
    monkeypatch.setattr(
        "control_dashboard.net.host_resolves",
        lambda host: host not in {"reachy-mini.local"},
    )
    monkeypatch.setattr("control_dashboard.net.port_open", lambda *args, **kwargs: False)
    result = check_reachy_daemon(spec, config, {})
    assert result.status == STATUS_OFFLINE
    assert "127.0.0.1:8000" in result.summary
    assert "virtual Reachy Mini" in result.summary
    assert "expected on port 8000" not in diagnose(spec, result, config)


def test_child_env_does_not_leak_dashboard_venv(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Speech-to-speech must not see the conversation app VIRTUAL_ENV or PYTHONPATH."""
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "conversation-venv"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "conversation-venv" / "Lib" / "site-packages"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "conversation-venv"))
    monkeypatch.setenv("GST_PYTHONPATH_1_0", str(tmp_path / "stale-gi") + os.pathsep + str(tmp_path / "stale-gi"))
    service_venv = tmp_path / "s2s-venv"
    scripts = service_venv / "Scripts"
    scripts.mkdir(parents=True)
    (service_venv / "pyvenv.cfg").write_text("home = python\n", encoding="utf-8")
    exe = scripts / "speech-to-speech.exe"
    exe.write_text("", encoding="utf-8")
    env = child_env(str(exe))
    assert env["VIRTUAL_ENV"] == str(service_venv)
    assert "PYTHONPATH" not in env
    assert "PYTHONHOME" not in env
    assert "GST_PYTHONPATH_1_0" not in env
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PATH"].split(os.pathsep)[0] == str(scripts.resolve())


def test_gui_start_opens_a_console_on_windows(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MuJoCo must keep a real child PID and a visible console, not a cmd.exe wrapper."""
    import sys

    from control_dashboard import proc as proc_mod

    popen = MagicMock(return_value=MagicMock(pid=4242))
    monkeypatch.setattr(proc_mod.subprocess, "Popen", popen)
    exe = tmp_path / "reachy-mini-daemon.exe"
    exe.write_text("", encoding="utf-8")
    pid = proc_mod.start_process(
        "reachy_daemon",
        str(exe),
        ["--sim"],
        tmp_path,
        tmp_path / "daemon.log",
        gui=True,
    )
    command = popen.call_args.args[0]
    kwargs = popen.call_args.kwargs
    if sys.platform == "win32":
        assert pid == 4242
        assert command == [str(exe), "--sim"]
        flags = kwargs.get("creationflags", 0)
        assert flags & proc_mod._CREATE_NEW_CONSOLE
        assert not flags & proc_mod._CREATE_NO_WINDOW
    else:
        assert pid == 4242
        assert kwargs.get("start_new_session") is True


def test_recover_does_not_restart_during_ready_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead wrapper PID must not relaunch MuJoCo while the daemon is still starting."""
    from datetime import datetime, timezone

    from control_dashboard.stack import StackController

    config = load_config()
    controller = StackController(config)
    controller._owned = {
        "reachy_daemon": {
            "pid": 1,
            "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
    }
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "not listening yet"),
    )
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    started = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    controller.recover_once()
    started.assert_not_called()


def test_recover_restarts_conversation_inside_ready_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crashed conversation app must be recovered even shortly after start."""
    from datetime import datetime, timezone

    from control_dashboard.stack import StackController

    config = load_config()
    controller = StackController(config)
    controller._owned = {
        "conversation": {
            "pid": 1,
            "started_by_dashboard": True,
            "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        }
    }
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    started = MagicMock(return_value={"ok": True, "status": STATUS_ONLINE})
    monkeypatch.setattr(controller, "start", started)
    controller.recover_once()
    started.assert_called_once()


def test_recover_adopts_listening_pid_when_wrapper_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once the daemon is healthy, replace the dead cmd.exe PID with the listener."""
    from control_dashboard.stack import StackController

    config = load_config()
    controller = StackController(config)
    controller._owned = {"reachy_daemon": {"pid": 1, "started_at": "2026-08-29T09:58:34+10:00"}}
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda pid: pid != 1)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [555])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [555])
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: "reachy-mini-daemon --sim --fastapi-port 8000",
    )
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_ONLINE, "up"),
    )
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    started = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    controller.recover_once()
    started.assert_not_called()
    assert controller._owned["reachy_daemon"]["pid"] == 555


def test_recover_does_not_relaunch_when_sim_is_already_listening(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead wrapper PID must not kill a live MuJoCo daemon that is still on port 8000."""
    from control_dashboard.stack import StackController

    config = load_config()
    controller = StackController(config)
    controller._owned = {"reachy_daemon": {"pid": 1, "started_at": "2020-01-01T00:00:00+10:00"}}
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda pid: pid != 1)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [777])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [777])
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: "reachy-mini-daemon --sim --fastapi-port 8000",
    )
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "probe missed"),
    )
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    started = MagicMock()
    monkeypatch.setattr(controller, "start", started)
    controller.recover_once()
    started.assert_not_called()
    assert controller._owned["reachy_daemon"]["pid"] == 777


def test_start_adopts_booting_simulator_instead_of_spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live MuJoCo process must be reused even before port 8000 answers."""
    config = load_config()
    reachy = config.service("reachy_daemon")
    assert reachy is not None
    controller = StackController(config)
    controller._owned = {}
    states = iter(
        [
            HealthResult(STATUS_OFFLINE, "not listening yet"),
            HealthResult(STATUS_ONLINE, "up"),
        ]
    )
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: next(states, HealthResult(STATUS_ONLINE, "up")),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [888])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: "reachy-mini-daemon --sim --fastapi-host 127.0.0.1 --fastapi-port 8000",
    )
    started = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    stopped = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.stop_pid", stopped)
    result = controller.start(reachy)
    assert result["ok"] is True
    started.assert_not_called()
    stopped.assert_not_called()
    assert "reachy_daemon" not in controller._owned
    assert result["external"] is True


def test_start_stops_leftover_matching_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A leftover conversation client must be killed before a new start claims the speech slot."""
    from control_dashboard.stack import StackController

    config = load_config()
    conversation = config.service("conversation")
    assert conversation is not None
    controller = StackController(config)
    states = iter(
        [
            HealthResult(STATUS_OFFLINE, "down"),
            HealthResult(STATUS_ONLINE, "up"),
        ]
    )
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: next(states, HealthResult(STATUS_ONLINE, "up")),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.shutil_which", lambda exe: exe)
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [99])
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    monkeypatch.setattr("control_dashboard.stack.proc.wait_until_gone", lambda _pid, timeout_s=8.0: True)
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: "python -m reachy_mini_conversation_app.main --ui",
    )
    stopped = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.stop_pid", stopped)
    started = MagicMock(return_value=123)
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    monkeypatch.setattr(
        "control_dashboard.stack.net.http_request",
        lambda *_a, **_k: HttpResult(
            True, 200, '{"size":1,"in_use":0,"units":[{"index":0,"state":"idle"}]}', 1.0, None
        ),
    )
    result = controller.start(conversation)
    assert result["ok"] is True
    stopped.assert_called_once_with(99)
    started.assert_called_once()


def test_speech_health_degrades_when_pool_slot_is_stuck(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quarantined speech-to-speech pipeline must surface as degraded, not healthy."""
    config = load_config()
    spec = config.service("speech")
    assert spec is not None
    monkeypatch.setattr("control_dashboard.checks.net.port_open", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "control_dashboard.checks._process_snapshot",
        lambda _spec, include_command=False: {
            "pids": [1],
            "command": "python -m speech_to_speech.s2s_pipeline --tts qwen3",
            "name": "python",
        },
    )

    def fake_http(url: str, **_k: object) -> HttpResult:
        if str(url).endswith("/v1/pool"):
            return HttpResult(True, 200, '{"size":1,"in_use":1,"units":[{"index":0,"state":"stuck"}]}', 1.0, None)
        return HttpResult(True, 200, '{"data":[]}', 1.0, None)

    monkeypatch.setattr("control_dashboard.checks.net.http_request", fake_http)
    result = check_speech(spec, config, {})
    assert result.status == STATUS_DEGRADED
    assert result.suggested_action == "Restart speech-to-speech"
    assert result.details["pool"]["in_use"] == 1


def test_start_waits_for_busy_speech_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conversation start must wait until speech-to-speech has an idle realtime unit."""
    config = load_config()
    conversation = config.service("conversation")
    assert conversation is not None
    controller = StackController(config)
    states = iter([HealthResult(STATUS_OFFLINE, "down"), HealthResult(STATUS_ONLINE, "up")])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: next(states, HealthResult(STATUS_ONLINE, "up")),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.shutil_which", lambda exe: exe)
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    started = MagicMock(return_value=123)
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    sleeps: list[float] = []
    monkeypatch.setattr("control_dashboard.stack.time.sleep", lambda seconds: sleeps.append(seconds))
    calls = {"n": 0}

    def fake_http(_url: str, **_k: object) -> HttpResult:
        calls["n"] += 1
        if calls["n"] == 1:
            return HttpResult(True, 200, '{"size":1,"in_use":1,"units":[{"index":0,"state":"active"}]}', 1.0, None)
        return HttpResult(True, 200, '{"size":1,"in_use":0,"units":[{"index":0,"state":"idle"}]}', 1.0, None)

    monkeypatch.setattr("control_dashboard.stack.net.http_request", fake_http)
    result = controller.start(conversation)
    assert result["ok"] is True
    assert calls["n"] >= 2
    assert sleeps == [0.5]
    started.assert_called_once()


def _dead_owned(controller: StackController, service_id: str) -> None:
    controller._owned = {
        service_id: {"pid": 1, "started_at": "2020-01-01T00:00:00+10:00"},
    }


def test_recover_stops_after_max_restarts(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the restart budget is spent, recover must log once and stop trying."""
    config = load_config()
    controller = StackController(config)
    _dead_owned(controller, "speech")
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    started = MagicMock(return_value={"ok": False, "error": "did not become healthy"})
    monkeypatch.setattr(controller, "start", started)

    for _ in range(config.auto_restart_max):
        controller.recover_once()
    assert started.call_count == config.auto_restart_max
    assert "speech" in controller._owned

    controller.recover_once()
    controller.recover_once()
    assert started.call_count == config.auto_restart_max
    assert "speech" not in controller._owned
    give_up = [item for item in controller._op_log if "stopping retries" in item["message"]]
    assert len(give_up) == 1


def test_recover_does_not_burn_retries_when_dependency_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead speech process must wait for llama.cpp instead of consuming the restart budget."""
    config = load_config()
    controller = StackController(config)
    _dead_owned(controller, "speech")
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(
        controller,
        "_blocked_by",
        lambda _spec, **_k: [{"id": "llama", "name": "llama.cpp", "status": STATUS_OFFLINE, "summary": "down"}],
    )
    started = MagicMock()
    monkeypatch.setattr(controller, "start", started)
    controller.recover_once()
    started.assert_not_called()
    assert controller._restarts.get("speech", 0) == 0
    assert "speech" in controller._owned


def test_recover_skips_service_that_is_already_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start all and recover must not launch a second copy of the same service."""
    config = load_config()
    controller = StackController(config)
    _dead_owned(controller, "hermes")
    controller._starting.add("hermes")
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    started = MagicMock()
    monkeypatch.setattr(controller, "start", started)
    controller.recover_once()
    started.assert_not_called()


def test_start_waits_when_service_is_already_starting(monkeypatch: pytest.MonkeyPatch) -> None:
    """A service that is still loading must be waited on, not reported as already up."""
    config = load_config()
    llama = config.service("llama")
    assert llama is not None
    controller = StackController(config)
    states = iter([HealthResult(STATUS_STARTING, "loading"), HealthResult(STATUS_ONLINE, "up")])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: next(states, HealthResult(STATUS_ONLINE, "up")),
    )
    started = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    result = controller.start(llama)
    assert result["ok"] is True
    started.assert_not_called()


def test_blocked_by_rechecks_stale_cache_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start must not fail because a previous poll left a dependency marked offline."""
    config = load_config()
    conversation = config.service("conversation")
    assert conversation is not None
    controller = StackController(config)
    for dep_id in conversation.depends_on:
        controller._last_health[dep_id] = HealthResult(STATUS_OFFLINE, "down")
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_ONLINE, "up"),
    )
    assert controller._blocked_by(conversation)
    assert controller._blocked_by(conversation, fresh=True) == []


def test_hermes_process_match_ignores_unrelated_commands() -> None:
    """Hermes leftover detection must not match the conversation app or test files."""
    from control_dashboard.proc import command_matches

    config = load_config()
    hermes = config.service("hermes")
    assert hermes is not None and hermes.process_match
    assert command_matches(
        r"C:\Users\me\AppData\Local\hermes\bin\hermes.exe gateway start",
        hermes.process_match,
    )
    assert command_matches(
        '"C:\\Users\\me\\hermes-agent\\venv\\Scripts\\python.exe" -m hermes_cli.main gateway run',
        hermes.process_match,
    )
    assert command_matches('"C:\\hermes.exe" gateway stop', hermes.process_match)
    assert command_matches("hermes-gateway --port 8642", hermes.process_match)
    assert not command_matches(
        r'"C:\Users\me\AppData\Local\hermes\hermes-agent\apps\desktop\release\win-unpacked\Hermes.exe"',
        hermes.process_match,
    )
    assert not command_matches("python -m hermes_cli.main serve --host 127.0.0.1 --port 0", hermes.process_match)
    assert not command_matches("python -m reachy_mini_conversation_app.main --ui", hermes.process_match)
    assert not command_matches("python tests/test_hermes_client.py", hermes.process_match)


def test_stop_refuses_stale_owned_pid_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stored PID must not be killed after it has been reused by another process."""
    config = load_config()
    llama = config.service("llama")
    assert llama is not None
    controller = StackController(config)
    controller._owned = {"llama": {"pid": 4242, "started_at": "2020-01-01T00:00:00+10:00"}}
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [4242])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.proc.process_command_line", lambda _pid: "notepad.exe")
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    killed = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.stop_pid", killed)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    result = controller.stop(llama)
    assert result["ok"] is False
    killed.assert_not_called()


def test_start_reuses_running_hermes_without_killing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live Hermes gateway must be adopted, not killed and relaunched."""
    config = load_config()
    hermes = config.service("hermes")
    assert hermes is not None
    controller = StackController(config)
    controller._owned = {}
    states = iter([HealthResult(STATUS_OFFLINE, "down"), HealthResult(STATUS_ONLINE, "up")])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: next(states, HealthResult(STATUS_ONLINE, "up")),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [555])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [555])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: '"C:\\Users\\me\\hermes-agent\\venv\\Scripts\\python.exe" -m hermes_cli.main gateway run',
    )
    started = MagicMock()
    stopped = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    monkeypatch.setattr("control_dashboard.stack.proc.stop_pid", stopped)
    monkeypatch.setattr("control_dashboard.stack.proc.run_logged", MagicMock(return_value=0))
    result = controller.start(hermes)
    assert result["ok"] is True
    started.assert_not_called()
    stopped.assert_not_called()
    assert "hermes" not in controller._owned
    assert result["external"] is True


def test_recover_does_not_restart_hermes_after_unexpected_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermes must not be auto-relaunched; unexpected exit is logged instead."""
    config = load_config()
    controller = StackController(config)
    controller._owned = {"hermes": {"pid": 1, "started_at": "2020-01-01T00:00:00+10:00"}}
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    started = MagicMock()
    monkeypatch.setattr(controller, "start", started)
    controller.recover_once()
    started.assert_not_called()
    unexpected = [item for item in controller._op_log if "UNEXPECTED EXIT" in item["message"]]
    assert unexpected


def test_stop_runs_hermes_stop_command_synchronously(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermes gateway stop must finish before stop() returns, so it cannot kill a later start."""
    config = load_config()
    hermes = config.service("hermes")
    assert hermes is not None
    controller = StackController(config)
    controller._owned = {"hermes": {"pid": 9, "started_at": "2020-01-01T00:00:00+10:00"}}
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.shutil_which", lambda exe: exe)
    run_logged = MagicMock(return_value=0)
    monkeypatch.setattr("control_dashboard.stack.proc.run_logged", run_logged)
    detached = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", detached)
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    result = controller.stop(hermes)
    assert result["ok"] is True
    run_logged.assert_called_once()
    assert run_logged.call_args.args[1] == ["gateway", "stop"]
    detached.assert_not_called()


def test_reachy_health_ignores_physical_robot_on_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wireless robot must not make the managed Reachy Mini card look online."""
    config = load_config()
    spec = config.service("reachy_daemon")
    assert spec is not None
    monkeypatch.setattr("control_dashboard.checks._configured_daemon_host", lambda _config: None)
    monkeypatch.setattr("control_dashboard.net.host_resolves", lambda _host: True)
    monkeypatch.setattr(
        "control_dashboard.net.port_open",
        lambda host, _port, timeout_s=1.0: host == "192.168.0.50",
    )
    monkeypatch.setattr(
        "control_dashboard.checks.net.http_request",
        lambda url, **_k: (
            HttpResult(True, 200, '{"state":"ok"}', 1.0, None)
            if "192.168.0.50" in url
            else HttpResult(False, None, "", 1.0, "refused")
        ),
    )
    result = check_reachy_daemon(spec, config, {})
    assert result.status == STATUS_OFFLINE
    assert "127.0.0.1:8000" in result.summary


def test_start_does_not_adopt_physical_desktop_on_port_8000(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Reachy Mini desktop app is not the MuJoCo simulator and must not be reused."""
    config = load_config()
    reachy = config.service("reachy_daemon")
    assert reachy is not None
    controller = StackController(config)
    controller._owned = {}
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [999])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: r"C:\Program Files\Reachy Mini\Reachy Mini.exe",
    )
    monkeypatch.setattr("control_dashboard.stack.shutil_which", lambda exe: exe)
    started = MagicMock(return_value=123)
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", started)
    result = controller.start(reachy)
    assert result["ok"] is False
    assert "Port 8000" in str(result.get("error"))
    started.assert_not_called()


def test_start_fails_fast_when_conversation_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation process that dies during connect must not hold Stop all for 90s."""
    config = load_config()
    conversation = config.service("conversation")
    assert conversation is not None
    controller = StackController(config)
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.shutil_which", lambda exe: exe)
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", MagicMock(return_value=99))
    monkeypatch.setattr(
        "control_dashboard.stack.net.http_request",
        lambda *_a, **_k: HttpResult(
            True, 200, '{"size":1,"in_use":0,"units":[{"index":0,"state":"idle"}]}', 1.0, None
        ),
    )
    sleeps: list[float] = []
    monkeypatch.setattr("control_dashboard.stack.time.sleep", lambda seconds: sleeps.append(seconds))
    result = controller.start(conversation)
    assert result["ok"] is False
    assert "exited before becoming healthy" in str(result.get("error"))
    assert 2.0 not in sleeps


def test_user_stopped_survives_controller_reload(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop all must still suppress auto-restart after the dashboard process restarts."""
    from control_dashboard import paths as dashboard_paths

    stopped_path = tmp_path / "stopped.json"
    monkeypatch.setattr(dashboard_paths, "STOPPED_PATH", stopped_path)
    config = load_config()
    first = StackController(config)
    conversation = config.service("conversation")
    assert conversation is not None
    first._mark_user_stopped(conversation)
    second = StackController(config)
    assert "conversation" in second._user_stopped


def test_acknowledge_intentional_stop_prevents_auto_restart(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sleep/ack_stop exit must not be restarted even while auto_restart stays enabled."""
    from control_dashboard import paths as dashboard_paths
    from control_dashboard.stack import StackController

    monkeypatch.setattr(dashboard_paths, "STOPPED_PATH", tmp_path / "stopped.json")
    monkeypatch.setattr(dashboard_paths, "OWNED_PATH", tmp_path / "owned.json")
    config = load_config()
    controller = StackController(config)
    conversation = config.service("conversation")
    assert conversation is not None
    controller._owned = {
        "conversation": {"pid": 1, "started_by_dashboard": True, "started_at": "2026-09-06T10:00:00+10:00"}
    }
    result = controller.acknowledge_intentional_stop(conversation, reason="sleep")
    assert result["ok"] is True
    assert "conversation" in controller._user_stopped
    assert "conversation" not in controller._owned

    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    started = MagicMock(return_value={"ok": True, "status": STATUS_ONLINE})
    monkeypatch.setattr(controller, "start", started)
    # Simulate crash ownership without clearing the intentional stop flag.
    controller._owned = {
        "conversation": {"pid": 1, "started_by_dashboard": True, "started_at": "2026-09-06T10:00:00+10:00"}
    }
    controller.recover_once()
    started.assert_not_called()


def test_recover_reloads_stopped_json_written_by_sleep(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recovery must honor stopped.json written by the Conversation App during sleep."""
    from control_dashboard import paths as dashboard_paths
    from control_dashboard.stack import StackController

    stopped_path = tmp_path / "stopped.json"
    monkeypatch.setattr(dashboard_paths, "STOPPED_PATH", stopped_path)
    monkeypatch.setattr(dashboard_paths, "OWNED_PATH", tmp_path / "owned.json")
    config = load_config()
    controller = StackController(config)
    controller._owned = {
        "conversation": {"pid": 1, "started_by_dashboard": True, "started_at": "2026-09-06T10:00:00+10:00"}
    }
    stopped_path.write_text('["conversation"]\n', encoding="utf-8")
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    started = MagicMock(return_value={"ok": True, "status": STATUS_ONLINE})
    monkeypatch.setattr(controller, "start", started)
    controller.recover_once()
    started.assert_not_called()
    assert "conversation" in controller._user_stopped


def test_reachy_health_rejects_physical_daemon_on_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wireless robot answering on 127.0.0.1:8000 is not the managed simulator."""
    config = load_config()
    spec = config.service("reachy_daemon")
    assert spec is not None
    monkeypatch.setattr("control_dashboard.checks._configured_daemon_host", lambda _config: None)
    monkeypatch.setattr("control_dashboard.net.host_resolves", lambda _host: True)
    monkeypatch.setattr("control_dashboard.net.port_open", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "control_dashboard.checks.net.http_request",
        lambda _url, **_k: HttpResult(
            True,
            200,
            '{"state":"stopped","wireless_version":true,"simulation_enabled":false,"desktop_app_daemon":false}',
            1.0,
            None,
        ),
    )
    monkeypatch.setattr(
        "control_dashboard.checks._process_snapshot",
        lambda *_a, **_k: {"pids": [9], "command": "Reachy Mini.exe", "name": "Reachy Mini.exe"},
    )
    result = check_reachy_daemon(spec, config, {})
    assert result.status == STATUS_OFFLINE
    assert "physical" in result.summary.lower()


def test_reachy_health_accepts_local_simulator_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /api/daemon/status with simulation_enabled is the local simulator."""
    config = load_config()
    spec = config.service("reachy_daemon")
    assert spec is not None
    monkeypatch.setattr("control_dashboard.checks._configured_daemon_host", lambda _config: None)
    monkeypatch.setattr("control_dashboard.net.host_resolves", lambda _host: True)
    monkeypatch.setattr("control_dashboard.net.port_open", lambda *_a, **_k: True)
    monkeypatch.setattr(
        "control_dashboard.checks.net.http_request",
        lambda _url, **_k: HttpResult(
            True,
            200,
            '{"state":"running","wireless_version":false,"simulation_enabled":true}',
            1.0,
            None,
        ),
    )
    monkeypatch.setattr(
        "control_dashboard.checks._process_snapshot",
        lambda *_a, **_k: {
            "pids": [8],
            "command": "reachy-mini-daemon --sim --fastapi-port 8000",
            "name": "reachy-mini-daemon.exe",
            "match": True,
        },
    )
    result = check_reachy_daemon(spec, config, {})
    assert result.status == STATUS_ONLINE
    assert result.details.get("environment") == "simulator"


def test_reachy_health_ignores_mdns_physical_robot(monkeypatch: pytest.MonkeyPatch) -> None:
    """reachy-mini.local must not make the managed Reachy Mini card look online."""
    config = load_config()
    spec = config.service("reachy_daemon")
    assert spec is not None
    monkeypatch.setattr("control_dashboard.checks._configured_daemon_host", lambda _config: None)
    monkeypatch.setattr("control_dashboard.net.host_resolves", lambda _host: True)
    monkeypatch.setattr(
        "control_dashboard.net.port_open",
        lambda host, _port, timeout_s=1.0: host == "reachy-mini.local",
    )
    monkeypatch.setattr(
        "control_dashboard.checks.net.http_request",
        lambda url, **_k: (
            HttpResult(
                True,
                200,
                '{"state":"stopped","wireless_version":true,"simulation_enabled":false}',
                1.0,
                None,
            )
            if "reachy-mini.local" in url
            else HttpResult(False, None, "", 1.0, "refused")
        ),
    )
    result = check_reachy_daemon(spec, config, {})
    assert result.status == STATUS_OFFLINE
    assert "127.0.0.1:8000" in result.summary


def test_stop_all_stops_matching_simulator_even_if_external(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop all must stop a local --sim daemon, including one this dashboard did not start."""
    config = load_config()
    reachy = config.service("reachy_daemon")
    assert reachy is not None
    controller = StackController(config)
    controller._owned = {}
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [777])
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [777])
    monkeypatch.setattr(
        "control_dashboard.stack.proc.process_command_line",
        lambda _pid: "reachy-mini-daemon --sim --fastapi-port 8000",
    )
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: True)
    monkeypatch.setattr("control_dashboard.stack.proc.wait_until_gone", lambda _pid, timeout_s=8.0: True)
    killed = MagicMock()
    monkeypatch.setattr("control_dashboard.stack.proc.stop_pid", killed)
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    monkeypatch.setattr(controller, "_wait_for_port_free", lambda _spec: None)
    result = controller.stop(reachy)
    assert result["ok"] is True
    killed.assert_called_once_with(777)
    assert 777 in result.get("stopped_pids", [])


def test_conversation_start_surfaces_log_error(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Conversation startup failure must include the captured log error."""
    from control_dashboard import paths as dashboard_paths

    monkeypatch.setattr(dashboard_paths, "LOG_DIR", tmp_path)
    log_path = tmp_path / "conversation.log"
    log_path.write_text(
        "Connection failed: Reachy Mini simulator is not listening on 127.0.0.1:8000.\n",
        encoding="utf-8",
    )
    config = load_config()
    conversation = config.service("conversation")
    assert conversation is not None
    controller = StackController(config)
    monkeypatch.setattr(
        controller,
        "health",
        lambda spec, probe=False: HealthResult(STATUS_OFFLINE, "down"),
    )
    monkeypatch.setattr(controller, "_blocked_by", lambda _spec, **_k: [])
    monkeypatch.setattr(controller, "_save_owned", lambda: None)
    monkeypatch.setattr(controller, "_save_stopped", lambda: None)
    monkeypatch.setattr("control_dashboard.stack.shutil_which", lambda exe: exe)
    monkeypatch.setattr("control_dashboard.stack.proc.pids_matching", lambda _pattern: [])
    monkeypatch.setattr("control_dashboard.stack.proc.listening_pids", lambda _port: [])
    monkeypatch.setattr("control_dashboard.stack.proc.pid_is_running", lambda _pid: False)
    monkeypatch.setattr("control_dashboard.stack.proc.start_process", MagicMock(return_value=99))
    monkeypatch.setattr(
        "control_dashboard.stack.net.http_request",
        lambda *_a, **_k: HttpResult(
            True, 200, '{"size":1,"in_use":0,"units":[{"index":0,"state":"idle"}]}', 1.0, None
        ),
    )
    result = controller.start(conversation)
    assert result["ok"] is False
    assert "exited before becoming healthy" in str(result.get("error"))
    assert "127.0.0.1:8000" in str(result.get("technical"))
