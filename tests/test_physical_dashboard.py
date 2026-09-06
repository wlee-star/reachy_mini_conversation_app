"""Tests for the physical Reachy Mini dashboard adapter."""

from __future__ import annotations
from typing import Any
from unittest.mock import MagicMock

import pytest

from control_dashboard import physical
from control_dashboard.checks import STATUS_ONLINE, STATUS_OFFLINE, STATUS_DEGRADED, HealthResult
from control_dashboard.physical import RobotTarget, CommandBlocked
from control_dashboard.registry import load_config


def test_resolve_target_uses_configured_physical_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-loopback REACHY_DAEMON_HOST must resolve as physical when identity is clear."""
    config = load_config()
    config.env["REACHY_DAEMON_HOST"] = "192.168.1.50"
    monkeypatch.setattr(
        "control_dashboard.checks._probe_daemon_http",
        lambda host, port: HealthResult(
            STATUS_ONLINE,
            "ok",
            details={"environment": "physical", "simulation_enabled": False, "host": host, "port": port},
        ),
    )
    target = physical.resolve_target(config)
    assert target.kind == physical.TARGET_PHYSICAL
    assert target.host == "192.168.1.50"


def test_resolve_target_local_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loopback daemon with simulation_enabled must stay simulator."""
    config = load_config()
    config.env.pop("REACHY_DAEMON_HOST", None)
    monkeypatch.setattr(
        "control_dashboard.checks._probe_daemon_http",
        lambda host, port: HealthResult(
            STATUS_ONLINE,
            "sim ok",
            details={"environment": "simulator", "simulation_enabled": True, "host": host, "port": port},
        ),
    )
    target = physical.resolve_target(config)
    assert target.kind == physical.TARGET_SIMULATOR


def test_resolve_target_remote_simulator_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured LAN host with simulation_enabled must not be treated as physical."""
    config = load_config()
    config.env["REACHY_DAEMON_HOST"] = "192.168.1.50"
    monkeypatch.setattr(
        "control_dashboard.checks._probe_daemon_http",
        lambda host, port: HealthResult(
            STATUS_ONLINE,
            "sim on lan",
            details={"environment": "simulator", "simulation_enabled": True, "host": host, "port": port},
        ),
    )
    target = physical.resolve_target(config)
    assert target.kind == physical.TARGET_SIMULATOR


def test_resolve_target_fail_closed_when_remote_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreachable configured host must be UNKNOWN, not guessed physical."""
    config = load_config()
    config.env["REACHY_DAEMON_HOST"] = "192.168.1.50"
    monkeypatch.setattr(
        "control_dashboard.checks._probe_daemon_http",
        lambda host, port: HealthResult(
            STATUS_OFFLINE,
            f"{host}:{port} refused TCP",
            details={"refused": True, "host": host, "port": port},
        ),
    )
    target = physical.resolve_target(config)
    assert target.kind == physical.TARGET_UNKNOWN
    assert target.host == "192.168.1.50"


def test_resolve_target_fail_closed_when_identity_unclear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Online daemon without clear physical/sim markers must be UNKNOWN."""
    config = load_config()
    config.env["REACHY_DAEMON_HOST"] = "192.168.1.50"
    monkeypatch.setattr(
        "control_dashboard.checks._probe_daemon_http",
        lambda host, port: HealthResult(
            STATUS_ONLINE,
            "ok",
            details={"environment": "unknown", "simulation_enabled": None, "host": host, "port": port},
        ),
    )
    target = physical.resolve_target(config)
    assert target.kind == physical.TARGET_UNKNOWN


def test_resolve_target_local_physical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Localhost physical daemon resolves as physical."""
    config = load_config()
    config.env.pop("REACHY_DAEMON_HOST", None)
    monkeypatch.setattr(
        "control_dashboard.checks._probe_daemon_http",
        lambda host, port: HealthResult(
            STATUS_ONLINE,
            "physical ok",
            details={"environment": "physical", "simulation_enabled": False, "host": host, "port": port},
        ),
    )
    target = physical.resolve_target(config)
    assert target.kind == physical.TARGET_PHYSICAL


def test_physical_commands_blocked_on_simulator() -> None:
    """Physical actions must not run when the target is the simulator."""
    target = RobotTarget(
        kind=physical.TARGET_SIMULATOR,
        host="127.0.0.1",
        port=8000,
        simulation_enabled=True,
        summary="sim",
    )
    with pytest.raises(CommandBlocked, match="COMMAND BLOCKED"):
        physical.assert_physical_command(target)


def test_physical_commands_blocked_on_unknown() -> None:
    """Fail-closed: unknown target blocks physical commands."""
    target = RobotTarget(
        kind=physical.TARGET_UNKNOWN,
        host="192.168.1.50",
        port=8000,
        simulation_enabled=None,
        summary="unresolved",
    )
    with pytest.raises(CommandBlocked, match="COMMAND BLOCKED"):
        physical.assert_physical_command(target)


def test_run_physical_action_blocked_for_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_physical_action must refuse simulator targets before calling conversation."""
    config = load_config()
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_SIMULATOR,
            host="127.0.0.1",
            port=8000,
            simulation_enabled=True,
            summary="sim",
        ),
    )
    called = MagicMock()
    monkeypatch.setattr(physical, "_conversation_json", called)
    with pytest.raises(CommandBlocked):
        physical.run_physical_action(config, "safe_stop", {})
    called.assert_not_called()


def test_run_physical_action_blocked_for_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_physical_action must refuse unknown targets before calling conversation."""
    config = load_config()
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_UNKNOWN,
            host=None,
            port=8000,
            simulation_enabled=None,
            summary="unknown",
        ),
    )
    called = MagicMock()
    monkeypatch.setattr(physical, "_conversation_json", called)
    with pytest.raises(CommandBlocked):
        physical.run_physical_action(config, "mic", {"muted": True})
    called.assert_not_called()


def test_camera_preview_off_does_not_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera Preview Off must stop preview fetches without opening another pipeline."""
    config = load_config()
    physical.set_camera_preview_enabled(False)
    fetch = MagicMock()
    monkeypatch.setattr(physical, "_http_bytes", fetch)
    jpeg, meta = physical.fetch_camera_jpeg(config)
    assert jpeg is None
    assert meta["status"] == "preview_off"
    assert "robot camera unchanged" in meta["summary"].lower() or "preview" in meta["summary"].lower()
    fetch.assert_not_called()
    physical.set_camera_preview_enabled(True)


def test_camera_fetch_blocked_on_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Camera JPEG fetch must not proxy when target is simulator."""
    config = load_config()
    physical.set_camera_preview_enabled(True)
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_SIMULATOR,
            host="127.0.0.1",
            port=8000,
            simulation_enabled=True,
            summary="sim",
        ),
    )
    fetch = MagicMock()
    monkeypatch.setattr(physical, "_http_bytes", fetch)
    jpeg, meta = physical.fetch_camera_jpeg(config)
    assert jpeg is None
    assert meta["status"] == "blocked"
    fetch.assert_not_called()


def test_camera_fetch_failure_uses_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failed camera fetch returns offline meta and increases backoff without raising."""
    config = load_config()
    physical.set_camera_preview_enabled(True)
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    monkeypatch.setattr(physical, "_http_bytes", lambda *_args, **_kwargs: None)
    with physical._CAMERA_LOCK:
        physical._CAMERA_LAST_ATTEMPT = 0.0
        physical._CAMERA_BACKOFF_S = physical._CAMERA_MIN_INTERVAL_S
    jpeg, meta = physical.fetch_camera_jpeg(config)
    assert jpeg is None
    assert meta["status"] == "offline"
    assert meta.get("backoff_s", 0) >= physical._CAMERA_MIN_INTERVAL_S


def test_camera_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful JPEG fetch reports live preview without a second SDK client."""
    config = load_config()
    physical.set_camera_preview_enabled(True)
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    jpeg_bytes = b"\xff\xd8\xff" + b"jpeg"
    monkeypatch.setattr(physical, "_http_bytes", lambda *_args, **_kwargs: jpeg_bytes)
    with physical._CAMERA_LOCK:
        physical._CAMERA_LAST_ATTEMPT = 0.0
        physical._CAMERA_BACKOFF_S = physical._CAMERA_MIN_INTERVAL_S
    jpeg, meta = physical.fetch_camera_jpeg(config)
    assert jpeg == jpeg_bytes
    assert meta["status"] == "live"


def test_run_physical_action_mic_mute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mic mute proxies through the conversation dashboard API on a physical target."""
    config = load_config()
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    monkeypatch.setattr(
        physical,
        "_conversation_json",
        lambda *_args, **_kwargs: (
            HealthResult(STATUS_ONLINE, "ok"),
            {"ok": True, "muted": True, "microphone_status": "muted"},
        ),
    )
    result = physical.run_physical_action(config, "mic", {"muted": True})
    assert result["ok"] is True
    assert result["result"]["muted"] is True


def test_run_physical_action_speaker_test_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speaker test API errors surface as CommandBlocked."""
    config = load_config()
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    monkeypatch.setattr(
        physical,
        "_conversation_json",
        lambda *_args, **_kwargs: (
            HealthResult(STATUS_ONLINE, "ok"),
            {"ok": False, "error": "Speaker is muted."},
        ),
    )
    with pytest.raises(CommandBlocked, match="Speaker is muted"):
        physical.run_physical_action(config, "speaker_test", {})


def test_run_physical_action_safe_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe stop proxies to conversation and returns motors_disabled status."""
    config = load_config()
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    monkeypatch.setattr(
        physical,
        "_conversation_json",
        lambda *_args, **_kwargs: (
            HealthResult(STATUS_ONLINE, "ok"),
            {"ok": True, "status": "motors_disabled"},
        ),
    )
    result = physical.run_physical_action(config, "safe_stop", {})
    assert result["result"]["status"] == "motors_disabled"


def test_run_physical_action_safe_stop_sdk_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Safe stop SDK failure from conversation is blocked as an error."""
    config = load_config()
    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    monkeypatch.setattr(
        physical,
        "_conversation_json",
        lambda *_args, **_kwargs: (
            HealthResult(STATUS_ONLINE, "ok"),
            {"ok": False, "error": "Safe stop failed: RuntimeError: motors"},
        ),
    )
    with pytest.raises(CommandBlocked, match="Safe stop failed"):
        physical.run_physical_action(config, "safe_stop", {})


def test_build_physical_status_includes_real_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Status payload must use health-check results, not hard-coded ONLINE."""
    config = load_config()

    def fake_health(spec: Any, probe: bool = False) -> HealthResult:
        if spec.id == "conversation":
            return HealthResult(STATUS_ONLINE, "Conversation UI up", details={"port": 7860})
        if spec.id == "reachy_daemon":
            return HealthResult(
                STATUS_OFFLINE,
                "no daemon",
                details={"environment": None},
            )
        if spec.id == "hermes":
            return HealthResult(STATUS_DEGRADED, "Hermes slow", details={"latency": 900})
        if spec.id == "home_assistant":
            return HealthResult(STATUS_ONLINE, "HA up", details={"version": "2024.1"})
        if spec.id == "apex":
            return HealthResult(STATUS_OFFLINE, "Apex offline")
        return HealthResult(STATUS_OFFLINE, f"{spec.id} offline")

    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical host configured",
        ),
    )
    monkeypatch.setattr(
        physical,
        "_conversation_json",
        lambda *_args, **_kwargs: (
            HealthResult(STATUS_OFFLINE, "media offline"),
            None,
        ),
    )
    payload = physical.build_physical_status(config, fake_health)
    assert payload["target"]["kind"] == "physical"
    assert payload["banners"]["connected"] is False
    conversation = next(row for row in payload["stack"] if row["id"] == "conversation")
    assert conversation["status"] == STATUS_ONLINE
    assert conversation["label"] == "ONLINE"
    assert any(row["id"] == "camera" and row["status"] == "offline" for row in payload["stack"])
    hermes = next(row for row in payload["stack"] if row["id"] == "hermes")
    assert hermes["status"] == STATUS_DEGRADED
    assert payload["camera_preview"]["enabled"] is physical.camera_preview_enabled()
    assert "preview" in payload["camera_preview"]["summary"].lower()
    assert payload["safe_stop"]["available"] is False
    assert "unavailable" in (payload["safe_stop"]["summary"] or "").lower()


def test_build_physical_status_media_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Media rows must reflect conversation dashboard status, not hard-coded READY."""
    config = load_config()

    def fake_health(spec: Any, probe: bool = False) -> HealthResult:
        return HealthResult(STATUS_ONLINE, f"{spec.id} up", details={"environment": "physical"})

    monkeypatch.setattr(
        physical,
        "resolve_target",
        lambda _config: RobotTarget(
            kind=physical.TARGET_PHYSICAL,
            host="192.168.1.50",
            port=8000,
            simulation_enabled=False,
            summary="physical",
        ),
    )
    monkeypatch.setattr(
        physical,
        "_conversation_json",
        lambda *_args, **_kwargs: (
            HealthResult(STATUS_ONLINE, "media ok"),
            {
                "microphone_status": "muted",
                "microphone_summary": "Microphone is muted.",
                "microphone_muted": True,
                "microphone_level": 0.0,
                "speaker_status": "ready",
                "speaker_summary": "Speaker is ready.",
                "speaker_muted": False,
                "speaker_level": 0.1,
                "volume_control": False,
                "camera_status": "ready",
                "camera_summary": "Camera ready",
                "safe_stop_available": True,
                "safe_stop_summary": "SAFE STOP: disable motors",
                "motors_status": "UNKNOWN",
                "robot_state": "IDLE",
            },
        ),
    )
    payload = physical.build_physical_status(config, fake_health)
    assert payload["robot"]["motors"] == "UNKNOWN"
    assert payload["robot"]["microphone"] == "MUTED"
    assert payload["robot"]["speaker"] == "READY"
    assert payload["media"]["volume_control"] is False
    mic = next(row for row in payload["stack"] if row["id"] == "microphone")
    assert mic["label"] == "CONNECTED"
