import threading
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import ANY, MagicMock, call

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import INIT_HEAD_POSE, SLEEP_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.wake_trace import WakeTrace
from reachy_mini_conversation_app.face_tracking import PostWakeTelemetrySnapshot
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def test_startup_freshness_recorder_retains_maxima_and_advancement() -> None:
    """Startup freshness reporting reflects the exact samples and publication lease."""
    recorder = app_lifecycle.StartupFreshnessRecorder()
    snapshot = PostWakeTelemetrySnapshot(
        monotonic_time=1.0,
        present_head_pose=np.eye(4),
        present_head_joints=(0.0,) * 7,
        present_body_yaw=0.0,
        present_antennas=(-0.1745, 0.1745),
        control_mode="enabled",
        daemon_timestamp="2026-09-14T10:00:00+00:00",
        daemon_ready=True,
        daemon_error=None,
        backend_last_alive=1.0,
        backend_last_alive_age_s=0.12,
        control_loop_frequency_hz=50.0,
        active_move_count=0,
        acquisition_elapsed_s=0.08,
    )

    recorder.observe(snapshot, daemon_advanced=True, backend_advanced=True)
    recorder.observe(
        snapshot.__class__(**{**snapshot.__dict__, "acquisition_elapsed_s": 0.2, "backend_last_alive_age_s": 0.22}),
        daemon_advanced=True,
        backend_advanced=True,
    )
    recorder.record_publication_lease(0.18)

    assert recorder.max_acquisition_s == pytest.approx(0.2)
    assert recorder.max_backend_age_s == pytest.approx(0.22)
    assert recorder.max_publication_lease_s == pytest.approx(0.18)
    assert recorder.daemon_advancement is True
    assert recorder.backend_advancement is True


def _authorized_snapshot(*, authorized_monotonic: float = 0.0) -> app_lifecycle.AuthorizedStartupSnapshot:
    return app_lifecycle.AuthorizedStartupSnapshot(
        head_transform=tuple(tuple(float(value) for value in row) for row in np.eye(4)),
        head_joints=(0.0,) * 7,
        body_yaw=0.0,
        antennas=(-0.1745, 0.1745),
        daemon_timestamp="2026-09-14T10:00:00+00:00",
        backend_last_alive=0.0,
        acquisition_monotonic=authorized_monotonic,
        authorized_monotonic=authorized_monotonic,
        validation_sample_count=10,
    )


def test_authorized_startup_snapshot_diagnostics_expose_provenance_pose_and_timestamp() -> None:
    """Snapshot diagnostics expose the authorized source without another SDK read."""
    head_pose = create_head_pose(yaw=5, degrees=True)
    snapshot = app_lifecycle.AuthorizedStartupSnapshot(
        head_transform=tuple(tuple(float(value) for value in row) for row in head_pose),
        head_joints=(0.0,) * 7,
        body_yaw=0.0,
        antennas=(-0.1745, 0.1745),
        daemon_timestamp="daemon-123",
        backend_last_alive=9.0,
        acquisition_monotonic=12.5,
        authorized_monotonic=12.75,
        validation_sample_count=10,
        provenance="NON_SLEEP_STARTUP_GATE",
    )

    diagnostics = app_lifecycle.authorized_startup_snapshot_diagnostics(snapshot)

    assert diagnostics["provenance"] == "NON_SLEEP_STARTUP_GATE"
    assert diagnostics["daemon_timestamp"] == "daemon-123"
    assert diagnostics["captured_monotonic"] == 12.5
    assert diagnostics["authorized_monotonic"] == 12.75
    assert diagnostics["head_rpy_deg"][2] == pytest.approx(5.0)


def test_request_stop_current_app_posts_to_daemon(monkeypatch) -> None:
    """The app stop request should call the connected Reachy daemon endpoint."""

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request, timeout):
        assert request.full_url == "http://192.168.1.42:8000/api/apps/stop-current-app"
        assert request.get_method() == "POST"
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", fake_urlopen)
    robot = SimpleNamespace(client=SimpleNamespace(host="192.168.1.42", port=8000))

    assert app_lifecycle.request_stop_current_app(robot, MagicMock())


def test_wake_up_if_sleeping_enables_motors_before_wake_up(monkeypatch) -> None:
    """Startup should enable sleeping motors before playing the wake-up movement."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    instrumentation = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", MagicMock(return_value=True))

    assert app_lifecycle.wake_up_if_sleeping(
        robot,
        MagicMock(),
        stage1_instrumentation=instrumentation,
    )

    assert robot.method_calls == [
        call.get_current_head_pose(),
        call.get_current_joint_positions(),
        call.enable_motors(),
        call.wake_up(),
        call.get_current_head_pose(),
        call.get_current_joint_positions(),
    ]


def test_prepare_robot_for_conversation_retries_enable_motors(monkeypatch) -> None:
    """Wireless power-on often rejects enable_motors until the daemon is ready."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.enable_motors.side_effect = [RuntimeError("not ready"), None]
    instrumentation = MagicMock()
    monkeypatch.setattr(
        app_lifecycle,
        "wait_for_non_sleep_startup_validation",
        MagicMock(return_value=app_lifecycle.NonSleepStartupValidation(True, None, 10, _authorized_snapshot())),
    )

    assert app_lifecycle.prepare_robot_for_conversation(
        robot,
        MagicMock(),
        attempts=2,
        delay_s=0,
        stage1_instrumentation=instrumentation,
    )

    assert robot.enable_motors.call_count == 2
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_wakes_when_pose_unreadable() -> None:
    """After power-off the pose read can fail even though the robot is asleep."""
    robot = MagicMock()
    robot.get_current_head_pose.side_effect = RuntimeError("motors disabled")

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_called_once()
    robot.wake_up.assert_called_once()


def test_wake_up_if_sleeping_skips_non_sleep_head_pose(monkeypatch) -> None:
    """Startup should leave an already-awake robot alone."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    convergence = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", convergence)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_called_once()
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()
    convergence.assert_not_called()


def test_sleep_pose_log_reports_each_geometric_condition(monkeypatch) -> None:
    """Startup attribution records the exact decision inputs without changing them."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    logger = MagicMock()
    instrumentation = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", MagicMock(return_value=True))

    assert app_lifecycle.wake_up_if_sleeping(robot, logger, stage1_instrumentation=instrumentation)

    assert any(
        call_args.args[0].startswith("Startup sleep test") and call_args.args[-1] is True
        for call_args in logger.info.call_args_list
    )


def test_wake_trace_surrounds_existing_sdk_wake_without_changing_lifecycle_order(monkeypatch) -> None:
    """The optional tracer observes the same enable-then-wake lifecycle."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    trace = MagicMock(spec=WakeTrace)
    trace.observe_sdk_waypoints.return_value = nullcontext()
    instrumentation = MagicMock()
    convergence = MagicMock(return_value=True)
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", convergence)

    assert app_lifecycle.wake_up_if_sleeping(
        robot,
        MagicMock(),
        wake_trace=trace,
        stage1_instrumentation=instrumentation,
    )

    trace.start.assert_called_once_with()
    trace.observe_sdk_waypoints.assert_called_once_with()
    robot.wake_up.assert_called_once_with()
    trace.wake_returned.assert_called_once()
    convergence.assert_called_once_with(instrumentation, ANY)
    trace.close.assert_not_called()


def test_run_go_to_sleep_tool_uses_runtime_callback() -> None:
    """Synchronous lifecycle paths should enter through the go_to_sleep tool."""
    expected = {"status": "sleeping"}
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = app_lifecycle.run_go_to_sleep_tool(deps, MagicMock())

    assert result == expected
    go_to_sleep.assert_called_once_with()


def test_acknowledge_dashboard_sleep_stop_writes_stopped_json(tmp_path, monkeypatch) -> None:
    """Sleep must mark conversation stopped before the process exits."""
    stopped_path = tmp_path / "stopped.json"
    monkeypatch.setattr(app_lifecycle, "_DASHBOARD_STOPPED_PATH", stopped_path)

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def read(self) -> bytes:
            return b'{"ok":true}'

    def fake_urlopen(request, timeout):
        assert request.get_method() == "POST"
        assert request.full_url.endswith("/api/services/conversation/ack_stop")
        assert timeout == 2.0
        return FakeResponse()

    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", fake_urlopen)
    logger = MagicMock()
    app_lifecycle.acknowledge_dashboard_sleep_stop(logger)
    assert stopped_path.read_text(encoding="utf-8").strip() == '[\n  "conversation"\n]'
    logger.info.assert_any_call("Marked Conversation App as intentionally stopped for sleep")


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


def _tracking_robot(*states: bool | None) -> MagicMock:
    robot = MagicMock()
    status = MagicMock()

    tracking_states = list(states)
    calls = 0

    def model_dump() -> dict[str, bool | None]:
        nonlocal calls
        state = tracking_states[min(calls, len(tracking_states) - 1)]
        calls += 1
        return {"head_tracking_enabled": state}

    status.model_dump.side_effect = model_dump
    robot.client.get_status.return_value = status
    return robot


class _StatusWithoutTrackingField:
    def model_dump(self) -> dict[str, object]:
        return {}


class _TrackingStatusResponse:
    def __init__(self, body: str) -> None:
        self.body = body.encode("utf-8")

    def __enter__(self) -> "_TrackingStatusResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.body


def _tracking_robot_without_sdk_field() -> MagicMock:
    robot = MagicMock()
    robot.client.host = "192.168.0.57"
    robot.client.port = 8000
    robot.client.get_status.return_value = _StatusWithoutTrackingField()
    return robot


def test_startup_tracking_false_needs_no_recovery_command() -> None:
    """An already-disabled daemon proceeds without an unnecessary command."""
    robot = _tracking_robot(False)

    result = app_lifecycle.ensure_daemon_head_tracking_disabled(
        robot,
        MagicMock(),
        reason="test_startup",
    )

    assert result == app_lifecycle.DaemonTrackingDisableResult(False, False, True)
    robot.stop_head_tracking.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_startup_tracking_false_survives_sdk_protocol_omission(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale SDK schema must not turn the daemon's authoritative false into unknown."""
    robot = _tracking_robot_without_sdk_field()
    urlopen = MagicMock(return_value=_TrackingStatusResponse('{"head_tracking_enabled": false}'))
    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", urlopen)

    result = app_lifecycle.ensure_daemon_head_tracking_disabled(
        robot,
        MagicMock(),
        reason="test_startup",
    )

    assert result == app_lifecycle.DaemonTrackingDisableResult(False, False, True)
    assert urlopen.call_args.args[0].full_url == "http://192.168.0.57:8000/api/daemon/status"
    robot.stop_head_tracking.assert_not_called()


def test_startup_tracking_true_survives_sdk_protocol_omission(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale SDK schema must still preserve true and run the existing disable-only path."""
    robot = _tracking_robot_without_sdk_field()
    urlopen = MagicMock(
        side_effect=[
            _TrackingStatusResponse('{"head_tracking_enabled": true}'),
            _TrackingStatusResponse('{"head_tracking_enabled": false}'),
        ]
    )
    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", urlopen)

    result = app_lifecycle.ensure_daemon_head_tracking_disabled(
        robot,
        MagicMock(),
        reason="test_startup",
    )

    assert result == app_lifecycle.DaemonTrackingDisableResult(True, True, True)
    robot.stop_head_tracking.assert_called_once_with()


@pytest.mark.parametrize(
    "body",
    [
        '{"face_target": {"detected": false}}',
        '{"head_tracking_enabled": null}',
        '{"head_tracking_enabled": "false"}',
    ],
)
def test_startup_tracking_unknown_payloads_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    body: str,
) -> None:
    """Missing or malformed tracking state is never synthesized from other fields."""
    robot = _tracking_robot_without_sdk_field()
    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", MagicMock(return_value=_TrackingStatusResponse(body)))
    clock = _FakeClock()

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="unknown"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_startup",
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    robot.stop_head_tracking.assert_not_called()


def test_startup_tracking_status_transport_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport failure cannot fall through into startup ownership."""
    robot = _tracking_robot_without_sdk_field()
    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", MagicMock(side_effect=OSError("offline")))
    clock = _FakeClock()

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="unavailable"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_startup",
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    robot.stop_head_tracking.assert_not_called()


def test_startup_tracking_waits_for_bounded_authoritative_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup may wait briefly for a Boolean, but still requires the Boolean."""
    robot = _tracking_robot_without_sdk_field()
    urlopen = MagicMock(
        side_effect=[
            _TrackingStatusResponse("{}"),
            _TrackingStatusResponse("{}"),
            _TrackingStatusResponse('{"head_tracking_enabled": false}'),
        ]
    )
    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", urlopen)
    clock = _FakeClock()

    result = app_lifecycle.ensure_daemon_head_tracking_disabled(
        robot,
        MagicMock(),
        reason="test_startup",
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result == app_lifecycle.DaemonTrackingDisableResult(False, False, True)
    assert urlopen.call_count == 3


def test_startup_tracking_unknown_through_deadline_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bounded wait does not weaken the unknown fail-safe."""
    robot = _tracking_robot_without_sdk_field()
    monkeypatch.setattr(app_lifecycle.urllib.request, "urlopen", MagicMock(return_value=_TrackingStatusResponse("{}")))
    clock = _FakeClock()

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="unknown"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_startup",
            timeout_s=0.1,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    robot.stop_head_tracking.assert_not_called()


def test_startup_tracking_true_is_disabled_once_and_confirmed() -> None:
    """Pre-owner recovery sends one disable and requires authoritative false."""
    robot = _tracking_robot(True, False)

    result = app_lifecycle.ensure_daemon_head_tracking_disabled(
        robot,
        MagicMock(),
        reason="test_startup",
    )

    assert result == app_lifecycle.DaemonTrackingDisableResult(True, True, True)
    robot.stop_head_tracking.assert_called_once_with()
    robot.start_head_tracking.assert_not_called()


def test_startup_tracking_disable_failure_remains_fail_closed() -> None:
    """A failed disable cannot fall through into normal startup ownership."""
    robot = _tracking_robot(True)
    robot.stop_head_tracking.side_effect = ConnectionError("lost SDK connection")

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="disable request failed"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_startup",
        )

    robot.stop_head_tracking.assert_called_once_with()
    robot.start_head_tracking.assert_not_called()


def test_startup_tracking_disable_command_is_bounded() -> None:
    """A stuck SDK send cannot hold startup indefinitely."""
    robot = _tracking_robot(True)
    release = threading.Event()
    robot.stop_head_tracking.side_effect = lambda: release.wait(timeout=1.0)

    try:
        with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="request timed out"):
            app_lifecycle.ensure_daemon_head_tracking_disabled(
                robot,
                MagicMock(),
                reason="test_startup",
                timeout_s=0.01,
            )
    finally:
        release.set()

    robot.stop_head_tracking.assert_called_once_with()
    robot.start_head_tracking.assert_not_called()


def test_startup_tracking_disable_requires_authoritative_false() -> None:
    """An accepted disable request still fails closed while daemon state remains true."""
    clock = _FakeClock()
    robot = _tracking_robot(True, True, True, True)

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="not authoritatively confirmed"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_startup",
            timeout_s=0.1,
            poll_interval_s=0.05,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    robot.stop_head_tracking.assert_called_once_with()
    robot.start_head_tracking.assert_not_called()


def test_startup_tracking_unknown_never_assumes_disabled_or_sends_a_command() -> None:
    """Unknown authoritative state blocks startup without toggling the daemon."""
    robot = _tracking_robot(None)

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="state is unknown"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_startup",
        )

    robot.stop_head_tracking.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_owned_shutdown_tracking_requires_bounded_authoritative_confirmation() -> None:
    """Graceful cleanup may disable an owned unknown state but must still confirm false."""
    clock = _FakeClock()
    robot = _tracking_robot(None, True, True, True)

    with pytest.raises(app_lifecycle.DaemonTrackingRecoveryError, match="not authoritatively confirmed"):
        app_lifecycle.ensure_daemon_head_tracking_disabled(
            robot,
            MagicMock(),
            reason="test_shutdown",
            application_owned=True,
            timeout_s=0.1,
            poll_interval_s=0.05,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert clock.value == pytest.approx(0.1)
    robot.stop_head_tracking.assert_called_once_with()
    robot.start_head_tracking.assert_not_called()


class _FakePostWakeInstrumentation:
    def __init__(
        self,
        clock: _FakeClock,
        rolls: list[float],
        *,
        advancing_liveness: bool = True,
        advancing_timestamp: bool = True,
        translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pitch: float = 0.0,
        yaw: float = 0.0,
        daemon_ready: bool = True,
        control_mode: str = "enabled",
        active_move_count: int = 0,
        freshness_s: float = 0.01,
        control_loop_frequency_hz: float = 50.0,
        malformed_pose: bool = False,
    ) -> None:
        self.clock = clock
        self.rolls = rolls
        self.advancing_liveness = advancing_liveness
        self.advancing_timestamp = advancing_timestamp
        self.translation_m = translation_m
        self.pitch = pitch
        self.yaw = yaw
        self.daemon_ready = daemon_ready
        self.control_mode = control_mode
        self.active_move_count = active_move_count
        self.freshness_s = freshness_s
        self.control_loop_frequency_hz = control_loop_frequency_hz
        self.malformed_pose = malformed_pose
        self.calls = 0
        self.preparation_calls = 0

    def prepare_telemetry_transport(self) -> object:
        self.preparation_calls += 1
        return SimpleNamespace(endpoint="/api/daemon/status", duration_s=2.1, result="pass")

    def read_post_wake_telemetry(self, *, deadline: float) -> PostWakeTelemetrySnapshot:
        assert self.clock.monotonic() < deadline
        roll = self.rolls[min(self.calls, len(self.rolls) - 1)]
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler("xyz", (roll, self.pitch, self.yaw), degrees=True).as_matrix()
        pose[0, 3], pose[1, 3], pose[2, 3] = self.translation_m
        if self.malformed_pose:
            pose[0, 0] = np.nan
        last_alive = float(self.calls + 1) if self.advancing_liveness else 1.0
        self.calls += 1
        timestamp_index = self.calls if self.advancing_timestamp else 1
        return PostWakeTelemetrySnapshot(
            monotonic_time=self.clock.monotonic(),
            present_head_pose=pose,
            present_head_joints=(0.0,) * 7,
            present_body_yaw=0.0,
            present_antennas=(-0.1745, 0.1745),
            control_mode=self.control_mode,
            daemon_timestamp=f"2026-09-14T10:00:{timestamp_index:02d}+00:00",
            daemon_ready=self.daemon_ready,
            daemon_error=None,
            backend_last_alive=last_alive,
            backend_last_alive_age_s=self.freshness_s,
            control_loop_frequency_hz=self.control_loop_frequency_hz,
            active_move_count=self.active_move_count,
            acquisition_elapsed_s=self.freshness_s,
        )


def test_orderly_shutdown_park_uses_existing_neutral_primitive_after_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controlled shutdown quiesces overlays before the existing two-second goto."""
    events: list[str] = []
    robot = MagicMock()
    instrumentation = MagicMock()
    snapshot = MagicMock()

    def record_preflight(*_args: object, **_kwargs: object) -> object:
        events.append("preflight")
        return snapshot

    def record_convergence(*_args: object, **_kwargs: object) -> object:
        events.append("converged")
        return MagicMock()

    monkeypatch.setattr(
        app_lifecycle,
        "ensure_daemon_head_tracking_disabled",
        MagicMock(side_effect=lambda *_args, **_kwargs: events.append("tracking_disabled")),
    )
    monkeypatch.setattr(
        app_lifecycle,
        "wait_for_orderly_shutdown_park_preconditions",
        MagicMock(side_effect=record_preflight),
    )
    monkeypatch.setattr(
        app_lifecycle,
        "wait_for_post_wake_convergence",
        MagicMock(side_effect=record_convergence),
    )
    robot.disable_wobbling.side_effect = lambda: events.append("wobble_disabled")
    robot.goto_target.side_effect = lambda **_kwargs: events.append("neutral_goto")

    app_lifecycle.park_robot_for_orderly_shutdown(robot, instrumentation, MagicMock())

    assert events == ["tracking_disabled", "wobble_disabled", "preflight", "neutral_goto", "converged"]
    kwargs = robot.goto_target.call_args.kwargs
    assert np.array_equal(kwargs["head"], INIT_HEAD_POSE)
    assert np.array_equal(kwargs["antennas"], INIT_ANTENNAS_JOINT_POSITIONS)
    assert kwargs["body_yaw"] == 0.0
    assert kwargs["duration"] == 2.0
    robot.client.disconnect.assert_not_called()


def test_orderly_shutdown_park_requires_measured_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Task completion alone cannot report a successful neutral park."""
    robot = MagicMock()
    monkeypatch.setattr(app_lifecycle, "ensure_daemon_head_tracking_disabled", MagicMock())
    monkeypatch.setattr(app_lifecycle, "wait_for_orderly_shutdown_park_preconditions", MagicMock())
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", MagicMock(return_value=None))

    with pytest.raises(app_lifecycle.OrderlyShutdownParkError, match="without measured convergence"):
        app_lifecycle.park_robot_for_orderly_shutdown(robot, MagicMock(), MagicMock())

    robot.goto_target.assert_called_once()
    robot.client.disconnect.assert_not_called()


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("active_move_count", 1, "another daemon movement is active"),
        ("freshness_s", 0.251, "telemetry is stale"),
        ("advancing_liveness", False, "backend liveness did not advance"),
        ("advancing_timestamp", False, "daemon timestamp did not advance"),
        ("malformed_pose", True, "finite"),
        ("control_mode", "disabled", "enabled motor control"),
    ],
)
def test_orderly_shutdown_park_preflight_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
    error: str,
) -> None:
    """Unsafe or untrustworthy pre-park state cannot reach the movement command."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0])
    setattr(instrumentation, attribute, value)
    monkeypatch.setattr(app_lifecycle, "_ORDERLY_SHUTDOWN_PREFLIGHT_TIMEOUT_S", 0.2)

    with pytest.raises(app_lifecycle.OrderlyShutdownParkError, match=error):
        app_lifecycle.wait_for_orderly_shutdown_park_preconditions(
            instrumentation,
            MagicMock(),
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def test_orderly_shutdown_park_preflight_accepts_fresh_advancing_idle_state() -> None:
    """Two fresh advancing idle samples authorize the existing park primitive."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0])

    snapshot = app_lifecycle.wait_for_orderly_shutdown_park_preconditions(
        instrumentation,
        MagicMock(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert instrumentation.calls == 2
    assert snapshot.active_move_count == 0
    assert snapshot.control_mode == "enabled"
    assert instrumentation.preparation_calls == 1


def test_orderly_shutdown_transport_preparation_failure_blocks_authoritative_read() -> None:
    """A warm-up failure fails closed before any authoritative shutdown telemetry."""

    class FailedPreparation:
        read_calls = 0

        def prepare_telemetry_transport(self) -> object:
            raise RuntimeError("connect failed")

        def read_post_wake_telemetry(self, *, deadline: float) -> PostWakeTelemetrySnapshot:
            self.read_calls += 1
            raise AssertionError("authoritative read must not start")

    instrumentation = FailedPreparation()

    with pytest.raises(app_lifecycle.OrderlyShutdownParkError, match="transport preparation failed"):
        app_lifecycle.wait_for_orderly_shutdown_park_preconditions(instrumentation, MagicMock())

    assert instrumentation.read_calls == 0


def test_orderly_shutdown_park_command_failure_is_not_reported_as_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed existing goto remains a fail-closed shutdown result."""
    robot = MagicMock()
    robot.goto_target.side_effect = ConnectionError("lost SDK connection")
    monkeypatch.setattr(app_lifecycle, "ensure_daemon_head_tracking_disabled", MagicMock())
    monkeypatch.setattr(app_lifecycle, "wait_for_orderly_shutdown_park_preconditions", MagicMock())
    convergence = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", convergence)

    with pytest.raises(app_lifecycle.OrderlyShutdownParkError, match="lost SDK connection"):
        app_lifecycle.park_robot_for_orderly_shutdown(robot, MagicMock(), MagicMock())

    convergence.assert_not_called()
    robot.client.disconnect.assert_not_called()


def _run_post_wake_gate(instrumentation: _FakePostWakeInstrumentation) -> bool:
    return (
        app_lifecycle.wait_for_post_wake_convergence(
            instrumentation,
            MagicMock(),
            monotonic=instrumentation.clock.monotonic,
            sleep=instrumentation.clock.sleep,
        )
        is not None
    )


def _run_non_sleep_gate(instrumentation: _FakePostWakeInstrumentation) -> app_lifecycle.NonSleepStartupValidation:
    return app_lifecycle.wait_for_non_sleep_startup_validation(
        instrumentation,
        MagicMock(),
        monotonic=instrumentation.clock.monotonic,
        sleep=instrumentation.clock.sleep,
    )


def _run_non_sleep_recovery_admission(instrumentation: _FakePostWakeInstrumentation) -> bool:
    return app_lifecycle.wait_for_non_sleep_startup_recovery_admission(
        instrumentation,
        MagicMock(),
        monotonic=instrumentation.clock.monotonic,
        sleep=instrumentation.clock.sleep,
    )


@pytest.mark.parametrize("roll", [0.0, 5.999, 6.0])
def test_non_sleep_gate_accepts_stable_pose_inside_provisional_envelope(roll: float) -> None:
    """Stable fresh awake poses inside the provisional envelope are authorized."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [roll]))

    assert result.accepted
    assert result.reason is None
    assert result.valid_samples >= 10
    assert result.authorized_snapshot is not None


def test_non_sleep_gate_rejects_just_above_provisional_envelope() -> None:
    """A geodesic rotation just above the provisional boundary is rejected."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [6.001]))

    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE


def test_observed_non_sleep_pose_is_recovery_eligible_but_not_directly_accepted() -> None:
    """The 6.117° physical pose can only enter the one-shot recovery path."""
    direct = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [6.117112740868852]))
    admission = _run_non_sleep_recovery_admission(_FakePostWakeInstrumentation(_FakeClock(), [6.117112740868852]))

    assert not direct.accepted
    assert direct.reason is app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE
    assert admission


@pytest.mark.parametrize("roll", [0.0, 7.0, 10.0])
def test_non_sleep_recovery_admission_requires_marginal_stage2i_pose(roll: float) -> None:
    """Already-valid and clearly unsafe poses are not eligible for recovery movement."""
    assert not _run_non_sleep_recovery_admission(_FakePostWakeInstrumentation(_FakeClock(), [roll]))


@pytest.mark.parametrize(
    "instrumentation",
    [
        _FakePostWakeInstrumentation(_FakeClock(), [6.117112740868852], freshness_s=0.3),
        _FakePostWakeInstrumentation(_FakeClock(), [6.117112740868852], control_loop_frequency_hz=39.9),
        _FakePostWakeInstrumentation(_FakeClock(), [6.117112740868852], active_move_count=1),
    ],
)
def test_non_sleep_recovery_admission_requires_safe_fresh_idle_telemetry(
    instrumentation: _FakePostWakeInstrumentation,
) -> None:
    """Recovery movement cannot be authorized by stale, unhealthy, or busy telemetry."""
    assert not _run_non_sleep_recovery_admission(instrumentation)


def test_non_sleep_gate_uses_combined_geodesic_rotation() -> None:
    """Combined Euler components are judged by geodesic distance."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [4.0], pitch=4.0, yaw=4.0))

    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE


def test_authorized_snapshot_is_deeply_immutable() -> None:
    """Nested authorized pose state cannot be changed through retained references."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [0.0]))
    authorized = result.authorized_snapshot

    assert authorized is not None
    with pytest.raises(TypeError):
        authorized.head_transform[0][0] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        authorized.head_joints[0] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        authorized.antennas[0] = 2.0  # type: ignore[index]


def test_authorized_snapshot_confirmation_only_confirms_matching_state() -> None:
    """Matching telemetry confirms without replacing the authorized object."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [1.0])
    result = _run_non_sleep_gate(instrumentation)
    authorized = result.authorized_snapshot
    assert authorized is not None
    original_transform = authorized.head_transform

    confirmed_at = app_lifecycle.confirm_authorized_startup_snapshot(
        instrumentation,
        authorized,
        MagicMock(),
        monotonic=clock.monotonic,
    )

    assert confirmed_at == pytest.approx(clock.value)
    assert authorized.head_transform == original_transform


def test_authorized_snapshot_confirmation_rejects_divergence_without_commands() -> None:
    """A changed physical pose invalidates authorization without movement."""
    clock = _FakeClock()
    initial = _FakePostWakeInstrumentation(clock, [0.0])
    authorized = _run_non_sleep_gate(initial).authorized_snapshot
    assert authorized is not None
    divergent = _FakePostWakeInstrumentation(clock, [3.0])
    divergent.calls = initial.calls

    with pytest.raises(app_lifecycle.NonSleepStartupValidationError) as error:
        app_lifecycle.confirm_authorized_startup_snapshot(
            divergent,
            authorized,
            MagicMock(),
            monotonic=clock.monotonic,
        )

    assert error.value.result.reason is app_lifecycle.NonSleepStartupFailureReason.AUTHORIZED_SNAPSHOT_DIVERGED


def test_authorized_snapshot_and_confirmation_leases_expire() -> None:
    """Both explicit 0.25-second leases fail closed."""
    authorized = _authorized_snapshot(authorized_monotonic=1.0)

    with pytest.raises(app_lifecycle.NonSleepStartupValidationError):
        app_lifecycle.require_fresh_authorized_startup_snapshot(authorized, monotonic=lambda: 1.251)
    with pytest.raises(app_lifecycle.NonSleepStartupValidationError):
        app_lifecycle.require_fresh_startup_confirmation(1.0, monotonic=lambda: 1.251)


def test_non_sleep_gate_blocks_stage2i_regression_pose() -> None:
    """The observed leaned startup pose cannot become an authoritative baseline."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [7.25], pitch=5.68, yaw=-3.75))

    assert not result.accepted
    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE


@pytest.mark.parametrize(
    ("instrumentation", "reason"),
    [
        (
            _FakePostWakeInstrumentation(_FakeClock(), [0.0], daemon_ready=False),
            app_lifecycle.NonSleepStartupFailureReason.DAEMON_UNHEALTHY,
        ),
        (
            _FakePostWakeInstrumentation(_FakeClock(), [0.0], active_move_count=1),
            app_lifecycle.NonSleepStartupFailureReason.ACTIVE_DAEMON_MOVE,
        ),
        (
            _FakePostWakeInstrumentation(_FakeClock(), [0.0], freshness_s=0.3),
            app_lifecycle.NonSleepStartupFailureReason.TELEMETRY_STALE,
        ),
        (
            _FakePostWakeInstrumentation(_FakeClock(), [0.0], advancing_timestamp=False),
            app_lifecycle.NonSleepStartupFailureReason.TIMESTAMP_NOT_ADVANCING,
        ),
        (
            _FakePostWakeInstrumentation(_FakeClock(), [0.0], malformed_pose=True),
            app_lifecycle.NonSleepStartupFailureReason.POSE_INVALID,
        ),
    ],
)
def test_non_sleep_gate_reports_structured_failures(
    instrumentation: _FakePostWakeInstrumentation,
    reason: app_lifecycle.NonSleepStartupFailureReason,
) -> None:
    """Unsafe telemetry produces a stable diagnostic category."""
    result = _run_non_sleep_gate(instrumentation)

    assert not result.accepted
    assert result.reason is reason


def test_non_sleep_gate_rejects_settling_or_unstable_pose() -> None:
    """Pose spread resets stability until the bounded deadline expires."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [0.0, 3.0] * 40))

    assert not result.accepted
    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.POSE_UNSTABLE


def test_non_sleep_gate_can_recover_after_bad_sample_reset() -> None:
    """A complete stable sequence after a bad sample can still authorize startup."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [0.0, 3.0] + [0.0] * 20))

    assert result.accepted
    assert result.reason is None


def test_non_sleep_gate_rejects_nonadvancing_backend_liveness() -> None:
    """Repeated backend liveness cannot count toward stable authorization."""
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [0.0], advancing_liveness=False))

    assert not result.accepted
    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.TIMESTAMP_NOT_ADVANCING


def test_non_sleep_gate_reports_unavailable_telemetry() -> None:
    """Read failures cannot authorize startup."""
    clock = _FakeClock()
    instrumentation = MagicMock()
    instrumentation.read_post_wake_telemetry.side_effect = RuntimeError("request failed")

    result = app_lifecycle.wait_for_non_sleep_startup_validation(
        instrumentation,
        MagicMock(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.TELEMETRY_UNAVAILABLE


def test_non_sleep_gate_preserves_pose_rejection_over_terminal_telemetry_timeout(monkeypatch) -> None:
    """A final transport timeout must not hide an established unsafe-pose rejection."""
    clock = _FakeClock()
    valid_source = _FakePostWakeInstrumentation(clock, [9.5])

    class UnsafeThenUnavailable:
        calls = 0

        def prepare_telemetry_transport(self) -> object:
            return SimpleNamespace(endpoint="/api/daemon/status", duration_s=0.01, result="pass")

        def read_post_wake_telemetry(self, *, deadline: float) -> PostWakeTelemetrySnapshot:
            self.calls += 1
            if self.calls <= 2:
                return valid_source.read_post_wake_telemetry(deadline=deadline)
            raise RuntimeError("post-wake telemetry request timed out")

    monkeypatch.setattr(app_lifecycle, "_NON_SLEEP_STARTUP_TIMEOUT_S", 0.2)
    result = app_lifecycle.wait_for_non_sleep_startup_validation(
        UnsafeThenUnavailable(),
        MagicMock(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.accepted is False
    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE


def test_non_sleep_gate_reports_deadline_before_stability(monkeypatch) -> None:
    """Valid telemetry cannot pass before the complete stability contract."""
    monkeypatch.setattr(app_lifecycle, "_NON_SLEEP_STARTUP_TIMEOUT_S", 0.3)
    result = _run_non_sleep_gate(_FakePostWakeInstrumentation(_FakeClock(), [0.0]))

    assert not result.accepted
    assert result.reason is app_lifecycle.NonSleepStartupFailureReason.DEADLINE_EXHAUSTED


def test_simulator_compatible_non_sleep_startup_is_supported(monkeypatch) -> None:
    """A simulator-like stable startup uses the same validation contract."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    validation = app_lifecycle.NonSleepStartupValidation(True, None, 10, _authorized_snapshot())
    validator = MagicMock(return_value=validation)
    instrumentation = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_validation", validator)

    assert app_lifecycle.prepare_robot_for_conversation(
        robot,
        MagicMock(),
        stage1_instrumentation=instrumentation,
    )

    validator.assert_called_once_with(instrumentation, ANY)
    robot.wake_up.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_prepare_robot_for_conversation_attempts_one_neutral_recovery_for_marginal_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marginal awake pose gets exactly one existing neutral recovery before startup."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    failure = app_lifecycle.NonSleepStartupValidation(
        False,
        app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE,
        0,
    )
    authorized = _authorized_snapshot()
    admission = MagicMock(return_value=True)
    recovery = MagicMock(return_value=authorized)
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_validation", MagicMock(return_value=failure))
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_recovery_admission", admission)
    monkeypatch.setattr(app_lifecycle, "park_robot_for_orderly_shutdown", recovery)
    instrumentation = MagicMock()
    recorder = app_lifecycle.StartupFreshnessRecorder()

    result = app_lifecycle.prepare_robot_for_conversation(
        robot,
        MagicMock(),
        stage1_instrumentation=instrumentation,
        freshness_recorder=recorder,
    )

    assert result is authorized
    admission.assert_called_once_with(instrumentation, ANY)
    recovery.assert_called_once_with(robot, instrumentation, ANY, freshness_recorder=recorder)
    robot.wake_up.assert_not_called()


def test_prepare_robot_for_conversation_does_not_retry_failed_neutral_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed marginal-pose recovery fails closed without a second movement attempt."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    failure = app_lifecycle.NonSleepStartupValidation(
        False,
        app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE,
        0,
    )
    recovery = MagicMock(side_effect=app_lifecycle.OrderlyShutdownParkError("did not converge"))
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_validation", MagicMock(return_value=failure))
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_recovery_admission", MagicMock(return_value=True))
    monkeypatch.setattr(app_lifecycle, "park_robot_for_orderly_shutdown", recovery)

    with pytest.raises(app_lifecycle.PostWakeConvergenceError, match="did not converge"):
        app_lifecycle.prepare_robot_for_conversation(
            robot,
            MagicMock(),
            stage1_instrumentation=MagicMock(),
        )

    recovery.assert_called_once()


def test_prepare_robot_for_conversation_rejects_non_recoverable_awake_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pose outside recovery admission preserves the original fail-closed result."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    failure = app_lifecycle.NonSleepStartupValidation(
        False,
        app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE,
        0,
    )
    recovery = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_validation", MagicMock(return_value=failure))
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_recovery_admission", MagicMock(return_value=False))
    monkeypatch.setattr(app_lifecycle, "park_robot_for_orderly_shutdown", recovery)

    with pytest.raises(app_lifecycle.NonSleepStartupValidationError):
        app_lifecycle.prepare_robot_for_conversation(
            robot,
            MagicMock(),
            stage1_instrumentation=MagicMock(),
        )

    recovery.assert_not_called()


def test_successful_auto_wake_skips_non_sleep_gate(monkeypatch) -> None:
    """Stage1Z success proceeds directly to Stage1M without a second gate."""
    robot = MagicMock()
    validator = MagicMock()
    monkeypatch.setattr(app_lifecycle, "wake_up_if_sleeping", MagicMock(return_value=True))
    monkeypatch.setattr(app_lifecycle, "wait_for_non_sleep_startup_validation", validator)

    assert app_lifecycle.prepare_robot_for_conversation(
        robot,
        MagicMock(),
        stage1_instrumentation=MagicMock(),
    )

    validator.assert_not_called()


def test_post_wake_gate_requires_time_and_sample_stability() -> None:
    """A neutral pose cannot pass before both stability requirements are met."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0])

    assert _run_post_wake_gate(instrumentation)
    assert instrumentation.calls >= 11
    assert clock.value >= 0.5


def test_post_wake_gate_accepts_pose_well_inside_rotation_threshold() -> None:
    """Healthy near-neutral poses must pass the gate."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [2.0])

    assert _run_post_wake_gate(instrumentation)


def test_post_wake_gate_accepts_boundary_below_rotation_threshold() -> None:
    """A stable residual just under 6.25° must pass."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [6.2])

    assert _run_post_wake_gate(instrumentation)


def test_post_wake_gate_rejects_boundary_above_rotation_threshold() -> None:
    """A residual just over 6.25° must fail."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [6.3])

    assert not _run_post_wake_gate(instrumentation)
    assert clock.value == pytest.approx(3.0)


def test_post_wake_gate_accepts_observed_compound_wake_residual() -> None:
    """The stable physical wake residual passes without relaxing other startup gates."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(
        clock,
        [3.8207770119718756],
        pitch=1.5525083766148091,
        yaw=-4.5454291316628215,
        translation_m=(-0.0032, 0.0018, -0.0021),
    )

    assert _run_post_wake_gate(instrumentation)


def test_post_wake_gate_rejects_clearly_unsafe_rotation() -> None:
    """A 10° lean must never be treated as converged."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [10.0])

    assert not _run_post_wake_gate(instrumentation)


def test_post_wake_gate_rejects_oscillating_pose_within_threshold() -> None:
    """Alternating in-tolerance poses that spread too far must not converge."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0, 2.5] * 40)

    assert not _run_post_wake_gate(instrumentation)


def test_post_wake_gate_rejects_insufficient_stable_samples(monkeypatch) -> None:
    """A late good pose that cannot gather 10 stable samples must fail."""
    monkeypatch.setattr(app_lifecycle, "_POST_WAKE_TIMEOUT_S", 0.3)
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0])

    assert not _run_post_wake_gate(instrumentation)
    assert instrumentation.calls < 10


def test_post_wake_gate_rejects_translation_outside_tolerance() -> None:
    """Large translation error fails even with a neutral orientation."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0], translation_m=(0.0, 0.0, 0.02))

    assert not _run_post_wake_gate(instrumentation)


def test_post_wake_gate_replays_stage1x_before_converging() -> None:
    """Stage 1X residual roll is rejected until the near-neutral pose is stable."""
    clock = _FakeClock()
    progression = [15.5, 12.0, 8.0, 7.0, 6.1, 5.5, 4.4, 3.9, 3.75, 3.7]
    instrumentation = _FakePostWakeInstrumentation(clock, progression)

    assert _run_post_wake_gate(instrumentation)
    assert instrumentation.calls >= len(progression) + 7


def test_post_wake_gate_resets_after_neutral_crossing() -> None:
    """A brief neutral crossing followed by delayed roll must time out."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [15.0, 0.0, 0.2, 20.0])

    assert not _run_post_wake_gate(instrumentation)
    assert clock.value == pytest.approx(3.0)


def test_post_wake_gate_times_out_without_motor_commands() -> None:
    """A persistent lean times out through read-only telemetry."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [15.5])

    assert not _run_post_wake_gate(instrumentation)
    assert instrumentation.calls == 61
    assert clock.value == pytest.approx(3.0)


def test_post_wake_gate_rejects_non_advancing_backend_liveness() -> None:
    """Repeated telemetry cannot accumulate stability without fresh backend life."""
    clock = _FakeClock()
    instrumentation = _FakePostWakeInstrumentation(clock, [0.0], advancing_liveness=False)

    assert not _run_post_wake_gate(instrumentation)


def test_post_wake_gate_handles_invalid_telemetry_without_unsafe_success() -> None:
    """Malformed telemetry remains a bounded non-converged result."""
    clock = _FakeClock()
    instrumentation = MagicMock()
    instrumentation.read_post_wake_telemetry.side_effect = ValueError("invalid pose")

    assert not app_lifecycle.wait_for_post_wake_convergence(
        instrumentation,
        MagicMock(),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )


def test_sleep_geometry_wake_failure_blocks_stage1m(monkeypatch) -> None:
    """A failed post-wake gate prevents normal startup from falling through."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()
    monkeypatch.setattr(app_lifecycle, "wait_for_post_wake_convergence", MagicMock(return_value=False))

    with pytest.raises(app_lifecycle.PostWakeConvergenceError):
        app_lifecycle.wake_up_if_sleeping(
            robot,
            MagicMock(),
            stage1_instrumentation=MagicMock(),
        )

    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
