import time
import threading
from unittest.mock import MagicMock, call
from collections.abc import Callable

import numpy as np
import pytest

from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset
from reachy_mini_conversation_app import moves
from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.wake_trace import WakeTrace
from reachy_mini_conversation_app.app_lifecycle import AuthorizedStartupSnapshot
from reachy_mini_conversation_app.face_tracking import (
    FaceBBoxSample,
    FaceBBoxWindow,
    Stage1Instrumentation,
    Stage1TelemetrySnapshot,
    pose_euler_degrees,
)
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


class _FakeMove:
    """Minimal non-emotion Move stub returning a fixed head pose."""

    def __init__(self, head: np.ndarray) -> None:
        self._head = head
        self.duration = 10.0

    def evaluate(self, t: float):
        return (self._head, np.array([0.0, 0.0]), 0.0)


def _wait_for(predicate: Callable[[], bool], timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _face_window(
    u: float = 640.0,
    v: float = 360.0,
    *,
    stable: bool = True,
    jitter_u: float = 0.0,
    jitter_v: float = 0.0,
) -> FaceBBoxWindow:
    samples = tuple(
        FaceBBoxSample(
            float(index),
            u + (-1.0) ** index * jitter_u,
            v + (-1.0) ** index * jitter_v,
            0.95,
        )
        for index in range(8)
    )
    return FaceBBoxWindow(samples, 8, 0, u, v, jitter_u, jitter_v, stable, 1280, 720)


def _telemetry(
    yaw: float = 3.0,
    *,
    target_yaw: float | None = None,
    acquisition_monotonic: float | None = None,
) -> Stage1TelemetrySnapshot:
    return Stage1TelemetrySnapshot(
        timestamp=1.0,
        acquisition_monotonic=time.monotonic() if acquisition_monotonic is None else acquisition_monotonic,
        present_head_pose=create_head_pose(yaw=yaw, degrees=True),
        present_head_joints=(0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        present_body_yaw=0.2,
        present_antennas=(0.3, -0.4),
        target_head_pose=create_head_pose(yaw=target_yaw, degrees=True) if target_yaw is not None else None,
        target_head_joints=None,
        target_body_yaw=None,
        target_antennas=None,
        daemon_timestamp="2026-09-12T11:00:00Z",
        acquisition_attempts=1,
        acquisition_elapsed_s=0.02,
        total_acquisition_elapsed_s=0.02,
        backend_last_alive=1.0,
        backend_last_alive_age_s=0.01,
        daemon_ready=True,
        daemon_error=None,
        control_loop_frequency_hz=50.0,
        active_move_count=0,
        head_tracking_enabled=False,
    )


def _configure_startup_pose(
    robot: MagicMock,
    *,
    yaw: float = 8.0,
    body_yaw: float = 0.2,
    antennas: tuple[float, float] = (0.3, -0.4),
) -> np.ndarray:
    head_pose = create_head_pose(x=0.001, y=-0.002, z=0.003, roll=1.0, pitch=2.0, yaw=yaw, degrees=True)
    robot.get_current_joint_positions.return_value = ([body_yaw, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], antennas)
    robot.get_current_head_pose.return_value = head_pose
    return head_pose


def _authorized_startup_snapshot(provenance: str) -> tuple[AuthorizedStartupSnapshot, np.ndarray]:
    head_pose = create_head_pose(x=0.004, y=-0.003, z=0.002, roll=1.5, pitch=-2.0, yaw=3.5, degrees=True)
    return (
        AuthorizedStartupSnapshot(
            head_transform=tuple(tuple(float(value) for value in row) for row in head_pose),
            head_joints=(0.31, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66),
            body_yaw=0.31,
            antennas=(-0.13, 0.17),
            daemon_timestamp="2026-09-20T00:00:00Z",
            backend_last_alive=1.0,
            acquisition_monotonic=1.0,
            authorized_monotonic=1.0,
            validation_sample_count=11,
            provenance=provenance,
        ),
        head_pose,
    )


def _stage1_manager(
    robot: MagicMock,
    *,
    commanded_yaw: float = 0.0,
    face_windows: list[FaceBBoxWindow] | None = None,
    telemetry: list[Stage1TelemetrySnapshot] | None = None,
) -> tuple[MovementManager, MagicMock]:
    instrumentation = MagicMock(spec=Stage1Instrumentation)
    instrumentation.capture_face_window.side_effect = face_windows or [_face_window(), _face_window()]
    instrumentation.read_telemetry.side_effect = telemetry or [_telemetry(), _telemetry()]
    _configure_startup_pose(robot)
    manager = MovementManager(robot, stage1_instrumentation=instrumentation)
    manager._last_commanded_pose = (
        create_head_pose(x=0.001, y=-0.002, z=0.003, roll=1.0, pitch=2.0, yaw=commanded_yaw, degrees=True),
        (0.3, -0.4),
        0.2,
    )
    manager._last_commanded_at = manager._now()
    settling = MagicMock()
    settling.to_dict.return_value = {"settled": True}
    manager._wait_for_stage1_stationary_head = MagicMock(return_value=settling)
    return manager, instrumentation


def _speech_handoff_manager(
    *,
    physical_yaw: float = 24.0,
    base_yaw: float = 0.0,
) -> tuple[MovementManager, MagicMock, MagicMock]:
    robot = MagicMock()
    _configure_startup_pose(robot, yaw=base_yaw)
    robot.get_tracked_face.return_value.detected = True
    instrumentation = MagicMock(spec=Stage1Instrumentation)
    instrumentation.read_telemetry.return_value = _telemetry(physical_yaw, target_yaw=base_yaw)
    manager = MovementManager(robot, stage1_instrumentation=instrumentation)
    manager._head_tracking = True
    manager._startup_first_target_logged = True
    manager._last_commanded_pose = (create_head_pose(yaw=base_yaw, degrees=True), (0.3, -0.4), 0.2)
    manager._last_commanded_at = manager._now()
    return manager, robot, instrumentation


def _start_without_publication(manager: MovementManager, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    issue_control_command = MagicMock(return_value=True)
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)
    manager.start()
    assert _wait_for(lambda: manager._thread is not None and manager._thread.is_alive())
    return issue_control_command


def test_non_neutral_physical_pose_is_adopted_without_fabricating_a_command() -> None:
    """Construction adopts every measured pose component without publishing."""
    robot = MagicMock()
    head_pose = _configure_startup_pose(robot, yaw=13.0, body_yaw=0.37, antennas=(0.22, -0.31))

    manager = MovementManager(robot)

    assert manager.state.last_primary_pose is not None
    assert np.allclose(manager.state.last_primary_pose[0], head_pose)
    assert manager.state.last_primary_pose[1] == pytest.approx((0.22, -0.31))
    assert manager.state.last_primary_pose[2] == pytest.approx(0.37)
    assert manager._last_commanded_pose is None
    assert manager._last_commanded_at is None
    assert manager.get_status()["startup_baseline"]["status"] == "adopted"
    robot.set_target.assert_not_called()


def test_first_worker_iteration_preserves_the_adopted_pose() -> None:
    """The first target matches the complete adopted pose instead of neutral."""
    robot = MagicMock()
    head_pose = _configure_startup_pose(robot, yaw=-11.0, body_yaw=-0.28, antennas=(-0.18, 0.27))
    manager = MovementManager(robot)
    robot.set_target.side_effect = lambda **kwargs: manager._stop_event.set()

    manager.working_loop()

    robot.set_target.assert_called_once()
    target = robot.set_target.call_args.kwargs
    assert np.allclose(target["head"], head_pose)
    assert target["antennas"] == pytest.approx((-0.18, 0.27))
    assert target["body_yaw"] == pytest.approx(-0.28)
    assert not np.allclose(target["head"], create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))
    assert manager._last_commanded_at is not None
    assert manager._startup_first_target_logged is True


def test_authorized_startup_snapshot_is_adopted_without_independent_read() -> None:
    """Stage1M and its first target use exactly the lifecycle-authorized state."""
    robot = MagicMock()
    head_pose = create_head_pose(x=0.001, y=-0.002, z=0.003, roll=2.0, pitch=1.0, yaw=-3.0, degrees=True)
    authorized = AuthorizedStartupSnapshot(
        head_transform=tuple(tuple(float(value) for value in row) for row in head_pose),
        head_joints=(0.21, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        body_yaw=0.21,
        antennas=(-0.12, 0.16),
        daemon_timestamp="2026-09-20T00:00:00Z",
        backend_last_alive=1.0,
        acquisition_monotonic=1.0,
        authorized_monotonic=1.0,
        validation_sample_count=11,
    )
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    robot.set_target.side_effect = lambda **kwargs: manager._stop_event.set()

    manager.working_loop()

    robot.get_current_head_pose.assert_not_called()
    robot.get_current_joint_positions.assert_not_called()
    target = robot.set_target.call_args.kwargs
    assert target["head"] == pytest.approx(head_pose)
    assert target["antennas"] == pytest.approx(authorized.antennas)
    assert target["body_yaw"] == pytest.approx(authorized.body_yaw)


@pytest.mark.parametrize("provenance", ["NON_SLEEP_STARTUP_GATE", "POST_WAKE_STAGE1Z"])
@pytest.mark.parametrize("authorization_age_s", [0.251, 1.204])
def test_expired_startup_publication_lease_blocks_before_motion_state_mutation(
    provenance: str,
    authorization_age_s: float,
) -> None:
    """An expired first-publication lease stops before commands, breathing, or publication."""
    robot = MagicMock()
    authorized, _ = _authorized_startup_snapshot(provenance)
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.0)
    manager._now = lambda: 1.0 + authorization_age_s
    manager.state.last_activity_time = 0.0
    queued_move = _FakeMove(create_head_pose(yaw=12.0, degrees=True))
    manager._command_queue.put(("queue_move", queued_move))

    manager.working_loop()
    manager.working_loop()

    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
    assert manager._startup_publication_event.is_set()
    assert manager.wait_for_startup_publication(0.0) is False
    assert manager._stop_event.is_set()
    assert manager._breathing_active is False
    assert not any(isinstance(move, moves.BreathingMove) for move in manager.move_queue)
    assert manager.state.current_move is None
    assert list(manager.move_queue) == []
    assert manager._command_queue.qsize() == 1
    assert manager._command_publication_count == 0
    first_publication = manager.get_status()["startup_observability"]["first_publication"]
    assert first_publication["diagnostic_status"] == "BLOCKED"
    assert first_publication["publication_monotonic"] is None
    assert first_publication["publication_authorization_age_ms"] == pytest.approx(authorization_age_s * 1000.0)
    assert first_publication["publication_authorization_lease_limit_ms"] == pytest.approx(250.0)
    assert first_publication["publication_authorization_lease_result"] == "FAIL"


@pytest.mark.parametrize("provenance", ["NON_SLEEP_STARTUP_GATE", "POST_WAKE_STAGE1Z"])
@pytest.mark.parametrize("authorization_age_s", [0.249, 0.25])
def test_valid_startup_publication_lease_publishes_authorized_snapshot(
    provenance: str,
    authorization_age_s: float,
) -> None:
    """The first valid publication uses the authorized snapshot for both startup provenances."""
    robot = MagicMock()
    authorized, head_pose = _authorized_startup_snapshot(provenance)
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.0)
    manager._now = lambda: 1.0 + authorization_age_s
    robot.set_target.side_effect = lambda **_kwargs: manager._stop_event.set()

    manager.working_loop()

    robot.set_target.assert_called_once()
    target = robot.set_target.call_args.kwargs
    assert target["head"] == pytest.approx(head_pose)
    assert target["antennas"] == pytest.approx(authorized.antennas)
    assert target["body_yaw"] == pytest.approx(authorized.body_yaw)
    assert manager.wait_for_startup_publication(0.0) is True


def test_startup_observability_marks_matching_first_publication() -> None:
    """First-publication diagnostics compare the target to the authorized snapshot."""
    robot = MagicMock()
    authorized, head_pose = _authorized_startup_snapshot("NON_SLEEP_STARTUP_GATE")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.1)
    manager._now = lambda: 1.2
    robot.set_target.side_effect = lambda **_kwargs: manager._stop_event.set()

    manager.working_loop()

    first_publication = manager.get_status()["startup_observability"]["first_publication"]
    assert first_publication["matches_authorized"] is True
    assert first_publication["target_snapshot_authorized_monotonic"] == pytest.approx(1.0)
    assert first_publication["target_snapshot_age_ms"] == pytest.approx(200.0)
    assert first_publication["publication_authorization_source"] == "final_consistency_confirmation"
    assert first_publication["publication_authorization_monotonic"] == pytest.approx(1.1)
    assert first_publication["publication_authorization_age_ms"] == pytest.approx(100.0)
    assert first_publication["publication_authorization_lease_limit_ms"] == pytest.approx(250.0)
    assert first_publication["publication_authorization_lease_result"] == "PASS"
    angles = pose_euler_degrees(head_pose)
    assert first_publication["head_rpy_deg"] == pytest.approx([angles.roll, angles.pitch, angles.yaw])


def test_old_target_snapshot_is_unambiguous_when_final_confirmation_is_fresh() -> None:
    """A fresh final confirmation can authorize the unchanged older target snapshot."""
    robot = MagicMock()
    authorized, _ = _authorized_startup_snapshot("POST_WAKE_STAGE1Z")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=2.204)
    manager._now = lambda: 2.204
    robot.set_target.side_effect = lambda **_kwargs: manager._stop_event.set()

    manager.working_loop()

    first_publication = manager.get_status()["startup_observability"]["first_publication"]
    assert first_publication["target_snapshot_age_ms"] == pytest.approx(1204.0)
    assert first_publication["publication_authorization_age_ms"] == pytest.approx(0.0)
    assert first_publication["publication_authorization_lease_result"] == "PASS"


def test_lease_expiring_after_early_worker_check_fails_closed_at_publication() -> None:
    """The exact pre-publication check closes the worker-loop timing gap."""
    robot = MagicMock()
    authorized, _ = _authorized_startup_snapshot("POST_WAKE_STAGE1Z")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.0)
    clock_values = iter([1.0, 1.2, 1.2, 1.2, 1.251])
    manager._now = lambda: next(clock_values, 1.251)

    manager.working_loop()

    robot.set_target.assert_not_called()
    assert manager.wait_for_startup_publication(0.0) is False
    first_publication = manager.get_status()["startup_observability"]["first_publication"]
    assert first_publication["diagnostic_status"] == "BLOCKED"
    assert first_publication["publication_monotonic"] is None
    assert first_publication["publication_authorization_age_ms"] == pytest.approx(251.0)
    assert first_publication["publication_authorization_lease_result"] == "FAIL"


def test_startup_observability_marks_different_first_publication() -> None:
    """A changed first target is diagnostic mismatch, not a loose movement tolerance pass."""
    robot = MagicMock()
    authorized, _ = _authorized_startup_snapshot("NON_SLEEP_STARTUP_GATE")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.1)
    manager._now = lambda: 1.2
    manager.state.last_primary_pose = (
        create_head_pose(x=0.004, y=-0.003, z=0.002, roll=1.5, pitch=-2.0, yaw=3.6, degrees=True),
        authorized.antennas,
        authorized.body_yaw,
    )
    robot.set_target.side_effect = lambda **_kwargs: manager._stop_event.set()

    manager.working_loop()

    first_publication = manager.get_status()["startup_observability"]["first_publication"]
    assert first_publication["matches_authorized"] is False
    assert first_publication["delta_rpy_deg"][2] == pytest.approx(0.1)


def test_first_publication_diagnostic_failure_does_not_block_valid_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observability errors are recorded without changing the existing set_target call."""
    robot = MagicMock()
    authorized, head_pose = _authorized_startup_snapshot("NON_SLEEP_STARTUP_GATE")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.1)
    manager._now = lambda: 1.2
    monkeypatch.setattr(moves, "pose_euler_degrees", MagicMock(side_effect=RuntimeError("diagnostic failed")))
    robot.set_target.side_effect = lambda **_kwargs: manager._stop_event.set()

    manager.working_loop()

    robot.set_target.assert_called_once()
    target = robot.set_target.call_args.kwargs
    assert target["head"] == pytest.approx(head_pose)
    assert target["antennas"] == pytest.approx(authorized.antennas)
    assert target["body_yaw"] == pytest.approx(authorized.body_yaw)
    first_publication = manager.get_status()["startup_observability"]["first_publication"]
    assert first_publication["diagnostic_status"] == "ERROR"
    assert "diagnostic failed" in first_publication["diagnostic_error"]


def test_startup_observability_records_breathing_after_first_publication() -> None:
    """Breathing attribution records when breathing follows the first publication."""
    robot = MagicMock()
    authorized, _ = _authorized_startup_snapshot("NON_SLEEP_STARTUP_GATE")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    manager.set_startup_confirmation(confirmed_at=1.1)
    manager._now = lambda: 1.2
    robot.set_target.side_effect = lambda **_kwargs: manager._stop_event.set()
    manager.working_loop()
    manager._stop_event.clear()
    robot.get_current_joint_positions.return_value = (authorized.head_joints, authorized.antennas)
    robot.get_current_head_pose.return_value = authorized.head_pose_array()
    manager.state.last_activity_time = 0.0

    manager._manage_breathing(1.2)

    breathing = manager.get_status()["startup_observability"]["breathing"]
    assert breathing["first_activation_monotonic"] == pytest.approx(1.2)
    assert breathing["before_first_publication"] is False


def test_startup_observability_records_breathing_before_first_publication() -> None:
    """Breathing attribution records if breathing becomes active before first publication."""
    robot = MagicMock()
    authorized, _ = _authorized_startup_snapshot("NON_SLEEP_STARTUP_GATE")
    manager = MovementManager(robot, authorized_startup_snapshot=authorized)
    robot.get_current_joint_positions.return_value = (authorized.head_joints, authorized.antennas)
    robot.get_current_head_pose.return_value = authorized.head_pose_array()
    manager._now = lambda: 1.2
    manager.state.last_activity_time = 0.0

    manager._manage_breathing(1.2)

    breathing = manager.get_status()["startup_observability"]["breathing"]
    assert breathing["first_activation_monotonic"] == pytest.approx(1.2)
    assert breathing["before_first_publication"] is True


def test_wake_trace_observes_stage1m_and_first_publication_without_adding_commands() -> None:
    """Startup tracing records existing MovementManager transitions only."""
    robot = MagicMock()
    head_pose = _configure_startup_pose(robot, yaw=-9.0, body_yaw=-0.2, antennas=(-0.1, 0.2))
    trace = MagicMock(spec=WakeTrace)
    manager = MovementManager(robot, wake_trace=trace)
    robot.set_target.side_effect = lambda **kwargs: manager._stop_event.set()

    manager.working_loop()

    robot.set_target.assert_called_once()
    assert np.allclose(robot.set_target.call_args.kwargs["head"], head_pose)
    phases = [entry.args[0] for entry in trace.record_transition.call_args_list]
    assert phases == ["stage1m_adoption", "movement_manager_first_publication"]


def test_startup_pose_read_failure_is_degraded_and_never_publishes() -> None:
    """An unreadable startup pose prevents the worker from publishing a target."""
    robot = MagicMock()
    robot.get_current_joint_positions.side_effect = RuntimeError("telemetry unavailable")
    manager = MovementManager(robot)

    manager.start()

    assert manager.state.last_primary_pose is None
    assert manager._thread is None
    assert manager._last_commanded_pose is None
    assert manager._last_commanded_at is None
    assert manager.get_status()["startup_baseline"] == {
        "status": "degraded",
        "adopted_at": None,
        "error": "telemetry unavailable",
    }
    robot.set_target.assert_not_called()


def test_incomplete_startup_pose_is_degraded_and_never_publishes() -> None:
    """Every head, body, and antenna component is required for safe adoption."""
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    robot.get_current_head_pose.return_value = np.eye(4)
    manager = MovementManager(robot)

    manager.start()

    assert manager.state.last_primary_pose is None
    assert manager._thread is None
    assert manager.get_status()["startup_baseline"]["status"] == "degraded"
    assert manager.get_status()["startup_baseline"]["error"] == "current head joints must contain seven finite values"
    robot.set_target.assert_not_called()


def test_startup_breathing_uses_the_adopted_non_neutral_baseline() -> None:
    """Initial breathing stays relative to the measured startup pose."""
    robot = MagicMock()
    head_pose = _configure_startup_pose(robot, yaw=16.0, body_yaw=0.42, antennas=(0.24, -0.29))
    manager = MovementManager(robot)
    current_time = manager._now()
    manager.state.last_activity_time = current_time - 1.0

    manager._manage_breathing(current_time)

    assert len(manager.move_queue) == 1
    breathing = manager.move_queue[0]
    assert isinstance(breathing, moves.BreathingMove)
    breathing_head, breathing_antennas, breathing_body_yaw = breathing.evaluate(1.0)
    assert breathing_head is not None and np.allclose(breathing_head, head_pose)
    assert breathing_antennas is not None and breathing_antennas == pytest.approx((0.24, -0.29))
    assert breathing_body_yaw == pytest.approx(0.42)
    assert not np.allclose(breathing_head, create_head_pose(0, 0, 0, 0, 0, 0, degrees=True))
    robot.set_target.assert_not_called()


def test_ordinary_queued_move_still_replaces_the_adopted_baseline() -> None:
    """Normal queued moves retain ownership after startup adoption."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    manager = MovementManager(robot)
    target_head = create_head_pose(yaw=24.0, degrees=True)
    requested_move = MagicMock(spec=moves.Move)
    requested_move.duration = 10.0
    requested_move.evaluate.return_value = (target_head, np.array([0.0, 0.0]), 0.0)
    current_time = manager._now()

    manager._handle_command("queue_move", requested_move, current_time)
    manager._manage_move_queue(current_time)
    head, antennas, body_yaw = manager._get_primary_pose(current_time)

    assert np.allclose(head, target_head)
    assert antennas == pytest.approx((0.0, 0.0))
    assert body_yaw == pytest.approx(0.0)
    assert manager._startup_breathing_pending is False
    robot.set_target.assert_not_called()


def test_stop_can_skip_neutral_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep shutdown should stop the movement loop without undoing the sleep pose."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    manager = MovementManager(robot)
    started = threading.Event()

    def fake_working_loop() -> None:
        started.set()
        while not manager._stop_event.is_set():
            time.sleep(0.001)

    monkeypatch.setattr(manager, "working_loop", fake_working_loop)

    manager.start()
    assert started.wait(timeout=1.0)

    manager.stop(reset_to_neutral=False)

    assert manager._thread is None
    robot.goto_target.assert_not_called()


def test_head_tracking_follows_speaking() -> None:
    """Once enabled, tracking owns the head when idle and releases it while the assistant speaks."""
    robot = MagicMock()
    _configure_startup_pose(robot, yaw=0.0, body_yaw=0.0, antennas=(0.0, 0.0))
    robot.get_tracked_face.return_value.detected = True
    instrumentation = MagicMock(spec=Stage1Instrumentation)
    instrumentation.read_telemetry.return_value = _telemetry(18.0, target_yaw=0.0)
    daemon_status = MagicMock()
    daemon_status.model_dump.side_effect = [
        {"head_tracking_enabled": True},
        {"head_tracking_enabled": False},
    ]
    robot.client.get_status.return_value = daemon_status
    manager = MovementManager(robot, stage1_instrumentation=instrumentation)
    manager.start()
    try:
        # The head_tracking tool enables tracking with full weight.
        manager.set_head_tracking(True)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)

        # Speaking with a locked face captures the anchor and releases the head.
        manager.set_speaking(True)
        assert _wait_for(lambda: call(weight=0.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is not None)

        # Done speaking hands the head back to tracking.
        robot.start_head_tracking.reset_mock()
        manager.set_speaking(False)
        assert _wait_for(lambda: call(weight=1.0) in robot.start_head_tracking.call_args_list)
        assert _wait_for(lambda: manager._track_anchor is None)
    finally:
        manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_called_once()


def test_speaking_handoff_publishes_tracked_physical_pose_before_pausing_tracking() -> None:
    """The tracked physical pose becomes the daemon base before tracking influence is removed."""
    manager, robot, instrumentation = _speech_handoff_manager(physical_yaw=24.0, base_yaw=0.0)
    order: list[str] = []

    def capture_telemetry(**_kwargs: object) -> Stage1TelemetrySnapshot:
        order.append("capture")
        return _telemetry(24.0)

    instrumentation.read_telemetry.side_effect = capture_telemetry
    robot.set_target.side_effect = lambda **kwargs: order.append("publish")
    robot.start_head_tracking.side_effect = lambda **kwargs: order.append("pause")

    manager._handle_command("set_speaking", True, manager._now())

    assert order == ["capture", "publish", "pause"]
    assert np.allclose(robot.set_target.call_args.kwargs["head"], create_head_pose(yaw=24.0, degrees=True))
    robot.start_head_tracking.assert_called_once_with(weight=0.0)


def test_speaking_handoff_replaces_stale_center_base_with_physical_anchor() -> None:
    """A centered app target cannot reappear after a fresh tracked pose is captured."""
    manager, _, _ = _speech_handoff_manager(physical_yaw=-19.0, base_yaw=0.0)

    manager._handle_command("set_speaking", True, manager._now())
    head, _, _ = manager._get_primary_pose(manager._now())

    assert np.allclose(head, create_head_pose(yaw=-19.0, degrees=True))
    assert manager.state.last_primary_pose is not None
    assert np.allclose(manager.state.last_primary_pose[0], head)


def test_speaking_handoff_cancels_centered_breathing_before_tracking_pause() -> None:
    """Idle breathing cannot republish its neutral base during tracked speech."""
    manager, robot, _ = _speech_handoff_manager(physical_yaw=21.0, base_yaw=0.0)
    manager.state.current_move = moves.BreathingMove(
        create_head_pose(yaw=0.0, degrees=True),
        (0.3, -0.4),
        base_head_pose=create_head_pose(yaw=0.0, degrees=True),
    )
    manager.state.move_start_time = manager._now()
    manager._breathing_active = True

    manager._handle_command("set_speaking", True, manager._now())

    assert manager.state.current_move is None
    assert manager._breathing_active is False
    assert np.allclose(robot.set_target.call_args.kwargs["head"], create_head_pose(yaw=21.0, degrees=True))


def test_speaking_suppresses_new_idle_breathing() -> None:
    """The neutral-based idle move cannot restart while the speech anchor owns the head."""
    manager, _, _ = _speech_handoff_manager()
    manager._handle_command("set_speaking", True, manager._now())
    manager.state.last_activity_time = manager._now() - 10.0

    manager._manage_breathing(manager._now())

    assert manager.state.current_move is None
    assert not manager.move_queue
    assert manager._breathing_active is False


def test_speaking_restoration_keeps_synchronized_anchor_as_app_base() -> None:
    """Tracking resumes before the anchor is released and cannot reveal the old center target."""
    manager, robot, _ = _speech_handoff_manager(physical_yaw=17.0, base_yaw=0.0)
    manager._handle_command("set_speaking", True, manager._now())
    robot.start_head_tracking.reset_mock()

    manager._handle_command("set_speaking", False, manager._now())
    head, _, _ = manager._get_primary_pose(manager._now())

    robot.start_head_tracking.assert_called_once_with(weight=1.0)
    assert manager._track_anchor is None
    assert np.allclose(head, create_head_pose(yaw=17.0, degrees=True))


def test_speaking_handoff_missing_telemetry_fails_closed() -> None:
    """Tracking remains fully active when no fresh telemetry reader is available."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    manager = MovementManager(robot)
    manager._head_tracking = True

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_speaking_handoff_stale_command_baseline_fails_closed() -> None:
    """Fresh physical telemetry cannot be paired with an expired app command baseline."""
    manager, robot, _ = _speech_handoff_manager()
    manager._last_commanded_at = manager._now() - moves.STAGE1_COMMAND_FRESHNESS_S - 0.001

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_speaking_handoff_stale_physical_telemetry_fails_closed() -> None:
    """A telemetry freshness rejection cannot remove daemon tracking influence."""
    manager, robot, instrumentation = _speech_handoff_manager()
    instrumentation.read_telemetry.side_effect = RuntimeError("telemetry sample exceeded the freshness bound")

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_speaking_handoff_expired_telemetry_lease_fails_before_state_mutation() -> None:
    """An expired local monotonic lease cannot publish or cancel breathing."""
    manager, robot, instrumentation = _speech_handoff_manager()
    manager._now = lambda: 100.0
    manager._last_commanded_at = 100.0
    instrumentation.read_telemetry.return_value = _telemetry(acquisition_monotonic=99.749)
    breathing = moves.BreathingMove(
        create_head_pose(yaw=0.0, degrees=True),
        (0.3, -0.4),
        base_head_pose=create_head_pose(yaw=0.0, degrees=True),
    )
    manager.state.current_move = breathing
    manager.state.move_start_time = 99.0
    manager._breathing_active = True
    robot.set_target.reset_mock()
    robot.start_head_tracking.reset_mock()

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    assert manager.state.current_move is breathing
    assert manager._breathing_active is True
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_speaking_handoff_waits_for_fresh_telemetry_before_mutating_tracking_state() -> None:
    """No anchor, breathing, or tracking-weight change occurs before telemetry succeeds."""
    manager, robot, instrumentation = _speech_handoff_manager()
    breathing = moves.BreathingMove(
        create_head_pose(yaw=0.0, degrees=True),
        (0.3, -0.4),
        base_head_pose=create_head_pose(yaw=0.0, degrees=True),
    )
    manager.state.current_move = breathing
    manager.state.move_start_time = manager._now()
    manager._breathing_active = True

    def reject_after_wait(**_kwargs: object) -> Stage1TelemetrySnapshot:
        assert manager._is_speaking is False
        assert manager._track_anchor is None
        assert manager.state.current_move is breathing
        assert manager._breathing_active is True
        robot.set_target.assert_not_called()
        robot.start_head_tracking.assert_not_called()
        raise RuntimeError("telemetry sample exceeded the freshness bound")

    instrumentation.read_telemetry.side_effect = reject_after_wait

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    assert manager.state.current_move is breathing
    assert manager._breathing_active is True
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_speaking_handoff_bad_control_loop_fails_before_anchor_breathing_or_tracking_weight() -> None:
    """A genuinely unhealthy loop cannot publish an anchor or reduce tracking influence."""
    manager, robot, instrumentation = _speech_handoff_manager()
    breathing = moves.BreathingMove(
        create_head_pose(yaw=0.0, degrees=True),
        (0.3, -0.4),
        base_head_pose=create_head_pose(yaw=0.0, degrees=True),
    )
    manager.state.current_move = breathing
    manager.state.move_start_time = manager._now()
    manager._breathing_active = True
    instrumentation.read_telemetry.side_effect = RuntimeError("Stage 1 telemetry control loop frequency is unhealthy")

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    assert manager.state.current_move is breathing
    assert manager._breathing_active is True
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_speaking_handoff_anchor_publication_failure_fails_closed() -> None:
    """A rejected base publication leaves tracking at full weight."""
    manager, robot, _ = _speech_handoff_manager()
    robot.set_target.side_effect = ConnectionError("publication failed")

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._is_speaking is False
    assert manager._track_anchor is None
    robot.start_head_tracking.assert_not_called()


def test_speaking_without_face_lock_leaves_tracking_unchanged() -> None:
    """Speech cannot pause tracking before the daemon has acquired a face."""
    manager, robot, instrumentation = _speech_handoff_manager()
    robot.get_tracked_face.return_value.detected = False

    manager._handle_command("set_speaking", True, manager._now())

    instrumentation.read_telemetry.assert_not_called()
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_tracking_disabled_speech_remains_motion_free() -> None:
    """The speech signal remains a no-op when tracking is disabled."""
    manager, robot, instrumentation = _speech_handoff_manager()
    manager._head_tracking = False

    manager._handle_command("set_speaking", True, manager._now())

    instrumentation.read_telemetry.assert_not_called()
    robot.set_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_non_speech_tracking_toggle_does_not_use_handoff_telemetry() -> None:
    """Ordinary tracking enablement remains independent of the speech handoff."""
    manager, robot, instrumentation = _speech_handoff_manager()
    manager._head_tracking = False

    manager._handle_command("set_head_tracking", True, manager._now())

    robot.start_head_tracking.assert_called_once_with(weight=1.0)
    instrumentation.read_telemetry.assert_not_called()


def test_speaking_handoff_marks_tick_publication_to_prevent_duplicate_target() -> None:
    """The ordered anchor preload remains the control loop's only publication for its tick."""
    manager, robot, _ = _speech_handoff_manager()

    manager._handle_command("set_speaking", True, manager._now())

    assert manager._handoff_published_this_tick is True
    robot.set_target.assert_called_once()


def test_tracking_toggle_failure_does_not_change_application_state() -> None:
    """A rejected SDK toggle cannot make local tracking state disagree with the daemon."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    manager = MovementManager(robot)
    robot.start_head_tracking.side_effect = ConnectionError("enable failed")

    manager._handle_command("set_head_tracking", True, manager._now())

    assert manager.get_head_tracking_enabled() is False
    robot.start_head_tracking.assert_called_once_with(weight=1.0)

    manager._head_tracking = True
    robot.stop_head_tracking.side_effect = ConnectionError("disable failed")
    manager._handle_command("set_head_tracking", False, manager._now())

    assert manager.get_head_tracking_enabled() is True
    robot.stop_head_tracking.assert_called_once_with()


def test_sdk_recovery_preserves_tracking_ownership_for_shutdown_cleanup() -> None:
    """SDK recovery must not erase the fact that this manager enabled daemon tracking."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    daemon_status = MagicMock()
    daemon_status.model_dump.side_effect = [
        {"head_tracking_enabled": True},
        {"head_tracking_enabled": False},
    ]
    robot.client.get_status.return_value = daemon_status
    manager = MovementManager(robot)
    manager._head_tracking = True

    manager._establish_sdk_recovery_boundary("test disconnect")

    assert manager.get_head_tracking_enabled() is True
    manager.stop(reset_to_neutral=False)
    assert manager.get_head_tracking_enabled() is False
    robot.stop_head_tracking.assert_called_once_with()


def test_graceful_shutdown_with_tracking_disabled_sends_no_tracking_command() -> None:
    """Stopping an already-disabled manager does not send a redundant daemon command."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    manager = MovementManager(robot)

    manager.stop(reset_to_neutral=False)

    robot.stop_head_tracking.assert_not_called()
    robot.client.get_status.assert_not_called()


def test_speaking_anchor_composes_emotions_and_holds_dances_from_neutral() -> None:
    """While speaking: hold the anchor, compose emotions onto it, play dances from neutral."""
    robot = MagicMock()
    _configure_startup_pose(robot)
    manager = MovementManager(robot)
    anchor = create_head_pose(0, 0, 0, 0, 0, 20, degrees=True)
    manager._track_anchor = anchor

    # No move: the head holds the captured look-at anchor.
    manager.state.current_move = None
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, anchor)

    # Emotion: composed onto the anchor exactly like the daemon wobble.
    emotion_head = create_head_pose(0, 0, 0, 0, 0, 15, degrees=True)
    recorded = MagicMock()
    recorded.get.return_value = _FakeMove(emotion_head)
    manager.state.current_move = EmotionQueueMove("happy", recorded)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, compose_world_offset(anchor, emotion_head))

    # Any other move (e.g. a dance) plays from its own neutral base, ignoring the anchor.
    dance_head = create_head_pose(0, 0, 0, 0, 25, 0, degrees=True)
    manager.state.current_move = _FakeMove(dance_head)
    manager.state.move_start_time = manager._now()
    head, _, _ = manager._get_primary_pose(manager._now())
    assert np.allclose(head, dance_head)


def test_stage1_calibration_publishes_one_commanded_baseline_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prepared session publishes once from the frozen command baseline."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(
        robot,
        commanded_yaw=0.0,
        face_windows=[_face_window(), _face_window(), _face_window(650.0)],
        telemetry=[_telemetry(3.0), _telemetry(3.0), _telemetry(3.0), _telemetry(1.3)],
    )
    monkeypatch.setattr(moves, "STAGE1_SETTLING_S", 0.0)

    preview = manager.prepare_stage1_horizontal_calibration(24.0)
    robot.set_target.assert_not_called()
    result = manager.execute_stage1_horizontal_calibration(preview.session_id)

    assert result.accepted is True
    assert result.command_count == 1
    assert result.preview.target.clamped_delta_yaw == pytest.approx(1.5)
    robot.set_target.assert_called_once()
    command = robot.set_target.call_args.kwargs
    angles = pose_euler_degrees(command["head"])
    assert angles.yaw == pytest.approx(1.5)
    assert angles.pitch == pytest.approx(2.0)
    assert command["antennas"] == (0.3, -0.4)
    assert command["body_yaw"] == pytest.approx(0.2)
    assert manager._stage1_calibration is None
    assert manager._thread is None
    assert manager._command_queue.empty()
    robot.goto_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()
    assert result.face_displacement is not None
    assert result.face_displacement.left_motion_consistent is True
    assert instrumentation.capture_face_window.call_count == 3

    with pytest.raises(ValueError, match="stale or consumed"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)
    robot.set_target.assert_called_once()


def test_stage1_calibration_rejects_stale_session_without_publication() -> None:
    """A stale authorization cannot reach the movement command path."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    preview = manager.prepare_stage1_horizontal_calibration(-20.0)

    with pytest.raises(ValueError, match="stale"):
        manager.execute_stage1_horizontal_calibration("wrong-session")
    assert manager._stage1_calibration is not None
    assert manager._stage1_calibration.session_id == preview.session_id
    assert manager._stage1_calibration.consumed is False
    robot.set_target.assert_not_called()


@pytest.mark.parametrize("failure_kind", ["face_not_stable", "no_face", "multiple_faces", "quality_failure"])
def test_stage1_preview_face_failure_preserves_running_manager_and_baseline(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    """A read-only face rejection leaves the active manager and baseline untouched."""
    robot = MagicMock()
    failed_window = (
        _face_window(stable=False)
        if failure_kind == "face_not_stable"
        else FaceBBoxWindow((), 8, 8, None, None, None, None, False)
    )
    manager, instrumentation = _stage1_manager(robot, face_windows=[failed_window])
    issue_control_command = MagicMock()
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)
    manager.start()
    assert _wait_for(lambda: manager._thread is not None and manager._thread.is_alive())
    baseline_pose = (
        manager._last_commanded_pose[0].copy(),
        manager._last_commanded_pose[1],
        manager._last_commanded_pose[2],
    )
    baseline_at = manager._last_commanded_at

    try:
        with pytest.raises(RuntimeError, match="FACE NOT STABLE"):
            manager.prepare_stage1_horizontal_calibration(1.0)

        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._command_queue.empty()
        assert manager._last_commanded_at == baseline_at
        assert np.array_equal(manager._last_commanded_pose[0], baseline_pose[0])
        assert manager._last_commanded_pose[1:] == baseline_pose[1:]
        manager._wait_for_stage1_stationary_head.assert_called_once()
        instrumentation.capture_face_window.assert_called_once_with()
    finally:
        manager.stop(reset_to_neutral=False)

    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_stage1_failed_preview_does_not_refresh_baseline_and_stale_retry_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preview failure preserves natural baseline aging and the stale guard."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot, face_windows=[_face_window(stable=False)])
    current_time = [100.0]

    def fake_now() -> float:
        return current_time[0]

    manager._now = fake_now
    manager._last_commanded_at = current_time[0]
    monkeypatch.setattr(manager, "_issue_control_command", MagicMock())
    manager.start()
    assert _wait_for(lambda: manager._thread is not None and manager._thread.is_alive())

    try:
        with pytest.raises(RuntimeError, match="FACE NOT STABLE"):
            manager.prepare_stage1_horizontal_calibration(1.0)
        assert manager._last_commanded_at == 100.0

        current_time[0] += moves.STAGE1_COMMAND_FRESHNESS_S + 0.01
        with pytest.raises(RuntimeError, match="baseline is stale"):
            manager.prepare_stage1_horizontal_calibration(1.0)

        manager._wait_for_stage1_stationary_head.assert_called_once()
        instrumentation.capture_face_window.assert_called_once_with()
        assert manager._last_commanded_at == 100.0
        assert manager._stage1_calibration is None
        assert manager._thread is not None and manager._thread.is_alive()
    finally:
        manager.stop(reset_to_neutral=False)

    robot.set_target.assert_not_called()


def test_stage1_preview_telemetry_failure_preserves_running_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-only telemetry failure occurs before calibration ownership."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot)
    manager._wait_for_stage1_stationary_head.side_effect = RuntimeError("telemetry unavailable")
    monkeypatch.setattr(manager, "_issue_control_command", MagicMock())
    manager.start()
    assert _wait_for(lambda: manager._thread is not None and manager._thread.is_alive())

    try:
        with pytest.raises(RuntimeError, match="telemetry unavailable"):
            manager.prepare_stage1_horizontal_calibration(1.0)

        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._command_queue.empty()
        instrumentation.capture_face_window.assert_not_called()
    finally:
        manager.stop(reset_to_neutral=False)

    robot.set_target.assert_not_called()


def test_stage1_successful_preview_acquires_one_session_without_movement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful preview freezes one owner but does not publish motion."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    issue_control_command = MagicMock()
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)

    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    assert manager.current_robot is robot
    assert manager._stage1_calibration is not None
    assert manager._stage1_calibration.session_id == preview.session_id
    assert manager._thread is None
    assert manager._command_queue.empty()
    issue_control_command.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_stage1_calibration_rejects_second_active_session() -> None:
    """Only one session can own the calibration boundary at a time."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    manager.prepare_stage1_horizontal_calibration(1.0)

    with pytest.raises(RuntimeError, match="already active"):
        manager.prepare_stage1_horizontal_calibration(-1.0)
    robot.set_target.assert_not_called()


def test_stage1_calibration_rejects_pose_change_after_preview() -> None:
    """Human approval cannot be reused after the reviewed pose changes."""
    robot = MagicMock()
    manager, _ = _stage1_manager(
        robot,
        face_windows=[_face_window(), _face_window()],
        telemetry=[_telemetry(3.0), _telemetry(3.3)],
    )
    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    with pytest.raises(RuntimeError, match="pose changed"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)
    robot.set_target.assert_not_called()


@pytest.mark.parametrize(
    "execution_face",
    [
        _face_window(640.0, 360.0),
        _face_window(650.0, 365.0, jitter_u=3.0, jitter_v=2.0),
    ],
    ids=["same_position", "small_detector_jitter"],
)
def test_stage1_calibration_accepts_consistent_face_position(
    monkeypatch: pytest.MonkeyPatch,
    execution_face: FaceBBoxWindow,
) -> None:
    """Fresh stable faces inside the one-percent position envelope remain eligible."""
    robot = MagicMock()
    manager, _ = _stage1_manager(
        robot,
        face_windows=[_face_window(), execution_face, execution_face],
        telemetry=[_telemetry(), _telemetry(), _telemetry(), _telemetry()],
    )
    monkeypatch.setattr(moves, "STAGE1_SETTLING_S", 0.0)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    result = manager.execute_stage1_horizontal_calibration(preview.session_id)

    assert result.command_count == 1
    robot.set_target.assert_called_once()


@pytest.mark.parametrize(
    "execution_face",
    [_face_window(660.0, 360.0), _face_window(640.0, 375.0), _face_window(680.0, 390.0)],
    ids=["horizontal", "vertical", "stable_new_location"],
)
def test_stage1_calibration_rejects_relocated_face(execution_face: FaceBBoxWindow) -> None:
    """A stable face at a materially different location cannot reuse the prepared target."""
    robot = MagicMock()
    manager, _ = _stage1_manager(
        robot,
        face_windows=[_face_window(), execution_face],
        telemetry=[_telemetry(), _telemetry()],
    )
    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    with pytest.raises(RuntimeError, match="FACE POSITION CHANGED"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)

    assert manager._stage1_calibration is None
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_stage1_calibration_rejects_expired_session_without_movement() -> None:
    """The existing age limit remains fail-safe and consumes the expired session."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    assert manager._stage1_calibration is not None
    manager._stage1_calibration.prepared_at -= moves.STAGE1_PREVIEW_MAX_AGE_S + 0.001

    with pytest.raises(RuntimeError, match="preview has expired"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)

    assert manager._stage1_calibration is None
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_stage1_calibration_preserves_execution_face_stability_gate() -> None:
    """An unstable fresh face remains rejected before position comparison or movement."""
    robot = MagicMock()
    manager, _ = _stage1_manager(
        robot,
        face_windows=[_face_window(), _face_window(stable=False)],
        telemetry=[_telemetry(), _telemetry()],
    )
    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    with pytest.raises(RuntimeError, match="FACE NOT STABLE"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)

    assert manager._stage1_calibration is None
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_stage1_telemetry_failure_consumes_session_without_motion() -> None:
    """A failed execution precondition is non-reusable and never reaches movement."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(
        robot,
        face_windows=[_face_window(), _face_window()],
        telemetry=[_telemetry(), _telemetry()],
    )
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    instrumentation.read_telemetry.side_effect = RuntimeError("telemetry unavailable")

    with pytest.raises(RuntimeError, match="telemetry unavailable"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)
    with pytest.raises(ValueError, match="stale or consumed"):
        manager.execute_stage1_horizontal_calibration(preview.session_id)

    assert manager._stage1_calibration is None
    assert manager._thread is None
    assert manager._command_queue.empty()
    robot.set_target.assert_not_called()


def test_stage1_after_failed_session_requires_new_preview() -> None:
    """A failed precondition cannot be retried without preparing a distinct session."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot)
    failed_preview = manager.prepare_stage1_horizontal_calibration(20.0)
    instrumentation.read_telemetry.side_effect = RuntimeError("telemetry unavailable")
    with pytest.raises(RuntimeError):
        manager.execute_stage1_horizontal_calibration(failed_preview.session_id)

    manager._last_commanded_at = manager._now()
    instrumentation.read_telemetry.side_effect = [_telemetry(), _telemetry()]
    instrumentation.capture_face_window.side_effect = [_face_window()]
    new_preview = manager.prepare_stage1_horizontal_calibration(20.0)

    assert new_preview.session_id != failed_preview.session_id
    robot.set_target.assert_not_called()


def test_stage1_running_prepare_rejection_preserves_running_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test A: rejection before session ownership cannot stop a running manager."""
    manager, _ = _stage1_manager(MagicMock())
    _start_without_publication(manager, monkeypatch)
    manager._stage1_instrumentation = None

    try:
        with pytest.raises(RuntimeError, match="instrumentation is unavailable"):
            manager.prepare_stage1_horizontal_calibration(1.0)
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_lifecycle is None
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_running_execute_telemetry_failure_restores_and_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test B: telemetry failure releases ownership and restores a running manager."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot)
    _start_without_publication(manager, monkeypatch)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    instrumentation.read_telemetry.side_effect = RuntimeError("telemetry unavailable")

    try:
        with pytest.raises(RuntimeError, match="telemetry unavailable"):
            manager.execute_stage1_horizontal_calibration(preview.session_id)
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._stage1_lifecycle is None
        assert manager._command_queue.empty()
        robot.set_target.assert_not_called()
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_running_stale_face_restores_before_any_calibration_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests C and I: stale-face validation stays ahead of motor publication."""
    robot = MagicMock()
    manager, _ = _stage1_manager(
        robot,
        face_windows=[_face_window(), _face_window(680.0, 390.0)],
        telemetry=[_telemetry(), _telemetry()],
    )
    issue_control_command = _start_without_publication(manager, monkeypatch)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    issue_control_command.reset_mock()

    try:
        with pytest.raises(RuntimeError, match="FACE POSITION CHANGED"):
            manager.execute_stage1_horizontal_calibration(preview.session_id)
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        issue_control_command.assert_not_called()
        robot.set_target.assert_not_called()
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_cancel_restores_running_state_and_removes_pending_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test D: cancellation releases the freeze and invalidates its command."""
    manager, _ = _stage1_manager(MagicMock())
    _start_without_publication(manager, monkeypatch)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    acknowledgement = moves._Stage1CalibrationAcknowledgement()
    manager._command_queue.put(("stage1_horizontal_calibration", (preview.session_id, acknowledgement)))

    try:
        manager.cancel_stage1_horizontal_calibration(preview.session_id)
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._stage1_lifecycle is None
        assert manager._command_queue.empty()
        assert acknowledgement.event.is_set() is False
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_expiry_restores_running_state_and_removes_pending_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test E: session expiry uses the same terminal release path."""
    manager, _ = _stage1_manager(MagicMock())
    _start_without_publication(manager, monkeypatch)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    try:
        manager._expire_stage1_horizontal_calibration(preview.session_id)
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._stage1_lifecycle is None
        assert manager._command_queue.empty()
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_stopped_state_remains_stopped_after_terminal_release() -> None:
    """Test F: temporary Stage 1 worker use cannot persistently start the manager."""
    manager, _ = _stage1_manager(MagicMock())
    preview = manager.prepare_stage1_horizontal_calibration(20.0)

    manager.cancel_stage1_horizontal_calibration(preview.session_id)

    assert manager._thread is None
    assert manager._stop_event.is_set()
    assert manager._stage1_calibration is None
    assert manager._stage1_lifecycle is None


def test_stage1_freeze_rejects_autonomous_and_direct_movement_without_accumulation() -> None:
    """Test G: queued moves and idle scheduling are suppressed for the owned freeze."""
    manager, _ = _stage1_manager(MagicMock())
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    emotion = MagicMock(spec=EmotionQueueMove)

    try:
        with pytest.raises(RuntimeError, match="rejected while Stage 1 owns"):
            manager.queue_move(emotion)
        assert manager.is_idle() is False
        assert manager._command_queue.empty()
        assert not manager.move_queue
    finally:
        manager.cancel_stage1_horizontal_calibration(preview.session_id)


def test_stage1_abnormal_release_purges_late_command_before_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test H: a late calibration command cannot execute after abnormal cleanup."""
    manager, _ = _stage1_manager(MagicMock())
    execute_on_worker = MagicMock()
    monkeypatch.setattr(manager, "_execute_stage1_horizontal_calibration", execute_on_worker)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    manager._command_queue.put(
        ("stage1_horizontal_calibration", (preview.session_id, moves._Stage1CalibrationAcknowledgement()))
    )

    manager._release_stage1_calibration(preview.session_id, "abnormal failure")
    assert manager._command_queue.empty()
    manager.start()
    time.sleep(0.05)
    manager.stop(reset_to_neutral=False)

    execute_on_worker.assert_not_called()
    assert manager._stage1_calibration is None


def test_stage1_lost_sdk_connection_is_not_counted_as_success_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test J: SDK publication failure rejects calibration and still restores lifecycle."""
    robot = MagicMock()
    manager, _ = _stage1_manager(
        robot,
        face_windows=[_face_window(), _face_window()],
        telemetry=[_telemetry(), _telemetry()],
    )
    issue_control_command = manager._issue_control_command
    _start_without_publication(manager, monkeypatch)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    robot.set_target.side_effect = RuntimeError("Lost connection with the server")
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)
    publication_count = manager._command_publication_count

    try:
        with pytest.raises(RuntimeError, match="command publication failed"):
            manager.execute_stage1_horizontal_calibration(preview.session_id)
        assert manager._command_publication_count == publication_count
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._stage1_lifecycle is None
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_double_cleanup_and_execute_failure_race_leave_one_live_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent failure and duplicate cleanup remain idempotent and motor-free."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot)
    _start_without_publication(manager, monkeypatch)
    preview = manager.prepare_stage1_horizontal_calibration(20.0)
    telemetry_entered = threading.Event()
    release_telemetry = threading.Event()

    def fail_telemetry(**_kwargs: object) -> Stage1TelemetrySnapshot:
        telemetry_entered.set()
        assert release_telemetry.wait(timeout=2.0)
        raise RuntimeError("telemetry failed")

    instrumentation.read_telemetry.side_effect = fail_telemetry
    execution_errors: list[Exception] = []

    def execute() -> None:
        try:
            manager.execute_stage1_horizontal_calibration(preview.session_id)
        except Exception as exc:
            execution_errors.append(exc)

    execution = threading.Thread(target=execute)
    execution.start()
    assert telemetry_entered.wait(timeout=1.0)
    cleanup_one = threading.Thread(
        target=manager._release_stage1_calibration,
        args=(preview.session_id, "concurrent cleanup"),
    )
    cleanup_two = threading.Thread(
        target=manager._release_stage1_calibration,
        args=(preview.session_id, "duplicate cleanup"),
    )
    cleanup_one.start()
    cleanup_two.start()
    release_telemetry.set()
    execution.join(timeout=3.0)
    cleanup_one.join(timeout=3.0)
    cleanup_two.join(timeout=3.0)

    try:
        assert not execution.is_alive()
        assert not cleanup_one.is_alive()
        assert not cleanup_two.is_alive()
        assert execution_errors and "telemetry failed" in str(execution_errors[0])
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._stage1_calibration is None
        assert manager._stage1_lifecycle is None
        assert manager._command_queue.empty()
        robot.set_target.assert_not_called()
    finally:
        manager.stop(reset_to_neutral=False)


@pytest.mark.parametrize(
    ("command_age", "expected_error"),
    [
        (moves.STAGE1_COMMAND_FRESHNESS_S, None),
        (moves.STAGE1_COMMAND_FRESHNESS_S + 0.001, "baseline is stale"),
    ],
)
def test_stage1_preview_baseline_capture_preserves_normal_freshness_rule(
    command_age: float,
    expected_error: str | None,
) -> None:
    """Preview admission uses the unchanged freshness rule for published commands."""
    manager, _ = _stage1_manager(MagicMock())
    manager._last_commanded_at = 100.0 - command_age
    acknowledgement = moves._Stage1PreviewAcknowledgement()

    manager._begin_stage1_preview_stabilization_on_worker("owner", acknowledgement, 100.0)

    assert acknowledgement.event.is_set()
    if expected_error is None:
        assert acknowledgement.error is None
        assert acknowledgement.baseline is not None
        assert acknowledgement.baseline.commanded_at == 100.0 - command_age
        assert acknowledgement.baseline.captured_at == 100.0
    else:
        assert acknowledgement.error == f"Stage 1 command {expected_error}"
        assert acknowledgement.baseline is None
        assert manager._stage1_preview_baseline is None


def test_stage1_preview_capture_rejects_tracking_and_primary_movement() -> None:
    """Only an otherwise valid Stage 1 worker state can own a preview baseline."""
    manager, _ = _stage1_manager(MagicMock())
    manager._last_commanded_at = 99.9
    manager._head_tracking = True
    tracking = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("tracking", tracking, 100.0)
    assert tracking.error == "legacy head tracking must be disabled for Stage 1 calibration"

    manager._head_tracking = False
    manager.state.current_move = MagicMock(spec=moves.Move)
    movement = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("movement", movement, 100.0)
    assert movement.error == "another primary movement is active"
    assert manager._stage1_preview_baseline is None


def test_stage1_three_second_stabilization_preserves_timestamps_and_issues_no_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stationary preview ages naturally while its explicit baseline remains valid."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot)
    current_time = [100.0]

    def fake_now() -> float:
        return current_time[0]

    def advance_clock(duration: float) -> None:
        current_time[0] += duration

    manager._now = fake_now
    manager._sleep = advance_clock
    manager._last_commanded_at = 99.9
    instrumentation.read_telemetry.side_effect = None
    instrumentation.read_telemetry.return_value = _telemetry()
    issue_control_command = MagicMock()
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)
    acknowledgement = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("owner", acknowledgement, current_time[0])

    settling = MovementManager._wait_for_stage1_stationary_head(manager, "owner")

    assert settling.duration >= moves.STAGE1_PREVIEW_SETTLING_DURATION_S
    assert settling.yaw_range == pytest.approx(0.0)
    assert settling.pitch_range == pytest.approx(0.0)
    assert manager._last_commanded_at == 99.9
    assert acknowledgement.baseline is not None
    assert acknowledgement.baseline.captured_at == 100.0
    assert manager._stage1_preview_suppresses_publication(current_time[0]) is True
    issue_control_command.assert_not_called()
    robot.set_target.assert_not_called()


def test_stage1_preview_suppression_keeps_worker_running_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live worker remains healthy while preview ownership suppresses output."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    issue_control_command = MagicMock()
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)
    commanded_at = manager._last_commanded_at

    baseline = manager._begin_stage1_preview_stabilization("owner")
    try:
        time.sleep(0.05)
        assert baseline.owner_session_id == "owner"
        assert manager._thread is not None and manager._thread.is_alive()
        assert manager._last_commanded_at == commanded_at
        issue_control_command.assert_not_called()
    finally:
        manager._end_stage1_preview_stabilization("owner", "test complete")
        manager.stop(reset_to_neutral=False)

    robot.set_target.assert_not_called()


def test_stage1_stabilized_baseline_can_only_be_consumed_by_its_owner() -> None:
    """A different session cannot use another preview's freshness exception."""
    manager, _ = _stage1_manager(MagicMock())
    manager._now = lambda: 100.1
    manager._last_commanded_at = 99.9
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("owner", begin, 100.0)

    wrong_owner = moves._Stage1FreezeAcknowledgement()
    manager._freeze_stage1_horizontal_calibration("other", wrong_owner, 100.1)

    assert wrong_owner.error == "Stage 1 preview baseline is not owned by this session"
    assert manager._stage1_preview_baseline is not None
    assert manager._stage1_preview_baseline.owner_session_id == "owner"
    assert not manager._stop_event.is_set()

    owner = moves._Stage1FreezeAcknowledgement()
    manager._freeze_stage1_horizontal_calibration("owner", owner, 100.1)

    assert owner.error is None
    assert owner.commanded_pose is not None
    assert owner.commanded_at == 99.9
    assert manager._stage1_preview_baseline is None
    assert manager._stop_event.is_set()


def test_stage1_preview_cancel_and_failure_remove_owned_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancellation and preview failure both remove the session-owned exemption."""
    manager, _ = _stage1_manager(MagicMock(), face_windows=[_face_window(stable=False)])
    manager._last_commanded_at = manager._now()
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("cancelled", begin, manager._now())
    manager._cancel_stage1_preview_stabilization("cancelled")
    assert manager._stage1_preview_baseline is None

    monkeypatch.setattr(manager, "_issue_control_command", MagicMock())
    manager.start()
    assert _wait_for(lambda: manager._thread is not None and manager._thread.is_alive())
    try:
        with pytest.raises(RuntimeError, match="FACE NOT STABLE"):
            manager.prepare_stage1_horizontal_calibration(1.0)
        assert manager._stage1_preview_baseline is None
        assert manager._thread is not None and manager._thread.is_alive()
    finally:
        manager.stop(reset_to_neutral=False)


def test_stage1_preview_timeout_and_command_mutation_invalidate_baseline() -> None:
    """Timeout and any effective command mutation revoke preview ownership."""
    manager, _ = _stage1_manager(MagicMock())
    manager._last_commanded_at = 99.9
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("timeout", begin, 100.0)

    assert manager._stage1_preview_suppresses_publication(100.0 + moves.STAGE1_PREVIEW_SETTLING_TIMEOUT_S + 0.1)
    assert manager._stage1_preview_baseline is None

    manager._stage1_preview_resume_at = None
    manager._last_commanded_at = 199.9
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("mutation", begin, 200.0)
    manager._last_commanded_pose = (
        create_head_pose(yaw=1.0, degrees=True),
        manager._last_commanded_pose[1],
        manager._last_commanded_pose[2],
    )

    assert manager._stage1_preview_suppresses_publication(200.1)
    assert manager._stage1_preview_baseline is None


def test_stage1_unexpected_move_invalidates_preview_and_remains_queued() -> None:
    """A real movement request cancels stabilization without swallowing the move."""
    manager, _ = _stage1_manager(MagicMock())
    manager._last_commanded_at = 99.9
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("owner", begin, 100.0)
    requested_move = MagicMock(spec=moves.Move)
    requested_move.duration = 1.0

    manager._handle_command("queue_move", requested_move, 100.1)

    assert manager._stage1_preview_baseline is None
    assert list(manager.move_queue) == [requested_move]


def test_stage1_preview_release_emits_no_cleanup_command_and_restores_breathing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release uses a grace period, then returns to ordinary breathing eligibility."""
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = ([0.0] * 6, [0.0, 0.0])
    robot.get_current_head_pose.return_value = np.eye(4)
    manager, _ = _stage1_manager(robot)
    manager._last_commanded_at = 99.9
    issue_control_command = MagicMock()
    monkeypatch.setattr(manager, "_issue_control_command", issue_control_command)
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("owner", begin, 100.0)
    release = moves._Stage1PreviewAcknowledgement()

    manager._end_stage1_preview_stabilization_on_worker("owner", "complete", release, 100.1)

    assert release.error is None
    assert manager._stage1_preview_baseline is None
    assert manager._stage1_preview_suppresses_publication(100.39)
    assert not manager._stage1_preview_suppresses_publication(100.41)
    manager._update_primary_motion(100.41)
    assert manager._breathing_active is True
    assert len(manager.move_queue) == 1
    assert isinstance(manager.move_queue[0], moves.BreathingMove)
    issue_control_command.assert_not_called()
    robot.set_target.assert_not_called()


def test_stage1_diagnostic_preview_is_owned_read_only_and_never_reaches_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic lifecycle stops at preview suppression and releases cleanly."""
    robot = MagicMock()
    manager, instrumentation = _stage1_manager(robot)
    calculate_target = MagicMock()
    monkeypatch.setattr(moves, "calculate_horizontal_calibration_target", calculate_target)
    manager.start()
    assert _wait_for(lambda: manager._command_publication_count > 0)

    preview = manager.start_stage1_preview_diagnostic()
    commanded_at = preview["source_command_timestamp"]
    publication_calls_at_start = robot.set_target.call_count
    session_id = preview["session_id"]
    baseline = manager._stage1_preview_baseline
    assert isinstance(session_id, str)
    assert baseline is not None
    assert preview["active"] is True
    assert preview["owner_id"] == session_id
    assert preview["baseline_valid"] is True
    assert preview["preview_publication_count"] == 0
    assert preview["movement_manager_alive"] is True

    with pytest.raises(RuntimeError, match="owned by another session"):
        manager.get_stage1_preview_diagnostic_status("other")
    with pytest.raises(RuntimeError, match="already active"):
        manager.start_stage1_preview_diagnostic()
    with pytest.raises(RuntimeError, match="owned by another session"):
        manager.release_stage1_preview_diagnostic("other")

    manager._now = lambda: baseline.captured_at + moves.STAGE1_PREVIEW_SETTLING_DURATION_S
    stabilized = manager.get_stage1_preview_diagnostic_status(session_id)
    assert stabilized["stabilization_complete"] is True
    assert stabilized["last_commanded_at"] == commanded_at
    assert stabilized["preview_publication_count"] == 0
    assert robot.set_target.call_count == publication_calls_at_start

    released = manager.release_stage1_preview_diagnostic(session_id)
    assert released["active"] is False
    assert released["owner_id"] is None
    assert released["baseline_valid"] is False
    assert released["suppression_active"] is False
    assert released["preview_publication_count"] == 0
    assert robot.set_target.call_count == publication_calls_at_start
    with pytest.raises(RuntimeError, match="stale or inactive"):
        manager.release_stage1_preview_diagnostic(session_id)

    manager.stop(reset_to_neutral=False)
    instrumentation.capture_face_window.assert_not_called()
    instrumentation.read_telemetry.assert_not_called()
    calculate_target.assert_not_called()


def test_stage1_diagnostic_preview_does_not_repair_a_stale_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public diagnostic wrapper preserves admission failures without side effects."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    commanded_at = manager._last_commanded_at
    begin = MagicMock(side_effect=RuntimeError("Stage 1 command baseline is stale"))
    monkeypatch.setattr(manager, "_begin_stage1_preview_stabilization", begin)
    manager._thread = MagicMock()
    manager._thread.is_alive.return_value = True

    with pytest.raises(RuntimeError, match="baseline is stale"):
        manager.start_stage1_preview_diagnostic()

    assert manager._last_commanded_at == commanded_at
    assert manager._stage1_preview_baseline is None
    robot.set_target.assert_not_called()


def test_stage1_diagnostic_preview_does_not_start_the_movement_manager() -> None:
    """Diagnostic admission rejects a stopped manager instead of reviving it."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)

    with pytest.raises(RuntimeError, match="MovementManager is not running"):
        manager.start_stage1_preview_diagnostic()

    assert manager._thread is None
    assert manager._stage1_preview_baseline is None
    robot.set_target.assert_not_called()


def test_stage1_manager_stop_invalidates_diagnostic_preview() -> None:
    """A stopped manager cannot retain or release an earlier preview owner."""
    robot = MagicMock()
    manager, _ = _stage1_manager(robot)
    manager._last_commanded_at = 99.9
    begin = moves._Stage1PreviewAcknowledgement()
    manager._begin_stage1_preview_stabilization_on_worker("owner", begin, 100.0)

    manager.stop(reset_to_neutral=False)

    status = manager.get_stage1_preview_diagnostic_status("owner")
    assert status["active"] is False
    assert status["baseline_invalidation_reason"] == "movement manager stopped"
    with pytest.raises(RuntimeError, match="stale or inactive"):
        manager.release_stage1_preview_diagnostic("owner")
    robot.set_target.assert_not_called()
