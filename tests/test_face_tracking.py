import time
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock

import httpx
import numpy as np
import pytest
from numpy.testing import assert_allclose

from reachy_mini.utils import create_head_pose
from reachy_mini_conversation_app.face_tracking import (
    EulerAngles,
    FaceBBoxSample,
    FaceBBoxWindow,
    PoseValidationError,
    Stage1Instrumentation,
    PostWakeTelemetrySnapshot,
    frame_center,
    angular_delta,
    bbox_center_pixel,
    pose_euler_degrees,
    validate_head_pose,
    compare_face_windows,
    normalized_image_error,
    calculate_face_geometry,
    calculate_incremental_clamp,
    inside_provisional_envelope,
    calculate_horizontal_calibration_target,
)
from reachy_mini_conversation_app.face_identity.types import DetectedFace, FaceLandmarks, QualityResult


class _GeometryRobot:
    def __init__(self, current_pose: np.ndarray, target_pose: np.ndarray) -> None:
        self.current_pose = current_pose
        self.target_pose = target_pose
        self.look_at_calls: list[tuple[int, int, float, bool]] = []
        self.set_target = Mock(side_effect=AssertionError("set_target must not be called"))
        self.goto_target = Mock(side_effect=AssertionError("goto_target must not be called"))
        self.start_head_tracking = Mock(side_effect=AssertionError("start_head_tracking must not be called"))
        self.enable_motors = Mock(side_effect=AssertionError("enable_motors must not be called"))

    def get_current_head_pose(self) -> np.ndarray:
        return self.current_pose

    def look_at_image(
        self,
        u: int,
        v: int,
        duration: float = 1.0,
        perform_movement: bool = True,
    ) -> np.ndarray:
        self.look_at_calls.append((u, v, duration, perform_movement))
        return self.target_pose


def test_bbox_and_frame_centres() -> None:
    """BBox and frame centres use the approved pixel geometry."""
    assert bbox_center_pixel((540.0, 260.0, 200.0, 200.0), 1280, 720) == (640, 360)
    assert frame_center(1280, 720) == (640.0, 360.0)


@pytest.mark.parametrize(
    ("target", "horizontal_sign", "vertical_sign"),
    [
        ((640, 360), 0, 0),
        ((320, 360), -1, 0),
        ((960, 360), 1, 0),
        ((640, 180), 0, -1),
        ((640, 540), 0, 1),
    ],
)
def test_normalized_image_error_signs(target: tuple[int, int], horizontal_sign: int, vertical_sign: int) -> None:
    """Normalized error signs follow image-left/right/high/low."""
    error_x, error_y = normalized_image_error(target, 1280, 720)
    assert np.sign(error_x) == horizontal_sign
    assert np.sign(error_y) == vertical_sign
    if horizontal_sign == 0:
        assert error_x == pytest.approx(0.0)
    if vertical_sign == 0:
        assert error_y == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ((-100.0, -100.0, 20.0, 20.0), (1, 1)),
        ((1270.0, 710.0, 100.0, 100.0), (1279, 719)),
    ],
)
def test_bbox_center_clamps_to_image_interior(
    bbox: tuple[float, float, float, float], expected: tuple[int, int]
) -> None:
    """Target pixels remain inside SDK-valid image coordinates."""
    assert bbox_center_pixel(bbox, 1280, 720) == expected


@pytest.mark.parametrize(
    "pose",
    [
        np.eye(3),
        np.zeros((4, 4)),
        np.diag([1.0, 1.0, -1.0, 1.0]),
    ],
)
def test_malformed_pose_is_rejected(pose: np.ndarray) -> None:
    """Non-rigid or incorrectly shaped poses fail closed."""
    with pytest.raises(PoseValidationError):
        validate_head_pose(pose)


def test_non_finite_pose_is_rejected() -> None:
    """Non-finite pose elements fail closed."""
    pose = np.eye(4)
    pose[0, 0] = np.nan
    with pytest.raises(PoseValidationError, match="finite"):
        validate_head_pose(pose)


def test_valid_pose_is_accepted_as_float64_copy() -> None:
    """Valid transforms are normalized to independent float64 arrays."""
    pose = np.eye(4, dtype=np.float32)
    validated = validate_head_pose(pose)
    assert validated.shape == (4, 4)
    assert validated.dtype == np.float64
    assert validated is not pose


def test_small_physical_pose_drift_is_accepted_without_modification() -> None:
    """Small physical FK drift is accepted without repairing the matrix."""
    pose = np.eye(4)
    pose[:3, :3] *= 1.0002
    validated = validate_head_pose(pose)
    assert_allclose(validated, pose)


def test_euler_extraction_matches_reachy_xyz_convention() -> None:
    """Euler extraction matches Reachy's XYZ roll-pitch-yaw convention."""
    pose = create_head_pose(roll=3.0, pitch=-7.0, yaw=11.0, degrees=True)
    angles = pose_euler_degrees(pose)
    assert angles.roll == pytest.approx(3.0)
    assert angles.pitch == pytest.approx(-7.0)
    assert angles.yaw == pytest.approx(11.0)


@pytest.mark.parametrize("roll", [20.0, -20.0, 45.0])
def test_euler_extraction_reports_known_roll_angles(roll: float) -> None:
    """Known roll-only matrices retain magnitude and sign."""
    angles = pose_euler_degrees(create_head_pose(roll=roll, degrees=True))
    assert angles.roll == pytest.approx(roll)
    assert angles.pitch == pytest.approx(0.0, abs=1e-12)
    assert angles.yaw == pytest.approx(0.0, abs=1e-12)


def test_euler_extraction_projects_accepted_imperfect_rotation_without_angle_distortion() -> None:
    """Small accepted FK scale noise cannot suppress a large roll."""
    pose = create_head_pose(roll=45.0, pitch=-8.0, yaw=12.0, degrees=True)
    pose[:3, :3] *= 1.0002

    angles = pose_euler_degrees(pose)

    assert angles.roll == pytest.approx(45.0, abs=1e-9)
    assert angles.pitch == pytest.approx(-8.0, abs=1e-9)
    assert angles.yaw == pytest.approx(12.0, abs=1e-9)


def test_angular_delta_wraps_to_shortest_direction() -> None:
    """Angular differences cross the wrap boundary by the shortest path."""
    delta = angular_delta(EulerAngles(0.0, 0.0, 179.0), EulerAngles(0.0, 0.0, -179.0))
    assert delta.yaw == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        (EulerAngles(0.0, 10.0, 15.0), True),
        (EulerAngles(0.0, 10.1, 0.0), False),
        (EulerAngles(0.0, 0.0, -15.1), False),
    ],
)
def test_provisional_envelope(angles: EulerAngles, expected: bool) -> None:
    """Provisional yaw and pitch envelope boundaries are diagnostic."""
    assert inside_provisional_envelope(angles) is expected


def test_incremental_clamp_limits_yaw_and_pitch() -> None:
    """Provisional per-update yaw and pitch limits are calculated."""
    result = calculate_incremental_clamp(
        EulerAngles(roll=0.0, pitch=1.0, yaw=-2.0),
        EulerAngles(roll=0.0, pitch=5.0, yaw=4.0),
    )
    assert result.requested_yaw == pytest.approx(6.0)
    assert result.requested_pitch == pytest.approx(4.0)
    assert result.limited_yaw == pytest.approx(1.5)
    assert result.limited_pitch == pytest.approx(1.0)
    assert result.yaw_was_limited is True
    assert result.pitch_was_limited is True


def test_geometry_is_calculation_only_and_has_no_movement_authority() -> None:
    """Geometry calls only the SDK calculation-only path."""
    current_pose = create_head_pose(roll=1.0, pitch=2.0, yaw=3.0, degrees=True)
    target_pose = create_head_pose(roll=1.0, pitch=4.0, yaw=8.0, degrees=True)
    robot = _GeometryRobot(current_pose, target_pose)

    result = calculate_face_geometry(robot, (540.0, 260.0, 200.0, 200.0), 1280, 720)

    assert robot.look_at_calls == [(640, 360, 0.0, False)]
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
    robot.start_head_tracking.assert_not_called()
    robot.enable_motors.assert_not_called()
    assert result.body_yaw_authorized is False
    assert result.normalized_error == pytest.approx((0.0, 0.0))
    assert result.delta_angles.yaw == pytest.approx(5.0)
    assert result.delta_angles.pitch == pytest.approx(2.0)
    assert result.incremental_clamp.limited_yaw == pytest.approx(1.5)
    assert result.incremental_clamp.limited_pitch == pytest.approx(1.0)
    assert_allclose(result.target_pose, target_pose)


def test_geometry_rejects_malformed_sdk_result() -> None:
    """Malformed SDK geometry cannot escape validation."""
    robot = _GeometryRobot(np.eye(4), np.eye(3))
    with pytest.raises(PoseValidationError):
        calculate_face_geometry(robot, (10.0, 20.0, 30.0, 40.0), 1280, 720)


@pytest.mark.parametrize(("raw_delta", "expected"), [(24.0, 1.5), (-28.0, -1.5)])
def test_horizontal_calibration_clamps_and_preserves_sign(raw_delta: float, expected: float) -> None:
    """Stage 1 keeps direction while limiting yaw to 1.5 degrees."""
    current = create_head_pose(yaw=-1.3, degrees=True)
    commanded = create_head_pose(x=0.002, y=-0.001, z=0.003, roll=2.0, pitch=3.0, yaw=1.0, degrees=True)
    target = calculate_horizontal_calibration_target(current, commanded, raw_delta)
    assert target.requested_calibration_delta_yaw == pytest.approx(expected)
    assert target.clamped_delta_yaw == pytest.approx(expected)
    assert target.target_angles.pitch == pytest.approx(target.commanded_angles.pitch)
    assert target.target_angles.roll == pytest.approx(target.commanded_angles.roll)
    assert target.target_angles.yaw == pytest.approx(target.commanded_angles.yaw + expected)
    assert_allclose(target.target_pose[:3, 3], commanded[:3, 3])
    assert target.within_stage1_limit is True


def test_horizontal_calibration_clamps_to_app_envelope() -> None:
    """The provisional absolute envelope can further reduce the tiny delta."""
    current = create_head_pose(yaw=14.5, degrees=True)
    target = calculate_horizontal_calibration_target(np.eye(4), current, 20.0)
    assert target.target_angles.yaw == pytest.approx(15.0)
    assert target.clamped_delta_yaw == pytest.approx(0.5)


@pytest.mark.parametrize("raw_delta", [float("nan"), float("inf"), 0.0])
def test_horizontal_calibration_rejects_invalid_delta(raw_delta: float) -> None:
    """Non-finite and directionless requests fail closed."""
    with pytest.raises(ValueError):
        calculate_horizontal_calibration_target(np.eye(4), np.eye(4), raw_delta)


def test_horizontal_calibration_rejects_non_finite_pose() -> None:
    """A non-finite head target source cannot reach calibration."""
    current = np.eye(4)
    current[0, 3] = np.inf
    with pytest.raises(PoseValidationError, match="finite"):
        calculate_horizontal_calibration_target(np.eye(4), current, 1.0)


def test_horizontal_calibration_rejects_pose_outside_envelope() -> None:
    """Stage 1 cannot start from outside its uncalibrated safety envelope."""
    with pytest.raises(PoseValidationError, match="outside"):
        calculate_horizontal_calibration_target(np.eye(4), create_head_pose(pitch=10.1, degrees=True), 1.0)


def _detected_face(u: float = 640.0, v: float = 360.0) -> DetectedFace:
    return DetectedFace(
        bbox=(u - 100.0, v - 100.0, 200.0, 200.0),
        confidence=0.95,
        landmarks=FaceLandmarks(
            right_eye=(u - 30.0, v - 20.0),
            left_eye=(u + 30.0, v - 20.0),
            nose=(u, v),
            right_mouth=(u - 25.0, v + 35.0),
            left_mouth=(u + 25.0, v + 35.0),
        ),
        frame_width=1280,
        frame_height=720,
    )


def test_stage1_face_window_retains_metadata_only() -> None:
    """Sampling uses YuNet and quality gating but returns no frame or biometric data."""
    robot = Mock()
    robot.media.get_frame.side_effect = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(5)]
    detector = Mock()
    detector.detect.return_value = [_detected_face()]
    instrumentation = Stage1Instrumentation(
        robot,
        detector=detector,
        quality_assessor=lambda _frame, _face: QualityResult(usable=True, score=1.0),
    )

    window = instrumentation.capture_face_window(attempts=5, interval_s=0.0)

    assert window.stable is True
    assert len(window.samples) == 5
    assert window.median_u == pytest.approx(640.0)
    assert window.frame_width == 1280
    assert window.frame_height == 720
    assert window.to_dict().keys().isdisjoint({"frame", "crop", "embedding", "identity"})
    robot.set_target.assert_not_called()


@pytest.mark.parametrize("detections", [[], [_detected_face(), _detected_face(700.0)]])
def test_stage1_face_window_rejects_non_single_face(detections: list[DetectedFace]) -> None:
    """No-face and multiple-face windows fail the movement precondition."""
    robot = Mock()
    robot.media.get_frame.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = Mock()
    detector.detect.return_value = detections
    instrumentation = Stage1Instrumentation(robot, detector=detector)

    window = instrumentation.capture_face_window(attempts=5, interval_s=0.0)

    assert window.stable is False
    assert window.invalid_attempts == 5


def test_stage1_face_window_rejects_quality_failure() -> None:
    """A quality-rejected face invalidates the metadata-only window."""
    robot = Mock()
    robot.media.get_frame.return_value = np.zeros((720, 1280, 3), dtype=np.uint8)
    detector = Mock()
    detector.detect.return_value = [_detected_face()]
    instrumentation = Stage1Instrumentation(
        robot,
        detector=detector,
        quality_assessor=lambda _frame, _face: QualityResult(usable=False, score=0.49),
    )

    window = instrumentation.capture_face_window(attempts=5, interval_s=0.0)

    assert window.stable is False
    assert window.samples == ()
    assert window.invalid_attempts == 5
    robot.set_target.assert_not_called()


def test_left_camera_displacement_must_exceed_pre_jitter() -> None:
    """A left turn requires positive image displacement beyond twice observed jitter."""
    pre_samples = tuple(FaceBBoxSample(float(index), 640.0 + (-1.0) ** index, 360.0, 0.95) for index in range(6))
    post_samples = tuple(FaceBBoxSample(float(index), 650.0, 360.0, 0.95) for index in range(6))
    pre = FaceBBoxWindow(pre_samples, 6, 0, 640.0, 360.0, 1.0, 0.0, True)
    post = FaceBBoxWindow(post_samples, 6, 0, 650.0, 360.0, 0.0, 0.0, True)

    displacement = compare_face_windows(pre, post)

    assert displacement.delta_u == pytest.approx(10.0)
    assert displacement.horizontal_jitter_ratio == pytest.approx(10.0)
    assert displacement.exceeds_horizontal_jitter is True
    assert displacement.left_motion_consistent is True


def _telemetry_payload(timestamp: str | None = None) -> dict[str, object]:
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "head_pose": {"m": np.eye(4).reshape(-1).tolist()},
        "head_joints": [0.0] * 7,
        "body_yaw": 0.0,
        "antennas_position": [0.1, -0.1],
        "control_mode": "enabled",
        "timestamp": timestamp,
    }


def _post_wake_state_payload() -> dict[str, object]:
    return {
        "head_pose": {"m": np.eye(4).reshape(-1).tolist()},
        "head_joints": [0.0] * 7,
        "body_yaw": 0.0,
        "antennas_position": [-0.1745, 0.1745],
        "control_mode": "enabled",
        "timestamp": "1970-01-01T00:01:40Z",
    }


def _post_wake_status_payload(
    *,
    last_alive: float = 99.99,
    tracking_enabled: bool = False,
    frequency_hz: float = 49.5,
    motor_controller_period_ms: float | None = None,
    ready: bool = True,
) -> dict[str, object]:
    control_loop_stats: dict[str, object] = {"mean_control_loop_frequency": frequency_hz}
    if motor_controller_period_ms is not None:
        control_loop_stats["motor_controller"] = (
            f"ControlLoopStats(period=~{motor_controller_period_ms:.2f}ms, "
            "read_dt=~2.68 ms, write_dt=~0.25 ms)"
        )
    return {
        "state": "running",
        "error": None,
        "head_tracking_enabled": tracking_enabled,
        "backend_status": {
            "ready": ready,
            "last_alive": last_alive,
            "error": None,
            "control_loop_stats": control_loop_stats,
        },
    }


def _response(payload: object) -> Mock:
    response = Mock()
    response.json.return_value = payload
    return response


def _mock_telemetry_client(monkeypatch: pytest.MonkeyPatch, side_effect: object) -> MagicMock:
    client = MagicMock()
    client.__enter__.return_value = client
    client.get.side_effect = side_effect if isinstance(side_effect, list) else None
    if not isinstance(side_effect, list):
        client.get.return_value = side_effect
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.httpx.Client", Mock(return_value=client))
    return client


class _ManualClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.current += seconds


def _telemetry_instrumentation() -> tuple[Stage1Instrumentation, Mock]:
    robot = Mock()
    robot.client.host = "robot.local"
    robot.client.port = 8000
    return Stage1Instrumentation(robot), robot


def _stage1_responses(
    *samples: tuple[str, float],
    tracking_enabled: bool = False,
    frequency_hz: float = 49.5,
    motor_controller_period_ms: float | None = None,
    moves: list[object] | None = None,
) -> list[Mock]:
    responses: list[Mock] = []
    for timestamp, last_alive in samples:
        responses.extend(
            [
                _response(_telemetry_payload(timestamp)),
                _response(
                    _post_wake_status_payload(
                        last_alive=last_alive,
                        tracking_enabled=tracking_enabled,
                        frequency_hz=frequency_hz,
                        motor_controller_period_ms=motor_controller_period_ms,
                    )
                ),
                _response([] if moves is None else moves),
            ]
        )
    return responses


def _physical_snapshot(
    timestamp: str,
    last_alive: float,
    *,
    acquisition_elapsed_s: float = 0.01,
    liveness_age_s: float = 0.01,
    tracking_enabled: bool = False,
    frequency_hz: float = 49.5,
    active_move_count: int = 0,
) -> PostWakeTelemetrySnapshot:
    return PostWakeTelemetrySnapshot(
        monotonic_time=0.0,
        present_head_pose=np.eye(4),
        present_head_joints=(0.0,) * 7,
        present_body_yaw=0.0,
        present_antennas=(0.1, -0.1),
        control_mode="enabled",
        daemon_timestamp=timestamp,
        daemon_ready=True,
        daemon_error=None,
        backend_last_alive=last_alive,
        backend_last_alive_age_s=liveness_age_s,
        control_loop_frequency_hz=frequency_hz,
        active_move_count=active_move_count,
        acquisition_elapsed_s=acquisition_elapsed_s,
        head_tracking_enabled=tracking_enabled,
    )


def test_stage1_telemetry_capture_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete advancing state/status/movement acquisition is read-only."""
    client = _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(
            ("1970-01-01T00:01:40.000Z", 99.99),
            ("1970-01-01T00:01:40.020Z", 100.01),
        ),
    )
    instrumentation, robot = _telemetry_instrumentation()

    snapshot = instrumentation.read_telemetry()

    assert client.get.call_count == 6
    assert snapshot.acquisition_attempts == 2
    assert snapshot.backend_last_alive_age_s == pytest.approx(0.01)
    assert snapshot.target_head_pose is None
    assert snapshot.target_head_joints is None
    robot.set_target.assert_not_called()


def test_stage1_telemetry_retries_one_timeout_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """One transient read timeout gets one bounded read-only retry."""
    client = _mock_telemetry_client(
        monkeypatch,
        [
            httpx.ReadTimeout("slow read"),
            *_stage1_responses(
                ("1970-01-01T00:01:40.000Z", 99.99),
                ("1970-01-01T00:01:40.020Z", 100.01),
            ),
        ],
    )
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", Mock())
    instrumentation, robot = _telemetry_instrumentation()

    snapshot = instrumentation.read_telemetry()

    assert snapshot.acquisition_attempts == 3
    assert client.get.call_count == 7
    robot.set_target.assert_not_called()


def test_stage1_telemetry_enforces_maximum_attempts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two timeouts fail closed without an unbounded third attempt."""
    client = _mock_telemetry_client(
        monkeypatch,
        [httpx.ReadTimeout("slow read"), httpx.ReadTimeout("slow read")],
    )
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", Mock())
    instrumentation, robot = _telemetry_instrumentation()
    caplog.set_level("WARNING")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        instrumentation.read_telemetry()

    assert client.get.call_count == 2
    robot.set_target.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"timestamp": "2026-09-12T11:00:00Z"},
        {**_telemetry_payload(timestamp="2026-09-12T11:00:00Z"), "head_joints": [0.0] * 6},
        {**_telemetry_payload(timestamp="2026-09-12T11:00:00Z"), "timestamp": "not-a-time"},
    ],
)
def test_stage1_telemetry_rejects_partial_or_malformed_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    """Malformed and partial snapshots fail immediately without retry or movement."""
    client = _mock_telemetry_client(
        monkeypatch,
        [
            _response(payload),
            _response(_post_wake_status_payload()),
            _response([]),
        ],
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(ValueError):
        instrumentation.read_telemetry()

    assert client.get.call_count == 3
    robot.set_target.assert_not_called()


def test_post_wake_telemetry_reads_state_health_and_moves_without_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The convergence snapshot uses only bounded read-only daemon endpoints."""
    responses = [
        _response(_post_wake_state_payload()),
        _response(_post_wake_status_payload()),
        _response([]),
    ]
    client = _mock_telemetry_client(monkeypatch, responses)
    instrumentation, robot = _telemetry_instrumentation()

    snapshot = instrumentation.read_post_wake_telemetry(deadline=time.monotonic() + 1.0)

    assert [entry.args[0] for entry in client.get.call_args_list] == [
        "http://robot.local:8000/api/state/full",
        "http://robot.local:8000/api/daemon/status",
        "http://robot.local:8000/api/move/running",
    ]
    assert snapshot.daemon_ready is True
    assert snapshot.present_head_joints == (0.0,) * 7
    assert snapshot.backend_last_alive_age_s == pytest.approx(0.01)
    assert snapshot.control_loop_frequency_hz == pytest.approx(49.5)
    assert snapshot.active_move_count == 0
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_transport_preparation_is_non_authorizing_and_uses_authorizing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slow transport setup does not count as an authorizing telemetry snapshot."""
    clock = _ManualClock()
    client = MagicMock()
    calls: list[str] = []

    def get(url: str, **_kwargs: object) -> Mock:
        calls.append(url)
        if url.endswith("/api/daemon/status") and len(calls) == 1:
            clock.current += 2.1
            return _response(_post_wake_status_payload())
        clock.current += 0.025
        if url.endswith("/api/state/full"):
            return _response(_post_wake_state_payload())
        if url.endswith("/api/daemon/status"):
            return _response(_post_wake_status_payload())
        return _response([])

    client.get.side_effect = get
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.httpx.Client", Mock(return_value=client))
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    instrumentation, robot = _telemetry_instrumentation()

    preparation = instrumentation.prepare_telemetry_transport()
    snapshot = instrumentation.read_post_wake_telemetry(deadline=clock.monotonic() + 1.0)

    assert preparation.duration_s == pytest.approx(2.1)
    assert snapshot.acquisition_elapsed_s == pytest.approx(0.075)
    assert snapshot.transport_preparation is preparation
    assert snapshot.endpoint_timings is not None
    assert snapshot.endpoint_timings.state_s == pytest.approx(0.025)
    assert snapshot.endpoint_timings.status_s == pytest.approx(0.025)
    assert snapshot.endpoint_timings.movement_s == pytest.approx(0.025)
    assert snapshot.endpoint_timings.total_s == pytest.approx(0.075)
    assert client.get.call_count == 4
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_transport_preparation_timeout_fails_before_authoritative_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed non-authorizing preparation cannot fall through to telemetry adoption."""
    client = _mock_telemetry_client(monkeypatch, [httpx.ReadTimeout("connect")])
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="preparation timed out"):
        instrumentation.prepare_telemetry_transport()

    assert client.get.call_count == 1
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_preparation_response_is_not_authoritative_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The warm-up status response is never reused as the physical state sample."""
    client = _mock_telemetry_client(
        monkeypatch,
        [
            _response(_post_wake_status_payload()),
            _response(_post_wake_state_payload()),
            _response(_post_wake_status_payload()),
            _response([]),
        ],
    )
    instrumentation, robot = _telemetry_instrumentation()

    instrumentation.prepare_telemetry_transport()
    snapshot = instrumentation.read_post_wake_telemetry(deadline=time.monotonic() + 1.0)

    assert [entry.args[0] for entry in client.get.call_args_list] == [
        "http://robot.local:8000/api/daemon/status",
        "http://robot.local:8000/api/state/full",
        "http://robot.local:8000/api/daemon/status",
        "http://robot.local:8000/api/move/running",
    ]
    assert snapshot.present_head_pose.shape == (4, 4)
    robot.set_target.assert_not_called()


def test_authoritative_acquisition_after_preparation_still_fails_above_250ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful warm-up never relaxes the authoritative acquisition budget."""
    clock = _ManualClock()
    client = MagicMock()
    responses = iter(
        [
            _response(_post_wake_status_payload()),
            _response(_post_wake_state_payload()),
            _response(_post_wake_status_payload()),
            _response([]),
        ]
    )

    def delayed_get(*_args: object, **_kwargs: object) -> Mock:
        if client.get.call_count > 1:
            clock.current += 0.09
        return next(responses)

    client.get.side_effect = delayed_get
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.httpx.Client", Mock(return_value=client))
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    instrumentation, robot = _telemetry_instrumentation()

    instrumentation.prepare_telemetry_transport()
    with pytest.raises(RuntimeError, match="freshness bound"):
        instrumentation.read_post_wake_telemetry(deadline=clock.monotonic() + 1.0)

    assert client.get.call_count == 4
    robot.goto_target.assert_not_called()


def _malformed_post_wake_states() -> list[dict[str, object]]:
    nan_pose = np.eye(4)
    nan_pose[0, 0] = np.nan
    infinity_pose = np.eye(4)
    infinity_pose[1, 1] = np.inf
    bad_homogeneous_pose = np.eye(4)
    bad_homogeneous_pose[3, 3] = 0.0
    invalid_rotation_pose = np.eye(4)
    invalid_rotation_pose[0, 0] = 2.0
    return [
        {**_post_wake_state_payload(), "head_pose": {"m": nan_pose.reshape(-1).tolist()}},
        {**_post_wake_state_payload(), "head_pose": {"m": infinity_pose.reshape(-1).tolist()}},
        {**_post_wake_state_payload(), "head_pose": {"m": np.eye(4).reshape(-1).tolist()[:-1]}},
        {**_post_wake_state_payload(), "head_pose": {"m": bad_homogeneous_pose.reshape(-1).tolist()}},
        {**_post_wake_state_payload(), "head_pose": {"m": invalid_rotation_pose.reshape(-1).tolist()}},
        {**_post_wake_state_payload(), "head_joints": [0.0] * 6},
        {**_post_wake_state_payload(), "body_yaw": np.inf},
        {**_post_wake_state_payload(), "antennas_position": [np.nan, 0.1745]},
    ]


@pytest.mark.parametrize("state_payload", _malformed_post_wake_states())
def test_post_wake_telemetry_rejects_malformed_physical_state(
    monkeypatch: pytest.MonkeyPatch,
    state_payload: dict[str, object],
) -> None:
    """Malformed physical values never become convergence evidence."""
    responses = [
        _response(state_payload),
        _response(_post_wake_status_payload()),
        _response([]),
    ]
    _mock_telemetry_client(monkeypatch, responses)
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(ValueError):
        instrumentation.read_post_wake_telemetry(deadline=time.monotonic() + 1.0)

    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_stage1_telemetry_rejects_non_advancing_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A daemon timestamp reused after an accepted snapshot is stale."""
    clock = _ManualClock()
    samples = [("1970-01-01T00:01:40.000Z", 99.99)] * 4
    client = _mock_telemetry_client(monkeypatch, _stage1_responses(*samples))
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.STAGE1_TELEMETRY_FRESH_SAMPLE_TIMEOUT_S", 0.10)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.STAGE1_TELEMETRY_FRESH_SAMPLE_POLL_S", 0.05)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", clock.sleep)
    instrumentation, robot = _telemetry_instrumentation()
    caplog.set_level("WARNING")

    with pytest.raises(RuntimeError, match="freshness bound"):
        instrumentation.read_telemetry()

    assert client.get.call_count >= 6
    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_complete_acquisition_over_250ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complete three-resource acquisition fails closed above 250 ms."""
    clock = _ManualClock()
    responses = iter(_stage1_responses(("1970-01-01T00:01:40.000Z", 99.99)))
    client = MagicMock()

    def delayed_get(*_args: object, **_kwargs: object) -> Mock:
        clock.current += 0.09
        return next(responses)

    client.get.side_effect = delayed_get
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.httpx.Client", Mock(return_value=client))
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="freshness bound"):
        instrumentation.read_telemetry()

    assert client.get.call_count == 3
    robot.set_target.assert_not_called()


def test_stage1_telemetry_accepts_exact_acquisition_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing inclusive 250 ms freshness boundary remains accepted."""
    instrumentation, robot = _telemetry_instrumentation()
    snapshots = iter(
        [
            _physical_snapshot(
                "1970-01-01T00:01:40.000Z",
                99.75,
                acquisition_elapsed_s=0.25,
                liveness_age_s=0.25,
            ),
            _physical_snapshot(
                "1970-01-01T00:01:40.020Z",
                99.77,
                acquisition_elapsed_s=0.25,
                liveness_age_s=0.25,
            ),
        ]
    )
    monkeypatch.setattr(instrumentation, "read_post_wake_telemetry", lambda *, deadline: next(snapshots))

    snapshot = instrumentation.read_telemetry()

    assert snapshot.acquisition_elapsed_s == pytest.approx(0.25)
    assert snapshot.backend_last_alive_age_s == pytest.approx(0.25)
    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_aged_daemon_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-daemon-clock backend liveness above 250 ms fails closed."""
    client = _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(("1970-01-01T00:01:40.000Z", 99.70)),
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="backend liveness"):
        instrumentation.read_telemetry()

    assert client.get.call_count == 3
    robot.set_target.assert_not_called()


def test_stage1_telemetry_waits_for_genuinely_fresh_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both daemon timestamp and backend liveness must strictly advance."""
    clock = _ManualClock()
    client = _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(
            ("1970-01-01T00:01:40.000Z", 99.99),
            ("1970-01-01T00:01:40.020Z", 99.99),
            ("1970-01-01T00:01:40.040Z", 100.03),
        ),
    )
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", clock.sleep)
    instrumentation, robot = _telemetry_instrumentation()

    snapshot = instrumentation.read_telemetry()

    assert snapshot.daemon_timestamp == "1970-01-01T00:01:40.040Z"
    assert snapshot.acquisition_attempts == 3
    assert clock.sleeps == [pytest.approx(0.05), pytest.approx(0.05)]
    assert client.get.call_count == 9
    robot.set_target.assert_not_called()


@pytest.mark.parametrize("wall_offset_s", [0.6, -0.6, 0.5883])
def test_stage1_telemetry_ignores_cross_host_wall_clock_offset(
    monkeypatch: pytest.MonkeyPatch,
    wall_offset_s: float,
) -> None:
    """Large apparent wall ages do not reject clock-independent fresh telemetry."""
    client = _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(
            ("1970-01-01T00:01:40.000Z", 99.99),
            ("1970-01-01T00:01:40.020Z", 100.01),
        ),
    )
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.time", lambda: 100.02 + wall_offset_s)
    instrumentation, robot = _telemetry_instrumentation()

    snapshot = instrumentation.read_telemetry()

    assert snapshot.backend_last_alive_age_s == pytest.approx(0.01)
    assert client.get.call_count == 6
    robot.set_target.assert_not_called()


def test_stage1_telemetry_ignores_wall_clock_jump(monkeypatch: pytest.MonkeyPatch) -> None:
    """An arbitrary local wall-clock jump cannot affect authorization."""
    client = _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(
            ("1970-01-01T00:01:40.000Z", 99.99),
            ("1970-01-01T00:01:40.020Z", 100.01),
        ),
    )
    wall_clock = Mock(side_effect=[-10_000.0, 10_000.0])
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.time", wall_clock)
    instrumentation, robot = _telemetry_instrumentation()

    first = instrumentation.read_telemetry()
    client.get.side_effect = _stage1_responses(("1970-01-01T00:01:40.040Z", 100.03))
    second = instrumentation.read_telemetry()

    assert first.timestamp == -10_000.0
    assert second.timestamp == 10_000.0
    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_nonadvancing_backend_liveness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advancing state alone cannot authorize a stale control backend."""
    clock = _ManualClock()
    samples = [
        ("1970-01-01T00:01:40.000Z", 99.99),
        ("1970-01-01T00:01:40.020Z", 99.99),
        ("1970-01-01T00:01:40.040Z", 99.99),
    ]
    _mock_telemetry_client(monkeypatch, _stage1_responses(*samples))
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.STAGE1_TELEMETRY_FRESH_SAMPLE_TIMEOUT_S", 0.10)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", clock.sleep)
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="freshness bound"):
        instrumentation.read_telemetry()

    robot.set_target.assert_not_called()


@pytest.mark.parametrize("last_alive", [None, float("nan"), float("inf")])
def test_stage1_telemetry_rejects_malformed_backend_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    last_alive: float | None,
) -> None:
    """Missing and nonfinite backend timestamps fail closed."""
    status = _post_wake_status_payload()
    backend = status["backend_status"]
    assert isinstance(backend, dict)
    backend["last_alive"] = last_alive
    _mock_telemetry_client(
        monkeypatch,
        [
            _response(_telemetry_payload("1970-01-01T00:01:40Z")),
            _response(status),
            _response([]),
        ],
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(ValueError, match="last_alive"):
        instrumentation.read_telemetry()

    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_unhealthy_control_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sub-threshold daemon control loop cannot authorize physical telemetry."""
    _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(("1970-01-01T00:01:40Z", 99.99), frequency_hz=39.9),
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="frequency is unhealthy"):
        instrumentation.read_telemetry()

    robot.set_target.assert_not_called()


def test_stage1_telemetry_accepts_daemon_motor_controller_period_with_observed_backend_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon motor-loop period is authoritative when the status payload exposes it."""
    _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(
            ("1970-01-01T00:01:40.000Z", 99.99),
            ("1970-01-01T00:01:40.020Z", 100.01),
            frequency_hz=19.39,
            motor_controller_period_ms=20.01,
        ),
    )
    instrumentation, robot = _telemetry_instrumentation()

    snapshot = instrumentation.read_telemetry()

    assert snapshot.control_loop_health_source == "motor_controller_period"
    assert snapshot.control_loop_frequency_hz == pytest.approx(49.975, rel=1e-3)
    assert snapshot.observed_control_loop_frequency_hz == pytest.approx(19.39)
    assert snapshot.motor_controller_period_s == pytest.approx(0.02001)
    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_genuinely_degraded_motor_controller_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow daemon motor-loop period still fails closed even when the mean field is healthy."""
    _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(("1970-01-01T00:01:40Z", 99.99), frequency_hz=50.0, motor_controller_period_ms=60.0),
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="frequency is unhealthy"):
        instrumentation.read_telemetry()

    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_missing_period_when_mean_frequency_is_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without motor-loop period evidence, low mean frequency remains fail-closed."""
    _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(("1970-01-01T00:01:40Z", 99.99), frequency_hz=19.39),
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="frequency is unhealthy"):
        instrumentation.read_telemetry()

    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_active_move_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """An active daemon move conflicts with Stage 1 authorization."""
    _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(("1970-01-01T00:01:40Z", 99.99), moves=[{"id": "active"}]),
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="active daemon movement"):
        instrumentation.read_telemetry()

    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_unsafe_authoritative_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each caller's required authoritative tracking state is enforced."""
    _mock_telemetry_client(
        monkeypatch,
        _stage1_responses(("1970-01-01T00:01:40Z", 99.99), tracking_enabled=True),
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="tracking state is unsafe"):
        instrumentation.read_telemetry(expected_tracking_enabled=False)

    robot.set_target.assert_not_called()


def test_stage1_telemetry_requires_new_timestamp_after_stale_sample(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reread of the same daemon timestamp cannot become the accepted fresh sample."""
    clock = _ManualClock()
    samples = [
        ("1970-01-01T00:01:40.000Z", 99.99),
        ("1970-01-01T00:01:40.000Z", 100.01),
        ("1970-01-01T00:01:40.000Z", 100.03),
    ]
    client = _mock_telemetry_client(monkeypatch, _stage1_responses(*samples))
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.STAGE1_TELEMETRY_FRESH_SAMPLE_TIMEOUT_S", 0.10)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.STAGE1_TELEMETRY_FRESH_SAMPLE_POLL_S", 0.05)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.monotonic", clock.monotonic)
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", clock.sleep)
    instrumentation, robot = _telemetry_instrumentation()
    caplog.set_level("WARNING")

    with pytest.raises(RuntimeError, match="freshness bound"):
        instrumentation.read_telemetry()

    assert client.get.call_count >= 6
    robot.set_target.assert_not_called()


def test_stage1_telemetry_rejects_non_finite_body_yaw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-finite joint-space values fail closed without movement."""
    client = _mock_telemetry_client(
        monkeypatch,
        [
            _response({**_telemetry_payload("1970-01-01T00:01:40Z"), "body_yaw": float("nan")}),
            _response(_post_wake_status_payload()),
            _response([]),
        ],
    )
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(ValueError, match="body_yaw"):
        instrumentation.read_telemetry()

    assert client.get.call_count == 3
    robot.set_target.assert_not_called()


def test_stage1_telemetry_enforces_total_deadline_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport timeout cannot start an unbounded retry sequence."""
    client = _mock_telemetry_client(
        monkeypatch,
        [httpx.ReadTimeout("slow read"), httpx.ReadTimeout("slow read")],
    )
    monkeypatch.setattr("reachy_mini_conversation_app.face_tracking.time.sleep", Mock())
    instrumentation, robot = _telemetry_instrumentation()

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        instrumentation.read_telemetry()

    assert client.get.call_count == 2
    robot.set_target.assert_not_called()
