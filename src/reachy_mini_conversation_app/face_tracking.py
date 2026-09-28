"""No-motion face-to-head geometry for staged physical validation."""

import re
import time
import logging
from typing import Callable, Protocol, TypeAlias
from datetime import datetime
from dataclasses import dataclass

import httpx
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from reachy_mini_conversation_app.face_identity.types import DetectedFace, QualityResult
from reachy_mini_conversation_app.face_identity.quality import assess_face_quality
from reachy_mini_conversation_app.face_identity.detector import YuNetDetector


logger = logging.getLogger(__name__)

FaceBBox: TypeAlias = tuple[float, float, float, float]

PROVISIONAL_YAW_ENVELOPE_DEG = 15.0
PROVISIONAL_PITCH_ENVELOPE_DEG = 10.0
PROVISIONAL_YAW_INCREMENT_DEG = 1.5
PROVISIONAL_PITCH_INCREMENT_DEG = 1.0
_ROTATION_ATOL = 1e-3
STAGE1_TELEMETRY_CONNECT_TIMEOUT_S = 1.0
STAGE1_TELEMETRY_READ_TIMEOUT_S = 3.0
STAGE1_TELEMETRY_WRITE_TIMEOUT_S = 0.25
STAGE1_TELEMETRY_POOL_TIMEOUT_S = 0.25
STAGE1_TELEMETRY_MAX_ATTEMPTS = 2
STAGE1_TELEMETRY_RETRY_BACKOFF_S = 0.15
# Fits two wireless read attempts plus one backoff under intermittent LAN spikes.
STAGE1_TELEMETRY_TOTAL_DEADLINE_S = 7.0
# Maximum complete local acquisition time and same-daemon-clock backend age.
STAGE1_TELEMETRY_FRESHNESS_S = 0.25
STAGE1_TELEMETRY_TRANSPORT_PREP_TIMEOUT_S = 3.0
STAGE1_TELEMETRY_FRESH_SAMPLE_TIMEOUT_S = 1.0
STAGE1_TELEMETRY_FRESH_SAMPLE_POLL_S = 0.05
POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ = 40.0
_MOTOR_CONTROLLER_PERIOD_RE = re.compile(r"period=~?\s*([0-9]+(?:\.[0-9]+)?)\s*ms")


class NoMotionLookAt(Protocol):
    """Read pose state and calculate image look-at geometry without moving."""

    def get_current_head_pose(self) -> npt.NDArray[np.float64]:
        """Return the current absolute head pose."""
        ...

    def look_at_image(
        self,
        u: int,
        v: int,
        duration: float = 1.0,
        perform_movement: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Calculate an absolute image look-at pose."""
        ...


class Stage1Media(Protocol):
    """Provide fresh SDK camera frames without creating another connection."""

    def get_frame(self) -> npt.NDArray[np.uint8] | None:
        """Return the latest BGR camera frame."""
        ...


class Stage1Client(Protocol):
    """Expose the daemon address already owned by the SDK client."""

    host: str
    port: int


class Stage1Robot(NoMotionLookAt, Protocol):
    """Read camera and daemon state through the existing Reachy connection."""

    media: Stage1Media
    client: Stage1Client


class PoseValidationError(ValueError):
    """Raised when a head pose is not a finite rigid transform."""


@dataclass(frozen=True)
class EulerAngles:
    """XYZ Euler angles in degrees, matching Reachy Mini pose conventions."""

    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class IncrementalClamp:
    """Diagnostic angular deltas before and after provisional limiting."""

    requested_yaw: float
    requested_pitch: float
    limited_yaw: float
    limited_pitch: float
    yaw_was_limited: bool
    pitch_was_limited: bool


@dataclass(frozen=True)
class FaceGeometryDiagnostic:
    """Complete no-motion geometry result for one face observation."""

    frame_width: int
    frame_height: int
    bbox: FaceBBox
    target_pixel: tuple[int, int]
    frame_center: tuple[float, float]
    normalized_error: tuple[float, float]
    current_pose: npt.NDArray[np.float64]
    target_pose: npt.NDArray[np.float64]
    current_angles: EulerAngles
    target_angles: EulerAngles
    delta_angles: EulerAngles
    incremental_clamp: IncrementalClamp
    inside_provisional_envelope: bool
    target_translation: tuple[float, float, float]
    translation_delta: tuple[float, float, float]
    body_yaw_authorized: bool = False


@dataclass(frozen=True)
class HorizontalCalibrationTarget:
    """One bounded horizontal target calculated from the current head pose."""

    current_pose: npt.NDArray[np.float64]
    commanded_pose: npt.NDArray[np.float64]
    target_pose: npt.NDArray[np.float64]
    current_angles: EulerAngles
    commanded_angles: EulerAngles
    target_angles: EulerAngles
    raw_geometry_delta_yaw: float
    requested_calibration_delta_yaw: float
    clamped_delta_yaw: float
    within_stage1_limit: bool


@dataclass(frozen=True)
class FaceBBoxSample:
    """One metadata-only face observation for physical motion validation."""

    timestamp: float
    u: float
    v: float
    confidence: float


@dataclass(frozen=True)
class FaceBBoxWindow:
    """A bounded metadata-only YuNet sample window."""

    samples: tuple[FaceBBoxSample, ...]
    attempts: int
    invalid_attempts: int
    median_u: float | None
    median_v: float | None
    jitter_u: float | None
    jitter_v: float | None
    stable: bool
    frame_width: int | None = None
    frame_height: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return camera metadata without retaining image data."""
        return {
            "attempts": self.attempts,
            "valid_samples": len(self.samples),
            "invalid_attempts": self.invalid_attempts,
            "samples": [
                {
                    "timestamp": sample.timestamp,
                    "u": sample.u,
                    "v": sample.v,
                    "confidence": sample.confidence,
                }
                for sample in self.samples
            ],
            "median_u": self.median_u,
            "median_v": self.median_v,
            "jitter_u": self.jitter_u,
            "jitter_v": self.jitter_v,
            "stable": self.stable,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }


@dataclass(frozen=True)
class TelemetryEndpointTimings:
    """Monotonic request timings for one physical telemetry acquisition."""

    state_s: float
    status_s: float
    movement_s: float
    total_s: float


@dataclass(frozen=True)
class TelemetryTransportPreparation:
    """Non-authorizing daemon transport preparation result."""

    endpoint: str
    duration_s: float
    result: str


@dataclass(frozen=True)
class Stage1TelemetrySnapshot:
    """Daemon present and commanded state captured without publishing a target."""

    timestamp: float
    acquisition_monotonic: float
    present_head_pose: npt.NDArray[np.float64]
    present_head_joints: tuple[float, ...]
    present_body_yaw: float
    present_antennas: tuple[float, float]
    target_head_pose: npt.NDArray[np.float64] | None
    target_head_joints: tuple[float, ...] | None
    target_body_yaw: float | None
    target_antennas: tuple[float, float] | None
    daemon_timestamp: str
    acquisition_attempts: int
    acquisition_elapsed_s: float
    total_acquisition_elapsed_s: float
    backend_last_alive: float
    backend_last_alive_age_s: float
    daemon_ready: bool
    daemon_error: str | None
    control_loop_frequency_hz: float
    active_move_count: int
    head_tracking_enabled: bool
    observed_control_loop_frequency_hz: float | None = None
    motor_controller_period_s: float | None = None
    control_loop_health_source: str = "mean_control_loop_frequency"
    endpoint_timings: TelemetryEndpointTimings | None = None
    transport_preparation: TelemetryTransportPreparation | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe pose and joint telemetry."""
        present_angles = pose_euler_degrees(self.present_head_pose)
        target_angles = pose_euler_degrees(self.target_head_pose) if self.target_head_pose is not None else None
        return {
            "timestamp": self.timestamp,
            "acquisition_monotonic": self.acquisition_monotonic,
            "daemon_timestamp": self.daemon_timestamp,
            "acquisition_attempts": self.acquisition_attempts,
            "acquisition_elapsed_s": self.acquisition_elapsed_s,
            "total_acquisition_elapsed_s": self.total_acquisition_elapsed_s,
            "freshness_bound_s": STAGE1_TELEMETRY_FRESHNESS_S,
            "backend_last_alive": self.backend_last_alive,
            "backend_last_alive_age_s": self.backend_last_alive_age_s,
            "daemon_ready": self.daemon_ready,
            "daemon_error": self.daemon_error,
            "control_loop_frequency_hz": self.control_loop_frequency_hz,
            "observed_control_loop_frequency_hz": self.observed_control_loop_frequency_hz,
            "motor_controller_period_s": self.motor_controller_period_s,
            "control_loop_health_source": self.control_loop_health_source,
            "active_move_count": self.active_move_count,
            "head_tracking_enabled": self.head_tracking_enabled,
            "endpoint_timings": (
                {
                    "state_s": self.endpoint_timings.state_s,
                    "status_s": self.endpoint_timings.status_s,
                    "movement_s": self.endpoint_timings.movement_s,
                    "total_s": self.endpoint_timings.total_s,
                }
                if self.endpoint_timings is not None
                else None
            ),
            "transport_preparation": (
                {
                    "endpoint": self.transport_preparation.endpoint,
                    "duration_s": self.transport_preparation.duration_s,
                    "result": self.transport_preparation.result,
                }
                if self.transport_preparation is not None
                else None
            ),
            "present_head_pose": self.present_head_pose.tolist(),
            "present_head_angles": {
                "roll": present_angles.roll,
                "pitch": present_angles.pitch,
                "yaw": present_angles.yaw,
            },
            "present_head_joints": list(self.present_head_joints),
            "present_body_yaw": self.present_body_yaw,
            "present_antennas": list(self.present_antennas),
            "target_head_pose": self.target_head_pose.tolist() if self.target_head_pose is not None else None,
            "target_head_angles": (
                {"roll": target_angles.roll, "pitch": target_angles.pitch, "yaw": target_angles.yaw}
                if target_angles is not None
                else None
            ),
            "target_head_joints": list(self.target_head_joints) if self.target_head_joints is not None else None,
            "target_body_yaw": self.target_body_yaw,
            "target_antennas": list(self.target_antennas) if self.target_antennas is not None else None,
        }


@dataclass(frozen=True)
class PostWakeTelemetrySnapshot:
    """Fresh physical state and daemon health used by the post-wake gate."""

    monotonic_time: float
    present_head_pose: npt.NDArray[np.float64]
    present_head_joints: tuple[float, ...]
    present_body_yaw: float
    present_antennas: tuple[float, float]
    control_mode: str
    daemon_timestamp: str
    daemon_ready: bool
    daemon_error: str | None
    backend_last_alive: float
    backend_last_alive_age_s: float
    control_loop_frequency_hz: float
    active_move_count: int
    acquisition_elapsed_s: float
    observed_control_loop_frequency_hz: float | None = None
    motor_controller_period_s: float | None = None
    control_loop_health_source: str = "mean_control_loop_frequency"
    target_head_pose: npt.NDArray[np.float64] | None = None
    target_head_joints: tuple[float, ...] | None = None
    target_body_yaw: float | None = None
    target_antennas: tuple[float, float] | None = None
    head_tracking_enabled: bool | None = None
    endpoint_timings: TelemetryEndpointTimings | None = None
    transport_preparation: TelemetryTransportPreparation | None = None


@dataclass(frozen=True)
class FaceBBoxDisplacement:
    """Post-minus-pre face displacement relative to observed detector jitter."""

    delta_u: float
    delta_v: float
    horizontal_jitter_ratio: float
    exceeds_horizontal_jitter: bool
    left_motion_consistent: bool


class Stage1Instrumentation:
    """Capture camera metadata and read-only daemon telemetry for Stage 1."""

    def __init__(
        self,
        robot: Stage1Robot,
        *,
        detector: YuNetDetector | None = None,
        quality_assessor: Callable[[npt.NDArray[np.uint8], DetectedFace], QualityResult] = assess_face_quality,
    ) -> None:
        """Bind instrumentation to the existing SDK-owned robot connection."""
        self._robot = robot
        self._detector = detector
        self._quality_assessor = quality_assessor
        self._last_daemon_timestamp: datetime | None = None
        self._last_backend_last_alive: float | None = None
        self._last_post_wake_daemon_timestamp: datetime | None = None
        self._http_client: httpx.Client | None = None
        self._last_transport_preparation: TelemetryTransportPreparation | None = None

    def close(self) -> None:
        """Close the read-only telemetry connection."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def capture_face_window(self, *, attempts: int = 8, interval_s: float = 0.1) -> FaceBBoxWindow:
        """Collect quality-valid single-face metadata without retaining frames."""
        if attempts < 5 or attempts > 10:
            raise ValueError("Stage 1 face sampling requires 5 to 10 attempts")
        if interval_s < 0.0 or interval_s > 0.2:
            raise ValueError("Stage 1 face sampling interval must be between 0 and 0.2 seconds")
        if self._detector is None:
            self._detector = YuNetDetector(score_threshold=0.70, allow_download=False)

        samples: list[FaceBBoxSample] = []
        invalid_attempts = 0
        frame_width: int | None = None
        frame_height: int | None = None
        for attempt in range(attempts):
            frame = self._robot.media.get_frame()
            if frame is None:
                invalid_attempts += 1
            else:
                frame_array = np.asarray(frame)
                faces = self._detector.detect(frame_array)
                if len(faces) != 1:
                    invalid_attempts += 1
                else:
                    face = faces[0]
                    quality = self._quality_assessor(frame_array, face)
                    if not quality.usable:
                        invalid_attempts += 1
                    else:
                        x, y, width, height = face.bbox
                        samples.append(
                            FaceBBoxSample(
                                timestamp=time.time(),
                                u=float(x + width / 2.0),
                                v=float(y + height / 2.0),
                                confidence=float(face.confidence),
                            )
                        )
                        frame_width = face.frame_width
                        frame_height = face.frame_height
            if attempt + 1 < attempts and interval_s > 0.0:
                time.sleep(interval_s)

        if not samples:
            return FaceBBoxWindow((), attempts, invalid_attempts, None, None, None, None, False)

        u_values = np.asarray([sample.u for sample in samples], dtype=np.float64)
        v_values = np.asarray([sample.v for sample in samples], dtype=np.float64)
        median_u = float(np.median(u_values))
        median_v = float(np.median(v_values))
        jitter_u = float(np.max(np.abs(u_values - median_u)))
        jitter_v = float(np.max(np.abs(v_values - median_v)))
        assert frame_width is not None and frame_height is not None
        # Sub-percent jitter keeps the expected ~1.5-degree image shift resolvable.
        stable = (
            invalid_attempts == 0
            and len(samples) == attempts
            and jitter_u <= frame_width * 0.005
            and jitter_v <= frame_height * 0.005
        )
        return FaceBBoxWindow(
            tuple(samples),
            attempts,
            invalid_attempts,
            median_u,
            median_v,
            jitter_u,
            jitter_v,
            stable,
            frame_width,
            frame_height,
        )

    def read_telemetry(self, *, expected_tracking_enabled: bool | None = None) -> Stage1TelemetrySnapshot:
        """Read advancing, bounded daemon state without issuing a robot command."""
        started_at = time.monotonic()
        fresh_sample_deadline = started_at + STAGE1_TELEMETRY_FRESH_SAMPLE_TIMEOUT_S
        total_deadline = started_at + STAGE1_TELEMETRY_TOTAL_DEADLINE_S
        baseline_daemon_timestamp = self._last_daemon_timestamp
        baseline_backend_last_alive = self._last_backend_last_alive
        acquisition_attempts = 0
        transport_failures = 0

        while True:
            now = time.monotonic()
            if now >= total_deadline:
                raise RuntimeError("Stage 1 telemetry acquisition exceeded its total deadline")
            if now >= fresh_sample_deadline:
                raise RuntimeError("Stage 1 telemetry sample exceeded the freshness bound")

            acquisition_attempts += 1
            try:
                physical = self.read_post_wake_telemetry(
                    deadline=min(total_deadline, fresh_sample_deadline, now + STAGE1_TELEMETRY_FRESHNESS_S)
                )
            except RuntimeError as exc:
                if not str(exc).startswith("physical telemetry request"):
                    raise
                transport_failures += 1
                if transport_failures >= STAGE1_TELEMETRY_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"Stage 1 telemetry timed out after {STAGE1_TELEMETRY_MAX_ATTEMPTS} attempts"
                    ) from exc
                retry_delay_s = min(
                    STAGE1_TELEMETRY_RETRY_BACKOFF_S,
                    total_deadline - time.monotonic(),
                    fresh_sample_deadline - time.monotonic(),
                )
                if retry_delay_s <= 0.0:
                    raise RuntimeError("Stage 1 telemetry sample exceeded the freshness bound") from exc
                time.sleep(retry_delay_s)
                continue
            except ValueError as exc:
                if "timestamp is stale" not in str(exc):
                    raise
                physical = None

            if physical is not None:
                daemon_timestamp = _parse_daemon_timestamp(physical.daemon_timestamp)
                tracking_enabled = physical.head_tracking_enabled
                if tracking_enabled is None:
                    raise ValueError("daemon head_tracking_enabled state is missing")
                if not physical.daemon_ready or physical.daemon_error is not None:
                    raise RuntimeError("Stage 1 telemetry daemon is unhealthy")
                if physical.control_mode != "enabled":
                    raise RuntimeError("Stage 1 telemetry control loop is not enabled")
                if physical.control_loop_frequency_hz < POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ:
                    raise RuntimeError("Stage 1 telemetry control loop frequency is unhealthy")
                if physical.active_move_count != 0:
                    raise RuntimeError("Stage 1 telemetry conflicts with an active daemon movement")
                if expected_tracking_enabled is not None and tracking_enabled is not expected_tracking_enabled:
                    raise RuntimeError("Stage 1 telemetry authoritative tracking state is unsafe")
                if physical.acquisition_elapsed_s > STAGE1_TELEMETRY_FRESHNESS_S:
                    raise RuntimeError("Stage 1 telemetry acquisition exceeded the freshness bound")
                if physical.backend_last_alive_age_s > STAGE1_TELEMETRY_FRESHNESS_S:
                    raise RuntimeError("Stage 1 telemetry backend liveness exceeded the freshness bound")

                if baseline_daemon_timestamp is None or baseline_backend_last_alive is None:
                    baseline_daemon_timestamp = daemon_timestamp
                    baseline_backend_last_alive = physical.backend_last_alive
                else:
                    daemon_advanced = daemon_timestamp > baseline_daemon_timestamp
                    backend_advanced = physical.backend_last_alive > baseline_backend_last_alive
                    if daemon_advanced and backend_advanced:
                        completed_at = time.monotonic()
                        logger.debug(
                            "Stage 1 telemetry category=SUCCESS acquisition_ms=%.1f total_ms=%.1f "
                            "daemon_timestamp_advanced=true backend_last_alive_advanced=true "
                            "backend_liveness_age_ms=%.1f freshness=pass",
                            physical.acquisition_elapsed_s * 1000.0,
                            (completed_at - started_at) * 1000.0,
                            physical.backend_last_alive_age_s * 1000.0,
                        )
                        snapshot = Stage1TelemetrySnapshot(
                            timestamp=time.time(),
                            acquisition_monotonic=completed_at,
                            present_head_pose=physical.present_head_pose,
                            present_head_joints=physical.present_head_joints,
                            present_body_yaw=physical.present_body_yaw,
                            present_antennas=physical.present_antennas,
                            target_head_pose=physical.target_head_pose,
                            target_head_joints=physical.target_head_joints,
                            target_body_yaw=physical.target_body_yaw,
                            target_antennas=physical.target_antennas,
                            daemon_timestamp=physical.daemon_timestamp,
                            acquisition_attempts=acquisition_attempts,
                            acquisition_elapsed_s=physical.acquisition_elapsed_s,
                            total_acquisition_elapsed_s=completed_at - started_at,
                            backend_last_alive=physical.backend_last_alive,
                            backend_last_alive_age_s=physical.backend_last_alive_age_s,
                            daemon_ready=physical.daemon_ready,
                            daemon_error=physical.daemon_error,
                            control_loop_frequency_hz=physical.control_loop_frequency_hz,
                            active_move_count=physical.active_move_count,
                            head_tracking_enabled=tracking_enabled,
                            observed_control_loop_frequency_hz=physical.observed_control_loop_frequency_hz,
                            motor_controller_period_s=physical.motor_controller_period_s,
                            control_loop_health_source=physical.control_loop_health_source,
                            endpoint_timings=physical.endpoint_timings,
                            transport_preparation=physical.transport_preparation,
                        )
                        self._last_daemon_timestamp = daemon_timestamp
                        self._last_backend_last_alive = physical.backend_last_alive
                        return snapshot
                    logger.warning(
                        "Stage 1 telemetry category=NOT_ADVANCING daemon_timestamp_advanced=%s "
                        "backend_last_alive_advanced=%s retry=true",
                        daemon_advanced,
                        backend_advanced,
                    )

            retry_delay_s = min(
                STAGE1_TELEMETRY_FRESH_SAMPLE_POLL_S,
                total_deadline - time.monotonic(),
                fresh_sample_deadline - time.monotonic(),
            )
            if retry_delay_s <= 0.0:
                raise RuntimeError("Stage 1 telemetry sample exceeded the freshness bound")
            time.sleep(retry_delay_s)

    def read_post_wake_telemetry(self, *, deadline: float) -> PostWakeTelemetrySnapshot:
        """Read bounded physical state and daemon health without commanding the robot."""
        self._ensure_http_client()

        started_at = time.monotonic()
        state_payload, state_elapsed_s = self._bounded_get_json(
            "/api/state/full",
            deadline=deadline,
            started_at=started_at,
            params={
                "with_control_mode": "true",
                "with_head_pose": "true",
                "with_target_head_pose": "true",
                "with_head_joints": "true",
                "with_target_head_joints": "true",
                "with_body_yaw": "true",
                "with_target_body_yaw": "true",
                "with_antenna_positions": "true",
                "with_target_antenna_positions": "true",
                "use_pose_matrix": "true",
            },
        )
        status_payload, status_elapsed_s = self._bounded_get_json(
            "/api/daemon/status",
            deadline=deadline,
            started_at=started_at,
        )
        running_moves_payload, movement_elapsed_s = self._bounded_get_json(
            "/api/move/running",
            deadline=deadline,
            started_at=started_at,
        )
        completed_at = time.monotonic()
        acquisition_elapsed_s = completed_at - started_at
        if acquisition_elapsed_s > STAGE1_TELEMETRY_FRESHNESS_S or completed_at > deadline:
            raise RuntimeError("post-wake telemetry response exceeded the freshness bound")

        state = _required_dict(state_payload, "post-wake state")
        status = _required_dict(status_payload, "daemon status")
        backend_status = _required_dict(status.get("backend_status"), "daemon backend status")
        control_loop_stats = _required_dict(backend_status.get("control_loop_stats"), "control loop stats")
        if not isinstance(running_moves_payload, list):
            raise ValueError("running daemon moves must be a list")

        daemon_timestamp_value = state.get("timestamp")
        daemon_timestamp = _parse_daemon_timestamp(daemon_timestamp_value)
        if (
            self._last_post_wake_daemon_timestamp is not None
            and daemon_timestamp <= self._last_post_wake_daemon_timestamp
        ):
            raise ValueError("daemon post-wake telemetry timestamp is stale")

        present_pose = _pose_from_telemetry(state.get("head_pose"), "head_pose")
        present_head_joints = _finite_tuple(state.get("head_joints"), 7, "head_joints")
        present_body_yaw = _finite_float(state.get("body_yaw"), "body_yaw")
        present_antennas_values = _finite_tuple(state.get("antennas_position"), 2, "antennas_position")
        target_pose_value = state.get("target_head_pose")
        target_joints_value = state.get("target_head_joints")
        target_body_yaw_value = state.get("target_body_yaw")
        target_antennas_value = state.get("target_antennas_position")
        target_head_pose = (
            _pose_from_telemetry(target_pose_value, "target_head_pose") if target_pose_value is not None else None
        )
        target_head_joints = (
            _finite_tuple(target_joints_value, 7, "target_head_joints") if target_joints_value is not None else None
        )
        target_body_yaw = (
            _finite_float(target_body_yaw_value, "target_body_yaw") if target_body_yaw_value is not None else None
        )
        target_antennas_values = (
            _finite_tuple(target_antennas_value, 2, "target_antennas_position")
            if target_antennas_value is not None
            else None
        )
        target_antennas = (
            (target_antennas_values[0], target_antennas_values[1]) if target_antennas_values is not None else None
        )
        control_mode = state.get("control_mode")
        if not isinstance(control_mode, str):
            raise ValueError("post-wake motor control mode is missing")

        backend_last_alive = _finite_float(backend_status.get("last_alive"), "backend last_alive")
        (
            control_loop_frequency_hz,
            observed_control_loop_frequency_hz,
            motor_controller_period_s,
            control_loop_health_source,
        ) = _control_loop_health_frequency(control_loop_stats)
        daemon_error = _optional_error(status.get("error"), "daemon error") or _optional_error(
            backend_status.get("error"),
            "backend error",
        )
        daemon_ready = status.get("state") == "running" and backend_status.get("ready") is True
        head_tracking_enabled = status.get("head_tracking_enabled")
        if not isinstance(head_tracking_enabled, bool):
            raise ValueError("daemon head_tracking_enabled state is missing")
        backend_last_alive_age_s = max(0.0, daemon_timestamp.timestamp() - backend_last_alive)

        snapshot = PostWakeTelemetrySnapshot(
            monotonic_time=completed_at,
            present_head_pose=present_pose,
            present_head_joints=present_head_joints,
            present_body_yaw=present_body_yaw,
            present_antennas=(present_antennas_values[0], present_antennas_values[1]),
            control_mode=control_mode,
            daemon_timestamp=str(daemon_timestamp_value),
            daemon_ready=daemon_ready,
            daemon_error=daemon_error,
            backend_last_alive=backend_last_alive,
            backend_last_alive_age_s=backend_last_alive_age_s,
            control_loop_frequency_hz=control_loop_frequency_hz,
            observed_control_loop_frequency_hz=observed_control_loop_frequency_hz,
            motor_controller_period_s=motor_controller_period_s,
            control_loop_health_source=control_loop_health_source,
            active_move_count=len(running_moves_payload),
            acquisition_elapsed_s=acquisition_elapsed_s,
            target_head_pose=target_head_pose,
            target_head_joints=target_head_joints,
            target_body_yaw=target_body_yaw,
            target_antennas=target_antennas,
            head_tracking_enabled=head_tracking_enabled,
            endpoint_timings=TelemetryEndpointTimings(
                state_s=state_elapsed_s,
                status_s=status_elapsed_s,
                movement_s=movement_elapsed_s,
                total_s=acquisition_elapsed_s,
            ),
            transport_preparation=self._last_transport_preparation,
        )
        self._last_post_wake_daemon_timestamp = daemon_timestamp
        return snapshot

    def prepare_telemetry_transport(
        self,
        *,
        timeout_s: float = STAGE1_TELEMETRY_TRANSPORT_PREP_TIMEOUT_S,
    ) -> TelemetryTransportPreparation:
        """Prepare the reusable daemon HTTP transport without authorizing telemetry."""
        if timeout_s <= 0.0:
            raise ValueError("telemetry transport preparation timeout must be positive")
        self._ensure_http_client()
        endpoint = "/api/daemon/status"
        started_at = time.monotonic()
        try:
            assert self._http_client is not None
            response = self._http_client.get(
                f"http://{self._robot.client.host}:{self._robot.client.port}{endpoint}",
                timeout=timeout_s,
            )
            response.raise_for_status()
            if not isinstance(response.json(), dict):
                raise RuntimeError("telemetry transport preparation returned malformed status")
        except httpx.TimeoutException as exc:
            duration_s = time.monotonic() - started_at
            self._last_transport_preparation = TelemetryTransportPreparation(endpoint, duration_s, "timeout")
            raise RuntimeError("telemetry transport preparation timed out") from exc
        except httpx.HTTPError as exc:
            duration_s = time.monotonic() - started_at
            self._last_transport_preparation = TelemetryTransportPreparation(endpoint, duration_s, "failed")
            raise RuntimeError(f"telemetry transport preparation failed: {exc}") from exc
        duration_s = time.monotonic() - started_at
        self._last_transport_preparation = TelemetryTransportPreparation(endpoint, duration_s, "pass")
        return self._last_transport_preparation

    def _ensure_http_client(self) -> None:
        if self._http_client is not None:
            return
        self._http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=STAGE1_TELEMETRY_CONNECT_TIMEOUT_S,
                read=STAGE1_TELEMETRY_READ_TIMEOUT_S,
                write=STAGE1_TELEMETRY_WRITE_TIMEOUT_S,
                pool=STAGE1_TELEMETRY_POOL_TIMEOUT_S,
            )
        )

    def _bounded_get_json(
        self,
        path: str,
        *,
        deadline: float,
        started_at: float,
        params: dict[str, str] | None = None,
    ) -> tuple[object, float]:
        """Fetch one physical telemetry resource inside the shared freshness deadline."""
        assert self._http_client is not None
        now = time.monotonic()
        request_timeout_s = min(deadline - now, STAGE1_TELEMETRY_FRESHNESS_S - (now - started_at))
        if request_timeout_s <= 0.0:
            raise RuntimeError("physical telemetry deadline expired")
        request_started_at = time.monotonic()
        try:
            response = self._http_client.get(
                f"http://{self._robot.client.host}:{self._robot.client.port}{path}",
                params=params,
                timeout=request_timeout_s,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise RuntimeError("physical telemetry request timed out") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"physical telemetry request failed: {exc}") from exc
        return response.json(), time.monotonic() - request_started_at


def compare_face_windows(pre: FaceBBoxWindow, post: FaceBBoxWindow) -> FaceBBoxDisplacement:
    """Compare stable metadata windows for the expected left-turn image sign."""
    if not pre.stable or not post.stable:
        raise ValueError("Stage 1 face windows must both be stable")
    if pre.median_u is None or pre.median_v is None or pre.jitter_u is None:
        raise ValueError("pre-motion face metadata is incomplete")
    if post.median_u is None or post.median_v is None:
        raise ValueError("post-motion face metadata is incomplete")

    delta_u = post.median_u - pre.median_u
    delta_v = post.median_v - pre.median_v
    # Two observed maximum deviations, with a one-pixel detector floor, avoids
    # calling a quantization-scale change material.
    jitter_reference = max(pre.jitter_u, 1.0)
    exceeds_horizontal_jitter = abs(delta_u) > 2.0 * jitter_reference
    return FaceBBoxDisplacement(
        delta_u=delta_u,
        delta_v=delta_v,
        horizontal_jitter_ratio=delta_u / jitter_reference,
        exceeds_horizontal_jitter=exceeds_horizontal_jitter,
        left_motion_consistent=delta_u > 0.0 and exceeds_horizontal_jitter,
    )


def frame_center(frame_width: int, frame_height: int) -> tuple[float, float]:
    """Return the centre of a valid image frame in pixels."""
    _validate_frame_dimensions(frame_width, frame_height)
    return frame_width / 2.0, frame_height / 2.0


def bbox_center_pixel(bbox: FaceBBox, frame_width: int, frame_height: int) -> tuple[int, int]:
    """Return a face bbox centre clamped to the SDK-valid image interior."""
    _validate_frame_dimensions(frame_width, frame_height)
    x, y, width, height = bbox
    if not np.all(np.isfinite(np.asarray(bbox, dtype=np.float64))):
        raise ValueError("face bbox must contain only finite values")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("face bbox width and height must be positive")

    u = int(round(x + width / 2.0))
    v = int(round(y + height / 2.0))
    return min(max(u, 1), frame_width - 1), min(max(v, 1), frame_height - 1)


def normalized_image_error(target_pixel: tuple[int, int], frame_width: int, frame_height: int) -> tuple[float, float]:
    """Return signed image error normalized by each frame half-dimension."""
    center_x, center_y = frame_center(frame_width, frame_height)
    u, v = target_pixel
    return (u - center_x) / center_x, (v - center_y) / center_y


def validate_head_pose(pose: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a copied float64 pose after validating its rigid transform."""
    try:
        pose_array = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PoseValidationError("head pose must be a numeric array") from exc
    if pose_array.shape != (4, 4):
        raise PoseValidationError(f"head pose must have shape (4, 4), got {pose_array.shape}")
    if not np.all(np.isfinite(pose_array)):
        raise PoseValidationError("head pose must contain only finite values")
    if not np.allclose(pose_array[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise PoseValidationError("head pose must have homogeneous bottom row [0, 0, 0, 1]")

    rotation = pose_array[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_ATOL):
        raise PoseValidationError("head pose rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_ROTATION_ATOL):
        raise PoseValidationError("head pose rotation must have determinant 1")
    return pose_array.copy()


def pose_euler_degrees(pose: npt.ArrayLike) -> EulerAngles:
    """Extract Reachy's XYZ roll, pitch, and yaw convention in degrees."""
    pose_array = validate_head_pose(pose)
    values = np.asarray(Rotation.from_matrix(pose_array[:3, :3]).as_euler("xyz", degrees=True), dtype=np.float64)
    return EulerAngles(roll=float(values[0]), pitch=float(values[1]), yaw=float(values[2]))


def angular_delta(current: EulerAngles, target: EulerAngles) -> EulerAngles:
    """Return shortest signed target-minus-current XYZ angular deltas."""
    return EulerAngles(
        roll=_shortest_angle_delta(current.roll, target.roll),
        pitch=_shortest_angle_delta(current.pitch, target.pitch),
        yaw=_shortest_angle_delta(current.yaw, target.yaw),
    )


def inside_provisional_envelope(
    target: EulerAngles,
    *,
    neutral: EulerAngles = EulerAngles(roll=0.0, pitch=0.0, yaw=0.0),
) -> bool:
    """Return whether target yaw and pitch are inside provisional neutral-relative limits."""
    offset = angular_delta(neutral, target)
    return abs(offset.yaw) <= PROVISIONAL_YAW_ENVELOPE_DEG and abs(offset.pitch) <= PROVISIONAL_PITCH_ENVELOPE_DEG


def calculate_incremental_clamp(current: EulerAngles, target: EulerAngles) -> IncrementalClamp:
    """Calculate provisional yaw and pitch delta limits without creating a target pose."""
    requested = angular_delta(current, target)
    limited_yaw = float(np.clip(requested.yaw, -PROVISIONAL_YAW_INCREMENT_DEG, PROVISIONAL_YAW_INCREMENT_DEG))
    limited_pitch = float(np.clip(requested.pitch, -PROVISIONAL_PITCH_INCREMENT_DEG, PROVISIONAL_PITCH_INCREMENT_DEG))
    return IncrementalClamp(
        requested_yaw=requested.yaw,
        requested_pitch=requested.pitch,
        limited_yaw=limited_yaw,
        limited_pitch=limited_pitch,
        yaw_was_limited=not np.isclose(limited_yaw, requested.yaw),
        pitch_was_limited=not np.isclose(limited_pitch, requested.pitch),
    )


def calculate_horizontal_calibration_target(
    current_pose: npt.ArrayLike,
    commanded_pose: npt.ArrayLike,
    raw_geometry_delta_yaw: float,
) -> HorizontalCalibrationTarget:
    """Build a Stage 1 yaw-only target from the active command baseline."""
    if not np.isfinite(raw_geometry_delta_yaw):
        raise ValueError("raw geometry yaw delta must be finite")
    if np.isclose(raw_geometry_delta_yaw, 0.0):
        raise ValueError("raw geometry yaw delta must indicate a direction")

    current_pose_array = validate_head_pose(current_pose)
    commanded_pose_array = validate_head_pose(commanded_pose)
    current_angles = pose_euler_degrees(current_pose_array)
    commanded_angles = pose_euler_degrees(commanded_pose_array)
    if not inside_provisional_envelope(commanded_angles):
        raise PoseValidationError("commanded pose is outside the provisional Stage 1 envelope")

    requested_delta = float(
        np.copysign(min(abs(raw_geometry_delta_yaw), PROVISIONAL_YAW_INCREMENT_DEG), raw_geometry_delta_yaw)
    )
    requested_target_yaw = commanded_angles.yaw + requested_delta
    target_yaw = float(
        np.clip(
            requested_target_yaw,
            -PROVISIONAL_YAW_ENVELOPE_DEG,
            PROVISIONAL_YAW_ENVELOPE_DEG,
        )
    )
    clamped_delta = _shortest_angle_delta(commanded_angles.yaw, target_yaw)
    if np.isclose(clamped_delta, 0.0) or np.sign(clamped_delta) != np.sign(raw_geometry_delta_yaw):
        raise PoseValidationError("no safe Stage 1 movement remains inside the yaw envelope")

    target_pose = commanded_pose_array.copy()
    target_pose[:3, :3] = Rotation.from_euler(
        "xyz",
        [commanded_angles.roll, commanded_angles.pitch, target_yaw],
        degrees=True,
    ).as_matrix()
    target_angles = pose_euler_degrees(target_pose)
    pitch_delta = _shortest_angle_delta(commanded_angles.pitch, target_angles.pitch)
    within_stage1_limit = (
        abs(clamped_delta) <= PROVISIONAL_YAW_INCREMENT_DEG
        and abs(pitch_delta) <= 0.25
        and inside_provisional_envelope(target_angles)
    )
    if not within_stage1_limit:
        raise PoseValidationError("calibration target exceeds Stage 1 limits")

    return HorizontalCalibrationTarget(
        current_pose=current_pose_array,
        commanded_pose=commanded_pose_array,
        target_pose=target_pose,
        current_angles=current_angles,
        commanded_angles=commanded_angles,
        target_angles=target_angles,
        raw_geometry_delta_yaw=float(raw_geometry_delta_yaw),
        requested_calibration_delta_yaw=requested_delta,
        clamped_delta_yaw=clamped_delta,
        within_stage1_limit=True,
    )


def calculate_face_geometry(
    robot: NoMotionLookAt,
    bbox: FaceBBox,
    frame_width: int,
    frame_height: int,
) -> FaceGeometryDiagnostic:
    """Calculate one bbox-to-head diagnostic without authorizing movement."""
    target_pixel = bbox_center_pixel(bbox, frame_width, frame_height)
    current_pose = validate_head_pose(robot.get_current_head_pose())
    target_pose = validate_head_pose(
        robot.look_at_image(
            target_pixel[0],
            target_pixel[1],
            duration=0.0,
            perform_movement=False,
        )
    )
    current_angles = pose_euler_degrees(current_pose)
    target_angles = pose_euler_degrees(target_pose)
    delta_angles = angular_delta(current_angles, target_angles)
    target_translation_array = target_pose[:3, 3]
    translation_delta_array = target_translation_array - current_pose[:3, 3]

    return FaceGeometryDiagnostic(
        frame_width=frame_width,
        frame_height=frame_height,
        bbox=bbox,
        target_pixel=target_pixel,
        frame_center=frame_center(frame_width, frame_height),
        normalized_error=normalized_image_error(target_pixel, frame_width, frame_height),
        current_pose=current_pose,
        target_pose=target_pose,
        current_angles=current_angles,
        target_angles=target_angles,
        delta_angles=delta_angles,
        incremental_clamp=calculate_incremental_clamp(current_angles, target_angles),
        inside_provisional_envelope=inside_provisional_envelope(target_angles),
        target_translation=(
            float(target_translation_array[0]),
            float(target_translation_array[1]),
            float(target_translation_array[2]),
        ),
        translation_delta=(
            float(translation_delta_array[0]),
            float(translation_delta_array[1]),
            float(translation_delta_array[2]),
        ),
    )


def _validate_frame_dimensions(frame_width: int, frame_height: int) -> None:
    if frame_width < 2 or frame_height < 2:
        raise ValueError("frame dimensions must both be at least 2 pixels")


def _shortest_angle_delta(current: float, target: float) -> float:
    return (target - current + 180.0) % 360.0 - 180.0


def _finite_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _finite_tuple(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    values = tuple(_finite_float(item, name) for item in value)
    return values


def _control_loop_health_frequency(
    control_loop_stats: dict[str, object],
) -> tuple[float, float, float | None, str]:
    mean_frequency_hz = _finite_float(
        control_loop_stats.get("mean_control_loop_frequency"),
        "mean control loop frequency",
    )
    motor_controller = control_loop_stats.get("motor_controller")
    if isinstance(motor_controller, str):
        match = _MOTOR_CONTROLLER_PERIOD_RE.search(motor_controller)
        if match is not None:
            period_ms = float(match.group(1))
            if not np.isfinite(period_ms) or period_ms <= 0.0:
                raise ValueError("motor controller period must be positive")
            period_s = period_ms / 1000.0
            return 1.0 / period_s, mean_frequency_hz, period_s, "motor_controller_period"
    return mean_frequency_hz, mean_frequency_hz, None, "mean_control_loop_frequency"


def _pose_from_telemetry(value: object, name: str) -> npt.NDArray[np.float64]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a pose object")
    matrix = _finite_tuple(value.get("m"), 16, name)
    return validate_head_pose(np.asarray(matrix, dtype=np.float64).reshape(4, 4))


def _parse_daemon_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("daemon Stage 1 telemetry timestamp is missing")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("daemon Stage 1 telemetry timestamp is invalid") from exc
    if timestamp.tzinfo is None:
        raise ValueError("daemon Stage 1 telemetry timestamp must include a timezone")
    return timestamp


def _required_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _optional_error(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value or None
