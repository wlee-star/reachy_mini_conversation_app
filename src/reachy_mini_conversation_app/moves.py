"""Movement system driving sequential primary moves.

Design overview
- Primary moves (emotions, dances, goto, breathing) are mutually exclusive and run
  sequentially.
- There is a single control point to the robot: `ReachyMini.set_target`.
- The control loop runs near 100 Hz and is phase-aligned via a monotonic clock.
- Idle behaviour starts an infinite `BreathingMove` after a short inactivity delay
  unless listening is active.

Threading model
- A dedicated worker thread owns all real-time state and issues `set_target`
  commands.
- Other threads communicate via a command queue (enqueue moves, mark activity,
  toggle listening).

Units and frames
- Antennas and `body_yaw` are in radians.

Safety
- Listening freezes antennas, then blends them back on unfreeze.
- Interpolations and blends are used to avoid jumps at all times.
- `set_target` errors are rate-limited in logs.
"""

from __future__ import annotations
import time
import uuid
import logging
import threading
from queue import Empty, Queue
from typing import Any, Dict, Tuple
from collections import deque
from dataclasses import field, dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from reachy_mini.utils.interpolation import compose_world_offset, linear_pose_interpolation
from reachy_mini_conversation_app.wake_trace import WakeTrace, WakeTraceCommand
from reachy_mini_conversation_app.app_lifecycle import (
    StartupFreshnessRecorder,
    AuthorizedStartupSnapshot,
    DaemonTrackingRecoveryError,
    ensure_daemon_head_tracking_disabled,
    authorized_startup_snapshot_diagnostics,
)
from reachy_mini_conversation_app.face_tracking import (
    STAGE1_TELEMETRY_FRESHNESS_S,
    FaceBBoxWindow,
    FaceBBoxDisplacement,
    Stage1Instrumentation,
    Stage1TelemetrySnapshot,
    HorizontalCalibrationTarget,
    angular_delta,
    pose_euler_degrees,
    compare_face_windows,
    calculate_horizontal_calibration_target,
)
from reachy_mini_conversation_app.dance_emotion_moves import EmotionQueueMove


logger = logging.getLogger(__name__)

_STARTUP_ATTRIBUTION_TOLERANCE_DEG = 1e-6

# Configuration constants
CONTROL_LOOP_FREQUENCY_HZ = 60.0  # Hz - Target frequency for the movement control loop
STAGE1_PREVIEW_MAX_AGE_S = 300.0
STAGE1_PREVIEW_POSE_TOLERANCE_DEG = 0.25
STAGE1_FACE_POSITION_TOLERANCE_RATIO = 0.01  # Twice the per-window jitter allowance, but only 1% of the frame.
STAGE1_COMMAND_FRESHNESS_S = 0.25
STAGE1_SETTLING_S = 0.4
STAGE1_PREVIEW_SETTLING_DURATION_S = 3.0
STAGE1_PREVIEW_SETTLING_TIMEOUT_S = 10.0
STAGE1_PREVIEW_SETTLING_INTERVAL_S = 0.1
STAGE1_PREVIEW_STATIONARY_LIMIT_DEG = 0.25
SDK_CONTROL_FAILURE_THRESHOLD = 5
SDK_CONTROL_LAST_SUCCESS_MAX_AGE_S = 0.25
SDK_CONTROL_PROBE_STALE_AFTER_S = 0.5
SDK_CONTROL_PROBE_FAILURE_THRESHOLD = 2
SDK_RECOVERY_ATTEMPT_LIMIT = 3
SDK_RECOVERY_BACKOFF_S = (0.0, 0.5, 1.0)

# Type definitions
FullBodyPose = Tuple[NDArray[np.float32], Tuple[float, float], float]  # (head_pose_4x4, antennas, body_yaw)


class BreathingMove(Move):  # type: ignore
    """Breathing move with interpolation to a base pose and continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float32],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
        base_head_pose: NDArray[np.float32] | None = None,
        base_antennas: Tuple[float, float] | None = None,
        base_body_yaw: float = 0.0,
    ):
        """Initialize breathing move.

        Args:
            interpolation_start_pose: 4x4 matrix of current head pose to interpolate from
            interpolation_start_antennas: Current antenna positions to interpolate from
            interpolation_duration: Duration of interpolation to the base pose (seconds)
            base_head_pose: Head pose around which breathing is applied
            base_antennas: Antenna positions around which breathing is applied
            base_body_yaw: Body yaw held while breathing

        """
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        self.base_head_pose = (
            base_head_pose.copy() if base_head_pose is not None else create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        )
        self.base_antennas = np.array(base_antennas if base_antennas is not None else (-0.1745, 0.1745))
        self.breathing_antennas = np.array(base_antennas if base_antennas is not None else (0.0, 0.0))
        self.base_body_yaw = base_body_yaw

        # Breathing parameters (gentle continuous sine; keep sway small to avoid stepped look)
        self.breathing_z_amplitude = 0.005  # 5mm gentle breathing
        self.breathing_frequency = 0.1  # Hz (6 breaths per minute)
        self.antenna_sway_amplitude = np.deg2rad(8)  # soft antenna sway
        self.antenna_frequency = 0.25  # Hz (slow sway, less mechanical)

    @property
    def duration(self) -> float:
        """Duration property required by official Move interface."""
        return float("inf")  # Continuous breathing (never ends naturally)

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate breathing move at time t."""
        if t < self.interpolation_duration:
            # Phase 1: Interpolate to the breathing base position
            interpolation_t = t / self.interpolation_duration

            # Interpolate head pose
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                self.base_head_pose,
                interpolation_t,
            )

            # Interpolate antennas
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.base_antennas
            antennas = antennas_interp.astype(np.float64)

        else:
            # Phase 2: Breathing patterns from the base pose
            breathing_time = t - self.interpolation_duration

            # Gentle z-axis breathing
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            breathing_offset = create_head_pose(
                x=0,
                y=0,
                z=z_offset,
                roll=0,
                pitch=0,
                yaw=0,
                degrees=True,
                mm=False,
            )
            head_pose = compose_world_offset(self.base_head_pose, breathing_offset)

            # Antenna sway (opposite directions)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = self.breathing_antennas + np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        # Return in official Move interface format: (head_pose, antennas_array, body_yaw)
        return (head_pose, antennas, self.base_body_yaw)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep copy of a full body pose tuple."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


def full_body_pose_equal(first: FullBodyPose, second: FullBodyPose) -> bool:
    """Return whether two commanded targets are identical within SDK precision."""
    return bool(
        np.allclose(first[0], second[0], atol=1e-5)
        and np.allclose(first[1], second[1], atol=1e-5)
        and np.isclose(first[2], second[2], atol=1e-5)
    )


@dataclass
class MovementState:
    """State tracking for the movement system."""

    # Primary move state
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0

    # Status flags
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        """Update the last activity time."""
        self.last_activity_time = time.monotonic()


@dataclass
class LoopFrequencyStats:
    """Track rolling loop frequency statistics."""

    mean: float = 0.0
    m2: float = 0.0
    min_freq: float = float("inf")
    count: int = 0
    last_freq: float = 0.0
    potential_freq: float = 0.0

    def reset(self) -> None:
        """Reset accumulators while keeping the last potential frequency."""
        self.mean = 0.0
        self.m2 = 0.0
        self.min_freq = float("inf")
        self.count = 0


@dataclass(frozen=True)
class Stage1CalibrationPreview:
    """Prepared one-shot horizontal calibration values."""

    session_id: str
    target: HorizontalCalibrationTarget
    antennas: Tuple[float, float]
    body_yaw: float
    pre_face: FaceBBoxWindow
    pre_telemetry: Stage1TelemetrySnapshot
    settling: _Stage1SettlingResult
    breathing_was_active: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostic values without pose matrices."""
        return {
            "session_id": self.session_id,
            "measured_present_yaw": self.target.current_angles.yaw,
            "current_commanded_yaw": self.target.commanded_angles.yaw,
            "present_vs_commanded_difference": angular_delta(
                self.target.commanded_angles, self.target.current_angles
            ).yaw,
            "geometry_direction": "left" if self.target.raw_geometry_delta_yaw > 0.0 else "right",
            "raw_geometry_delta_yaw": self.target.raw_geometry_delta_yaw,
            "requested_calibration_delta_yaw": self.target.requested_calibration_delta_yaw,
            "clamped_delta_yaw": self.target.clamped_delta_yaw,
            "target_commanded_yaw": self.target.target_angles.yaw,
            "commanded_pitch": self.target.commanded_angles.pitch,
            "target_pitch": self.target.target_angles.pitch,
            "body_yaw_change": 0.0,
            "antenna_change": 0.0,
            "within_stage1_limit": self.target.within_stage1_limit,
            "movement_count_planned": 1,
            "movement_executed": False,
            "breathing_was_active": self.breathing_was_active,
            "command_baseline_source": "fresh MovementManager command, frozen by its worker",
            "daemon_target_telemetry_available": self.pre_telemetry.target_head_pose is not None,
            "settling": self.settling.to_dict(),
            "pre_face": self.pre_face.to_dict(),
            "pre_telemetry": self.pre_telemetry.to_dict(),
        }


@dataclass(frozen=True)
class Stage1CalibrationResult:
    """Acknowledgement for one consumed calibration session."""

    preview: Stage1CalibrationPreview
    accepted: bool
    command_count: int
    execution_pre_face: FaceBBoxWindow | None = None
    execution_post_face: FaceBBoxWindow | None = None
    execution_pre_telemetry: Stage1TelemetrySnapshot | None = None
    execution_post_telemetry: Stage1TelemetrySnapshot | None = None
    face_displacement: FaceBBoxDisplacement | None = None
    post_capture_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe acknowledgement values."""
        payload: dict[str, object] = {
            **self.preview.to_dict(),
            "accepted": self.accepted,
            "command_count": self.command_count,
            "movement_executed": self.command_count == 1,
            "post_capture_error": self.post_capture_error,
        }
        if self.execution_pre_face is not None:
            payload["execution_pre_face"] = self.execution_pre_face.to_dict()
        if self.execution_post_face is not None:
            payload["execution_post_face"] = self.execution_post_face.to_dict()
        if self.execution_pre_telemetry is not None:
            payload["execution_pre_telemetry"] = self.execution_pre_telemetry.to_dict()
        if self.execution_post_telemetry is not None:
            payload["execution_post_telemetry"] = self.execution_post_telemetry.to_dict()
        if self.face_displacement is not None:
            payload["face_displacement"] = {
                "delta_u": self.face_displacement.delta_u,
                "delta_v": self.face_displacement.delta_v,
                "horizontal_jitter_ratio": self.face_displacement.horizontal_jitter_ratio,
                "exceeds_horizontal_jitter": self.face_displacement.exceeds_horizontal_jitter,
                "left_motion_consistent": self.face_displacement.left_motion_consistent,
            }
        return payload


@dataclass
class _Stage1CalibrationRequest:
    session_id: str
    raw_geometry_delta_yaw: float
    preview: Stage1CalibrationPreview
    prepared_at: float
    consumed: bool = False


@dataclass
class _Stage1Lifecycle:
    session_id: str
    movement_was_running: bool
    owns_movement_freeze: bool = False
    worker_was_stopped: bool = False
    expiry_timer: threading.Timer | None = None
    releasing: bool = False
    released: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Stage1CalibrationAcknowledgement:
    event: threading.Event = field(default_factory=threading.Event)
    result: Stage1CalibrationResult | None = None
    error: str | None = None


@dataclass
class _Stage1FreezeAcknowledgement:
    event: threading.Event = field(default_factory=threading.Event)
    commanded_pose: FullBodyPose | None = None
    commanded_at: float | None = None
    breathing_was_active: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _Stage1PreviewBaseline:
    owner_session_id: str
    commanded_pose: FullBodyPose
    commanded_at: float
    captured_at: float
    breathing_was_active: bool
    publication_count: int


@dataclass
class _Stage1PreviewAcknowledgement:
    event: threading.Event = field(default_factory=threading.Event)
    baseline: _Stage1PreviewBaseline | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Stage1SettlingSample:
    sampled_at: float
    yaw: float
    pitch: float


@dataclass(frozen=True)
class _Stage1SettlingResult:
    duration: float
    samples: tuple[_Stage1SettlingSample, ...]
    yaw_range: float
    pitch_range: float

    def to_dict(self) -> dict[str, object]:
        return {
            "duration": self.duration,
            "sample_count": len(self.samples),
            "head_yaw_start": self.samples[0].yaw,
            "head_yaw_end": self.samples[-1].yaw,
            "head_yaw_range": self.yaw_range,
            "head_pitch_start": self.samples[0].pitch,
            "head_pitch_end": self.samples[-1].pitch,
            "head_pitch_range": self.pitch_range,
            "settled": True,
        }


class MovementManager:
    """Coordinate sequential moves and robot output at 100 Hz.

    Responsibilities:
    - Own a real-time loop that samples the current primary move (if any) and calls
      `set_target` exactly once per tick.
    - Start an idle `BreathingMove` after `idle_inactivity_delay` when not
      listening and no moves are queued.
    - Expose thread-safe APIs so other threads can enqueue moves or mark activity
      without touching internal state.

    Timing:
    - All elapsed-time calculations rely on `time.monotonic()` through `self._now`
      to avoid wall-clock jumps.
    - The loop attempts 100 Hz

    Concurrency:
    - External threads communicate via `_command_queue` messages.
    """

    def __init__(
        self,
        current_robot: ReachyMini,
        stage1_instrumentation: Stage1Instrumentation | None = None,
        wake_trace: WakeTrace | None = None,
        sdk_recovery_callback: Callable[[], bool] | None = None,
        sdk_connection_probe: Callable[[], bool] | None = None,
        authorized_startup_snapshot: AuthorizedStartupSnapshot | None = None,
        startup_confirmation_at: float | None = None,
        startup_confirmation_lease_s: float = 0.25,
        startup_freshness_recorder: StartupFreshnessRecorder | None = None,
    ):
        """Initialize movement manager."""
        self.current_robot = current_robot
        self._stage1_instrumentation = stage1_instrumentation
        self._wake_trace = wake_trace
        self._head_tracking = False

        # Single timing source for durations
        self._now = time.monotonic

        # Movement state
        self.state = MovementState()
        self.state.last_activity_time = self._now()

        # Move queue (primary moves)
        self.move_queue: deque[Move] = deque()

        # Configuration
        self.idle_inactivity_delay = 0.3  # seconds
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker_lock = threading.RLock()
        self._is_listening = False
        # Speaking pauses tracking; the captured look-at pose anchors queued moves.
        self._is_speaking = False
        self._track_anchor: NDArray[np.float32] | None = None
        self._handoff_published_this_tick = False
        self._last_commanded_pose: FullBodyPose | None = None
        self._last_commanded_at: float | None = None
        self._listening_antennas: Tuple[float, float] = (0.0, 0.0)
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4  # seconds to blend back after listening
        self._last_listening_blend_time = self._now()
        self._breathing_active = False  # true when breathing move is running or queued
        self._photo_stillness = False  # suppress breathing/wobble during photo capture
        self._photo_wobble_disabled = False
        self._listening_debounce_s = 0.15
        self._last_listening_toggle_time = self._now()
        self._last_set_target_err = 0.0
        self._set_target_err_interval = 5.0
        self._set_target_err_suppressed = 0
        self._command_publication_count = 0

        self._sdk_health_lock = threading.Lock()
        self._sdk_health_started_at = self._now()
        self._sdk_control_state = "SUSPECT"
        self._sdk_last_successful_publication: float | None = None
        self._sdk_consecutive_failures = 0
        self._sdk_probe_failures = 0
        self._sdk_last_failure: str | None = None
        self._sdk_last_failure_type: str | None = None
        self._sdk_recovery_in_progress = False
        self._sdk_recovery_attempts = 0
        self._sdk_last_recovery_result: str | None = None
        self._sdk_recovery_callback = sdk_recovery_callback
        self._sdk_connection_probe = sdk_connection_probe
        self._sdk_recovery_thread: threading.Thread | None = None
        self._sdk_probe_thread: threading.Thread | None = None
        self._sdk_shutdown_event = threading.Event()

        self._stage1_calibration_lock = threading.Lock()
        self._stage1_calibration: _Stage1CalibrationRequest | None = None
        self._stage1_lifecycle_lock = threading.Lock()
        self._stage1_lifecycle: _Stage1Lifecycle | None = None
        self._stage1_preview_lock = threading.Lock()
        self._stage1_preview_baseline: _Stage1PreviewBaseline | None = None
        self._stage1_preview_resume_at: float | None = None
        self._stage1_preview_last_session_id: str | None = None
        self._stage1_preview_invalidation_reason: str | None = None
        self._stage1_preview_last_publication_count = 0
        self._sleep = time.sleep

        # Cross-thread signalling
        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()

        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening
        self._status_lock = threading.Lock()
        self._freq_stats = LoopFrequencyStats()
        self._freq_snapshot = LoopFrequencyStats()

        self._startup_baseline_adopted_at: float | None = None
        self._startup_baseline_error: str | None = None
        self._startup_breathing_pending = False
        self._startup_first_target_logged = False
        self._startup_confirmation_at = startup_confirmation_at
        self._startup_confirmation_lease_s = startup_confirmation_lease_s
        self._startup_freshness_recorder = startup_freshness_recorder
        self._startup_publication_event = threading.Event()
        self._startup_publication_succeeded = False
        self._authorized_startup_snapshot = authorized_startup_snapshot
        self._startup_observability: dict[str, object] = {
            "branch": authorized_startup_snapshot.provenance
            if authorized_startup_snapshot is not None
            else "UNAUTHORIZED_READ",
            "authorized_snapshot": (
                authorized_startup_snapshot_diagnostics(authorized_startup_snapshot)
                if authorized_startup_snapshot is not None
                else None
            ),
            "final_consistency": None,
            "first_publication": None,
            "breathing": {
                "first_activation_monotonic": None,
                "before_first_publication": "unknown",
            },
        }
        try:
            if authorized_startup_snapshot is None:
                head_joints, antennas = self.current_robot.get_current_joint_positions()
                head_pose = np.asarray(self.current_robot.get_current_head_pose(), dtype=np.float64)
            else:
                head_joints = authorized_startup_snapshot.head_joints
                antennas = authorized_startup_snapshot.antennas
                head_pose = authorized_startup_snapshot.head_pose_array()
            head_joint_values = tuple(float(value) for value in head_joints)
            antenna_values = tuple(float(value) for value in antennas)
            if head_pose.shape != (4, 4) or not np.isfinite(head_pose).all():
                raise ValueError("current head pose must be a finite 4x4 matrix")
            if not np.allclose(head_pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
                raise ValueError("current head pose must be a homogeneous transform")
            if len(head_joint_values) != 7 or not np.isfinite(head_joint_values).all():
                raise ValueError("current head joints must contain seven finite values")
            if len(antenna_values) != 2 or not np.isfinite(antenna_values).all():
                raise ValueError("current antennas must contain two finite values")
            body_yaw = (
                authorized_startup_snapshot.body_yaw
                if authorized_startup_snapshot is not None
                else head_joint_values[0]
            )
            startup_baseline = (
                head_pose.copy(),
                (antenna_values[0], antenna_values[1]),
                body_yaw,
            )
            self.state.last_primary_pose = clone_full_body_pose(startup_baseline)
            self._listening_antennas = startup_baseline[1]
            self._startup_baseline_adopted_at = self._now()
            self._startup_breathing_pending = True
            startup_angles = pose_euler_degrees(startup_baseline[0])
            logger.info(
                "Movement manager startup baseline adopted head_pose=%s head_rpy_deg=%s body_yaw=%.9f "
                "antennas=%s head_joints=%s",
                startup_baseline[0].tolist(),
                (startup_angles.roll, startup_angles.pitch, startup_angles.yaw),
                startup_baseline[2],
                startup_baseline[1],
                head_joint_values,
            )
            if self._wake_trace is not None:
                self._wake_trace.record_transition(
                    "stage1m_adoption",
                    command=WakeTraceCommand(
                        source="stage1m_adopted_baseline",
                        head=startup_baseline[0].copy(),
                        antennas=startup_baseline[1],
                        body_yaw=startup_baseline[2],
                        duration=None,
                        interpolation=None,
                    ),
                    active_task=None,
                )
        except Exception as exc:
            self._startup_baseline_error = str(exc)
            logger.error("Movement manager startup baseline unavailable: %s", exc)

    def set_sdk_recovery_callback(self, callback: Callable[[], bool] | None) -> None:
        """Set the process-scoped SDK recovery request callback."""
        with self._sdk_health_lock:
            self._sdk_recovery_callback = callback

    def get_sdk_control_status(self) -> dict[str, object]:
        """Return a thread-safe SDK control-health snapshot."""
        now = self._now()
        with self._sdk_health_lock:
            last_success = self._sdk_last_successful_publication
            return {
                "state": self._sdk_control_state,
                "last_successful_publication_age_s": (
                    round(max(0.0, now - last_success), 3) if last_success is not None else None
                ),
                "consecutive_failures": self._sdk_consecutive_failures,
                "connection_probe_failures": self._sdk_probe_failures,
                "last_failure": self._sdk_last_failure,
                "last_failure_type": self._sdk_last_failure_type,
                "recovery_in_progress": self._sdk_recovery_in_progress,
                "recovery_attempts": self._sdk_recovery_attempts,
                "last_recovery_result": self._sdk_last_recovery_result,
                "publication_count": self._command_publication_count,
            }

    def wait_for_sdk_recovery(self, timeout_s: float = 5.0) -> None:
        """Wait for the current recovery worker, if any, to finish."""
        with self._sdk_health_lock:
            thread = self._sdk_recovery_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def _record_sdk_publication_success(self, now: float) -> None:
        with self._sdk_health_lock:
            previous_state = self._sdk_control_state
            self._sdk_last_successful_publication = now
            self._sdk_consecutive_failures = 0
            self._sdk_probe_failures = 0
            if previous_state not in {"RECOVERING", "FAILED"}:
                self._sdk_control_state = "HEALTHY"
        if previous_state == "SUSPECT":
            logger.info("SDK control health recovered after a successful set_target publication")

    def _record_sdk_publication_failure(self, exc: Exception, now: float) -> None:
        transition_to_suspect = False
        begin_recovery = False
        with self._sdk_health_lock:
            if self._sdk_shutdown_event.is_set() or self._sdk_control_state in {"RECOVERING", "FAILED"}:
                return
            self._sdk_consecutive_failures += 1
            self._sdk_last_failure = str(exc)
            self._sdk_last_failure_type = type(exc).__name__
            if self._sdk_control_state == "HEALTHY":
                self._sdk_control_state = "SUSPECT"
                transition_to_suspect = True
            last_success = self._sdk_last_successful_publication or self._sdk_health_started_at
            begin_recovery = (
                self._sdk_consecutive_failures >= SDK_CONTROL_FAILURE_THRESHOLD
                and now - last_success >= SDK_CONTROL_LAST_SUCCESS_MAX_AGE_S
            )
        if transition_to_suspect:
            logger.warning("SDK control health transitioned HEALTHY -> SUSPECT after set_target failure")
        if begin_recovery:
            self._begin_sdk_recovery("persistent set_target failure")

    def _record_sdk_probe_result(self, connected: bool) -> None:
        if connected:
            with self._sdk_health_lock:
                self._sdk_probe_failures = 0
            return
        begin_recovery = False
        transition_to_suspect = False
        now = self._now()
        with self._sdk_health_lock:
            if self._sdk_shutdown_event.is_set() or self._sdk_control_state in {"RECOVERING", "FAILED"}:
                return
            self._sdk_probe_failures += 1
            self._sdk_last_failure = "SDK control heartbeat unavailable"
            self._sdk_last_failure_type = "ConnectionError"
            if self._sdk_control_state == "HEALTHY":
                self._sdk_control_state = "SUSPECT"
                transition_to_suspect = True
            last_success = self._sdk_last_successful_publication or self._sdk_health_started_at
            begin_recovery = (
                self._sdk_probe_failures >= SDK_CONTROL_PROBE_FAILURE_THRESHOLD
                and now - last_success >= SDK_CONTROL_LAST_SUCCESS_MAX_AGE_S
            )
        if transition_to_suspect:
            logger.warning("SDK control health transitioned HEALTHY -> SUSPECT after heartbeat failure")
        if begin_recovery:
            self._begin_sdk_recovery("persistent SDK heartbeat failure")

    def _begin_sdk_recovery(self, reason: str) -> None:
        with self._sdk_health_lock:
            if self._sdk_shutdown_event.is_set() or self._sdk_recovery_in_progress:
                return
            if self._sdk_control_state == "FAILED":
                return
            self._sdk_control_state = "DISCONNECTED"
            self._sdk_recovery_in_progress = True
            self._sdk_last_recovery_result = None
            self._stop_event.set()
            thread = threading.Thread(
                target=self._run_sdk_recovery,
                args=(reason,),
                daemon=True,
                name="sdk-control-recovery",
            )
            self._sdk_recovery_thread = thread
        logger.error("SDK control health transitioned to DISCONNECTED: %s", reason)
        thread.start()

    def _run_sdk_recovery(self, reason: str) -> None:
        self._establish_sdk_recovery_boundary(reason)
        with self._sdk_health_lock:
            if self._sdk_shutdown_event.is_set():
                self._sdk_recovery_in_progress = False
                self._sdk_last_recovery_result = "cancelled by application shutdown"
                return
            self._sdk_control_state = "RECOVERING"
            callback = self._sdk_recovery_callback
        logger.warning("SDK control health transitioned DISCONNECTED -> RECOVERING")

        if callback is None:
            with self._sdk_health_lock:
                self._sdk_control_state = "FAILED"
                self._sdk_recovery_in_progress = False
                self._sdk_last_recovery_result = "no safe recovery owner available"
            logger.error("SDK control recovery failed: no safe recovery owner available")
            return

        for attempt, delay_s in enumerate(SDK_RECOVERY_BACKOFF_S, start=1):
            if delay_s > 0 and self._sdk_shutdown_event.wait(delay_s):
                with self._sdk_health_lock:
                    self._sdk_recovery_in_progress = False
                    self._sdk_last_recovery_result = "cancelled by application shutdown"
                return
            with self._sdk_health_lock:
                self._sdk_recovery_attempts = attempt
            logger.warning(
                "Requesting conversation process recovery attempt %s/%s", attempt, SDK_RECOVERY_ATTEMPT_LIMIT
            )
            try:
                accepted = callback()
            except Exception as exc:
                accepted = False
                result = f"{type(exc).__name__}: {exc}"
                logger.error("Conversation process recovery attempt %s failed: %s", attempt, exc)
            else:
                result = "conversation process restart requested" if accepted else "recovery request rejected"
            with self._sdk_health_lock:
                self._sdk_last_recovery_result = result
            if accepted:
                logger.warning("Conversation process restart requested for SDK control recovery")
                return

        with self._sdk_health_lock:
            self._sdk_control_state = "FAILED"
            self._sdk_recovery_in_progress = False
        logger.error("SDK control recovery exhausted %s attempts; entering FAILED", SDK_RECOVERY_ATTEMPT_LIMIT)

    def _establish_sdk_recovery_boundary(self, reason: str) -> None:
        with self._stage1_lifecycle_lock:
            lifecycle = self._stage1_lifecycle
        if lifecycle is not None:
            self._release_stage1_calibration(
                lifecycle.session_id,
                f"SDK recovery: {reason}",
                restore_movement=False,
            )

        with self._worker_lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        with self._worker_lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None

        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            if command == "stage1_horizontal_calibration":
                _session_id, acknowledgement = payload
                acknowledgement.error = "SDK control recovery invalidated Stage 1 calibration"
                acknowledgement.event.set()
            elif command in {
                "begin_stage1_preview_stabilization",
                "end_stage1_preview_stabilization",
                "freeze_stage1_horizontal_calibration",
            }:
                acknowledgement = payload[-1]
                acknowledgement.error = "SDK control recovery invalidated Stage 1 calibration"
                acknowledgement.event.set()

        with self._stage1_calibration_lock:
            self._stage1_calibration = None
        self._cancel_stage1_preview_stabilization(f"SDK recovery: {reason}")
        self.move_queue.clear()
        self.state.current_move = None
        self.state.move_start_time = None
        self._breathing_active = False
        self._is_speaking = False
        self._track_anchor = None
        logger.warning("SDK recovery movement boundary established; queued and active movement discarded")

    def _sdk_probe_loop(self) -> None:
        while not self._sdk_shutdown_event.wait(0.25):
            with self._sdk_health_lock:
                last_success = self._sdk_last_successful_publication
                recovery_active = self._sdk_recovery_in_progress
                probe = self._sdk_connection_probe
            if recovery_active or probe is None:
                continue
            if last_success is not None and self._now() - last_success < SDK_CONTROL_PROBE_STALE_AFTER_S:
                continue
            try:
                connected = probe()
            except Exception as exc:
                logger.warning("SDK connection health probe failed: %s", exc)
                connected = False
            self._record_sdk_probe_result(connected)

    def prepare_stage1_horizontal_calibration(self, raw_geometry_delta_yaw: float) -> Stage1CalibrationPreview:
        """Prepare one bounded calibration target without moving the robot."""
        with self._stage1_calibration_lock:
            if self._stage1_calibration is not None:
                raise RuntimeError("a Stage 1 calibration session is already active")
            if self._head_tracking:
                raise RuntimeError("legacy head tracking must be disabled for Stage 1 calibration")
            if self._stage1_instrumentation is None:
                raise RuntimeError("Stage 1 instrumentation is unavailable")

        session_id = uuid.uuid4().hex
        movement_was_running = self._thread is not None and self._thread.is_alive()
        lifecycle = _Stage1Lifecycle(session_id, movement_was_running, owns_movement_freeze=True)
        with self._stage1_lifecycle_lock:
            if self._stage1_lifecycle is not None:
                raise RuntimeError("a Stage 1 calibration session is already active")
            self._stage1_lifecycle = lifecycle

        try:
            baseline = self._begin_stage1_preview_stabilization(session_id)
            settling = self._wait_for_stage1_stationary_head(session_id)
            pre_face = self._stage1_instrumentation.capture_face_window()
            self._require_stage1_preview_baseline(session_id)
            if not pre_face.stable:
                raise RuntimeError("PRECONDITION FAILED — FACE NOT STABLE")

            freeze = _Stage1FreezeAcknowledgement()
            self._command_queue.put(("freeze_stage1_horizontal_calibration", (session_id, freeze)))
            if not freeze.event.wait(2.0):
                raise TimeoutError("Stage 1 command freeze timed out")
            if freeze.error is not None:
                raise RuntimeError(freeze.error)
            lifecycle.worker_was_stopped = True
            thread = self._thread
            if thread is not None:
                thread.join(timeout=2.0)
                with self._worker_lock:
                    if self._thread is thread and not thread.is_alive():
                        self._thread = None
            if freeze.commanded_pose is None or freeze.commanded_at is None:
                raise RuntimeError("Stage 1 command baseline is unavailable")

            telemetry = self._stage1_instrumentation.read_telemetry(expected_tracking_enabled=False)

            commanded_head, antennas, body_yaw = freeze.commanded_pose
            if telemetry.target_head_pose is not None and not np.allclose(
                telemetry.target_head_pose, commanded_head, atol=1e-5
            ):
                raise RuntimeError("daemon and MovementManager command targets disagree")
            if telemetry.target_antennas is not None and not np.allclose(
                telemetry.target_antennas, antennas, atol=1e-5
            ):
                raise RuntimeError("daemon and MovementManager antenna targets disagree")
            if telemetry.target_body_yaw is not None and not np.isclose(
                telemetry.target_body_yaw, body_yaw, atol=1e-5
            ):
                raise RuntimeError("daemon and MovementManager body targets disagree")

            target = calculate_horizontal_calibration_target(
                telemetry.present_head_pose,
                commanded_head,
                raw_geometry_delta_yaw,
            )
            preview = Stage1CalibrationPreview(
                session_id=session_id,
                target=target,
                antennas=antennas,
                body_yaw=body_yaw,
                pre_face=pre_face,
                pre_telemetry=telemetry,
                settling=settling,
                breathing_was_active=baseline.breathing_was_active,
            )
            with self._stage1_calibration_lock:
                self._stage1_calibration = _Stage1CalibrationRequest(
                    session_id=session_id,
                    raw_geometry_delta_yaw=raw_geometry_delta_yaw,
                    preview=preview,
                    prepared_at=self._now(),
                )
            expiry_timer = threading.Timer(
                STAGE1_PREVIEW_MAX_AGE_S,
                self._expire_stage1_horizontal_calibration,
                args=(session_id,),
            )
            expiry_timer.daemon = True
            lifecycle.expiry_timer = expiry_timer
            expiry_timer.start()
            return preview
        except Exception:
            self._release_stage1_calibration(session_id, "prepare failed")
            raise

    def cancel_stage1_horizontal_calibration(self, session_id: str) -> None:
        """Cancel a prepared Stage 1 session and restore movement lifecycle state."""
        with self._stage1_lifecycle_lock:
            lifecycle = self._stage1_lifecycle
            if lifecycle is None or lifecycle.session_id != session_id:
                raise ValueError("stale or unknown Stage 1 calibration session")
        self._release_stage1_calibration(session_id, "cancelled")

    def _expire_stage1_horizontal_calibration(self, session_id: str) -> None:
        self._release_stage1_calibration(session_id, "expired")

    def _stage1_owns_movement_freeze(self) -> bool:
        with self._stage1_lifecycle_lock:
            lifecycle = self._stage1_lifecycle
            return lifecycle is not None and lifecycle.owns_movement_freeze

    def _release_stage1_calibration(
        self,
        session_id: str,
        reason: str,
        *,
        restore_movement: bool = True,
    ) -> None:
        with self._stage1_lifecycle_lock:
            lifecycle = self._stage1_lifecycle
            if lifecycle is None or lifecycle.session_id != session_id:
                return
            if lifecycle.releasing:
                released = lifecycle.released
            else:
                lifecycle.releasing = True
                released = None
        if released is not None:
            released.wait(timeout=5.0)
            return

        try:
            if lifecycle.expiry_timer is not None and lifecycle.expiry_timer is not threading.current_thread():
                lifecycle.expiry_timer.cancel()
            with self._stage1_calibration_lock:
                self._stage1_calibration = None

            self._stop_event.set()
            with self._worker_lock:
                thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=10.0)
            with self._worker_lock:
                if self._thread is not None and not self._thread.is_alive():
                    self._thread = None
                worker_still_alive = self._thread is not None and self._thread.is_alive()
            if worker_still_alive:
                logger.error("Stage 1 worker did not stop during lifecycle release session_id=%s", session_id)

            with self._stage1_preview_lock:
                baseline = self._stage1_preview_baseline
                if baseline is not None and baseline.owner_session_id == session_id:
                    self._stage1_preview_baseline = None
                    self._stage1_preview_resume_at = self._now() + self.idle_inactivity_delay
                    self._stage1_preview_last_session_id = session_id
                    self._stage1_preview_invalidation_reason = reason
                    with self._status_lock:
                        publication_count = self._command_publication_count
                    self._stage1_preview_last_publication_count = publication_count - baseline.publication_count

            retained_commands: list[tuple[str, Any]] = []
            while True:
                try:
                    command, payload = self._command_queue.get_nowait()
                except Empty:
                    break
                if command in {
                    "begin_stage1_preview_stabilization",
                    "end_stage1_preview_stabilization",
                    "freeze_stage1_horizontal_calibration",
                    "stage1_horizontal_calibration",
                }:
                    continue
                if command in {
                    "queue_move",
                    "clear_queue",
                    "set_moving_state",
                    "set_head_tracking",
                    "freeze_head_tracking",
                    "restore_head_tracking",
                    "hold_still_for_capture",
                    "release_capture_stillness",
                    "set_speaking",
                }:
                    continue
                retained_commands.append((command, payload))
            for retained_command in retained_commands:
                self._command_queue.put(retained_command)

            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False

            if restore_movement and lifecycle.movement_was_running:
                self.start()
            logger.info(
                "Stage 1 lifecycle released session_id=%s reason=%s movement_was_running=%s",
                session_id,
                reason,
                lifecycle.movement_was_running,
            )
        finally:
            with self._stage1_lifecycle_lock:
                if self._stage1_lifecycle is lifecycle:
                    self._stage1_lifecycle = None
                lifecycle.released.set()

    def start_stage1_preview_diagnostic(self) -> dict[str, object]:
        """Start a session-owned preview without entering calibration preparation."""
        if self._thread is None or not self._thread.is_alive():
            raise RuntimeError("MovementManager is not running")
        with self._stage1_calibration_lock:
            if self._stage1_calibration is not None:
                raise RuntimeError("a Stage 1 calibration session is already active")
        session_id = uuid.uuid4().hex
        self._begin_stage1_preview_stabilization(session_id, start_if_stopped=False)
        return self.get_stage1_preview_diagnostic_status(session_id)

    def get_stage1_preview_diagnostic_status(self, session_id: str | None = None) -> dict[str, object]:
        """Return read-only state for a preview-only diagnostic session."""
        now = self._now()
        with self._stage1_preview_lock:
            baseline = self._stage1_preview_baseline
            resume_at = self._stage1_preview_resume_at
            last_session_id = self._stage1_preview_last_session_id
            invalidation_reason = self._stage1_preview_invalidation_reason
            last_preview_publication_count = self._stage1_preview_last_publication_count
        if baseline is not None and session_id != baseline.owner_session_id:
            raise RuntimeError("Stage 1 preview baseline is owned by another session")
        if baseline is None and session_id is not None and session_id != last_session_id:
            raise RuntimeError("Stage 1 preview session is stale or unknown")

        with self._status_lock:
            last_commanded_at = self._last_commanded_at
            last_commanded_pose = (
                clone_full_body_pose(self._last_commanded_pose) if self._last_commanded_pose is not None else None
            )
            publication_count = self._command_publication_count

        baseline_age = now - baseline.captured_at if baseline is not None else None
        command_unchanged = (
            baseline is not None
            and last_commanded_at == baseline.commanded_at
            and last_commanded_pose is not None
            and full_body_pose_equal(last_commanded_pose, baseline.commanded_pose)
        )
        expired = baseline_age is not None and baseline_age > STAGE1_PREVIEW_SETTLING_TIMEOUT_S
        baseline_valid = bool(baseline is not None and baseline_age is not None and not expired and command_unchanged)
        active_invalidation_reason = (
            "timeout" if expired else "command baseline mutation" if not command_unchanged else None
        )
        baseline_pose = baseline.commanded_pose if baseline is not None else None
        baseline_angles = pose_euler_degrees(baseline_pose[0]) if baseline_pose is not None else None
        current_move = self.state.current_move
        non_preview_queue = [type(move).__name__ for move in self.move_queue if not isinstance(move, BreathingMove)]
        recovery_suppression_active = resume_at is not None and now < resume_at

        return {
            "active": baseline is not None,
            "session_id": baseline.owner_session_id if baseline is not None else None,
            "owner_id": baseline.owner_session_id if baseline is not None else None,
            "baseline_valid": baseline_valid,
            "baseline_invalidation_reason": active_invalidation_reason
            if baseline is not None
            else invalidation_reason,
            "baseline_captured_at": baseline.captured_at if baseline is not None else None,
            "source_command_timestamp": baseline.commanded_at if baseline is not None else None,
            "baseline_age": baseline_age,
            "suppression_active": baseline is not None,
            "publication_suppressed": baseline is not None or recovery_suppression_active,
            "recovery_suppression_active": recovery_suppression_active,
            "stabilization_elapsed": baseline_age,
            "stabilization_required": STAGE1_PREVIEW_SETTLING_DURATION_S,
            "stabilization_complete": bool(
                baseline_age is not None and baseline_age >= STAGE1_PREVIEW_SETTLING_DURATION_S
            ),
            "movement_manager_alive": self._thread is not None and self._thread.is_alive(),
            "current_movement_type": type(current_move).__name__ if current_move is not None else None,
            "non_preview_queue": non_preview_queue,
            "tracking_enabled": self._head_tracking,
            "breathing_active": self._breathing_active,
            "breathing_was_active": baseline.breathing_was_active if baseline is not None else None,
            "last_commanded_at": last_commanded_at,
            "command_publication_count": publication_count,
            "publication_count_at_start": baseline.publication_count if baseline is not None else None,
            "preview_publication_count": (
                publication_count - baseline.publication_count
                if baseline is not None
                else last_preview_publication_count
            ),
            "baseline_pose": {
                "head": baseline_pose[0].tolist(),
                "head_rpy_deg": {
                    "roll": baseline_angles.roll,
                    "pitch": baseline_angles.pitch,
                    "yaw": baseline_angles.yaw,
                },
                "antennas": baseline_pose[1],
                "body_yaw": baseline_pose[2],
            }
            if baseline_pose is not None and baseline_angles is not None
            else None,
            "last_commanded_pose": {
                "head": last_commanded_pose[0].tolist(),
                "antennas": last_commanded_pose[1],
                "body_yaw": last_commanded_pose[2],
            }
            if last_commanded_pose is not None
            else None,
        }

    def release_stage1_preview_diagnostic(self, session_id: str) -> dict[str, object]:
        """Release an owned preview without publishing a cleanup target."""
        self._end_stage1_preview_stabilization(session_id, "diagnostic release")
        return self.get_stage1_preview_diagnostic_status()

    def _begin_stage1_preview_stabilization(
        self,
        session_id: str,
        *,
        start_if_stopped: bool = True,
    ) -> _Stage1PreviewBaseline:
        acknowledgement = _Stage1PreviewAcknowledgement()
        worker_running = self._thread is not None and self._thread.is_alive()
        if not worker_running and not start_if_stopped:
            raise RuntimeError("MovementManager is not running")
        self._command_queue.put(("begin_stage1_preview_stabilization", (session_id, acknowledgement)))
        if not worker_running:
            self.start()
        if not acknowledgement.event.wait(2.0):
            raise TimeoutError("Stage 1 preview stabilization acknowledgement timed out")
        if acknowledgement.error is not None:
            raise RuntimeError(acknowledgement.error)
        if acknowledgement.baseline is None:
            raise RuntimeError("Stage 1 preview baseline is unavailable")
        return acknowledgement.baseline

    def _end_stage1_preview_stabilization(self, session_id: str, reason: str) -> None:
        acknowledgement = _Stage1PreviewAcknowledgement()
        if self._thread is None or not self._thread.is_alive():
            self._end_stage1_preview_stabilization_on_worker(session_id, reason, acknowledgement, self._now())
        else:
            self._command_queue.put(("end_stage1_preview_stabilization", (session_id, reason, acknowledgement)))
        if not acknowledgement.event.wait(2.0):
            raise TimeoutError("Stage 1 preview stabilization release timed out")
        if acknowledgement.error is not None:
            raise RuntimeError(acknowledgement.error)

    def _require_stage1_preview_baseline(self, session_id: str) -> _Stage1PreviewBaseline:
        with self._stage1_preview_lock:
            baseline = self._stage1_preview_baseline
        if baseline is None or baseline.owner_session_id != session_id:
            raise RuntimeError("Stage 1 preview baseline is not owned by this session")
        if self._now() - baseline.captured_at > STAGE1_PREVIEW_SETTLING_TIMEOUT_S:
            raise RuntimeError("Stage 1 preview stabilization timed out")
        with self._status_lock:
            commanded_at = self._last_commanded_at
            if self._last_commanded_pose is None:
                raise RuntimeError("Stage 1 command baseline is unavailable")
            commanded_pose = clone_full_body_pose(self._last_commanded_pose)
        if commanded_at != baseline.commanded_at or not full_body_pose_equal(commanded_pose, baseline.commanded_pose):
            raise RuntimeError("Stage 1 preview baseline changed during stabilization")
        return baseline

    def _wait_for_stage1_stationary_head(self, session_id: str) -> _Stage1SettlingResult:
        samples: list[_Stage1SettlingSample] = []
        started_at = self._now()
        while self._now() - started_at <= STAGE1_PREVIEW_SETTLING_TIMEOUT_S:
            self._require_stage1_preview_baseline(session_id)
            if self._stage1_instrumentation is None:
                raise RuntimeError("Stage 1 instrumentation is unavailable")
            telemetry = self._stage1_instrumentation.read_telemetry(expected_tracking_enabled=False)
            sampled_at = self._now()
            angles = pose_euler_degrees(telemetry.present_head_pose)
            samples.append(_Stage1SettlingSample(sampled_at, angles.yaw, angles.pitch))
            qualifying_starts = [
                index
                for index, sample in enumerate(samples)
                if sampled_at - sample.sampled_at >= STAGE1_PREVIEW_SETTLING_DURATION_S
            ]
            if qualifying_starts:
                window = samples[qualifying_starts[-1] :]
                yaw_values = [sample.yaw for sample in window]
                pitch_values = [sample.pitch for sample in window]
                yaw_range = max(yaw_values) - min(yaw_values)
                pitch_range = max(pitch_values) - min(pitch_values)
                if (
                    yaw_range <= STAGE1_PREVIEW_STATIONARY_LIMIT_DEG
                    and pitch_range <= STAGE1_PREVIEW_STATIONARY_LIMIT_DEG
                ):
                    return _Stage1SettlingResult(
                        duration=window[-1].sampled_at - window[0].sampled_at,
                        samples=tuple(window),
                        yaw_range=yaw_range,
                        pitch_range=pitch_range,
                    )
            self._sleep(STAGE1_PREVIEW_SETTLING_INTERVAL_S)
        raise RuntimeError("PRECONDITION FAILED — HEAD NOT SETTLED")

    def execute_stage1_horizontal_calibration(
        self,
        session_id: str,
        *,
        timeout_s: float = 5.0,
    ) -> Stage1CalibrationResult:
        """Consume one prepared calibration session through the movement worker."""
        with self._stage1_calibration_lock:
            request = self._stage1_calibration
            if request is None or request.session_id != session_id or request.consumed:
                raise ValueError("stale or consumed Stage 1 calibration session")
            request.consumed = True

        acknowledgement = _Stage1CalibrationAcknowledgement()
        try:
            self._command_queue.put(("stage1_horizontal_calibration", (session_id, acknowledgement)))
            if self._thread is None or not self._thread.is_alive():
                self.start()
            if not acknowledgement.event.wait(timeout_s):
                self._stop_event.set()
                raise TimeoutError("Stage 1 calibration acknowledgement timed out")

            thread = self._thread
            if thread is not None:
                thread.join(timeout=timeout_s)
                with self._worker_lock:
                    if self._thread is thread and not thread.is_alive():
                        self._thread = None
            if acknowledgement.error is not None:
                raise RuntimeError(acknowledgement.error)
            if acknowledgement.result is None:
                raise RuntimeError("Stage 1 calibration returned no acknowledgement")
            return acknowledgement.result
        finally:
            self._release_stage1_calibration(session_id, "execute complete")

    def queue_move(self, move: Move) -> None:
        """Queue a primary move to run after the currently executing one.

        Thread-safe: the move is enqueued via the worker command queue so the
        control loop remains the sole mutator of movement state.
        """
        if self._stage1_owns_movement_freeze():
            raise RuntimeError(f"{type(move).__name__} rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        """Stop the active move and discard any queued primary moves.

        Thread-safe: executed by the worker thread via the command queue.
        """
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("move queue changes are rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("clear_queue", None))

    def set_moving_state(self, duration: float) -> None:
        """Mark the robot as actively moving for the provided duration.

        Legacy hook used by goto helpers to keep inactivity and breathing logic
        aware of manual motions. Thread-safe via the command queue.
        """
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("movement state changes are rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("set_moving_state", duration))

    def is_idle(self) -> bool:
        """Return True when the robot has been inactive longer than the idle delay."""
        if self._stage1_owns_movement_freeze():
            return False
        with self._shared_state_lock:
            last_activity = self._shared_last_activity_time
            listening = self._shared_is_listening

        if listening:
            return False

        return self._now() - last_activity >= self.idle_inactivity_delay

    def set_listening(self, listening: bool) -> None:
        """Enable or disable listening mode without touching shared state directly.

        While listening:
        - Antenna positions are frozen at the last commanded values.
        - Blending is reset so that upon unfreezing the antennas return smoothly.
        - Idle breathing is suppressed.

        Thread-safe: the change is posted to the worker command queue.
        """
        with self._shared_state_lock:
            if self._shared_is_listening == listening:
                return
        self._command_queue.put(("set_listening", listening))

    def set_head_tracking(self, enabled: bool) -> None:
        """Start or stop following the user's face; thread-safe via the command queue."""
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("head tracking changes are rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("set_head_tracking", enabled))

    def get_head_tracking_enabled(self) -> bool:
        """Return whether head tracking is currently enabled."""
        return self._head_tracking

    def freeze_head_tracking(self) -> None:
        """Hold the head still by setting tracking weight to 0 while leaving tracking enabled."""
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("head tracking changes are rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("freeze_head_tracking", None))

    def restore_head_tracking(self, was_enabled: bool) -> None:
        """Restore the exact prior tracking enabled/disabled state after a temporary freeze."""
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("head tracking changes are rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("restore_head_tracking", bool(was_enabled)))

    def hold_still_for_capture(self) -> None:
        """Stop breathing/moves and pause daemon wobble for a short photo capture."""
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("capture movement is rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("hold_still_for_capture", None))

    def release_capture_stillness(self) -> None:
        """Allow breathing again and re-enable daemon wobble after photo capture."""
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("capture movement is rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("release_capture_stillness", None))

    def set_speaking(self, speaking: bool) -> None:
        """Pause head tracking while the assistant speaks, resume it afterwards.

        On speaking, the current look-at pose is captured as an anchor so queued
        moves stay on the user; when speaking stops the head is handed back to
        daemon tracking. Thread-safe: posted to the worker command queue.
        """
        if self._stage1_owns_movement_freeze():
            raise RuntimeError("speaking movement is rejected while Stage 1 owns the movement freeze")
        self._command_queue.put(("set_speaking", speaking))

    def _poll_signals(self, current_time: float) -> None:
        """Apply queued commands."""
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)
            if self._stop_event.is_set():
                break

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        """Handle a single cross-thread command."""
        if self._stage1_owns_movement_freeze() and command in {
            "queue_move",
            "clear_queue",
            "set_moving_state",
            "set_head_tracking",
            "freeze_head_tracking",
            "restore_head_tracking",
            "hold_still_for_capture",
            "release_capture_stillness",
            "set_speaking",
        }:
            logger.warning("Suppressed %s while Stage 1 owns the movement freeze", command)
            return
        if command in {"queue_move", "clear_queue", "set_moving_state", "hold_still_for_capture"} or (
            command == "set_head_tracking" and bool(payload)
        ):
            self._cancel_stage1_preview_stabilization("unexpected movement request")
        if command in {"queue_move", "set_moving_state"} or (command == "set_head_tracking" and bool(payload)):
            self._startup_breathing_pending = False
        if command == "queue_move":
            if isinstance(payload, Move):
                self.move_queue.append(payload)
                self.state.update_activity()
                duration = getattr(payload, "duration", None)
                if duration is not None:
                    try:
                        duration_str = f"{float(duration):.2f}"
                    except (TypeError, ValueError):
                        duration_str = str(duration)
                else:
                    duration_str = "?"
                logger.debug(
                    "Queued move with duration %ss, queue size: %s",
                    duration_str,
                    len(self.move_queue),
                )
            else:
                logger.warning("Ignored queue_move command with invalid payload: %s", payload)
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.info("Cleared move queue and stopped current move")
        elif command == "set_moving_state":
            try:
                duration = float(payload)
            except (TypeError, ValueError):
                logger.warning("Invalid moving state duration: %s", payload)
                return
            self.state.update_activity()
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired_state = bool(payload)
            now = self._now()
            if now - self._last_listening_toggle_time < self._listening_debounce_s:
                return
            self._last_listening_toggle_time = now

            if self._is_listening == desired_state:
                return

            self._is_listening = desired_state
            self._last_listening_blend_time = now
            if desired_state:
                # Freeze: snapshot current commanded antennas and reset blend
                antenna_pose = self._last_commanded_pose or self.state.last_primary_pose
                if antenna_pose is None:
                    logger.error("Cannot enter listening mode without a valid movement baseline")
                    return
                self._listening_antennas = (
                    float(antenna_pose[1][0]),
                    float(antenna_pose[1][1]),
                )
                self._antenna_unfreeze_blend = 0.0
            else:
                # Unfreeze: restart blending from frozen pose
                self._antenna_unfreeze_blend = 0.0
            self.state.update_activity()
        elif command == "set_head_tracking":
            enabled = bool(payload)
            if self._head_tracking == enabled:
                return
            try:
                if enabled:
                    self.current_robot.start_head_tracking(weight=1.0)
                else:
                    self.current_robot.stop_head_tracking()
            except Exception as e:
                logger.warning("Head-tracking toggle failed: %s", e)
                return
            self._head_tracking = enabled
            self._track_anchor = None
            # set_speaking is gated off while disabled, so its state would go stale across a toggle.
            self._is_speaking = False
        elif command == "freeze_head_tracking":
            if not self._head_tracking:
                return
            try:
                self.current_robot.start_head_tracking(weight=0.0)
            except Exception as e:
                logger.warning("Head-tracking freeze failed: %s", e)
        elif command == "restore_head_tracking":
            was_enabled = bool(payload)
            self._track_anchor = None
            self._is_speaking = False
            try:
                if was_enabled:
                    self._head_tracking = True
                    self.current_robot.start_head_tracking(weight=1.0)
                else:
                    self._head_tracking = False
                    self.current_robot.stop_head_tracking()
            except Exception as e:
                logger.warning("Head-tracking restore failed: %s", e)
        elif command == "hold_still_for_capture":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            self._photo_stillness = True
            self._photo_wobble_disabled = False
            self.state.update_activity()
            disable_wobble = getattr(self.current_robot, "disable_wobbling", None)
            if disable_wobble is not None:
                try:
                    disable_wobble()
                    self._photo_wobble_disabled = True
                except Exception as e:
                    logger.warning("Failed to disable wobbling for photo capture: %s", e)
            logger.info("Photo capture stillness enabled")
        elif command == "release_capture_stillness":
            self._photo_stillness = False
            self.state.update_activity()
            if self._photo_wobble_disabled:
                enable_wobble = getattr(self.current_robot, "enable_wobbling", None)
                if enable_wobble is not None:
                    try:
                        enable_wobble()
                    except Exception as e:
                        logger.warning("Failed to re-enable wobbling after photo capture: %s", e)
                self._photo_wobble_disabled = False
            logger.info("Photo capture stillness released")
        elif command == "set_speaking":
            if not self._head_tracking:
                return
            speaking = bool(payload)
            if self._is_speaking == speaking:
                return
            if speaking:
                if self._stage1_instrumentation is None:
                    logger.warning("Head-tracking speaking handoff rejected: fresh physical telemetry is unavailable")
                    return
                try:
                    if not self.current_robot.get_tracked_face(wait=False).detected:
                        # Keep tracking active so speech cannot block initial face acquisition.
                        logger.info("Head-tracking speaking handoff skipped: no face is locked")
                        return
                    telemetry = self._stage1_instrumentation.read_telemetry(expected_tracking_enabled=True)
                    captured_at = self._now()
                    with self._status_lock:
                        commanded_pose = (
                            clone_full_body_pose(self._last_commanded_pose)
                            if self._last_commanded_pose is not None
                            else None
                        )
                        commanded_at = self._last_commanded_at
                    if (
                        commanded_pose is None
                        or commanded_at is None
                        or captured_at - commanded_at > STAGE1_COMMAND_FRESHNESS_S
                    ):
                        logger.warning("Head-tracking speaking handoff rejected: app command baseline is unavailable or stale")
                        return

                    anchor: NDArray[np.float32] = np.asarray(telemetry.present_head_pose, dtype=np.float32).copy()
                    _, antennas, body_yaw = commanded_pose
                    if self._now() - telemetry.acquisition_monotonic > STAGE1_TELEMETRY_FRESHNESS_S:
                        logger.warning("Head-tracking speaking handoff rejected: telemetry authorization lease expired")
                        return
                    if not self._issue_control_command(anchor, antennas, body_yaw):
                        logger.warning("Head-tracking speaking handoff rejected: anchor base publication failed")
                        return
                    if isinstance(self.state.current_move, BreathingMove):
                        self.state.current_move = None
                        self.state.move_start_time = None
                    self.move_queue = deque(move for move in self.move_queue if not isinstance(move, BreathingMove))
                    self._breathing_active = False
                    self.state.last_primary_pose = clone_full_body_pose((anchor, antennas, body_yaw))
                    self._track_anchor = anchor
                    self._handoff_published_this_tick = True
                    self.current_robot.start_head_tracking(weight=0.0)
                except (RuntimeError, ValueError, AssertionError) as e:
                    self._track_anchor = None
                    logger.warning("Head-tracking speaking handoff rejected: %s", e)
                    return
                except Exception as e:
                    self._track_anchor = None
                    logger.warning("Head-tracking speaking handoff failed: %s", e)
                    return

                self._is_speaking = True
                self.state.update_activity()
                anchor_angles = pose_euler_degrees(anchor)
                base_angles = (
                    pose_euler_degrees(telemetry.target_head_pose)
                    if telemetry.target_head_pose is not None
                    else None
                )
                logger.info(
                    "Head-tracking speaking handoff anchored before pause "
                    "daemon_timestamp=%s acquisition_elapsed_ms=%.1f "
                    "physical_rpy_deg=(%.3f,%.3f,%.3f) prior_base_rpy_deg=%s",
                    telemetry.daemon_timestamp,
                    telemetry.total_acquisition_elapsed_s * 1000.0,
                    anchor_angles.roll,
                    anchor_angles.pitch,
                    anchor_angles.yaw,
                    (
                        f"({base_angles.roll:.3f},{base_angles.pitch:.3f},{base_angles.yaw:.3f})"
                        if base_angles is not None
                        else "unavailable"
                    ),
                )
            else:
                try:
                    # Restore tracking while the synchronized anchor still owns the app base.
                    self.current_robot.start_head_tracking(weight=1.0)
                except Exception as e:
                    logger.warning("Head-tracking speaking restoration failed: %s", e)
                    return
                self._is_speaking = False
                self._track_anchor = None
                self.state.update_activity()
        elif command == "begin_stage1_preview_stabilization":
            session_id, acknowledgement = payload
            self._begin_stage1_preview_stabilization_on_worker(session_id, acknowledgement, current_time)
        elif command == "end_stage1_preview_stabilization":
            session_id, reason, acknowledgement = payload
            self._end_stage1_preview_stabilization_on_worker(session_id, reason, acknowledgement, current_time)
        elif command == "freeze_stage1_horizontal_calibration":
            session_id, acknowledgement = payload
            self._freeze_stage1_horizontal_calibration(session_id, acknowledgement, current_time)
        elif command == "stage1_horizontal_calibration":
            session_id, acknowledgement = payload
            self._execute_stage1_horizontal_calibration(session_id, acknowledgement)
        else:
            logger.warning("Unknown command received by MovementManager: %s", command)

    def _begin_stage1_preview_stabilization_on_worker(
        self,
        session_id: str,
        acknowledgement: _Stage1PreviewAcknowledgement,
        current_time: float,
    ) -> None:
        try:
            with self._stage1_preview_lock:
                if self._stage1_preview_baseline is not None:
                    raise RuntimeError("a Stage 1 preview stabilization is already active")
            if self._head_tracking:
                raise RuntimeError("legacy head tracking must be disabled for Stage 1 calibration")
            if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
                raise RuntimeError("another primary movement is active")
            if any(not isinstance(move, BreathingMove) for move in self.move_queue):
                raise RuntimeError("another primary movement is queued")
            with self._status_lock:
                if self._last_commanded_at is None:
                    raise RuntimeError("Stage 1 command baseline has never been published")
                if current_time - self._last_commanded_at > STAGE1_COMMAND_FRESHNESS_S:
                    raise RuntimeError("Stage 1 command baseline is stale")
                if self._last_commanded_pose is None:
                    raise RuntimeError("Stage 1 command baseline is unavailable")
                commanded_pose = clone_full_body_pose(self._last_commanded_pose)
                commanded_at = self._last_commanded_at
                publication_count = self._command_publication_count
            baseline = _Stage1PreviewBaseline(
                owner_session_id=session_id,
                commanded_pose=commanded_pose,
                commanded_at=commanded_at,
                captured_at=current_time,
                breathing_was_active=self._breathing_active,
                publication_count=publication_count,
            )
            with self._stage1_preview_lock:
                self._stage1_preview_baseline = baseline
                self._stage1_preview_resume_at = None
                self._stage1_preview_invalidation_reason = None
            if isinstance(self.state.current_move, BreathingMove):
                self.state.current_move = None
                self.state.move_start_time = None
            self.move_queue.clear()
            self._breathing_active = False
            acknowledgement.baseline = baseline
            logger.info("Stage 1 preview stabilization started session_id=%s", session_id)
        except RuntimeError as exc:
            acknowledgement.error = str(exc)
        finally:
            acknowledgement.event.set()

    def _end_stage1_preview_stabilization_on_worker(
        self,
        session_id: str,
        reason: str,
        acknowledgement: _Stage1PreviewAcknowledgement,
        current_time: float,
    ) -> None:
        with self._stage1_preview_lock:
            baseline = self._stage1_preview_baseline
            if baseline is not None and baseline.owner_session_id == session_id:
                with self._status_lock:
                    publication_count = self._command_publication_count
                self._stage1_preview_baseline = None
                self._stage1_preview_resume_at = current_time + self.idle_inactivity_delay
                self._stage1_preview_last_session_id = session_id
                self._stage1_preview_invalidation_reason = reason
                self._stage1_preview_last_publication_count = publication_count - baseline.publication_count
                self.state.last_activity_time = current_time
                logger.info("Stage 1 preview stabilization released: %s", reason)
            elif baseline is not None:
                acknowledgement.error = "Stage 1 preview baseline is owned by another session"
            else:
                acknowledgement.error = "Stage 1 preview session is stale or inactive"
        acknowledgement.event.set()

    def _cancel_stage1_preview_stabilization(self, reason: str) -> None:
        with self._stage1_preview_lock:
            baseline = self._stage1_preview_baseline
            had_preview_state = self._stage1_preview_baseline is not None or self._stage1_preview_resume_at is not None
            self._stage1_preview_baseline = None
            self._stage1_preview_resume_at = None
            if baseline is not None:
                with self._status_lock:
                    publication_count = self._command_publication_count
                self._stage1_preview_last_session_id = baseline.owner_session_id
                self._stage1_preview_invalidation_reason = reason
                self._stage1_preview_last_publication_count = publication_count - baseline.publication_count
        if had_preview_state:
            logger.info("Stage 1 preview stabilization cancelled: %s", reason)

    def _stage1_preview_suppresses_publication(self, current_time: float) -> bool:
        with self._stage1_preview_lock:
            baseline = self._stage1_preview_baseline
            resume_at = self._stage1_preview_resume_at
        if baseline is not None:
            with self._status_lock:
                commanded_at = self._last_commanded_at
                commanded_pose = (
                    clone_full_body_pose(self._last_commanded_pose) if self._last_commanded_pose is not None else None
                )
                publication_count = self._command_publication_count
            expired = current_time - baseline.captured_at > STAGE1_PREVIEW_SETTLING_TIMEOUT_S
            mutated = (
                commanded_at != baseline.commanded_at
                or commanded_pose is None
                or not full_body_pose_equal(commanded_pose, baseline.commanded_pose)
            )
            if expired or mutated:
                reason = "timeout" if expired else "command baseline mutation"
                with self._stage1_preview_lock:
                    if self._stage1_preview_baseline is baseline:
                        self._stage1_preview_baseline = None
                        self._stage1_preview_resume_at = current_time + self.idle_inactivity_delay
                        self._stage1_preview_last_session_id = baseline.owner_session_id
                        self._stage1_preview_invalidation_reason = reason
                        self._stage1_preview_last_publication_count = publication_count - baseline.publication_count
                        self.state.last_activity_time = current_time
                logger.warning("Stage 1 preview stabilization invalidated: %s", reason)
            return True
        if resume_at is None:
            return False
        if current_time < resume_at:
            return True
        with self._stage1_preview_lock:
            if self._stage1_preview_resume_at == resume_at:
                self._stage1_preview_resume_at = None
        return False

    def _freeze_stage1_horizontal_calibration(
        self,
        session_id: str,
        acknowledgement: _Stage1FreezeAcknowledgement,
        current_time: float,
    ) -> None:
        """Freeze the worker at its latest published command without republishing."""
        acquired = False
        try:
            with self._stage1_lifecycle_lock:
                lifecycle = self._stage1_lifecycle
                if lifecycle is not None and (lifecycle.session_id != session_id or lifecycle.releasing):
                    raise RuntimeError("Stage 1 lifecycle is stale or inactive")
            baseline = self._require_stage1_preview_baseline(session_id)
            with self._status_lock:
                if (
                    self._last_commanded_at != baseline.commanded_at
                    or self._last_commanded_pose is None
                    or not full_body_pose_equal(self._last_commanded_pose, baseline.commanded_pose)
                ):
                    raise RuntimeError("Stage 1 preview baseline changed during stabilization")
            acknowledgement.commanded_pose = clone_full_body_pose(baseline.commanded_pose)
            acknowledgement.commanded_at = baseline.commanded_at
            acknowledgement.breathing_was_active = baseline.breathing_was_active
            with self._stage1_preview_lock:
                self._stage1_preview_baseline = None
                self._stage1_preview_resume_at = None
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            acquired = True
        except RuntimeError as exc:
            acknowledgement.error = str(exc)
            self._end_stage1_preview_stabilization_on_worker(
                session_id,
                "freeze rejected",
                _Stage1PreviewAcknowledgement(),
                current_time,
            )
        finally:
            if acquired:
                self._stop_event.set()
            acknowledgement.event.set()

    def _execute_stage1_horizontal_calibration(
        self,
        session_id: str,
        acknowledgement: _Stage1CalibrationAcknowledgement,
    ) -> None:
        """Issue one prepared calibration command, then stop the worker."""
        try:
            with self._stage1_calibration_lock:
                request = self._stage1_calibration
                if request is None or request.session_id != session_id or not request.consumed:
                    acknowledgement.error = "stale or unconsumed Stage 1 calibration session"
                    return
                if self._now() - request.prepared_at > STAGE1_PREVIEW_MAX_AGE_S:
                    raise ValueError("Stage 1 calibration preview has expired")

            if self._stage1_instrumentation is None:
                raise RuntimeError("Stage 1 instrumentation is unavailable")
            execution_pre_face = self._stage1_instrumentation.capture_face_window()
            if not execution_pre_face.stable:
                raise ValueError("PRECONDITION FAILED - FACE NOT STABLE")
            execution_pre_telemetry = self._stage1_instrumentation.read_telemetry(expected_tracking_enabled=False)

            preview = request.preview
            prepared_face = preview.pre_face
            if (
                prepared_face.median_u is None
                or prepared_face.median_v is None
                or prepared_face.frame_width is None
                or prepared_face.frame_height is None
                or execution_pre_face.median_u is None
                or execution_pre_face.median_v is None
                or execution_pre_face.frame_width is None
                or execution_pre_face.frame_height is None
            ):
                raise ValueError("PRECONDITION FAILED — FACE POSITION UNAVAILABLE")
            if (
                execution_pre_face.frame_width != prepared_face.frame_width
                or execution_pre_face.frame_height != prepared_face.frame_height
            ):
                raise ValueError("PRECONDITION FAILED — FACE FRAME SIZE CHANGED")
            delta_u = abs(execution_pre_face.median_u - prepared_face.median_u)
            delta_v = abs(execution_pre_face.median_v - prepared_face.median_v)
            limit_u = prepared_face.frame_width * STAGE1_FACE_POSITION_TOLERANCE_RATIO
            limit_v = prepared_face.frame_height * STAGE1_FACE_POSITION_TOLERANCE_RATIO
            if delta_u > limit_u or delta_v > limit_v:
                raise ValueError(
                    "PRECONDITION FAILED — FACE POSITION CHANGED "
                    f"(delta_u={delta_u:.3f}px limit_u={limit_u:.3f}px, "
                    f"delta_v={delta_v:.3f}px limit_v={limit_v:.3f}px)"
                )
            with self._status_lock:
                if self._last_commanded_pose is None:
                    raise RuntimeError("Stage 1 command baseline is unavailable")
                active_command = clone_full_body_pose(self._last_commanded_pose)
            if not np.allclose(active_command[0], preview.target.commanded_pose, atol=1e-5):
                raise ValueError("commanded head target changed after the Stage 1 preview")
            if not np.allclose(active_command[1], preview.antennas, atol=1e-5):
                raise ValueError("commanded antenna target changed after the Stage 1 preview")
            if not np.isclose(active_command[2], preview.body_yaw, atol=1e-5):
                raise ValueError("commanded body target changed after the Stage 1 preview")

            present_drift = angular_delta(
                preview.target.current_angles,
                pose_euler_degrees(execution_pre_telemetry.present_head_pose),
            )
            if (
                abs(present_drift.yaw) > STAGE1_PREVIEW_POSE_TOLERANCE_DEG
                or abs(present_drift.pitch) > STAGE1_PREVIEW_POSE_TOLERANCE_DEG
            ):
                raise ValueError("present head pose changed after the Stage 1 calibration preview")
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            command_head = preview.target.target_pose.astype(np.float32)
            with self._stage1_lifecycle_lock:
                lifecycle = self._stage1_lifecycle
                if lifecycle is None or lifecycle.session_id != session_id or lifecycle.releasing:
                    raise RuntimeError("Stage 1 lifecycle is stale or inactive")
                if self._now() - execution_pre_telemetry.acquisition_monotonic > STAGE1_TELEMETRY_FRESHNESS_S:
                    raise RuntimeError("Stage 1 telemetry authorization lease expired")
                accepted = self._issue_control_command(command_head, preview.antennas, preview.body_yaw)
            if not accepted:
                raise RuntimeError("Stage 1 calibration command publication failed")
            execution_post_telemetry = None
            execution_post_face = None
            face_displacement = None
            post_capture_error = None
            if accepted:
                self.state.last_primary_pose = (command_head.copy(), preview.antennas, preview.body_yaw)
                time.sleep(STAGE1_SETTLING_S)
                try:
                    execution_post_telemetry = self._stage1_instrumentation.read_telemetry(expected_tracking_enabled=False)
                    execution_post_face = self._stage1_instrumentation.capture_face_window()
                    if execution_post_face.stable:
                        face_displacement = compare_face_windows(execution_pre_face, execution_post_face)
                    else:
                        post_capture_error = "post-motion face window was unstable"
                        logger.warning("Stage 1 post-motion face window was unstable")
                except (RuntimeError, ValueError) as exc:
                    post_capture_error = str(exc)
                    logger.warning("Stage 1 post-motion instrumentation failed: %s", exc)
            acknowledgement.result = Stage1CalibrationResult(
                preview=preview,
                accepted=accepted,
                command_count=1 if accepted else 0,
                execution_pre_face=execution_pre_face,
                execution_post_face=execution_post_face,
                execution_pre_telemetry=execution_pre_telemetry,
                execution_post_telemetry=execution_post_telemetry,
                face_displacement=face_displacement,
                post_capture_error=post_capture_error,
            )
            logger.info(
                "[FACE_TRACK_STAGE1] session=%s direction=%s commanded_yaw=%.3f delta_yaw=%.3f "
                "target_yaw=%.3f pitch_hold=%.3f body_yaw_hold=%.3f command_count=%s accepted=%s",
                session_id,
                "left" if preview.target.clamped_delta_yaw > 0.0 else "right",
                preview.target.commanded_angles.yaw,
                preview.target.clamped_delta_yaw,
                preview.target.target_angles.yaw,
                preview.target.target_angles.pitch,
                preview.body_yaw,
                acknowledgement.result.command_count,
                accepted,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            acknowledgement.error = str(exc)
            logger.warning("Stage 1 calibration rejected: %s", exc)
        finally:
            self._stop_event.set()
            acknowledgement.event.set()

    def _publish_shared_state(self) -> None:
        """Expose idle-related state for external threads."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        """Manage the primary move queue (sequential execution)."""
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None

            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                # Any real move cancels breathing mode flag
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)
                logger.debug(f"Starting new move, duration: {self.state.current_move.duration}s")

    def _manage_breathing(self, current_time: float) -> None:
        """Manage automatic breathing when idle."""
        if (
            self.state.current_move is None
            and not self.move_queue
            and not self._is_listening
            and not self._is_speaking
            and not self._photo_stillness
            and not self._breathing_active
        ):
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    # These 2 functions return the latest available sensor data from the robot, but don't perform I/O synchronously.
                    # Therefore, we accept calling them inside the control loop.
                    _, current_antennas = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()

                    breathing_base = self.state.last_primary_pose if self._startup_breathing_pending else None

                    if self._wake_trace is not None:
                        base_pose = breathing_base or (
                            current_head_pose,
                            (float(current_antennas[0]), float(current_antennas[1])),
                            0.0,
                        )
                        self._wake_trace.record_transition(
                            "breathing_eligible",
                            command=WakeTraceCommand(
                                source="breathing_base",
                                head=np.asarray(base_pose[0], dtype=np.float64).copy(),
                                antennas=base_pose[1],
                                body_yaw=base_pose[2],
                                duration=None,
                                interpolation=None,
                            ),
                            active_task="BreathingMove",
                        )

                    self._breathing_active = True
                    self._startup_breathing_pending = False
                    breathing = self._startup_observability.get("breathing")
                    if isinstance(breathing, dict) and breathing.get("first_activation_monotonic") is None:
                        activation_at = self._now()
                        breathing["first_activation_monotonic"] = activation_at
                        breathing["before_first_publication"] = (
                            self._startup_observability.get("first_publication") is None
                        )
                        logger.info(
                            "BREATHING_STARTUP_ATTRIBUTION first_activation_monotonic=%.6f "
                            "before_first_publication=%s",
                            activation_at,
                            breathing["before_first_publication"],
                        )
                    self.state.update_activity()

                    breathing_move = BreathingMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        interpolation_duration=1.0,
                        base_head_pose=breathing_base[0] if breathing_base is not None else None,
                        base_antennas=breathing_base[1] if breathing_base is not None else None,
                        base_body_yaw=breathing_base[2] if breathing_base is not None else 0.0,
                    )
                    self.move_queue.append(breathing_move)
                    logger.debug("Started breathing after %.1fs of inactivity", idle_for)
                except Exception as e:
                    self._breathing_active = False
                    logger.error("Failed to start breathing: %s", e)

        if isinstance(self.state.current_move, BreathingMove) and self.move_queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
            logger.debug("Stopping breathing due to new move activity")

        if self.state.current_move is not None and not isinstance(self.state.current_move, BreathingMove):
            self._breathing_active = False

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Get the primary full body pose from the current move or prior pose."""
        # When a primary move is playing, sample it and cache the resulting pose
        if self.state.current_move is not None and self.state.move_start_time is not None:
            move_time = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(move_time)

            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([-0.1745, 0.1745])  # ~10° offset
            if body_yaw is None:
                body_yaw = 0.0

            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            head_copy = head.copy()
            primary_full_body_pose = (
                head_copy,
                antennas_tuple,
                float(body_yaw),
            )

            self.state.last_primary_pose = clone_full_body_pose(primary_full_body_pose)
        # Otherwise reuse the last primary pose so we avoid jumps between moves
        elif self.state.last_primary_pose is not None:
            primary_full_body_pose = clone_full_body_pose(self.state.last_primary_pose)
        else:
            raise RuntimeError("Movement manager has no valid startup baseline")

        # Speaking pauses tracking: hold the look-at anchor, overlay emotions on it, dance from neutral.
        if self._track_anchor is not None:
            head_pose, antennas, body_yaw = primary_full_body_pose
            move = self.state.current_move
            if move is None:
                head_pose = self._track_anchor.copy()
            elif isinstance(move, EmotionQueueMove):
                head_pose = compose_world_offset(self._track_anchor, head_pose)
            primary_full_body_pose = (head_pose, antennas, body_yaw)

        return primary_full_body_pose

    def _update_primary_motion(self, current_time: float) -> None:
        """Advance queue state and idle behaviours for this tick."""
        self._manage_move_queue(current_time)
        self._manage_breathing(current_time)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        """Blend target antennas with listening freeze state and update blending."""
        now = self._now()
        listening = self._is_listening
        listening_antennas = self._listening_antennas
        blend = self._antenna_unfreeze_blend
        blend_duration = self._antenna_blend_duration
        last_update = self._last_listening_blend_time
        self._last_listening_blend_time = now

        if listening:
            antennas_cmd = listening_antennas
            new_blend = 0.0
        else:
            dt = max(0.0, now - last_update)
            if blend_duration <= 0:
                new_blend = 1.0
            else:
                new_blend = min(1.0, blend + dt / blend_duration)
            antennas_cmd = (
                listening_antennas[0] * (1.0 - new_blend) + target_antennas[0] * new_blend,
                listening_antennas[1] * (1.0 - new_blend) + target_antennas[1] * new_blend,
            )

        if listening:
            self._antenna_unfreeze_blend = 0.0
        else:
            self._antenna_unfreeze_blend = new_blend
            if new_blend >= 1.0:
                self._listening_antennas = (
                    float(target_antennas[0]),
                    float(target_antennas[1]),
                )

        return antennas_cmd

    def _block_expired_startup_publication(self, checked_at: float) -> None:
        confirmation_at = self._startup_confirmation_at
        if confirmation_at is None:
            raise RuntimeError("startup confirmation timestamp disappeared")
        authorization_age_ms = (checked_at - confirmation_at) * 1000.0
        lease_limit_ms = self._startup_confirmation_lease_s * 1000.0
        self._startup_baseline_error = "startup confirmation lease expired before first publication"
        self._startup_observability["first_publication"] = {
            "diagnostic_status": "BLOCKED",
            "publication_monotonic": None,
            "authorization_checked_monotonic": checked_at,
            "publication_authorization_source": "final_consistency_confirmation",
            "publication_authorization_monotonic": confirmation_at,
            "publication_authorization_age_ms": authorization_age_ms,
            "publication_authorization_lease_limit_ms": lease_limit_ms,
            "publication_authorization_lease_result": "FAIL",
        }
        logger.error(
            "FIRST_STARTUP_PUBLICATION_BLOCKED publication_authorization_source="
            "final_consistency_confirmation publication_authorization_monotonic=%.6f "
            "authorization_checked_monotonic=%.6f publication_authorization_age_ms=%.3f "
            "publication_authorization_lease_limit_ms=%.3f publication_authorization_lease_result=FAIL",
            confirmation_at,
            checked_at,
            authorization_age_ms,
            lease_limit_ms,
        )

    def _issue_control_command(
        self, head: NDArray[np.float32], antennas: Tuple[float, float], body_yaw: float
    ) -> bool:
        """Send the pose to the robot with throttled error logging."""
        self._cancel_stage1_preview_stabilization("control command issued")
        first_target = not self._startup_first_target_logged
        publication_monotonic = self._now()
        if first_target and self._startup_confirmation_at is not None:
            if publication_monotonic - self._startup_confirmation_at > self._startup_confirmation_lease_s:
                self._block_expired_startup_publication(publication_monotonic)
                return False
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
        except Exception as e:
            now = self._now()
            if now - self._last_set_target_err >= self._set_target_err_interval:
                msg = f"Failed to set robot target: {e}"
                if self._set_target_err_suppressed:
                    msg += f" (suppressed {self._set_target_err_suppressed} repeats)"
                    self._set_target_err_suppressed = 0
                logger.error(msg)
                self._last_set_target_err = now
            else:
                self._set_target_err_suppressed += 1
            self._record_sdk_publication_failure(e, now)
            return False
        else:
            now = self._now()
            with self._status_lock:
                self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))
                self._last_commanded_at = now
                self._command_publication_count += 1
            self._record_sdk_publication_success(now)
            if first_target:
                self._record_first_publication_diagnostics(
                    head,
                    antennas,
                    body_yaw,
                    publication_monotonic=publication_monotonic,
                )
            if first_target and self._wake_trace is not None:
                self._wake_trace.record_transition(
                    "movement_manager_first_publication",
                    command=WakeTraceCommand(
                        source="movement_manager_first_target",
                        head=np.asarray(head, dtype=np.float64).copy(),
                        antennas=antennas,
                        body_yaw=body_yaw,
                        duration=None,
                        interpolation=None,
                    ),
                    active_task="MovementManager",
                )
            return True

    def _record_first_publication_diagnostics(
        self,
        head: NDArray[np.float32],
        antennas: Tuple[float, float],
        body_yaw: float,
        *,
        publication_monotonic: float,
    ) -> None:
        try:
            target_angles = pose_euler_degrees(head)
            authorized_angles = None
            delta_rpy_deg = None
            matches_authorized: bool | None = None
            authorized_age_ms = None
            if self._authorized_startup_snapshot is not None:
                authorized_pose = self._authorized_startup_snapshot.head_pose_array()
                authorized_pose_angles = pose_euler_degrees(authorized_pose)
                authorized_angles = (
                    authorized_pose_angles.roll,
                    authorized_pose_angles.pitch,
                    authorized_pose_angles.yaw,
                )
                delta_rpy_deg = (
                    target_angles.roll - authorized_pose_angles.roll,
                    target_angles.pitch - authorized_pose_angles.pitch,
                    target_angles.yaw - authorized_pose_angles.yaw,
                )
                authorized_age_ms = (
                    publication_monotonic - self._authorized_startup_snapshot.authorized_monotonic
                ) * 1000.0
                matches_authorized = (
                    np.allclose(head, authorized_pose, rtol=0.0, atol=1e-12)
                    and max(
                        abs(value - expected)
                        for value, expected in zip(antennas, self._authorized_startup_snapshot.antennas)
                    )
                    <= 1e-12
                    and abs(body_yaw - self._authorized_startup_snapshot.body_yaw) <= 1e-12
                    and max(abs(value) for value in delta_rpy_deg) <= _STARTUP_ATTRIBUTION_TOLERANCE_DEG
                )
            lease_age_ms = None
            lease_result = "NO_EXISTING_LEASE"
            if self._startup_confirmation_at is not None:
                lease_age_ms = (publication_monotonic - self._startup_confirmation_at) * 1000.0
                lease_result = "PASS" if lease_age_ms <= self._startup_confirmation_lease_s * 1000.0 else "FAIL"
            first_publication_diagnostics = {
                "diagnostic_status": "PASS",
                "head_rpy_deg": [target_angles.roll, target_angles.pitch, target_angles.yaw],
                "authorized_head_rpy_deg": list(authorized_angles) if authorized_angles is not None else None,
                "delta_rpy_deg": list(delta_rpy_deg) if delta_rpy_deg is not None else None,
                "publication_monotonic": publication_monotonic,
                "target_snapshot_authorized_monotonic": (
                    self._authorized_startup_snapshot.authorized_monotonic
                    if self._authorized_startup_snapshot is not None
                    else None
                ),
                "target_snapshot_age_ms": authorized_age_ms,
                "publication_authorization_source": (
                    "final_consistency_confirmation" if self._startup_confirmation_at is not None else None
                ),
                "publication_authorization_monotonic": self._startup_confirmation_at,
                "publication_authorization_age_ms": lease_age_ms,
                "publication_authorization_lease_limit_ms": self._startup_confirmation_lease_s * 1000.0
                if self._startup_confirmation_at is not None
                else None,
                "publication_authorization_lease_result": lease_result,
                "matches_authorized": matches_authorized,
                "equality_tolerance_deg": _STARTUP_ATTRIBUTION_TOLERANCE_DEG,
            }
            self._startup_observability["first_publication"] = first_publication_diagnostics
            if self._startup_freshness_recorder is not None and lease_age_ms is not None:
                self._startup_freshness_recorder.record_publication_lease(lease_age_ms / 1000.0)
            breathing = self._startup_observability.get("breathing")
            if isinstance(breathing, dict) and breathing.get("first_activation_monotonic") is None:
                breathing["before_first_publication"] = False
            logger.info(
                "Movement manager first target head_pose=%s head_rpy_deg=%s body_yaw=%.9f antennas=%s",
                head.tolist(),
                (target_angles.roll, target_angles.pitch, target_angles.yaw),
                body_yaw,
                antennas,
            )
            logger.info(
                "FIRST_STARTUP_PUBLICATION target_head_rpy_deg=%s authorized_head_rpy_deg=%s "
                "delta_rpy_deg=%s matches_authorized=%s publication_monotonic=%.6f "
                "target_snapshot_authorized_monotonic=%s target_snapshot_age_ms=%s "
                "publication_authorization_source=%s publication_authorization_monotonic=%s "
                "publication_authorization_age_ms=%s publication_authorization_lease_limit_ms=%s "
                "publication_authorization_lease_result=%s",
                (target_angles.roll, target_angles.pitch, target_angles.yaw),
                authorized_angles,
                delta_rpy_deg,
                matches_authorized,
                publication_monotonic,
                first_publication_diagnostics["target_snapshot_authorized_monotonic"],
                None if authorized_age_ms is None else round(authorized_age_ms, 3),
                first_publication_diagnostics["publication_authorization_source"],
                first_publication_diagnostics["publication_authorization_monotonic"],
                None if lease_age_ms is None else round(lease_age_ms, 3),
                first_publication_diagnostics["publication_authorization_lease_limit_ms"],
                lease_result,
            )
        except Exception as exc:
            self._startup_observability["first_publication"] = {
                "diagnostic_status": "ERROR",
                "diagnostic_error": f"{type(exc).__name__}: {exc}",
            }
            logger.warning("First startup publication diagnostics failed: %s", exc)
        finally:
            self._startup_first_target_logged = True

    def _update_frequency_stats(
        self,
        loop_start: float,
        prev_loop_start: float,
        stats: LoopFrequencyStats,
    ) -> LoopFrequencyStats:
        """Update frequency statistics based on the current loop start time."""
        period = loop_start - prev_loop_start
        if period > 0:
            stats.last_freq = 1.0 / period
            stats.count += 1
            delta = stats.last_freq - stats.mean
            stats.mean += delta / stats.count
            stats.m2 += delta * (stats.last_freq - stats.mean)
            stats.min_freq = min(stats.min_freq, stats.last_freq)
        return stats

    def _schedule_next_tick(self, loop_start: float, stats: LoopFrequencyStats) -> Tuple[float, LoopFrequencyStats]:
        """Compute sleep time to maintain target frequency and update potential freq."""
        computation_time = self._now() - loop_start
        stats.potential_freq = 1.0 / computation_time if computation_time > 0 else float("inf")
        sleep_time = max(0.0, self.target_period - computation_time)
        return sleep_time, stats

    def _record_frequency_snapshot(self, stats: LoopFrequencyStats) -> None:
        """Store a thread-safe snapshot of current frequency statistics."""
        with self._status_lock:
            self._freq_snapshot = LoopFrequencyStats(
                mean=stats.mean,
                m2=stats.m2,
                min_freq=stats.min_freq,
                count=stats.count,
                last_freq=stats.last_freq,
                potential_freq=stats.potential_freq,
            )

    def _maybe_log_frequency(self, loop_count: int, print_interval_loops: int, stats: LoopFrequencyStats) -> None:
        """Emit frequency telemetry when enough loops have elapsed."""
        if loop_count % print_interval_loops != 0 or stats.count == 0:
            return

        variance = stats.m2 / stats.count if stats.count > 0 else 0.0
        lowest = stats.min_freq if stats.min_freq != float("inf") else 0.0
        logger.debug(
            "Loop freq - avg: %.2fHz, variance: %.4f, min: %.2fHz, last: %.2fHz, potential: %.2fHz, target: %.1fHz",
            stats.mean,
            variance,
            lowest,
            stats.last_freq,
            stats.potential_freq,
            self.target_frequency,
        )
        stats.reset()

    def start(self) -> None:
        """Start the worker thread that drives the 100 Hz control loop."""
        with self._worker_lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning("Move worker already running; start() ignored")
                return
            if self.state.last_primary_pose is None or self._startup_baseline_error is not None:
                logger.error(
                    "Move worker not started because the startup baseline is unavailable: %s",
                    self._startup_baseline_error or "unknown validation failure",
                )
                return
            self._sdk_shutdown_event.clear()
            with self._sdk_health_lock:
                if self._sdk_probe_thread is None or not self._sdk_probe_thread.is_alive():
                    self._sdk_probe_thread = threading.Thread(
                        target=self._sdk_probe_loop,
                        daemon=True,
                        name="sdk-control-health",
                    )
                    self._sdk_probe_thread.start()
            self._stop_event.clear()
            logger.info("Movement manager worker starting")
            if self._wake_trace is not None:
                self._wake_trace.record_transition("movement_manager_start", active_task="MovementManager")
            self._thread = threading.Thread(target=self.working_loop, daemon=True)
            self._thread.start()
        logger.debug("Move worker started")

    def wait_for_startup_publication(self, timeout: float) -> bool:
        """Wait until the first authorized publication succeeds or startup fails."""
        if self._startup_confirmation_at is None:
            return True
        self._startup_publication_event.wait(timeout=max(0.0, timeout))
        return self._startup_publication_succeeded

    def set_startup_confirmation(self, confirmed_at: float, lease_s: float = 0.25) -> None:
        """Arm the lease that gates the worker's first startup publication."""
        self._startup_confirmation_at = confirmed_at
        self._startup_confirmation_lease_s = lease_s
        authorized = self._startup_observability.get("authorized_snapshot")
        self._startup_observability["final_consistency"] = {
            "result": "PASS",
            "timestamp": None,
            "monotonic": confirmed_at,
            "publication_authorization_source": "final_consistency_confirmation",
            "publication_authorization_monotonic": confirmed_at,
            "publication_authorization_lease_limit_ms": lease_s * 1000.0,
            "authorized_head_rpy_deg": authorized.get("head_rpy_deg") if isinstance(authorized, dict) else None,
        }

    def stop(self, reset_to_neutral: bool = True) -> None:
        """Request the worker thread to stop and wait for it to exit.

        Optionally resets the robot to a neutral position after stopping.
        """
        self._sdk_shutdown_event.set()
        with self._sdk_health_lock:
            recovery_thread = self._sdk_recovery_thread
        if recovery_thread is not None and recovery_thread is not threading.current_thread():
            recovery_thread.join(timeout=5.0)
        with self._stage1_lifecycle_lock:
            lifecycle = self._stage1_lifecycle
        if lifecycle is not None:
            self._release_stage1_calibration(
                lifecycle.session_id,
                "movement manager stopped",
                restore_movement=False,
            )
        self._cancel_stage1_preview_stabilization("movement manager stopped")
        worker_was_running = False
        with self._worker_lock:
            if self._thread is not None and self._thread.is_alive():
                worker_was_running = True
            if reset_to_neutral:
                logger.info("Stopping movement manager and resetting to neutral position...")
            else:
                logger.info("Stopping movement manager...")

            if worker_was_running:
                # Clear any queued moves and stop current move
                self.clear_move_queue()

                # Stop the worker thread first so it doesn't interfere
                self._stop_event.set()
                assert self._thread is not None
                self._thread.join()
                self._thread = None
            else:
                logger.debug("Move worker not running during stop")
        logger.debug("Move worker stopped")

        if self._head_tracking:
            try:
                ensure_daemon_head_tracking_disabled(
                    self.current_robot,
                    logger,
                    reason="movement_manager_shutdown",
                    application_owned=True,
                )
            except DaemonTrackingRecoveryError as exc:
                logger.error("Failed to confirm head tracking disabled during shutdown: %s", exc)
            else:
                self._head_tracking = False
                self._track_anchor = None
                self._is_speaking = False

        if not reset_to_neutral or not worker_was_running:
            return

        # Reset to neutral position using goto_target (same approach as wake_up)
        try:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            neutral_antennas = [-0.1745, 0.1745]  # ~10° offset to reduce shaking
            neutral_body_yaw = 0.0

            # Use goto_target directly on the robot
            self.current_robot.goto_target(
                head=neutral_head_pose,
                antennas=neutral_antennas,
                duration=2.0,
                body_yaw=neutral_body_yaw,
            )

            logger.info("Reset to neutral position completed")

        except Exception as e:
            logger.error(f"Failed to reset to neutral position: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return a lightweight status snapshot for observability."""
        with self._worker_lock:
            worker_alive = self._thread is not None and self._thread.is_alive()
        with self._status_lock:
            pose_snapshot = (
                clone_full_body_pose(self._last_commanded_pose) if self._last_commanded_pose is not None else None
            )
            freq_snapshot = LoopFrequencyStats(
                mean=self._freq_snapshot.mean,
                m2=self._freq_snapshot.m2,
                min_freq=self._freq_snapshot.min_freq,
                count=self._freq_snapshot.count,
                last_freq=self._freq_snapshot.last_freq,
                potential_freq=self._freq_snapshot.potential_freq,
            )

        head_matrix = pose_snapshot[0].tolist() if pose_snapshot else None
        antennas = pose_snapshot[1] if pose_snapshot else None
        body_yaw = pose_snapshot[2] if pose_snapshot else None

        return {
            "worker_alive": worker_alive,
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
            "startup_baseline": {
                "status": "adopted" if self._startup_baseline_adopted_at is not None else "degraded",
                "adopted_at": self._startup_baseline_adopted_at,
                "error": self._startup_baseline_error,
            },
            "last_commanded_pose": {
                "head": head_matrix,
                "antennas": antennas,
                "body_yaw": body_yaw,
            },
            "loop_frequency": {
                "last": freq_snapshot.last_freq,
                "mean": freq_snapshot.mean,
                "min": freq_snapshot.min_freq,
                "potential": freq_snapshot.potential_freq,
                "samples": freq_snapshot.count,
            },
            "sdk_control": self.get_sdk_control_status(),
            "stage1_session": self._stage1_lifecycle.session_id if self._stage1_lifecycle is not None else None,
            "calibration_queued": self._stage1_calibration is not None,
            "startup_observability": self._startup_observability,
        }

    def working_loop(self) -> None:
        """Run the primary-move control loop with a single set_target() call per tick."""
        logger.debug("Starting enhanced movement control loop (100Hz)")

        loop_count = 0
        prev_loop_start = self._now()
        print_interval_loops = max(1, int(self.target_frequency * 2))
        freq_stats = self._freq_stats

        while not self._stop_event.is_set():
            loop_start = self._now()
            loop_count += 1

            if loop_count > 1:
                freq_stats = self._update_frequency_stats(loop_start, prev_loop_start, freq_stats)
            prev_loop_start = loop_start

            first_startup_publication = (
                not self._startup_first_target_logged and self._startup_confirmation_at is not None
            )
            if first_startup_publication:
                confirmation_at = self._startup_confirmation_at
                if confirmation_at is None:
                    raise RuntimeError("startup confirmation timestamp disappeared")
                checked_at = self._now()
                if checked_at - confirmation_at > self._startup_confirmation_lease_s:
                    self._block_expired_startup_publication(checked_at)
                    self._stop_event.set()
                    self._startup_publication_event.set()
                    break

            # 1) Poll external commands
            self._handoff_published_this_tick = False
            self._poll_signals(loop_start)
            if self._stop_event.is_set():
                break
            if self._handoff_published_this_tick:
                sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
                self._publish_shared_state()
                self._record_frequency_snapshot(freq_stats)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue
            if self._stage1_owns_movement_freeze() or self._stage1_preview_suppresses_publication(loop_start):
                sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
                self._publish_shared_state()
                self._record_frequency_snapshot(freq_stats)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # 2) Manage the primary move queue (start new move, end finished move, breathing)
            self._update_primary_motion(loop_start)

            # 3) Build the primary full-body pose for this tick
            head, antennas, body_yaw = self._get_primary_pose(loop_start)

            # 4) Apply listening antenna freeze or blend-back
            antennas_cmd = self._calculate_blended_antennas(antennas)

            # 5) Single set_target call - the only control point
            publication_succeeded = self._issue_control_command(head, antennas_cmd, body_yaw)
            if first_startup_publication:
                self._startup_publication_succeeded = publication_succeeded
                self._startup_publication_event.set()
                if not publication_succeeded:
                    self._stop_event.set()
                    break

            # 6) Adaptive sleep to align to next tick, then publish shared state
            sleep_time, freq_stats = self._schedule_next_tick(loop_start, freq_stats)
            self._publish_shared_state()
            self._record_frequency_snapshot(freq_stats)

            # 7) Periodic telemetry on loop frequency
            self._maybe_log_frequency(loop_count, print_interval_loops, freq_stats)

            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.debug("Movement control loop stopped")
