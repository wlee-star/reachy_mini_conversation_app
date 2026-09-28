"""Helpers for app startup and shutdown lifecycle behavior."""

import os
import json
import time
import asyncio
import logging
import threading
import urllib.error
import urllib.request
from enum import Enum
from typing import Protocol
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import INIT_HEAD_POSE, SLEEP_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS
from reachy_mini.utils.interpolation import distance_between_poses
from reachy_mini_conversation_app.config import PROJECT_ROOT, config, set_custom_profile
from reachy_mini_conversation_app.wake_trace import WakeTrace
from reachy_mini_conversation_app.face_tracking import (
    STAGE1_TELEMETRY_FRESHNESS_S,
    POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ,
    Stage1Instrumentation,
    PostWakeTelemetrySnapshot,
    pose_euler_degrees,
    validate_head_pose,
)
from reachy_mini_conversation_app.profile_store import DEFAULT_PROFILE_NAME, migrate_legacy_profiles
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, initialize_tools
from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep


_STOP_CURRENT_APP_PATH = "/api/apps/stop-current-app"
_STOP_CURRENT_APP_TIMEOUT_S = 2.0
_SLEEP_HEAD_TRANSLATION_TOLERANCE_M = 0.05
_SLEEP_HEAD_ROTATION_TOLERANCE_RAD = 0.35
_DASHBOARD_STOPPED_PATH = PROJECT_ROOT / "control_dashboard" / "runtime" / "stopped.json"
_CONVERSATION_SERVICE_ID = "conversation"
_DASHBOARD_ACK_STOP_TIMEOUT_S = 2.0
_POST_WAKE_HEAD_TRANSLATION_TOLERANCE_M = 0.010
# Physical SDK-wake residuals are stable up to 6.19° geodesic from INIT;
# 6.25° admits that bounded hardware residual while still rejecting ≥7°/faults.
_POST_WAKE_HEAD_ROTATION_TOLERANCE_RAD = np.deg2rad(6.25)
_NON_SLEEP_STARTUP_RECOVERY_ADMISSION_ROTATION_RAD = _POST_WAKE_HEAD_ROTATION_TOLERANCE_RAD
_POST_WAKE_BODY_YAW_TOLERANCE_RAD = 0.05
_POST_WAKE_ANTENNA_TOLERANCE_RAD = 0.05
_POST_WAKE_SAMPLE_INTERVAL_S = 0.05
_POST_WAKE_STABILITY_DURATION_S = 0.5
_POST_WAKE_MIN_STABLE_SAMPLES = 10
_POST_WAKE_ROTATION_SPREAD_RAD = np.deg2rad(1.5)
_POST_WAKE_TRANSLATION_SPREAD_M = 0.002
_POST_WAKE_TIMEOUT_S = 3.0
_NON_SLEEP_STARTUP_TIMEOUT_S = 3.0
_NON_SLEEP_STARTUP_SAMPLE_INTERVAL_S = 0.05
_NON_SLEEP_STARTUP_STABILITY_DURATION_S = 0.5
_NON_SLEEP_STARTUP_MIN_STABLE_SAMPLES = 10
_NON_SLEEP_STARTUP_ROTATION_SPREAD_RAD = np.deg2rad(1.5)
_NON_SLEEP_STARTUP_TRANSLATION_SPREAD_M = 0.002
# Provisional auto-accept envelope pending dedicated physical validation.
_NON_SLEEP_STARTUP_PROVISIONAL_ROTATION_ENVELOPE_RAD = np.deg2rad(6.0)
_AUTHORIZED_STARTUP_MAX_AGE_S = STAGE1_TELEMETRY_FRESHNESS_S
_AUTHORIZED_STARTUP_BODY_YAW_TOLERANCE_RAD = 0.05
_AUTHORIZED_STARTUP_ANTENNA_TOLERANCE_RAD = 0.05
_TRACKING_DISABLE_TIMEOUT_S = 1.5
_TRACKING_DISABLE_POLL_INTERVAL_S = 0.05
_TRACKING_STATUS_HTTP_TIMEOUT_S = 0.5
_ORDERLY_SHUTDOWN_PREFLIGHT_TIMEOUT_S = 1.0


@dataclass
class StartupFreshnessRecorder:
    """Accumulate freshness values already evaluated by startup safety gates."""

    max_acquisition_s: float = 0.0
    max_backend_age_s: float = 0.0
    max_publication_lease_s: float = 0.0
    daemon_advancement: bool = True
    backend_advancement: bool = True
    observed_samples: int = 0
    publication_lease_observed: bool = False

    def observe(
        self,
        snapshot: "PostWakeTelemetrySnapshot",
        *,
        daemon_advanced: bool,
        backend_advanced: bool,
    ) -> None:
        """Record one physical sample without changing its safety result."""
        self.max_acquisition_s = max(self.max_acquisition_s, snapshot.acquisition_elapsed_s)
        self.max_backend_age_s = max(self.max_backend_age_s, snapshot.backend_last_alive_age_s)
        self.daemon_advancement = self.daemon_advancement and daemon_advanced
        self.backend_advancement = self.backend_advancement and backend_advanced
        self.observed_samples += 1

    def record_publication_lease(self, lease_age_s: float) -> None:
        """Record the lease age checked at first publication."""
        self.max_publication_lease_s = max(self.max_publication_lease_s, lease_age_s)
        self.publication_lease_observed = True

    def log_summary(self, logger: logging.Logger) -> None:
        """Log the complete startup freshness summary."""
        result = (
            self.daemon_advancement
            and self.backend_advancement
            and self.max_acquisition_s <= STAGE1_TELEMETRY_FRESHNESS_S
            and self.max_backend_age_s <= STAGE1_TELEMETRY_FRESHNESS_S
            and self.max_publication_lease_s <= STAGE1_TELEMETRY_FRESHNESS_S
            and self.observed_samples > 0
            and self.publication_lease_observed
        )
        logger.info(
            "STARTUP_FRESHNESS max_acquisition_ms=%.3f max_backend_age_ms=%.3f "
            "max_publication_lease_ms=%.3f daemon_advancement=%s backend_advancement=%s "
            "limit_ms=%.1f samples=%s result=%s",
            self.max_acquisition_s * 1000.0,
            self.max_backend_age_s * 1000.0,
            self.max_publication_lease_s * 1000.0,
            str(self.daemon_advancement).lower(),
            str(self.backend_advancement).lower(),
            STAGE1_TELEMETRY_FRESHNESS_S * 1000.0,
            self.observed_samples,
            str(result).lower(),
        )


_ORDERLY_SHUTDOWN_PREFLIGHT_MIN_SAMPLES = 2
_ORDERLY_SHUTDOWN_NEUTRAL_DURATION_S = 2.0


class PostWakeConvergenceError(RuntimeError):
    """Raised when an automatic wake does not converge safely."""


class DaemonTrackingRecoveryError(RuntimeError):
    """Raised when daemon tracking cannot be authoritatively disabled."""


class OrderlyShutdownParkError(RuntimeError):
    """Raised when a controlled shutdown cannot verify a neutral park."""


@dataclass(frozen=True)
class DaemonTrackingDisableResult:
    """Describe one bounded disable-only daemon tracking check."""

    initial_state: bool | None
    command_issued: bool
    confirmed: bool


def _coerce_tracking_enabled(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _read_daemon_tracking_enabled_via_http(robot: ReachyMini) -> bool | None:
    host = getattr(robot.client, "host", None)
    port = getattr(robot.client, "port", None)
    if not isinstance(host, str) or not isinstance(port, int):
        return None

    request = urllib.request.Request(f"http://{host}:{port}/api/daemon/status", method="GET")
    with urllib.request.urlopen(request, timeout=_TRACKING_STATUS_HTTP_TIMEOUT_S) as response:
        payload: object = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        return None
    return _coerce_tracking_enabled(payload.get("head_tracking_enabled"))


def _read_daemon_tracking_enabled(robot: ReachyMini) -> bool | None:
    status = robot.client.get_status(wait=False)
    model_dump = getattr(status, "model_dump", None)
    if callable(model_dump):
        status_payload = model_dump()
        if isinstance(status_payload, dict):
            if "head_tracking_enabled" in status_payload:
                return _coerce_tracking_enabled(status_payload.get("head_tracking_enabled"))

    if "head_tracking_enabled" in vars(status):
        return _coerce_tracking_enabled(vars(status).get("head_tracking_enabled"))

    enabled = _read_daemon_tracking_enabled_via_http(robot)
    return enabled if isinstance(enabled, bool) else None


def ensure_daemon_head_tracking_disabled(
    robot: ReachyMini,
    logger: logging.Logger,
    *,
    reason: str,
    application_owned: bool = False,
    timeout_s: float = _TRACKING_DISABLE_TIMEOUT_S,
    poll_interval_s: float = _TRACKING_DISABLE_POLL_INTERVAL_S,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> DaemonTrackingDisableResult:
    """Disable daemon tracking once when required and require authoritative confirmation."""
    deadline = monotonic() + timeout_s
    initial_state: bool | None = None
    last_read_error: Exception | None = None
    while monotonic() <= deadline:
        try:
            initial_state = _read_daemon_tracking_enabled(robot)
            last_read_error = None
        except Exception as exc:
            last_read_error = exc
            initial_state = None
        if initial_state is not None:
            break
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            break
        sleep(min(poll_interval_s, remaining))

    if last_read_error is not None:
        logger.error("Tracking fail-safe state read failed reason=%s error=%s", reason, last_read_error)
        raise DaemonTrackingRecoveryError("authoritative daemon tracking state unavailable") from last_read_error

    if initial_state is False:
        logger.info(
            "Tracking fail-safe reason=%s initial=false command_issued=false confirmed=true",
            reason,
        )
        return DaemonTrackingDisableResult(False, False, True)
    if initial_state is None and not application_owned:
        logger.error(
            "Tracking fail-safe reason=%s initial=unknown command_issued=false confirmed=false",
            reason,
        )
        raise DaemonTrackingRecoveryError("authoritative daemon tracking state is unknown")

    logger.warning(
        "Tracking fail-safe reason=%s initial=%s command_issued=true",
        reason,
        "true" if initial_state is True else "unknown_owned",
    )
    command_completed = threading.Event()
    command_errors: list[Exception] = []

    def request_disable() -> None:
        try:
            robot.stop_head_tracking()
        except Exception as exc:
            command_errors.append(exc)
        finally:
            command_completed.set()

    threading.Thread(
        target=request_disable,
        daemon=True,
        name="tracking-fail-safe-disable",
    ).start()
    if not command_completed.wait(timeout=max(0.0, timeout_s)):
        logger.error("Tracking fail-safe disable timed out reason=%s", reason)
        raise DaemonTrackingRecoveryError("daemon tracking disable request timed out")
    if command_errors:
        error = command_errors[0]
        logger.error("Tracking fail-safe disable failed reason=%s error=%s", reason, error)
        raise DaemonTrackingRecoveryError("daemon tracking disable request failed") from error

    last_state: bool | None = initial_state
    last_error: Exception | None = None
    while monotonic() <= deadline:
        try:
            last_state = _read_daemon_tracking_enabled(robot)
            last_error = None
        except Exception as exc:
            last_error = exc
        if last_state is False:
            logger.info(
                "Tracking fail-safe reason=%s initial=%s command_issued=true confirmed=true",
                reason,
                initial_state,
            )
            return DaemonTrackingDisableResult(initial_state, True, True)
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            break
        sleep(min(poll_interval_s, remaining))

    logger.error(
        "Tracking fail-safe confirmation failed reason=%s final_state=%s last_error=%s",
        reason,
        last_state,
        last_error,
    )
    raise DaemonTrackingRecoveryError("daemon tracking disable was not authoritatively confirmed")


def wait_for_orderly_shutdown_park_preconditions(
    instrumentation: "_PostWakeTelemetrySource",
    logger: logging.Logger,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> PostWakeTelemetrySnapshot:
    """Require fresh advancing idle telemetry before an orderly shutdown park."""
    try:
        _prepare_telemetry_transport(instrumentation, logger)
    except (RuntimeError, ValueError) as exc:
        raise OrderlyShutdownParkError(f"shutdown park transport preparation failed: {exc}") from exc

    deadline = monotonic() + _ORDERLY_SHUTDOWN_PREFLIGHT_TIMEOUT_S
    accepted_samples: list[PostWakeTelemetrySnapshot] = []
    last_daemon_timestamp: str | None = None
    last_backend_alive: float | None = None
    last_error = "no telemetry received"

    while monotonic() < deadline:
        try:
            snapshot = instrumentation.read_post_wake_telemetry(deadline=deadline)
            validate_head_pose(snapshot.present_head_pose)
            if not np.isfinite(snapshot.present_body_yaw) or not np.isfinite(snapshot.present_antennas).all():
                raise ValueError("shutdown park full-body telemetry contains nonfinite values")
            if snapshot.control_mode != "enabled":
                raise RuntimeError("shutdown park requires enabled motor control")
            if not snapshot.daemon_ready or snapshot.daemon_error is not None:
                raise RuntimeError("shutdown park requires a healthy daemon")
            if snapshot.control_loop_frequency_hz < POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ:
                raise RuntimeError("shutdown park control-loop frequency is unhealthy")
            if snapshot.active_move_count != 0:
                raise RuntimeError("shutdown park rejected because another daemon movement is active")
            if (
                snapshot.acquisition_elapsed_s > STAGE1_TELEMETRY_FRESHNESS_S
                or snapshot.backend_last_alive_age_s > STAGE1_TELEMETRY_FRESHNESS_S
            ):
                raise RuntimeError("shutdown park telemetry is stale")

            if last_daemon_timestamp is not None and snapshot.daemon_timestamp == last_daemon_timestamp:
                raise RuntimeError("shutdown park daemon timestamp did not advance")
            if last_backend_alive is not None and snapshot.backend_last_alive <= last_backend_alive:
                raise RuntimeError("shutdown park backend liveness did not advance")

            last_daemon_timestamp = snapshot.daemon_timestamp
            last_backend_alive = snapshot.backend_last_alive
            accepted_samples.append(snapshot)
            if len(accepted_samples) >= _ORDERLY_SHUTDOWN_PREFLIGHT_MIN_SAMPLES:
                _log_authoritative_telemetry_timing(snapshot, logger)
                logger.info(
                    "ORDERLY_SHUTDOWN_PARK_PREFLIGHT result=PASS samples=%s active_moves=0 control_mode=enabled",
                    len(accepted_samples),
                )
                return snapshot
        except (RuntimeError, ValueError) as exc:
            last_error = str(exc)
            accepted_samples.clear()
            logger.warning("ORDERLY_SHUTDOWN_PARK_PREFLIGHT sample_rejected=%s", exc)

        remaining = deadline - monotonic()
        if remaining > 0.0:
            sleep(min(_POST_WAKE_SAMPLE_INTERVAL_S, remaining))

    raise OrderlyShutdownParkError(f"shutdown park preflight failed: {last_error}")


def park_robot_for_orderly_shutdown(
    robot: ReachyMini,
    instrumentation: "_PostWakeTelemetrySource",
    logger: logging.Logger,
    *,
    freshness_recorder: StartupFreshnessRecorder | None = None,
) -> "AuthorizedStartupSnapshot":
    """Park at SDK neutral and require measured convergence before disconnect."""
    try:
        ensure_daemon_head_tracking_disabled(
            robot,
            logger,
            reason="orderly_shutdown_park",
            application_owned=True,
        )
        robot.disable_wobbling()
        wait_for_orderly_shutdown_park_preconditions(instrumentation, logger)
        robot.goto_target(
            head=INIT_HEAD_POSE,
            antennas=INIT_ANTENNAS_JOINT_POSITIONS,
            duration=_ORDERLY_SHUTDOWN_NEUTRAL_DURATION_S,
            body_yaw=0.0,
        )
        authorized_snapshot = wait_for_post_wake_convergence(
            instrumentation,
            logger,
            freshness_recorder=freshness_recorder,
        )
        if authorized_snapshot is None:
            raise OrderlyShutdownParkError("neutral command completed without measured convergence")
    except OrderlyShutdownParkError:
        raise
    except Exception as exc:
        raise OrderlyShutdownParkError(f"orderly shutdown neutral park failed: {exc}") from exc

    logger.info("ORDERLY_SHUTDOWN_PARK result=PASS measured_convergence=true")
    return authorized_snapshot


class NonSleepStartupFailureReason(str, Enum):
    """Reason a non-sleep startup pose was not authorized."""

    TELEMETRY_UNAVAILABLE = "TELEMETRY_UNAVAILABLE"
    TELEMETRY_STALE = "TELEMETRY_STALE"
    TIMESTAMP_NOT_ADVANCING = "TIMESTAMP_NOT_ADVANCING"
    DAEMON_UNHEALTHY = "DAEMON_UNHEALTHY"
    ACTIVE_DAEMON_MOVE = "ACTIVE_DAEMON_MOVE"
    POSE_INVALID = "POSE_INVALID"
    POSE_UNSTABLE = "POSE_UNSTABLE"
    POSE_OUTSIDE_PROVISIONAL_ENVELOPE = "POSE_OUTSIDE_PROVISIONAL_ENVELOPE"
    DEADLINE_EXHAUSTED = "DEADLINE_EXHAUSTED"
    AUTHORIZATION_EXPIRED = "AUTHORIZATION_EXPIRED"
    AUTHORIZED_SNAPSHOT_DIVERGED = "AUTHORIZED_SNAPSHOT_DIVERGED"


@dataclass(frozen=True)
class AuthorizedStartupSnapshot:
    """Immutable physical state authorized by the non-sleep startup gate."""

    head_transform: tuple[tuple[float, ...], ...]
    head_joints: tuple[float, ...]
    body_yaw: float
    antennas: tuple[float, float]
    daemon_timestamp: str
    backend_last_alive: float
    acquisition_monotonic: float
    authorized_monotonic: float
    validation_sample_count: int
    provenance: str = "NON_SLEEP_STARTUP_GATE"

    def head_pose_array(self) -> npt.NDArray[np.float64]:
        """Return a consumer-owned array for the authorized head transform."""
        return np.asarray(self.head_transform, dtype=np.float64)


def authorized_startup_snapshot_diagnostics(snapshot: AuthorizedStartupSnapshot) -> dict[str, object]:
    """Return passive diagnostics for a lifecycle-authorized startup snapshot."""
    angles = pose_euler_degrees(snapshot.head_pose_array())
    return {
        "provenance": snapshot.provenance,
        "head_rpy_deg": [angles.roll, angles.pitch, angles.yaw],
        "daemon_timestamp": snapshot.daemon_timestamp,
        "captured_monotonic": snapshot.acquisition_monotonic,
        "authorized_monotonic": snapshot.authorized_monotonic,
        "validation_sample_count": snapshot.validation_sample_count,
    }


@dataclass(frozen=True)
class NonSleepStartupValidation:
    """Outcome of read-only validation for an already-awake startup pose."""

    accepted: bool
    reason: NonSleepStartupFailureReason | None
    valid_samples: int
    authorized_snapshot: AuthorizedStartupSnapshot | None = None


class NonSleepStartupValidationError(RuntimeError):
    """Raised when an already-awake startup pose cannot be safely adopted."""

    def __init__(self, result: NonSleepStartupValidation) -> None:
        """Retain the structured validation outcome."""
        self.result = result
        super().__init__(result.reason.value if result.reason is not None else "unknown startup validation failure")


class _PostWakeTelemetrySource(Protocol):
    def prepare_telemetry_transport(self) -> object: ...

    def read_post_wake_telemetry(self, *, deadline: float) -> PostWakeTelemetrySnapshot: ...


def _prepare_telemetry_transport(instrumentation: _PostWakeTelemetrySource, logger: logging.Logger) -> None:
    preparation = instrumentation.prepare_telemetry_transport()
    endpoint = getattr(preparation, "endpoint", "unknown")
    raw_duration_s = getattr(preparation, "duration_s", 0.0)
    duration_s = float(raw_duration_s) if isinstance(raw_duration_s, int | float) else 0.0
    result = getattr(preparation, "result", "unknown")
    logger.info(
        "TELEMETRY_TRANSPORT_PREPARATION endpoint=%s duration_ms=%.3f result=%s authorizing=false",
        endpoint,
        duration_s * 1000.0,
        result,
    )


def _log_authoritative_telemetry_timing(snapshot: PostWakeTelemetrySnapshot, logger: logging.Logger) -> None:
    timings = snapshot.endpoint_timings
    if timings is None:
        return
    logger.info(
        "AUTHORITATIVE_TELEMETRY_TIMING state_ms=%.3f status_ms=%.3f movement_ms=%.3f total_ms=%.3f",
        timings.state_s * 1000.0,
        timings.status_s * 1000.0,
        timings.movement_s * 1000.0,
        timings.total_s * 1000.0,
    )


def _non_sleep_exception_reason(exc: RuntimeError | ValueError) -> NonSleepStartupFailureReason:
    message = str(exc).lower()
    if "timestamp" in message and "stale" in message:
        return NonSleepStartupFailureReason.TIMESTAMP_NOT_ADVANCING
    if "stale" in message or "freshness" in message:
        return NonSleepStartupFailureReason.TELEMETRY_STALE
    if isinstance(exc, ValueError):
        return NonSleepStartupFailureReason.POSE_INVALID
    return NonSleepStartupFailureReason.TELEMETRY_UNAVAILABLE


def wait_for_non_sleep_startup_validation(
    instrumentation: _PostWakeTelemetrySource,
    logger: logging.Logger,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    freshness_recorder: StartupFreshnessRecorder | None = None,
) -> NonSleepStartupValidation:
    """Authorize a stable near-neutral awake pose without commanding the robot."""
    try:
        _prepare_telemetry_transport(instrumentation, logger)
    except (RuntimeError, ValueError) as exc:
        logger.error("NON_SLEEP_STARTUP_TRANSPORT_PREPARATION_FAILED reason=%s", exc)
        return NonSleepStartupValidation(False, NonSleepStartupFailureReason.TELEMETRY_UNAVAILABLE, 0)

    started_at = monotonic()
    deadline = started_at + _NON_SLEEP_STARTUP_TIMEOUT_S
    stable_started_at: float | None = None
    stable_samples: list[PostWakeTelemetrySnapshot] = []
    last_backend_alive: float | None = None
    last_daemon_timestamp: str | None = None
    last_reason: NonSleepStartupFailureReason | None = None
    last_sample_rejection: NonSleepStartupFailureReason | None = None
    telemetry_error_count = 0
    last_telemetry_error: str | None = None
    saw_valid_pose = False
    logger.info(
        "NON_SLEEP_STARTUP_VALIDATION_START timeout_s=%.3f sample_rate_hz=%.1f stability_s=%.3f "
        "min_samples=%s provisional_rotation_envelope_deg=%.1f",
        _NON_SLEEP_STARTUP_TIMEOUT_S,
        1.0 / _NON_SLEEP_STARTUP_SAMPLE_INTERVAL_S,
        _NON_SLEEP_STARTUP_STABILITY_DURATION_S,
        _NON_SLEEP_STARTUP_MIN_STABLE_SAMPLES,
        np.degrees(_NON_SLEEP_STARTUP_PROVISIONAL_ROTATION_ENVELOPE_RAD),
    )

    while monotonic() < deadline:
        try:
            snapshot = instrumentation.read_post_wake_telemetry(deadline=deadline)
            current_time = monotonic()
            pose = validate_head_pose(snapshot.present_head_pose)
            if not np.isfinite(snapshot.present_body_yaw) or not np.isfinite(snapshot.present_antennas).all():
                raise ValueError("startup full-body pose contains nonfinite values")
            saw_valid_pose = True
            _, neutral_rotation, _ = distance_between_poses(pose, INIT_HEAD_POSE)
            timestamp_advanced = last_daemon_timestamp is None or snapshot.daemon_timestamp != last_daemon_timestamp
            backend_advanced = last_backend_alive is None or snapshot.backend_last_alive > last_backend_alive
            if freshness_recorder is not None:
                freshness_recorder.observe(
                    snapshot,
                    daemon_advanced=timestamp_advanced,
                    backend_advanced=backend_advanced,
                )
            last_daemon_timestamp = snapshot.daemon_timestamp
            last_backend_alive = max(snapshot.backend_last_alive, last_backend_alive or snapshot.backend_last_alive)

            reason: NonSleepStartupFailureReason | None = None
            if (
                not snapshot.daemon_ready
                or snapshot.daemon_error is not None
                or snapshot.control_loop_frequency_hz < POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ
            ):
                reason = NonSleepStartupFailureReason.DAEMON_UNHEALTHY
            elif snapshot.control_mode != "enabled":
                reason = NonSleepStartupFailureReason.DAEMON_UNHEALTHY
            elif snapshot.active_move_count != 0:
                reason = NonSleepStartupFailureReason.ACTIVE_DAEMON_MOVE
            elif (
                snapshot.acquisition_elapsed_s > STAGE1_TELEMETRY_FRESHNESS_S
                or snapshot.backend_last_alive_age_s > STAGE1_TELEMETRY_FRESHNESS_S
            ):
                reason = NonSleepStartupFailureReason.TELEMETRY_STALE
            elif not timestamp_advanced or not backend_advanced:
                reason = NonSleepStartupFailureReason.TIMESTAMP_NOT_ADVANCING
            elif neutral_rotation > _NON_SLEEP_STARTUP_PROVISIONAL_ROTATION_ENVELOPE_RAD + 1e-12:
                reason = NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE

            if reason is not None:
                last_reason = reason
                last_sample_rejection = reason
                stable_samples.clear()
                stable_started_at = None
            else:
                candidate_samples = [*stable_samples, snapshot]
                translation_spread = max(
                    np.linalg.norm(first.present_head_pose[:3, 3] - second.present_head_pose[:3, 3])
                    for first in candidate_samples
                    for second in candidate_samples
                )
                rotation_spread = max(
                    distance_between_poses(first.present_head_pose, second.present_head_pose)[1]
                    for first in candidate_samples
                    for second in candidate_samples
                )
                if (
                    translation_spread > _NON_SLEEP_STARTUP_TRANSLATION_SPREAD_M
                    or rotation_spread > _NON_SLEEP_STARTUP_ROTATION_SPREAD_RAD
                ):
                    last_reason = NonSleepStartupFailureReason.POSE_UNSTABLE
                    last_sample_rejection = NonSleepStartupFailureReason.POSE_UNSTABLE
                    stable_samples = [snapshot]
                    stable_started_at = current_time
                else:
                    last_sample_rejection = None
                    stable_samples = candidate_samples
                    if stable_started_at is None:
                        stable_started_at = current_time
                    stability_duration = current_time - stable_started_at
                    if (
                        len(stable_samples) >= _NON_SLEEP_STARTUP_MIN_STABLE_SAMPLES
                        and stability_duration >= _NON_SLEEP_STARTUP_STABILITY_DURATION_S
                    ):
                        logger.info(
                            "NON_SLEEP_STARTUP_VALIDATION_ACCEPTED elapsed_s=%.3f samples=%s "
                            "stability_s=%.3f neutral_rotation_deg=%.3f",
                            current_time - started_at,
                            len(stable_samples),
                            stability_duration,
                            np.degrees(neutral_rotation),
                        )
                        _log_authoritative_telemetry_timing(snapshot, logger)
                        authorized_at = monotonic()
                        authorized = AuthorizedStartupSnapshot(
                            head_transform=tuple(tuple(float(value) for value in row) for row in pose),
                            head_joints=tuple(snapshot.present_head_joints),
                            body_yaw=snapshot.present_body_yaw,
                            antennas=snapshot.present_antennas,
                            daemon_timestamp=snapshot.daemon_timestamp,
                            backend_last_alive=snapshot.backend_last_alive,
                            acquisition_monotonic=snapshot.monotonic_time,
                            authorized_monotonic=authorized_at,
                            validation_sample_count=len(stable_samples),
                        )
                        authorized_angles = pose_euler_degrees(authorized.head_pose_array())
                        logger.info(
                            "AUTHORIZED_STARTUP_SNAPSHOT provenance=%s head_rpy_deg=%s daemon_timestamp=%s "
                            "captured_monotonic=%.6f authorized_monotonic=%.6f",
                            authorized.provenance,
                            (authorized_angles.roll, authorized_angles.pitch, authorized_angles.yaw),
                            authorized.daemon_timestamp,
                            authorized.acquisition_monotonic,
                            authorized.authorized_monotonic,
                        )
                        return NonSleepStartupValidation(
                            True,
                            None,
                            len(stable_samples),
                            authorized,
                        )
                    last_reason = None
        except (RuntimeError, ValueError) as exc:
            last_reason = _non_sleep_exception_reason(exc)
            telemetry_error_count += 1
            last_telemetry_error = f"{type(exc).__name__}: {exc}"
            stable_samples.clear()
            stable_started_at = None

        remaining_s = deadline - monotonic()
        if remaining_s > 0.0:
            sleep(min(_NON_SLEEP_STARTUP_SAMPLE_INTERVAL_S, remaining_s))

    if last_sample_rejection is not None:
        if last_reason is not last_sample_rejection:
            logger.warning(
                "NON_SLEEP_STARTUP retained sample rejection=%s over terminal telemetry error=%s",
                last_sample_rejection.value,
                last_telemetry_error,
            )
        last_reason = last_sample_rejection
    elif last_reason is None:
        last_reason = (
            NonSleepStartupFailureReason.DEADLINE_EXHAUSTED
            if saw_valid_pose
            else NonSleepStartupFailureReason.TELEMETRY_UNAVAILABLE
        )
    logger.error(
        "NON_SLEEP_STARTUP_VALIDATION_FAILED elapsed_s=%.3f reason=%s valid_samples=%s "
        "telemetry_errors=%s last_telemetry_error=%s",
        monotonic() - started_at,
        last_reason.value,
        len(stable_samples),
        telemetry_error_count,
        last_telemetry_error,
    )
    return NonSleepStartupValidation(False, last_reason, len(stable_samples))


def wait_for_non_sleep_startup_recovery_admission(
    instrumentation: _PostWakeTelemetrySource,
    logger: logging.Logger,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Authorize one bounded neutral recovery for a marginal awake startup pose."""
    try:
        _prepare_telemetry_transport(instrumentation, logger)
    except (RuntimeError, ValueError) as exc:
        logger.error("NON_SLEEP_STARTUP_RECOVERY_ADMISSION_TRANSPORT_FAILED reason=%s", exc)
        return False

    started_at = monotonic()
    deadline = started_at + _NON_SLEEP_STARTUP_TIMEOUT_S
    stable_started_at: float | None = None
    stable_samples: list[PostWakeTelemetrySnapshot] = []
    last_backend_alive: float | None = None
    last_daemon_timestamp: str | None = None
    last_reason = "no eligible telemetry"
    logger.info(
        "NON_SLEEP_STARTUP_RECOVERY_ADMISSION_START timeout_s=%.3f normal_envelope_deg=%.1f "
        "admission_envelope_deg=%.2f",
        _NON_SLEEP_STARTUP_TIMEOUT_S,
        np.degrees(_NON_SLEEP_STARTUP_PROVISIONAL_ROTATION_ENVELOPE_RAD),
        np.degrees(_NON_SLEEP_STARTUP_RECOVERY_ADMISSION_ROTATION_RAD),
    )

    while monotonic() < deadline:
        try:
            snapshot = instrumentation.read_post_wake_telemetry(deadline=deadline)
            current_time = monotonic()
            pose = validate_head_pose(snapshot.present_head_pose)
            if not np.isfinite(snapshot.present_body_yaw) or not np.isfinite(snapshot.present_antennas).all():
                raise ValueError("startup recovery full-body pose contains nonfinite values")
            _, neutral_rotation, _ = distance_between_poses(pose, INIT_HEAD_POSE)
            body_yaw_error = abs(snapshot.present_body_yaw)
            antenna_errors = (
                abs(snapshot.present_antennas[0] - INIT_ANTENNAS_JOINT_POSITIONS[0]),
                abs(snapshot.present_antennas[1] - INIT_ANTENNAS_JOINT_POSITIONS[1]),
            )
            timestamp_advanced = last_daemon_timestamp is None or snapshot.daemon_timestamp != last_daemon_timestamp
            backend_advanced = last_backend_alive is None or snapshot.backend_last_alive > last_backend_alive
            last_daemon_timestamp = snapshot.daemon_timestamp
            if last_backend_alive is None or snapshot.backend_last_alive > last_backend_alive:
                last_backend_alive = snapshot.backend_last_alive

            eligible = (
                snapshot.daemon_ready
                and snapshot.daemon_error is None
                and snapshot.control_mode == "enabled"
                and snapshot.control_loop_frequency_hz >= POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ
                and snapshot.active_move_count == 0
                and snapshot.acquisition_elapsed_s <= STAGE1_TELEMETRY_FRESHNESS_S
                and snapshot.backend_last_alive_age_s <= STAGE1_TELEMETRY_FRESHNESS_S
                and timestamp_advanced
                and backend_advanced
                and body_yaw_error <= _POST_WAKE_BODY_YAW_TOLERANCE_RAD
                and max(antenna_errors) <= _POST_WAKE_ANTENNA_TOLERANCE_RAD
                and neutral_rotation > _NON_SLEEP_STARTUP_PROVISIONAL_ROTATION_ENVELOPE_RAD + 1e-12
                and neutral_rotation <= _NON_SLEEP_STARTUP_RECOVERY_ADMISSION_ROTATION_RAD + 1e-12
            )
            if not eligible:
                last_reason = (
                    "pose_or_health_outside_recovery_admission"
                    if snapshot.daemon_ready and snapshot.daemon_error is None
                    else "daemon_unhealthy"
                )
                stable_samples.clear()
                stable_started_at = None
            else:
                candidate_samples = [*stable_samples, snapshot]
                translation_spread = max(
                    np.linalg.norm(first.present_head_pose[:3, 3] - second.present_head_pose[:3, 3])
                    for first in candidate_samples
                    for second in candidate_samples
                )
                rotation_spread = max(
                    distance_between_poses(first.present_head_pose, second.present_head_pose)[1]
                    for first in candidate_samples
                    for second in candidate_samples
                )
                if (
                    translation_spread > _NON_SLEEP_STARTUP_TRANSLATION_SPREAD_M
                    or rotation_spread > _NON_SLEEP_STARTUP_ROTATION_SPREAD_RAD
                ):
                    last_reason = "pose_unstable"
                    stable_samples = [snapshot]
                    stable_started_at = current_time
                else:
                    stable_samples = candidate_samples
                    if stable_started_at is None:
                        stable_started_at = current_time
                    stability_duration = current_time - stable_started_at
                    if (
                        len(stable_samples) >= _NON_SLEEP_STARTUP_MIN_STABLE_SAMPLES
                        and stability_duration >= _NON_SLEEP_STARTUP_STABILITY_DURATION_S
                    ):
                        logger.info(
                            "NON_SLEEP_STARTUP_RECOVERY_ADMISSION_ACCEPTED elapsed_s=%.3f samples=%s "
                            "stability_s=%.3f neutral_rotation_deg=%.3f",
                            current_time - started_at,
                            len(stable_samples),
                            stability_duration,
                            np.degrees(neutral_rotation),
                        )
                        _log_authoritative_telemetry_timing(snapshot, logger)
                        return True
                    last_reason = "insufficient_stability"
        except (RuntimeError, ValueError) as exc:
            last_reason = f"{type(exc).__name__}: {exc}"
            stable_samples.clear()
            stable_started_at = None

        remaining_s = deadline - monotonic()
        if remaining_s > 0.0:
            sleep(min(_NON_SLEEP_STARTUP_SAMPLE_INTERVAL_S, remaining_s))

    logger.error(
        "NON_SLEEP_STARTUP_RECOVERY_ADMISSION_FAILED elapsed_s=%.3f samples=%s reason=%s",
        monotonic() - started_at,
        len(stable_samples),
        last_reason,
    )
    return False


def require_fresh_authorized_startup_snapshot(
    snapshot: AuthorizedStartupSnapshot,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Reject an authorized snapshot whose explicit handoff lease expired."""
    if monotonic() - snapshot.authorized_monotonic > _AUTHORIZED_STARTUP_MAX_AGE_S:
        raise NonSleepStartupValidationError(
            NonSleepStartupValidation(False, NonSleepStartupFailureReason.AUTHORIZATION_EXPIRED, 0)
        )


def confirm_authorized_startup_snapshot(
    instrumentation: _PostWakeTelemetrySource,
    authorized: AuthorizedStartupSnapshot,
    logger: logging.Logger,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> float:
    """Confirm without replacing an authorized startup snapshot before publication."""
    deadline = monotonic() + _AUTHORIZED_STARTUP_MAX_AGE_S
    try:
        current = instrumentation.read_post_wake_telemetry(deadline=deadline)
        completed_at = monotonic()
        current_pose = validate_head_pose(current.present_head_pose)
        translation, rotation, _ = distance_between_poses(authorized.head_pose_array(), current_pose)
        healthy = (
            current.daemon_ready
            and current.daemon_error is None
            and current.control_mode == "enabled"
            and current.control_loop_frequency_hz >= POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ
            and current.active_move_count == 0
            and current.acquisition_elapsed_s <= STAGE1_TELEMETRY_FRESHNESS_S
            and current.backend_last_alive_age_s <= STAGE1_TELEMETRY_FRESHNESS_S
            and current.daemon_timestamp != authorized.daemon_timestamp
            and current.backend_last_alive > authorized.backend_last_alive
            and len(current.present_head_joints) == 7
            and np.isfinite(current.present_head_joints).all()
        )
        consistent = (
            translation <= _NON_SLEEP_STARTUP_TRANSLATION_SPREAD_M
            and rotation <= _NON_SLEEP_STARTUP_ROTATION_SPREAD_RAD
            and abs(current.present_body_yaw - authorized.body_yaw) <= _AUTHORIZED_STARTUP_BODY_YAW_TOLERANCE_RAD
            and max(abs(value - expected) for value, expected in zip(current.present_antennas, authorized.antennas))
            <= _AUTHORIZED_STARTUP_ANTENNA_TOLERANCE_RAD
        )
        authorized_angles = pose_euler_degrees(authorized.head_pose_array())
        current_angles = pose_euler_degrees(current_pose)
        if completed_at > deadline or not healthy or not consistent:
            raise ValueError("authorized startup snapshot diverged")
    except (RuntimeError, ValueError) as exc:
        logger.error("AUTHORIZED_STARTUP_CONFIRMATION_FAILED error=%s", exc)
        raise NonSleepStartupValidationError(
            NonSleepStartupValidation(False, NonSleepStartupFailureReason.AUTHORIZED_SNAPSHOT_DIVERGED, 0)
        ) from exc
    logger.info(
        "STARTUP_FINAL_CONSISTENCY result=PASS authorized_head_rpy_deg=%s confirmed_head_rpy_deg=%s "
        "confirmation_timestamp=%s confirmation_monotonic=%.6f",
        (authorized_angles.roll, authorized_angles.pitch, authorized_angles.yaw),
        (current_angles.roll, current_angles.pitch, current_angles.yaw),
        current.daemon_timestamp,
        completed_at,
    )
    return completed_at


def require_fresh_startup_confirmation(
    confirmed_at: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Reject a final confirmation whose first-publication lease expired."""
    if monotonic() - confirmed_at > _AUTHORIZED_STARTUP_MAX_AGE_S:
        raise NonSleepStartupValidationError(
            NonSleepStartupValidation(False, NonSleepStartupFailureReason.AUTHORIZATION_EXPIRED, 0)
        )


def acknowledge_dashboard_sleep_stop(logger: logging.Logger) -> None:
    """Mark Conversation App as intentionally stopped before exiting for sleep.

    Writes ``stopped.json`` so the dashboard recovery loop will not treat the
    exit as a crash, and best-effort notifies the live dashboard process.
    """
    try:
        _append_stopped_service(_CONVERSATION_SERVICE_ID)
        logger.info("Marked Conversation App as intentionally stopped for sleep")
    except OSError as exc:
        logger.warning("Failed to write dashboard stopped state for sleep: %s", exc)

    base = (os.environ.get("CONTROL_DASHBOARD_URL") or "http://127.0.0.1:8788").rstrip("/")
    url = f"{base}/api/services/{_CONVERSATION_SERVICE_ID}/ack_stop"
    request = urllib.request.Request(url, data=b"{}", method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=_DASHBOARD_ACK_STOP_TIMEOUT_S) as response:
            response.read()
        logger.info("Dashboard acknowledged intentional sleep stop")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Dashboard ack_stop unavailable during sleep: %s", exc)


def _append_stopped_service(service_id: str) -> None:
    """Add ``service_id`` to the dashboard stopped-service file."""
    _DASHBOARD_STOPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    if _DASHBOARD_STOPPED_PATH.is_file():
        try:
            payload: object = json.loads(_DASHBOARD_STOPPED_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = []
        if isinstance(payload, list):
            ids = [str(item) for item in payload]
        elif isinstance(payload, dict) and isinstance(payload.get("ids"), list):
            ids = [str(item) for item in payload["ids"]]
    if service_id not in ids:
        ids.append(service_id)
    _DASHBOARD_STOPPED_PATH.write_text(json.dumps(sorted(ids), indent=2) + "\n", encoding="utf-8")


def initialize_tools_with_default_fallback(
    instance_path: str | Path | None,
    logger: logging.Logger,
) -> str | None:
    """Load the tool registry, degrading to the default profile if need be.

    Returns the profile that was abandoned, or None when the selection loaded.
    """
    # User profiles predating profile.md were never migrated on disk; convert
    # them here so the strict readers below (and the settings UI) see them.
    try:
        migrate_legacy_profiles(config.user_personalities_root())
    except Exception as exc:
        logger.warning("Legacy profile migration failed: %s", exc)

    try:
        initialize_tools(instance_path=instance_path)
        return None
    except Exception as exc:
        selected_profile = config.REACHY_MINI_CUSTOM_PROFILE
        if not selected_profile or selected_profile == DEFAULT_PROFILE_NAME:
            raise

        # set_custom_profile is a no-op while LOCKED_PROFILE is pinned, so
        # confirm the switch took effect before announcing a fallback.
        set_custom_profile(DEFAULT_PROFILE_NAME)
        if config.REACHY_MINI_CUSTOM_PROFILE != DEFAULT_PROFILE_NAME:
            logger.error(
                "Profile %r could not be loaded (%s) and this build is locked to it, "
                "so there is no profile to fall back to.",
                selected_profile,
                exc,
            )
            raise

        logger.error(
            "Profile %r could not be loaded (%s); starting on the %r profile instead. "
            "Reselect or repair it from the settings UI to restore it.",
            selected_profile,
            exc,
            DEFAULT_PROFILE_NAME,
        )
        initialize_tools(instance_path=instance_path, force=True)
        return selected_profile


def request_stop_current_app(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Request the Reachy Mini daemon to stop the current app."""
    stop_current_app_url = f"http://{robot.client.host}:{robot.client.port}{_STOP_CURRENT_APP_PATH}"
    request = urllib.request.Request(stop_current_app_url, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_STOP_CURRENT_APP_TIMEOUT_S) as response:
            response.read()
    except urllib.error.URLError as e:
        logger.error("Failed to request current app stop via %s: %s", stop_current_app_url, e)
        return False

    logger.info("Requested current app stop via %s", stop_current_app_url)
    return True


def _is_sleep_head_pose(head_pose: npt.ArrayLike) -> bool:
    try:
        current_head_pose: npt.NDArray[np.float64] = np.asarray(head_pose, dtype=np.float64)
    except (TypeError, ValueError):
        return False

    if current_head_pose.shape != (4, 4):
        return False

    pose_distances = distance_between_poses(current_head_pose, SLEEP_HEAD_POSE)
    translation_distance = float(pose_distances[0])
    rotation_angle = float(pose_distances[1])
    return (
        translation_distance <= _SLEEP_HEAD_TRANSLATION_TOLERANCE_M
        and rotation_angle <= _SLEEP_HEAD_ROTATION_TOLERANCE_RAD
    )


def _log_startup_pose(
    robot: ReachyMini,
    logger: logging.Logger,
    phase: str,
    head_pose: npt.ArrayLike | None = None,
) -> None:
    """Log cached robot state without issuing a command or waiting for new telemetry."""
    try:
        pose = np.asarray(robot.get_current_head_pose() if head_pose is None else head_pose, dtype=np.float64)
        head_joints, antennas = robot.get_current_joint_positions()
        angles = pose_euler_degrees(pose)
    except Exception as exc:
        logger.warning("Startup pose snapshot phase=%s unavailable: %s", phase, exc)
        return

    motor_mode = "unavailable"
    try:
        daemon_status = robot.client.get_status(wait=False)
        if daemon_status.backend_status is not None:
            motor_mode = daemon_status.backend_status.motor_control_mode.value
    except Exception as exc:
        logger.warning("Startup motor-mode snapshot phase=%s unavailable: %s", phase, exc)

    logger.info(
        "Startup pose snapshot phase=%s head_pose=%s head_rpy_deg=%s body_yaw=%.9f antennas=%s motor_mode=%s",
        phase,
        pose.tolist(),
        (angles.roll, angles.pitch, angles.yaw),
        float(head_joints[0]),
        tuple(float(position) for position in antennas),
        motor_mode,
    )


def wait_for_post_wake_convergence(
    instrumentation: _PostWakeTelemetrySource,
    logger: logging.Logger,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    freshness_recorder: StartupFreshnessRecorder | None = None,
) -> AuthorizedStartupSnapshot | None:
    """Wait for fresh physical telemetry to converge after an automatic wake."""
    try:
        _prepare_telemetry_transport(instrumentation, logger)
    except (RuntimeError, ValueError) as exc:
        logger.error("POST_WAKE_TRANSPORT_PREPARATION_FAILED reason=%s", exc)
        return None

    started_at = monotonic()
    deadline = started_at + _POST_WAKE_TIMEOUT_S
    stable_started_at: float | None = None
    stable_samples: list[PostWakeTelemetrySnapshot] = []
    last_backend_alive: float | None = None
    last_daemon_timestamp: str | None = None
    last_metrics: tuple[float, float, tuple[float, float, float], float, tuple[float, float]] | None = None
    logger.info(
        "POST_WAKE_CONVERGENCE_START timeout_s=%.3f sample_rate_hz=%.1f stability_s=%.3f min_samples=%s",
        _POST_WAKE_TIMEOUT_S,
        1.0 / _POST_WAKE_SAMPLE_INTERVAL_S,
        _POST_WAKE_STABILITY_DURATION_S,
        _POST_WAKE_MIN_STABLE_SAMPLES,
    )

    while monotonic() < deadline:
        try:
            snapshot = instrumentation.read_post_wake_telemetry(deadline=deadline)
            current_time = monotonic()
            pose = validate_head_pose(snapshot.present_head_pose)
            translation_error, rotation_error, _ = distance_between_poses(pose, INIT_HEAD_POSE)
            angles = pose_euler_degrees(pose)
            body_yaw_error = abs(snapshot.present_body_yaw)
            antenna_errors = (
                abs(snapshot.present_antennas[0] - INIT_ANTENNAS_JOINT_POSITIONS[0]),
                abs(snapshot.present_antennas[1] - INIT_ANTENNAS_JOINT_POSITIONS[1]),
            )
            last_metrics = (
                translation_error,
                rotation_error,
                (angles.roll, angles.pitch, angles.yaw),
                body_yaw_error,
                antenna_errors,
            )

            daemon_timestamp_advanced = (
                last_daemon_timestamp is None or snapshot.daemon_timestamp != last_daemon_timestamp
            )
            liveness_advanced = last_backend_alive is None or snapshot.backend_last_alive > last_backend_alive
            if freshness_recorder is not None:
                freshness_recorder.observe(
                    snapshot,
                    daemon_advanced=daemon_timestamp_advanced,
                    backend_advanced=liveness_advanced,
                )
            last_daemon_timestamp = snapshot.daemon_timestamp
            if last_backend_alive is None or snapshot.backend_last_alive > last_backend_alive:
                last_backend_alive = snapshot.backend_last_alive
            conditions = {
                "head_translation": translation_error <= _POST_WAKE_HEAD_TRANSLATION_TOLERANCE_M,
                "head_rotation": rotation_error <= _POST_WAKE_HEAD_ROTATION_TOLERANCE_RAD,
                "body_yaw": body_yaw_error <= _POST_WAKE_BODY_YAW_TOLERANCE_RAD,
                "antennas": max(antenna_errors) <= _POST_WAKE_ANTENNA_TOLERANCE_RAD,
                "control_mode": snapshot.control_mode == "enabled",
                "daemon_ready": snapshot.daemon_ready,
                "daemon_error": snapshot.daemon_error is None,
                "backend_recent": snapshot.backend_last_alive_age_s <= STAGE1_TELEMETRY_FRESHNESS_S,
                "backend_advanced": liveness_advanced,
                "control_loop": snapshot.control_loop_frequency_hz >= POST_WAKE_MIN_CONTROL_LOOP_FREQUENCY_HZ,
                "active_move": snapshot.active_move_count == 0,
                "acquisition": snapshot.acquisition_elapsed_s <= STAGE1_TELEMETRY_FRESHNESS_S,
            }
            failed_conditions = tuple(name for name, passed in conditions.items() if not passed)
            logger.debug(
                "POST_WAKE_SAMPLE elapsed_s=%.3f translation_m=%.6f rotation_deg=%.3f rpy_deg=%s "
                "body_yaw_error_rad=%.6f antenna_errors_rad=%s freshness_s=%.6f failed=%s",
                current_time - started_at,
                translation_error,
                np.degrees(rotation_error),
                (angles.roll, angles.pitch, angles.yaw),
                body_yaw_error,
                antenna_errors,
                snapshot.acquisition_elapsed_s,
                failed_conditions,
            )
            if failed_conditions:
                if stable_samples:
                    logger.info(
                        "POST_WAKE_STABILITY_RESET elapsed_s=%.3f stable_samples=%s stability_s=%.3f reason=%s",
                        current_time - started_at,
                        len(stable_samples),
                        current_time - stable_started_at if stable_started_at is not None else 0.0,
                        failed_conditions,
                    )
                stable_samples.clear()
                stable_started_at = None
            else:
                candidate_samples = [*stable_samples, snapshot]
                translation_spread = max(
                    np.linalg.norm(first.present_head_pose[:3, 3] - second.present_head_pose[:3, 3])
                    for first in candidate_samples
                    for second in candidate_samples
                )
                rotation_spread = max(
                    distance_between_poses(first.present_head_pose, second.present_head_pose)[1]
                    for first in candidate_samples
                    for second in candidate_samples
                )
                if (
                    translation_spread > _POST_WAKE_TRANSLATION_SPREAD_M
                    or rotation_spread > _POST_WAKE_ROTATION_SPREAD_RAD
                ):
                    logger.info(
                        "POST_WAKE_STABILITY_RESET elapsed_s=%.3f stable_samples=%s "
                        "translation_spread_m=%.6f rotation_spread_deg=%.3f reason=pose_spread",
                        current_time - started_at,
                        len(stable_samples),
                        translation_spread,
                        np.degrees(rotation_spread),
                    )
                    stable_samples = [snapshot]
                    stable_started_at = current_time
                else:
                    stable_samples = candidate_samples
                    if stable_started_at is None:
                        stable_started_at = current_time
                        logger.info(
                            "POST_WAKE_WITHIN_TOLERANCE elapsed_s=%.3f translation_m=%.6f "
                            "rotation_deg=%.3f rpy_deg=%s",
                            current_time - started_at,
                            translation_error,
                            np.degrees(rotation_error),
                            (angles.roll, angles.pitch, angles.yaw),
                        )

                stability_duration = current_time - stable_started_at
                if (
                    len(stable_samples) >= _POST_WAKE_MIN_STABLE_SAMPLES
                    and stability_duration >= _POST_WAKE_STABILITY_DURATION_S
                ):
                    logger.info(
                        "POST_WAKE_CONVERGED elapsed_s=%.3f translation_m=%.6f rotation_deg=%.3f "
                        "rpy_deg=%s body_yaw_error_rad=%.6f antenna_errors_rad=%s stable_samples=%s "
                        "stability_s=%.3f freshness_s=%.6f",
                        current_time - started_at,
                        translation_error,
                        np.degrees(rotation_error),
                        (angles.roll, angles.pitch, angles.yaw),
                        body_yaw_error,
                        antenna_errors,
                        len(stable_samples),
                        stability_duration,
                        snapshot.acquisition_elapsed_s,
                    )
                    _log_authoritative_telemetry_timing(snapshot, logger)
                    authorized_at = monotonic()
                    return AuthorizedStartupSnapshot(
                        head_transform=tuple(tuple(float(value) for value in row) for row in pose),
                        head_joints=tuple(snapshot.present_head_joints),
                        body_yaw=snapshot.present_body_yaw,
                        antennas=snapshot.present_antennas,
                        daemon_timestamp=snapshot.daemon_timestamp,
                        backend_last_alive=snapshot.backend_last_alive,
                        acquisition_monotonic=snapshot.monotonic_time,
                        authorized_monotonic=authorized_at,
                        validation_sample_count=len(stable_samples),
                        provenance="POST_WAKE_STAGE1Z",
                    )
        except (RuntimeError, ValueError) as exc:
            current_time = monotonic()
            if stable_samples:
                logger.info(
                    "POST_WAKE_STABILITY_RESET elapsed_s=%.3f stable_samples=%s stability_s=%.3f reason=%s",
                    current_time - started_at,
                    len(stable_samples),
                    current_time - stable_started_at if stable_started_at is not None else 0.0,
                    exc,
                )
            else:
                logger.debug("POST_WAKE_SAMPLE elapsed_s=%.3f invalid=%s", current_time - started_at, exc)
            stable_samples.clear()
            stable_started_at = None

        remaining_s = deadline - monotonic()
        if remaining_s > 0.0:
            sleep(min(_POST_WAKE_SAMPLE_INTERVAL_S, remaining_s))

    if last_metrics is None:
        logger.warning("POST_WAKE_TIMEOUT elapsed_s=%.3f no_valid_telemetry=True", monotonic() - started_at)
    else:
        translation_error, rotation_error, rpy, body_yaw_error, antenna_errors = last_metrics
        logger.warning(
            "POST_WAKE_TIMEOUT elapsed_s=%.3f translation_m=%.6f rotation_deg=%.3f rpy_deg=%s "
            "body_yaw_error_rad=%.6f antenna_errors_rad=%s stable_samples=%s",
            monotonic() - started_at,
            translation_error,
            np.degrees(rotation_error),
            rpy,
            body_yaw_error,
            antenna_errors,
            len(stable_samples),
        )
    return None


def hold_post_wake_convergence_failure(
    logger: logging.Logger,
    stop_event: threading.Event | None,
) -> None:
    """Retain robot ownership without movement until a human stops the app."""
    logger.error(
        "Startup validation failed; startup is blocked in a non-moving degraded hold. "
        "MovementManager and conversation services were not initialized."
    )
    (stop_event or threading.Event()).wait()


def prepare_robot_for_conversation(
    robot: ReachyMini,
    logger: logging.Logger,
    *,
    attempts: int = 8,
    delay_s: float = 1.0,
    wake_trace: WakeTrace | None = None,
    stage1_instrumentation: Stage1Instrumentation | None = None,
    freshness_recorder: StartupFreshnessRecorder | None = None,
) -> bool | AuthorizedStartupSnapshot:
    """Enable motors after power-off, then wake if the head is in the sleep pose."""
    _log_startup_pose(robot, logger, "before_motor_enable")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            robot.enable_motors()
            last_error = None
            break
        except Exception as e:
            last_error = e
            logger.warning("Failed to enable motors (attempt %s/%s): %s", attempt, attempts, e)
            if attempt < attempts:
                time.sleep(delay_s)
    if last_error is not None:
        logger.error("Could not enable motors before conversation start: %s", last_error)
    woke_from_sleep = wake_up_if_sleeping(
        robot,
        logger,
        wake_trace=wake_trace,
        stage1_instrumentation=stage1_instrumentation,
        freshness_recorder=freshness_recorder,
    )
    if isinstance(woke_from_sleep, AuthorizedStartupSnapshot):
        return woke_from_sleep
    if woke_from_sleep:
        return True
    if stage1_instrumentation is None:
        raise NonSleepStartupValidationError(
            NonSleepStartupValidation(False, NonSleepStartupFailureReason.TELEMETRY_UNAVAILABLE, 0)
        )
    if freshness_recorder is None:
        validation = wait_for_non_sleep_startup_validation(stage1_instrumentation, logger)
    else:
        validation = wait_for_non_sleep_startup_validation(
            stage1_instrumentation, logger, freshness_recorder=freshness_recorder
        )
    if not validation.accepted:
        if (
            validation.reason is NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE
            and wait_for_non_sleep_startup_recovery_admission(stage1_instrumentation, logger)
        ):
            logger.info("NON_SLEEP_STARTUP_RECOVERY_ATTEMPT count=1 target=INIT_HEAD_POSE")
            try:
                return park_robot_for_orderly_shutdown(
                    robot,
                    stage1_instrumentation,
                    logger,
                    freshness_recorder=freshness_recorder,
                )
            except OrderlyShutdownParkError as exc:
                raise PostWakeConvergenceError(f"non-sleep neutral recovery failed: {exc}") from exc
        raise NonSleepStartupValidationError(validation)
    if validation.authorized_snapshot is None:
        raise NonSleepStartupValidationError(
            NonSleepStartupValidation(False, NonSleepStartupFailureReason.TELEMETRY_UNAVAILABLE, 0)
        )
    return validation.authorized_snapshot


def _wake_up_with_trace(robot: ReachyMini, wake_trace: WakeTrace | None) -> None:
    """Run the SDK wake while optional instrumentation observes its public waypoints."""
    if wake_trace is None:
        robot.wake_up()
        return
    wake_trace.start()
    try:
        with wake_trace.observe_sdk_waypoints():
            robot.wake_up()
    except Exception:
        wake_trace.record_transition("wake_failed", active_task=None)
        wake_trace.close()
        raise
    wake_trace.wake_returned(INIT_HEAD_POSE)


def wake_up_if_sleeping(
    robot: ReachyMini,
    logger: logging.Logger,
    *,
    wake_trace: WakeTrace | None = None,
    stage1_instrumentation: Stage1Instrumentation | None = None,
    freshness_recorder: StartupFreshnessRecorder | None = None,
) -> bool | AuthorizedStartupSnapshot:
    """Run the SDK wake-up movement when Reachy starts from the sleep pose."""
    try:
        head_pose = robot.get_current_head_pose()
    except Exception as e:
        logger.warning("Could not read robot pose before startup wake-up check: %s", e)
        logger.info("Startup wake invoked reason=unreadable_head_pose")
        try:
            robot.enable_motors()
            _wake_up_with_trace(robot, wake_trace)
        except Exception as wake_error:
            logger.error("Failed to wake Reachy after an unreadable pose: %s", wake_error)
            return False
        logger.info("Startup wake completed reason=unreadable_head_pose")
        _log_startup_pose(robot, logger, "after_wake")
        return True

    pose_distances = distance_between_poses(np.asarray(head_pose, dtype=np.float64), SLEEP_HEAD_POSE)
    translation_distance = float(pose_distances[0])
    rotation_angle = float(pose_distances[1])
    translation_within_tolerance = translation_distance <= _SLEEP_HEAD_TRANSLATION_TOLERANCE_M
    rotation_within_tolerance = rotation_angle <= _SLEEP_HEAD_ROTATION_TOLERANCE_RAD
    sleep_pose_detected = _is_sleep_head_pose(head_pose)
    _log_startup_pose(robot, logger, "sleep_test", head_pose)
    logger.info(
        "Startup sleep test translation_m=%.9f translation_limit_m=%.9f translation_pass=%s "
        "rotation_rad=%.9f rotation_limit_rad=%.9f rotation_pass=%s body_yaw_considered=False "
        "antennas_considered=False motor_mode_considered=False decision=%s",
        translation_distance,
        _SLEEP_HEAD_TRANSLATION_TOLERANCE_M,
        translation_within_tolerance,
        rotation_angle,
        _SLEEP_HEAD_ROTATION_TOLERANCE_RAD,
        rotation_within_tolerance,
        sleep_pose_detected,
    )
    if not sleep_pose_detected:
        return False

    logger.info("Robot is in sleep pose; running wake-up movement.")
    logger.info("Startup wake invoked reason=sleep_pose_geometry")
    try:
        robot.enable_motors()
        _wake_up_with_trace(robot, wake_trace)
    except Exception as e:
        logger.error("Failed to run wake-up movement: %s", e)
        return False
    if stage1_instrumentation is None:
        raise PostWakeConvergenceError("post-wake convergence telemetry is unavailable")
    if freshness_recorder is None:
        authorized_snapshot = wait_for_post_wake_convergence(stage1_instrumentation, logger)
    else:
        authorized_snapshot = wait_for_post_wake_convergence(
            stage1_instrumentation, logger, freshness_recorder=freshness_recorder
        )
    if not authorized_snapshot:
        raise PostWakeConvergenceError("post-wake physical convergence timed out")
    logger.info("STAGE1M_ADOPTION_AFTER_WAKE_CONVERGENCE pending=True")
    logger.info("Startup wake completed reason=sleep_pose_geometry")
    _log_startup_pose(robot, logger, "after_wake")
    return authorized_snapshot


def run_go_to_sleep_tool(deps: ToolDependencies, logger: logging.Logger) -> dict[str, object]:
    """Run the shared go_to_sleep tool from synchronous shutdown paths."""
    try:
        return asyncio.run(GoToSleep()(deps))
    except Exception as e:
        logger.error("Failed to run go_to_sleep tool during shutdown: %s", e)
        return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
