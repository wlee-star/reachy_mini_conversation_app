"""Read-only startup wake trajectory instrumentation."""

import time
import logging
import threading
from types import TracebackType
from typing import Protocol
from contextlib import AbstractContextManager
from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from reachy_mini.io.protocol import DaemonStatus
from reachy_mini.utils.interpolation import InterpolationTechnique
from reachy_mini_conversation_app.face_tracking import EulerAngles, angular_delta, pose_euler_degrees


WAKE_TRACE_FREQUENCY_HZ = 20.0
WAKE_TRACE_POST_RETURN_S = 5.0


class WakeTraceClient(Protocol):
    """Expose cached daemon status without waiting for new telemetry."""

    def get_status(self, wait: bool = True, timeout: float = 5.0) -> DaemonStatus:
        """Return the latest cached daemon status."""
        ...


class WakeTraceRobot(Protocol):
    """Expose cached state and the public SDK wake calls being observed."""

    @property
    def client(self) -> WakeTraceClient:
        """Return the existing SDK client."""
        ...

    def get_current_head_pose(self) -> npt.NDArray[np.float64]:
        """Return the latest cached head pose."""
        ...

    def get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        """Return the latest cached head and antenna joints."""
        ...

    def goto_target(
        self,
        head: npt.NDArray[np.float64] | None = None,
        antennas: npt.NDArray[np.float64] | list[float] | None = None,
        duration: float = 0.5,
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,
        body_yaw: float | None = 0.0,
    ) -> None:
        """Run the SDK target interpolation."""
        ...


@dataclass(frozen=True)
class WakeTraceCommand:
    """One target observed at an existing startup transition."""

    source: str
    head: npt.NDArray[np.float64] | None
    antennas: tuple[float, float] | None
    body_yaw: float | None
    duration: float | None
    interpolation: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe target values."""
        angles = pose_euler_degrees(self.head) if self.head is not None else None
        return {
            "source": self.source,
            "head": self.head.tolist() if self.head is not None else None,
            "head_rpy_deg": (
                {"roll": angles.roll, "pitch": angles.pitch, "yaw": angles.yaw} if angles is not None else None
            ),
            "antennas": list(self.antennas) if self.antennas is not None else None,
            "body_yaw": self.body_yaw,
            "duration": self.duration,
            "interpolation": self.interpolation,
        }


@dataclass(frozen=True)
class WakeTraceSample:
    """One cached physical-state observation."""

    wall_time: float
    monotonic_time: float
    phase: str
    head: npt.NDArray[np.float64]
    angles: EulerAngles
    body_yaw: float
    antennas: tuple[float, float]
    motor_mode: str
    active_task: str | None
    command: WakeTraceCommand | None
    translation_error_m: float | None
    rotation_error_deg: float | None
    angular_error: EulerAngles | None
    body_yaw_error: float | None
    antenna_error: tuple[float, float] | None

    def to_dict(self) -> dict[str, object]:
        """Return a durable log representation."""
        return {
            "wall_time": self.wall_time,
            "monotonic_time": self.monotonic_time,
            "phase": self.phase,
            "head": self.head.tolist(),
            "head_rpy_deg": {
                "roll": self.angles.roll,
                "pitch": self.angles.pitch,
                "yaw": self.angles.yaw,
            },
            "body_yaw": self.body_yaw,
            "antennas": list(self.antennas),
            "motor_mode": self.motor_mode,
            "active_task": self.active_task,
            "command": self.command.to_dict() if self.command is not None else None,
            "translation_error_m": self.translation_error_m,
            "rotation_error_deg": self.rotation_error_deg,
            "angular_error_deg": (
                {
                    "roll": self.angular_error.roll,
                    "pitch": self.angular_error.pitch,
                    "yaw": self.angular_error.yaw,
                }
                if self.angular_error is not None
                else None
            ),
            "body_yaw_error": self.body_yaw_error,
            "antenna_error": list(self.antenna_error) if self.antenna_error is not None else None,
        }


@dataclass(frozen=True)
class WakeTraceMaximum:
    """Largest absolute value observed for one Euler axis."""

    axis: str
    value: float
    sample: WakeTraceSample


class _WakeWaypointObservation(AbstractContextManager[None]):
    """Observe the SDK's existing wake waypoints without replacing their targets."""

    def __init__(self, trace: "WakeTrace", robot: WakeTraceRobot) -> None:
        self._trace = trace
        self._robot = robot
        self._original_goto: Callable[..., None] | None = None
        self._waypoint_index = 0

    def __enter__(self) -> None:
        self._original_goto = self._robot.goto_target

        def observed_goto(
            head: npt.NDArray[np.float64] | None = None,
            antennas: npt.NDArray[np.float64] | list[float] | None = None,
            duration: float = 0.5,
            method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,
            body_yaw: float | None = 0.0,
        ) -> None:
            assert self._original_goto is not None
            self._waypoint_index += 1
            name = {
                1: "wake_neutral",
                2: "wake_roll_left",
                3: "wake_return_neutral",
            }.get(self._waypoint_index, f"wake_waypoint_{self._waypoint_index}")
            antenna_values = None if antennas is None else (float(antennas[0]), float(antennas[1]))
            command = WakeTraceCommand(
                source=name,
                head=None if head is None else np.asarray(head, dtype=np.float64).copy(),
                antennas=antenna_values,
                body_yaw=body_yaw,
                duration=duration,
                interpolation=method.value,
            )
            self._trace.record_transition(f"{name}_before", command=command, active_task="sdk_goto_target")
            try:
                self._original_goto(
                    head=head,
                    antennas=antennas,
                    duration=duration,
                    method=method,
                    body_yaw=body_yaw,
                )
            finally:
                self._trace.record_transition(f"{name}_after", command=command, active_task=None)

        setattr(self._robot, "goto_target", observed_goto)
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._original_goto is not None:
            setattr(self._robot, "goto_target", self._original_goto)
        return None


class WakeTrace:
    """Sample cached robot state around an automatic startup wake."""

    def __init__(
        self,
        robot: WakeTraceRobot,
        logger: logging.Logger,
        *,
        frequency_hz: float = WAKE_TRACE_FREQUENCY_HZ,
        post_return_s: float = WAKE_TRACE_POST_RETURN_S,
    ) -> None:
        """Bind the tracer to one existing SDK-owned robot connection."""
        if frequency_hz <= 0.0 or frequency_hz > WAKE_TRACE_FREQUENCY_HZ:
            raise ValueError(f"wake trace frequency must be within (0, {WAKE_TRACE_FREQUENCY_HZ}]")
        if post_return_s < 0.0:
            raise ValueError("wake trace post-return duration must be non-negative")
        self._robot = robot
        self._logger = logger
        self._period = 1.0 / frequency_hz
        self._post_return_s = post_return_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active = False
        self._phase = "inactive"
        self._active_task: str | None = None
        self._command: WakeTraceCommand | None = None
        self._post_return_deadline: float | None = None
        self._samples: list[WakeTraceSample] = []
        self._maxima: dict[str, WakeTraceMaximum] = {}
        self._summary_logged = False

    @property
    def active(self) -> bool:
        """Return whether a wake trace is collecting samples."""
        with self._lock:
            return self._active

    @property
    def samples(self) -> tuple[WakeTraceSample, ...]:
        """Return an immutable snapshot of captured samples."""
        with self._lock:
            return tuple(self._samples)

    @property
    def maxima(self) -> dict[str, WakeTraceMaximum]:
        """Return the largest observed absolute Euler values."""
        with self._lock:
            return dict(self._maxima)

    def observe_sdk_waypoints(self) -> AbstractContextManager[None]:
        """Observe the public SDK goto calls made by wake_up()."""
        return _WakeWaypointObservation(self, self._robot)

    def start(self) -> None:
        """Start read-only cached-state sampling before automatic wake."""
        with self._lock:
            if self._active:
                raise RuntimeError("wake trace is already active")
            self._active = True
            self._phase = "before_wake"
            self._active_task = "sdk_wake_up"
            self._command = None
            self._post_return_deadline = None
            self._samples.clear()
            self._maxima.clear()
            self._summary_logged = False
            self._stop.clear()
        self._capture_sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True, name="wake-trace")
        self._thread.start()

    def record_transition(
        self,
        phase: str,
        *,
        command: WakeTraceCommand | None = None,
        active_task: str | None = None,
    ) -> None:
        """Record an immediate cached sample at a startup transition."""
        with self._lock:
            if not self._active:
                return
            self._phase = phase
            self._active_task = active_task
            if command is not None:
                self._command = command
        self._capture_sample()

    def wake_returned(self, final_target: npt.NDArray[np.float64]) -> None:
        """Mark SDK wake completion and begin the post-wake trace window."""
        with self._lock:
            if not self._active:
                return
            command = self._command
            if command is None or command.source != "wake_return_neutral":
                command = WakeTraceCommand(
                    source="wake_final_target",
                    head=np.asarray(final_target, dtype=np.float64).copy(),
                    antennas=None,
                    body_yaw=0.0,
                    duration=None,
                    interpolation=None,
                )
            self._post_return_deadline = time.monotonic() + self._post_return_s
        self.record_transition("wake_returned", command=command, active_task=None)

    def close(self) -> None:
        """Stop sampling and join the tracer thread without commanding the robot."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(self._period * 2.0, 0.2))
        with self._lock:
            self._active = False
        self._log_summary()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._period):
            with self._lock:
                deadline = self._post_return_deadline
            if deadline is not None and time.monotonic() >= deadline:
                break
            self._capture_sample()
        with self._lock:
            self._active = False
        self._log_summary()

    def _capture_sample(self) -> None:
        with self._lock:
            if not self._active:
                return
            phase = self._phase
            active_task = self._active_task
            command = self._command
        try:
            monotonic_time = time.monotonic()
            wall_time = time.time()
            head = np.asarray(self._robot.get_current_head_pose(), dtype=np.float64).copy()
            head_joints, antenna_positions = self._robot.get_current_joint_positions()
            angles = pose_euler_degrees(head)
            motor_mode = "unavailable"
            status = self._robot.client.get_status(wait=False)
            if status.backend_status is not None:
                motor_mode = status.backend_status.motor_control_mode.value

            translation_error: float | None = None
            rotation_error: float | None = None
            angular_error: EulerAngles | None = None
            if command is not None and command.head is not None:
                target_angles = pose_euler_degrees(command.head)
                translation_error = float(np.linalg.norm(head[:3, 3] - command.head[:3, 3]))
                rotation_error = float(
                    np.degrees(
                        (
                            Rotation.from_matrix(command.head[:3, :3]).inv() * Rotation.from_matrix(head[:3, :3])
                        ).magnitude()
                    )
                )
                angular_error = angular_delta(target_angles, angles)

            body_yaw = float(head_joints[0])
            antennas = (float(antenna_positions[0]), float(antenna_positions[1]))
            body_yaw_error = None if command is None or command.body_yaw is None else body_yaw - command.body_yaw
            antenna_error = (
                None
                if command is None or command.antennas is None
                else (antennas[0] - command.antennas[0], antennas[1] - command.antennas[1])
            )
            sample = WakeTraceSample(
                wall_time=wall_time,
                monotonic_time=monotonic_time,
                phase=phase,
                head=head,
                angles=angles,
                body_yaw=body_yaw,
                antennas=antennas,
                motor_mode=motor_mode,
                active_task=active_task,
                command=command,
                translation_error_m=translation_error,
                rotation_error_deg=rotation_error,
                angular_error=angular_error,
                body_yaw_error=body_yaw_error,
                antenna_error=antenna_error,
            )
        except (
            AssertionError,
            AttributeError,
            ConnectionError,
            IndexError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            self._logger.warning("[WAKE_TRACE] sample unavailable phase=%s error=%s", phase, exc)
            return

        with self._lock:
            self._samples.append(sample)
            for axis, value in (
                ("roll", sample.angles.roll),
                ("pitch", sample.angles.pitch),
                ("yaw", sample.angles.yaw),
            ):
                maximum = self._maxima.get(axis)
                if maximum is None or abs(value) > abs(maximum.value):
                    self._maxima[axis] = WakeTraceMaximum(axis=axis, value=value, sample=sample)
        self._logger.info("[WAKE_TRACE] sample=%s", sample.to_dict())

    def _log_summary(self) -> None:
        with self._lock:
            if not self._samples or self._summary_logged:
                return
            self._summary_logged = True
            summary = {
                axis: {
                    "value": maximum.value,
                    "wall_time": maximum.sample.wall_time,
                    "monotonic_time": maximum.sample.monotonic_time,
                    "phase": maximum.sample.phase,
                    "head": maximum.sample.head.tolist(),
                    "active_task": maximum.sample.active_task,
                    "command": maximum.sample.command.to_dict() if maximum.sample.command is not None else None,
                }
                for axis, maximum in self._maxima.items()
            }
        self._logger.info("[WAKE_TRACE] complete samples=%s maxima=%s", len(self.samples), summary)
