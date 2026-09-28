import time
import logging
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.reachy_mini import INIT_HEAD_POSE, INIT_ANTENNAS_JOINT_POSITIONS
from reachy_mini.utils.interpolation import InterpolationTechnique
from reachy_mini_conversation_app.wake_trace import WakeTrace, WakeTraceCommand


class _FakeMotorMode:
    value = "enabled"


class _FakeBackendStatus:
    motor_control_mode = _FakeMotorMode()


class _FakeStatus:
    backend_status = _FakeBackendStatus()


class _FakeClient:
    def __init__(self) -> None:
        self.get_status_calls: list[bool] = []
        self.send_command = Mock(side_effect=AssertionError("wake tracing must not send commands"))

    def get_status(self, wait: bool = True, timeout: float = 5.0) -> _FakeStatus:
        self.get_status_calls.append(wait)
        return _FakeStatus()


class _FakeMedia:
    def __init__(self) -> None:
        self.play_sound = Mock()


class _FakeRobot:
    def __init__(self) -> None:
        self.client = _FakeClient()
        self.media = _FakeMedia()
        self.head = np.eye(4)
        self.body_yaw = 0.0
        self.antennas = (0.0, 0.0)
        self.goto_calls: list[dict[str, object]] = []
        self.set_target = Mock(side_effect=AssertionError("wake tracing must not add set_target calls"))
        self.prepare_stage1_horizontal_calibration = Mock(
            side_effect=AssertionError("wake tracing must not reach Stage 1 calibration")
        )

    def get_current_head_pose(self) -> np.ndarray:
        return self.head.copy()

    def get_current_joint_positions(self) -> tuple[list[float], list[float]]:
        return ([self.body_yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], list(self.antennas))

    def goto_target(
        self,
        head: np.ndarray | None = None,
        antennas: np.ndarray | list[float] | None = None,
        duration: float = 0.5,
        method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,
        body_yaw: float | None = 0.0,
    ) -> None:
        self.goto_calls.append(
            {
                "head": None if head is None else head.copy(),
                "antennas": None if antennas is None else list(antennas),
                "duration": duration,
                "method": method,
                "body_yaw": body_yaw,
            }
        )
        if head is not None:
            self.head = head.copy()
        if antennas is not None:
            self.antennas = (float(antennas[0]), float(antennas[1]))
        if body_yaw is not None:
            self.body_yaw = body_yaw


def test_tracer_samples_cached_state_without_issuing_commands() -> None:
    """Periodic samples use cached reads and never enter a control path."""
    robot = _FakeRobot()
    trace = WakeTrace(robot, logging.getLogger(__name__))

    trace.start()
    trace.close()

    assert trace.samples
    assert robot.client.get_status_calls
    assert set(robot.client.get_status_calls) == {False}
    robot.client.send_command.assert_not_called()
    robot.set_target.assert_not_called()
    robot.prepare_stage1_horizontal_calibration.assert_not_called()


def test_waypoint_observer_preserves_installed_sdk_wake_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Observation preserves every argument in the installed SDK wake sequence."""
    robot = _FakeRobot()
    trace = WakeTrace(robot, logging.getLogger(__name__))
    original_goto = robot.goto_target
    monkeypatch.setattr("reachy_mini.reachy_mini.time.sleep", lambda _duration: None)

    trace.start()
    with trace.observe_sdk_waypoints():
        ReachyMini.wake_up(robot)  # type: ignore[arg-type]
    trace.close()

    assert robot.goto_target == original_goto
    assert len(robot.goto_calls) == 3
    first, second, third = robot.goto_calls
    assert np.array_equal(first["head"], INIT_HEAD_POSE)
    assert first["antennas"] == INIT_ANTENNAS_JOINT_POSITIONS
    assert first["duration"] == 2
    assert first["method"] is InterpolationTechnique.MIN_JERK
    assert first["body_yaw"] == 0.0
    assert Rotation.from_matrix(np.asarray(second["head"])[:3, :3]).as_euler("xyz", degrees=True) == pytest.approx(
        [20.0, 0.0, 0.0]
    )
    assert second["antennas"] is None
    assert second["duration"] == pytest.approx(0.2)
    assert second["method"] is InterpolationTechnique.MIN_JERK
    assert second["body_yaw"] == 0.0
    assert np.array_equal(third["head"], INIT_HEAD_POSE)
    assert third["antennas"] is None
    assert third["duration"] == pytest.approx(0.2)
    assert third["method"] is InterpolationTechnique.MIN_JERK
    assert third["body_yaw"] == 0.0
    robot.media.play_sound.assert_called_once_with("wake_up.wav")
    robot.client.send_command.assert_not_called()
    robot.set_target.assert_not_called()
    assert [sample.phase for sample in trace.samples] == [
        "before_wake",
        "wake_neutral_before",
        "wake_neutral_after",
        "wake_roll_left_before",
        "wake_roll_left_after",
        "wake_return_neutral_before",
        "wake_return_neutral_after",
    ]


def test_maximum_angles_and_command_error_are_recorded() -> None:
    """Samples retain peak angles and target-versus-measurement errors."""
    robot = _FakeRobot()
    trace = WakeTrace(robot, logging.getLogger(__name__))
    target = create_head_pose(x=0.02, roll=20.0, pitch=-5.0, yaw=4.0, degrees=True)
    command = WakeTraceCommand(
        source="test_target",
        head=target,
        antennas=(0.2, -0.3),
        body_yaw=0.4,
        duration=0.2,
        interpolation="minjerk",
    )
    trace.start()
    robot.head = create_head_pose(x=0.01, roll=10.0, pitch=-2.0, yaw=1.0, degrees=True)
    robot.body_yaw = 0.1
    robot.antennas = (0.1, -0.1)
    trace.record_transition("test_measurement", command=command, active_task="test_task")
    trace.close()

    sample = trace.samples[-1]
    assert sample.translation_error_m == pytest.approx(0.01)
    assert sample.rotation_error_deg == pytest.approx(11.028, abs=0.01)
    assert sample.angular_error is not None
    assert sample.angular_error.roll == pytest.approx(-10.0, abs=0.2)
    assert sample.angular_error.pitch == pytest.approx(3.0, abs=0.2)
    assert sample.angular_error.yaw == pytest.approx(-3.0, abs=0.2)
    assert sample.body_yaw_error == pytest.approx(-0.3)
    assert sample.antenna_error == pytest.approx((-0.1, 0.2))
    assert trace.maxima["roll"].value == pytest.approx(10.0)
    assert trace.maxima["roll"].sample.phase == "test_measurement"


def test_post_wake_window_stops_and_shutdown_is_idempotent() -> None:
    """The bounded post-wake trace stops and emits one summary."""
    robot = _FakeRobot()
    logger = Mock(spec=logging.Logger)
    trace = WakeTrace(robot, logger, frequency_hz=20.0, post_return_s=0.05)

    trace.start()
    trace.wake_returned(np.eye(4))
    deadline = time.monotonic() + 1.0
    while trace.active and time.monotonic() < deadline:
        time.sleep(0.005)

    assert not trace.active
    trace.close()
    trace.close()
    completion_logs = [
        entry for entry in logger.info.call_args_list if entry.args[0].startswith("[WAKE_TRACE] complete")
    ]
    assert len(completion_logs) == 1


def test_wake_return_retains_the_exact_observed_final_waypoint() -> None:
    """Wake-return comparison retains duration and interpolation from the final SDK call."""
    robot = _FakeRobot()
    trace = WakeTrace(robot, logging.getLogger(__name__), post_return_s=0.0)
    final_command = WakeTraceCommand(
        source="wake_return_neutral",
        head=np.eye(4),
        antennas=None,
        body_yaw=0.0,
        duration=0.2,
        interpolation="minjerk",
    )

    trace.start()
    trace.record_transition("wake_return_neutral_after", command=final_command)
    trace.wake_returned(np.eye(4))
    trace.close()

    returned = trace.samples[-1]
    assert returned.phase == "wake_returned"
    assert returned.command is final_command
    assert returned.command.duration == pytest.approx(0.2)
    assert returned.command.interpolation == "minjerk"


def test_sample_failure_is_logged_and_tracer_still_stops() -> None:
    """Unavailable cached telemetry is visible and does not strand the thread."""
    robot = _FakeRobot()
    robot.get_current_head_pose = Mock(side_effect=RuntimeError("cached pose unavailable"))
    logger = Mock(spec=logging.Logger)
    trace = WakeTrace(robot, logger)

    trace.start()
    trace.close()

    assert not trace.active
    assert not trace.samples
    logger.warning.assert_called()
