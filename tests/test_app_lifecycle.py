from types import SimpleNamespace
from unittest.mock import MagicMock, call

import numpy as np

from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini_conversation_app import app_lifecycle
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


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


def test_wake_up_if_sleeping_enables_motors_before_wake_up() -> None:
    """Startup should enable sleeping motors before playing the wake-up movement."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = SLEEP_HEAD_POSE.copy()

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    assert robot.method_calls == [
        call.get_current_head_pose(),
        call.enable_motors(),
        call.wake_up(),
    ]


def test_prepare_robot_for_conversation_retries_enable_motors() -> None:
    """Wireless power-on often rejects enable_motors until the daemon is ready."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)
    robot.enable_motors.side_effect = [RuntimeError("not ready"), None]

    assert app_lifecycle.prepare_robot_for_conversation(robot, MagicMock(), attempts=2, delay_s=0)

    assert robot.enable_motors.call_count == 2
    robot.wake_up.assert_not_called()


def test_wake_up_if_sleeping_wakes_when_pose_unreadable() -> None:
    """After power-off the pose read can fail even though the robot is asleep."""
    robot = MagicMock()
    robot.get_current_head_pose.side_effect = RuntimeError("motors disabled")

    assert app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.enable_motors.assert_called_once()
    robot.wake_up.assert_called_once()


def test_wake_up_if_sleeping_skips_non_sleep_head_pose() -> None:
    """Startup should leave an already-awake robot alone."""
    robot = MagicMock()
    robot.get_current_head_pose.return_value = np.eye(4)

    assert not app_lifecycle.wake_up_if_sleeping(robot, MagicMock())

    robot.get_current_joint_positions.assert_not_called()
    robot.enable_motors.assert_not_called()
    robot.wake_up.assert_not_called()


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
