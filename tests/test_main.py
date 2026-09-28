"""Tests for app-level runtime behavior."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import reachy_mini_conversation_app.main as main_mod
from reachy_mini_conversation_app.app_lifecycle import AuthorizedStartupSnapshot
from reachy_mini_conversation_app.face_tracking import PostWakeTelemetrySnapshot


@pytest.fixture(autouse=True)
def _disabled_daemon_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_mod.app_lifecycle,
        "ensure_daemon_head_tracking_disabled",
        MagicMock(
            return_value=main_mod.app_lifecycle.DaemonTrackingDisableResult(
                initial_state=False,
                command_issued=False,
                confirmed=True,
            )
        ),
    )


def test_inactivity_timeout_thread_goes_to_sleep() -> None:
    """The watchdog should use the shared sleep shutdown path once activity is too old."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())
    go_to_sleep = MagicMock(return_value={"status": "sleeping"})

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
        go_to_sleep=go_to_sleep,
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    go_to_sleep.assert_called_once_with()
    stream_manager.close.assert_not_called()


def test_inactivity_timeout_thread_closes_stream_manager_without_sleep_callback() -> None:
    """The watchdog should still close the stream when no sleep callback is available."""
    stream_manager = SimpleNamespace(seconds_since_activity=lambda: 10.0, close=MagicMock())

    thread = main_mod._start_inactivity_timeout_thread(
        timeout_minutes=0.0001,
        stream_manager=stream_manager,
        logger=MagicMock(),
        app_stop_event=threading.Event(),
    )

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    stream_manager.close.assert_called_once_with()


def test_post_wake_failure_blocks_movement_manager_and_conversation(monkeypatch) -> None:
    """A convergence failure parks startup before movement or conversation construction."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True)
    robot = MagicMock()
    logger = MagicMock()
    movement_manager = MagicMock()
    hold = MagicMock()
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(
        main_mod.app_lifecycle,
        "prepare_robot_for_conversation",
        MagicMock(side_effect=main_mod.app_lifecycle.PostWakeConvergenceError("timeout")),
    )
    monkeypatch.setattr(main_mod.app_lifecycle, "hold_post_wake_convergence_failure", hold)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", movement_manager)

    main_mod.run(args, robot=robot)

    hold.assert_called_once_with(logger, None)
    movement_manager.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_tracking_recovery_failure_blocks_before_movement_ownership(monkeypatch) -> None:
    """Unknown or unconfirmed daemon tracking blocks before Stage 2I and MovementManager."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True)
    robot = MagicMock()
    logger = MagicMock()
    hold = MagicMock()
    prepare = MagicMock()
    movement_manager = MagicMock()
    recovery = MagicMock(side_effect=main_mod.app_lifecycle.DaemonTrackingRecoveryError("unknown"))
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(main_mod.app_lifecycle, "ensure_daemon_head_tracking_disabled", recovery)
    monkeypatch.setattr(main_mod.app_lifecycle, "prepare_robot_for_conversation", prepare)
    monkeypatch.setattr(main_mod.app_lifecycle, "hold_post_wake_convergence_failure", hold)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", movement_manager)

    main_mod.run(args, robot=robot)

    recovery.assert_called_once_with(robot, logger, reason="startup_pre_movement_owner")
    hold.assert_called_once_with(logger, None)
    prepare.assert_not_called()
    movement_manager.assert_not_called()
    robot.start_head_tracking.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_tracking_recovery_precedes_stage2i_and_movement_manager(monkeypatch) -> None:
    """Disable-only recovery completes before pose validation or normal movement ownership."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True)
    robot = MagicMock()
    logger = MagicMock()
    hold = MagicMock()
    events: list[str] = []

    def recover(*_args, **_kwargs):
        events.append("recover")
        return main_mod.app_lifecycle.DaemonTrackingDisableResult(True, True, True)

    def reject_pose(*_args, **_kwargs):
        events.append("stage2i")
        raise main_mod.app_lifecycle.PostWakeConvergenceError("test stop")

    movement_manager = MagicMock(side_effect=lambda *_args, **_kwargs: events.append("manager"))
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(main_mod.app_lifecycle, "ensure_daemon_head_tracking_disabled", recover)
    monkeypatch.setattr(main_mod.app_lifecycle, "prepare_robot_for_conversation", reject_pose)
    monkeypatch.setattr(main_mod.app_lifecycle, "hold_post_wake_convergence_failure", hold)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", movement_manager)

    main_mod.run(args, robot=robot)

    assert events == ["recover", "stage2i"]
    movement_manager.assert_not_called()
    robot.start_head_tracking.assert_not_called()


def test_non_sleep_validation_failure_blocks_stage1m_and_all_movement(monkeypatch) -> None:
    """An ambiguous awake pose parks startup before MovementManager construction."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True, ui=False)
    robot = MagicMock()
    logger = MagicMock()
    movement_manager = MagicMock()
    hold = MagicMock()
    failure = main_mod.app_lifecycle.NonSleepStartupValidation(
        False,
        main_mod.app_lifecycle.NonSleepStartupFailureReason.POSE_OUTSIDE_PROVISIONAL_ENVELOPE,
        0,
    )
    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(
        main_mod.app_lifecycle,
        "prepare_robot_for_conversation",
        MagicMock(side_effect=main_mod.app_lifecycle.NonSleepStartupValidationError(failure)),
    )
    monkeypatch.setattr(main_mod.app_lifecycle, "hold_post_wake_convergence_failure", hold)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", movement_manager)

    main_mod.run(args, robot=robot)

    hold.assert_called_once_with(logger, None)
    movement_manager.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
    robot.wake_up.assert_not_called()


def test_stage2i_pose_real_gate_enters_degraded_hold_before_stage1m(monkeypatch) -> None:
    """The observed physical pose is rejected through the real main lifecycle gate."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", (7.25, 5.68, -3.75), degrees=True).as_matrix()
    robot = MagicMock()
    robot.get_current_head_pose.return_value = pose
    robot.get_current_joint_positions.return_value = ([0.0] * 7, [0.0, 0.0])
    logger = MagicMock()
    hold = MagicMock()
    movement_manager = MagicMock()

    class Instrumentation:
        calls = 0

        def prepare_telemetry_transport(self) -> object:
            return SimpleNamespace(endpoint="/api/daemon/status", duration_s=0.01, result="pass")

        def read_post_wake_telemetry(self, *, deadline: float) -> PostWakeTelemetrySnapshot:
            self.calls += 1
            return PostWakeTelemetrySnapshot(
                monotonic_time=0.0,
                present_head_pose=pose.copy(),
                present_head_joints=(0.0,) * 7,
                present_body_yaw=0.0,
                present_antennas=(0.0, 0.0),
                control_mode="enabled",
                daemon_timestamp=f"2026-09-20T00:00:{self.calls:02d}Z",
                daemon_ready=True,
                daemon_error=None,
                backend_last_alive=float(self.calls),
                backend_last_alive_age_s=0.01,
                control_loop_frequency_hz=50.0,
                active_move_count=0,
                acquisition_elapsed_s=0.01,
            )

        def close(self) -> None:
            pass

    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(main_mod, "Stage1Instrumentation", lambda _robot: Instrumentation())
    monkeypatch.setattr(main_mod.app_lifecycle, "_NON_SLEEP_STARTUP_TIMEOUT_S", 0.01)
    monkeypatch.setattr(main_mod.app_lifecycle, "_NON_SLEEP_STARTUP_SAMPLE_INTERVAL_S", 0.001)
    monkeypatch.setattr(main_mod.app_lifecycle, "hold_post_wake_convergence_failure", hold)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", movement_manager)

    main_mod.run(args, robot=robot)

    hold.assert_called_once_with(logger, None)
    movement_manager.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()
    robot.wake_up.assert_not_called()


def test_startup_publication_failure_enters_degraded_hold(monkeypatch) -> None:
    """Lifecycle observes a worker-side first-publication lease failure."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True, ui=False)
    robot = MagicMock()
    robot.client.host = "127.0.0.1"
    robot.client.port = 8000
    robot.client.is_connected.return_value = True
    logger = MagicMock()
    hold = MagicMock()
    snapshot = AuthorizedStartupSnapshot(
        head_transform=tuple(tuple(float(value) for value in row) for row in np.eye(4)),
        head_joints=(0.0,) * 7,
        body_yaw=0.0,
        antennas=(-0.1745, 0.1745),
        daemon_timestamp="2026-09-20T00:00:00Z",
        backend_last_alive=1.0,
        acquisition_monotonic=1.0,
        authorized_monotonic=1.0,
        validation_sample_count=10,
    )
    movement_manager = MagicMock()
    movement_manager.wait_for_startup_publication.return_value = False

    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(main_mod.app_lifecycle, "prepare_robot_for_conversation", MagicMock(return_value=snapshot))
    monkeypatch.setattr(main_mod.app_lifecycle, "require_fresh_authorized_startup_snapshot", MagicMock())
    monkeypatch.setattr(main_mod.app_lifecycle, "confirm_authorized_startup_snapshot", MagicMock(return_value=1.0))
    monkeypatch.setattr(main_mod.app_lifecycle, "require_fresh_startup_confirmation", MagicMock())
    monkeypatch.setattr(main_mod.app_lifecycle, "hold_post_wake_convergence_failure", hold)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", MagicMock(return_value=movement_manager))
    monkeypatch.setattr("reachy_mini_conversation_app.console.LocalStream", MagicMock())
    monkeypatch.setattr("reachy_mini_conversation_app.conversation_handler.ConversationHandler", MagicMock())
    monkeypatch.setattr("reachy_mini_conversation_app.huggingface_realtime.HuggingFaceRealtimeHandler", MagicMock())

    main_mod.run(args, robot=robot)

    movement_manager.set_startup_confirmation.assert_called_once()
    movement_manager.start.assert_called_once()
    movement_manager.wait_for_startup_publication.assert_called_once_with(0.25)
    hold.assert_called_once_with(logger, None)
    robot.enable_wobbling.assert_not_called()
    robot.set_target.assert_not_called()
    robot.goto_target.assert_not_called()


def test_stream_completion_reaches_existing_orderly_shutdown_finally(monkeypatch) -> None:
    """A graceful stream close must retain the production park-and-disconnect lifecycle."""
    args = SimpleNamespace(debug=False, no_motion=False, no_camera=True, ui=False)
    robot = MagicMock()
    robot.client.host = "127.0.0.1"
    robot.client.port = 8000
    robot.client.is_connected.return_value = True
    logger = MagicMock()
    instrumentation = MagicMock()
    wake_trace = MagicMock()
    movement_manager = MagicMock()
    movement_manager.get_sdk_control_status.return_value = {"state": "HEALTHY"}
    stream = MagicMock()
    park = MagicMock()

    monkeypatch.setattr(main_mod, "setup_logger", MagicMock(return_value=logger))
    monkeypatch.setattr(main_mod, "Stage1Instrumentation", MagicMock(return_value=instrumentation))
    monkeypatch.setattr(main_mod, "WakeTrace", MagicMock(return_value=wake_trace))
    monkeypatch.setattr(main_mod.app_lifecycle, "prepare_robot_for_conversation", MagicMock(return_value=False))
    monkeypatch.setattr(main_mod.app_lifecycle, "initialize_tools_with_default_fallback", MagicMock())
    monkeypatch.setattr(main_mod.app_lifecycle, "park_robot_for_orderly_shutdown", park)
    monkeypatch.setattr("reachy_mini_conversation_app.moves.MovementManager", MagicMock(return_value=movement_manager))
    monkeypatch.setattr("reachy_mini_conversation_app.console.LocalStream", MagicMock(return_value=stream))
    monkeypatch.setattr("reachy_mini_conversation_app.conversation_handler.ConversationHandler", MagicMock())
    monkeypatch.setattr("reachy_mini_conversation_app.huggingface_realtime.HuggingFaceRealtimeHandler", MagicMock())
    monkeypatch.setattr(
        "reachy_mini_conversation_app.config.resolve_app_timeout_minutes", MagicMock(return_value=None)
    )

    main_mod.run(args, robot=robot)

    stream.launch.assert_called_once_with()
    movement_manager.stop.assert_called_once_with(reset_to_neutral=False)
    park.assert_called_once_with(robot, instrumentation, logger)
    robot.client.disconnect.assert_called_once_with()
