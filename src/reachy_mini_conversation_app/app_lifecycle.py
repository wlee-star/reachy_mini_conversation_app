"""Helpers for app startup and shutdown lifecycle behavior."""

import os
import json
import time
import asyncio
import logging
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import numpy.typing as npt

from reachy_mini import ReachyMini
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini.utils.interpolation import distance_between_poses
from reachy_mini_conversation_app.config import PROJECT_ROOT, config, set_custom_profile
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


def prepare_robot_for_conversation(
    robot: ReachyMini,
    logger: logging.Logger,
    *,
    attempts: int = 8,
    delay_s: float = 1.0,
) -> bool:
    """Enable motors after power-off, then wake if the head is in the sleep pose."""
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
    return wake_up_if_sleeping(robot, logger) or last_error is None


def wake_up_if_sleeping(robot: ReachyMini, logger: logging.Logger) -> bool:
    """Run the SDK wake-up movement when Reachy starts from the sleep pose."""
    try:
        head_pose = robot.get_current_head_pose()
    except Exception as e:
        logger.warning("Could not read robot pose before startup wake-up check: %s", e)
        try:
            robot.enable_motors()
            robot.wake_up()
        except Exception as wake_error:
            logger.error("Failed to wake Reachy after an unreadable pose: %s", wake_error)
            return False
        return True

    if not _is_sleep_head_pose(head_pose):
        return False

    logger.info("Robot is in sleep pose; running wake-up movement.")
    try:
        robot.enable_motors()
        robot.wake_up()
    except Exception as e:
        logger.error("Failed to run wake-up movement: %s", e)
        return False
    return True


def run_go_to_sleep_tool(deps: ToolDependencies, logger: logging.Logger) -> dict[str, object]:
    """Run the shared go_to_sleep tool from synchronous shutdown paths."""
    try:
        return asyncio.run(GoToSleep()(deps))
    except Exception as e:
        logger.error("Failed to run go_to_sleep tool during shutdown: %s", e)
        return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
