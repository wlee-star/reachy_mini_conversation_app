"""Local administrative control for the authoritative head-tracking path."""

import time
import logging
from typing import Any, Protocol
from collections.abc import Callable


logger = logging.getLogger(__name__)

TRACKING_ACK_TIMEOUT_S = 1.5
TRACKING_ACK_POLL_INTERVAL_S = 0.05


class TrackingMovementManager(Protocol):
    """Expose the MovementManager operations needed by administrative tracking."""

    def set_head_tracking(self, enabled: bool) -> None:
        """Queue a tracking-state change on the movement worker."""
        ...

    def get_head_tracking_enabled(self) -> bool:
        """Return the application tracking state."""
        ...

    def get_status(self) -> dict[str, Any]:
        """Return movement-worker health and diagnostics."""
        ...

    def get_sdk_control_status(self) -> dict[str, object]:
        """Return SDK publication health."""
        ...


class HeadTrackingAdmin:
    """Control tracking through one existing MovementManager and verify daemon state."""

    def __init__(
        self,
        movement_manager: TrackingMovementManager | None,
        daemon_face_status: Callable[[], object] | None,
        *,
        daemon_tracking_status: Callable[[], object] | None = None,
        acknowledgement_timeout_s: float = TRACKING_ACK_TIMEOUT_S,
        poll_interval_s: float = TRACKING_ACK_POLL_INTERVAL_S,
    ) -> None:
        """Bind the existing manager and read-only daemon status source."""
        self._movement_manager = movement_manager
        self._daemon_face_status = daemon_face_status
        self._daemon_tracking_status = daemon_tracking_status
        self._acknowledgement_timeout_s = acknowledgement_timeout_s
        self._poll_interval_s = poll_interval_s

    def status(self) -> dict[str, object]:
        """Return application and daemon tracking state without changing either."""
        manager = self._movement_manager
        if manager is None:
            return self._unavailable_status("movement_manager_unavailable")

        try:
            movement_status = manager.get_status()
            sdk_control = manager.get_sdk_control_status()
            application_enabled = manager.get_head_tracking_enabled()
        except Exception as exc:
            logger.warning("Administrative tracking status failed: %s", exc)
            return self._unavailable_status("movement_manager_unavailable")

        worker_alive = movement_status.get("worker_alive") is True
        sdk_healthy = sdk_control.get("state") == "HEALTHY"
        face_payload = self._read_daemon_face()
        daemon_enabled = self._read_daemon_tracking_enabled()
        if daemon_enabled is True:
            daemon_state = "ENABLED"
        elif daemon_enabled is False:
            daemon_state = "DISABLED"
        else:
            daemon_state = "UNKNOWN"

        requested_daemon_state = "ENABLED" if application_enabled else "DISABLED"
        return {
            "available": worker_alive and sdk_healthy,
            "worker_alive": worker_alive,
            "sdk_control_state": sdk_control.get("state", "UNKNOWN"),
            "application_state": "ENABLED" if application_enabled else "DISABLED",
            "daemon_state": daemon_state,
            "daemon_tracking_enabled": daemon_enabled,
            "daemon_confirmed": daemon_state == requested_daemon_state,
            "tracking_state": self._tracking_state(application_enabled, daemon_enabled, face_payload),
            "face_detected": face_payload["detected"],
            "target_timestamp": face_payload["timestamp"],
            "target_age_s": None,
            "target_x": face_payload["x"],
            "target_y": face_payload["y"],
        }

    def set_enabled(self, enabled: bool) -> dict[str, object]:
        """Queue one state change and wait briefly for independent daemon evidence."""
        requested_state = "ENABLED" if enabled else "DISABLED"
        initial = self.status()
        if initial.get("available") is not True:
            return {
                **initial,
                "requested_state": requested_state,
                "command_accepted": False,
                "command_issued": False,
                "idempotent": False,
                "error": "tracking_control_unavailable",
            }
        if initial.get("application_state") == requested_state and initial.get("daemon_confirmed") is True:
            return {
                **initial,
                "requested_state": requested_state,
                "command_accepted": True,
                "command_issued": False,
                "idempotent": True,
            }

        manager = self._movement_manager
        if manager is None:
            return {
                **initial,
                "requested_state": requested_state,
                "command_accepted": False,
                "command_issued": False,
                "idempotent": False,
                "error": "movement_manager_unavailable",
            }
        try:
            manager.set_head_tracking(enabled)
        except Exception as exc:
            logger.warning("Administrative tracking command rejected: %s", exc)
            return {
                **initial,
                "requested_state": requested_state,
                "command_accepted": False,
                "command_issued": False,
                "idempotent": False,
                "error": f"tracking_command_rejected: {type(exc).__name__}",
            }

        deadline = time.monotonic() + self._acknowledgement_timeout_s
        latest = initial
        while True:
            latest = self.status()
            if latest.get("application_state") == requested_state and latest.get("daemon_confirmed") is True:
                return {
                    **latest,
                    "requested_state": requested_state,
                    "command_accepted": True,
                    "command_issued": True,
                    "idempotent": False,
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self._poll_interval_s, remaining))

        return {
            **latest,
            "requested_state": requested_state,
            "command_accepted": True,
            "command_issued": True,
            "idempotent": False,
            "daemon_confirmed": False,
            "error": "daemon_acknowledgement_timeout",
        }

    def _read_daemon_face(self) -> dict[str, object]:
        empty: dict[str, object] = {"detected": False, "timestamp": None, "x": None, "y": None}
        if self._daemon_face_status is None:
            return empty
        try:
            face = self._daemon_face_status()
        except Exception as exc:
            logger.warning("Daemon tracking status unavailable: %s", exc)
            return empty

        if isinstance(face, dict):
            detected = bool(face.get("detected", False))
            timestamp = face.get("ts")
            x = face.get("x")
            y = face.get("y")
        else:
            detected = bool(getattr(face, "detected", False))
            timestamp = getattr(face, "ts", None)
            x = getattr(face, "x", None)
            y = getattr(face, "y", None)
        return {"detected": detected, "timestamp": timestamp, "x": x, "y": y}

    def _read_daemon_tracking_enabled(self) -> bool | None:
        if self._daemon_tracking_status is None:
            return None
        try:
            enabled = self._daemon_tracking_status()
        except Exception as exc:
            logger.warning("Daemon tracking state unavailable: %s", exc)
            return None
        return enabled if isinstance(enabled, bool) else None

    @staticmethod
    def _tracking_state(
        application_enabled: bool,
        daemon_enabled: bool | None,
        face_payload: dict[str, object],
    ) -> str:
        if daemon_enabled is None:
            return "UNKNOWN"
        if application_enabled and not daemon_enabled:
            return "ENABLE_REQUESTED"
        if not application_enabled and daemon_enabled:
            return "DISABLE_REQUESTED"
        if not daemon_enabled:
            return "DISABLED"
        if face_payload["detected"] is True:
            return "ENABLED_FACE_DETECTED"
        if face_payload["timestamp"] is None:
            return "ENABLED_NO_OBSERVATION"
        return "ENABLED_NO_FACE"

    @staticmethod
    def _unavailable_status(error: str) -> dict[str, object]:
        return {
            "available": False,
            "worker_alive": False,
            "sdk_control_state": "UNKNOWN",
            "application_state": "UNKNOWN",
            "daemon_state": "UNKNOWN",
            "daemon_tracking_enabled": None,
            "daemon_confirmed": False,
            "tracking_state": "UNKNOWN",
            "face_detected": False,
            "target_timestamp": None,
            "target_age_s": None,
            "target_x": None,
            "target_y": None,
            "error": error,
        }
