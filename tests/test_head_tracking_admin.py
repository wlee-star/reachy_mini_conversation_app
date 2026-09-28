"""Safe administrative head-tracking control regressions."""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.console import LocalStream
from reachy_mini_conversation_app.head_tracking_admin import HeadTrackingAdmin


class FakeMovementManager:
    """Minimal authoritative manager fake with observable queue requests."""

    def __init__(self, *, enabled: bool = False, worker_alive: bool = True, sdk_state: str = "HEALTHY") -> None:
        """Create a controllable manager state."""
        self.enabled = enabled
        self.worker_alive = worker_alive
        self.sdk_state = sdk_state
        self.requests: list[bool] = []
        self.rejection: Exception | None = None

    def set_head_tracking(self, enabled: bool) -> None:
        """Record one queued tracking request."""
        if self.rejection is not None:
            raise self.rejection
        self.requests.append(enabled)
        self.enabled = enabled

    def get_head_tracking_enabled(self) -> bool:
        """Return the simulated application state."""
        return self.enabled

    def get_status(self) -> dict[str, object]:
        """Return simulated worker health."""
        return {"worker_alive": self.worker_alive}

    def get_sdk_control_status(self) -> dict[str, object]:
        """Return simulated SDK health."""
        return {"state": self.sdk_state}


def daemon_face_for(manager: FakeMovementManager) -> SimpleNamespace:
    """Represent daemon acknowledgement without exposing a control operation."""
    return SimpleNamespace(
        detected=manager.enabled,
        ts=12.5 if manager.enabled else None,
        x=0.1 if manager.enabled else None,
        y=-0.2 if manager.enabled else None,
    )


def daemon_tracking_for(manager: FakeMovementManager) -> bool:
    """Represent authoritative daemon tracking state."""
    return manager.enabled


def test_enable_uses_existing_movement_manager_and_confirms_daemon() -> None:
    """Enable routes through the supplied manager and waits for daemon evidence."""
    manager = FakeMovementManager()
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    result = admin.set_enabled(True)

    assert manager.requests == [True]
    assert result["command_accepted"] is True
    assert result["application_state"] == "ENABLED"
    assert result["daemon_state"] == "ENABLED"
    assert result["daemon_confirmed"] is True


def test_disable_uses_existing_movement_manager_and_confirms_daemon() -> None:
    """Disable routes through the supplied manager and waits for daemon clearing."""
    manager = FakeMovementManager(enabled=True)
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    result = admin.set_enabled(False)

    assert manager.requests == [False]
    assert result["application_state"] == "DISABLED"
    assert result["daemon_state"] == "DISABLED"
    assert result["daemon_confirmed"] is True


def test_status_is_read_only_and_exposes_face_target() -> None:
    """Status reads existing state without queuing a command."""
    manager = FakeMovementManager(enabled=True)
    daemon_status = MagicMock(return_value=daemon_face_for(manager))
    admin = HeadTrackingAdmin(
        manager,
        daemon_status,
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    result = admin.status()

    assert manager.requests == []
    assert result["face_detected"] is True
    assert result["target_timestamp"] == 12.5
    assert result["target_x"] == 0.1
    assert result["target_y"] == -0.2
    daemon_status.assert_called_once_with()


def test_unavailable_or_unhealthy_manager_fails_closed() -> None:
    """Missing ownership or unhealthy SDK state prevents commands."""
    absent = HeadTrackingAdmin(None, MagicMock())
    unhealthy_manager = FakeMovementManager(sdk_state="UNHEALTHY")
    unhealthy = HeadTrackingAdmin(
        unhealthy_manager,
        lambda: daemon_face_for(unhealthy_manager),
        daemon_tracking_status=lambda: daemon_tracking_for(unhealthy_manager),
    )

    absent_result = absent.set_enabled(True)
    unhealthy_result = unhealthy.set_enabled(True)

    assert absent_result["command_accepted"] is False
    assert unhealthy_result["command_accepted"] is False
    assert unhealthy_manager.requests == []


def test_command_rejection_fails_closed() -> None:
    """A rejected manager request is never reported as accepted."""
    manager = FakeMovementManager()
    manager.rejection = RuntimeError("preview owns movement")
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    result = admin.set_enabled(True)

    assert result["command_accepted"] is False
    assert result["daemon_confirmed"] is True
    assert manager.requests == []


def test_daemon_acknowledgement_timeout_is_bounded_and_distinct() -> None:
    """Manager acceptance remains distinct from bounded daemon confirmation."""
    manager = FakeMovementManager()
    cleared = SimpleNamespace(detected=False, ts=None, x=None, y=None)
    admin = HeadTrackingAdmin(
        manager,
        lambda: cleared,
        daemon_tracking_status=lambda: None,
        acknowledgement_timeout_s=0.01,
        poll_interval_s=0.001,
    )

    started = time.monotonic()
    result = admin.set_enabled(True)
    elapsed = time.monotonic() - started

    assert elapsed < 0.2
    assert result["command_accepted"] is True
    assert result["application_state"] == "ENABLED"
    assert result["daemon_state"] == "UNKNOWN"
    assert result["daemon_confirmed"] is False
    assert result["error"] == "daemon_acknowledgement_timeout"
    assert manager.requests == [True]


def test_repeated_disable_is_idempotent() -> None:
    """Repeated confirmed disable requests do not queue transitions."""
    manager = FakeMovementManager()
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    first = admin.set_enabled(False)
    second = admin.set_enabled(False)

    assert first["idempotent"] is True
    assert second["idempotent"] is True
    assert manager.requests == []


def test_repeated_enable_issues_only_one_command() -> None:
    """A confirmed enable makes the next enable request idempotent."""
    manager = FakeMovementManager()
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    first = admin.set_enabled(True)
    second = admin.set_enabled(True)

    assert first["command_issued"] is True
    assert second["idempotent"] is True
    assert second["command_issued"] is False
    assert manager.requests == [True]


def test_authoritative_enabled_without_observation_is_distinct() -> None:
    """Authoritative enable does not require a detector observation."""
    manager = FakeMovementManager(enabled=True)
    cleared = SimpleNamespace(detected=False, ts=None, x=None, y=None)
    admin = HeadTrackingAdmin(manager, lambda: cleared, daemon_tracking_status=lambda: True)

    result = admin.status()

    assert result["daemon_state"] == "ENABLED"
    assert result["daemon_confirmed"] is True
    assert result["tracking_state"] == "ENABLED_NO_OBSERVATION"


def test_enable_acknowledges_authoritative_state_without_face() -> None:
    """Enable succeeds from daemon truth before the first detector observation."""
    manager = FakeMovementManager()
    cleared = SimpleNamespace(detected=False, ts=None, x=None, y=None)
    admin = HeadTrackingAdmin(
        manager,
        lambda: cleared,
        daemon_tracking_status=lambda: daemon_tracking_for(manager),
    )

    result = admin.set_enabled(True)

    assert manager.requests == [True]
    assert result["command_accepted"] is True
    assert result["daemon_confirmed"] is True
    assert result["tracking_state"] == "ENABLED_NO_OBSERVATION"


def test_authoritative_enabled_without_face_is_distinct() -> None:
    """A no-face observation remains independent of daemon enable state."""
    manager = FakeMovementManager(enabled=True)
    no_face = SimpleNamespace(detected=False, ts=12.5, x=None, y=None)
    admin = HeadTrackingAdmin(manager, lambda: no_face, daemon_tracking_status=lambda: True)

    result = admin.status()

    assert result["daemon_state"] == "ENABLED"
    assert result["daemon_confirmed"] is True
    assert result["tracking_state"] == "ENABLED_NO_FACE"


def test_authoritative_enabled_with_face_is_distinct() -> None:
    """A detected face is reported without acting as enable proof."""
    manager = FakeMovementManager(enabled=True)
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: True,
    )

    result = admin.status()

    assert result["tracking_state"] == "ENABLED_FACE_DETECTED"


def test_authoritative_disabled_ignores_stale_face_for_confirmation() -> None:
    """Daemon false remains authoritative when face metadata is stale."""
    manager = FakeMovementManager()
    stale_face = SimpleNamespace(detected=True, ts=12.5, x=0.1, y=-0.2)
    admin = HeadTrackingAdmin(manager, lambda: stale_face, daemon_tracking_status=lambda: False)

    result = admin.status()

    assert result["daemon_state"] == "DISABLED"
    assert result["daemon_confirmed"] is True
    assert result["tracking_state"] == "DISABLED"


def test_missing_authoritative_field_is_unknown() -> None:
    """Older daemon status never aliases a missing field to disabled."""
    manager = FakeMovementManager()
    admin = HeadTrackingAdmin(
        manager,
        lambda: daemon_face_for(manager),
        daemon_tracking_status=lambda: None,
    )

    result = admin.status()

    assert result["daemon_tracking_enabled"] is None
    assert result["daemon_state"] == "UNKNOWN"
    assert result["daemon_confirmed"] is False
    assert result["tracking_state"] == "UNKNOWN"


def test_application_daemon_disagreements_fail_closed() -> None:
    """Requested transitions remain unconfirmed until daemon truth agrees."""
    enabling = HeadTrackingAdmin(
        FakeMovementManager(enabled=True),
        lambda: SimpleNamespace(detected=False, ts=None, x=None, y=None),
        daemon_tracking_status=lambda: False,
    ).status()
    disabling = HeadTrackingAdmin(
        FakeMovementManager(),
        lambda: SimpleNamespace(detected=False, ts=None, x=None, y=None),
        daemon_tracking_status=lambda: True,
    ).status()

    assert enabling["tracking_state"] == "ENABLE_REQUESTED"
    assert enabling["daemon_confirmed"] is False
    assert disabling["tracking_state"] == "DISABLE_REQUESTED"
    assert disabling["daemon_confirmed"] is False


def test_internal_routes_are_loopback_only_and_forward_to_admin() -> None:
    """Internal routes reject LAN clients and reuse the supplied controller."""
    app = FastAPI()
    admin = MagicMock()
    admin.status.return_value = {"available": True, "daemon_confirmed": True}
    admin.set_enabled.return_value = {"available": True, "daemon_confirmed": True}
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=MagicMock()),
        head_tracking_admin=admin,
    )
    stream._mount_dashboard_routes(app)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/internal/head-tracking/status").status_code == 200
        assert client.post("/api/internal/head-tracking/enable").status_code == 200
        assert client.post("/api/internal/head-tracking/disable").status_code == 200

    admin.status.assert_called_once_with()
    assert admin.set_enabled.call_args_list[0].args == (True,)
    assert admin.set_enabled.call_args_list[1].args == (False,)

    with TestClient(app, client=("192.168.0.20", 50000)) as client:
        assert client.get("/api/internal/head-tracking/status").status_code == 403
        assert client.post("/api/internal/head-tracking/enable").status_code == 403
        assert client.post("/api/internal/head-tracking/disable").status_code == 403

    assert admin.set_enabled.call_count == 2


def test_unconfirmed_daemon_state_returns_service_unavailable() -> None:
    """The route fails closed when the manager command lacks daemon proof."""
    app = FastAPI()
    admin = MagicMock()
    admin.set_enabled.return_value = {
        "available": True,
        "command_accepted": True,
        "daemon_confirmed": False,
    }
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=MagicMock()),
        head_tracking_admin=admin,
    )
    stream._mount_dashboard_routes(app)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post("/api/internal/head-tracking/enable")

    assert response.status_code == 503
    assert response.json()["command_accepted"] is True
    assert response.json()["daemon_confirmed"] is False
