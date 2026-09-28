"""Presentation telemetry and single-flight camera route regressions."""

import time
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reachy_mini_conversation_app.console import LocalStream


@pytest.mark.parametrize("muted,age,expected", [(False, 0, 0.7), (False, 2, 0.0), (True, 0, 0.0)])
def test_audio_status_uses_recent_real_levels(muted: bool, age: float, expected: float) -> None:
    """Both meters stop on silence or mute without altering audio processing."""
    stream = LocalStream(MagicMock(), SimpleNamespace(media=MagicMock()))
    stream._last_user_level = stream._last_assistant_level = 0.7
    stream._last_level_emit = {"user": time.monotonic() - age, "assistant": time.monotonic() - age}
    stream._mic_muted = stream._speaker_muted = muted
    payload = stream._dashboard_status_payload()
    assert payload["microphone_level"] == expected
    assert payload["speaker_level"] == expected
    assert stream._last_user_level == 0.7
    assert stream._last_assistant_level == 0.7


def test_emit_level_updates_dashboard_without_gradio_rpc() -> None:
    """Control-dashboard meters must work even when no Gradio browser client is connected."""
    stream = LocalStream(MagicMock(), SimpleNamespace(media=MagicMock()))
    stream._rpc = None
    frame = np.full(1600, 0.2, dtype=np.float32)
    stream._emit_level("user", frame)
    stream._emit_level("assistant", frame)
    assert stream._last_user_level > 0.0
    assert stream._last_assistant_level > 0.0
    payload = stream._dashboard_status_payload()
    assert payload["microphone_level"] == round(stream._last_user_level, 3)
    assert payload["speaker_level"] == round(stream._last_assistant_level, 3)


def test_emit_level_silence_stays_near_idle() -> None:
    """Near-silent frames map to a near-zero presentation level."""
    stream = LocalStream(MagicMock(), SimpleNamespace(media=MagicMock()))
    stream._rpc = None
    stream._emit_level("user", np.zeros(1600, dtype=np.float32))
    assert stream._last_user_level == 0.0
    assert stream._dashboard_status_payload()["microphone_level"] == 0.0


def test_camera_returns_current_frame_without_queue_or_cache() -> None:
    """Each request fetches the current SDK JPEG and never opens a new camera."""
    app = FastAPI()
    media = MagicMock()
    media.get_frame_jpeg.side_effect = [b"first", b"latest", None, RuntimeError("offline"), b"reconnected"]
    stream = LocalStream(MagicMock(), SimpleNamespace(media=media))
    stream._mount_dashboard_routes(app)
    with TestClient(app) as client:
        first = client.get("/api/dashboard/camera.jpg")
        latest = client.get("/api/dashboard/camera.jpg")
        assert first.content == b"first"
        assert latest.content == b"latest"
        assert latest.headers["cache-control"] == "no-store"
        assert "camera;dur=" in latest.headers["server-timing"]
        assert client.get("/api/dashboard/camera.jpg").status_code == 503
        assert client.get("/api/dashboard/camera.jpg").status_code == 503
        assert client.get("/api/dashboard/camera.jpg").content == b"reconnected"
    assert media.get_frame_jpeg.call_count == 5
    media.get_frame.assert_not_called()


def test_busy_camera_does_not_queue_another_sdk_call() -> None:
    """Overlapping requests fail promptly and a later request can recover."""
    app = FastAPI()
    media = MagicMock()
    media.get_frame_jpeg.return_value = b"latest"
    stream = LocalStream(MagicMock(), SimpleNamespace(media=media))
    stream._mount_dashboard_routes(app)
    with TestClient(app) as client:
        with stream._dashboard_camera_lock:
            assert client.get("/api/dashboard/camera.jpg").status_code == 503
            media.get_frame_jpeg.assert_not_called()
        assert client.get("/api/dashboard/camera.jpg").content == b"latest"


def test_camera_status_distinguishes_dashboard_request_freshness_from_unknown_source() -> None:
    """Dashboard request freshness is separate from unavailable source-frame timestamps."""
    media = MagicMock()
    stream = LocalStream(MagicMock(), SimpleNamespace(media=media))
    stream._dashboard_camera_last_request_at = time.monotonic() - 5.0
    stream._dashboard_camera_last_success_at = time.monotonic() - 5.0
    stream._dashboard_camera_last_frame_at = stream._dashboard_camera_last_success_at

    payload = stream._dashboard_status_payload()

    assert payload["camera_frame_status"]["state"] == "STALE"
    freshness = payload["camera_freshness"]
    assert freshness["camera_source_last_frame_at"] == "UNKNOWN"
    assert freshness["camera_app_last_frame_at"] == "UNKNOWN"
    assert freshness["camera_jpeg_age_s"] >= 5.0
    assert freshness["dashboard_camera_request_age_s"] >= 5.0
    assert freshness["dashboard_camera_last_failure_reason"] is None


def test_camera_busy_failure_reports_request_failure_without_source_failure() -> None:
    """Single-flight rejection is attributed to dashboard request contention."""
    app = FastAPI()
    media = MagicMock()
    media.get_frame_jpeg.return_value = b"latest"
    stream = LocalStream(MagicMock(), SimpleNamespace(media=media))
    stream._mount_dashboard_routes(app)
    with TestClient(app) as client:
        with stream._dashboard_camera_lock:
            assert client.get("/api/dashboard/camera.jpg").status_code == 503
        payload = stream._dashboard_status_payload()

    freshness = payload["camera_freshness"]
    assert freshness["dashboard_camera_last_failure_reason"] == "single-flight busy"
    assert freshness["camera_single_flight_busy"] is False
    assert freshness["camera_source_last_frame_at"] == "UNKNOWN"
    media.get_frame_jpeg.assert_not_called()


def test_camera_single_flight_busy_tracks_owner_lifetime() -> None:
    """Current busy state follows the owning request and is not cleared by a rejected request."""
    app = FastAPI()
    media = MagicMock()
    owner_started = threading.Event()
    release_owner = threading.Event()

    def blocking_jpeg() -> bytes:
        owner_started.set()
        assert release_owner.wait(timeout=2.0)
        return b"latest"

    media.get_frame_jpeg.side_effect = blocking_jpeg
    stream = LocalStream(MagicMock(), SimpleNamespace(media=media))
    stream._mount_dashboard_routes(app)

    with TestClient(app) as client:
        owner_response: list[object] = []

        def request_owner() -> None:
            owner_response.append(client.get("/api/dashboard/camera.jpg"))

        owner = threading.Thread(target=request_owner)
        owner.start()
        try:
            assert owner_started.wait(timeout=1.0)
            assert client.get("/api/dashboard/camera.jpg").status_code == 503
            busy_payload = stream._dashboard_status_payload()
            assert busy_payload["camera_freshness"]["camera_single_flight_busy"] is True

            release_owner.set()
            owner.join(timeout=2.0)
            assert not owner.is_alive()
            assert owner_response[0].content == b"latest"
            idle_payload = stream._dashboard_status_payload()
            assert idle_payload["camera_freshness"]["camera_single_flight_busy"] is False
            assert idle_payload["camera_freshness"]["dashboard_camera_last_failure_reason"] is None
        finally:
            release_owner.set()
            owner.join(timeout=2.0)


def test_stage1_internal_routes_forward_preview_and_one_shot() -> None:
    """The loopback-only route forwards session diagnostics without camera data."""
    app = FastAPI()
    prepare = MagicMock(return_value={"session_id": "session-1", "movement_count_planned": 1})
    execute = MagicMock(return_value={"session_id": "session-1", "accepted": True, "command_count": 1})
    cancel = MagicMock()
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=MagicMock()),
        stage1_prepare=prepare,
        stage1_execute=execute,
        stage1_cancel=cancel,
    )
    stream._mount_dashboard_routes(app)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        preview = client.post(
            "/api/internal/face-track-stage1/prepare",
            json={"raw_geometry_delta_yaw": 24.0},
        )
        result = client.post(
            "/api/internal/face-track-stage1/execute",
            json={"session_id": "session-1"},
        )
        cancelled = client.post(
            "/api/internal/face-track-stage1/cancel",
            json={"session_id": "session-1"},
        )

    assert preview.json() == {"ok": True, "session_id": "session-1", "movement_count_planned": 1}
    assert result.json() == {
        "ok": True,
        "session_id": "session-1",
        "accepted": True,
        "command_count": 1,
    }
    prepare.assert_called_once_with(24.0)
    execute.assert_called_once_with("session-1")
    assert cancelled.json() == {"ok": True, "status": "cancelled"}
    cancel.assert_called_once_with("session-1")


def test_stage1_internal_routes_reject_non_loopback_clients() -> None:
    """The calibration control path is unavailable to LAN clients."""
    app = FastAPI()
    prepare = MagicMock()
    cancel = MagicMock()
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=MagicMock()),
        stage1_prepare=prepare,
        stage1_cancel=cancel,
    )
    stream._mount_dashboard_routes(app)

    with TestClient(app, client=("192.168.0.20", 50000)) as client:
        response = client.post(
            "/api/internal/face-track-stage1/prepare",
            json={"raw_geometry_delta_yaw": 24.0},
        )
        cancel_response = client.post(
            "/api/internal/face-track-stage1/cancel",
            json={"session_id": "session-1"},
        )

    assert response.status_code == 403
    prepare.assert_not_called()
    assert cancel_response.status_code == 403
    cancel.assert_not_called()


def test_stage1_preview_only_routes_start_inspect_and_release() -> None:
    """Loopback diagnostics expose only the owned preview lifecycle."""
    app = FastAPI()
    start = MagicMock(return_value={"active": True, "session_id": "preview-1", "owner_id": "preview-1"})
    status = MagicMock(return_value={"active": True, "session_id": "preview-1", "stabilization_complete": True})
    release = MagicMock(return_value={"active": False, "session_id": None, "owner_id": None})
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=MagicMock()),
        stage1_preview_start=start,
        stage1_preview_status=status,
        stage1_preview_release=release,
    )
    stream._mount_dashboard_routes(app)

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        started = client.post("/api/internal/face-track-stage1/preview/start")
        inspected = client.get(
            "/api/internal/face-track-stage1/preview/status",
            params={"session_id": "preview-1"},
        )
        released = client.post(
            "/api/internal/face-track-stage1/preview/release",
            json={"session_id": "preview-1"},
        )

    assert started.json() == {"ok": True, "active": True, "session_id": "preview-1", "owner_id": "preview-1"}
    assert inspected.json() == {
        "ok": True,
        "active": True,
        "session_id": "preview-1",
        "stabilization_complete": True,
    }
    assert released.json() == {"ok": True, "active": False, "session_id": None, "owner_id": None}
    start.assert_called_once_with()
    status.assert_called_once_with("preview-1")
    release.assert_called_once_with("preview-1")


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/internal/face-track-stage1/preview/start", None),
        ("get", "/api/internal/face-track-stage1/preview/status", None),
        ("post", "/api/internal/face-track-stage1/preview/release", {"session_id": "preview-1"}),
    ],
)
def test_stage1_preview_only_routes_reject_non_loopback_clients(
    method: str,
    path: str,
    payload: dict[str, str] | None,
) -> None:
    """Preview diagnostics remain confined to the local application host."""
    app = FastAPI()
    start = MagicMock()
    status = MagicMock()
    release = MagicMock()
    stream = LocalStream(
        MagicMock(),
        SimpleNamespace(media=MagicMock()),
        stage1_preview_start=start,
        stage1_preview_status=status,
        stage1_preview_release=release,
    )
    stream._mount_dashboard_routes(app)

    with TestClient(app, client=("192.168.0.20", 50000)) as client:
        response = client.request(method, path, json=payload)

    assert response.status_code == 403
    start.assert_not_called()
    status.assert_not_called()
    release.assert_not_called()
