"""Presentation telemetry and single-flight camera route regressions."""

import time
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
