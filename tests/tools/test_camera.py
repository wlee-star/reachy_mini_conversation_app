"""Tests for the camera tool."""

import base64
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.camera import (
    VISION_UNAVAILABLE_SPOKEN,
    Camera,
    match_camera_look_request,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("Reachy, look at what's in front of you.", True),
        ("use your camera and look at what's in front of you", True),
        ("what am I wearing?", True),
        ("can you see me?", True),
        ("who's this?", False),
        ("what time is it?", False),
    ],
)
def test_match_camera_look_request(transcript: str, expected: bool) -> None:
    """Look/camera scene asks match; FACE-ID and unrelated chat do not."""
    assert match_camera_look_request(transcript) is expected


@pytest.mark.asyncio
async def test_camera_tool_returns_base64_of_sdk_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool base64-encodes the JPEG bytes returned by the SDK."""
    monkeypatch.setenv("HF_VISION_ENABLED", "true")
    jpeg_bytes = b"\xff\xd8jpeg\xff\xd9"
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = jpeg_bytes

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert result["b64_im"] == base64.b64encode(jpeg_bytes).decode("utf-8")


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_no_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no frame available the tool returns an error."""
    monkeypatch.setenv("HF_VISION_ENABLED", "true")
    reachy_mini = MagicMock()
    reachy_mini.media.get_frame_jpeg.return_value = None

    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result


@pytest.mark.asyncio
async def test_camera_tool_reports_error_when_camera_disabled() -> None:
    """With the camera disabled the tool returns an error and never reads a frame."""
    reachy_mini = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=False,
    )

    result = await Camera()(deps, question="What color is this?")

    assert "error" in result
    reachy_mini.media.get_frame_jpeg.assert_not_called()


@pytest.mark.asyncio
async def test_camera_tool_fails_fast_when_vision_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text-only models get a structured vision_unavailable result and no capture."""
    monkeypatch.setenv("HF_VISION_ENABLED", "false")
    reachy_mini = MagicMock()
    deps = ToolDependencies(
        reachy_mini=reachy_mini,
        movement_manager=MagicMock(),
        camera_enabled=True,
    )

    result = await Camera()(deps, question="Who is this?")

    assert result == {
        "status": "vision_unavailable",
        "reason": "active_model_does_not_support_images",
        "spoken": VISION_UNAVAILABLE_SPOKEN,
    }
    reachy_mini.media.get_frame_jpeg.assert_not_called()
    assert Camera().wants_spoken_followup(result, None) is False
