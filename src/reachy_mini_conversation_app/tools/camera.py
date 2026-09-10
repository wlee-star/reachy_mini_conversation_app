import re
import base64
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.config import hf_vision_input_enabled
from reachy_mini_conversation_app.activation import strip_transcript_name_prefix
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

VISION_UNAVAILABLE_REASON = "active_model_does_not_support_images"
VISION_UNAVAILABLE_SPOKEN = "I can capture the camera image, but my current local model can't interpret images."

# Explicit look/camera scene requests (not person-identity "who's this?").
_CAMERA_LOOK_RE = re.compile(
    r"(?:"
    r"\b(?:use|using)\b.{0,24}\bcamera\b"
    r"|\blook(?:\s+at)?\b.{0,40}\b(?:front|camera|holding|wearing|this|that|here)\b"
    r"|\bwhat(?:'s| is| am)\b.{0,24}\b(?:wearing|holding|in front|in the (?:frame|picture|photo))\b"
    r"|\b(?:can you|do you)\s+see\b"
    r"|\bdescribe\b.{0,24}\b(?:scene|what you see|what'?s in front)\b"
    r")",
    re.IGNORECASE,
)


def match_camera_look_request(transcript: str) -> bool:
    """Return whether the user asked Reachy to look via the camera (not FACE-ID who-is-this)."""
    text = strip_transcript_name_prefix(transcript).strip()
    if not text:
        return False
    return _CAMERA_LOOK_RE.search(text) is not None


class Camera(Tool):
    """Take a picture with the camera to see what is in front of the robot."""

    name = "camera"
    description = (
        "Take a picture with the camera to see what is in front of the robot. "
        "Use this when the user asks you to look at something, see what they are holding, "
        "check their appearance, describe the scene, or comment on how they look. "
        "Also use it when the user asks what you can see or wants your visual opinion. "
        "The camera is live, each call captures the current moment. "
        "If the user asks you to look without saying at what, do not ask for clarification, call this tool and describe what you see. "
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to observe or ask about in the picture. "
                    "Examples: what is the user holding, describe the user's outfit, "
                    "what do you see around you, how does the user look today."
                ),
            },
        },
        "required": ["question"],
    }

    def wants_spoken_followup(self, result: dict[str, Any] | None, error: str | None) -> bool:
        """Vision-unavailable replies are spoken deterministically by the realtime loop."""
        if error is not None:
            return True
        if isinstance(result, dict) and result.get("status") == "vision_unavailable":
            return False
        return self.needs_response

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and return the base64-encoded JPEG."""
        question = (kwargs.get("question") or "").strip()
        if not question:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", question[:120])

        if not deps.camera_enabled:
            logger.error("Camera is disabled")
            return {"error": "Camera is disabled"}

        if not hf_vision_input_enabled():
            logger.info("camera: skipping capture; active model does not support image input")
            return {
                "status": "vision_unavailable",
                "reason": VISION_UNAVAILABLE_REASON,
                "spoken": VISION_UNAVAILABLE_SPOKEN,
            }

        jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
        if jpeg_bytes is None:
            logger.error("No frame available from camera")
            return {"error": "No frame available"}

        return {"b64_im": base64.b64encode(jpeg_bytes).decode("utf-8")}
