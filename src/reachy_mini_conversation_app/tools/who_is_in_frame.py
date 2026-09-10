"""On-demand identity probe for the current Reachy camera frame."""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.face_identity.settings import (
    face_memory_enabled,
    face_memory_on_demand_recognition_enabled,
)
from reachy_mini_conversation_app.face_identity.speech_authority import spoken_for_recognition_result


logger = logging.getLogger(__name__)


class WhoIsInFrame(Tool):
    """Identify who is in the current Reachy camera frame (explicit user request only)."""

    name = "who_is_in_frame"
    description = (
        "Identify who is in the current camera frame using face memory (YuNet + SFace). "
        "Use only when the user asks who this/that person is. "
        "Do not use for automatic greetings or continuous scanning. "
        "Do not use the camera tool for person identity questions. "
        "Returns known/unknown/multiple_faces/no_face with an authoritative spoken field."
    )
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    needs_response = False

    def wants_spoken_followup(self, result: dict[str, Any] | None, error: str | None) -> bool:
        """Spoken identity is delivered deterministically from the tool result."""
        return False

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Recognize the current frame window once."""
        if not face_memory_enabled() or not face_memory_on_demand_recognition_enabled():
            payload: dict[str, Any] = {
                "error": "on_demand_recognition_disabled" if face_memory_enabled() else "face_memory_disabled",
                "status": "disabled",
            }
            payload["spoken"] = spoken_for_recognition_result(payload)
            return payload
        service = deps.face_memory_service
        if service is None:
            payload = {"error": "face_memory_unavailable", "status": "error"}
            payload["spoken"] = spoken_for_recognition_result(payload)
            return payload
        logger.info("Tool call: who_is_in_frame")
        return await asyncio.to_thread(service.who_is_in_frame, deps)
