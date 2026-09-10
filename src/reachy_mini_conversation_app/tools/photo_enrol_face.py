"""Explicit photo enrolment for personal face memory."""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.face_identity.settings import (
    face_memory_enabled,
    face_memory_photo_enrolment_enabled,
)
from reachy_mini_conversation_app.face_identity.speech_authority import (
    face_memory_photo_enrolment_disabled_result,
)


logger = logging.getLogger(__name__)


class PhotoEnrolFace(Tool):
    """Enter temporary photo enrolment mode when the user shows a picture."""

    # Kept for later dashboard/camera reuse; not exposed in the default profile while paused.
    name = "photo_enrol_face"
    description = (
        "Start or advance PHOTO ENROLMENT when the user wants Reachy to remember someone from a held-up photo "
        "or picture. Currently paused unless FACE_MEMORY_PHOTO_ENROLMENT_ENABLED=true. "
        "This is NOT live face recognition. Use action=scan, consent, set_name, set_details, or cancel."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["scan", "consent", "set_name", "set_details", "cancel"],
                "description": "Photo enrolment step to run.",
            },
            "accepted": {
                "type": "boolean",
                "description": "For action=consent: whether the user agreed to remember the person.",
            },
            "name": {
                "type": "string",
                "description": "For action=set_name: the person's display name.",
            },
            "details": {
                "type": "string",
                "description": "For action=set_details: freeform facts to store in the person profile.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Run one photo-enrolment step."""
        if not face_memory_enabled():
            return {
                "error": "face_memory_disabled",
                "status": "disabled",
                "persisted": False,
                "spoken": "Face memory is turned off right now.",
            }
        if not face_memory_photo_enrolment_enabled():
            return face_memory_photo_enrolment_disabled_result()

        service = deps.face_memory_service
        if service is None:
            return {"error": "face_memory_unavailable", "persisted": False}

        action = str(kwargs.get("action") or "").strip().lower()
        logger.info("Tool call: photo_enrol_face action=%s", action)

        if action == "scan":
            return await asyncio.to_thread(service.scan_photo, deps)
        if action == "consent":
            accepted = bool(kwargs.get("accepted", False))
            return await asyncio.to_thread(service.consent, accepted, deps.movement_manager)
        if action == "set_name":
            name = kwargs.get("name")
            if not isinstance(name, str) or not name.strip():
                return {"error": "name must be a non-empty string", "persisted": False}
            return await asyncio.to_thread(service.set_name_and_persist, name)
        if action == "set_details":
            details = kwargs.get("details")
            if not isinstance(details, str):
                details = ""
            return await asyncio.to_thread(service.save_details, details, deps.movement_manager)
        if action == "cancel":
            await asyncio.to_thread(service.abort_photo_enrolment, deps.movement_manager)
            return {
                "status": "cancelled",
                "persisted": False,
                "spoken": "Okay, I cancelled the photo enrolment.",
            }
        return {"error": f"unknown action: {action}", "persisted": False}
