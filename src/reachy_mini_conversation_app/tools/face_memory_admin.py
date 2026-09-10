"""Privacy controls for remembered face identities and profiles."""

from __future__ import annotations
import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.face_identity.settings import face_memory_enabled


logger = logging.getLogger(__name__)


class FaceMemoryAdmin(Tool):
    """List, forget, or update remembered people (profile + face identity)."""

    name = "face_memory_admin"
    description = (
        "Manage remembered people from face memory. Use list when asked who you remember. "
        "Use forget when asked to forget someone (removes embeddings and profile). "
        "Use update_profile to add freeform details about someone already remembered. "
        "Do not invent people who are not stored. Never claim success unless this tool returns "
        "status forgotten/updated with persisted=true, or status ok for list."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "forget", "update_profile"],
                "description": "Admin action to perform.",
            },
            "name": {
                "type": "string",
                "description": "Person display name for forget/update_profile.",
            },
            "person_id": {
                "type": "string",
                "description": "Optional stable person_id when known.",
            },
            "details": {
                "type": "string",
                "description": "Freeform details to append for update_profile.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Run a face-memory privacy/admin action."""
        if not face_memory_enabled():
            return {"error": "face_memory_disabled", "status": "disabled", "persisted": False}

        service = deps.face_memory_service
        if service is None:
            return {"error": "face_memory_unavailable", "persisted": False}

        action = str(kwargs.get("action") or "").strip().lower()
        logger.info("Tool call: face_memory_admin action=%s", action)

        pipeline = service.pipeline()
        if action == "list":
            people = await asyncio.to_thread(pipeline.list_remembered)
            names = [row["name"] or row["person_id"] for row in people]
            spoken = (
                "I don't remember anyone yet." if not names else "I remember " + ", ".join(str(n) for n in names) + "."
            )
            return {"status": "ok", "people": people, "persisted": False, "spoken": spoken}
        if action == "forget":
            name = kwargs.get("name") if isinstance(kwargs.get("name"), str) else None
            person_id = kwargs.get("person_id") if isinstance(kwargs.get("person_id"), str) else None
            result = await asyncio.to_thread(pipeline.forget_person, person_id=person_id, name=name)
            if result.get("status") == "forgotten":
                forgotten_id = str(result.get("person_id") or "")
                still_identity = pipeline.identities.get(forgotten_id) if forgotten_id else object()
                still_profile = pipeline.profiles.get(forgotten_id) if forgotten_id else object()
                if still_identity is None and still_profile is None:
                    label = result.get("name") or result.get("person_id")
                    result["persisted"] = True
                    result["spoken"] = f"Okay, I've forgotten {label}."
                else:
                    result = {
                        "status": "persistence_failed",
                        "persisted": False,
                        "person_id": forgotten_id or None,
                        "spoken": "I could not verify that person was forgotten.",
                    }
            else:
                result["persisted"] = False
                result["spoken"] = "I couldn't find that person in my memory."
            return result
        if action == "update_profile":
            name = kwargs.get("name")
            details = kwargs.get("details")
            if not isinstance(name, str) or not name.strip():
                return {"error": "name must be a non-empty string", "persisted": False}
            if not isinstance(details, str) or not details.strip():
                return {"error": "details must be a non-empty string", "persisted": False}
            profile = pipeline.profiles.get_by_name(name)
            if profile is None:
                return {
                    "status": "not_found",
                    "persisted": False,
                    "spoken": f"I don't have a profile for {name}.",
                }

            def _update() -> dict[str, Any]:
                updated = pipeline.profiles.upsert(profile.person_id, profile.name, append_note=details.strip())
                reread = pipeline.profiles.get(updated.person_id)
                if reread is None:
                    return {
                        "status": "persistence_failed",
                        "persisted": False,
                        "person_id": updated.person_id,
                        "name": updated.name,
                        "spoken": "I could not verify that profile update.",
                    }
                return {
                    "status": "updated",
                    "person_id": updated.person_id,
                    "name": updated.name,
                    "persisted": True,
                    "spoken": f"Updated what I remember about {updated.name}.",
                }

            return await asyncio.to_thread(_update)
        return {"error": f"unknown action: {action}", "persisted": False}
