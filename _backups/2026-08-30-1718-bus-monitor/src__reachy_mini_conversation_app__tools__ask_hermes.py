import uuid
import logging
from typing import Any

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.tools.apex import Apex
from reachy_mini_conversation_app.hermes_client import (
    HermesClientError,
    HermesTimeoutError,
    HermesNotConfiguredError,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _spoken_apex_readings(snapshot: dict[str, Any]) -> str | None:
    if "error" in snapshot:
        return None
    probes = snapshot.get("water_parameters")
    if not isinstance(probes, dict) or not probes:
        return None
    parts: list[str] = []
    for name, probe in probes.items():
        if not isinstance(probe, dict):
            continue
        value = probe.get("value")
        if value is None:
            continue
        parts.append(f"{name} {value}")
        if len(parts) >= 6:
            break
    if not parts:
        return None
    return ", ".join(parts)


async def _live_apex_fallback_reply(deps: ToolDependencies) -> str:
    try:
        snapshot = await Apex()(deps, action="get_water_parameters")
    except Exception as exc:
        logger.warning("Apex fallback after Hermes narration failed: %s", exc)
        return "Historical trend data isn't available right now."
    spoken = _spoken_apex_readings(snapshot)
    if spoken is None:
        return "Historical trend data isn't available right now."
    return f"Historical trend data isn't available right now. Your current Apex readings are {spoken}."


class AskHermes(Tool):
    """Forward advanced delegated tasks to the Hermes Agent API server."""

    name = "ask_hermes"
    description = (
        "Ask Hermes for advanced delegated tasks: multi-step household requests, buses or trains, research, "
        "and cached reef tank trends, threading/thread summaries, parameter history, and ATO history. "
        "Use this immediately for 'trending', 'treading', 'threading', or 'tank trends' — do not use apex or "
        "reef_status for those. Do not use this for live reef tank status or current Apex numbers; use apex. "
        "This can take up to a few minutes. If a check is already running, do not call this again. "
        "Do not use this for simple Home Assistant lights/scenes/entity states or simple Apex current "
        "readings; use home_assistant or apex instead. Do not use this for chit-chat, jokes, "
        "general knowledge, or robot motion (head, dance, camera, sleep). "
        "Pass the user's request as a clear question or command."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The live-data question or device command to send to Hermes.",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Send one query to Hermes and return the assistant reply."""
        query = kwargs.get("query")
        if not isinstance(query, str) or not query.strip():
            logger.warning("ask_hermes: empty query")
            return {"error": "query must be a non-empty string"}

        if hermes_client.hermes_is_busy():
            logger.info("ask_hermes: previous request still running")
            return {
                "status": "already_running",
                "message": (
                    "A previous check is still running. Tell the user you are still on it. "
                    "Do not call ask_hermes again."
                ),
            }

        request_id = str(uuid.uuid4())
        logger.info("Hermes request started request_id=%s", request_id)
        try:
            reply = await hermes_client.send_to_hermes(
                query,
                hermes_client.get_hermes_session_id(),
                request_id=request_id,
            )
        except HermesNotConfiguredError:
            logger.warning("ask_hermes: Hermes Gateway is not configured")
            return {"error": "Hermes Gateway is not configured"}
        except HermesTimeoutError:
            logger.warning("ask_hermes: Hermes Gateway timed out request_id=%s", request_id)
            return {
                "error": "That check took too long. Ask me again if you still want it.",
            }
        except HermesClientError as exc:
            logger.warning("ask_hermes failed request_id=%s: %s", request_id, exc)
            return {"error": "I couldn't reach the household data service."}

        if hermes_client.is_trend_query(query) and hermes_client.is_process_narration(reply):
            logger.warning(
                "Hermes result rejected as process narration request_id=%s",
                request_id,
            )
            fallback = await _live_apex_fallback_reply(deps)
            return {
                "reply": fallback,
                "hermes_request_id": request_id,
                "trend_available": False,
                "source": "apex_live_fallback",
            }

        logger.info("Hermes request completed request_id=%s chars=%s", request_id, len(reply))
        payload: dict[str, Any] = {
            "reply": reply,
            "hermes_request_id": request_id,
            "source": "hermes",
        }
        if hermes_client.is_trend_query(query):
            payload["trend_available"] = True
        return payload
