import logging
from typing import Any

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HermesClientError,
    HermesTimeoutError,
    HermesNotConfiguredError,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


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

        try:
            reply = await hermes_client.send_to_hermes(query, hermes_client.get_hermes_session_id())
        except HermesNotConfiguredError:
            logger.warning("ask_hermes: Hermes Gateway is not configured")
            return {"error": "Hermes Gateway is not configured"}
        except HermesTimeoutError:
            logger.warning("ask_hermes: Hermes Gateway timed out")
            return {
                "error": "That check took too long. Ask me again if you still want it.",
            }
        except HermesClientError as exc:
            logger.warning("ask_hermes failed: %s", exc)
            return {"error": "I couldn't reach the household data service."}

        logger.info("Tool call: ask_hermes chars=%s", len(reply))
        return {"reply": reply}
