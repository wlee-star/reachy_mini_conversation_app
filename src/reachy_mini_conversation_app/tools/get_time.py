"""LLM-callable wrapper around the deterministic local-time utility."""

import logging
from typing import Any

from reachy_mini_conversation_app.local_time import current_local_time
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class GetTime(Tool):
    """Return the current civil time from the system clock."""

    name = "get_time"
    description = (
        "Get the current local date and time from the system clock. "
        "Call this immediately for any time or date question, including "
        "'what time is it', 'what time is it in Sydney', 'what's today's date', "
        "and 'what day is it'. Speak the returned local_time and local_date exactly. "
        "Do not invent, guess, or adjust the numbers."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Optional IANA timezone. Empty uses the configured local timezone (Australia/Sydney).",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return structured current time from the system clock."""
        del deps
        timezone_raw = kwargs.get("timezone")
        timezone_name = timezone_raw.strip() if isinstance(timezone_raw, str) else None
        result = current_local_time(timezone_name=timezone_name)
        logger.info("[TIME] get_time supplied to model local_time=%s", result.get("local_time"))
        return result
