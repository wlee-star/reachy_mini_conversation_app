import re
import asyncio
import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_SLEEP_COMMAND_RE = re.compile(
    r"\b(?:"
    r"(?:please\s+)?go\s+to\s+sleep|"
    r"(?:please\s+)?go\s+to\s+bed|"
    r"(?:please\s+)?sleep\s+now|"
    r"time\s+to\s+sleep|"
    r"put\s+(?:yourself|the\s+robot)\s+to\s+sleep"
    r")\b",
    re.IGNORECASE,
)
_SLEEP_NON_COMMAND_RE = re.compile(
    r"\b(?:"
    r"why\b|"
    r"what\s+time\s+should|"
    r"when\s+(?:should|do|does|did)|"
    r"how\s+(?:do|does|did|should|long)|"
    r"should\s+i\b|"
    r"do\s+people\b|"
    r"tell\s+me\s+(?:a\s+)?story|"
    r"story\s+about|"
    r"about\s+going\s+to\s+sleep|"
    r"(?:couldn'?t|can'?t|could\s+not|cannot)\s+go\s+to\s+sleep|"
    r"i\s+(?:couldn'?t|can'?t|could\s+not|cannot)\b|"
    r"people\s+go\s+to\s+sleep"
    r")\b",
    re.IGNORECASE,
)


def match_sleep_intent(transcript: str) -> bool:
    """Return whether the utterance is a direct robot sleep command."""
    text = transcript.lower().strip().replace("'", "")
    text = re.sub(r"[.!?,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False
    if _SLEEP_NON_COMMAND_RE.search(text):
        return False
    return _SLEEP_COMMAND_RE.search(text) is not None


class GoToSleep(Tool):
    """Put Reachy Mini to sleep and stop the current app."""

    name = "go_to_sleep"
    description = (
        "Use when you are sure the user wants the Reachy Mini robot to go to sleep, stop the current app, shut down this app, "
        "or end the conversation. Do not use for idle turns, sleepy emotions, silence, or ambiguous requests."
    )
    needs_response = False
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Put Reachy Mini to sleep and request app shutdown."""
        if deps.go_to_sleep is None:
            return {"error": "go_to_sleep is unavailable in this runtime"}

        logger.info("Tool call: go_to_sleep")
        try:
            return await asyncio.to_thread(deps.go_to_sleep)
        except Exception as e:
            logger.error("go_to_sleep failed: %s", e)
            return {"error": f"go_to_sleep failed: {type(e).__name__}: {e}"}
