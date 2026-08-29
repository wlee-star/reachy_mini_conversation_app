"""HTTP client for the Hermes Agent OpenAI-compatible API server."""

import time
import uuid
import asyncio
import logging

import httpx

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

# Hermes can load a large system prompt and run multi-step tool loops; delegated
# queries measured 45-90s. Cap at 180s so a hung gateway cannot stall a turn for 5 minutes.
HERMES_REQUEST_TIMEOUT_S = 180.0
_HERMES_REQUEST_LOCK = asyncio.Lock()
HERMES_CHAT_MODEL = "hermes-agent"
HERMES_SESSION_HEADER = "X-Hermes-Session-Id"
# Layered on Hermes's own system prompt so the agent skips essays and extra skills.
HERMES_VOICE_SYSTEM_PROMPT = (
    "You are answering a talking robot. Reply in 1-2 short spoken sentences. "
    "Answer the user's question. Use advanced reasoning and external tools only when the request "
    "cannot be handled by Reachy's local tools. "
    "Do not search the web, use the terminal, or narrate files, tools, agents, or your process."
)
HERMES_REEF_TREND_INSTRUCTION = (
    "Answer the reef-trend question with actual historical changes. "
    "Inspect historical reef data if available, analyse trends, and report meaningful changes. "
    "Do not narrate files, tools, agents, internal reasoning, or your process. "
    "If historical data is unavailable, say that clearly in one short sentence. "
    "Never invent historical values."
)
_TREND_MARKERS: tuple[str, ...] = (
    "trend",
    "trending",
    "treading",
    "threading",
    "history",
    "historical",
    "over time",
    "trajectory",
    "pattern",
)
_PROCESS_NARRATION_MARKERS: tuple[str, ...] = (
    "can't access the file",
    "cannot access the file",
    "couldn't access the file",
    "could not access the file",
    "unable to access",
    "don't have access to the file",
    "do not have access to the file",
    "issue accessing",
    "issue with accessing",
    "problem accessing",
    "need to inspect",
    "i need to inspect",
    "i will inspect",
    "i'll inspect",
    "i will now attempt",
    "i'll now attempt",
    "attempt to use the tool",
    "using the tool",
    "tried to access",
    "accessing the file",
    "file content",
    "there seems to be an issue",
)

_PROCESS_SESSION_ID: str | None = None


class HermesClientError(RuntimeError):
    """Base error for Hermes Gateway calls."""


class HermesNotConfiguredError(HermesClientError):
    """Raised when the gateway URL or API key is missing."""


class HermesTimeoutError(HermesClientError):
    """Raised when the gateway does not respond in time."""


class HermesRequestError(HermesClientError):
    """Raised when the gateway returns a failure or an unusable body."""


def get_hermes_session_id() -> str:
    """Return a process-lifetime session id for Hermes conversation context."""
    global _PROCESS_SESSION_ID
    if _PROCESS_SESSION_ID is None:
        _PROCESS_SESSION_ID = str(uuid.uuid4())
    return _PROCESS_SESSION_ID


def hermes_is_busy() -> bool:
    """Return whether a Hermes Gateway request is currently in flight."""
    return _HERMES_REQUEST_LOCK.locked()


def is_trend_query(text: str) -> bool:
    """Return whether the utterance asks for historical trends rather than a live snapshot."""
    lowered = text.lower()
    return any(marker in lowered for marker in _TREND_MARKERS)


def is_process_narration(text: str) -> bool:
    """Return whether Hermes narrated file/tool process instead of answering."""
    lowered = text.lower().strip()
    if not lowered:
        return True
    if any(marker in lowered for marker in _PROCESS_NARRATION_MARKERS):
        return True
    mentions_file = "file" in lowered
    mentions_access = "access" in lowered
    mentions_inspect = "inspect" in lowered
    mentions_tool = "tool" in lowered
    return (mentions_file and mentions_access) or (mentions_inspect and (mentions_file or mentions_tool))


def _require_hermes_config() -> tuple[str, str]:
    gateway_url = (config.HERMES_GATEWAY_URL or "").strip()
    api_key = (config.HERMES_API_KEY or "").strip()
    if not gateway_url or not api_key:
        raise HermesNotConfiguredError(
            "HERMES_GATEWAY_URL and HERMES_API_KEY must be set before calling the Hermes Gateway."
        )
    return gateway_url, api_key


def _reply_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        raise HermesRequestError("Hermes Gateway response must be a chat.completion JSON object.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HermesRequestError("Hermes Gateway response is missing choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise HermesRequestError("Hermes Gateway response is missing a non-empty reply string.")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise HermesRequestError("Hermes Gateway response is missing a non-empty reply string.")
    reply = message.get("content")
    if not isinstance(reply, str) or not reply.strip():
        raise HermesRequestError("Hermes Gateway response is missing a non-empty reply string.")
    return reply.strip()


async def send_to_hermes(text: str, session_id: str, request_id: str | None = None) -> str:
    """POST a user message to Hermes /v1/chat/completions and return the assistant text."""
    utterance = text.strip()
    if not utterance:
        raise HermesRequestError("Cannot send an empty utterance to the Hermes Gateway.")

    gateway_url, api_key = _require_hermes_config()
    hermes_request_id = request_id or str(uuid.uuid4())
    user_content = utterance
    if is_trend_query(utterance):
        user_content = f"{utterance}\n\n{HERMES_REEF_TREND_INSTRUCTION}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        HERMES_SESSION_HEADER: session_id,
    }
    body: dict[str, object] = {
        "model": HERMES_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": HERMES_VOICE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    started = time.monotonic()
    logger.info("Hermes request started request_id=%s session=%s", hermes_request_id, session_id)
    try:
        async with _HERMES_REQUEST_LOCK:
            async with httpx.AsyncClient(timeout=HERMES_REQUEST_TIMEOUT_S) as http_client:
                response = await http_client.post(gateway_url, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
    except httpx.TimeoutException as exc:
        logger.warning(
            "Hermes Gateway timed out after %.1fs request_id=%s: %s",
            HERMES_REQUEST_TIMEOUT_S,
            hermes_request_id,
            exc,
        )
        raise HermesTimeoutError("Hermes Gateway timed out.") from exc
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "Hermes Gateway HTTP %s request_id=%s: %s",
            exc.response.status_code,
            hermes_request_id,
            exc,
        )
        raise HermesRequestError(f"Hermes Gateway returned HTTP {exc.response.status_code}.") from exc
    except httpx.RequestError as exc:
        logger.warning("Hermes Gateway request failed request_id=%s: %s", hermes_request_id, exc)
        raise HermesRequestError("Hermes Gateway request failed.") from exc
    except ValueError as exc:
        logger.warning("Hermes Gateway returned malformed JSON request_id=%s: %s", hermes_request_id, exc)
        raise HermesRequestError("Hermes Gateway returned malformed JSON.") from exc

    reply = _reply_from_payload(payload)
    logger.info(
        "Hermes request completed request_id=%s chars=%s in %.1fs session=%s",
        hermes_request_id,
        len(reply),
        time.monotonic() - started,
        session_id,
    )
    return reply
