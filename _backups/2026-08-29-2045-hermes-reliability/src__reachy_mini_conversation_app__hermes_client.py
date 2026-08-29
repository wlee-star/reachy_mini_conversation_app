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
    "Use advanced reasoning and external tools only when the request cannot be handled by Reachy's local tools. "
    "Do not search the web, use the terminal, or explain your process."
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


async def send_to_hermes(text: str, session_id: str) -> str:
    """POST a user message to Hermes /v1/chat/completions and return the assistant text."""
    utterance = text.strip()
    if not utterance:
        raise HermesRequestError("Cannot send an empty utterance to the Hermes Gateway.")

    gateway_url, api_key = _require_hermes_config()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        HERMES_SESSION_HEADER: session_id,
    }
    body: dict[str, object] = {
        "model": HERMES_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": HERMES_VOICE_SYSTEM_PROMPT},
            {"role": "user", "content": utterance},
        ],
    }

    started = time.monotonic()
    try:
        async with _HERMES_REQUEST_LOCK:
            async with httpx.AsyncClient(timeout=HERMES_REQUEST_TIMEOUT_S) as http_client:
                response = await http_client.post(gateway_url, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
    except httpx.TimeoutException as exc:
        logger.warning("Hermes Gateway timed out after %.1fs: %s", HERMES_REQUEST_TIMEOUT_S, exc)
        raise HermesTimeoutError("Hermes Gateway timed out.") from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("Hermes Gateway HTTP %s: %s", exc.response.status_code, exc)
        raise HermesRequestError(f"Hermes Gateway returned HTTP {exc.response.status_code}.") from exc
    except httpx.RequestError as exc:
        logger.warning("Hermes Gateway request failed: %s", exc)
        raise HermesRequestError("Hermes Gateway request failed.") from exc
    except ValueError as exc:
        logger.warning("Hermes Gateway returned malformed JSON: %s", exc)
        raise HermesRequestError("Hermes Gateway returned malformed JSON.") from exc

    reply = _reply_from_payload(payload)
    logger.info(
        "Hermes Gateway replied (%s chars) in %.1fs for session %s",
        len(reply),
        time.monotonic() - started,
        session_id,
    )
    return reply
