"""HTTP client for the Hermes Agent OpenAI-compatible API server."""

import os
import json
import time
import uuid
import asyncio
import logging
from typing import Any

import httpx

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)
REEF_THREAD_PATH = os.path.expanduser("~/reef-monitor/reef_thread.jsonl")
_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"
HISTORY_UNAVAILABLE = "Historical reef data is currently unavailable."

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
    "This is a reef trend/history request. Use the Reefy cache report included in this "
    "message as the only data source. Speak the actual numbers, 6-hour slopes, and ATO "
    "time-to-empty. Do not narrate files, tools, agents, or your process. Do not invent "
    "historical values. Do not use live Apex readings. If no cache report is included, "
    "say that historical reef data is currently unavailable."
)
HERMES_REEF_VOICE_SYSTEM_PROMPT = (
    "You are answering a talking robot. Reply in 1-2 short spoken sentences. "
    "A Reefy historical cache report is included in the user message. Use those numbers. "
    "Do not search the web. Do not invent values. Do not mention files, tools, or Apex."
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
    "6 hour",
    "6-hour",
    "six hour",
    "last 6",
    "changed over",
    "reef tank report",
    "reef report",
    "tank report",
    "full reef",
    "time to empty",
    "time-to-empty",
    "ato history",
    "been using",
    "how much ato",
    "ato usage",
)
_REEF_CONTEXT_MARKERS: tuple[str, ...] = ("reef", "tank", "apex", "ato")
_REEF_HISTORY_MARKERS: tuple[str, ...] = (
    "report",
    "analyse",
    "analyze",
    "analysis",
    "improving",
    "worsening",
    "getting worse",
    "getting better",
    "been doing",
    "been going",
    "changed",
    "changing",
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
    "trouble accessing",
    "having trouble accessing",
    "let me try again",
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
    if any(marker in lowered for marker in _TREND_MARKERS):
        return True
    reef_context = any(marker in lowered for marker in _REEF_CONTEXT_MARKERS)
    return reef_context and any(marker in lowered for marker in _REEF_HISTORY_MARKERS)


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


def chat_completions_url(gateway_url: str) -> str:
    """Return the OpenAI chat-completions path, even if only host:port was configured."""
    url = gateway_url.strip().rstrip("/")
    if not url:
        return url
    if url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}{_CHAT_COMPLETIONS_SUFFIX}"


def _handoff_from_thread_entry(entry: dict[str, Any]) -> dict[str, Any]:
    handoff = entry.get("handoff")
    return handoff if isinstance(handoff, dict) else {}


def _for_reachy_from_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    for_reachy = handoff.get("for_reachy")
    return for_reachy if isinstance(for_reachy, dict) else {}


def _summary_from_thread_entry(entry: dict[str, Any]) -> str | None:
    summary = entry.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    for_reachy = _for_reachy_from_handoff(_handoff_from_thread_entry(entry))
    nested = for_reachy.get("summary")
    if isinstance(nested, str) and nested.strip():
        return nested.strip()
    return None


def _source_from_thread_entry(entry: dict[str, Any]) -> str | None:
    for_reachy = _for_reachy_from_handoff(_handoff_from_thread_entry(entry))
    source = for_reachy.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    source = entry.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return None


def load_latest_reef_thread(path: str | None = None) -> dict[str, Any] | None:
    """Return the latest Reefy thread run, or None if the cache is missing/empty."""
    thread_path = path or REEF_THREAD_PATH
    try:
        with open(thread_path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        logger.warning("[HERMES] reef_thread unreadable path=%s error=%s", thread_path, exc)
        return None
    latest: dict[str, Any] | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("[HERMES] reef_thread skipped malformed line")
            continue
        if not isinstance(entry, dict) or entry.get("type") != "run":
            continue
        summary = _summary_from_thread_entry(entry)
        if summary is None:
            continue
        trends = entry.get("trends")
        ato = entry.get("ato")
        handoff = _handoff_from_thread_entry(entry)
        latest = {
            "generated_at": entry.get("ts"),
            "data_timestamp": entry.get("cache_ts") or entry.get("ts"),
            "report": summary,
            "trends": trends if isinstance(trends, dict) else {},
            "ato": ato if isinstance(ato, dict) else {},
            "ato_hours_until_low": entry.get("ato_hours_until_low"),
            "handoff": handoff,
            "source": _source_from_thread_entry(entry),
        }
    if latest is None:
        logger.info("[HERMES] reef_thread empty path=%s", thread_path)
        return None
    logger.info(
        "[HERMES] reef_thread loaded path=%s generated_at=%s data_timestamp=%s ato_hours=%s trend_keys=%s",
        thread_path,
        latest.get("generated_at"),
        latest.get("data_timestamp"),
        latest.get("ato_hours_until_low"),
        sorted(str(key) for key in latest["trends"]),
    )
    return latest


def reef_history_query(user_query: str, cache: dict[str, Any] | None) -> str:
    """Build an unambiguous Hermes trend request, attaching the Reefy cache when present."""
    parts = [f"Reef trend/history request: {user_query.strip()}", HERMES_REEF_TREND_INSTRUCTION]
    if cache is None:
        parts.append("Reefy cache is unavailable. Do not invent historical values.")
        return "\n".join(parts)
    parts.append("Use this Reefy cache report as the only data source. Do not invent values.")
    parts.append(f"generated_at: {cache.get('generated_at')}")
    parts.append(f"data_timestamp: {cache.get('data_timestamp')}")
    parts.append(f"report: {cache.get('report')}")
    if cache.get("ato_hours_until_low") is not None:
        parts.append(f"ato_hours_until_low: {cache['ato_hours_until_low']}")
    trends = cache.get("trends")
    if isinstance(trends, dict) and trends:
        parts.append(f"trends: {json.dumps(trends)}")
    return "\n".join(parts)


def _require_hermes_config() -> tuple[str, str]:
    gateway_url = (config.HERMES_GATEWAY_URL or "").strip()
    api_key = (config.HERMES_API_KEY or "").strip()
    if not gateway_url or not api_key:
        raise HermesNotConfiguredError(
            "HERMES_GATEWAY_URL and HERMES_API_KEY must be set before calling the Hermes Gateway."
        )
    return chat_completions_url(gateway_url), api_key


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
    history_request = is_trend_query(utterance)
    user_content = utterance
    if history_request and HERMES_REEF_TREND_INSTRUCTION not in utterance:
        user_content = f"{utterance}\n\n{HERMES_REEF_TREND_INSTRUCTION}"
    system_prompt = HERMES_REEF_VOICE_SYSTEM_PROMPT if history_request else HERMES_VOICE_SYSTEM_PROMPT
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        HERMES_SESSION_HEADER: session_id,
    }
    body: dict[str, object] = {
        "model": HERMES_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    started = time.monotonic()
    logger.info(
        "[HERMES] request started request_id=%s url=%s history=%s query_chars=%s session=%s",
        hermes_request_id,
        gateway_url,
        history_request,
        len(user_content),
        session_id,
    )
    http_status: int | None = None
    payload: object | None = None
    try:
        async with _HERMES_REQUEST_LOCK:
            async with httpx.AsyncClient(timeout=HERMES_REQUEST_TIMEOUT_S) as http_client:
                response = await http_client.post(gateway_url, headers=headers, json=body)
                http_status = response.status_code
                response.raise_for_status()
                payload = response.json()
    except httpx.TimeoutException as exc:
        logger.warning(
            "[HERMES] gateway timed out after %.1fs request_id=%s url=%s: %s",
            HERMES_REQUEST_TIMEOUT_S,
            hermes_request_id,
            gateway_url,
            exc,
        )
        raise HermesTimeoutError("Hermes Gateway timed out.") from exc
    except httpx.HTTPStatusError as exc:
        http_status = exc.response.status_code
        logger.warning(
            "[HERMES] gateway HTTP %s request_id=%s url=%s",
            http_status,
            hermes_request_id,
            gateway_url,
        )
        raise HermesRequestError(f"Hermes Gateway returned HTTP {http_status}.") from exc
    except httpx.RequestError as exc:
        logger.warning("[HERMES] gateway request failed request_id=%s url=%s: %s", hermes_request_id, gateway_url, exc)
        raise HermesRequestError("Hermes Gateway request failed.") from exc
    except ValueError as exc:
        logger.warning(
            "[HERMES] gateway malformed JSON request_id=%s http_status=%s: %s",
            hermes_request_id,
            http_status,
            exc,
        )
        raise HermesRequestError("Hermes Gateway returned malformed JSON.") from exc

    if payload is None:
        raise HermesRequestError("Hermes Gateway returned an empty body.")
    reply = _reply_from_payload(payload)
    logger.info(
        "[HERMES] gateway reply request_id=%s http_status=%s chars=%s process_narration=%s elapsed=%.1fs prefix=%s",
        hermes_request_id,
        http_status,
        len(reply),
        is_process_narration(reply),
        time.monotonic() - started,
        reply[:240],
    )
    return reply
