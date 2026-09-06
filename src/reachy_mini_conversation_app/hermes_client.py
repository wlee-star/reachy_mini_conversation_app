"""HTTP client for the Hermes Agent OpenAI-compatible API server."""

import os
import json
import time
import uuid
import asyncio
import logging
from typing import Any
from datetime import datetime, timezone

import httpx

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)
REEF_MONITOR_DIR = os.path.expanduser("~/reef-monitor")
REEF_THREAD_PATH = os.path.join(REEF_MONITOR_DIR, "reef_thread.jsonl")
REEF_CACHE_PATH = os.path.join(REEF_MONITOR_DIR, "reef_cache.json")
# reef-monitor 1-minute cache cron; allow one missed tick of jitter.
REEF_LIVE_CACHE_MAX_AGE_SECONDS = 120
# reef-monitor history logger + thread updater cron (*/30).
REEF_TREND_INTERVAL_SECONDS = 1800
_CHAT_COMPLETIONS_SUFFIX = "/v1/chat/completions"
HISTORY_UNAVAILABLE = "Historical reef data is currently unavailable."
REQUEST_KIND_CURRENT = "reef_current"
REQUEST_KIND_HISTORY = "reef_history"
REQUEST_KIND_DELEGATED = "delegated"

# Non-Reef delegated queries can take 45-90s; config.HERMES_REQUEST_TIMEOUT_SECONDS
# caps them at 180s. Interactive Reef waits use config.HERMES_REEF_REQUEST_TIMEOUT_SECONDS (15s).
_HERMES_REQUEST_LOCK = asyncio.Lock()
_HERMES_IN_FLIGHT_REQUEST_ID: str | None = None
_CIRCUIT_CLOSED = "closed"
_CIRCUIT_OPEN = "open"
_CIRCUIT_HALF_OPEN = "half_open"
HERMES_CHAT_MODEL = "hermes-agent"
HERMES_SESSION_HEADER = "X-Hermes-Session-Id"
# Layered on Hermes's own system prompt so the agent skips essays and extra skills.
HERMES_VOICE_SYSTEM_PROMPT = (
    "You are answering Reachy Mini, a talking robot assistant. "
    "Reply in 1-2 short spoken sentences. "
    "Answer the user's question. Use advanced reasoning and external tools only when the request "
    "cannot be handled by Reachy Mini's local tools. "
    "Do not search the web, use the terminal, or narrate files, tools, agents, or your process."
)
HERMES_REEF_TREND_INSTRUCTION = (
    "This is a historical reef request. Use the Reefy thread report included in this "
    "message as the historical data source. Speak the actual numbers, 6-hour slopes, and ATO "
    "time-to-empty. Do not narrate files, tools, agents, or your process. Do not invent "
    "historical values. If no historical report is included, "
    "say that historical reef data is currently unavailable."
)
HERMES_REEF_CURRENT_INSTRUCTION = (
    "This is a current/latest reef request, not a historical one. "
    "Use the current reef-monitor live cache in this message as the live probe source. "
    "Use the latest 30-minute slope report only for 6-hour trends. "
    "Do not use prior conversation context. Do not use an old reef_thread.jsonl report "
    "as the only source. Do not invent values. Speak the current numbers and 6-hour slopes."
)
HERMES_REEF_VOICE_SYSTEM_PROMPT = (
    "You are answering Reachy Mini, a talking robot assistant. "
    "Reply in 1-2 short spoken sentences. "
    "A Reefy historical cache report is included in the user message. Use those numbers. "
    "Do not search the web. Do not invent values. Do not mention files, tools, or Apex."
)
HERMES_REEF_CURRENT_VOICE_SYSTEM_PROMPT = (
    "You are answering Reachy Mini, a talking robot assistant. "
    "Reply in 1-2 short spoken sentences. "
    "A current reef-monitor snapshot is included in the user message. Use those numbers. "
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
    "hermes report",
    "latest hermes",
    "full reef",
    "time to empty",
    "time-to-empty",
    "ato history",
    "been using",
    "how much ato",
    "ato usage",
)
_REEF_CONTEXT_MARKERS: tuple[str, ...] = ("reef", "tank", "apex", "ato")
_REEF_REPORT_MARKERS: tuple[str, ...] = (
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
    "happening",
)
_HISTORICAL_REEF_MARKERS: tuple[str, ...] = (
    "history",
    "historical",
    "over time",
    "last 6",
    "last six",
    "last few hours",
    "changed over",
    "been using",
    "ato history",
    "parameter history",
    "been doing",
    "been going",
    "how has",
    "how have",
    "what has changed",
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

    category = "HERMES_ERROR"

    def __init__(self, message: str, *, category: str | None = None) -> None:
        """Attach an optional failure category to a Hermes error."""
        super().__init__(message)
        if category is not None:
            self.category = category


class HermesNotConfiguredError(HermesClientError):
    """Raised when the gateway URL or API key is missing."""

    category = "HERMES_NOT_CONFIGURED"


class HermesTimeoutError(HermesClientError):
    """Raised when the gateway does not respond in time."""

    category = "HERMES_TIMEOUT"


class HermesCircuitOpenError(HermesClientError):
    """Raised when Hermes is temporarily skipped after repeated failures."""

    category = "HERMES_CIRCUIT_OPEN"


class HermesRequestError(HermesClientError):
    """Raised when the gateway returns a failure or an unusable body."""

    category = "HERMES_HTTP_ERROR"


class _HermesCircuit:
    def __init__(self) -> None:
        self.state = _CIRCUIT_CLOSED
        self.failures = 0
        self.opened_at: float | None = None
        self.half_open_probe = False

    def reset(self) -> None:
        self.state = _CIRCUIT_CLOSED
        self.failures = 0
        self.opened_at = None
        self.half_open_probe = False

    def allow_request(self) -> bool:
        if self.state == _CIRCUIT_CLOSED:
            return True
        cooldown = float(config.HERMES_CIRCUIT_COOLDOWN_SECONDS)
        if self.state == _CIRCUIT_OPEN:
            if self.opened_at is not None and time.monotonic() - self.opened_at >= cooldown:
                self.state = _CIRCUIT_HALF_OPEN
                self.half_open_probe = False
            else:
                return False
        if self.state == _CIRCUIT_HALF_OPEN:
            if self.half_open_probe:
                return False
            self.half_open_probe = True
            return True
        return True

    def record_success(self) -> None:
        if self.state != _CIRCUIT_CLOSED or self.failures:
            logger.info("[HERMES] circuit closed after recovery")
        self.reset()

    def record_failure(self) -> None:
        self.failures += 1
        self.half_open_probe = False
        threshold = int(config.HERMES_CIRCUIT_FAILURE_THRESHOLD)
        if self.state == _CIRCUIT_HALF_OPEN or self.failures >= threshold:
            self.state = _CIRCUIT_OPEN
            self.opened_at = time.monotonic()
            logger.warning(
                "[HERMES] circuit opened failures=%s cooldown=%.1fs",
                self.failures,
                float(config.HERMES_CIRCUIT_COOLDOWN_SECONDS),
            )

    def abandon_probe(self) -> None:
        self.half_open_probe = False


_HERMES_CIRCUIT = _HermesCircuit()


def get_hermes_session_id() -> str:
    """Return a process-lifetime session id for Hermes conversation context."""
    global _PROCESS_SESSION_ID
    if _PROCESS_SESSION_ID is None:
        _PROCESS_SESSION_ID = str(uuid.uuid4())
    return _PROCESS_SESSION_ID


def new_hermes_session_id() -> str:
    """Return a one-off session id so a current reef request cannot reuse old context."""
    return str(uuid.uuid4())


def hermes_is_busy() -> bool:
    """Return whether a Hermes Gateway request is currently in flight."""
    return _HERMES_REQUEST_LOCK.locked()


def hermes_in_flight_request_id() -> str | None:
    """Return the request id currently holding the Hermes lock, if any."""
    return _HERMES_IN_FLIGHT_REQUEST_ID


def hermes_circuit_state() -> str:
    """Return the Hermes circuit state: closed, open, or half_open."""
    return _HERMES_CIRCUIT.state


def reset_hermes_circuit() -> None:
    """Reset the Hermes circuit breaker. Used by tests."""
    _HERMES_CIRCUIT.reset()


def hermes_request_timeout_s(*, history_request: bool = False, reef_request: bool | None = None) -> float:
    """Return the live-wait timeout for a Hermes call."""
    if reef_request is None:
        reef_request = history_request
    if reef_request:
        return float(config.HERMES_REEF_REQUEST_TIMEOUT_SECONDS)
    return float(config.HERMES_REQUEST_TIMEOUT_SECONDS)


def is_trend_query(text: str) -> bool:
    """Return whether the utterance is a reef trend/report request for Hermes."""
    lowered = text.lower()
    if any(marker in lowered for marker in _TREND_MARKERS):
        return True
    reef_context = any(marker in lowered for marker in _REEF_CONTEXT_MARKERS)
    if "latest report" in lowered and "weather" not in lowered:
        return True
    return reef_context and any(marker in lowered for marker in _REEF_REPORT_MARKERS)


def is_historical_reef_query(text: str) -> bool:
    """Return whether the utterance asks for historical reef data rather than current trends."""
    if not is_trend_query(text):
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _HISTORICAL_REEF_MARKERS)


def is_current_reef_query(text: str) -> bool:
    """Return whether the utterance asks for the current/latest reef trend or report."""
    return is_trend_query(text) and not is_historical_reef_query(text)


def reef_request_kind(text: str) -> str | None:
    """Return reef_current, reef_history, or None for a non-reef utterance."""
    if is_historical_reef_query(text):
        return REQUEST_KIND_HISTORY
    if is_trend_query(text):
        return REQUEST_KIND_CURRENT
    return None


def current_reef_is_fresh(age_seconds: float | None) -> bool:
    """Return whether a current-reef data timestamp is within the live-cache freshness window."""
    return age_seconds is not None and age_seconds <= REEF_LIVE_CACHE_MAX_AGE_SECONDS


def reef_slopes_are_fresh(age_seconds: float | None) -> bool:
    """Return whether a reef_thread slope report is within the 30-minute trend cadence."""
    return age_seconds is not None and age_seconds <= REEF_TREND_INTERVAL_SECONDS


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
    latest["cache_age_seconds"] = reef_cache_age_seconds(
        generated_at=latest.get("generated_at"),
        data_timestamp=latest.get("data_timestamp"),
        path=thread_path,
    )
    logger.info(
        "[HERMES] reef_thread loaded path=%s generated_at=%s data_timestamp=%s age_seconds=%s ato_hours=%s trend_keys=%s",
        thread_path,
        latest.get("generated_at"),
        latest.get("data_timestamp"),
        latest.get("cache_age_seconds"),
        latest.get("ato_hours_until_low"),
        sorted(str(key) for key in latest["trends"]),
    )
    return latest


def parse_reef_timestamp(raw: object) -> datetime | None:
    """Parse an ISO-8601 report timestamp, including a trailing Z."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def reef_cache_age_seconds(
    *,
    generated_at: object = None,
    data_timestamp: object = None,
    path: str | None = None,
) -> float | None:
    """Return cache age in seconds from a report timestamp, else the file mtime."""
    now = datetime.now(timezone.utc)
    for raw in (generated_at, data_timestamp):
        parsed = parse_reef_timestamp(raw)
        if parsed is not None:
            return max(0.0, (now - parsed).total_seconds())
    if path:
        try:
            return max(0.0, time.time() - os.path.getmtime(path))
        except OSError:
            return None
    return None


def reef_cache_max_age_seconds() -> float:
    """Return the configured Reef cache freshness threshold in seconds."""
    return float(config.REEF_CACHE_MAX_AGE_SECONDS)


def reef_cache_status_for_age(age_seconds: float | None) -> str:
    """Classify a cache fallback as degraded or stale. Never success."""
    max_age = reef_cache_max_age_seconds()
    if age_seconds is not None and age_seconds <= max_age:
        return "degraded"
    return "stale"


def load_reef_live_cache(path: str | None = None) -> dict[str, Any] | None:
    """Return the current reef-monitor live cache, or None if it is missing."""
    cache_path = path or REEF_CACHE_PATH
    try:
        with open(cache_path, encoding="utf-8") as handle:
            raw: object = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[HERMES] reef_cache unreadable path=%s error=%s", cache_path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("[HERMES] reef_cache invalid path=%s", cache_path)
        return None
    cached_at = raw.get("cached_at")
    data_timestamp = cached_at if isinstance(cached_at, str) and cached_at.strip() else None
    probes_raw = raw.get("probes")
    ato_raw = raw.get("ato")
    probes: dict[str, Any] = probes_raw if isinstance(probes_raw, dict) else {}
    ato: dict[str, Any] = ato_raw if isinstance(ato_raw, dict) else {}
    age_seconds = reef_cache_age_seconds(
        generated_at=data_timestamp,
        data_timestamp=data_timestamp,
        path=cache_path,
    )
    snapshot: dict[str, Any] = {
        "cached_at": data_timestamp,
        "data_timestamp": data_timestamp,
        "probes": {str(key): value for key, value in probes.items()},
        "ato": {str(key): value for key, value in ato.items()},
        "age_seconds": age_seconds,
        "source": "reef_monitor",
    }
    logger.info(
        "[HERMES] reef_cache loaded path=%s cached_at=%s age_seconds=%s probe_keys=%s",
        cache_path,
        data_timestamp,
        age_seconds,
        sorted(str(key) for key in probes),
    )
    return snapshot


def reef_history_query(user_query: str, cache: dict[str, Any] | None) -> str:
    """Build a historical Hermes request, attaching the Reefy thread when present."""
    parts = [f"Reef historical request: {user_query.strip()}", HERMES_REEF_TREND_INSTRUCTION]
    if cache is None:
        parts.append("Reefy historical report is unavailable. Do not invent historical values.")
        return "\n".join(parts)
    parts.append("Use this Reefy thread report as the historical data source. Do not invent values.")
    parts.append(f"generated_at: {cache.get('generated_at')}")
    parts.append(f"data_timestamp: {cache.get('data_timestamp')}")
    parts.append(f"report: {cache.get('report')}")
    if cache.get("ato_hours_until_low") is not None:
        parts.append(f"ato_hours_until_low: {cache['ato_hours_until_low']}")
    trends = cache.get("trends")
    if isinstance(trends, dict) and trends:
        parts.append(f"trends: {json.dumps(trends)}")
    return "\n".join(parts)


def reef_current_query(
    user_query: str,
    live_cache: dict[str, Any] | None,
    thread: dict[str, Any] | None,
) -> str:
    """Build a current Hermes request from the live cache plus latest 30-minute slopes."""
    parts = [f"Reef current request: {user_query.strip()}", HERMES_REEF_CURRENT_INSTRUCTION]
    if live_cache is None:
        parts.append("Live reef-monitor cache is unavailable. Do not invent values.")
    else:
        parts.append("Current reef-monitor live cache:")
        parts.append(f"cached_at: {live_cache.get('cached_at')}")
        parts.append(f"data_timestamp: {live_cache.get('data_timestamp')}")
        parts.append(f"age_seconds: {live_cache.get('age_seconds')}")
        probes = live_cache.get("probes")
        if isinstance(probes, dict) and probes:
            parts.append(f"probes: {json.dumps(probes)}")
        ato = live_cache.get("ato")
        if isinstance(ato, dict) and ato:
            parts.append(f"ato: {json.dumps(ato)}")
    if thread is not None:
        parts.append("Latest 30-minute slope report for 6-hour trends only. Not a substitute for live values.")
        parts.append(f"slope_generated_at: {thread.get('generated_at')}")
        parts.append(f"slope_data_timestamp: {thread.get('data_timestamp')}")
        trends = thread.get("trends")
        if isinstance(trends, dict) and trends:
            parts.append(f"trends: {json.dumps(trends)}")
        if thread.get("ato_hours_until_low") is not None:
            parts.append(f"ato_hours_until_low: {thread['ato_hours_until_low']}")
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
        raise HermesRequestError(
            "Hermes Gateway response must be a chat.completion JSON object.",
            category="HERMES_INVALID_RESPONSE",
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HermesRequestError("Hermes Gateway response is missing choices.", category="HERMES_INVALID_RESPONSE")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise HermesRequestError(
            "Hermes Gateway response is missing a non-empty reply string.",
            category="HERMES_INVALID_RESPONSE",
        )
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise HermesRequestError(
            "Hermes Gateway response is missing a non-empty reply string.",
            category="HERMES_INVALID_RESPONSE",
        )
    reply = message.get("content")
    if not isinstance(reply, str) or not reply.strip():
        raise HermesRequestError(
            "Hermes Gateway response is missing a non-empty reply string.",
            category="HERMES_INVALID_RESPONSE",
        )
    return reply.strip()


def _log_hermes_timeout(request_id: str, *, elapsed: float, reef_request: bool) -> None:
    logger.warning(
        "[HERMES] request timed out/failed request_id=%s elapsed=%.1fs reef=%s failure_category=%s",
        request_id,
        elapsed,
        reef_request,
        HermesTimeoutError.category,
    )
    if reef_request:
        logger.warning(
            "[HERMES] Reef request timed out request_id=%s elapsed=%.1fs failure_category=%s",
            request_id,
            elapsed,
            HermesTimeoutError.category,
        )
        logger.warning("[HERMES] timeout after %.1f seconds", elapsed)
    logger.info("[HERMES] cancelling timed-out request request_id=%s", request_id)


def _should_retry_hermes(exc: BaseException, attempt: int) -> bool:
    if attempt > 1:
        return False
    if isinstance(exc, httpx.TimeoutException):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code <= 599
    return isinstance(exc, httpx.RequestError)


def _resolve_request_kind(utterance: str, request_kind: str | None) -> str:
    if request_kind in {REQUEST_KIND_CURRENT, REQUEST_KIND_HISTORY, REQUEST_KIND_DELEGATED}:
        return request_kind
    if utterance.startswith("Reef current request:"):
        return REQUEST_KIND_CURRENT
    if utterance.startswith("Reef historical request:") or utterance.startswith("Reef trend/history request:"):
        return REQUEST_KIND_HISTORY
    return reef_request_kind(utterance) or REQUEST_KIND_DELEGATED


async def send_to_hermes(
    text: str,
    session_id: str,
    request_id: str | None = None,
    request_kind: str | None = None,
) -> str:
    """POST a user message to Hermes /v1/chat/completions and return the assistant text."""
    global _HERMES_IN_FLIGHT_REQUEST_ID
    utterance = text.strip()
    if not utterance:
        raise HermesRequestError(
            "Cannot send an empty utterance to the Hermes Gateway.",
            category="HERMES_INVALID_RESPONSE",
        )

    gateway_url, api_key = _require_hermes_config()
    hermes_request_id = request_id or str(uuid.uuid4())
    resolved_kind = _resolve_request_kind(utterance, request_kind)
    reef_request = resolved_kind in {REQUEST_KIND_CURRENT, REQUEST_KIND_HISTORY}
    history_request = resolved_kind == REQUEST_KIND_HISTORY
    request_type = resolved_kind
    if not _HERMES_CIRCUIT.allow_request():
        logger.warning(
            "[HERMES] circuit open request_id=%s request_type=%s failure_category=%s",
            hermes_request_id,
            request_type,
            HermesCircuitOpenError.category,
        )
        raise HermesCircuitOpenError("Hermes Gateway circuit is open.")
    user_content = utterance
    if resolved_kind == REQUEST_KIND_CURRENT and HERMES_REEF_CURRENT_INSTRUCTION not in utterance:
        user_content = f"{utterance}\n\n{HERMES_REEF_CURRENT_INSTRUCTION}"
    elif history_request and HERMES_REEF_TREND_INSTRUCTION not in utterance:
        user_content = f"{utterance}\n\n{HERMES_REEF_TREND_INSTRUCTION}"
    if resolved_kind == REQUEST_KIND_CURRENT:
        system_prompt = HERMES_REEF_CURRENT_VOICE_SYSTEM_PROMPT
    elif history_request:
        system_prompt = HERMES_REEF_VOICE_SYSTEM_PROMPT
    else:
        system_prompt = HERMES_VOICE_SYSTEM_PROMPT
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

    timeout_s = hermes_request_timeout_s(history_request=history_request, reef_request=reef_request)
    started = time.monotonic()
    requested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    logger.info("[HERMES] request started request_id=%s", hermes_request_id)
    logger.info("[HERMES] request_type=%s request_id=%s", request_type, hermes_request_id)
    logger.info("[HERMES] history=%s request_id=%s", str(history_request).lower(), hermes_request_id)
    logger.info("[HERMES] gateway=%s request_id=%s", gateway_url, hermes_request_id)
    logger.info("[HERMES] requested_at=%s request_id=%s", requested_at, hermes_request_id)
    logger.info(
        "[HERMES] request started request_id=%s request_type=%s url=%s history=%s query_chars=%s session=%s timeout=%.1f",
        hermes_request_id,
        request_type,
        gateway_url,
        history_request,
        len(user_content),
        session_id,
        timeout_s,
    )
    if reef_request:
        logger.info("[HERMES] Reef request started request_id=%s timeout=%.1f", hermes_request_id, timeout_s)
    http_status: int | None = None
    payload: object | None = None
    try:
        async with _HERMES_REQUEST_LOCK:
            _HERMES_IN_FLIGHT_REQUEST_ID = hermes_request_id
            try:
                async with httpx.AsyncClient(timeout=timeout_s) as http_client:
                    for attempt in (1, 2):
                        try:
                            response = await asyncio.wait_for(
                                http_client.post(gateway_url, headers=headers, json=body),
                                timeout=timeout_s,
                            )
                        except TimeoutError:
                            _log_hermes_timeout(
                                hermes_request_id,
                                elapsed=time.monotonic() - started,
                                reef_request=reef_request,
                            )
                            raise
                        except httpx.TimeoutException:
                            _log_hermes_timeout(
                                hermes_request_id,
                                elapsed=time.monotonic() - started,
                                reef_request=reef_request,
                            )
                            raise
                        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                            if _should_retry_hermes(exc, attempt):
                                logger.info(
                                    "[HERMES] retry request_id=%s attempt=2 failure_category=HERMES_RETRY",
                                    hermes_request_id,
                                )
                                continue
                            raise
                        http_status = response.status_code
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            if _should_retry_hermes(exc, attempt):
                                logger.info(
                                    "[HERMES] retry request_id=%s attempt=2 failure_category=HERMES_RETRY",
                                    hermes_request_id,
                                )
                                continue
                            raise
                        payload = response.json()
                        break
            finally:
                _HERMES_IN_FLIGHT_REQUEST_ID = None
                logger.info("[HERMES] request cleanup complete request_id=%s", hermes_request_id)
    except asyncio.CancelledError:
        _HERMES_CIRCUIT.abandon_probe()
        raise
    except TimeoutError as exc:
        _HERMES_CIRCUIT.record_failure()
        raise HermesTimeoutError("Hermes Gateway timed out.") from exc
    except httpx.TimeoutException as exc:
        _HERMES_CIRCUIT.record_failure()
        raise HermesTimeoutError("Hermes Gateway timed out.") from exc
    except httpx.HTTPStatusError as exc:
        _HERMES_CIRCUIT.record_failure()
        http_status = exc.response.status_code
        logger.warning(
            "[HERMES] gateway HTTP %s request_id=%s url=%s failure_category=HERMES_HTTP_ERROR",
            http_status,
            hermes_request_id,
            gateway_url,
        )
        raise HermesRequestError(f"Hermes Gateway returned HTTP {http_status}.") from exc
    except httpx.RequestError as exc:
        _HERMES_CIRCUIT.record_failure()
        logger.warning(
            "[HERMES] gateway request failed request_id=%s url=%s failure_category=HERMES_CONNECTION_ERROR: %s",
            hermes_request_id,
            gateway_url,
            exc,
        )
        raise HermesRequestError("Hermes Gateway request failed.", category="HERMES_CONNECTION_ERROR") from exc
    except ValueError as exc:
        _HERMES_CIRCUIT.record_failure()
        logger.warning(
            "[HERMES] gateway malformed JSON request_id=%s http_status=%s failure_category=HERMES_INVALID_RESPONSE: %s",
            hermes_request_id,
            http_status,
            exc,
        )
        logger.warning("[HERMES] parse_error=%s", exc)
        raise HermesRequestError(
            "Hermes Gateway returned malformed JSON.",
            category="HERMES_INVALID_RESPONSE",
        ) from exc

    if payload is None:
        _HERMES_CIRCUIT.record_failure()
        raise HermesRequestError("Hermes Gateway returned an empty body.", category="HERMES_INVALID_RESPONSE")
    try:
        reply = _reply_from_payload(payload)
    except HermesRequestError:
        _HERMES_CIRCUIT.record_failure()
        raise
    _HERMES_CIRCUIT.record_success()
    logger.info("[HERMES] response received request_id=%s", hermes_request_id)
    logger.info("[HERMES] http_status=%s request_id=%s", http_status, hermes_request_id)
    logger.info(
        "[HERMES] gateway reply request_id=%s request_type=%s http_status=%s chars=%s "
        "process_narration=%s elapsed=%.1fs prefix=%s",
        hermes_request_id,
        request_type,
        http_status,
        len(reply),
        is_process_narration(reply),
        time.monotonic() - started,
        reply[:240],
    )
    return reply
