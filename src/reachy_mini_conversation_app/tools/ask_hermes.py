import time
import uuid
import logging
from typing import Any
from datetime import datetime, timezone

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.local_time import format_time_12h, read_local_moment
from reachy_mini_conversation_app.hermes_client import (
    HISTORY_UNAVAILABLE,
    REQUEST_KIND_CURRENT,
    REQUEST_KIND_HISTORY,
    REQUEST_KIND_DELEGATED,
    HermesClientError,
    HermesTimeoutError,
    HermesCircuitOpenError,
    HermesNotConfiguredError,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


def _reason_from_hermes_error(exc: HermesClientError) -> str:
    message = str(exc)
    if isinstance(exc, HermesNotConfiguredError):
        return "not_configured"
    if isinstance(exc, HermesTimeoutError):
        return "timeout"
    if isinstance(exc, HermesCircuitOpenError):
        return "circuit_open"
    if "HTTP " in message:
        return f"gateway_http_{message.rsplit('HTTP ', 1)[-1].rstrip('.')}"
    if "malformed JSON" in message:
        return "gateway_malformed_json"
    return "gateway_error"


def _failure_category(reason: str) -> str:
    if reason.startswith("gateway_http_"):
        return "HERMES_HTTP_ERROR"
    mapping = {
        "timeout": "HERMES_TIMEOUT",
        "not_configured": "HERMES_NOT_CONFIGURED",
        "circuit_open": "HERMES_CIRCUIT_OPEN",
        "empty_or_narration": "HERMES_INVALID_RESPONSE",
        "gateway_malformed_json": "HERMES_INVALID_RESPONSE",
        "gateway_error": "HERMES_CONNECTION_ERROR",
    }
    return mapping.get(reason, "HERMES_ERROR")


def _cached_report_text(cache: dict[str, Any] | None) -> str | None:
    if cache is None:
        return None
    report = cache.get("report")
    if isinstance(report, str) and report.strip():
        return report.strip()
    return None


def _cache_age_seconds(cache: dict[str, Any] | None) -> float | None:
    if cache is None:
        return None
    existing = cache.get("cache_age_seconds")
    if isinstance(existing, (int, float)):
        return float(existing)
    return hermes_client.reef_cache_age_seconds(
        generated_at=cache.get("generated_at"),
        data_timestamp=cache.get("data_timestamp"),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _optional_timestamp(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _report_timestamp(*sources: dict[str, Any] | None) -> str | None:
    for cache in sources:
        if cache is None:
            continue
        for key in ("generated_at", "cached_at", "data_timestamp", "report_timestamp"):
            timestamp = _optional_timestamp(cache.get(key))
            if timestamp is not None:
                return timestamp
    return None


def _data_timestamp(*sources: dict[str, Any] | None) -> str | None:
    for cache in sources:
        if cache is None:
            continue
        for key in ("data_timestamp", "cached_at", "generated_at"):
            timestamp = _optional_timestamp(cache.get(key))
            if timestamp is not None:
                return timestamp
    return None


def _clock_label(timestamp: str | None) -> str | None:
    parsed = hermes_client.parse_reef_timestamp(timestamp)
    if parsed is None:
        return None
    return format_time_12h(read_local_moment(parsed))


def _log_hermes_result(
    *,
    request_id: str,
    status: str,
    source: str,
    fresh: bool,
    data_timestamp: str | None,
    report_timestamp: str | None,
    requested_at: str,
    retrieved_at: str,
    age_seconds: float | None,
    cache_used: bool,
    data_source: str,
) -> None:
    logger.info("[HERMES] data_timestamp=%s request_id=%s", data_timestamp, request_id)
    logger.info("[HERMES] report_timestamp=%s request_id=%s", report_timestamp, request_id)
    logger.info("[HERMES] requested_at=%s request_id=%s", requested_at, request_id)
    logger.info("[HERMES] retrieved_at=%s request_id=%s", retrieved_at, request_id)
    logger.info("[HERMES] age_seconds=%s request_id=%s", age_seconds, request_id)
    logger.info("[HERMES] cache_used=%s request_id=%s", str(cache_used).lower(), request_id)
    logger.info("[HERMES] fresh=%s request_id=%s", str(fresh).lower(), request_id)
    logger.info("[HERMES] source=%s request_id=%s", source, request_id)
    logger.info("[HERMES] data_source=%s request_id=%s", data_source, request_id)
    logger.info("[HERMES] status=%s request_id=%s", status, request_id)


def _age_when_phrase(age_seconds: float | None, timestamp: str | None) -> str:
    clock = _clock_label(timestamp)
    if clock is not None:
        return clock
    if age_seconds is None:
        return "an unknown time"
    minutes = max(1, int(round(age_seconds / 60.0)))
    unit = "minute" if minutes == 1 else "minutes"
    return f"approximately {minutes} {unit} ago"


def _cached_spoken(
    report: str,
    age_seconds: float | None,
    cache: dict[str, Any] | None,
    *,
    pending: bool = False,
) -> str:
    when = _age_when_phrase(age_seconds, _report_timestamp(cache))
    if pending:
        return f"The live Reef check is still running, so here is a cached Hermes report from {when}. {report}"
    return f"Hermes is currently unavailable. I have an older cached report from {when}. {report}"


def _not_current_spoken(report: str, age_seconds: float | None, timestamp: str | None) -> str:
    when = _age_when_phrase(age_seconds, timestamp)
    return f"This reef report is not current. The underlying data is from {when}. {report}"


def _reef_provenance(
    *,
    request_kind: str,
    live_cache: dict[str, Any] | None,
    thread: dict[str, Any] | None,
    retrieved_at: str,
) -> tuple[str, str | None, str | None, float | None]:
    if request_kind == REQUEST_KIND_CURRENT and live_cache is not None:
        data_timestamp = _data_timestamp(live_cache)
        report_timestamp = _report_timestamp(thread) or retrieved_at
        age_seconds = live_cache.get("age_seconds")
        if not isinstance(age_seconds, (int, float)):
            age_seconds = hermes_client.reef_cache_age_seconds(
                generated_at=data_timestamp,
                data_timestamp=data_timestamp,
            )
        return "reef_monitor", data_timestamp, report_timestamp, age_seconds
    if thread is not None:
        data_timestamp = _data_timestamp(thread)
        report_timestamp = _report_timestamp(thread) or retrieved_at
        return (
            "reef_thread_cache",
            data_timestamp,
            report_timestamp,
            _cache_age_seconds(thread),
        )
    return "hermes", None, retrieved_at, None


def _trends_from_cache(cache: dict[str, Any] | None) -> dict[str, Any]:
    trends = cache.get("trends") if cache else None
    return trends if isinstance(trends, dict) else {}


def _trend_keys(cache: dict[str, Any] | None) -> list[str]:
    return sorted(str(key) for key in _trends_from_cache(cache))


def _hermes_reef_result(
    report: str,
    *,
    request_id: str,
    request_kind: str,
    requested_at: str,
    retrieved_at: str,
    data_timestamp: str | None,
    report_timestamp: str | None,
    age_seconds: float | None,
    data_source: str,
    cache: dict[str, Any] | None,
    live_cache: dict[str, Any] | None,
    elapsed: float,
) -> dict[str, Any]:
    trends = _trends_from_cache(cache)
    trend_keys = _trend_keys(cache)
    historical = request_kind == REQUEST_KIND_HISTORY
    # HTTP 200 is not freshness: age comes from Apex/live-cache or reef_thread timestamps.
    fresh = False if historical else hermes_client.current_reef_is_fresh(age_seconds)
    latest = fresh
    if historical:
        status = "success"
        spoken = report
    elif fresh:
        status = "success"
        spoken = report
    else:
        status = "stale"
        spoken = _not_current_spoken(report, age_seconds, data_timestamp or report_timestamp)
    generated_at = None
    if live_cache is not None:
        generated_at = _optional_timestamp(live_cache.get("cached_at")) or data_timestamp
    elif cache is not None:
        generated_at = _optional_timestamp(cache.get("generated_at"))
    logger.info("[HERMES] response received request_id=%s", request_id)
    logger.info("[HERMES] http_status=200 request_id=%s", request_id)
    logger.info(
        "[REEF] Hermes report received request_kind=%s source=hermes status=%s fresh=%s "
        "data_source=%s age_seconds=%s cache_used=false elapsed=%.1fs",
        request_kind,
        status,
        str(fresh).lower(),
        data_source,
        age_seconds,
        elapsed,
    )
    logger.info("[REEF] returning Hermes Reef report to Reachy request_id=%s trend_keys=%s", request_id, trend_keys)
    _log_hermes_result(
        request_id=request_id,
        status=status,
        source="hermes",
        fresh=fresh,
        data_timestamp=data_timestamp,
        report_timestamp=report_timestamp,
        requested_at=requested_at,
        retrieved_at=retrieved_at,
        age_seconds=age_seconds,
        cache_used=False,
        data_source=data_source,
    )
    return {
        "status": status,
        "stale": not fresh,
        "fresh": fresh,
        "latest": latest,
        "source": "hermes",
        "live": fresh,
        "degraded": not fresh,
        "generated_at": generated_at,
        "data_timestamp": data_timestamp,
        "report_timestamp": report_timestamp,
        "requested_at": requested_at,
        "retrieved_at": retrieved_at,
        "age_seconds": age_seconds,
        "report": report,
        "spoken": spoken,
        "reply": spoken,
        "trend_available": True,
        "hermes_request_id": request_id,
        "cache_used": False,
        "cache_age_seconds": None,
        "data_source": data_source,
        "request_kind": request_kind,
        "trends": trends,
        "trend_keys": trend_keys,
    }


def _cached_history_result(
    report: str,
    *,
    request_id: str,
    cache: dict[str, Any],
    reason: str,
    requested_at: str | None = None,
) -> dict[str, Any]:
    age_seconds = _cache_age_seconds(cache)
    status = hermes_client.reef_cache_status_for_age(age_seconds)
    spoken = _cached_spoken(report, age_seconds, cache, pending=reason == "already_running")
    trends = _trends_from_cache(cache)
    trend_keys = _trend_keys(cache)
    retrieved_at = _utc_now()
    started_at = requested_at or retrieved_at
    data_timestamp = _data_timestamp(cache)
    report_timestamp = _report_timestamp(cache)
    if reason == "timeout":
        logger.info(
            "[REEF] Hermes timeout; returning validated cache source=cache status=%s stale=true cache_used=true",
            status,
        )
    logger.info("[REEF] cache present request_id=%s", request_id)
    logger.info("[REEF] using cached Hermes Reef report request_id=%s", request_id)
    logger.info("[REEF] cache age=%s request_id=%s", age_seconds, request_id)
    logger.info("[REEF] returning cached report to Reachy request_id=%s status=%s", request_id, status)
    logger.info("[REEF] stale=true source=cache cache_used=true request_id=%s trend_keys=%s", request_id, trend_keys)
    _log_hermes_result(
        request_id=request_id,
        status=status,
        source="cache",
        fresh=False,
        data_timestamp=data_timestamp,
        report_timestamp=report_timestamp,
        requested_at=started_at,
        retrieved_at=retrieved_at,
        age_seconds=age_seconds,
        cache_used=True,
        data_source="reef_thread_cache",
    )
    result = {
        "status": status,
        "stale": True,
        "fresh": False,
        "latest": False,
        "source": "cache",
        "live": False,
        "degraded": True,
        "generated_at": cache.get("generated_at"),
        "data_timestamp": data_timestamp,
        "report_timestamp": report_timestamp,
        "requested_at": started_at,
        "retrieved_at": retrieved_at,
        "age_seconds": age_seconds,
        "report": report,
        "spoken": spoken,
        "reply": spoken,
        "trend_available": True,
        "hermes_request_id": request_id,
        "cache_used": True,
        "cache_age_seconds": age_seconds,
        "data_source": "reef_thread_cache",
        "request_kind": REQUEST_KIND_HISTORY,
        "reason": reason,
        "trends": trends,
        "trend_keys": trend_keys,
    }
    if reason != "already_running":
        result["failure_category"] = _failure_category(reason)
    return result


def _history_error(
    request_id: str,
    reason: str,
    cache: dict[str, Any] | None = None,
    requested_at: str | None = None,
) -> dict[str, Any]:
    generated_at = cache.get("generated_at") if cache else None
    data_timestamp = _data_timestamp(cache)
    retrieved_at = _utc_now()
    started_at = requested_at or retrieved_at
    report_timestamp = _report_timestamp(cache)
    age_seconds = _cache_age_seconds(cache)
    logger.warning("[REEF] live request failed request_id=%s reason=%s", request_id, reason)
    logger.warning("[REEF] no usable cache request_id=%s", request_id)
    logger.warning("[REEF] returning error request_id=%s source=none", request_id)
    _log_hermes_result(
        request_id=request_id,
        status="error",
        source="hermes",
        fresh=False,
        data_timestamp=data_timestamp,
        report_timestamp=report_timestamp,
        requested_at=started_at,
        retrieved_at=retrieved_at,
        age_seconds=age_seconds,
        cache_used=False,
        data_source="none",
    )
    result = {
        "status": "error",
        "stale": True,
        "fresh": False,
        "latest": False,
        "source": "none",
        "live": False,
        "degraded": True,
        "reason": reason,
        "generated_at": generated_at,
        "data_timestamp": data_timestamp,
        "report_timestamp": report_timestamp,
        "requested_at": started_at,
        "retrieved_at": retrieved_at,
        "age_seconds": age_seconds,
        "report": None,
        "spoken": HISTORY_UNAVAILABLE,
        "error": HISTORY_UNAVAILABLE,
        "trend_available": False,
        "hermes_request_id": request_id,
        "cache_used": False,
        "cache_age_seconds": None,
        "data_source": "none",
        "request_kind": REQUEST_KIND_HISTORY,
    }
    if reason != "already_running":
        result["failure_category"] = _failure_category(reason)
    return result


def _fallback_to_cache_or_error(
    request_id: str,
    reason: str,
    cache: dict[str, Any] | None,
    requested_at: str | None = None,
) -> dict[str, Any]:
    logger.warning("[REEF] live Hermes unavailable request_id=%s reason=%s", request_id, reason)
    logger.info("[REEF] checking Reef cache request_id=%s", request_id)
    report = _cached_report_text(cache)
    if report is None or cache is None:
        return _history_error(request_id, reason, cache, requested_at=requested_at)
    return _cached_history_result(
        report,
        request_id=request_id,
        cache=cache,
        reason=reason,
        requested_at=requested_at,
    )


class AskHermes(Tool):
    """Forward advanced delegated tasks to the Hermes Agent API server."""

    name = "ask_hermes"
    description = (
        "Ask Hermes for advanced delegated tasks: multi-step household requests, other buses or trains (not live Route 311), research, "
        "and reef tank trends, threading/thread summaries, parameter history, ATO history, 6-hour changes, "
        "ATO time-to-empty, and reef tank reports. "
        "Use this immediately for 'trending', 'treading', 'threading', 'tank trends', 'reef tank report', "
        "'changed over the last 6 hours', or 'how much ATO have I been using' — do not use apex or "
        "reef_status for those. Do not use this for live reef tank status or current Apex numbers; use apex. "
        "Use this immediately for 'latest Hermes report' or 'latest report'. "
        "Reef results include status, stale, fresh, latest, data_timestamp, report_timestamp, "
        "requested_at, retrieved_at, age_seconds, data_source, and source. "
        "source=hermes and fresh=true is current Hermes data; speak it as current. "
        "fresh=false or status=stale is not current: still use the report numbers, but tell the user "
        "the data is not current and never call it the latest report. "
        "source=cache and stale=true and fresh=false is a cached Hermes Reef report: still use the report "
        "to answer, but tell the user it is cached/stale and never call it the latest report. "
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
        _ = deps
        if not isinstance(query, str) or not query.strip():
            logger.warning("[HERMES] ask_hermes empty query")
            return {"error": "query must be a non-empty string"}

        request_id = str(uuid.uuid4())
        requested_at = _utc_now()
        request_kind = hermes_client.reef_request_kind(query) or REQUEST_KIND_DELEGATED
        reef_request = request_kind in {REQUEST_KIND_CURRENT, REQUEST_KIND_HISTORY}
        history_request = request_kind == REQUEST_KIND_HISTORY
        live_cache = hermes_client.load_reef_live_cache() if request_kind == REQUEST_KIND_CURRENT else None
        thread = hermes_client.load_latest_reef_thread() if reef_request else None
        logger.info("[HERMES] request started request_id=%s", request_id)
        logger.info("[HERMES] intent=%s request_id=%s", request_kind, request_id)
        logger.info("[HERMES] request_type=%s request_id=%s", request_kind, request_id)
        logger.info("[HERMES] history=%s request_id=%s", str(history_request).lower(), request_id)
        logger.info("[HERMES] requested_at=%s request_id=%s", requested_at, request_id)
        logger.info(
            "[REEF] request received request_id=%s request_kind=%s history=%s live_cache=%s thread=%s query=%s",
            request_id,
            request_kind,
            history_request,
            "present" if live_cache else "missing",
            "present" if thread else "missing",
            query.strip(),
        )

        if hermes_client.hermes_is_busy():
            pending_id = hermes_client.hermes_in_flight_request_id()
            logger.info("[HERMES] existing request pending request_id=%s", pending_id)
            if reef_request:
                logger.info("[REEF] previous Hermes request still running request_id=%s", request_id)
                report = _cached_report_text(thread)
                if report is not None and thread is not None:
                    logger.info("[REEF] valid cache available request_id=%s", request_id)
                    logger.info(
                        "[REEF] returning cached Reef report rather than empty result request_id=%s", request_id
                    )
                    return _cached_history_result(
                        report,
                        request_id=request_id,
                        cache=thread,
                        reason="already_running",
                        requested_at=requested_at,
                    )
                return _history_error(request_id, "already_running", thread, requested_at=requested_at)
            logger.info("[HERMES] ask_hermes previous request still running")
            return {
                "status": "already_running",
                "message": (
                    "A previous check is still running. Tell the user you are still on it. "
                    "Do not call ask_hermes again."
                ),
            }

        if request_kind == REQUEST_KIND_CURRENT:
            outbound = hermes_client.reef_current_query(query, live_cache, thread)
            # Fresh session so prior Hermes chat cannot substitute for live reef data.
            session_id = hermes_client.new_hermes_session_id()
        elif history_request:
            outbound = hermes_client.reef_history_query(query, thread)
            session_id = hermes_client.new_hermes_session_id()
        else:
            outbound = query.strip()
            session_id = hermes_client.get_hermes_session_id()
        logger.info(
            "[HERMES] gateway=%s request_id=%s", (config.HERMES_GATEWAY_URL or "").strip() or "unset", request_id
        )
        if reef_request:
            logger.info(
                "[REEF] attempting Hermes request request_id=%s request_kind=%s live_cache=%s thread=%s",
                request_id,
                request_kind,
                "present" if live_cache else "missing",
                "present" if thread else "missing",
            )

        started = time.monotonic()
        try:
            reply = await hermes_client.send_to_hermes(
                outbound,
                session_id,
                request_id=request_id,
                request_kind=request_kind,
            )
        except HermesNotConfiguredError as exc:
            logger.warning("[HERMES] ask_hermes not configured request_id=%s", request_id)
            if reef_request:
                return _fallback_to_cache_or_error(request_id, "not_configured", thread, requested_at=requested_at)
            return {
                "error": "Hermes Gateway is not configured",
                "failure_category": exc.category,
            }
        except HermesTimeoutError as exc:
            logger.warning("[HERMES] request timed out/failed request_id=%s", request_id)
            if reef_request:
                return _fallback_to_cache_or_error(request_id, "timeout", thread, requested_at=requested_at)
            return {
                "error": "That check took too long. Ask me again if you still want it.",
                "failure_category": exc.category,
            }
        except HermesClientError as exc:
            reason = _reason_from_hermes_error(exc)
            logger.warning("[HERMES] request timed out/failed request_id=%s reason=%s: %s", request_id, reason, exc)
            if reef_request:
                return _fallback_to_cache_or_error(request_id, reason, thread, requested_at=requested_at)
            return {
                "error": "I couldn't reach the household data service.",
                "failure_category": exc.category,
            }

        if reef_request and hermes_client.is_process_narration(reply):
            logger.warning("[HERMES] result rejected as process narration request_id=%s", request_id)
            return _fallback_to_cache_or_error(request_id, "empty_or_narration", thread, requested_at=requested_at)

        retrieved_at = _utc_now()
        if reef_request:
            post_cache = hermes_client.load_reef_live_cache() if request_kind == REQUEST_KIND_CURRENT else None
            effective_live = post_cache or live_cache
            data_source, data_timestamp, report_timestamp, age_seconds = _reef_provenance(
                request_kind=request_kind,
                live_cache=effective_live,
                thread=thread,
                retrieved_at=retrieved_at,
            )
            return _hermes_reef_result(
                reply,
                request_id=request_id,
                request_kind=request_kind,
                requested_at=requested_at,
                retrieved_at=retrieved_at,
                data_timestamp=data_timestamp,
                report_timestamp=report_timestamp,
                age_seconds=age_seconds,
                data_source=data_source,
                cache=thread,
                live_cache=effective_live if request_kind == REQUEST_KIND_CURRENT else None,
                elapsed=time.monotonic() - started,
            )

        logger.info("[HERMES] ask_hermes completed request_id=%s chars=%s", request_id, len(reply))
        _log_hermes_result(
            request_id=request_id,
            status="success",
            source="hermes",
            fresh=True,
            data_timestamp=None,
            report_timestamp=retrieved_at,
            requested_at=requested_at,
            retrieved_at=retrieved_at,
            age_seconds=0.0,
            cache_used=False,
            data_source="hermes",
        )
        return {
            "status": "success",
            "stale": False,
            "fresh": True,
            "latest": True,
            "reply": reply,
            "spoken": reply,
            "report": reply,
            "data_timestamp": None,
            "report_timestamp": retrieved_at,
            "requested_at": requested_at,
            "retrieved_at": retrieved_at,
            "age_seconds": 0.0,
            "hermes_request_id": request_id,
            "source": "hermes",
            "cache_used": False,
            "data_source": "hermes",
        }
