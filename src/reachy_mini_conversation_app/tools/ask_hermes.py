import time
import uuid
import logging
from typing import Any

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HISTORY_UNAVAILABLE,
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


def _cached_spoken(report: str, age_seconds: float | None, *, pending: bool = False) -> str:
    if age_seconds is None:
        age_phrase = "an unknown time"
    else:
        minutes = max(1, int(round(age_seconds / 60.0)))
        unit = "minute" if minutes == 1 else "minutes"
        age_phrase = f"approximately {minutes} {unit}"
    if pending:
        return (
            f"The live Reef check is still running, so here is a cached Hermes report from {age_phrase} ago. {report}"
        )
    return f"I couldn't reach the live Reef data, but I have a cached Hermes report from {age_phrase} ago. {report}"


def _trends_from_cache(cache: dict[str, Any] | None) -> dict[str, Any]:
    trends = cache.get("trends") if cache else None
    return trends if isinstance(trends, dict) else {}


def _trend_keys(cache: dict[str, Any] | None) -> list[str]:
    return sorted(str(key) for key in _trends_from_cache(cache))


def _live_history_result(
    report: str,
    *,
    request_id: str,
    cache: dict[str, Any] | None,
    elapsed: float,
) -> dict[str, Any]:
    trends = _trends_from_cache(cache)
    trend_keys = _trend_keys(cache)
    logger.info("[HERMES] response received request_id=%s", request_id)
    logger.info(
        "[REEF] Hermes live report received source=live status=success stale=false cache_used=false elapsed=%.1fs",
        elapsed,
    )
    logger.info("[REEF] live Hermes Reef report received request_id=%s chars=%s", request_id, len(report))
    logger.info("[REEF] returning live report to Reachy request_id=%s trend_keys=%s", request_id, trend_keys)
    return {
        "status": "success",
        "stale": False,
        "source": "live",
        "live": True,
        "degraded": False,
        "generated_at": cache.get("generated_at") if cache else None,
        "data_timestamp": cache.get("data_timestamp") if cache else None,
        "report": report,
        "spoken": report,
        "reply": report,
        "trend_available": True,
        "hermes_request_id": request_id,
        "cache_used": False,
        "cache_age_seconds": None,
        "trends": trends,
        "trend_keys": trend_keys,
    }


def _cached_history_result(
    report: str,
    *,
    request_id: str,
    cache: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    age_seconds = _cache_age_seconds(cache)
    status = hermes_client.reef_cache_status_for_age(age_seconds)
    spoken = _cached_spoken(report, age_seconds, pending=reason == "already_running")
    trends = _trends_from_cache(cache)
    trend_keys = _trend_keys(cache)
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
    result = {
        "status": status,
        "stale": True,
        "source": "cache",
        "live": False,
        "degraded": True,
        "generated_at": cache.get("generated_at"),
        "data_timestamp": cache.get("data_timestamp"),
        "report": report,
        "spoken": spoken,
        "reply": spoken,
        "trend_available": True,
        "hermes_request_id": request_id,
        "cache_used": True,
        "cache_age_seconds": age_seconds,
        "reason": reason,
        "trends": trends,
        "trend_keys": trend_keys,
    }
    if reason != "already_running":
        result["failure_category"] = _failure_category(reason)
    return result


def _history_error(request_id: str, reason: str, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    generated_at = cache.get("generated_at") if cache else None
    data_timestamp = cache.get("data_timestamp") if cache else None
    logger.warning("[REEF] live request failed request_id=%s reason=%s", request_id, reason)
    logger.warning("[REEF] no usable cache request_id=%s", request_id)
    logger.warning("[REEF] returning error request_id=%s source=none", request_id)
    result = {
        "status": "error",
        "stale": True,
        "source": "none",
        "live": False,
        "degraded": True,
        "reason": reason,
        "generated_at": generated_at,
        "data_timestamp": data_timestamp,
        "report": None,
        "spoken": HISTORY_UNAVAILABLE,
        "error": HISTORY_UNAVAILABLE,
        "trend_available": False,
        "hermes_request_id": request_id,
        "cache_used": False,
        "cache_age_seconds": None,
    }
    if reason != "already_running":
        result["failure_category"] = _failure_category(reason)
    return result


def _fallback_to_cache_or_error(
    request_id: str,
    reason: str,
    cache: dict[str, Any] | None,
) -> dict[str, Any]:
    logger.warning("[REEF] live Hermes unavailable request_id=%s reason=%s", request_id, reason)
    logger.info("[REEF] checking Reef cache request_id=%s", request_id)
    report = _cached_report_text(cache)
    if report is None or cache is None:
        return _history_error(request_id, reason, cache)
    return _cached_history_result(report, request_id=request_id, cache=cache, reason=reason)


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
        "Reef results include status, stale, and source. source=live and stale=false is current Hermes data; "
        "speak it as current. source=cache and stale=true is a cached Hermes Reef report: still use the report "
        "to answer, but tell the user it is cached/stale and not current. "
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
        history_request = hermes_client.is_trend_query(query)
        cache = hermes_client.load_latest_reef_thread() if history_request else None
        logger.info(
            "[REEF] request received request_id=%s history=%s cache=%s query=%s",
            request_id,
            history_request,
            "present" if cache else "missing",
            query.strip(),
        )

        if hermes_client.hermes_is_busy():
            pending_id = hermes_client.hermes_in_flight_request_id()
            logger.info("[HERMES] existing request pending request_id=%s", pending_id)
            if history_request:
                logger.info("[REEF] previous Hermes request still running request_id=%s", request_id)
                report = _cached_report_text(cache)
                if report is not None and cache is not None:
                    logger.info("[REEF] valid cache available request_id=%s", request_id)
                    logger.info(
                        "[REEF] returning cached Reef report rather than empty result request_id=%s", request_id
                    )
                    return _cached_history_result(
                        report,
                        request_id=request_id,
                        cache=cache,
                        reason="already_running",
                    )
                return _history_error(request_id, "already_running", cache)
            logger.info("[HERMES] ask_hermes previous request still running")
            return {
                "status": "already_running",
                "message": (
                    "A previous check is still running. Tell the user you are still on it. "
                    "Do not call ask_hermes again."
                ),
            }

        outbound = hermes_client.reef_history_query(query, cache) if history_request else query.strip()
        if history_request:
            logger.info("[REEF] attempting live Hermes request request_id=%s", request_id)

        started = time.monotonic()
        try:
            reply = await hermes_client.send_to_hermes(
                outbound,
                hermes_client.get_hermes_session_id(),
                request_id=request_id,
            )
        except HermesNotConfiguredError as exc:
            logger.warning("[HERMES] ask_hermes not configured request_id=%s", request_id)
            if history_request:
                return _fallback_to_cache_or_error(request_id, "not_configured", cache)
            return {
                "error": "Hermes Gateway is not configured",
                "failure_category": exc.category,
            }
        except HermesTimeoutError as exc:
            logger.warning("[HERMES] request timed out/failed request_id=%s", request_id)
            if history_request:
                return _fallback_to_cache_or_error(request_id, "timeout", cache)
            return {
                "error": "That check took too long. Ask me again if you still want it.",
                "failure_category": exc.category,
            }
        except HermesClientError as exc:
            reason = _reason_from_hermes_error(exc)
            logger.warning("[HERMES] request timed out/failed request_id=%s reason=%s: %s", request_id, reason, exc)
            if history_request:
                return _fallback_to_cache_or_error(request_id, reason, cache)
            return {
                "error": "I couldn't reach the household data service.",
                "failure_category": exc.category,
            }

        if history_request and hermes_client.is_process_narration(reply):
            logger.warning("[HERMES] result rejected as process narration request_id=%s", request_id)
            return _fallback_to_cache_or_error(request_id, "empty_or_narration", cache)

        if history_request:
            return _live_history_result(
                reply,
                request_id=request_id,
                cache=cache,
                elapsed=time.monotonic() - started,
            )

        logger.info("[HERMES] ask_hermes completed request_id=%s chars=%s", request_id, len(reply))
        return {
            "status": "success",
            "stale": False,
            "reply": reply,
            "spoken": reply,
            "hermes_request_id": request_id,
            "source": "hermes",
            "cache_used": False,
        }
