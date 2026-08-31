import uuid
import logging
from typing import Any

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HISTORY_UNAVAILABLE,
    HermesClientError,
    HermesTimeoutError,
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
    if "HTTP " in message:
        return f"gateway_http_{message.rsplit('HTTP ', 1)[-1].rstrip('.')}"
    if "malformed JSON" in message:
        return "gateway_malformed_json"
    return "gateway_error"


def _history_success(
    report: str,
    *,
    request_id: str,
    source: str,
    cache: dict[str, Any] | None,
    cache_used: bool = False,
) -> dict[str, Any]:
    generated_at = cache.get("generated_at") if cache else None
    data_timestamp = cache.get("data_timestamp") if cache else None
    trends = cache.get("trends") if cache else None
    return {
        "status": "success",
        "source": source,
        "generated_at": generated_at,
        "data_timestamp": data_timestamp,
        "report": report,
        "spoken": report,
        "reply": report,
        "trend_available": True,
        "hermes_request_id": request_id,
        "cache_used": cache_used,
        "trends": trends if isinstance(trends, dict) else {},
    }


def _history_error(request_id: str, reason: str, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    generated_at = cache.get("generated_at") if cache else None
    data_timestamp = cache.get("data_timestamp") if cache else None
    logger.warning(
        "[HERMES] history unavailable request_id=%s reason=%s cache=%s",
        request_id,
        reason,
        "present" if cache else "missing",
    )
    return {
        "status": "error",
        "source": "hermes",
        "reason": reason,
        "generated_at": generated_at,
        "data_timestamp": data_timestamp,
        "report": None,
        "spoken": HISTORY_UNAVAILABLE,
        "error": HISTORY_UNAVAILABLE,
        "trend_available": False,
        "hermes_request_id": request_id,
        "cache_used": False,
    }


class AskHermes(Tool):
    """Forward advanced delegated tasks to the Hermes Agent API server."""

    name = "ask_hermes"
    description = (
        "Ask Hermes for advanced delegated tasks: multi-step household requests, other buses or trains (not live Route 311), research, "
        "and cached reef tank trends, threading/thread summaries, parameter history, ATO history, 6-hour changes, "
        "ATO time-to-empty, and reef tank reports. "
        "Use this immediately for 'trending', 'treading', 'threading', 'tank trends', 'reef tank report', "
        "'changed over the last 6 hours', or 'how much ATO have I been using' — do not use apex or "
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
        _ = deps
        if not isinstance(query, str) or not query.strip():
            logger.warning("[HERMES] ask_hermes empty query")
            return {"error": "query must be a non-empty string"}

        request_id = str(uuid.uuid4())
        history_request = hermes_client.is_trend_query(query)
        cache = hermes_client.load_latest_reef_thread() if history_request else None
        cached_report = cache.get("report") if cache else None
        logger.info(
            "[HERMES] ask_hermes invoked request_id=%s history=%s cache=%s query=%s",
            request_id,
            history_request,
            "present" if cache else "missing",
            query.strip(),
        )
        if history_request and isinstance(cached_report, str) and cached_report.strip():
            logger.info(
                "[HERMES] ask_hermes cache hit request_id=%s source=hermes cache_used=true",
                request_id,
            )
            return _history_success(
                cached_report.strip(),
                request_id=request_id,
                source="hermes",
                cache=cache,
                cache_used=True,
            )

        if hermes_client.hermes_is_busy():
            logger.info("[HERMES] ask_hermes previous request still running")
            return {
                "status": "already_running",
                "message": (
                    "A previous check is still running. Tell the user you are still on it. "
                    "Do not call ask_hermes again."
                ),
            }

        outbound = hermes_client.reef_history_query(query, cache) if history_request else query.strip()

        try:
            reply = await hermes_client.send_to_hermes(
                outbound,
                hermes_client.get_hermes_session_id(),
                request_id=request_id,
            )
        except HermesNotConfiguredError:
            logger.warning("[HERMES] ask_hermes not configured request_id=%s", request_id)
            if history_request and cache is not None:
                return _history_success(
                    str(cache["report"]),
                    request_id=request_id,
                    source="reefy",
                    cache=cache,
                    cache_used=True,
                )
            if history_request:
                return _history_error(request_id, "not_configured")
            return {"error": "Hermes Gateway is not configured"}
        except HermesTimeoutError:
            logger.warning("[HERMES] ask_hermes timed out request_id=%s", request_id)
            if history_request and cache is not None:
                return _history_success(
                    str(cache["report"]),
                    request_id=request_id,
                    source="reefy",
                    cache=cache,
                    cache_used=True,
                )
            if history_request:
                return _history_error(request_id, "timeout", cache)
            return {
                "error": "That check took too long. Ask me again if you still want it.",
            }
        except HermesClientError as exc:
            reason = _reason_from_hermes_error(exc)
            logger.warning("[HERMES] ask_hermes failed request_id=%s reason=%s: %s", request_id, reason, exc)
            if history_request and cache is not None:
                return _history_success(
                    str(cache["report"]),
                    request_id=request_id,
                    source="reefy",
                    cache=cache,
                    cache_used=True,
                )
            if history_request:
                return _history_error(request_id, reason, cache)
            return {"error": "I couldn't reach the household data service."}

        if history_request and hermes_client.is_process_narration(reply):
            logger.warning("[HERMES] result rejected as process narration request_id=%s", request_id)
            if cache is not None:
                return _history_success(
                    str(cache["report"]),
                    request_id=request_id,
                    source="reefy",
                    cache=cache,
                    cache_used=True,
                )
            return _history_error(request_id, "empty_or_narration")

        if history_request:
            payload = _history_success(reply, request_id=request_id, source="hermes", cache=cache)
            logger.info("[HERMES] ask_hermes result request_id=%s status=success source=hermes", request_id)
            return payload

        logger.info("[HERMES] ask_hermes completed request_id=%s chars=%s", request_id, len(reply))
        return {
            "status": "success",
            "reply": reply,
            "spoken": reply,
            "hermes_request_id": request_id,
            "source": "hermes",
            "cache_used": False,
        }
