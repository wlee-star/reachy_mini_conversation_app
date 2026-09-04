"""Authoritative local time from the system clock and IANA timezone.

Startup diagnostics establish the session timezone context. Current-time
requests always re-read the system clock. The LLM never calculates time.
"""

import re
import logging
from typing import Any
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from reachy_mini_conversation_app.config import config


logger = logging.getLogger(__name__)

SYDNEY_TIMEZONE = "Australia/Sydney"
_STARTUP_TIME_CONTEXT: dict[str, str] | None = None
_TIME_QUERY_RE = re.compile(
    r"\b(?:"
    r"what(?:s| is) the time|"
    r"what time is it(?: now| in sydney)?|"
    r"what(?:s| is) (?:today'?s )?date|"
    r"what date is it|"
    r"what day is it|"
    r"the time in sydney|"
    r"current time|"
    r"tell me the time|"
    r"have the time"
    r")\b",
    re.IGNORECASE,
)


def local_timezone_name() -> str:
    """Return the configured IANA timezone, defaulting to Australia/Sydney."""
    configured = (config.LOCAL_TIMEZONE or "").strip()
    return configured or SYDNEY_TIMEZONE


def resolve_timezone(timezone_name: str | None = None) -> str:
    """Return a valid IANA timezone, falling back to the configured local zone."""
    candidate = (timezone_name or "").strip()
    if not candidate:
        return local_timezone_name()
    lowered = candidate.lower().replace(" ", "_")
    if lowered in {"sydney", "australia/sydney"}:
        return SYDNEY_TIMEZONE
    try:
        ZoneInfo(candidate)
    except (ZoneInfoNotFoundError, KeyError, ValueError):
        logger.warning("[TIME] unknown timezone=%s; using %s", candidate, local_timezone_name())
        return local_timezone_name()
    return candidate


def utc_offset_label(moment: datetime) -> str:
    """Return a +HH:MM UTC offset from an aware datetime."""
    offset = moment.utcoffset()
    if offset is None:
        return "+00:00"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def read_local_moment(at: datetime | None = None, timezone_name: str | None = None) -> datetime:
    """Read civil time from the system clock in the given IANA timezone."""
    tz = ZoneInfo(resolve_timezone(timezone_name))
    if at is None:
        return datetime.now(tz)
    if at.tzinfo is None:
        return at.replace(tzinfo=tz)
    return at.astimezone(tz)


def format_time_12h(moment: datetime) -> str:
    """Return a 12-hour clock label such as 8:52 AM."""
    return moment.strftime("%I:%M %p").lstrip("0")


def reset_startup_time_context() -> None:
    """Clear the process startup time context. Tests only."""
    global _STARTUP_TIME_CONTEXT
    _STARTUP_TIME_CONTEXT = None


def get_startup_time_context() -> dict[str, str] | None:
    """Return the startup timezone context, if it has been established."""
    if _STARTUP_TIME_CONTEXT is None:
        return None
    return dict(_STARTUP_TIME_CONTEXT)


def establish_startup_time_context(at: datetime | None = None) -> dict[str, str]:
    """Store the authoritative local-time context once per process."""
    global _STARTUP_TIME_CONTEXT
    if _STARTUP_TIME_CONTEXT is not None:
        return dict(_STARTUP_TIME_CONTEXT)
    moment = read_local_moment(at)
    timezone_name = local_timezone_name()
    context = {
        "timezone": timezone_name,
        "startup_local_datetime": moment.isoformat(timespec="seconds"),
        "startup_utc_offset": utc_offset_label(moment),
        "startup_timestamp": f"{moment.timestamp():.3f}",
    }
    _STARTUP_TIME_CONTEXT = context
    logger.info("[TIME] startup timezone=%s", timezone_name)
    logger.info("[TIME] startup local_datetime=%s", context["startup_local_datetime"])
    logger.info("[TIME] startup utc_offset=%s", context["startup_utc_offset"])
    logger.info("[TIME] source=system_clock")
    return dict(context)


def current_local_time(
    *,
    at: datetime | None = None,
    timezone_name: str | None = None,
) -> dict[str, Any]:
    """Return structured current time from the system clock."""
    resolved = resolve_timezone(timezone_name)
    moment = read_local_moment(at, resolved)
    local_time = format_time_12h(moment)
    local_date = moment.strftime("%Y-%m-%d")
    weekday = moment.strftime("%A")
    place = "Sydney" if resolved == SYDNEY_TIMEZONE else resolved
    result: dict[str, Any] = {
        "status": "success",
        "timezone": resolved,
        "local_datetime": moment.isoformat(timespec="seconds"),
        "local_date": local_date,
        "local_time": local_time,
        "utc_offset": utc_offset_label(moment),
        "weekday": weekday,
        "source": "system_clock",
        "spoken": f"The current time in {place} is {local_time}.",
    }
    logger.info("[TIME] current local_datetime=%s", result["local_datetime"])
    logger.info("[TIME] current local_time=%s timezone=%s utc_offset=%s", local_time, resolved, result["utc_offset"])
    logger.info("[TIME] source=system_clock")
    return result


def match_time_intent(transcript: str) -> bool:
    """Return whether the utterance is a current time or date question."""
    text = transcript.lower().strip().replace("'", "")
    text = re.sub(r"[.!?,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return False
    return _TIME_QUERY_RE.search(text) is not None


def startup_time_instructions() -> str:
    """Return session instructions that pin timezone without letting the LLM clock-watch."""
    context = get_startup_time_context()
    timezone_name = context["timezone"] if context is not None else local_timezone_name()
    lines = [
        f"The system's local timezone is {timezone_name}.",
        "Never invent the current time or date from memory or conversation history.",
        "Call get_time for any time or date question and speak the returned local_time exactly.",
        "Do not calculate elapsed time from startup; call get_time again.",
    ]
    if context is not None:
        lines.insert(
            1,
            (
                f"Startup established local datetime {context['startup_local_datetime']} "
                f"with UTC offset {context['startup_utc_offset']}."
            ),
        )
    return "\n".join(lines)
