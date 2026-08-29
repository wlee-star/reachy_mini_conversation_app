"""In-memory live event log for the dashboard."""

from __future__ import annotations
import time
import threading
from typing import Any
from datetime import datetime, timezone


_MAX_EVENTS = 400
_lock = threading.Lock()
_events: list[dict[str, Any]] = []
_next_id = 1


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def emit(level: str, service: str, message: str, *, technical: str | None = None) -> dict[str, Any]:
    """Append one dashboard event and return it."""
    global _next_id
    event = {
        "id": 0,
        "ts": _now(),
        "epoch": time.time(),
        "level": level,
        "service": service,
        "message": message,
        "technical": technical,
    }
    with _lock:
        event["id"] = _next_id
        _next_id += 1
        _events.append(event)
        overflow = len(_events) - _MAX_EVENTS
        if overflow > 0:
            del _events[:overflow]
    return event


def list_events(
    *,
    after_id: int = 0,
    service: str | None = None,
    errors_only: bool = False,
) -> list[dict[str, Any]]:
    """Return events newer than after_id, optionally filtered."""
    with _lock:
        selected = [event for event in _events if event["id"] > after_id]
    if service:
        selected = [event for event in selected if event["service"] == service]
    if errors_only:
        selected = [event for event in selected if event["level"] in {"error", "warning"}]
    return selected


def clear_events() -> None:
    """Drop the in-memory event buffer."""
    with _lock:
        _events.clear()
