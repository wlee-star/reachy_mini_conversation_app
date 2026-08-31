"""LLM-callable wrapper around the local Home Assistant 311 bus monitor."""

import logging
from typing import Any

from reachy_mini_conversation_app.bus_monitor import (
    DEFAULT_PREPARATION_MINUTES,
    get_bus_monitor,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class MonitorBus(Tool):
    """Query live Route 311 data and manage background arrival alerts."""

    name = "monitor_bus"
    description = (
        "Query the live Route 311 arrival from the existing Home Assistant sensor and optionally "
        "start, switch, or cancel background monitoring. Always call query first for a current 311 time. "
        "Speak the returned spoken field. Use start only after the user confirms they want alerts. "
        "Use switch only when they explicitly ask to watch the following 311 instead. "
        "Use continuous only when they ask to keep monitoring later 311s. "
        "Use cancel when they want to stop watching. Do not use ask_hermes for live Route 311."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["query", "start", "cancel", "status", "switch", "continuous"],
                "description": "query reads live Home Assistant data; start begins watching; switch moves to the following service; continuous keeps watching later 311s; cancel stops; status reports the watch.",
            },
            "preparation_threshold": {
                "type": "integer",
                "enum": [10, 15],
                "description": "Preparation warning in minutes. Default 15. Use 10 only when the user asked for a 10-minute warning.",
            },
        },
        "required": ["action"],
    }

    def wants_spoken_followup(self, result: dict[str, Any] | None, error: str | None) -> bool:
        """Skip a second spoken turn when the live arrival was already announced."""
        if error is not None:
            return True
        if isinstance(result, dict) and result.get("already_spoken"):
            return False
        return True

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Run one bus-monitor operation and return a spoken-ready payload."""
        del deps
        action_raw = kwargs.get("action")
        action = action_raw if isinstance(action_raw, str) else ""
        if action not in {"query", "start", "cancel", "status", "switch", "continuous"}:
            return {"error": "action must be one of query, start, cancel, status, switch, continuous"}

        threshold_raw = kwargs.get("preparation_threshold")
        threshold = DEFAULT_PREPARATION_MINUTES
        if isinstance(threshold_raw, int) and threshold_raw in {10, 15}:
            threshold = threshold_raw

        manager = get_bus_monitor()
        try:
            if action == "query":
                result = await manager.query(preparation_threshold=threshold)
            elif action == "start":
                result = await manager.start(preparation_threshold=threshold)
            elif action == "switch":
                result = await manager.switch()
            elif action == "continuous":
                result = await manager.keep_monitoring(preparation_threshold=threshold)
            elif action == "cancel":
                result = await manager.cancel()
            else:
                result = manager.status()
        except Exception as exc:
            logger.warning("monitor_bus failed: %s", exc)
            return {"error": "Bus monitoring is unavailable right now."}

        logger.info("monitor_bus action=%s status=%s", action, result.get("status") or result.get("offer"))
        return result
