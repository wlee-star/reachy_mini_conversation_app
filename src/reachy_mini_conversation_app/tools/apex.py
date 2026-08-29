import logging
from typing import Any

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.reef_status import _load_reef_snapshot


logger = logging.getLogger(__name__)


def _dict_field(cache: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = cache.get(field_name)
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _list_field(cache: dict[str, Any], field_name: str) -> list[Any]:
    value = cache.get(field_name)
    return value if isinstance(value, list) else []


def _filter_probes(probes: dict[str, Any], include: object) -> dict[str, Any]:
    if include is None:
        return probes
    if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
        return {}
    requested = {item.strip() for item in include if item.strip()}
    return {key: value for key, value in probes.items() if key in requested}


class Apex(Tool):
    """Local deterministic Apex status tool."""

    name = "apex"
    description = (
        "Get current Neptune Apex reef status from the local Apex /status URL when APEX_STATUS_URL "
        "is set, otherwise the reef cache. Use immediately for live tank status, current temperature, "
        "pH, ORP, salinity, equipment, alarm, or alert. Not for historical trends, threading, or ATO "
        "history — use ask_hermes for those. Returns structured current data."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get_apex_status", "get_water_parameters", "get_equipment_status", "get_alerts"],
                "description": "The Apex status operation to perform.",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional water parameter names to include for get_water_parameters.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Read one deterministic Apex status view from /status or the reef cache."""
        action_raw = kwargs.get("action")
        action = action_raw if isinstance(action_raw, str) else ""
        if action not in {"get_apex_status", "get_water_parameters", "get_equipment_status", "get_alerts"}:
            return {
                "error": "action must be one of get_apex_status, get_water_parameters, "
                "get_equipment_status, get_alerts"
            }

        cache, error = await _load_reef_snapshot()
        if error is not None and cache is None:
            return {"error": error or "Apex reef cache is unavailable."}
        if cache is None:
            return {"error": "Apex reef cache is unavailable."}

        logger.info("[APEX] executing local status read: %s", action)
        if action == "get_water_parameters":
            return self._water_parameters(cache, kwargs.get("include"))
        if action == "get_equipment_status":
            return self._equipment_status(cache)
        if action == "get_alerts":
            return self._alerts(cache)
        return self._apex_status(cache)

    def _apex_status(self, cache: dict[str, Any]) -> dict[str, Any]:
        return {
            "apex_status": {
                "water_parameters": _dict_field(cache, "probes"),
                "equipment": self._equipment_payload(cache),
                "alerts": _list_field(cache, "alerts"),
                "alarms": _dict_field(cache, "alarms"),
                "controller": cache.get("controller"),
                "cached_at": cache.get("cached_at"),
                "age_seconds": cache.get("age_seconds"),
                "stale": cache.get("stale", False),
            },
            "source": _source(cache),
        }

    def _water_parameters(self, cache: dict[str, Any], include: object) -> dict[str, Any]:
        probes = _dict_field(cache, "probes")
        filtered_probes = _filter_probes(probes, include)
        if include is not None and not filtered_probes:
            return {"error": "include must be a list of known water parameter names"}
        return {
            "water_parameters": filtered_probes,
            "cached_at": cache.get("cached_at"),
            "age_seconds": cache.get("age_seconds"),
            "stale": cache.get("stale", False),
            "source": _source(cache),
        }

    def _equipment_status(self, cache: dict[str, Any]) -> dict[str, Any]:
        return {
            "equipment_status": self._equipment_payload(cache),
            "cached_at": cache.get("cached_at"),
            "age_seconds": cache.get("age_seconds"),
            "stale": cache.get("stale", False),
            "source": _source(cache),
        }

    def _alerts(self, cache: dict[str, Any]) -> dict[str, Any]:
        return {
            "alerts": _list_field(cache, "alerts"),
            "alarms": _dict_field(cache, "alarms"),
            "cached_at": cache.get("cached_at"),
            "age_seconds": cache.get("age_seconds"),
            "stale": cache.get("stale", False),
            "source": _source(cache),
        }

    def _equipment_payload(self, cache: dict[str, Any]) -> dict[str, Any]:
        outlets = cache.get("outlets")
        return {
            "ato": _dict_field(cache, "ato"),
            "controller": cache.get("controller"),
            "outlets": outlets if isinstance(outlets, list) else [],
        }


def _source(cache: dict[str, Any]) -> str:
    source = cache.get("source")
    return source if isinstance(source, str) else "reef_cache_direct"
