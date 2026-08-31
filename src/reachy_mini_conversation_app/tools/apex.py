import re
import logging
from typing import Any
from dataclasses import dataclass

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies
from reachy_mini_conversation_app.tools.reef_status import _load_reef_snapshot


logger = logging.getLogger(__name__)

_WAKE_PREFIX = re.compile(
    r"^(?:hey |ok |okay )?(?:reachy|erichi|richie|rishi|ricci|ritchie|i reach a|i reachy|reach it)\s+"
)
_POLITE_PREFIX = re.compile(r"^(?:please |can you |could you |would you |tell me )+")
_TREND_RE = re.compile(
    r"\b(?:trend(?:ing|s)?|treading|thread(?:ing)?|history|historical|over time|"
    r"trajector(?:y|ies)|pattern|improving|worsening|getting (?:worse|better)|"
    r"been (?:doing|going)|changed|changing|happening|"
    r"(?:last\s+)?(?:\d+|few|six)\s*-?\s*hours?|time[- ]to[- ]empty|"
    r"been using|ato usage|slopes?)\b"
)
_REPORT_RE = re.compile(r"\b(?:report|repot|repo|analys(?:e|is)|analyze|detailed)\b")
_ASK_HERMES_RE = re.compile(r"\bask(?:ing)?\s+hermes\b")
_HERMES_SOURCE_QUESTION_RE = re.compile(
    r"\b(?:did|was|is) (?:that|this|it) (?:come from|from) hermes\b"
    r"|\bdid hermes (?:give you that|give that|send that|tell you that)\b"
    r"|\bdid you get that from hermes\b"
)
_TREND_REPORT_RE = re.compile(
    r"\b(?:trend(?:ing|s)?|treading|thread(?:ing)?|history)\s+report\b"
    r"|\breport\s+(?:on\s+)?(?:the\s+)?(?:trend(?:ing|s)?|treading|thread(?:ing)?|history)\b"
)
_LLSATO_RE = re.compile(r"\bl\s*l\s*s\s*a\s*t\s*o\b")
_ATO_RE = re.compile(r"\bato\b")
_PH_RE = re.compile(r"\bp\s*h\b")
_ORP_RE = re.compile(r"\borp\b")
_FS100_RE = re.compile(r"\bfs\s*100\b")
_TEMP_RE = re.compile(r"\b(?:temp(?:erature)?|tmp)\b")
_SALINITY_RE = re.compile(r"\bsalinity\b")
_REEF_RE = re.compile(r"\b(?:reef|tank|apex)\b")


@dataclass(frozen=True)
class ApexUtterance:
    """A spoken live-reef request that should call the local Apex tool."""

    metric: str


@dataclass(frozen=True)
class ReefRoute:
    """Deterministic reef-intent owner for the current turn."""

    intent: str
    route: str
    metric: str | None = None
    explicit_hermes_request: bool = False


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


def _normalize_reef_utterance(transcript: str) -> str:
    text = transcript.lower().strip().replace("'", "")
    text = re.sub(r"[.!?,;:]+", " ", text)
    text = _WAKE_PREFIX.sub("", text)
    text = _POLITE_PREFIX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _reef_metric(text: str) -> str | None:
    if _LLSATO_RE.search(text):
        return "llsato"
    if _ATO_RE.search(text):
        return "ato"
    if _PH_RE.search(text):
        return "ph"
    if _ORP_RE.search(text):
        return "orp"
    if _FS100_RE.search(text):
        return "fs100"
    if _SALINITY_RE.search(text):
        return "salinity"
    if _TEMP_RE.search(text):
        return "tmp"
    if _REEF_RE.search(text):
        return "status"
    return None


def classify_reef_intent(transcript: str) -> ReefRoute | None:
    """Choose local Apex for current readings, ask_hermes for report/trend/analysis."""
    text = _normalize_reef_utterance(transcript)
    if not text:
        return None
    explicit = bool(_ASK_HERMES_RE.search(text))
    metric = _reef_metric(text)
    has_reef = metric is not None
    has_trend = bool(_TREND_RE.search(text))
    has_report = bool(_REPORT_RE.search(text))
    standalone_trend_report = bool(_TREND_REPORT_RE.search(text))
    if explicit and has_reef:
        if has_trend or standalone_trend_report:
            return ReefRoute(intent="trends", route="ask_hermes", explicit_hermes_request=True)
        return ReefRoute(intent="detailed_report", route="ask_hermes", explicit_hermes_request=True)
    if has_trend and (has_reef or has_report or standalone_trend_report):
        return ReefRoute(intent="trends", route="ask_hermes", explicit_hermes_request=explicit)
    if has_report and has_reef:
        return ReefRoute(intent="detailed_report", route="ask_hermes", explicit_hermes_request=explicit)
    if has_reef:
        return ReefRoute(intent="current_stats", route="local", metric=metric)
    return None


def match_apex_intent(transcript: str) -> ApexUtterance | None:
    """Parse a live reef/Apex request without using the LLM."""
    route = classify_reef_intent(transcript)
    if route is None or route.route != "local" or route.metric is None:
        return None
    return ApexUtterance(metric=route.metric)


def match_reef_source_question(transcript: str) -> bool:
    """Return whether the user is asking which source produced the last reef answer."""
    return bool(_HERMES_SOURCE_QUESTION_RE.search(_normalize_reef_utterance(transcript)))


def spoken_reef_source_answer(source: str | None, response_type: str | None) -> str:
    """Answer a source follow-up from stored reef metadata, not the LLM."""
    if source == "hermes" and response_type == "trends":
        return "Yes. That came from Hermes' Reef Tank trend data."
    if source == "hermes":
        return "Yes. That came from Hermes' Reef Tank report."
    if source == "home_assistant":
        return "No. That came directly from the current reef tank data in Home Assistant."
    return "I don't have a previous reef reading to attribute."


def _probes_from_result(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("apex_status")
    if isinstance(status, dict):
        probes = status.get("water_parameters")
        if isinstance(probes, dict):
            return probes
    probes = result.get("water_parameters")
    if isinstance(probes, dict):
        return probes
    return {}


def _ato_from_result(result: dict[str, Any]) -> dict[str, Any]:
    status = result.get("apex_status")
    if isinstance(status, dict):
        equipment = status.get("equipment")
        if isinstance(equipment, dict):
            ato = equipment.get("ato")
            if isinstance(ato, dict):
                return ato
    equipment = result.get("equipment_status")
    if isinstance(equipment, dict):
        ato = equipment.get("ato")
        if isinstance(ato, dict):
            return ato
    return {}


def _raw_probe_value(probes: dict[str, Any], *names: str) -> object:
    by_lower = {key.lower(): value for key, value in probes.items()}
    for name in names:
        item = by_lower.get(name.lower())
        if isinstance(item, dict) and "value" in item:
            return item.get("value")
        if item is not None and not isinstance(item, dict):
            return item
    return None


def _format_raw(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _ato_raw_value(result: dict[str, Any], probes: dict[str, Any]) -> object:
    llsato = _raw_probe_value(probes, "LLSATO")
    if llsato is not None:
        return llsato
    ato = _ato_from_result(result)
    if "llsato" in ato:
        return ato.get("llsato")
    if "level" in ato:
        return ato.get("level")
    return None


def _named_reading(probes: dict[str, Any], label: str, *names: str) -> str:
    value = _raw_probe_value(probes, *names)
    if value is None:
        return f"{label} is not in the current Apex reading."
    return f"{label} is {_format_raw(value)}."


def spoken_apex_update(result: dict[str, Any], metric: str) -> str:
    """Build a spoken line from raw Apex values with no conversion or invented units."""
    if result.get("error"):
        return "I could not read the live Apex status right now."
    probes = _probes_from_result(result)
    if metric in {"llsato", "ato"}:
        value = _ato_raw_value(result, probes)
        if value is None:
            return "LLSATO is not in the current Apex reading."
        return f"LLSATO is {_format_raw(value)}."
    if metric == "tmp":
        return _named_reading(probes, "Temperature", "Tmp", "temperature", "temp")
    if metric == "ph":
        return _named_reading(probes, "pH", "pH", "ph")
    if metric == "orp":
        return _named_reading(probes, "ORP", "ORP")
    if metric == "fs100":
        return _named_reading(probes, "FS100", "FS100")
    if metric == "salinity":
        value = _raw_probe_value(probes, "salinity", "cond", "conductivity")
        if value is None:
            return "There is no salinity reading in the current Apex status."
        return f"Salinity is {_format_raw(value)}."
    parts: list[str] = []
    for name, item in probes.items():
        value = item.get("value") if isinstance(item, dict) else item
        if value is None:
            continue
        parts.append(f"{name} {_format_raw(value)}")
    if not parts:
        return "The current Apex reading has no probe values."
    return "Current readings: " + ", ".join(parts) + "."


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
        logger.info("[APEX] apex tool invoked action=%s", action or "(missing)")
        if action not in {"get_apex_status", "get_water_parameters", "get_equipment_status", "get_alerts"}:
            result = {
                "error": "action must be one of get_apex_status, get_water_parameters, "
                "get_equipment_status, get_alerts"
            }
            logger.info("[APEX] apex tool result=%s", result)
            return result

        cache, error = await _load_reef_snapshot()
        if error is not None and cache is None:
            result = {"error": error or "Apex reef cache is unavailable."}
            logger.info("[APEX] apex tool result=%s", result)
            return result
        if cache is None:
            result = {"error": "Apex reef cache is unavailable."}
            logger.info("[APEX] apex tool result=%s", result)
            return result

        if action == "get_water_parameters":
            result = self._water_parameters(cache, kwargs.get("include"))
        elif action == "get_equipment_status":
            result = self._equipment_status(cache)
        elif action == "get_alerts":
            result = self._alerts(cache)
        else:
            result = self._apex_status(cache)
        logger.info("[APEX] apex tool result=%s", result)
        return result

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
