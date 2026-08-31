import re
import time
import logging
from typing import Any
from urllib.parse import quote

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 5.0
_CONTROL_CACHE_TTL_S = 15.0
_DEFAULT_BUS_ENTITY_ID = "sensor.route_311_at_rockwall_cres"
_BEDROOM_LAMP_ENTITY_ID = "light.yeelink_sg_269831873_lamp4_s_2_light"
SCREEN_UP_ENTITY_ID = "button.screen_up"
_CONTROL_ACTIONS = frozenset(
    {
        "turn_light_on",
        "turn_light_off",
        "turn_switch_on",
        "turn_switch_off",
        "press_button",
        "activate_scene",
        "set_bedroom_lamp",
    }
)
_SWITCH_BY_LAMP_NUMBER = {
    "1": "switch.living_room_lamp_1",
    "2": "switch.lamp_1",
    "3": "switch.lamp_3",
}
_LAMP_NUMBER_WORDS = {
    "one": "1",
    "1": "1",
    "two": "2",
    "2": "2",
    "three": "3",
    "3": "3",
}
_FAST_QUESTION_PREFIX = re.compile(r"^(?:is|are|was|whats|what is|did you|have you)\b")
_FAST_WAKE_PREFIX = re.compile(
    r"^(?:hey |ok |okay )?(?:reachy|erichi|richie|rishi|ricci|ritchie|i reach a|i reachy|reach it)\s+"
)
_FAST_POLITE_PREFIX = re.compile(r"^(?:please |can you |could you |would you )+")
_FAST_LAMP_TURN = re.compile(r"(?:turn|switch)\s+(on|off)\s+(?:the\s+)?(?:lamp|light)\s+(one|two|three|[123])\b")
_FAST_LAMP_TURN_NOUN_FIRST = re.compile(
    r"(?:turn|switch)\s+(?:the\s+)?(?:lamp|light)\s+(one|two|three|[123])\s+(on|off)\b"
)
_FAST_LAMP_TRAILING = re.compile(r"\b(?:lamp|light)\s+(one|two|three|[123])\s+(on|off)\b")
_FAST_LIVING_ROOM = re.compile(r"(?:turn|switch)\s+(on|off)\s+(?:the\s+)?living\s*room\s+(?:lamp|light)\b")
_FAST_BEDROOM = re.compile(r"(?:turn|switch)\s+(on|off)\s+(?:the\s+)?bedroom\s+(?:lamp|light)\b")
_FAST_SCREEN = re.compile(r"\bscreen\s+(up|down)\b")
_OPPOSITE_CONTROL_ACTION = {
    "turn_switch_on": "turn_switch_off",
    "turn_switch_off": "turn_switch_on",
    "turn_light_on": "turn_light_off",
    "turn_light_off": "turn_light_on",
    "set_bedroom_lamp": "turn_light_off",
}
_recent_control_results: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}


class HomeAssistantConfigError(RuntimeError):
    """Raised when Home Assistant is not configured."""


def _home_assistant_config() -> tuple[str, str]:
    base_url = (config.HA_URL or "").strip().rstrip("/")
    token = (config.HA_TOKEN or "").strip()
    if not base_url or not token:
        raise HomeAssistantConfigError("HA_URL and HA_TOKEN must be set before using Home Assistant tools.")
    return base_url, token


def _require_string(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _bus_entity_id(kwargs: dict[str, Any]) -> str:
    return (
        _require_string(kwargs.get("entity_id"))
        or _require_string(getattr(config, "HA_BUS_ENTITY_ID", None))
        or _DEFAULT_BUS_ENTITY_ID
    )


def _first_string(mapping: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _minutes_from_value(value: object, *, seconds: bool = False) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return max(0, int(value) // 60 if seconds else int(value))
    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text or text in {"unknown", "unavailable", "none"}:
        return None
    if text.isdigit():
        number = int(text)
        return max(0, number // 60 if seconds else number)

    match = re.search(r"(\d+)\s*(?:min|mins|minute|minutes|m)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)\s*(?:sec|secs|second|seconds|s)\b", text)
    if match:
        return max(0, int(match.group(1)) // 60)
    return None


def _arrival_details(source: dict[str, Any], entity_id: str, attributes: dict[str, Any]) -> dict[str, Any] | None:
    minutes = _minutes_from_value(
        source.get("minutes")
        or source.get("due_in")
        or source.get("due_in_minutes")
        or source.get("minutes_to_arrival")
        or source.get("arrival_minutes")
    )
    if minutes is None:
        minutes = _minutes_from_value(
            source.get("seconds") or source.get("due_in_seconds") or source.get("arrival_seconds"),
            seconds=True,
        )
    route = _first_string(
        source,
        ("route", "route_id", "route_short_name", "route_name", "line", "line_name", "service"),
    ) or _first_string(attributes, ("route", "route_id", "route_short_name", "route_name", "line", "line_name"))
    destination = _first_string(source, ("destination", "headsign", "trip_headsign", "direction", "towards"))
    eta_display = _first_string(source, ("eta_display", "display", "arrival_time", "due_at", "scheduled"))
    realtime = bool(source.get("realtime") or source.get("is_realtime"))

    if minutes is None and route is None and destination is None and eta_display is None:
        return None

    result: dict[str, Any] = {
        "entity_id": entity_id,
        "friendly_name": attributes.get("friendly_name"),
    }
    if minutes is not None:
        result["minutes"] = minutes
    if route is not None:
        result["route"] = route
    if destination is not None:
        result["destination"] = destination
    if eta_display is not None:
        result["eta_display"] = eta_display
    result["realtime"] = realtime
    return result


def _extract_bus_arrival(payload: object, entity_id: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    for key in ("arrivals", "next_arrivals", "departures", "next_departures", "services", "buses"):
        candidates = attributes.get(key)
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    details = _arrival_details(candidate, entity_id, attributes)
                    if details is not None:
                        return details

    details = _arrival_details(attributes, entity_id, attributes)
    if details is not None:
        return details

    minutes = _minutes_from_value(payload.get("state"), seconds=True)
    if minutes is not None:
        return {
            "minutes": minutes,
            "entity_id": entity_id,
            "friendly_name": attributes.get("friendly_name"),
            "from_state_seconds": True,
        }
    return None


def _state_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"error": "Home Assistant returned an unexpected state response."}

    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    return {
        "entity_id": payload.get("entity_id"),
        "state": payload.get("state"),
        "friendly_name": attributes.get("friendly_name"),
        "attributes": attributes,
    }


async def _request_json(
    http_client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json_body: dict[str, object] | None = None,
) -> tuple[object | None, str | None]:
    try:
        response = await http_client.request(method, url, json=json_body)
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("Home Assistant request timed out")
        return None, "Home Assistant is currently unavailable."
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        logger.warning("Home Assistant HTTP %s for %s %s", status_code, method, url)
        if status_code in {401, 403}:
            return None, "Home Assistant authentication failed."
        if status_code == 404:
            return None, "Home Assistant could not find that entity or service."
        return None, "Home Assistant rejected the request."
    except httpx.RequestError as exc:
        logger.warning("Home Assistant request failed: %s", exc)
        return None, "Home Assistant is currently unavailable."

    if not response.content:
        return {}, None
    try:
        return response.json(), None
    except ValueError:
        logger.warning("Home Assistant returned malformed JSON")
        return None, "Home Assistant returned an unexpected response."


def _normalize_fast_ha_transcript(transcript: str) -> str:
    text = transcript.lower().strip()
    text = text.replace("'", "")
    text = re.sub(r"[.!?,;:]+", " ", text)
    text = text.replace("lamb", "lamp")
    text = text.replace("tunoff", "turn off")
    text = text.replace("turnoff", "turn off")
    text = text.replace("turnon", "turn on")
    text = text.replace("tune on", "turn on")
    text = text.replace("tune off", "turn off")
    text = text.replace("tun on", "turn on")
    text = text.replace("tun off", "turn off")
    text = _FAST_WAKE_PREFIX.sub("", text)
    text = _FAST_POLITE_PREFIX.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _switch_command(direction: str, lamp_token: str) -> dict[str, Any] | None:
    lamp_number = _LAMP_NUMBER_WORDS.get(lamp_token)
    entity_id = _SWITCH_BY_LAMP_NUMBER.get(lamp_number) if lamp_number else None
    if entity_id is None:
        return None
    action = "turn_switch_on" if direction == "on" else "turn_switch_off"
    return {"action": action, "entity_id": entity_id}


def match_fast_ha_commands(transcript: str) -> list[dict[str, Any]]:
    """Parse spoken Home Assistant commands that can run before the LLM tool call."""
    text = _normalize_fast_ha_transcript(transcript)
    if not text or _FAST_QUESTION_PREFIX.search(text):
        return []

    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for match in _FAST_LAMP_TURN.finditer(text):
        command = _switch_command(match.group(1), match.group(2))
        if command is not None:
            candidates.append((match.start(), match.end(), command))
    for match in _FAST_LAMP_TURN_NOUN_FIRST.finditer(text):
        command = _switch_command(match.group(2), match.group(1))
        if command is not None:
            candidates.append((match.start(), match.end(), command))
    for match in _FAST_LAMP_TRAILING.finditer(text):
        command = _switch_command(match.group(2), match.group(1))
        if command is not None:
            candidates.append((match.start(), match.end(), command))
    for match in _FAST_LIVING_ROOM.finditer(text):
        action = "turn_switch_on" if match.group(1) == "on" else "turn_switch_off"
        candidates.append((match.start(), match.end(), {"action": action, "entity_id": _SWITCH_BY_LAMP_NUMBER["1"]}))
    for match in _FAST_BEDROOM.finditer(text):
        if match.group(1) == "off":
            command = {"action": "turn_light_off", "entity_id": _BEDROOM_LAMP_ENTITY_ID}
        else:
            command = {"action": "set_bedroom_lamp"}
        candidates.append((match.start(), match.end(), command))
    for match in _FAST_SCREEN.finditer(text):
        candidates.append(
            (match.start(), match.end(), {"action": "press_button", "entity_id": f"button.screen_{match.group(1)}"})
        )

    candidates.sort(key=lambda item: (item[0], item[0] - item[1]))
    commands: list[dict[str, Any]] = []
    last_end = -1
    for start, end, command in candidates:
        if start < last_end:
            continue
        commands.append(command)
        last_end = end
    return commands


def _control_target(action: str, kwargs: dict[str, Any]) -> str | None:
    if action == "activate_scene":
        return _require_string(kwargs.get("scene_id")) or _require_string(kwargs.get("entity_id"))
    if action == "set_bedroom_lamp":
        return _BEDROOM_LAMP_ENTITY_ID
    return _require_string(kwargs.get("entity_id"))


def _cached_control_result(action: str, entity_id: str | None) -> dict[str, Any] | None:
    if entity_id is None:
        return None
    cached = _recent_control_results.get((action, entity_id))
    if cached is None:
        return None
    stored_at, result = cached
    if time.monotonic() - stored_at > _CONTROL_CACHE_TTL_S:
        _recent_control_results.pop((action, entity_id), None)
        return None
    return result


def is_control_action(action: object) -> bool:
    """Return whether this Home Assistant action changes a device."""
    return isinstance(action, str) and action in _CONTROL_ACTIONS


def is_device_control_success(result: dict[str, Any] | None, error: str | None) -> bool:
    """Return whether a Home Assistant result is a confirmed control completion."""
    if error is not None or not isinstance(result, dict) or "error" in result:
        return False
    return result.get("status") == "success" and bool(result.get("service"))


def is_screen_up_command(action: object, entity_id: object) -> bool:
    """Return whether this Home Assistant call is the Screen Up button press."""
    return action == "press_button" and entity_id == SCREEN_UP_ENTITY_ID


def is_screen_up_success(result: dict[str, Any] | None, error: str | None) -> bool:
    """Return whether Home Assistant confirmed a successful Screen Up press."""
    if not is_device_control_success(result, error) or result is None:
        return False
    return result.get("entity_id") == SCREEN_UP_ENTITY_ID


def _remember_control_result(action: str, entity_id: str, result: dict[str, Any]) -> None:
    _recent_control_results[(action, entity_id)] = (time.monotonic(), result)
    opposite = _OPPOSITE_CONTROL_ACTION.get(action)
    if opposite is not None:
        _recent_control_results.pop((opposite, entity_id), None)
    if action == "turn_light_off":
        _recent_control_results.pop(("set_bedroom_lamp", entity_id), None)


class HomeAssistant(Tool):
    """Local Home Assistant REST API tool."""

    name = "home_assistant"
    description = (
        "Control or read Home Assistant locally without Hermes. Use for simple Home Assistant requests:\n"
        "- get_entity_state: read any entity (light, switch, sensor, button, etc.)\n"
        "- turn_light_on/turn_light_off: control lights with optional brightness_pct (0-100)\n"
        "- set_bedroom_lamp: convenience for bedroom lamp (always uses light.yeelink_sg_269831873_lamp4_s_2_light) with brightness_pct (default 100) and color_temp_kelvin (1500-6500)\n"
        "- turn_switch_on/turn_switch_off: control switches (e.g., switch.lamp_3, switch.lamp_1, switch.living_room_lamp_1)\n"
        "- press_button: press momentary buttons; screen up/down requests always use "
        "button.screen_up/button.screen_down, never move_head. Call this in the same turn; "
        "do not only say you will press the screen.\n"
        "- get_bus_arrival: next bus from the configured Home Assistant bus sensor "
        "(returns route, minutes, destination, realtime flag)\n"
        "- activate_scene: run a scene (scene.*)\n"
        "Provides exact entity_id values for common entities."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "get_entity_state",
                    "turn_light_on",
                    "turn_light_off",
                    "turn_switch_on",
                    "turn_switch_off",
                    "press_button",
                    "activate_scene",
                    "get_bus_arrival",
                    "set_bedroom_lamp",
                ],
                "description": "The Home Assistant operation to perform.",
            },
            "entity_id": {
                "type": "string",
                "description": "Entity ID for state reads, light/switch/button control, or get_bus_arrival override. Common: light.yeelink_sg_269831873_lamp4_s_2_light (bedroom lamp), switch.lamp_3 (lamp 3), switch.lamp_1 (lamp 2), switch.living_room_lamp_1 (lamp 1), button.screen_up, button.screen_down, sensor.route_311_at_rockwall_cres (bus).",
            },
            "scene_id": {
                "type": "string",
                "description": "Scene entity ID for activate_scene.",
            },
            "brightness_pct": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Brightness percentage for turn_light_on or set_bedroom_lamp (default 100).",
            },
            "color_temp_kelvin": {
                "type": "integer",
                "minimum": 1500,
                "maximum": 6500,
                "description": "Color temperature in Kelvin for bedroom lamp (e.g., 2700 warm, 4000 neutral, 6500 cool).",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Execute one narrow Home Assistant operation over the local REST API."""
        action_raw = kwargs.get("action")
        action = action_raw if isinstance(action_raw, str) else ""
        valid_actions = {
            "get_entity_state",
            "turn_light_on",
            "turn_light_off",
            "turn_switch_on",
            "turn_switch_off",
            "press_button",
            "activate_scene",
            "get_bus_arrival",
            "set_bedroom_lamp",
        }
        if action not in valid_actions:
            return {"error": f"action must be one of {', '.join(sorted(valid_actions))}"}

        if action in _CONTROL_ACTIONS:
            cached = _cached_control_result(action, _control_target(action, kwargs))
            if cached is not None:
                logger.info("[HA] skipping duplicate local service call: %s", action)
                return cached

        try:
            base_url, token = _home_assistant_config()
        except HomeAssistantConfigError:
            logger.warning("home_assistant: Home Assistant is not configured")
            return {"error": "Home Assistant is not configured"}

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S, headers=headers) as http_client:
            if action == "get_entity_state":
                result = await self._get_entity_state(http_client, base_url, kwargs)
            elif action in {"turn_light_on", "turn_light_off"}:
                result = await self._call_light_service(http_client, base_url, action, kwargs)
            elif action in {"turn_switch_on", "turn_switch_off"}:
                result = await self._call_switch_service(http_client, base_url, action, kwargs)
            elif action == "press_button":
                result = await self._press_button(http_client, base_url, kwargs)
            elif action == "get_bus_arrival":
                result = await self._get_bus_arrival(http_client, base_url, kwargs)
            elif action == "set_bedroom_lamp":
                result = await self._set_bedroom_lamp(http_client, base_url, kwargs)
            else:
                result = await self._activate_scene(http_client, base_url, kwargs)

        if action in _CONTROL_ACTIONS and isinstance(result, dict) and result.get("status") == "success":
            entity_id = result.get("entity_id")
            if isinstance(entity_id, str):
                _remember_control_result(action, entity_id, result)
        return result

    async def _get_entity_state(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        entity_id = _require_string(kwargs.get("entity_id"))
        if entity_id is None:
            return {"error": "entity_id is required"}

        logger.info("[HA] reading local entity state: %s", entity_id)
        payload, error = await _request_json(
            http_client,
            "GET",
            f"{base_url}/api/states/{quote(entity_id, safe='')}",
        )
        if error is not None:
            return {"error": error}
        state = _state_payload(payload)
        if "error" not in state:
            logger.info("[HA] state read succeeded: %s", entity_id)
        return state

    async def _call_light_service(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        action: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        entity_id = _require_string(kwargs.get("entity_id"))
        if entity_id is None:
            return {"error": "entity_id is required"}
        if not entity_id.startswith("light."):
            return {"error": "light control requires a light.* entity_id"}

        service = "turn_on" if action == "turn_light_on" else "turn_off"
        logger.info("[HA] executing local service call: light.%s %s", service, entity_id)
        _payload, error = await _request_json(
            http_client,
            "POST",
            f"{base_url}/api/services/light/{service}",
            json_body={"entity_id": entity_id},
        )
        if error is not None:
            return {"error": error}
        logger.info("[HA] service call succeeded: light.%s %s", service, entity_id)
        return {"status": "success", "service": f"light.{service}", "entity_id": entity_id}

    async def _activate_scene(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        scene_id = _require_string(kwargs.get("scene_id")) or _require_string(kwargs.get("entity_id"))
        if scene_id is None:
            return {"error": "scene_id is required"}
        if not scene_id.startswith("scene."):
            return {"error": "activate_scene requires a scene.* entity_id"}

        logger.info("[HA] executing local service call: scene.turn_on %s", scene_id)
        _payload, error = await _request_json(
            http_client,
            "POST",
            f"{base_url}/api/services/scene/turn_on",
            json_body={"entity_id": scene_id},
        )
        if error is not None:
            return {"error": error}
        logger.info("[HA] service call succeeded: scene.turn_on %s", scene_id)
        return {"status": "success", "service": "scene.turn_on", "entity_id": scene_id}

    async def _call_switch_service(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        action: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        entity_id = _require_string(kwargs.get("entity_id"))
        if entity_id is None:
            return {"error": "entity_id is required"}
        if not entity_id.startswith("switch."):
            return {"error": "switch control requires a switch.* entity_id"}

        service = "turn_on" if action == "turn_switch_on" else "turn_off"
        logger.info("[HA] executing local service call: switch.%s %s", service, entity_id)
        _payload, error = await _request_json(
            http_client,
            "POST",
            f"{base_url}/api/services/switch/{service}",
            json_body={"entity_id": entity_id},
        )
        if error is not None:
            return {"error": error}
        logger.info("[HA] service call succeeded: switch.%s %s", service, entity_id)
        return {"status": "success", "service": f"switch.{service}", "entity_id": entity_id}

    async def _press_button(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        entity_id = _require_string(kwargs.get("entity_id"))
        if entity_id is None:
            return {"error": "entity_id is required"}
        if not entity_id.startswith("button."):
            return {"error": "button press requires a button.* entity_id"}

        logger.info("[HA] executing local service call: button.press %s", entity_id)
        _payload, error = await _request_json(
            http_client,
            "POST",
            f"{base_url}/api/services/button/press",
            json_body={"entity_id": entity_id},
        )
        if error is not None:
            return {"error": error}
        logger.info("[HA] service call succeeded: button.press %s", entity_id)
        return {"status": "success", "service": "button.press", "entity_id": entity_id}

    async def _get_bus_arrival(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Get next bus arrival from the configured Home Assistant sensor."""
        entity_id = _bus_entity_id(kwargs)
        logger.info("[HA] reading bus arrival: %s", entity_id)
        payload, error = await _request_json(
            http_client,
            "GET",
            f"{base_url}/api/states/{quote(entity_id, safe='')}",
        )
        if error is not None:
            return {"error": error, "entity_id": entity_id}

        arrival = _extract_bus_arrival(payload, entity_id)
        if arrival is not None:
            logger.info("[HA] bus arrival read succeeded: %s", arrival)
            return arrival

        logger.warning("[HA] bus arrival unavailable")
        return {"error": "Bus arrival data unavailable", "entity_id": entity_id}

    async def _set_bedroom_lamp(
        self,
        http_client: httpx.AsyncClient,
        base_url: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Control bedroom lamp (Yeelink) with brightness and color temperature."""
        entity_id = _BEDROOM_LAMP_ENTITY_ID

        brightness_pct = kwargs.get("brightness_pct")
        if brightness_pct is not None:
            try:
                brightness_pct = int(brightness_pct)
            except (TypeError, ValueError):
                brightness_pct = None
        if brightness_pct is None:
            brightness_pct = 100  # Default to 100% (lamp sometimes reverts to ~1%)
        brightness_pct = max(0, min(100, brightness_pct))

        color_temp_kelvin = kwargs.get("color_temp_kelvin")
        if color_temp_kelvin is not None:
            try:
                color_temp_kelvin = int(color_temp_kelvin)
            except (TypeError, ValueError):
                color_temp_kelvin = None
        if color_temp_kelvin is not None:
            color_temp_kelvin = max(1500, min(6500, color_temp_kelvin))

        payload = {"entity_id": entity_id, "brightness_pct": brightness_pct}
        if color_temp_kelvin is not None:
            payload["color_temp_kelvin"] = color_temp_kelvin

        logger.info(
            "[HA] executing local service call: light.turn_on %s (brightness=%s%%, color_temp=%sK)",
            entity_id,
            brightness_pct,
            color_temp_kelvin or "unchanged",
        )
        _payload, error = await _request_json(
            http_client,
            "POST",
            f"{base_url}/api/services/light/turn_on",
            json_body=payload,
        )
        if error is not None:
            return {"error": error}
        logger.info("[HA] bedroom lamp set succeeded")
        return {
            "status": "success",
            "service": "light.turn_on",
            "entity_id": entity_id,
            "brightness_pct": brightness_pct,
            "color_temp_kelvin": color_temp_kelvin,
        }
