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
_FAST_BEDROOM_NOUN_FIRST = re.compile(r"(?:turn|switch)\s+(?:the\s+)?bedroom\s+(?:lamp|light)\s+(on|off)\b")
_BEDROOM_DEVICE = r"(?:the\s+)?bedroom\s+(?:lamp|light)"
_SPOKEN_BRIGHTNESS = (
    r"\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty(?:[\s-]?five)?|thirty|forty|fifty|sixty|seventy(?:[\s-]?five)?|eighty|ninety|hundred"
)
_FAST_BEDROOM_BRIGHTNESS = re.compile(
    rf"(?:(?:set|change|put)\s+)?{_BEDROOM_DEVICE}\s+"
    rf"(?:brightness\s+)?(?:to\s+|at\s+)?(?P<brightness>{_SPOKEN_BRIGHTNESS})\s*(?:%|percent)(?:\s*brightness)?"
    rf"(?:\s+and\s+(?P<kelvin>\d{{4}})\s*(?:k|kelvin))?"
)
_FAST_BEDROOM_RELATIVE_BRIGHTNESS = re.compile(
    rf"(?:make\s+{_BEDROOM_DEVICE}\s+(?P<make>brighter|dimmer)"
    rf"|(?P<verb>brighten|dim)\s+{_BEDROOM_DEVICE}"
    rf"|(?P<dir>increase|decrease)\s+{_BEDROOM_DEVICE}\s+brightness)"
)
_FAST_BEDROOM_KELVIN = re.compile(
    rf"(?:(?:set|change)\s+)?{_BEDROOM_DEVICE}\s+"
    rf"(?:(?:colou?r\s+temp(?:erature)?)\s+)?"
    rf"(?:to\s+|at\s+)?(?P<kelvin>\d{{4}})\s*(?:k|kelvin)"
    rf"(?:\s+and\s+(?P<brightness>{_SPOKEN_BRIGHTNESS})\s*(?:%|percent))?"
)
_FAST_BEDROOM_WARM_COOL = re.compile(
    rf"(?:make\s+{_BEDROOM_DEVICE}\s+(?P<make>warmer|cooler|warm|cool)"
    rf"|(?:set|change)\s+{_BEDROOM_DEVICE}\s+(?:to\s+)?(?P<set>warmer|cooler|warm|cool))"
)
_TRAILING_BRIGHTNESS = re.compile(rf"^\s+(?:and\s+)?(?:at\s+|to\s+)?({_SPOKEN_BRIGHTNESS})\s*(?:%|(?:percent)\b)")
_TRAILING_KELVIN = re.compile(r"^\s+(?:and\s+)?(?:at\s+|to\s+)?(\d{4})\s*(?:k|kelvin)\b")
_TRAILING_WARM_COOL = re.compile(r"^\s+(?:and\s+)?(?P<adj>warmer|cooler|warm|cool)\b")
_FAST_SCREEN = re.compile(r"\bscreen\s+(up|down)\b")
_BRIGHTNESS_STEP_PCT = 20
_KELVIN_STEP = 500
_WARM_KELVIN = 2700
_COOL_KELVIN = 6500
_MIN_COLOR_TEMP_K = 1500
_MAX_COLOR_TEMP_K = 6500
_BRIGHTNESS_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty five": 25,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "seventy five": 75,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
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


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _spoken_brightness_pct(token: str) -> int | None:
    normalized = re.sub(r"\s+", " ", token.strip().lower().replace("-", " "))
    if normalized.isdigit():
        return max(0, min(100, int(normalized)))
    value = _BRIGHTNESS_NUMBER_WORDS.get(normalized)
    if value is None:
        return None
    return value


def _kelvin_for_warm_cool(token: str) -> tuple[int | None, int | None]:
    if token == "warmer":
        return None, -_KELVIN_STEP
    if token == "cooler":
        return None, _KELVIN_STEP
    if token == "warm":
        return _WARM_KELVIN, None
    if token == "cool":
        return _COOL_KELVIN, None
    return None, None


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


def _bedroom_set_command(brightness: int | None, kelvin: int | None) -> dict[str, Any]:
    command: dict[str, Any] = {"action": "set_bedroom_lamp"}
    if brightness is not None:
        command["brightness_pct"] = brightness
    if kelvin is not None:
        command["color_temp_kelvin"] = max(_MIN_COLOR_TEMP_K, min(_MAX_COLOR_TEMP_K, kelvin))
    return command


def _bedroom_power_command(text: str, match: re.Match[str]) -> tuple[int, int, dict[str, Any]]:
    if match.group(1) == "off":
        return (match.start(), match.end(), {"action": "turn_light_off", "entity_id": _BEDROOM_LAMP_ENTITY_ID})

    command: dict[str, Any] = {"action": "set_bedroom_lamp"}
    end = match.end()
    rest = text[end:]
    brightness_match = _TRAILING_BRIGHTNESS.match(rest)
    if brightness_match is not None:
        brightness = _spoken_brightness_pct(brightness_match.group(1))
        if brightness is not None:
            command["brightness_pct"] = brightness
            end += brightness_match.end()
            rest = text[end:]
    kelvin_match = _TRAILING_KELVIN.match(rest)
    if kelvin_match is not None:
        command["color_temp_kelvin"] = max(_MIN_COLOR_TEMP_K, min(_MAX_COLOR_TEMP_K, int(kelvin_match.group(1))))
        end += kelvin_match.end()
        rest = text[end:]
    warm_match = _TRAILING_WARM_COOL.match(rest)
    if warm_match is not None and "color_temp_kelvin" not in command:
        kelvin, kelvin_delta = _kelvin_for_warm_cool(warm_match.group("adj"))
        if kelvin is not None:
            command["color_temp_kelvin"] = kelvin
            end += warm_match.end()
        elif kelvin_delta is not None:
            command["color_temp_delta_kelvin"] = kelvin_delta
            end += warm_match.end()
    return (match.start(), end, command)


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
        candidates.append(_bedroom_power_command(text, match))
    for match in _FAST_BEDROOM_NOUN_FIRST.finditer(text):
        candidates.append(_bedroom_power_command(text, match))
    for match in _FAST_BEDROOM_BRIGHTNESS.finditer(text):
        brightness = _spoken_brightness_pct(match.group("brightness"))
        if brightness is None:
            continue
        kelvin_token = match.group("kelvin")
        kelvin = int(kelvin_token) if kelvin_token is not None else None
        candidates.append((match.start(), match.end(), _bedroom_set_command(brightness, kelvin)))
    for match in _FAST_BEDROOM_RELATIVE_BRIGHTNESS.finditer(text):
        token = match.group("make") or match.group("verb") or match.group("dir")
        if token in {"brighter", "brighten", "increase"}:
            delta = _BRIGHTNESS_STEP_PCT
        elif token in {"dimmer", "dim", "decrease"}:
            delta = -_BRIGHTNESS_STEP_PCT
        else:
            continue
        candidates.append((match.start(), match.end(), {"action": "set_bedroom_lamp", "brightness_delta_pct": delta}))
    for match in _FAST_BEDROOM_KELVIN.finditer(text):
        kelvin = int(match.group("kelvin"))
        brightness_token = match.group("brightness")
        brightness = _spoken_brightness_pct(brightness_token) if brightness_token is not None else None
        candidates.append((match.start(), match.end(), _bedroom_set_command(brightness, kelvin)))
    for match in _FAST_BEDROOM_WARM_COOL.finditer(text):
        token = match.group("make") or match.group("set")
        if token is None:
            continue
        kelvin, kelvin_delta = _kelvin_for_warm_cool(token)
        warm_command: dict[str, Any] = {"action": "set_bedroom_lamp"}
        if kelvin is not None:
            warm_command["color_temp_kelvin"] = kelvin
        elif kelvin_delta is not None:
            warm_command["color_temp_delta_kelvin"] = kelvin_delta
        else:
            continue
        candidates.append((match.start(), match.end(), warm_command))
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


def _is_relative_bedroom_kwargs(kwargs: dict[str, Any]) -> bool:
    return kwargs.get("brightness_delta_pct") is not None or kwargs.get("color_temp_delta_kelvin") is not None


def _control_result_cache_key(action: str, kwargs: dict[str, Any], result: dict[str, Any] | None = None) -> str | None:
    entity_id = _control_target(action, kwargs)
    if result is not None:
        remembered = result.get("entity_id")
        if isinstance(remembered, str):
            entity_id = remembered
    if entity_id is None:
        return None
    if action != "set_bedroom_lamp":
        return entity_id
    if result is None and _is_relative_bedroom_kwargs(kwargs):
        return None
    source = result if result is not None else kwargs
    brightness = source.get("brightness_pct")
    kelvin = source.get("color_temp_kelvin")
    if result is None and brightness is None and kelvin is None:
        brightness = 100
    return f"{entity_id}:{brightness}:{kelvin}"


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


def _forget_bedroom_lamp_sets(entity_id: str) -> None:
    for key in list(_recent_control_results):
        if key[0] == "set_bedroom_lamp" and key[1].startswith(entity_id):
            _recent_control_results.pop(key, None)


def _remember_control_result(action: str, cache_id: str, entity_id: str, result: dict[str, Any]) -> None:
    _recent_control_results[(action, cache_id)] = (time.monotonic(), result)
    opposite = _OPPOSITE_CONTROL_ACTION.get(action)
    if opposite is not None:
        _recent_control_results.pop((opposite, entity_id), None)
    if action == "turn_light_off":
        _forget_bedroom_lamp_sets(entity_id)


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
            cached = _cached_control_result(action, _control_result_cache_key(action, kwargs))
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
            cache_id = _control_result_cache_key(action, kwargs, result)
            if isinstance(entity_id, str) and cache_id is not None:
                _remember_control_result(action, cache_id, entity_id, result)
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
        brightness_pct = _optional_int(kwargs.get("brightness_pct"))
        color_temp_kelvin = _optional_int(kwargs.get("color_temp_kelvin"))
        brightness_delta = _optional_int(kwargs.get("brightness_delta_pct"))
        color_temp_delta = _optional_int(kwargs.get("color_temp_delta_kelvin"))

        if brightness_delta is not None or color_temp_delta is not None:
            logger.info("[HA] reading local entity state: %s", entity_id)
            state_payload, error = await _request_json(
                http_client,
                "GET",
                f"{base_url}/api/states/{quote(entity_id, safe='')}",
            )
            if error is not None:
                logger.warning("[HA] service call failed: %s %s", entity_id, error)
                return {"error": error}
            if not isinstance(state_payload, dict):
                return {"error": "Home Assistant returned an unexpected state response."}
            state = state_payload.get("state")
            if state in {"unavailable", "unknown"}:
                logger.warning("[HA] service call failed: %s unavailable", entity_id)
                return {"error": "Bedroom lamp is currently unavailable."}
            attributes = state_payload.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}

            if brightness_delta is not None:
                current_brightness: int | None = None
                raw_pct = attributes.get("brightness_pct")
                if not isinstance(raw_pct, bool) and isinstance(raw_pct, int | float):
                    current_brightness = max(0, min(100, int(raw_pct)))
                else:
                    raw_brightness = attributes.get("brightness")
                    if not isinstance(raw_brightness, bool) and isinstance(raw_brightness, int | float):
                        current_brightness = max(0, min(100, int(round(raw_brightness / 255 * 100))))
                    elif state == "off":
                        current_brightness = 0
                if current_brightness is None:
                    logger.warning("[HA] service call failed: %s brightness unavailable", entity_id)
                    return {"error": "Bedroom lamp brightness is unavailable."}
                brightness_pct = max(0, min(100, current_brightness + brightness_delta))

            if color_temp_delta is not None:
                current_kelvin: int | None = None
                raw_kelvin = attributes.get("color_temp_kelvin")
                if not isinstance(raw_kelvin, bool) and isinstance(raw_kelvin, int | float):
                    current_kelvin = int(raw_kelvin)
                else:
                    mireds = attributes.get("color_temp")
                    if not isinstance(mireds, bool) and isinstance(mireds, int | float) and mireds > 0:
                        current_kelvin = int(round(1_000_000 / float(mireds)))
                if current_kelvin is None:
                    logger.warning("[HA] service call failed: %s color temperature unavailable", entity_id)
                    return {"error": "Bedroom lamp color temperature is unavailable."}
                min_k = _optional_int(attributes.get("min_color_temp_kelvin")) or _MIN_COLOR_TEMP_K
                max_k = _optional_int(attributes.get("max_color_temp_kelvin")) or _MAX_COLOR_TEMP_K
                color_temp_kelvin = max(min_k, min(max_k, current_kelvin + color_temp_delta))

        if brightness_pct is not None:
            brightness_pct = max(0, min(100, brightness_pct))
        elif color_temp_kelvin is None:
            brightness_pct = 100  # Default to 100% (lamp sometimes reverts to ~1%)

        if color_temp_kelvin is not None:
            color_temp_kelvin = max(_MIN_COLOR_TEMP_K, min(_MAX_COLOR_TEMP_K, color_temp_kelvin))

        payload: dict[str, object] = {"entity_id": entity_id}
        if brightness_pct is not None:
            payload["brightness_pct"] = brightness_pct
        if color_temp_kelvin is not None:
            payload["color_temp_kelvin"] = color_temp_kelvin
        if "brightness_pct" not in payload and "color_temp_kelvin" not in payload:
            logger.warning("[HA] service call failed: %s missing brightness and color temperature", entity_id)
            return {"error": "Bedroom lamp command is missing brightness and color temperature."}

        logger.info(
            "[HA] executing local service call: light.turn_on %s (brightness=%s%%, color_temp=%sK)",
            entity_id,
            brightness_pct if brightness_pct is not None else "unchanged",
            color_temp_kelvin if color_temp_kelvin is not None else "unchanged",
        )
        _payload, error = await _request_json(
            http_client,
            "POST",
            f"{base_url}/api/services/light/turn_on",
            json_body=payload,
        )
        if error is not None:
            logger.warning("[HA] service call failed: light.turn_on %s %s", entity_id, error)
            return {"error": error}
        logger.info("[HA] bedroom lamp set succeeded")
        return {
            "status": "success",
            "service": "light.turn_on",
            "entity_id": entity_id,
            "brightness_pct": brightness_pct,
            "color_temp_kelvin": color_temp_kelvin,
        }
