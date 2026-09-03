from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app.tools import home_assistant as home_assistant_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.home_assistant import (
    SCREEN_UP_ENTITY_ID,
    HomeAssistant,
    is_control_action,
    is_screen_up_success,
    match_fast_ha_commands,
    is_device_control_success,
)


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


class _FakeResponse:
    def __init__(self, status_code: int, payload: object | None = None, content: bytes = b"{}") -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://ha.local/api")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> object:
        return self._payload


class _FakeAsyncClient:
    requests: list[tuple[str, str, dict[str, object] | None]] = []
    headers: dict[str, str] = {}
    response = _FakeResponse(200)
    responses: list[_FakeResponse] = []
    error: Exception | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        raw_headers = kwargs.get("headers") or {}
        self.__class__.headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, object] | None = None,
    ) -> _FakeResponse:
        self.__class__.requests.append((method, url, json))
        if self.__class__.error is not None:
            raise self.__class__.error
        if self.__class__.responses:
            return self.__class__.responses.pop(0)
        return self.__class__.response


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.headers = {}
    _FakeAsyncClient.response = _FakeResponse(200)
    _FakeAsyncClient.responses = []
    _FakeAsyncClient.error = None
    monkeypatch.setattr(home_assistant_mod.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(home_assistant_mod.config, "HA_URL", "http://ha.local")
    monkeypatch.setattr(home_assistant_mod.config, "HA_TOKEN", "ha-token")
    monkeypatch.setattr(home_assistant_mod.config, "HA_BUS_ENTITY_ID", None, raising=False)
    home_assistant_mod._recent_control_results.clear()


@pytest.mark.asyncio
async def test_home_assistant_reports_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing HA env config becomes a tool error."""
    monkeypatch.setattr(home_assistant_mod.config, "HA_URL", None)
    monkeypatch.setattr(home_assistant_mod.config, "HA_TOKEN", None)

    result = await HomeAssistant()(_deps(), action="get_entity_state", entity_id="light.lounge")

    assert result == {"error": "Home Assistant is not configured"}


@pytest.mark.asyncio
async def test_get_entity_state_reads_local_state() -> None:
    """State reads call the local Home Assistant states endpoint."""
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {
            "entity_id": "light.lounge",
            "state": "off",
            "attributes": {"friendly_name": "Lounge", "brightness": 120},
        },
    )

    result = await HomeAssistant()(_deps(), action="get_entity_state", entity_id="light.lounge")

    assert result["entity_id"] == "light.lounge"
    assert result["state"] == "off"
    assert result["friendly_name"] == "Lounge"
    assert _FakeAsyncClient.headers["Authorization"] == "Bearer ha-token"
    assert _FakeAsyncClient.requests == [("GET", "http://ha.local/api/states/light.lounge", None)]


@pytest.mark.asyncio
async def test_turn_light_off_calls_light_service() -> None:
    """Light off uses the local Home Assistant light service."""
    result = await HomeAssistant()(_deps(), action="turn_light_off", entity_id="light.lounge")

    assert result == {"status": "success", "service": "light.turn_off", "entity_id": "light.lounge"}
    assert _FakeAsyncClient.requests == [
        ("POST", "http://ha.local/api/services/light/turn_off", {"entity_id": "light.lounge"})
    ]


@pytest.mark.asyncio
async def test_activate_scene_calls_scene_service() -> None:
    """Scene activation uses the local Home Assistant scene service."""
    result = await HomeAssistant()(_deps(), action="activate_scene", scene_id="scene.evening")

    assert result == {"status": "success", "service": "scene.turn_on", "entity_id": "scene.evening"}
    assert _FakeAsyncClient.requests == [
        ("POST", "http://ha.local/api/services/scene/turn_on", {"entity_id": "scene.evening"})
    ]


@pytest.mark.asyncio
async def test_home_assistant_reports_auth_failure() -> None:
    """Auth failures are returned as concise tool errors."""
    _FakeAsyncClient.response = _FakeResponse(401)

    result = await HomeAssistant()(_deps(), action="turn_light_on", entity_id="light.lounge")

    assert result == {"error": "Home Assistant authentication failed."}


@pytest.mark.asyncio
async def test_home_assistant_reports_timeout() -> None:
    """Request timeouts are returned as concise availability errors."""
    request = httpx.Request("GET", "http://ha.local/api")
    _FakeAsyncClient.error = httpx.TimeoutException("timeout", request=request)

    result = await HomeAssistant()(_deps(), action="get_entity_state", entity_id="light.lounge")

    assert result == {"error": "Home Assistant is currently unavailable."}


@pytest.mark.asyncio
async def test_light_service_rejects_non_light_entity() -> None:
    """Light service actions reject non-light entity IDs."""
    result = await HomeAssistant()(_deps(), action="turn_light_on", entity_id="switch.lounge")

    assert result == {"error": "light control requires a light.* entity_id"}
    assert _FakeAsyncClient.requests == []


@pytest.mark.asyncio
async def test_get_bus_arrival_reads_configured_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bus arrivals use the configured Home Assistant sensor."""
    monkeypatch.setattr(home_assistant_mod.config, "HA_BUS_ENTITY_ID", "sensor.my_bus")
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {
            "entity_id": "sensor.my_bus",
            "state": "120",
            "attributes": {"friendly_name": "My bus"},
        },
    )

    result = await HomeAssistant()(_deps(), action="get_bus_arrival")

    assert result == {
        "minutes": 2,
        "entity_id": "sensor.my_bus",
        "friendly_name": "My bus",
        "from_state_seconds": True,
    }
    assert _FakeAsyncClient.requests == [("GET", "http://ha.local/api/states/sensor.my_bus", None)]


@pytest.mark.asyncio
async def test_get_bus_arrival_parses_route_departure_attributes() -> None:
    """Bus arrivals accept common transit attribute shapes, not only arrivals[].minutes."""
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {
            "entity_id": "sensor.route_311",
            "state": "on",
            "attributes": {
                "friendly_name": "Route 311",
                "next_departures": [
                    {
                        "route_short_name": "311",
                        "due_in": "7 min",
                        "headsign": "Central",
                        "is_realtime": True,
                    }
                ],
            },
        },
    )

    result = await HomeAssistant()(_deps(), action="get_bus_arrival", entity_id="sensor.route_311")

    assert result == {
        "entity_id": "sensor.route_311",
        "friendly_name": "Route 311",
        "minutes": 7,
        "route": "311",
        "destination": "Central",
        "realtime": True,
    }


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        (
            "Erichi Tun on lamp three.",
            [{"action": "turn_switch_on", "entity_id": "switch.lamp_3"}],
        ),
        (
            "I reach a turn on lamp three.",
            [{"action": "turn_switch_on", "entity_id": "switch.lamp_3"}],
        ),
        (
            "Turn off lamb three.",
            [{"action": "turn_switch_off", "entity_id": "switch.lamp_3"}],
        ),
        (
            "can you turn off lamp 3",
            [{"action": "turn_switch_off", "entity_id": "switch.lamp_3"}],
        ),
        (
            "lamp 3 off",
            [{"action": "turn_switch_off", "entity_id": "switch.lamp_3"}],
        ),
        (
            "screen down",
            [{"action": "press_button", "entity_id": "button.screen_down"}],
        ),
        (
            "screen up",
            [{"action": "press_button", "entity_id": "button.screen_up"}],
        ),
        (
            "turn screen up on",
            [{"action": "press_button", "entity_id": "button.screen_up"}],
        ),
        (
            "turn on screen up",
            [{"action": "press_button", "entity_id": "button.screen_up"}],
        ),
        (
            "screen up please",
            [{"action": "press_button", "entity_id": "button.screen_up"}],
        ),
        (
            "can you turn screen up on",
            [{"action": "press_button", "entity_id": "button.screen_up"}],
        ),
        (
            "Rishi, turn on lamp three and turn off lamp three.",
            [
                {"action": "turn_switch_on", "entity_id": "switch.lamp_3"},
                {"action": "turn_switch_off", "entity_id": "switch.lamp_3"},
            ],
        ),
        (
            "Rishi Tune on Lamb Three. And Tunoff Lamb Three. Ricci Tune off Lamb Three.",
            [
                {"action": "turn_switch_on", "entity_id": "switch.lamp_3"},
                {"action": "turn_switch_off", "entity_id": "switch.lamp_3"},
                {"action": "turn_switch_off", "entity_id": "switch.lamp_3"},
            ],
        ),
        (
            "turn on the bedroom light",
            [{"action": "set_bedroom_lamp"}],
        ),
        (
            "turn off the bedroom light",
            [
                {
                    "action": "turn_light_off",
                    "entity_id": "light.yeelink_sg_269831873_lamp4_s_2_light",
                }
            ],
        ),
        (
            "turn the bedroom light on",
            [{"action": "set_bedroom_lamp"}],
        ),
        (
            "Ricci can you set the bedroom light to one percent brightness?",
            [{"action": "set_bedroom_lamp", "brightness_pct": 1}],
        ),
        (
            "set the bedroom light to 1% brightness",
            [{"action": "set_bedroom_lamp", "brightness_pct": 1}],
        ),
        (
            "set the bedroom light to 25%",
            [{"action": "set_bedroom_lamp", "brightness_pct": 25}],
        ),
        (
            "set the bedroom light to 50% brightness",
            [{"action": "set_bedroom_lamp", "brightness_pct": 50}],
        ),
        (
            "set the bedroom light to 100%",
            [{"action": "set_bedroom_lamp", "brightness_pct": 100}],
        ),
        (
            "make the bedroom light brighter",
            [{"action": "set_bedroom_lamp", "brightness_delta_pct": 20}],
        ),
        (
            "make the bedroom light dimmer",
            [{"action": "set_bedroom_lamp", "brightness_delta_pct": -20}],
        ),
        (
            "increase the bedroom light brightness",
            [{"action": "set_bedroom_lamp", "brightness_delta_pct": 20}],
        ),
        (
            "decrease the bedroom light brightness",
            [{"action": "set_bedroom_lamp", "brightness_delta_pct": -20}],
        ),
        (
            "brighten the bedroom light",
            [{"action": "set_bedroom_lamp", "brightness_delta_pct": 20}],
        ),
        (
            "dim the bedroom light",
            [{"action": "set_bedroom_lamp", "brightness_delta_pct": -20}],
        ),
        (
            "set the bedroom light to 2700K",
            [{"action": "set_bedroom_lamp", "color_temp_kelvin": 2700}],
        ),
        (
            "set the bedroom light to 3000K",
            [{"action": "set_bedroom_lamp", "color_temp_kelvin": 3000}],
        ),
        (
            "set the bedroom light to 4000K",
            [{"action": "set_bedroom_lamp", "color_temp_kelvin": 4000}],
        ),
        (
            "make the bedroom light warmer",
            [{"action": "set_bedroom_lamp", "color_temp_delta_kelvin": -500}],
        ),
        (
            "make the bedroom light cooler",
            [{"action": "set_bedroom_lamp", "color_temp_delta_kelvin": 500}],
        ),
        (
            "turn the bedroom light on at 30%",
            [{"action": "set_bedroom_lamp", "brightness_pct": 30}],
        ),
        (
            "turn the bedroom light on at 50% and 3000K",
            [{"action": "set_bedroom_lamp", "brightness_pct": 50, "color_temp_kelvin": 3000}],
        ),
        (
            "turn the bedroom light on warm",
            [{"action": "set_bedroom_lamp", "color_temp_kelvin": 2700}],
        ),
        ("is lamp 3 on", []),
        ("what's the weather", []),
        ("change the bedroom light colour temperature", []),
    ],
)
def test_match_fast_ha_commands(transcript: str, expected: list[dict[str, Any]]) -> None:
    """Common lamp and screen phrases map to local HA actions before the LLM tool call."""
    assert match_fast_ha_commands(transcript) == expected


def test_control_success_wants_spoken_followup() -> None:
    """A completed lamp toggle still needs a short spoken confirmation."""
    tool = HomeAssistant()
    assert (
        tool.wants_spoken_followup(
            {"status": "success", "service": "switch.turn_off", "entity_id": "switch.lamp_3"},
            None,
        )
        is True
    )


def test_device_control_success_requires_confirmed_service_result() -> None:
    """Queued-looking payloads and errors are not treated as completed control."""
    assert is_control_action("turn_switch_on") is True
    assert is_control_action("get_entity_state") is False
    assert (
        is_device_control_success(
            {"status": "success", "service": "switch.turn_on", "entity_id": "switch.lamp_3"},
            None,
        )
        is True
    )
    assert is_device_control_success({"status": "accepted", "entity_id": "switch.lamp_3"}, None) is False
    assert is_device_control_success({"error": "Home Assistant could not find that entity or service."}, None) is False
    assert is_device_control_success({"minutes": 4, "route": "311"}, None) is False
    assert (
        is_device_control_success(
            {
                "status": "uncertain",
                "confirmation": "uncertain",
                "service": "light.turn_on",
                "entity_id": "light.lounge",
                "error": "I couldn't confirm that the device changed.",
            },
            None,
        )
        is False
    )


def test_screen_up_success_requires_confirmed_button_press() -> None:
    """Screen Up success is the confirmed button.screen_up press, not a queued or failed call."""
    assert (
        is_screen_up_success(
            {"status": "success", "service": "button.press", "entity_id": SCREEN_UP_ENTITY_ID},
            None,
        )
        is True
    )
    assert (
        is_screen_up_success(
            {"status": "success", "service": "switch.turn_on", "entity_id": "switch.lamp_3"},
            None,
        )
        is False
    )
    assert (
        is_screen_up_success(
            {"status": "success", "service": "button.press", "entity_id": "button.screen_down"},
            None,
        )
        is False
    )
    assert is_screen_up_success({"status": "accepted", "entity_id": SCREEN_UP_ENTITY_ID}, None) is False
    assert is_screen_up_success({"error": "Home Assistant is currently unavailable."}, None) is False


def test_query_and_error_still_need_spoken_followup() -> None:
    """State reads and failures still need the model to speak."""
    tool = HomeAssistant()
    assert tool.wants_spoken_followup({"minutes": 4, "route": "311"}, None) is True
    assert tool.wants_spoken_followup({"error": "Home Assistant is currently unavailable."}, None) is True


@pytest.mark.asyncio
async def test_duplicate_control_call_skips_second_http() -> None:
    """A later identical control call reuses the fast-path result instead of hitting HA again."""
    first = await HomeAssistant()(_deps(), action="turn_switch_on", entity_id="switch.lamp_3")
    second = await HomeAssistant()(_deps(), action="turn_switch_on", entity_id="switch.lamp_3")

    assert first == {"status": "success", "service": "switch.turn_on", "entity_id": "switch.lamp_3"}
    assert second == first
    assert _FakeAsyncClient.requests == [
        ("POST", "http://ha.local/api/services/switch/turn_on", {"entity_id": "switch.lamp_3"})
    ]


@pytest.mark.asyncio
async def test_opposite_control_invalidates_duplicate_cache() -> None:
    """Turning a switch off must not leave a cached on-result that skips the next on."""
    await HomeAssistant()(_deps(), action="turn_switch_on", entity_id="switch.lamp_3")
    await HomeAssistant()(_deps(), action="turn_switch_off", entity_id="switch.lamp_3")
    await HomeAssistant()(_deps(), action="turn_switch_on", entity_id="switch.lamp_3")

    assert [item[1] for item in _FakeAsyncClient.requests] == [
        "http://ha.local/api/services/switch/turn_on",
        "http://ha.local/api/services/switch/turn_off",
        "http://ha.local/api/services/switch/turn_on",
    ]


@pytest.mark.asyncio
async def test_press_screen_up_calls_button_service(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Screen Up uses the existing button.press Home Assistant service."""
    with caplog.at_level("INFO"):
        result = await HomeAssistant()(_deps(), action="press_button", entity_id=SCREEN_UP_ENTITY_ID)

    assert result == {"status": "success", "service": "button.press", "entity_id": SCREEN_UP_ENTITY_ID}
    assert is_screen_up_success(result, None) is True
    assert _FakeAsyncClient.requests == [
        ("POST", "http://ha.local/api/services/button/press", {"entity_id": SCREEN_UP_ENTITY_ID})
    ]
    assert "[HA] executing local service call: button.press button.screen_up" in caplog.text
    assert "[HA] service call succeeded: button.press button.screen_up" in caplog.text


@pytest.mark.asyncio
async def test_press_screen_up_failure_is_not_success() -> None:
    """A rejected Screen Up press is a tool error, not a confirmed activation."""
    _FakeAsyncClient.response = _FakeResponse(404)

    result = await HomeAssistant()(_deps(), action="press_button", entity_id=SCREEN_UP_ENTITY_ID)

    assert result == {"error": "Home Assistant could not find that entity or service."}
    assert is_screen_up_success(result, None) is False
    assert is_device_control_success(result, None) is False


_BEDROOM_ENTITY = "light.yeelink_sg_269831873_lamp4_s_2_light"
_BEDROOM_TURN_ON = "http://ha.local/api/services/light/turn_on"
_BEDROOM_STATE = f"http://ha.local/api/states/{_BEDROOM_ENTITY}"


@pytest.mark.asyncio
async def test_set_bedroom_lamp_on_sends_default_brightness(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bedroom ON still uses set_bedroom_lamp at 100% with no Kelvin change."""
    with caplog.at_level("INFO"):
        result = await HomeAssistant()(_deps(), action="set_bedroom_lamp")

    assert result == {
        "status": "success",
        "service": "light.turn_on",
        "entity_id": _BEDROOM_ENTITY,
        "brightness_pct": 100,
        "color_temp_kelvin": None,
    }
    assert _FakeAsyncClient.requests == [
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 100})
    ]
    assert "[HA] executing local service call: light.turn_on" in caplog.text
    assert "[HA] bedroom lamp set succeeded" in caplog.text


@pytest.mark.asyncio
async def test_set_bedroom_lamp_absolute_brightness_payload() -> None:
    """Absolute brightness uses light.turn_on with brightness_pct on the bedroom entity."""
    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_pct=1)

    assert result["status"] == "success"
    assert result["brightness_pct"] == 1
    assert _FakeAsyncClient.requests == [
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 1})
    ]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_kelvin_does_not_reset_brightness() -> None:
    """Kelvin-only commands must not force brightness back to 100%."""
    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", color_temp_kelvin=2700)

    assert result == {
        "status": "success",
        "service": "light.turn_on",
        "entity_id": _BEDROOM_ENTITY,
        "brightness_pct": None,
        "color_temp_kelvin": 2700,
    }
    assert _FakeAsyncClient.requests == [
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "color_temp_kelvin": 2700})
    ]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_on_then_brightness_is_not_cached() -> None:
    """A later brightness change must not reuse the ON cache and skip Home Assistant."""
    await HomeAssistant()(_deps(), action="set_bedroom_lamp")
    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_pct=1)

    assert result["brightness_pct"] == 1
    assert _FakeAsyncClient.requests == [
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 100}),
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 1}),
    ]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_relative_brightness_reads_state() -> None:
    """Brighter/dimmer uses the existing state endpoint, then a bounded light.turn_on."""
    _FakeAsyncClient.responses = [
        _FakeResponse(
            200,
            {
                "entity_id": _BEDROOM_ENTITY,
                "state": "on",
                "attributes": {"brightness": 128, "color_temp_kelvin": 3000},
            },
        ),
        _FakeResponse(200),
    ]

    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_delta_pct=20)

    assert result["brightness_pct"] == 70
    assert _FakeAsyncClient.requests == [
        ("GET", _BEDROOM_STATE, None),
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 70}),
    ]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_relative_brightness_clamps() -> None:
    """Relative brightness never sends values outside 0-100."""
    _FakeAsyncClient.responses = [
        _FakeResponse(
            200,
            {
                "entity_id": _BEDROOM_ENTITY,
                "state": "on",
                "attributes": {"brightness_pct": 95},
            },
        ),
        _FakeResponse(200),
    ]

    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_delta_pct=20)

    assert result["brightness_pct"] == 100
    assert _FakeAsyncClient.requests[1][2] == {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 100}


@pytest.mark.asyncio
async def test_set_bedroom_lamp_relative_kelvin_uses_entity_range() -> None:
    """Warmer/cooler stays inside the entity's supported Kelvin range."""
    _FakeAsyncClient.responses = [
        _FakeResponse(
            200,
            {
                "entity_id": _BEDROOM_ENTITY,
                "state": "on",
                "attributes": {
                    "brightness_pct": 40,
                    "color_temp_kelvin": 2700,
                    "min_color_temp_kelvin": 2700,
                    "max_color_temp_kelvin": 6500,
                },
            },
        ),
        _FakeResponse(200),
    ]

    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", color_temp_delta_kelvin=-500)

    assert result["color_temp_kelvin"] == 2700
    assert _FakeAsyncClient.requests == [
        ("GET", _BEDROOM_STATE, None),
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "color_temp_kelvin": 2700}),
    ]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_unavailable_relative_does_not_post() -> None:
    """Relative changes fail safely when the bedroom lamp state is unavailable."""
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {"entity_id": _BEDROOM_ENTITY, "state": "unavailable", "attributes": {}},
    )

    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_delta_pct=20)

    assert result == {"error": "Bedroom lamp is currently unavailable."}
    assert _FakeAsyncClient.requests == [("GET", _BEDROOM_STATE, None)]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_missing_brightness_attribute_does_not_post() -> None:
    """Relative brightness does not guess when the ON entity has no brightness."""
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {"entity_id": _BEDROOM_ENTITY, "state": "on", "attributes": {}},
    )

    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_delta_pct=-20)

    assert result == {"error": "Bedroom lamp brightness is unavailable."}
    assert _FakeAsyncClient.requests == [("GET", _BEDROOM_STATE, None)]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_clamps_out_of_range_values() -> None:
    """Out-of-range brightness and Kelvin are clamped instead of sent malformed."""
    too_bright = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_pct=101)
    too_dim = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_pct=-10)
    too_hot = await HomeAssistant()(_deps(), action="set_bedroom_lamp", color_temp_kelvin=99999)

    assert too_bright["brightness_pct"] == 100
    assert too_dim["brightness_pct"] == 0
    assert too_hot["color_temp_kelvin"] == 6500
    assert [item[2] for item in _FakeAsyncClient.requests] == [
        {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 100},
        {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 0},
        {"entity_id": _BEDROOM_ENTITY, "color_temp_kelvin": 6500},
    ]


@pytest.mark.asyncio
async def test_set_bedroom_lamp_ignores_non_numeric_kelvin() -> None:
    """Non-numeric Kelvin is dropped rather than sent to Home Assistant."""
    result = await HomeAssistant()(_deps(), action="set_bedroom_lamp", color_temp_kelvin="hot")

    assert result["color_temp_kelvin"] is None
    assert result["brightness_pct"] == 100
    assert _FakeAsyncClient.requests == [
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 100})
    ]


@pytest.mark.asyncio
async def test_bedroom_light_timeout_is_uncertain_and_not_repeated() -> None:
    """A timed-out bedroom control is uncertain and is not sent to HA a second time."""
    request = httpx.Request("POST", _BEDROOM_TURN_ON)
    _FakeAsyncClient.error = httpx.TimeoutException("timeout", request=request)

    first = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_pct=50)
    second = await HomeAssistant()(_deps(), action="set_bedroom_lamp", brightness_pct=50)

    assert first["status"] == "uncertain"
    assert first["confirmation"] == "uncertain"
    assert first["spoken"] == "I couldn't confirm that the bedroom light changed."
    assert is_device_control_success(first, None) is False
    assert second == first
    assert _FakeAsyncClient.requests == [
        ("POST", _BEDROOM_TURN_ON, {"entity_id": _BEDROOM_ENTITY, "brightness_pct": 50})
    ]


@pytest.mark.asyncio
async def test_light_timeout_is_uncertain_not_success() -> None:
    """A timed-out light service call does not claim success and is not retried."""
    request = httpx.Request("POST", "http://ha.local/api/services/light/turn_on")
    _FakeAsyncClient.error = httpx.TimeoutException("timeout", request=request)

    first = await HomeAssistant()(_deps(), action="turn_light_on", entity_id="light.lounge")
    second = await HomeAssistant()(_deps(), action="turn_light_on", entity_id="light.lounge")

    assert first["status"] == "uncertain"
    assert first["confirmation"] == "uncertain"
    assert "couldn't confirm" in first["spoken"]
    assert is_device_control_success(first, None) is False
    assert second == first
    assert _FakeAsyncClient.requests == [
        ("POST", "http://ha.local/api/services/light/turn_on", {"entity_id": "light.lounge"})
    ]
