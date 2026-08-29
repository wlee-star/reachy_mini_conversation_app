from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app.tools import home_assistant as home_assistant_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.home_assistant import HomeAssistant, match_fast_ha_commands


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
        return self.__class__.response


@pytest.fixture(autouse=True)
def _reset_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.headers = {}
    _FakeAsyncClient.response = _FakeResponse(200)
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
        ("is lamp 3 on", []),
        ("what's the weather", []),
    ],
)
def test_match_fast_ha_commands(transcript: str, expected: list[dict[str, str]]) -> None:
    """Common lamp and screen phrases map to local HA actions before the LLM tool call."""
    assert match_fast_ha_commands(transcript) == expected


def test_control_success_skips_spoken_followup() -> None:
    """A completed lamp toggle should not wait for a spoken model follow-up."""
    tool = HomeAssistant()
    assert (
        tool.wants_spoken_followup(
            {"status": "success", "service": "switch.turn_off", "entity_id": "switch.lamp_3"},
            None,
        )
        is False
    )


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
