"""Tests for the ask_hermes tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HermesRequestError,
    HermesTimeoutError,
    HermesNotConfiguredError,
)
from reachy_mini_conversation_app.tools.ask_hermes import AskHermes, _spoken_apex_readings
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def test_ask_hermes_description_keeps_simple_local_tools_out() -> None:
    """Hermes should remain available without owning simple HA/Apex calls."""
    description = AskHermes.description
    assert "advanced delegated tasks" in description
    assert "use home_assistant or apex instead" in description
    assert "tank trends" in description
    assert "do not use apex or reef_status" in description.lower()
    assert "already running" in description.lower()
    assert "live reef tank status" in description.lower()


@pytest.mark.asyncio
async def test_ask_hermes_returns_gateway_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful Hermes call is returned as reply text."""

    async def _send(text: str, session_id: str, request_id: str | None = None) -> str:
        assert text == "what's the reef temperature"
        assert session_id
        assert request_id
        return "Reef temperature is 26.1 C."

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="what's the reef temperature")

    assert result["reply"] == "Reef temperature is 26.1 C."
    assert result["source"] == "hermes"
    assert "hermes_request_id" in result


@pytest.mark.asyncio
async def test_ask_hermes_reports_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing gateway config becomes a tool error dict, not an exception."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise HermesNotConfiguredError("missing")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef status")

    assert result == {"error": "Hermes Gateway is not configured"}


@pytest.mark.asyncio
async def test_ask_hermes_reports_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway failures degrade to an error payload for the conversation loop."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise HermesRequestError("HTTP 503")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="turn on the tank lights")

    assert result == {"error": "I couldn't reach the household data service."}


@pytest.mark.asyncio
async def test_ask_hermes_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway timeout becomes a spoken-ready error, not a crash."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise HermesTimeoutError("timed out")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef status")

    assert result == {"error": "That check took too long. Ask me again if you still want it."}


@pytest.mark.asyncio
async def test_ask_hermes_rejects_empty_query() -> None:
    """Empty queries do not call the gateway."""
    result = await AskHermes()(_deps(), query="   ")
    assert result == {"error": "query must be a non-empty string"}


@pytest.mark.asyncio
async def test_ask_hermes_reports_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second ask_hermes while one is in flight returns immediately so the robot can speak."""
    posts: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        posts.append("sent")
        return "should not run"

    monkeypatch.setattr(hermes_client, "hermes_is_busy", lambda: True)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef tank report")

    assert result["status"] == "already_running"
    assert "still on it" in result["message"]
    assert posts == []


@pytest.mark.asyncio
async def test_ask_hermes_rejects_process_narration_and_falls_back_to_apex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file/process error is not treated as a reef trend and falls back to live Apex."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        return "It seems there is an issue with accessing the file content."

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["trend_available"] is False
    assert result["source"] == "apex_live_fallback"
    assert result["reply"].startswith("Historical trend data isn't available right now")
    assert "file" not in result["reply"].lower()
    assert "inspect" not in result["reply"].lower()


@pytest.mark.asyncio
async def test_ask_hermes_accepts_genuine_trend_without_apex_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real historical trend is spoken as-is and does not call Apex."""
    apex_calls: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        return "Nitrate has been falling from 20 to 8 over the recorded period, while temperature stayed near 26."

    class _FakeApex:
        async def __call__(self, _deps: ToolDependencies, **_kwargs: object) -> dict[str, object]:
            apex_calls.append("called")
            return {"water_parameters": {"temperature": {"value": 26.1}}}

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)
    monkeypatch.setattr("reachy_mini_conversation_app.tools.ask_hermes.Apex", _FakeApex)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["trend_available"] is True
    assert result["source"] == "hermes"
    assert "Nitrate has been falling" in result["reply"]
    assert apex_calls == []


def test_spoken_apex_readings_lists_probe_values() -> None:
    """Apex fallback speech uses live values and does not invent a trend."""
    spoken = _spoken_apex_readings(
        {
            "water_parameters": {
                "temperature": {"value": 26.1},
                "ph": {"value": 8.1},
            }
        }
    )
    assert spoken == "temperature 26.1, ph 8.1"
    assert _spoken_apex_readings({"error": "unavailable"}) is None
