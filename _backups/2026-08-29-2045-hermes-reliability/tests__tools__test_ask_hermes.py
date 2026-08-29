"""Tests for the ask_hermes tool."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HermesRequestError,
    HermesTimeoutError,
    HermesNotConfiguredError,
)
from reachy_mini_conversation_app.tools.ask_hermes import AskHermes
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

    async def _send(text: str, session_id: str) -> str:
        assert text == "what's the reef temperature"
        assert session_id
        return "Reef temperature is 26.1 C."

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="what's the reef temperature")

    assert result == {"reply": "Reef temperature is 26.1 C."}


@pytest.mark.asyncio
async def test_ask_hermes_reports_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing gateway config becomes a tool error dict, not an exception."""

    async def _send(_text: str, _session_id: str) -> str:
        raise HermesNotConfiguredError("missing")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef status")

    assert result == {"error": "Hermes Gateway is not configured"}


@pytest.mark.asyncio
async def test_ask_hermes_reports_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway failures degrade to an error payload for the conversation loop."""

    async def _send(_text: str, _session_id: str) -> str:
        raise HermesRequestError("HTTP 503")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="turn on the tank lights")

    assert result == {"error": "I couldn't reach the household data service."}


@pytest.mark.asyncio
async def test_ask_hermes_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway timeout becomes a spoken-ready error, not a crash."""

    async def _send(_text: str, _session_id: str) -> str:
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

    async def _send(_text: str, _session_id: str) -> str:
        posts.append("sent")
        return "should not run"

    monkeypatch.setattr(hermes_client, "hermes_is_busy", lambda: True)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef tank report")

    assert result["status"] == "already_running"
    assert "still on it" in result["message"]
    assert posts == []
