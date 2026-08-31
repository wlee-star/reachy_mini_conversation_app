"""Tests for the ask_hermes tool."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HISTORY_UNAVAILABLE,
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
    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef tank report")

    assert result["status"] == "already_running"
    assert "still on it" in result["message"]
    assert posts == []


def _write_reef_thread(path: Path) -> None:
    path.write_text(
        (
            '{"type":"open","ts":"2026-08-31T03:00:00Z","text":"opened"}\n'
            '{"type":"run","ts":"2026-08-31T04:00:04Z","cache_ts":"2026-08-31T03:59:03Z",'
            '"summary":"Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill).",'
            '"trends":{"Tmp":{"trend_6h":-0.012,"trend_str":"-0.012/6h"},'
            '"LLSATO":{"trend_6h":-0.071,"trend_str":"-0.071/6h"}},'
            '"ato_hours_until_low":204.0}\n'
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_ask_hermes_returns_cached_report_without_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid Reefy cache is the ask_hermes fast path; the gateway is not called."""
    thread = tmp_path / "reef_thread.jsonl"
    _write_reef_thread(thread)

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise AssertionError("cached report must not call the Hermes gateway")

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What is my reef tank report?")

    assert result["status"] == "success"
    assert result["source"] == "hermes"
    assert result["cache_used"] is True
    assert result["report"] == "Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill)."
    assert "24.0C" in result["spoken"]


@pytest.mark.asyncio
async def test_ask_hermes_rejects_process_narration_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file/process error is not spoken as a trend when the Reefy cache is missing."""

    async def _send(text: str, _session_id: str, request_id: str | None = None) -> str:
        assert "Reef trend/history request" in text
        assert request_id
        return "It seems there is an issue with accessing the file content."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] == "error"
    assert result["reason"] == "empty_or_narration"
    assert result["spoken"] == HISTORY_UNAVAILABLE
    assert result["cache_used"] is False


@pytest.mark.asyncio
async def test_ask_hermes_history_error_does_not_use_live_apex(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Hermes fails and the Reefy cache is missing, say history is unavailable."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise HermesRequestError("Hermes Gateway returned HTTP 404.")

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] == "error"
    assert result["reason"] == "gateway_http_404"
    assert result["spoken"] == HISTORY_UNAVAILABLE
    assert result["trend_available"] is False
    assert "try again" not in result["spoken"].lower()


@pytest.mark.asyncio
async def test_ask_hermes_gateway_404_without_cache_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway 404 without cache must not be replaced by live Apex numbers."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise HermesRequestError("Hermes Gateway returned HTTP 404.")

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="Give me a reef tank report.")

    assert result["status"] == "error"
    assert result["reason"] == "gateway_http_404"
    assert result["spoken"] == HISTORY_UNAVAILABLE
    assert result["trend_available"] is False
    assert result["cache_used"] is False


@pytest.mark.asyncio
async def test_ask_hermes_accepts_genuine_trend_without_apex_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real historical trend is spoken as-is and does not call Apex."""

    async def _send(text: str, _session_id: str, request_id: str | None = None) -> str:
        assert "Reef trend/history request: how is my reef tank trending?" in text
        assert request_id
        return "Nitrate has been falling from 20 to 8 over the recorded period, while temperature stayed near 26."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] == "success"
    assert result["trend_available"] is True
    assert result["source"] == "hermes"
    assert "Nitrate has been falling" in result["report"]
