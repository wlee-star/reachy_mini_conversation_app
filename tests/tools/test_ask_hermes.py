"""Tests for the ask_hermes tool."""

import time
import asyncio
from typing import Any
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HISTORY_UNAVAILABLE,
    HermesRequestError,
    HermesTimeoutError,
    HermesCircuitOpenError,
    HermesNotConfiguredError,
)
from reachy_mini_conversation_app.tools.ask_hermes import AskHermes
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import _hermes_result_text


TREND_QUERY = "Can you tell me what my reef tank is trending at?"
_TREND_KEYS = ["FS100", "LLSATO", "ORP", "Tmp", "pH"]


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _iso(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fresh_live_cache(*, age_seconds: float = 12.0) -> dict[str, Any]:
    timestamp = _iso(timedelta(seconds=age_seconds))
    return {
        "cached_at": timestamp,
        "data_timestamp": timestamp,
        "age_seconds": age_seconds,
        "probes": {"Tmp": {"value": 24.5, "status": "ok"}},
        "ato": {"value": 13.4, "status": "HIGH"},
        "source": "reef_monitor",
    }


def _stale_live_cache() -> dict[str, Any]:
    return _fresh_live_cache(age_seconds=1201.0)


def _write_reef_thread(path: Path, *, generated_at: str | None = None, summary: str | None = None) -> None:
    ts = generated_at or _iso(timedelta(minutes=5))
    report = summary or "Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill)."
    path.write_text(
        (
            '{"type":"open","ts":"2026-08-31T03:00:00Z","text":"opened"}\n'
            f'{{"type":"run","ts":"{ts}","cache_ts":"{ts}",'
            f'"summary":"{report}",'
            '"trends":{"FS100":{"trend_6h":0.4716,"trend_str":"+0.472/6h"},'
            '"LLSATO":{"trend_6h":-0.071,"trend_str":"-0.071/6h"},'
            '"ORP":{"trend_6h":-0.4714,"trend_str":"-0.471/6h"},'
            '"Tmp":{"trend_6h":-0.012,"trend_str":"-0.012/6h"},'
            '"pH":{"trend_6h":0.0118,"trend_str":"+0.012/6h"}},'
            '"ato_hours_until_low":204.0}\n'
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _fresh_hermes_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own lock and must not read the real reef-monitor cache."""
    hermes_client._HERMES_REQUEST_LOCK = asyncio.Lock()
    hermes_client._HERMES_IN_FLIGHT_REQUEST_ID = None
    hermes_client.reset_hermes_circuit()
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: None)


def test_ask_hermes_description_keeps_simple_local_tools_out() -> None:
    """Hermes should remain available without owning simple HA/Apex calls."""
    description = AskHermes.description
    assert "advanced delegated tasks" in description
    assert "use home_assistant or apex instead" in description
    assert "tank trends" in description
    assert "do not use apex or reef_status" in description.lower()
    assert "already running" in description.lower()
    assert "live reef tank status" in description.lower()
    assert "source=hermes" in description
    assert "source=cache" in description
    assert "stale=true" in description
    assert "fresh=true" in description


@pytest.mark.asyncio
async def test_ask_hermes_latest_report_attempts_live_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A latest-Hermes-report question must call Hermes, not answer from cache alone."""
    posts: list[str] = []

    async def _send(text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append(text)
        return "Hermes latest Reef report: temp 24.1C."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What's the latest Hermes report?")

    assert posts, "latest report must call Hermes"
    assert "Reef current request:" in posts[0]
    assert "Reef historical request:" not in posts[0]
    assert result["status"] == "success"
    assert result["fresh"] is True
    assert result["latest"] is True
    assert result["source"] == "hermes"
    assert result["data_source"] == "reef_monitor"
    assert result["data_timestamp"]
    assert result["report_timestamp"]
    assert result["requested_at"]
    assert result["retrieved_at"]
    assert result["age_seconds"] == 12.0
    assert result["report"] == "Hermes latest Reef report: temp 24.1C."


@pytest.mark.asyncio
async def test_ask_hermes_returns_gateway_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful Hermes call is returned as reply text."""

    async def _send(text: str, session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
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

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesNotConfiguredError("missing")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef status")

    assert result["error"] == "Hermes Gateway is not configured"
    assert result["failure_category"] == "HERMES_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_ask_hermes_reports_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway failures degrade to an error payload for the conversation loop."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesRequestError("HTTP 503")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="turn on the tank lights")

    assert result["error"] == "I couldn't reach the household data service."
    assert result["failure_category"] == "HERMES_HTTP_ERROR"


@pytest.mark.asyncio
async def test_ask_hermes_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway timeout becomes a spoken-ready error, not a crash."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesTimeoutError("timed out")

    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="reef status")

    assert result["error"] == "That check took too long. Ask me again if you still want it."
    assert result["failure_category"] == "HERMES_TIMEOUT"


@pytest.mark.asyncio
async def test_ask_hermes_rejects_empty_query() -> None:
    """Empty queries do not call the gateway."""
    result = await AskHermes()(_deps(), query="   ")
    assert result == {"error": "query must be a non-empty string"}


@pytest.mark.asyncio
async def test_ask_hermes_reports_already_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second non-Reef ask_hermes while one is in flight returns immediately so the robot can speak."""
    posts: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append("sent")
        return "should not run"

    monkeypatch.setattr(hermes_client, "hermes_is_busy", lambda: True)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="when's the next 311")

    assert result["status"] == "already_running"
    assert "still on it" in result["message"]
    assert posts == []


@pytest.mark.asyncio
async def test_ask_hermes_pending_without_cache_is_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stuck Hermes Reef request with no cache must not hang; it returns a controlled error."""
    posts: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append("sent")
        return "should not run"

    monkeypatch.setattr(hermes_client, "hermes_is_busy", lambda: True)
    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query=TREND_QUERY)

    assert posts == []
    assert result["status"] == "error"
    assert result["stale"] is True
    assert result["source"] == "none"
    assert result["cache_used"] is False
    assert result["reason"] == "already_running"
    assert result["spoken"] == HISTORY_UNAVAILABLE


@pytest.mark.asyncio
async def test_ask_hermes_live_reef_report_reaches_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A live Hermes Reef report is returned to the registered Reachy tool and LLM payload."""
    live_report = "Hermes live Reef report: temp 24.1C, nitrate falling, ATO 2.9."
    posts: list[str] = []

    async def _send(text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append(text)
        assert request_id
        return live_report

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What did Hermes report about my Reef?")

    assert posts, "Hermes must be called"
    assert result["status"] == "success"
    assert result["stale"] is False
    assert result["fresh"] is True
    assert result["latest"] is True
    assert result["source"] == "hermes"
    assert result["data_source"] == "reef_monitor"
    assert result["report"] == live_report
    assert result["data_timestamp"]
    assert result["report_timestamp"]
    assert result["requested_at"]
    assert result["retrieved_at"]
    assert result["cache_used"] is False
    assert result["trend_keys"] == []
    assert _hermes_result_text(result) == live_report
    assert "latest" not in live_report.lower() or result["latest"] is True


@pytest.mark.asyncio
async def test_ask_hermes_cache_does_not_bypass_live_hermes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A present reef_thread.jsonl must not skip a working Hermes request."""
    thread = tmp_path / "reef_thread.jsonl"
    _write_reef_thread(thread)
    live_report = "Live Hermes Reef report: temp 24.1C and nitrate falling."
    posts: list[str] = []

    async def _send(text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append(text)
        assert "Reef current request" in text
        assert "only data source" not in text
        assert "24.5" in text
        return live_report

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What is my reef tank report?")

    assert posts, "cache must not bypass Hermes"
    assert result["status"] == "success"
    assert result["stale"] is False
    assert result["source"] == "hermes"
    assert result["data_source"] == "reef_monitor"
    assert result["cache_used"] is False
    assert set(result["trend_keys"]) == set(_TREND_KEYS)
    assert result["report"] == live_report
    assert _hermes_result_text(result) == live_report


@pytest.mark.asyncio
async def test_ask_hermes_timeout_returns_stale_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Hermes timeout still returns the cached Reef report, marked stale."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill)."
    _write_reef_thread(thread, summary=cached)
    posts: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append("sent")
        raise HermesTimeoutError("timed out")

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What did Hermes report about my Reef?")

    assert posts == ["sent"]
    assert result["status"] != "success"
    assert result["stale"] is True
    assert result["fresh"] is False
    assert result["latest"] is False
    assert result["source"] == "cache"
    assert result["report"] == cached
    assert result["report_timestamp"]
    assert result["retrieved_at"]
    assert result["cache_used"] is True
    assert set(result["trend_keys"]) == set(_TREND_KEYS)
    spoken = _hermes_result_text(result)
    assert cached in spoken
    assert "cached" in spoken.lower()
    assert "latest" not in spoken.lower()


@pytest.mark.asyncio
async def test_ask_hermes_error_returns_stale_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Hermes error still returns the cached Reef report, marked stale."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill)."
    _write_reef_thread(thread, summary=cached)

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesRequestError("Hermes Gateway returned HTTP 503.")

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="Show me the latest Reef report.")

    assert result["status"] != "success"
    assert result["stale"] is True
    assert result["source"] == "cache"
    assert result["report"] == cached
    assert result["cache_used"] is True
    assert cached in _hermes_result_text(result)


@pytest.mark.asyncio
async def test_ask_hermes_error_with_old_cache_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old cached report is still returned, with age and status=stale."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Old Hermes Reef report: temp 23.8C."
    _write_reef_thread(thread, generated_at=_iso(timedelta(hours=5)), summary=cached)
    monkeypatch.setattr(hermes_client.config, "REEF_CACHE_MAX_AGE_SECONDS", 3600)

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesRequestError("Hermes Gateway returned HTTP 503.")

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What was the latest Reef report?")

    assert result["status"] == "stale"
    assert result["stale"] is True
    assert result["fresh"] is False
    assert result["latest"] is False
    assert result["source"] == "cache"
    assert result["report"] == cached
    assert result["cache_age_seconds"] is not None
    assert result["cache_age_seconds"] > 3600
    spoken = _hermes_result_text(result)
    assert cached in spoken
    assert "latest" not in spoken.lower()


@pytest.mark.asyncio
async def test_ask_hermes_error_without_cache_is_not_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermes failure with no cache is an error, never success."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesRequestError("Hermes Gateway returned HTTP 404.")

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="What did the Reef monitor last report?")

    assert result["status"] == "error"
    assert result["stale"] is True
    assert result["source"] == "none"
    assert result["report"] is None
    assert result["spoken"] == HISTORY_UNAVAILABLE


@pytest.mark.asyncio
async def test_ask_hermes_rejects_process_narration_without_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """A file/process error is not spoken as a trend when the Reefy cache is missing."""

    async def _send(text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        assert "Reef current request" in text
        assert request_id
        return "It seems there is an issue with accessing the file content."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] == "error"
    assert result["reason"] == "empty_or_narration"
    assert result["source"] == "none"
    assert result["spoken"] == HISTORY_UNAVAILABLE
    assert result["cache_used"] is False


@pytest.mark.asyncio
async def test_ask_hermes_process_narration_falls_back_to_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process narration is rejected and the cached Reef report is returned stale."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill)."
    _write_reef_thread(thread, summary=cached)

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        return "It seems there is an issue with accessing the file content."

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] != "success"
    assert result["stale"] is True
    assert result["source"] == "cache"
    assert result["report"] == cached


@pytest.mark.asyncio
async def test_ask_hermes_history_error_does_not_use_live_apex(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Hermes fails and the Reefy cache is missing, say history is unavailable."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesRequestError("Hermes Gateway returned HTTP 404.")

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] == "error"
    assert result["reason"] == "gateway_http_404"
    assert result["source"] == "none"
    assert result["spoken"] == HISTORY_UNAVAILABLE
    assert result["trend_available"] is False
    assert "try again" not in result["spoken"].lower()


@pytest.mark.asyncio
async def test_ask_hermes_gateway_404_without_cache_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway 404 without cache must not be replaced by live Apex numbers."""

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
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

    async def _send(text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        assert "Reef current request: how is my reef tank trending?" in text
        assert "Reef historical request:" not in text
        assert request_id
        return "Nitrate has been falling from 20 to 8 over the recorded period, while temperature stayed near 26."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="how is my reef tank trending?")

    assert result["status"] == "success"
    assert result["stale"] is False
    assert result["fresh"] is True
    assert result["trend_available"] is True
    assert result["source"] == "hermes"
    assert result["data_source"] == "reef_monitor"
    assert "Nitrate has been falling" in result["report"]
    assert "Nitrate has been falling" in _hermes_result_text(result)


@pytest.mark.asyncio
async def test_ask_hermes_pending_with_cache_returns_cached_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A second Reef request while Hermes is pending must use the cache, not an empty already_running."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef trends: Tmp -0.012/6h, pH +0.012/6h, ORP -0.471/6h, FS100 +0.472/6h, LLSATO -0.071/6h."
    _write_reef_thread(thread, summary=cached)
    posts: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append("sent")
        return "should not run"

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "hermes_is_busy", lambda: True)
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    with caplog.at_level("INFO"):
        result = await AskHermes()(_deps(), query=TREND_QUERY)

    assert posts == []
    assert result["status"] != "success"
    assert result["stale"] is True
    assert result["source"] == "cache"
    assert result["cache_used"] is True
    assert result["report"] == cached
    assert set(result["trend_keys"]) == set(_TREND_KEYS)
    assert set(result["trends"]) == set(_TREND_KEYS)
    spoken = _hermes_result_text(result)
    assert cached in spoken
    assert "still running" in spoken.lower()
    assert "previous Hermes request still running" in caplog.text
    assert "valid cache available" in caplog.text
    assert "returning cached Reef report rather than empty result" in caplog.text


@pytest.mark.asyncio
async def test_ask_hermes_second_request_uses_cache_while_first_is_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-flight live request continues; the second Reef request returns the cached report."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef trends: Tmp -0.012/6h, pH +0.012/6h, ORP falling, FS100 rising, LLSATO stable."
    _write_reef_thread(thread, summary=cached)
    started = asyncio.Event()
    release = asyncio.Event()
    posts: list[str] = []

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append("sent")
        async with hermes_client._HERMES_REQUEST_LOCK:
            started.set()
            await release.wait()
        return "Live Hermes Reef report: temp 24.1C, nitrate falling."

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    first = asyncio.create_task(AskHermes()(_deps(), query=TREND_QUERY))
    await started.wait()
    second = await AskHermes()(_deps(), query=TREND_QUERY)

    assert second["status"] != "success"
    assert second["stale"] is True
    assert second["source"] == "cache"
    assert second["cache_used"] is True
    assert second["report"] == cached
    assert set(second["trend_keys"]) == set(_TREND_KEYS)
    assert cached in _hermes_result_text(second)
    assert posts == ["sent"]

    release.set()
    first_result = await first
    assert first_result["status"] == "success"
    assert first_result["stale"] is False
    assert first_result["source"] == "hermes"
    assert first_result["cache_used"] is False
    assert "nitrate falling" in first_result["report"]
    assert set(first_result["trend_keys"]) == set(_TREND_KEYS)


@pytest.mark.asyncio
async def test_ask_hermes_hung_gateway_returns_cache_within_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A hung live Hermes Reef request must stop at the configured timeout and return cache."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Temp +0.007C/6h, 24.1C — CRITICAL LOW (24.5-26.5); ATO -0.078/6h."
    _write_reef_thread(thread, summary=cached)
    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client.config, "HERMES_GATEWAY_URL", "http://127.0.0.1:8642/v1/chat/completions")
    monkeypatch.setattr(hermes_client.config, "HERMES_API_KEY", "test-hermes-key")
    monkeypatch.setattr(hermes_client.config, "HERMES_REEF_REQUEST_TIMEOUT_SECONDS", 0.05)

    class HungAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "HungAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> object:
            del url, headers, json
            await asyncio.sleep(5)
            raise AssertionError("timed-out POST must not complete")

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", HungAsyncClient)

    started = time.monotonic()
    with caplog.at_level("INFO"):
        result = await AskHermes()(_deps(), query=TREND_QUERY)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    assert result["status"] != "success"
    assert result["stale"] is True
    assert result["source"] == "cache"
    assert result["cache_used"] is True
    assert result["report"] == cached
    assert set(result["trend_keys"]) == set(_TREND_KEYS)
    assert cached in _hermes_result_text(result)
    assert hermes_client.hermes_is_busy() is False
    assert hermes_client.hermes_in_flight_request_id() is None
    assert "Hermes timeout; returning validated cache" in caplog.text
    assert "request cleanup complete" in caplog.text


@pytest.mark.asyncio
async def test_ask_hermes_after_timeout_can_run_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out Reef request must not leave Hermes permanently busy."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef stable - temp 24.0C (-0.012/6h)."
    _write_reef_thread(thread, summary=cached)
    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client.config, "HERMES_GATEWAY_URL", "http://127.0.0.1:8642/v1/chat/completions")
    monkeypatch.setattr(hermes_client.config, "HERMES_API_KEY", "test-hermes-key")
    monkeypatch.setattr(hermes_client.config, "HERMES_REEF_REQUEST_TIMEOUT_SECONDS", 0.05)
    posts = 0

    class FirstHungThenLiveClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "FirstHungThenLiveClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> object:
            nonlocal posts
            posts += 1
            del url, headers, json
            if posts == 1:
                await asyncio.sleep(5)
                raise AssertionError("timed-out POST must not complete")
            request = httpx.Request("POST", "http://127.0.0.1:8642/v1/chat/completions")
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": "hermes-agent",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Live Reef report: temp 24.1C."},
                            "finish_reason": "stop",
                        }
                    ],
                },
                request=request,
            )

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", FirstHungThenLiveClient)

    first = await AskHermes()(_deps(), query=TREND_QUERY)
    assert first["source"] == "cache"
    assert first["cache_used"] is True
    assert hermes_client.hermes_is_busy() is False

    second = await AskHermes()(_deps(), query=TREND_QUERY)
    assert second["status"] == "success"
    assert second["stale"] is False
    assert second["source"] == "hermes"
    assert second["live"] is True
    assert second["degraded"] is False
    assert second["cache_used"] is False
    assert second["report"] == "Live Reef report: temp 24.1C."
    assert posts == 2


@pytest.mark.asyncio
async def test_ask_hermes_circuit_open_uses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open Hermes circuit still returns a validated cache, never as live."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill)."
    _write_reef_thread(thread, summary=cached)

    async def _send(_text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        raise HermesCircuitOpenError("Hermes Gateway circuit is open.")

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query=TREND_QUERY)

    assert result["status"] != "success"
    assert result["source"] == "cache"
    assert result["live"] is False
    assert result["degraded"] is True
    assert result["stale"] is True
    assert result["cache_used"] is True
    assert result["failure_category"] == "HERMES_CIRCUIT_OPEN"
    assert "cached" in result["spoken"].lower()


@pytest.mark.asyncio
async def test_ask_hermes_http_200_with_old_live_cache_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermes HTTP success with old underlying data must not be labelled fresh/latest."""
    posts: list[str] = []

    async def _send(text: str, _session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append(text)
        assert "Reef current request:" in text
        assert request_id
        return "Temperature 24.5C +0.034/6h; ATO +0.718/6h."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _stale_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query=TREND_QUERY)

    assert posts, "Hermes must still be called"
    assert result["status"] == "stale"
    assert result["fresh"] is False
    assert result["latest"] is False
    assert result["stale"] is True
    assert result["live"] is False
    assert result["source"] == "hermes"
    assert result["cache_used"] is False
    assert result["data_source"] == "reef_monitor"
    assert result["request_kind"] == "reef_current"
    assert result["age_seconds"] == 1201.0
    assert result["data_timestamp"]
    assert result["report_timestamp"]
    assert result["requested_at"]
    assert result["retrieved_at"]
    spoken = _hermes_result_text(result)
    assert "not current" in spoken.lower()
    assert "Temperature 24.5C" in spoken


@pytest.mark.asyncio
async def test_ask_hermes_historical_query_uses_history_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit historical reef questions stay on reef_history and reef_thread data."""
    thread = tmp_path / "reef_thread.jsonl"
    cached = "Reef history: temp rose 0.2C over 6h."
    _write_reef_thread(thread, summary=cached)
    posts: list[str] = []
    sessions: list[str] = []

    async def _send(text: str, session_id: str, request_id: str | None = None, **_kwargs: object) -> str:
        posts.append(text)
        sessions.append(session_id)
        assert "Reef historical request:" in text
        assert "Reef current request:" not in text
        assert cached in text
        return cached

    monkeypatch.setattr(hermes_client, "REEF_THREAD_PATH", str(thread))
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="How has my reef tank changed over the last 6 hours?")

    assert posts
    assert result["status"] == "success"
    assert result["fresh"] is False
    assert result["latest"] is False
    assert result["request_kind"] == "reef_history"
    assert result["data_source"] == "reef_thread_cache"
    assert result["report"] == cached
    assert sessions[0]


@pytest.mark.asyncio
async def test_ask_hermes_trending_does_not_use_history_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current trending question must use reef_current, not history=True."""
    kinds: list[str] = []

    async def _send(text: str, _session_id: str, request_id: str | None = None, **kwargs: object) -> str:
        kinds.append(str(kwargs.get("request_kind")))
        assert "Reef current request:" in text
        assert "Reef historical request:" not in text
        assert request_id
        return "Temperature 24.5C +0.034/6h."

    monkeypatch.setattr(hermes_client, "load_latest_reef_thread", lambda path=None: None)
    monkeypatch.setattr(hermes_client, "load_reef_live_cache", lambda path=None: _fresh_live_cache())
    monkeypatch.setattr(hermes_client, "send_to_hermes", _send)

    result = await AskHermes()(_deps(), query="Wally, can you tell me where my reef tank is trending at?")

    assert kinds == ["reef_current"]
    assert result["request_kind"] == "reef_current"
    assert result["fresh"] is True
    assert result["status"] == "success"
    assert result["data_source"] == "reef_monitor"
