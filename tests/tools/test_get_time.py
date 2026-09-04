"""Tests for the local get_time tool."""

from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app import local_time
from reachy_mini_conversation_app.local_time import SYDNEY_TIMEZONE, reset_startup_time_context
from reachy_mini_conversation_app.tools.get_time import GetTime
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.fixture(autouse=True)
def _reset_time_context() -> None:
    reset_startup_time_context()
    yield
    reset_startup_time_context()


@pytest.mark.asyncio
async def test_get_time_returns_system_clock_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool returns structured system-clock time, not an LLM guess."""
    moment = datetime(2026, 9, 4, 8, 52, tzinfo=ZoneInfo(SYDNEY_TIMEZONE))
    monkeypatch.setattr(local_time, "read_local_moment", lambda at=None, timezone_name=None: moment)

    result = await GetTime()(_deps())
    assert result["status"] == "success"
    assert result["source"] == "system_clock"
    assert result["timezone"] == SYDNEY_TIMEZONE
    assert result["local_time"] == "8:52 AM"
    assert result["local_date"] == "2026-09-04"
    assert "8:52 AM" in result["spoken"]


@pytest.mark.asyncio
async def test_get_time_sydney_alias_uses_local_zone() -> None:
    """A Sydney request still uses the authoritative local timezone."""
    result = await GetTime()(_deps(), timezone="Sydney")
    assert result["timezone"] == SYDNEY_TIMEZONE
    assert result["source"] == "system_clock"
    assert result["status"] == "success"
