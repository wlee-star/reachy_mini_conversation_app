from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.bus_monitor import BusMonitorManager, reset_bus_monitor_for_tests
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.monitor_bus import MonitorBus


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.fixture(autouse=True)
def _manager() -> BusMonitorManager:
    manager = BusMonitorManager()
    reset_bus_monitor_for_tests(manager)
    return manager


@pytest.mark.asyncio
async def test_monitor_bus_query_returns_live_spoken_field(
    monkeypatch: pytest.MonkeyPatch, _manager: BusMonitorManager
) -> None:
    """Query reads live Home Assistant data through the isolated monitor."""

    async def _query(*, preparation_threshold: int = 15) -> dict[str, object]:
        return {"minutes": 22, "spoken": "The next 311 is currently about 22 minutes away.", "offer": "offer_prepare"}

    monkeypatch.setattr(_manager, "query", _query)
    result = await MonitorBus()(_deps(), action="query")
    assert result["minutes"] == 22
    assert result["offer"] == "offer_prepare"


@pytest.mark.asyncio
async def test_monitor_bus_rejects_unknown_action() -> None:
    """Unknown actions are tool errors, not exceptions into the loop."""
    result = await MonitorBus()(_deps(), action="explode")
    assert "error" in result


def test_already_spoken_skips_followup() -> None:
    """A fast-path announcement should not make the model speak the arrival twice."""
    tool = MonitorBus()
    assert tool.wants_spoken_followup({"already_spoken": True, "minutes": 8}, None) is False
    assert tool.wants_spoken_followup({"minutes": 8}, None) is True
