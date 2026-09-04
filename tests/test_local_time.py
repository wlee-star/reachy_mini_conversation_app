"""Tests for the deterministic local-time context and utility."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from reachy_mini_conversation_app.local_time import (
    SYDNEY_TIMEZONE,
    utc_offset_label,
    match_time_intent,
    current_local_time,
    local_timezone_name,
    get_startup_time_context,
    reset_startup_time_context,
    establish_startup_time_context,
)


@pytest.fixture(autouse=True)
def _reset_time_context() -> None:
    reset_startup_time_context()
    yield
    reset_startup_time_context()


def test_startup_context_uses_sydney_timezone() -> None:
    """Startup stores the configured Australia/Sydney timezone."""
    context = establish_startup_time_context()
    assert local_timezone_name() == SYDNEY_TIMEZONE
    assert context["timezone"] == SYDNEY_TIMEZONE
    assert context["startup_local_datetime"]
    assert context["startup_utc_offset"] in {"+10:00", "+11:00"}
    assert get_startup_time_context() == context


def test_startup_context_uses_provided_timestamp() -> None:
    """A known Sydney moment is stored without the LLM inventing a clock."""
    moment = datetime(2026, 9, 4, 8, 48, tzinfo=ZoneInfo(SYDNEY_TIMEZONE))
    context = establish_startup_time_context(moment)
    assert context["startup_local_datetime"] == "2026-09-04T08:48:00+10:00"
    assert context["startup_utc_offset"] == "+10:00"


def test_current_time_uses_system_clock_not_startup() -> None:
    """A later current-time read re-reads the clock instead of freezing startup."""
    startup = datetime(2026, 9, 4, 8, 48, tzinfo=ZoneInfo(SYDNEY_TIMEZONE))
    later = datetime(2026, 9, 4, 8, 52, tzinfo=ZoneInfo(SYDNEY_TIMEZONE))
    establish_startup_time_context(startup)
    result = current_local_time(at=later)
    assert result["status"] == "success"
    assert result["source"] == "system_clock"
    assert result["timezone"] == SYDNEY_TIMEZONE
    assert result["local_time"] == "8:52 AM"
    assert result["local_date"] == "2026-09-04"
    assert result["utc_offset"] == "+10:00"
    assert result["spoken"] == "The current time in Sydney is 8:52 AM."
    assert get_startup_time_context() is not None
    assert get_startup_time_context()["startup_local_datetime"] == "2026-09-04T08:48:00+10:00"


def test_sydney_dst_offsets() -> None:
    """Australia/Sydney follows DST rather than a fixed UTC offset."""
    utc = timezone.utc
    summer = datetime(2026, 1, 15, 0, 0, tzinfo=utc).astimezone(ZoneInfo(SYDNEY_TIMEZONE))
    winter = datetime(2026, 7, 15, 0, 0, tzinfo=utc).astimezone(ZoneInfo(SYDNEY_TIMEZONE))
    assert utc_offset_label(summer) == "+11:00"
    assert utc_offset_label(winter) == "+10:00"
    assert current_local_time(at=summer)["utc_offset"] == "+11:00"
    assert current_local_time(at=winter)["utc_offset"] == "+10:00"


def test_time_intent_matches_common_questions() -> None:
    """Spoken time and date questions are routed to the time utility."""
    assert match_time_intent("What time is it?")
    assert match_time_intent("What's the time?")
    assert match_time_intent("What time is it in Sydney?")
    assert match_time_intent("What's today's date?")
    assert match_time_intent("What date is it?")
    assert match_time_intent("What day is it?")
    assert match_time_intent("What time is it now?")
    assert match_time_intent("when is the 311") is False
    assert match_time_intent("what time is the 311") is False


def test_current_time_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The value supplied for speech is logged from the system clock."""
    moment = datetime(2026, 9, 4, 8, 52, tzinfo=ZoneInfo(SYDNEY_TIMEZONE))
    with caplog.at_level("INFO"):
        result = current_local_time(at=moment)
    assert result["local_time"] == "8:52 AM"
    assert "[TIME] current local_datetime=" in caplog.text
    assert "[TIME] source=system_clock" in caplog.text
    assert "8:52" in caplog.text
