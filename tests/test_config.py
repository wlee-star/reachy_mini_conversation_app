"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


def test_assistant_and_robot_names_default_to_wally_and_reachy_mini() -> None:
    """Assistant identity is Wally; the robot display name stays Reachy Mini."""
    assert config.config.ASSISTANT_NAME == "Wally"
    assert config.config.WAKE_NAME == "Wally"
    assert config.config.ROBOT_NAME == "Reachy Mini"
    assert config.config.ACTIVE_SESSION_TIMEOUT_SECONDS == 30
    assert config.config.LOCAL_TIMEZONE == "Australia/Sydney"
