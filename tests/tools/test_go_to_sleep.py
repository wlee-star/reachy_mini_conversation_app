from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep, match_sleep_intent


def test_go_to_sleep_has_no_required_arguments() -> None:
    """The tool should be callable without a confirmation argument."""
    assert GoToSleep.parameters_schema == {
        "type": "object",
        "properties": {},
        "required": [],
    }


@pytest.mark.asyncio
async def test_go_to_sleep_returns_unavailable_without_runtime_callback() -> None:
    """The tool should fail gracefully if the runtime did not inject a sleep callback."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    result = await GoToSleep()(deps)

    assert result == {"error": "go_to_sleep is unavailable in this runtime"}


@pytest.mark.asyncio
async def test_go_to_sleep_calls_runtime_callback() -> None:
    """The tool should delegate the actual movement and app stop to the host runtime."""
    expected = {
        "status": "sleeping",
        "stop_current_app_requested": True,
        "local_stop_requested": True,
    }
    go_to_sleep = MagicMock(return_value=expected)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=go_to_sleep,
    )

    result = await GoToSleep()(deps)

    assert result == expected
    go_to_sleep.assert_called_once_with()


@pytest.mark.parametrize(
    "transcript",
    [
        "go to sleep",
        "Go to sleep.",
        "please go to sleep",
        "time to sleep",
        "go to bed",
        "sleep now",
        "put yourself to sleep",
    ],
)
def test_match_sleep_intent_accepts_direct_commands(transcript: str) -> None:
    """Imperative robot sleep phrases must match after wake-word stripping."""
    assert match_sleep_intent(transcript) is True


@pytest.mark.parametrize(
    "transcript",
    [
        "Why do people go to sleep?",
        "What time should I go to sleep?",
        "Tell me a story about going to sleep.",
        "I couldn't go to sleep last night.",
        "I can't go to sleep",
        "Should I go to sleep?",
        "How long do people go to sleep?",
        "what's the weather",
        "hello",
        "I watched a sleep video",
        "sleeping is important",
    ],
)
def test_match_sleep_intent_rejects_non_commands(transcript: str) -> None:
    """Conversational or informational sleep mentions must not trigger the tool."""
    assert match_sleep_intent(transcript) is False
