"""Wally activation, identity, and Reachy Mini technical naming."""

from reachy_mini_conversation_app.config import DEFAULT_ROBOT_NAME, config
from reachy_mini_conversation_app.prompts import get_session_instructions, assistant_identity_instructions
from reachy_mini_conversation_app.activation import (
    ActivationSession,
    split_wake_prefix,
    wake_reminder_text,
    strip_transcript_name_prefix,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def test_wally_wake_at_start_authorizes_and_strips_command() -> None:
    """A leading Wally is activation; the remainder is the command."""
    detected, remainder = split_wake_prefix("Wally, turn on lamp three.")
    assert detected is True
    assert remainder.lower() == "turn on lamp three."


def test_missing_wally_does_not_activate() -> None:
    """Side-effecting requests without Wally stay unauthorized."""
    session = ActivationSession(clock=lambda: 0.0)
    decision = session.evaluate("Turn on lamp three.")
    assert decision.authorized is False
    assert decision.wake_detected is False


def test_reachy_is_not_the_assistant_wake_name() -> None:
    """Reachy / Reachy Mini must not activate the assistant."""
    session = ActivationSession(clock=lambda: 0.0)
    for transcript in (
        "Reachy, turn on lamp three.",
        "Hey Reachy, dance.",
        "Reachy Mini, what's the reef temperature?",
        "Hey Reachy Mini, look left.",
    ):
        decision = session.evaluate(transcript)
        assert decision.authorized is False, transcript
        assert decision.wake_detected is False, transcript


def test_mid_sentence_wally_is_not_activation() -> None:
    """A mention of Wally that is not the utterance start must not activate."""
    session = ActivationSession(clock=lambda: 0.0)
    decision = session.evaluate("I was talking to Wally yesterday.")
    assert decision.authorized is False
    assert decision.wake_detected is False


def test_hey_wally_is_activation() -> None:
    """Optional vocatives before Wally still count as a wake."""
    detected, remainder = split_wake_prefix("Hey Wally, dance.")
    assert detected is True
    assert remainder.lower() == "dance."


def test_follow_up_allowed_until_timeout() -> None:
    """After Wally, follow-ups work until the session timeout."""
    now = {"t": 0.0}
    session = ActivationSession(clock=lambda: now["t"])
    first = session.evaluate("Wally, what's the reef temperature?")
    assert first.authorized is True
    follow = session.evaluate("What is the salinity?")
    assert follow.authorized is True
    assert follow.wake_detected is False
    now["t"] = 31.0
    expired = session.evaluate("What is the alkalinity?")
    assert expired.authorized is False


def test_wake_reminder_uses_configured_name() -> None:
    """Unactivated speech is reminded with the configured wake name."""
    assert wake_reminder_text() == "Please say Wally first."


def test_legacy_reachy_stt_prefix_still_strips_for_matchers() -> None:
    """Command matchers may still parse a leftover Reachy STT prefix; the gate does not activate on it."""
    assert strip_transcript_name_prefix("Rishi, turn on lamp three.").lower() == "turn on lamp three."
    assert strip_transcript_name_prefix("Wally, turn on lamp three.").lower() == "turn on lamp three."


def test_identity_prompt_says_wally_not_reachy() -> None:
    """The system prompt must identify Wally as the assistant and Reachy Mini as the robot."""
    identity = assistant_identity_instructions()
    assert "You are Wally, the conversational assistant." in identity
    assert "Never identify yourself as Reachy." in identity
    assert "When asked your name, say Wally." in identity
    assert DEFAULT_ROBOT_NAME in identity
    assert "Reachy Mini is the robot" in identity


def test_session_instructions_include_wally_identity(tmp_path) -> None:
    """Every backend session prompt carries the Wally identity block."""
    instructions = get_session_instructions(instance_path=tmp_path)
    assert instructions.startswith("You are Wally, the conversational assistant.")
    assert "Never identify yourself as Reachy." in instructions
    lowered = instructions.lower()
    assert "when asked your name, say wally." in lowered
    assert "when asked who you are, say you are wally" in lowered
    assert "when asked if you are reachy" in lowered


def test_reachy_mini_sdk_field_is_unchanged() -> None:
    """Tool dependencies still use the Reachy Mini robot object, not a Wally type."""
    assert "reachy_mini" in ToolDependencies.__dataclass_fields__
    assert "wally" not in ToolDependencies.__dataclass_fields__
    assert config.ROBOT_NAME == "Reachy Mini"
    assert config.ASSISTANT_NAME == "Wally"
    assert config.WAKE_NAME == "Wally"
