"""Reachy activation and identity."""

from reachy_mini_conversation_app.config import DEFAULT_ROBOT_NAME, config
from reachy_mini_conversation_app.prompts import get_session_instructions, assistant_identity_instructions
from reachy_mini_conversation_app.activation import (
    ActivationSession,
    split_wake_prefix,
    wake_reminder_text,
    strip_transcript_name_prefix,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def test_reachy_wake_at_start_authorizes_and_strips_command() -> None:
    """A leading Reachy is activation; the remainder is the command."""
    detected, remainder = split_wake_prefix("Reachy, turn on lamp three.")
    assert detected is True
    assert remainder.lower() == "turn on lamp three."


def test_missing_reachy_does_not_activate() -> None:
    """Side-effecting requests without Reachy stay unauthorized."""
    session = ActivationSession(clock=lambda: 0.0)
    decision = session.evaluate("Turn on lamp three.")
    assert decision.authorized is False
    assert decision.wake_detected is False


def test_reachy_and_reachy_mini_activate_the_assistant() -> None:
    """Reachy and Reachy Mini both activate the assistant."""
    session = ActivationSession(clock=lambda: 0.0)
    for transcript in (
        "Reachy, turn on lamp three.",
        "Hey Reachy, dance.",
        "Reachy Mini, what's the reef temperature?",
        "Hey Reachy Mini, look left.",
    ):
        decision = session.evaluate(transcript)
        assert decision.authorized is True, transcript
        assert decision.wake_detected is True, transcript


def test_mid_sentence_reachy_is_not_activation() -> None:
    """A mention of Reachy that is not the utterance start must not activate."""
    session = ActivationSession(clock=lambda: 0.0)
    decision = session.evaluate("I was talking to Reachy yesterday.")
    assert decision.authorized is False
    assert decision.wake_detected is False


def test_hey_reachy_is_activation() -> None:
    """Optional vocatives before Reachy still count as a wake."""
    detected, remainder = split_wake_prefix("Hey Reachy, dance.")
    assert detected is True
    assert remainder.lower() == "dance."


def test_follow_up_allowed_until_timeout() -> None:
    """After Reachy, follow-ups work until the session timeout."""
    now = {"t": 0.0}
    session = ActivationSession(clock=lambda: now["t"])
    first = session.evaluate("Reachy, what's the reef temperature?")
    assert first.authorized is True
    follow = session.evaluate("What is the salinity?")
    assert follow.authorized is True
    assert follow.wake_detected is False
    now["t"] = 31.0
    expired = session.evaluate("What is the alkalinity?")
    assert expired.authorized is False


def test_wake_reminder_uses_configured_name() -> None:
    """Unactivated speech is reminded with the configured wake name."""
    assert wake_reminder_text() == "Please say Reachy first."


def test_reachy_stt_variants_strip_for_matchers() -> None:
    """Command matchers parse common Reachy STT variants."""
    assert strip_transcript_name_prefix("Rishi, turn on lamp three.").lower() == "turn on lamp three."
    assert strip_transcript_name_prefix("Reachy, turn on lamp three.").lower() == "turn on lamp three."
    assert strip_transcript_name_prefix("Reachie, hello").lower() == "hello"
    assert strip_transcript_name_prefix("Reach it, hello").lower() == "hello"
    assert strip_transcript_name_prefix("Reaching.").lower() == ""


def test_common_reachy_stt_mishears_activate() -> None:
    """Frequent STT mishears of Reachy must still open the session."""
    session = ActivationSession(clock=lambda: 0.0)
    for transcript in ("Reach it.", "Reaching.", "Reachie, hello", "Harichi, can you hear me?"):
        decision = session.evaluate(transcript)
        assert decision.authorized is True, transcript
        assert decision.wake_detected is True, transcript


def test_identity_prompt_says_reachy_mini() -> None:
    """The system prompt must identify the assistant as Reachy Mini."""
    identity = assistant_identity_instructions()
    assert "You are Reachy Mini, a friendly conversational robot assistant." in identity
    assert "When asked your name, say Reachy Mini." in identity
    assert DEFAULT_ROBOT_NAME in identity
    assert "Reachy Mini is the robot" in identity


def test_session_instructions_include_reachy_identity(tmp_path) -> None:
    """Every backend session prompt carries the Reachy identity block."""
    instructions = get_session_instructions(instance_path=tmp_path)
    assert instructions.startswith("You are Reachy Mini, a friendly conversational robot assistant.")
    lowered = instructions.lower()
    assert "when asked your name, say reachy mini." in lowered
    assert "when asked who you are, say you are reachy mini" in lowered


def test_reachy_mini_sdk_field_is_unchanged() -> None:
    """Tool dependencies continue to use the official Reachy Mini robot object."""
    assert "reachy_mini" in ToolDependencies.__dataclass_fields__
    assert config.ROBOT_NAME == "Reachy Mini"
    assert config.ASSISTANT_NAME == "Reachy Mini"
    assert config.WAKE_NAME == "Reachy"
