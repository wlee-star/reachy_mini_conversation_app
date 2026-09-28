"""Reachy activation and identity."""

from reachy_mini_conversation_app.config import DEFAULT_ROBOT_NAME, config
from reachy_mini_conversation_app.prompts import get_session_instructions, assistant_identity_instructions
from reachy_mini_conversation_app.activation import (
    WAKE_GATE_MENTION,
    WAKE_GATE_FOLLOWUP,
    MENTION_PROMPT_TEXT,
    WAKE_GATE_DUPLICATE,
    WAKE_GATE_DIRECT_WAKE,
    WAKE_GATE_UNADDRESSED,
    WAKE_GATE_MENTION_DECLINED,
    WAKE_GATE_MENTION_CONFIRMATION,
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
    assert decision.kind == WAKE_GATE_UNADDRESSED


def test_background_speech_is_silently_unauthorized() -> None:
    """Ordinary background phrases must not authorize tools or open follow-up."""
    session = ActivationSession(clock=lambda: 0.0)
    for transcript in (
        "Welcome back to my YouTube channel.",
        "What are you doing tomorrow?",
        "Can you call me later?",
        "The weather looks pretty good.",
        "I think I'll buy that one.",
        "What time are you coming home?",
        "go to sleep",
    ):
        decision = session.evaluate(transcript)
        assert decision.authorized is False, transcript
        assert decision.kind == WAKE_GATE_UNADDRESSED, transcript
        assert decision.speak_text is None, transcript
        assert session.is_active() is False


def test_reachy_and_reachy_mini_activate_the_assistant() -> None:
    """Reachy and Reachy Mini both activate the assistant."""
    session = ActivationSession(clock=lambda: 0.0)
    for transcript in (
        "Reachy, turn on lamp three.",
        "Hey Reachy, dance.",
        "Reachy Mini, what's the reef temperature?",
        "Hey Reachy Mini, look left.",
        "Hi Reachy, what time is it?",
        "Okay Reachy, what's the next bus?",
        "Um Reachy, could you turn the light on?",
    ):
        decision = session.evaluate(transcript)
        assert decision.authorized is True, transcript
        assert decision.wake_detected is True, transcript
        assert decision.kind == WAKE_GATE_DIRECT_WAKE, transcript


def test_mid_sentence_reachy_is_not_a_command() -> None:
    """A third-person Reachy mention must not authorize tools as a direct wake."""
    session = ActivationSession(clock=lambda: 0.0)
    decision = session.evaluate("I was showing Mum what Reachy can do.")
    assert decision.authorized is False
    assert decision.wake_detected is False
    assert decision.kind == WAKE_GATE_MENTION
    assert decision.speak_text == MENTION_PROMPT_TEXT


def test_direct_wake_still_works_when_mentioning_capability() -> None:
    """Leading Reachy remains a direct invocation even when describing capability."""
    session = ActivationSession(clock=lambda: 0.0)
    decision = session.evaluate("Reachy, show Mum what you can do.")
    assert decision.authorized is True
    assert decision.kind == WAKE_GATE_DIRECT_WAKE
    assert decision.command_text.lower() == "show mum what you can do."


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
    assert follow.kind == WAKE_GATE_FOLLOWUP
    now["t"] = 31.0
    expired = session.evaluate("What is the alkalinity?")
    assert expired.authorized is False
    assert expired.kind == WAKE_GATE_UNADDRESSED


def test_ha_style_follow_up_without_wake_name() -> None:
    """A device follow-up after a valid wake stays authorized."""
    session = ActivationSession(clock=lambda: 0.0)
    first = session.evaluate("Reachy, turn on the bedroom light.")
    assert first.authorized is True
    second = session.evaluate("Set the bedroom light to 1%.")
    assert second.authorized is True
    assert second.kind == WAKE_GATE_FOLLOWUP


def test_bus_style_follow_up_without_wake_name() -> None:
    """A bus follow-up after a valid wake stays authorized."""
    session = ActivationSession(clock=lambda: 0.0)
    first = session.evaluate("Reachy, what's the next 311 bus?")
    assert first.authorized is True
    second = session.evaluate("And the following bus?")
    assert second.authorized is True
    assert second.kind == WAKE_GATE_FOLLOWUP


def test_expired_follow_up_rejects_unrelated_command_silently() -> None:
    """After expiry, an imperative without Reachy stays silent and unauthorized."""
    now = {"t": 0.0}
    session = ActivationSession(clock=lambda: now["t"])
    session.evaluate("Reachy, turn on the bedroom light.")
    now["t"] = 31.0
    later = session.evaluate("Turn off the light.")
    assert later.authorized is False
    assert later.kind == WAKE_GATE_UNADDRESSED
    assert later.speak_text is None


def test_mention_confirmation_opens_follow_up() -> None:
    """A positive answer after 'Did you ask for me?' opens conversational context."""
    now = {"t": 0.0}
    session = ActivationSession(clock=lambda: now["t"])
    mention = session.evaluate("I was actually talking about Reachy.")
    assert mention.kind == WAKE_GATE_MENTION
    yes = session.evaluate("Yes.")
    assert yes.authorized is True
    assert yes.kind == WAKE_GATE_MENTION_CONFIRMATION
    assert yes.command_text == ""
    follow = session.evaluate("Turn on the bedroom light.")
    assert follow.authorized is True
    assert follow.kind == WAKE_GATE_FOLLOWUP


def test_mention_decline_closes_session() -> None:
    """A negative mention confirmation must not authorize later background speech."""
    session = ActivationSession(clock=lambda: 0.0)
    mention = session.evaluate("I was testing Reachy earlier.")
    assert mention.kind == WAKE_GATE_MENTION
    no = session.evaluate("No, I was talking to someone else.")
    assert no.authorized is False
    assert no.kind == WAKE_GATE_MENTION_DECLINED
    later = session.evaluate("Turn off the light.")
    assert later.authorized is False
    assert later.kind == WAKE_GATE_UNADDRESSED


def test_mention_cooldown_and_duplicate_protection() -> None:
    """Repeated identical Reachy mentions must not keep prompting."""
    now = {"t": 0.0}
    session = ActivationSession(clock=lambda: now["t"])
    first = session.evaluate("I was showing John what Reachy can do.")
    assert first.kind == WAKE_GATE_MENTION
    duplicate = session.evaluate("I was showing John what Reachy can do.")
    assert duplicate.kind == WAKE_GATE_DUPLICATE
    now["t"] = 10.0
    cooled = session.evaluate("I told Mum that Reachy controls my lights.")
    assert cooled.kind == WAKE_GATE_UNADDRESSED
    assert cooled.speak_text is None
    now["t"] = 70.0
    again = session.evaluate("I told Mum that Reachy controls my lights.")
    assert again.kind == WAKE_GATE_MENTION


def test_clear_expires_follow_up() -> None:
    """Sleep/stop clearing must close the follow-up window."""
    session = ActivationSession(clock=lambda: 0.0)
    session.evaluate("Reachy, hello")
    assert session.is_active() is True
    session.clear(reason="sleep")
    assert session.is_active() is False
    later = session.evaluate("Turn off the light.")
    assert later.authorized is False


def test_wake_reminder_uses_configured_name() -> None:
    """Legacy reminder text still uses the configured wake name when referenced."""
    assert wake_reminder_text() == "Please say Reachy first."


def test_reachy_stt_variants_strip_for_matchers() -> None:
    """Command matchers parse common Reachy STT variants."""
    assert strip_transcript_name_prefix("Rishi, turn on lamp three.").lower() == "turn on lamp three."
    assert strip_transcript_name_prefix("Reachy, turn on lamp three.").lower() == "turn on lamp three."
    assert strip_transcript_name_prefix("Reachie, hello").lower() == "hello"
    assert strip_transcript_name_prefix("Rachie, what time is it?").lower() == "what time is it?"
    assert strip_transcript_name_prefix("Reach it, hello").lower() == "hello"
    assert strip_transcript_name_prefix("Reaching.").lower() == ""


def test_common_reachy_stt_mishears_activate() -> None:
    """Frequent STT mishears of Reachy must still open the session."""
    session = ActivationSession(clock=lambda: 0.0)
    for transcript in (
        "Reachy, what time is it?",
        "Reachie, what time is it?",
        "Rachie, what time is it?",
        "Reach it.",
        "Reaching.",
        "Harichi, can you hear me?",
    ):
        decision = session.evaluate(transcript)
        assert decision.authorized is True, transcript
        assert decision.wake_detected is True, transcript


def test_wake_aliases_do_not_enable_fuzzy_matching() -> None:
    """Similar unrelated leading words and mid-sentence aliases stay unauthorized."""
    for transcript in (
        "Rachel, what time is it?",
        "Archie, what time is it?",
        "Actually, what time is it?",
        "I heard Rachie ask what time it is.",
    ):
        decision = ActivationSession(clock=lambda: 0.0).evaluate(transcript)
        assert decision.authorized is False, transcript
        assert decision.wake_detected is False, transcript
        assert decision.kind == WAKE_GATE_UNADDRESSED, transcript


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
