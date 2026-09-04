"""Application-level Wally wake/activation gate.

Wally is the conversational assistant. Reachy Mini is the robot. This module
authorizes user requests before tools or other side effects run. The LLM prompt
is not the security boundary.
"""

import re
import time
import logging
from dataclasses import dataclass
from collections.abc import Callable

from reachy_mini_conversation_app.config import (
    DEFAULT_WAKE_NAME,
    config,
)


logger = logging.getLogger(__name__)

_VOCATIVE = r"(?:hey |hi |hello |ok |okay |yo )?"
_WALLY_STT_VARIANTS = ("wally", "wall-e", "walle", "wali", "wolly")
# Legacy Reachy Mini STT mishears, stripped only so command matchers still parse.
# They must never activate the assistant.
_LEGACY_ROBOT_STT_ALT = r"reachy|erichi|richie|rishi|ricci|ritchie|i reach a|i reachy|reach it"


@dataclass(frozen=True)
class ActivationDecision:
    """Result of checking one user utterance against Wally activation rules."""

    authorized: bool
    wake_detected: bool
    command_text: str
    session_active: bool


def configured_wake_names() -> tuple[str, ...]:
    """Return lowercase wake tokens, including Wally STT variants when applicable."""
    wake = (config.WAKE_NAME or DEFAULT_WAKE_NAME).strip().lower()
    names = [wake] if wake else [DEFAULT_WAKE_NAME.lower()]
    if names[0] == "wally":
        names.extend(_WALLY_STT_VARIANTS)
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return tuple(unique)


def _wake_alt_pattern() -> str:
    return "|".join(re.escape(name) for name in configured_wake_names())


def assistant_wake_prefix_re() -> re.Pattern[str]:
    """Match Wally (or the configured wake name) only at the start of an utterance."""
    return re.compile(rf"^{_VOCATIVE}(?:{_wake_alt_pattern()})\b[\s,.\-:]*", re.IGNORECASE)


def transcript_name_prefix_re() -> re.Pattern[str]:
    """Strip an opening assistant or legacy robot STT name so command matchers can run."""
    return re.compile(
        rf"^{_VOCATIVE}(?:{_wake_alt_pattern()}|{_LEGACY_ROBOT_STT_ALT})[\s,.\-:]+",
        re.IGNORECASE,
    )


def split_wake_prefix(transcript: str) -> tuple[bool, str]:
    """Return whether Wally starts the utterance, and the remainder after that prefix."""
    text = transcript.strip()
    if not text:
        return False, ""
    match = assistant_wake_prefix_re().match(text)
    if match is None:
        return False, text
    remainder = text[match.end() :].strip()
    return True, remainder


def strip_transcript_name_prefix(transcript: str) -> str:
    """Remove a leading assistant/legacy STT name so fast-path matchers see the command."""
    text = transcript.strip()
    if not text:
        return ""
    return transcript_name_prefix_re().sub("", text, count=1).strip()


def wake_reminder_text() -> str:
    """Return the short spoken reminder for unactivated speech."""
    wake = (config.WAKE_NAME or DEFAULT_WAKE_NAME).strip() or DEFAULT_WAKE_NAME
    return f"Please say {wake} first."


class ActivationSession:
    """Authorize Wally at utterance start, then accept follow-ups until timeout."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Start an inactive session using a monotonic clock."""
        self._clock = clock
        self._authorized_until = 0.0

    def is_active(self, now: float | None = None) -> bool:
        """Return whether a Wally session is still open."""
        moment = self._clock() if now is None else now
        return moment < self._authorized_until

    def evaluate(self, transcript: str, *, now: float | None = None) -> ActivationDecision:
        """Authorize a user utterance, refreshing the follow-up window when allowed."""
        moment = self._clock() if now is None else now
        timeout = float(config.ACTIVE_SESSION_TIMEOUT_SECONDS)
        wake_detected, remainder = split_wake_prefix(transcript)
        session_was_active = moment < self._authorized_until
        if wake_detected:
            self._authorized_until = moment + timeout
            command_text = remainder
            logger.info("Wally activated")
            return ActivationDecision(
                authorized=True,
                wake_detected=True,
                command_text=command_text,
                session_active=True,
            )
        if session_was_active:
            self._authorized_until = moment + timeout
            return ActivationDecision(
                authorized=True,
                wake_detected=False,
                command_text=transcript.strip(),
                session_active=True,
            )
        logger.info("Wally activation required; ignoring unactivated user request")
        return ActivationDecision(
            authorized=False,
            wake_detected=False,
            command_text=transcript.strip(),
            session_active=False,
        )
