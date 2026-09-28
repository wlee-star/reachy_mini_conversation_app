"""Application-level Reachy wake/activation gate.

This module authorizes user requests before tools or other side effects run.
The LLM prompt is not the security boundary.

Unaddressed background speech is silently ignored. Direct wake names and an
active follow-up window authorize requests. Mid-utterance Reachy mentions may
conservatively prompt "Did you ask for me?" without opening tools.
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

# Optional openers before a direct wake ("Hey Reachy", "Um Reachy", ...).
_VOCATIVE = r"(?:(?:hey|hi|hello|ok|okay|yo|um|uh|oh|erm|well|so)\s+)*"
_REACHY_STT_VARIANTS = (
    "reachy mini",
    "reachy",
    "reachie",
    "rachie",
    "reach it",
    "reaching",
    "erichi",
    "harichi",
    "richie",
    "rishi",
    "ricci",
    "ritchie",
)
# Mid-utterance mentions use clearer name forms only; noisy STT variants stay wake-prefix-only.
_MENTION_NAME_VARIANTS = ("reachy mini", "reachy", "reachie")

MENTION_PROMPT_TEXT = "Did you ask for me?"
DEFAULT_MENTION_CONFIRM_TIMEOUT_SECONDS = 20
DEFAULT_MENTION_COOLDOWN_SECONDS = 60
DEFAULT_DUPLICATE_TRANSCRIPT_WINDOW_SECONDS = 8

WAKE_GATE_DIRECT_WAKE = "direct_wake"
WAKE_GATE_FOLLOWUP = "followup"
WAKE_GATE_MENTION = "mention"
WAKE_GATE_MENTION_CONFIRMATION = "mention_confirmation"
WAKE_GATE_MENTION_DECLINED = "mention_declined"
WAKE_GATE_UNADDRESSED = "unaddressed"
WAKE_GATE_DUPLICATE = "duplicate"

_AFFIRM_RE = re.compile(
    r"^(?:yes|yeah|yep|yup|sure|ok|okay|correct|i did)(?:\b|[\s,.!?]|$)",
    re.IGNORECASE,
)
_DECLINE_RE = re.compile(
    r"^(?:no|nope|nah|not really|no thanks)(?:\b|[\s,.!?]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ActivationDecision:
    """Result of checking one user utterance against Reachy activation rules."""

    authorized: bool
    wake_detected: bool
    command_text: str
    session_active: bool
    kind: str = WAKE_GATE_UNADDRESSED
    speak_text: str | None = None


def configured_wake_names() -> tuple[str, ...]:
    """Return lowercase wake tokens, including Reachy STT variants when applicable."""
    wake = (config.WAKE_NAME or DEFAULT_WAKE_NAME).strip().lower()
    names = [wake] if wake else [DEFAULT_WAKE_NAME.lower()]
    if names[0] in {"reachy", "reachy mini"}:
        names.extend(_REACHY_STT_VARIANTS)
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return tuple(sorted(unique, key=len, reverse=True))


def _wake_alt_pattern() -> str:
    return "|".join(re.escape(name) for name in configured_wake_names())


def _mention_alt_pattern() -> str:
    wake = (config.WAKE_NAME or DEFAULT_WAKE_NAME).strip().lower() or DEFAULT_WAKE_NAME.lower()
    names = [wake]
    if wake in {"reachy", "reachy mini"}:
        names.extend(_MENTION_NAME_VARIANTS)
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return "|".join(re.escape(name) for name in sorted(unique, key=len, reverse=True))


def assistant_wake_prefix_re() -> re.Pattern[str]:
    """Match Reachy (or the configured wake name) only at the start of an utterance."""
    return re.compile(rf"^{_VOCATIVE}(?:{_wake_alt_pattern()})\b[\s,.\-:]*", re.IGNORECASE)


def transcript_name_prefix_re() -> re.Pattern[str]:
    """Strip an opening assistant or legacy robot STT name so command matchers can run."""
    return re.compile(
        rf"^{_VOCATIVE}(?:{_wake_alt_pattern()})[\s,.\-:]+",
        re.IGNORECASE,
    )


def mention_name_re() -> re.Pattern[str]:
    """Match a whole-word Reachy mention anywhere in an utterance."""
    return re.compile(rf"\b(?:{_mention_alt_pattern()})\b", re.IGNORECASE)


def split_wake_prefix(transcript: str) -> tuple[bool, str]:
    """Return whether Reachy starts the utterance, and the remainder after that prefix."""
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
    """Return the legacy spoken reminder text (kept for compatibility; not used for ordinary rejection)."""
    wake = (config.WAKE_NAME or DEFAULT_WAKE_NAME).strip() or DEFAULT_WAKE_NAME
    return f"Please say {wake} first."


def mention_prompt_text() -> str:
    """Return the conservative spoken prompt for a third-person Reachy mention."""
    return MENTION_PROMPT_TEXT


def normalize_transcript_key(transcript: str) -> str:
    """Normalize a transcript for near-duplicate suppression."""
    lowered = transcript.strip().lower()
    collapsed = re.sub(r"\s+", " ", lowered)
    return re.sub(r"[^\w\s]", "", collapsed).strip()


def log_wake_gate(event: str, **fields: object) -> None:
    """Emit a structured wake-gate debug line without dumping unnecessary audio content."""
    parts = [f"wake_gate: {event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value!r}")
    logger.info(" ".join(parts))


class ActivationSession:
    """Authorize Reachy at utterance start, then accept follow-ups until timeout."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Start an inactive session using a monotonic clock."""
        self._clock = clock
        self._authorized_until = 0.0
        self._had_followup_session = False
        self._mention_pending_until = 0.0
        self._mention_cooldown_until = 0.0
        self._last_transcript_key = ""
        self._last_transcript_at = 0.0

    def is_active(self, now: float | None = None) -> bool:
        """Return whether a Reachy follow-up session is still open."""
        moment = self._clock() if now is None else now
        return moment < self._authorized_until

    def clear(self, *, reason: str = "cleared") -> None:
        """Expire follow-up and mention-confirmation state (sleep, stop, cancel)."""
        was_active = self._authorized_until > 0 or self._mention_pending_until > 0
        self._authorized_until = 0.0
        self._mention_pending_until = 0.0
        self._had_followup_session = False
        if was_active:
            log_wake_gate("FOLLOWUP_EXPIRED", reason=reason)

    def evaluate(self, transcript: str, *, now: float | None = None) -> ActivationDecision:
        """Authorize a user utterance, refreshing the follow-up window when allowed."""
        moment = self._clock() if now is None else now
        timeout = float(config.ACTIVE_SESSION_TIMEOUT_SECONDS)
        text = transcript.strip()
        key = normalize_transcript_key(text)

        if (
            key
            and key == self._last_transcript_key
            and (moment - self._last_transcript_at) < DEFAULT_DUPLICATE_TRANSCRIPT_WINDOW_SECONDS
        ):
            log_wake_gate("DUPLICATE_TRANSCRIPT_IGNORED")
            return ActivationDecision(
                authorized=False,
                wake_detected=False,
                command_text=text,
                session_active=moment < self._authorized_until,
                kind=WAKE_GATE_DUPLICATE,
            )

        wake_detected, remainder = split_wake_prefix(text)
        session_was_active = moment < self._authorized_until
        mention_pending = moment < self._mention_pending_until

        if self._had_followup_session and not session_was_active and not wake_detected:
            log_wake_gate("FOLLOWUP_EXPIRED")
            self._had_followup_session = False

        if wake_detected:
            self._open_followup(moment, timeout)
            self._mention_pending_until = 0.0
            self._remember_transcript(key, moment)
            log_wake_gate("DIRECT_WAKE_ACCEPTED")
            logger.info("Reachy activated")
            return ActivationDecision(
                authorized=True,
                wake_detected=True,
                command_text=remainder,
                session_active=True,
                kind=WAKE_GATE_DIRECT_WAKE,
            )

        if mention_pending:
            affirm_remainder = self._confirmation_remainder(text, affirmative=True)
            if affirm_remainder is not None:
                self._open_followup(moment, timeout)
                self._mention_pending_until = 0.0
                self._remember_transcript(key, moment)
                log_wake_gate("MENTION_CONFIRMATION_ACCEPTED")
                return ActivationDecision(
                    authorized=True,
                    wake_detected=False,
                    command_text=affirm_remainder,
                    session_active=True,
                    kind=WAKE_GATE_MENTION_CONFIRMATION,
                )
            if self._confirmation_remainder(text, affirmative=False) is not None:
                self._mention_pending_until = 0.0
                self._authorized_until = 0.0
                self._had_followup_session = False
                self._remember_transcript(key, moment)
                log_wake_gate("MENTION_DECLINED")
                return ActivationDecision(
                    authorized=False,
                    wake_detected=False,
                    command_text=text,
                    session_active=False,
                    kind=WAKE_GATE_MENTION_DECLINED,
                )
            # Unrelated speech while awaiting confirmation must not become a follow-up.
            self._mention_pending_until = 0.0
            log_wake_gate("UNADDRESSED_SPEECH_IGNORED", reason="mention_pending_unrelated")
            self._remember_transcript(key, moment)
            return ActivationDecision(
                authorized=False,
                wake_detected=False,
                command_text=text,
                session_active=False,
                kind=WAKE_GATE_UNADDRESSED,
            )

        if session_was_active:
            self._open_followup(moment, timeout)
            self._remember_transcript(key, moment)
            log_wake_gate("FOLLOWUP_ACCEPTED")
            return ActivationDecision(
                authorized=True,
                wake_detected=False,
                command_text=text,
                session_active=True,
                kind=WAKE_GATE_FOLLOWUP,
            )

        if self._should_offer_mention(text, moment):
            self._mention_pending_until = moment + DEFAULT_MENTION_CONFIRM_TIMEOUT_SECONDS
            self._mention_cooldown_until = moment + DEFAULT_MENTION_COOLDOWN_SECONDS
            self._remember_transcript(key, moment)
            log_wake_gate("MENTION_DETECTED")
            return ActivationDecision(
                authorized=False,
                wake_detected=False,
                command_text=text,
                session_active=False,
                kind=WAKE_GATE_MENTION,
                speak_text=mention_prompt_text(),
            )

        log_wake_gate("UNADDRESSED_SPEECH_IGNORED")
        logger.info("wake_gate: rejected_unaddressed_speech")
        self._remember_transcript(key, moment)
        return ActivationDecision(
            authorized=False,
            wake_detected=False,
            command_text=text,
            session_active=False,
            kind=WAKE_GATE_UNADDRESSED,
        )

    def _open_followup(self, moment: float, timeout: float) -> None:
        self._authorized_until = moment + timeout
        self._had_followup_session = True

    def _remember_transcript(self, key: str, moment: float) -> None:
        if key:
            self._last_transcript_key = key
            self._last_transcript_at = moment

    def _should_offer_mention(self, text: str, moment: float) -> bool:
        """Conservatively detect a conversational Reachy mention without treating it as a command."""
        if not text or moment < self._mention_cooldown_until:
            return False
        if mention_name_re().search(text) is None:
            return False
        # Prefer silence when the utterance looks like an imperative command with a mid name.
        lowered = text.lower()
        if re.match(r"^(?:please\s+)?(?:turn|set|get|show|tell|look|go|stop|start)\b", lowered):
            return False
        return True

    def _confirmation_remainder(self, text: str, *, affirmative: bool) -> str | None:
        """Return remainder after yes/no (and optional wake), or None if not a confirmation."""
        candidate = text.strip()
        if not candidate:
            return None
        wake_detected, after_wake = split_wake_prefix(candidate)
        if wake_detected:
            candidate = after_wake or candidate
        pattern = _AFFIRM_RE if affirmative else _DECLINE_RE
        match = pattern.match(candidate)
        if match is None:
            return None
        return candidate[match.end() :].lstrip(" ,.-").strip()
