"""Tool-authoritative face-memory speech and photo-enrolment intent helpers."""

from __future__ import annotations
import re
from typing import Any, Mapping

from reachy_mini_conversation_app.activation import strip_transcript_name_prefix


PHOTO_ENROLMENT_DISABLED_SPOKEN = "Photo enrolment from my camera is currently disabled."
NO_VERIFIED_ENROLMENT_SPOKEN = "I haven't confirmed any face enrolment — nothing was saved."
ON_DEMAND_DISABLED_SPOKEN = "On-demand face recognition is turned off right now."
UNKNOWN_PERSON_SPOKEN = "I don't recognise this person."
NO_CLEAR_FACE_SPOKEN = "I can't see a clear face right now."
MULTIPLE_FACES_SPOKEN = "I can see more than one person. Could you show me who you mean?"
RECOGNITION_ERROR_SPOKEN = "I had trouble identifying that person."
KNOWN_WITHOUT_NAME_SPOKEN = "I recognise this person, but I don't have a name for them."

# Statuses that may authorize success speech when the tool result also verifies persistence.
SUCCESS_STATUSES = frozenset(
    {
        "enrolled",
        "complete",
        "forgotten",
        "updated",
        "known",
    }
)

# Non-success / incomplete statuses must never become success speech.
NON_SUCCESS_STATUSES = frozenset(
    {
        "disabled",
        "no_clear_face",
        "multiple_faces",
        "quality_failed",
        "awaiting_consent",
        "await_consent",
        "consent_declined",
        "declined",
        "awaiting_name",
        "await_name",
        "awaiting_details",
        "await_details",
        "cancelled",
        "aborted_for_sleep",
        "persistence_failed",
        "not_found",
        "error",
        "rejected",
        "ok",  # list-only; not an enrolment/forget success claim
        "unknown",
        "ambiguous",
        "no_face",
        "unusable",
        "model_mismatch",
    }
)

_ROBOT_NAME_VARIANTS = frozenset(
    {
        "reachy",
        "reachy mini",
        "richie",
        "ricci",
        "ritchie",
        "rishi",
        "ray g",
        "raygee",
    }
)

_SUCCESS_CLAIM_RE = re.compile(
    r"\b(?:"
    r"i(?:'ve| have)\s+(?:enrolled|remembered|saved|stored|forgotten)|"
    r"i(?:'ll| will)\s+remember|"
    r"(?:enrolled|remembered|saved|stored)\s+(?:\w+\s+){0,3}(?:in\s+my\s+memory|successfully)|"
    r"face\s+memory\s+(?:updated|saved)|"
    r"i\s+know\s+(?:who|them|him|her)"
    r")\b",
    re.IGNORECASE,
)

_PHOTO_ENROL_REQUEST_RE = re.compile(
    r"\b(?:"
    r"i\s+have\s+a\s+photo|"
    r"scan\s+(?:this|the|his|her|their)?\s*photo|"
    r"hold(?:ing)?\s+up\s+a\s+photo|"
    r"show(?:ing)?\s+(?:you\s+)?(?:a\s+)?photo|"
    r"remember\s+(?:this|that)\s+person|"
    r"enr?ol(?:l)?\s+(?:this|that|the|him|her|them)|"
    r"remember\s+(?:someone|somebody)\s+from\s+(?:a\s+)?(?:photo|picture)|"
    r"from\s+(?:a\s+)?(?:held[- ]?up\s+)?(?:photo|picture)"
    r")\b",
    re.IGNORECASE,
)

_PAST_ENROL_NARRATION_RE = re.compile(
    r"\b(?:"
    r"(?:scanned|scan)\s+(?:his|her|their|the|this)?\s*photo|"
    r"enr?ol(?:l)?ed\s+(?:him|her|them|this|that)|"
    r"remembered\s+(?:him|her|them)"
    r").*\b(?:"
    r"enr?ol(?:l)?ed|remembered|scanned"
    r")\b|"
    r"\b(?:"
    r"enr?ol(?:l)?ed|remembered"
    r")\s+(?:him|her|them)\b|"
    r"\bscanned\s+(?:his|her|their)\s+photo\s+and\s+(?:enr?ol(?:l)?ed|remembered)\b",
    re.IGNORECASE,
)

_ENROL_STATUS_QUESTION_RE = re.compile(
    r"\b(?:"
    r"did\s+you\s+(?:enr?ol(?:l)?|remember|save|store)|"
    r"have\s+you\s+(?:enr?ol(?:l)?ed|remembered|saved|stored)|"
    r"is\s+(?:she|he|they|that\s+person)\s+(?:enr?ol(?:l)?ed|remembered|saved)|"
    r"did\s+(?:the\s+)?enrol(?:l)?ment\s+(?:work|succeed|finish)"
    r")\b",
    re.IGNORECASE,
)

# Apostrophes are stripped by _normalize_transcript ("who's" -> "whos").
_SELF_IDENTITY_RE = re.compile(
    r"\b(?:"
    r"who\s+are\s+you|"
    r"whore\s+you|"
    r"what(?:s|\s+is)\s+your\s+name|"
    r"what\s+are\s+you\s+called|"
    r"tell\s+me\s+your\s+name|"
    r"introduce\s+yourself"
    r")\b",
    re.IGNORECASE,
)

_PERSON_IDENTITY_RE = re.compile(
    r"\b(?:"
    r"who(?:s|\s+is)\s+this(?:\s+person)?(?:\s+in\s+(?:the\s+)?(?:picture|photo|image))?|"
    r"who(?:s|\s+is)\s+that(?:\s+person)?(?:\s+in\s+(?:the\s+)?(?:picture|photo|image))?|"
    r"do\s+you\s+know\s+who\s+(?:this|that)\s+is|"
    r"do\s+you\s+recognise\s+(?:this|that)(?:\s+person)?|"
    r"do\s+you\s+recognize\s+(?:this|that)(?:\s+person)?|"
    r"can\s+you\s+(?:tell|identify)\s+who\s+(?:this|that)\s+is|"
    r"identify\s+(?:this|that)\s+person"
    r")\b",
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[.!?,;:]+", " ", name.strip().lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def is_robot_name_variant(name: str) -> bool:
    """Return True when a candidate person name is a Reachy STT variant."""
    return _normalize_name(name) in _ROBOT_NAME_VARIANTS


def face_memory_photo_enrolment_disabled_result() -> dict[str, Any]:
    """Structured result when camera photo enrolment is paused."""
    return {
        "status": "disabled",
        "persisted": False,
        "spoken": PHOTO_ENROLMENT_DISABLED_SPOKEN,
    }


def is_confirmed_success(result: Mapping[str, Any] | None) -> bool:
    """Return whether a tool result may authorize face-memory success speech."""
    if not isinstance(result, Mapping):
        return False
    if result.get("error"):
        return False
    status = str(result.get("status") or "").strip().lower()
    if status in NON_SUCCESS_STATUSES or status not in SUCCESS_STATUSES:
        return False
    if status in {"enrolled", "complete", "forgotten", "updated"}:
        return result.get("persisted") is True
    # Recognition "known" still requires an explicit person_id.
    if status == "known":
        return bool(result.get("person_id"))
    return False


def allows_success_claim(result: Mapping[str, Any] | None, *, claim: str) -> bool:
    """Return whether success wording of the given claim kind is authorized."""
    if result is None or not is_confirmed_success(result):
        return False
    status = str(result.get("status") or "").strip().lower()
    claim_key = claim.strip().lower()
    allowed = {
        "enrolled": {"enrolled", "complete"},
        "remembered": {"enrolled", "complete", "known"},
        "forgotten": {"forgotten"},
        "updated": {"updated", "complete"},
        "recognised": {"known"},
        "recognized": {"known"},
        "saved": {"enrolled", "complete", "updated"},
        "stored": {"enrolled", "complete", "updated"},
    }
    return status in allowed.get(claim_key, set())


def proposed_speech_claims_face_memory_success(text: str) -> bool:
    """Detect spoken claims that enrolment/forget/update succeeded."""
    return _SUCCESS_CLAIM_RE.search(text or "") is not None


def authorize_proposed_speech(
    proposed_speech: str,
    tool_result: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Authorize assistant success speech only from a verified tool result."""
    if not proposed_speech_claims_face_memory_success(proposed_speech):
        return True, "no_face_memory_success_claim"
    if is_confirmed_success(tool_result):
        return True, "verified_tool_success"
    return False, "missing_verified_tool_success"


def match_photo_enrolment_request(transcript: str) -> bool:
    """Return whether the user asked for physical camera photo enrolment."""
    text = _normalize_transcript(transcript)
    if not text:
        return False
    return _PHOTO_ENROL_REQUEST_RE.search(text) is not None


def match_face_memory_success_narration(transcript: str) -> bool:
    """Detect STT mangling that narrates enrolment success as a user utterance."""
    text = _normalize_transcript(transcript)
    if not text:
        return False
    return _PAST_ENROL_NARRATION_RE.search(text) is not None


def match_enrolment_status_question(transcript: str) -> bool:
    """Detect questions asking whether enrolment already happened."""
    text = _normalize_transcript(transcript)
    if not text:
        return False
    return _ENROL_STATUS_QUESTION_RE.search(text) is not None


def identity_command_text(transcript: str) -> str:
    """Strip a leading wake-name variant before identity intent classification."""
    return _normalize_transcript(strip_transcript_name_prefix(transcript or ""))


def match_self_identity_question(transcript: str) -> bool:
    """Return whether the user asked Reachy about Reachy's own identity."""
    text = identity_command_text(transcript)
    if not text:
        return False
    return _SELF_IDENTITY_RE.search(text) is not None


def match_person_identity_question(transcript: str) -> bool:
    """Return whether the user asked who is in the current camera frame."""
    text = identity_command_text(transcript)
    if not text:
        return False
    if match_self_identity_question(transcript):
        return False
    return _PERSON_IDENTITY_RE.search(text) is not None


def format_known_person_spoken(*, name: str, relationship: str | None = None) -> str:
    """Build the short spoken line for a verified known recognition."""
    cleaned_name = name.strip()
    if not cleaned_name:
        return KNOWN_WITHOUT_NAME_SPOKEN
    rel = (relationship or "").strip()
    if not rel:
        return f"That's {cleaned_name}."
    lowered = rel.lower()
    if lowered.startswith("your "):
        return f"That's {cleaned_name}, {lowered}."
    return f"That's {cleaned_name}, your {lowered}."


def spoken_for_recognition_result(result: Mapping[str, Any] | None) -> str:
    """Return tool-authoritative speech for an on-demand recognition result."""
    if not isinstance(result, Mapping):
        return RECOGNITION_ERROR_SPOKEN
    if result.get("error"):
        error = str(result.get("error") or "")
        if error in {"face_memory_disabled", "on_demand_recognition_disabled"}:
            return ON_DEMAND_DISABLED_SPOKEN
        if error == "camera_disabled":
            return "The camera is disabled, so I cannot look."
        return RECOGNITION_ERROR_SPOKEN
    explicit = result.get("spoken")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    status = str(result.get("status") or "").strip().lower()
    if status == "known":
        name = result.get("name")
        if is_confirmed_success(result) and isinstance(name, str) and name.strip():
            relationship = result.get("relationship")
            rel = relationship if isinstance(relationship, str) else None
            return format_known_person_spoken(name=name, relationship=rel)
        if is_confirmed_success(result):
            return KNOWN_WITHOUT_NAME_SPOKEN
        return UNKNOWN_PERSON_SPOKEN
    if status in {"unknown", "ambiguous"}:
        return UNKNOWN_PERSON_SPOKEN
    if status in {"no_face", "unusable", "no_clear_face"}:
        return NO_CLEAR_FACE_SPOKEN
    if status == "multiple_faces":
        return MULTIPLE_FACES_SPOKEN
    return RECOGNITION_ERROR_SPOKEN


def honest_reply_for_unverified_enrolment() -> str:
    """Spoken text when no verified enrolment exists."""
    return NO_VERIFIED_ENROLMENT_SPOKEN


def verify_persisted_enrolment(
    *,
    identities: Any,
    profiles: Any,
    person_id: str,
    name: str,
    min_embeddings: int = 1,
) -> dict[str, Any]:
    """Re-read identity + profile stores and confirm enrolment really landed."""
    cleaned_id = person_id.strip()
    cleaned_name = name.strip()
    if not cleaned_id or not cleaned_name:
        return {
            "status": "persistence_failed",
            "persisted": False,
            "spoken": "I could not verify that person was saved.",
            "reason": "missing_person_id_or_name",
        }
    identity = identities.get(cleaned_id)
    profile = profiles.get(cleaned_id)
    if identity is None:
        return {
            "status": "persistence_failed",
            "persisted": False,
            "person_id": cleaned_id,
            "name": cleaned_name,
            "spoken": "I could not verify that person was saved.",
            "reason": "identity_missing",
        }
    if not identity.embeddings or len(identity.embeddings) < min_embeddings:
        return {
            "status": "persistence_failed",
            "persisted": False,
            "person_id": cleaned_id,
            "name": cleaned_name,
            "spoken": "I could not verify that person was saved.",
            "reason": "embeddings_missing",
        }
    if profile is None or profile.name.strip().lower() != cleaned_name.lower():
        return {
            "status": "persistence_failed",
            "persisted": False,
            "person_id": cleaned_id,
            "name": cleaned_name,
            "spoken": "I could not verify that person was saved.",
            "reason": "profile_missing_or_mismatch",
        }
    return {
        "status": "enrolled",
        "person_id": cleaned_id,
        "name": profile.name,
        "embedding_count": len(identity.embeddings),
        "persisted": True,
    }


def _normalize_transcript(transcript: str) -> str:
    text = (transcript or "").lower().strip().replace("'", "")
    text = re.sub(r"[.!?,;:]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
