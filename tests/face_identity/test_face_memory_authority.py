"""Regression tests for paused photo enrolment and tool-authoritative success speech."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.profile_store import read_packaged_default_profile
from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    FaceEmbedding,
    QualityResult,
)
from reachy_mini_conversation_app.face_identity.profile_store import PersonProfileStore
from reachy_mini_conversation_app.face_identity.identity_store import IdentityStore
from reachy_mini_conversation_app.face_identity.speech_authority import (
    PHOTO_ENROLMENT_DISABLED_SPOKEN,
    allows_success_claim,
    is_confirmed_success,
    is_robot_name_variant,
    authorize_proposed_speech,
    verify_persisted_enrolment,
    match_photo_enrolment_request,
    match_enrolment_status_question,
    match_face_memory_success_narration,
    honest_reply_for_unverified_enrolment,
    proposed_speech_claims_face_memory_success,
)


def _emb() -> FaceEmbedding:
    return FaceEmbedding(
        vector=(1.0, 0.0, 0.0, 0.0),
        model_id=MODEL_ID_SFACE,
        model_version=MODEL_VERSION_SFACE,
        quality=QualityResult(usable=True, score=0.9, reasons=()),
        created_at=1.0,
    )


def test_no_tool_result_blocks_enrolment_success_speech() -> None:
    """LLM may not claim enrolment without a verified tool result."""
    proposed = "I've enrolled Carol in my memory."
    assert proposed_speech_claims_face_memory_success(proposed)
    allowed, reason = authorize_proposed_speech(proposed, None)
    assert allowed is False
    assert reason == "missing_verified_tool_success"


@pytest.mark.parametrize(
    "result",
    [
        {"status": "disabled", "persisted": False},
        {"status": "no_clear_face", "persisted": False},
        {"status": "multiple_faces", "persisted": False},
        {"status": "persistence_failed", "persisted": False},
        {"error": "face_memory_disabled"},
    ],
)
def test_non_success_tool_statuses_block_success_claims(result: dict[str, object]) -> None:
    """Disabled / incomplete / failed statuses never authorize success speech."""
    assert is_confirmed_success(result) is False
    assert allows_success_claim(result, claim="enrolled") is False
    allowed, _ = authorize_proposed_speech("I've enrolled Carol.", result)
    assert allowed is False


def test_empty_identity_store_blocks_enrolment_success_claim(tmp_path: Path) -> None:
    """Empty stores cannot authorize enrolment success wording."""
    identities = IdentityStore(tmp_path)
    profiles = PersonProfileStore(tmp_path)
    assert identities.list_identities() == []
    assert profiles.list_profiles() == []
    assert is_confirmed_success({"status": "enrolled", "name": "Carol", "persisted": False}) is False


def test_ricci_transcript_is_success_narration_not_enrolment_request() -> None:
    """STT mangling must not become a new identity or success state."""
    transcript = "Ricci scanned his photo and enrolled him."
    assert match_face_memory_success_narration(transcript)
    assert is_robot_name_variant("Ricci")
    assert is_robot_name_variant("Richie")
    assert is_robot_name_variant("Ritchie")
    assert is_robot_name_variant("Reachy")
    # Narration alone must not create store rows.
    assert not is_confirmed_success(None)


def test_did_you_enrol_without_verification_is_honest_no() -> None:
    """Status questions without verified enrolment must answer not confirmed."""
    assert match_enrolment_status_question("Did you enrol her?")
    allowed, _ = authorize_proposed_speech("Yes, I've enrolled her.", None)
    assert allowed is False
    assert "haven't confirmed" in honest_reply_for_unverified_enrolment().lower()


def test_verified_persisted_enrolment_allows_success_speech(tmp_path: Path) -> None:
    """Mocked persisted enrolment authorizes success wording."""
    identities = IdentityStore(tmp_path)
    profiles = PersonProfileStore(tmp_path)
    record = identities.enrol([_emb(), _emb()])
    profiles.upsert(record.person_id, "Carol")
    verified = verify_persisted_enrolment(
        identities=identities,
        profiles=profiles,
        person_id=record.person_id,
        name="Carol",
        min_embeddings=2,
    )
    assert verified["status"] == "enrolled"
    assert verified["persisted"] is True
    assert allows_success_claim(verified, claim="enrolled")
    allowed, reason = authorize_proposed_speech("I'll remember Carol.", verified)
    assert allowed is True
    assert reason == "verified_tool_success"


def test_forget_and_update_require_persisted_true() -> None:
    """Forget/update confirmations require verified persisted tool success."""
    assert allows_success_claim({"status": "forgotten", "persisted": False}, claim="forgotten") is False
    assert allows_success_claim({"status": "forgotten", "persisted": True}, claim="forgotten") is True
    assert allows_success_claim({"status": "updated", "persisted": False}, claim="updated") is False
    assert allows_success_claim({"status": "updated", "persisted": True}, claim="updated") is True


def test_photo_enrolment_request_detected() -> None:
    """Physical photo-enrolment phrases are recognized for the pause guard."""
    assert match_photo_enrolment_request("Reachy, I have a photo.")
    assert match_photo_enrolment_request("Reachy, scan this photo.")
    assert match_photo_enrolment_request("remember this person from a picture")


@pytest.mark.asyncio
async def test_photo_enrol_tool_paused_by_photo_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """With photo enrolment off, scan returns disabled and does not start a session."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "true")
    monkeypatch.setenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", "false")
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.face_identity.service import FaceMemoryService
    from reachy_mini_conversation_app.tools.photo_enrol_face import PhotoEnrolFace

    service = FaceMemoryService(instance_path=None)
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        camera_enabled=True,
        face_memory_service=service,
    )
    result = await PhotoEnrolFace()(deps, action="scan")
    assert result["status"] == "disabled"
    assert result["persisted"] is False
    assert result["spoken"] == PHOTO_ENROLMENT_DISABLED_SPOKEN
    assert service.session.active is False
    assert service.session.phase == "idle"
    assert service.session.temporary_embeddings == []


@pytest.mark.asyncio
async def test_set_name_rejects_robot_name_variant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reachy/Ricci/Richie must never become a person name."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "true")
    monkeypatch.setenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", "true")
    from reachy_mini_conversation_app.face_identity.service import FaceMemoryService

    service = FaceMemoryService(instance_path=tmp_path)
    service.session.active = True
    service.session.phase = "await_name"
    service.session.temporary_embeddings = [_emb(), _emb()]
    result = service.set_name_and_persist("Ricci")
    assert result["persisted"] is False
    assert result.get("reason") == "robot_name_variant"
    assert service.pipeline().identities.list_identities() == []


def test_default_profile_does_not_expose_photo_enrol_face() -> None:
    """Paused photo enrolment must not be registered for the default conversation profile."""
    profile = read_packaged_default_profile()
    assert "photo_enrol_face" not in profile.default_tools
    assert "face_memory_admin" in profile.default_tools
    assert "who_is_in_frame" in profile.default_tools
    instructions = profile.instructions.lower()
    assert "never claim that a person was enrolled" in instructions
    assert "photo enrolment from the camera is currently disabled" in instructions


def test_photo_enrolment_flag_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Photo enrolment and live recognition stay off by default."""
    monkeypatch.delenv("FACE_MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_LIVE_RECOGNITION_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED", raising=False)
    from reachy_mini_conversation_app.face_identity.settings import (
        face_memory_enabled,
        face_memory_photo_enrolment_enabled,
        face_memory_live_recognition_enabled,
        face_memory_on_demand_recognition_enabled,
    )

    assert face_memory_enabled() is False
    assert face_memory_photo_enrolment_enabled() is False
    assert face_memory_live_recognition_enabled() is False
    assert face_memory_on_demand_recognition_enabled() is False
