"""Tests for on-demand person identity routing and recognition."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.profile_store import read_packaged_default_profile
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    FaceEmbedding,
    QualityResult,
    IdentityRecord,
)
from reachy_mini_conversation_app.face_identity.service import (
    FaceMemoryService,
    aggregate_on_demand_results,
)
from reachy_mini_conversation_app.face_identity.settings import (
    face_memory_photo_enrolment_enabled,
    face_memory_live_recognition_enabled,
    face_memory_on_demand_recognition_enabled,
)
from reachy_mini_conversation_app.face_identity.profile_store import PersonProfileStore
from reachy_mini_conversation_app.face_identity.identity_store import IdentityStore
from reachy_mini_conversation_app.face_identity.speech_authority import (
    NO_CLEAR_FACE_SPOKEN,
    MULTIPLE_FACES_SPOKEN,
    UNKNOWN_PERSON_SPOKEN,
    is_confirmed_success,
    format_known_person_spoken,
    match_self_identity_question,
    spoken_for_recognition_result,
    match_person_identity_question,
)


@pytest.mark.parametrize(
    "transcript",
    [
        "Reachy, who is this?",
        "Richie, who is this?",
        "Rishi, who's this?",
        "Ritchie, who's this in the picture?",
        "Ricci, who is this person?",
        "Do you know who this is?",
        "Reachy, do you know who this is?",
        "who is that person?",
    ],
)
def test_person_identity_intent_routes_to_face_memory(transcript: str) -> None:
    """Wake variants + identity questions must match on-demand face ID, not self-ID."""
    assert match_person_identity_question(transcript) is True
    assert match_self_identity_question(transcript) is False


@pytest.mark.parametrize(
    "transcript",
    [
        "Reachy, who are you?",
        "Who are you?",
        "What's your name?",
        "Reachy, what's your name?",
        "What is your name?",
        "Tell me your name",
    ],
)
def test_self_identity_questions_do_not_route_to_face_memory(transcript: str) -> None:
    """Self-identity questions must stay with Reachy's own identity handling."""
    assert match_person_identity_question(transcript) is False
    assert match_self_identity_question(transcript) is True


def test_wake_name_stripped_before_identity_classification() -> None:
    """Richie/Rishi prefixes must not turn 'who is this' into a self-name question."""
    assert match_person_identity_question("Richie, who is this?") is True
    assert match_person_identity_question("Rishi who's this?") is True
    assert match_self_identity_question("Richie, who are you?") is True


def test_spoken_known_with_relationship() -> None:
    """Known results speak the stored name, optionally with relationship."""
    assert format_known_person_spoken(name="Carol") == "That's Carol."
    assert format_known_person_spoken(name="Carol", relationship="Mum") == "That's Carol, your mum."
    spoken = spoken_for_recognition_result(
        {
            "status": "known",
            "person_id": "person_0001",
            "name": "Carol",
            "relationship": "Mum",
        }
    )
    assert spoken == "That's Carol, your mum."
    assert is_confirmed_success({"status": "known", "person_id": "person_0001", "name": "Carol"})


def test_spoken_fail_safe_statuses() -> None:
    """Unknown / no-face / multiple-faces never invent a name."""
    assert spoken_for_recognition_result({"status": "unknown"}) == UNKNOWN_PERSON_SPOKEN
    assert spoken_for_recognition_result({"status": "ambiguous"}) == UNKNOWN_PERSON_SPOKEN
    assert spoken_for_recognition_result({"status": "no_face"}) == NO_CLEAR_FACE_SPOKEN
    assert spoken_for_recognition_result({"status": "multiple_faces"}) == MULTIPLE_FACES_SPOKEN
    assert "Carol" not in spoken_for_recognition_result({"status": "unknown", "similarity": 0.39, "person_id": None})


def test_spoken_disabled_is_exact_without_vision_offer() -> None:
    """Disabled on-demand recognition uses the fixed line and no vision invitation."""
    from reachy_mini_conversation_app.face_identity.speech_authority import ON_DEMAND_DISABLED_SPOKEN

    spoken = spoken_for_recognition_result({"error": "on_demand_recognition_disabled", "status": "disabled"})
    assert spoken == ON_DEMAND_DISABLED_SPOKEN
    lowered = spoken.lower()
    assert "frame" not in lowered
    assert "camera" not in lowered
    assert "image" not in lowered
    assert "look" not in lowered


def test_aggregate_known_requires_agreeing_frames() -> None:
    """Temporal confirmation needs a consistent winner across usable frames."""
    frames = [
        {"status": "known", "person_id": "person_0001", "name": "Carol", "similarity": 0.72, "margin": 0.3},
        {"status": "known", "person_id": "person_0001", "name": "Carol", "similarity": 0.71, "margin": 0.28},
        {"status": "unknown", "similarity": 0.2},
        {"status": "no_face"},
    ]
    result = aggregate_on_demand_results(frames)
    assert result["status"] == "known"
    assert result["person_id"] == "person_0001"
    assert result["agreeing_frames"] >= 2


def test_aggregate_unknown_below_threshold() -> None:
    """Below-threshold matches stay unknown with no nearest-name guess."""
    frames = [
        {"status": "unknown", "similarity": 0.2, "reasons": ["below_threshold"]},
        {"status": "unknown", "similarity": 0.25, "reasons": ["below_threshold"]},
    ]
    result = aggregate_on_demand_results(frames)
    assert result["status"] == "unknown"
    assert result.get("name") is None
    assert spoken_for_recognition_result(result) == UNKNOWN_PERSON_SPOKEN


def test_aggregate_ambiguous_margin() -> None:
    """Ambiguous margin must not guess."""
    frames = [
        {"status": "unknown", "similarity": 0.7, "reasons": ["ambiguous_margin"]},
        {"status": "unknown", "similarity": 0.69, "reasons": ["ambiguous_margin"]},
    ]
    result = aggregate_on_demand_results(frames)
    assert result["status"] == "ambiguous"
    assert spoken_for_recognition_result(result) == UNKNOWN_PERSON_SPOKEN


def test_aggregate_no_face_and_multiple() -> None:
    """No usable face and multiple faces fail safe."""
    assert aggregate_on_demand_results([{"status": "no_face"}, {"status": "no_face"}])["status"] == "no_face"
    multi = aggregate_on_demand_results(
        [{"status": "multiple_faces", "face_count": 2}, {"status": "multiple_faces", "face_count": 2}]
    )
    assert multi["status"] == "multiple_faces"
    assert spoken_for_recognition_result(multi) == MULTIPLE_FACES_SPOKEN


def test_aggregate_conflicting_known_winners_is_ambiguous() -> None:
    """Disagreeing known winners across frames must not pick arbitrarily."""
    frames = [
        {"status": "known", "person_id": "person_0001", "name": "Carol", "similarity": 0.8},
        {"status": "known", "person_id": "person_0002", "name": "Dave", "similarity": 0.79},
        {"status": "known", "person_id": "person_0001", "name": "Carol", "similarity": 0.81},
        {"status": "known", "person_id": "person_0002", "name": "Dave", "similarity": 0.8},
    ]
    result = aggregate_on_demand_results(frames)
    assert result["status"] == "ambiguous"


def test_on_demand_recognition_does_not_mutate_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognition is read-only against IdentityStore and PersonProfileStore."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "true")
    monkeypatch.setenv("FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED", "true")
    identities = IdentityStore(tmp_path)
    profiles = PersonProfileStore(tmp_path)
    vector = [1.0] + [0.0] * 127
    identities.enrol(
        [
            FaceEmbedding(
                vector=tuple(vector),
                model_id=MODEL_ID_SFACE,
                model_version=MODEL_VERSION_SFACE,
                quality=QualityResult(usable=True, score=0.9, reasons=()),
                created_at=1.0,
            ),
            FaceEmbedding(
                vector=tuple(vector),
                model_id=MODEL_ID_SFACE,
                model_version=MODEL_VERSION_SFACE,
                quality=QualityResult(usable=True, score=0.9, reasons=()),
                created_at=2.0,
            ),
        ]
    )
    person_id = identities.list_identities()[0].person_id
    profiles.upsert(person_id, "Carol", relationship="Mum")

    identity_path = identities.path
    profile_path = profiles.path
    identity_before = identity_path.read_bytes()
    profile_before = profile_path.read_bytes()
    embedding_count_before = len(identities.get(person_id).embeddings)  # type: ignore[union-attr]

    service = FaceMemoryService(instance_path=tmp_path)
    # Bypass YuNet/SFace: feed already-recognized frame payloads through the aggregator.
    result = aggregate_on_demand_results(
        [
            {
                "status": "known",
                "person_id": person_id,
                "name": "Carol",
                "similarity": 0.9,
                "margin": 0.5,
            },
            {
                "status": "known",
                "person_id": person_id,
                "name": "Carol",
                "similarity": 0.88,
                "margin": 0.4,
            },
        ],
        profiles=service.pipeline().profiles,
    )
    assert result["status"] == "known"
    assert result["name"] == "Carol"

    assert identity_path.read_bytes() == identity_before
    assert profile_path.read_bytes() == profile_before
    assert len(identities.get(person_id).embeddings) == embedding_count_before  # type: ignore[union-attr]
    assert profiles.get(person_id) is not None
    assert profiles.get(person_id).name == "Carol"  # type: ignore[union-attr]
    assert profiles.get(person_id).relationship == "Mum"  # type: ignore[union-attr]


def test_who_is_in_frame_tool_uses_camera_not_llm_vision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On-demand ID uses media.get_frame and never attaches JPEG for LLM vision."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "true")
    monkeypatch.setenv("FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED", "true")

    from reachy_mini_conversation_app.tools.who_is_in_frame import WhoIsInFrame

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    media = MagicMock()
    media.get_frame.return_value = frame
    media.get_frame_jpeg = MagicMock(side_effect=AssertionError("JPEG vision path must not run"))
    robot = MagicMock()
    robot.media = media

    service = FaceMemoryService(instance_path=tmp_path)
    service.recognize_frame_window = MagicMock(  # type: ignore[method-assign]
        return_value={
            "status": "unknown",
            "usable_frames": 2,
            "spoken": UNKNOWN_PERSON_SPOKEN,
        }
    )
    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=MagicMock(),
        instance_path=tmp_path,
        camera_enabled=True,
        face_memory_service=service,
    )

    import asyncio

    result = asyncio.run(WhoIsInFrame()(deps))
    assert result["status"] == "unknown"
    assert "b64_im" not in result
    media.get_frame_jpeg.assert_not_called()
    assert media.get_frame.call_count >= 1


def test_default_profile_includes_who_is_in_frame() -> None:
    """Default profile exposes on-demand ID and keeps photo enrolment paused."""
    profile = read_packaged_default_profile()
    assert "who_is_in_frame" in profile.default_tools
    assert "photo_enrol_face" not in profile.default_tools
    instructions = profile.instructions.lower()
    assert "who is this" in instructions
    assert "do not use the camera tool for person identity" in instructions
    assert "do not offer to look at the camera" in instructions
    assert "cannot interpret images" in instructions


def test_on_demand_flag_defaults_on_when_face_memory_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """On-demand recognition defaults on; live recognition and photo enrolment stay off."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "true")
    monkeypatch.delenv("FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_LIVE_RECOGNITION_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", raising=False)
    assert face_memory_on_demand_recognition_enabled() is True
    assert face_memory_live_recognition_enabled() is False
    assert face_memory_photo_enrolment_enabled() is False


def test_model_error_does_not_hallucinate_name() -> None:
    """Structured errors speak a safe line without inventing identity."""
    spoken = spoken_for_recognition_result({"error": "capture_failed: RuntimeError", "status": "error"})
    assert "Carol" not in spoken
    assert spoken


def test_identity_record_gallery_match_known(tmp_path: Path) -> None:
    """Enrolled gallery vectors still match through the existing matcher path."""
    from reachy_mini_conversation_app.face_identity.matcher import match_embedding

    probe = [1.0, 0.0, 0.0]
    identities = [
        IdentityRecord(
            person_id="person_0001",
            model_id=MODEL_ID_SFACE,
            model_version=MODEL_VERSION_SFACE,
            embeddings=[[1.0, 0.0, 0.0], [0.99, 0.01, 0.0]],
        )
    ]
    result = match_embedding(probe, identities, threshold=0.40, margin=0.05)
    assert result.status == "known"
    assert result.person_id == "person_0001"
