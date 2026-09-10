"""Unit tests for face-identity matcher, stores, quality, and head-freeze helpers."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini_conversation_app.moves import MovementManager
from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    DetectedFace,
    FaceEmbedding,
    FaceLandmarks,
    QualityResult,
    IdentityRecord,
)
from reachy_mini_conversation_app.face_identity.matcher import match_embedding, cosine_similarity
from reachy_mini_conversation_app.face_identity.quality import assess_face_quality
from reachy_mini_conversation_app.face_identity.photo_session import HeadTrackingGuard, collect_photo_embeddings
from reachy_mini_conversation_app.face_identity.profile_store import PersonProfileStore
from reachy_mini_conversation_app.face_identity.identity_store import IdentityStore


def _vec(*values: float) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr / max(float(np.linalg.norm(arr)), 1e-12)
    return [float(v) for v in arr.tolist()]


def _face(
    *,
    width: int = 640,
    height: int = 480,
    bbox: tuple[float, float, float, float] = (200, 120, 180, 180),
    confidence: float = 0.95,
) -> DetectedFace:
    x, y, w, h = bbox
    return DetectedFace(
        bbox=bbox,
        confidence=confidence,
        landmarks=FaceLandmarks(
            right_eye=(x + w * 0.3, y + h * 0.35),
            left_eye=(x + w * 0.7, y + h * 0.35),
            nose=(x + w * 0.5, y + h * 0.55),
            right_mouth=(x + w * 0.35, y + h * 0.75),
            left_mouth=(x + w * 0.65, y + h * 0.75),
        ),
        frame_width=width,
        frame_height=height,
    )


def test_cosine_similarity_identical() -> None:
    """Identical normalized vectors score ~1."""
    probe = _vec(0.2, 0.4, 0.8)
    assert cosine_similarity(probe, probe) == pytest.approx(1.0, abs=1e-5)


def test_match_known_with_margin() -> None:
    """Clear best match above threshold and margin returns known."""
    identities = [
        IdentityRecord(
            person_id="person_0001",
            model_id=MODEL_ID_SFACE,
            model_version=MODEL_VERSION_SFACE,
            embeddings=[_vec(1, 0, 0), _vec(0.95, 0.05, 0)],
        ),
        IdentityRecord(
            person_id="person_0002",
            model_id=MODEL_ID_SFACE,
            model_version=MODEL_VERSION_SFACE,
            embeddings=[_vec(0, 1, 0)],
        ),
    ]
    result = match_embedding(_vec(1, 0, 0), identities, threshold=0.4, margin=0.05)
    assert result.status == "known"
    assert result.person_id == "person_0001"
    assert result.similarity is not None and result.similarity > 0.9


def test_match_ambiguous_margin_returns_unknown() -> None:
    """Near-tied top scores must return UNKNOWN."""
    identities = [
        IdentityRecord(
            person_id="person_0001",
            model_id=MODEL_ID_SFACE,
            model_version=MODEL_VERSION_SFACE,
            embeddings=[_vec(0.8, 0.2, 0)],
        ),
        IdentityRecord(
            person_id="person_0002",
            model_id=MODEL_ID_SFACE,
            model_version=MODEL_VERSION_SFACE,
            embeddings=[_vec(0.78, 0.22, 0)],
        ),
    ]
    result = match_embedding(_vec(0.79, 0.21, 0), identities, threshold=0.3, margin=0.05)
    assert result.status == "unknown"
    assert "ambiguous_margin" in result.reasons


def test_match_model_version_mismatch() -> None:
    """Gallery embeddings from another model version are not compared silently."""
    identities = [
        IdentityRecord(
            person_id="person_0001",
            model_id=MODEL_ID_SFACE,
            model_version="old_sface",
            embeddings=[_vec(1, 0, 0)],
        )
    ]
    result = match_embedding(_vec(1, 0, 0), identities)
    assert result.status == "model_mismatch"


def test_identity_and_profile_stores_separate(tmp_path: Path) -> None:
    """Names live in profile store; embeddings live in identity store under person_id."""
    identities = IdentityStore(tmp_path)
    profiles = PersonProfileStore(tmp_path)
    emb = FaceEmbedding(
        vector=tuple(_vec(1, 0, 0, 0)),
        model_id=MODEL_ID_SFACE,
        model_version=MODEL_VERSION_SFACE,
        quality=QualityResult(usable=True, score=0.9, reasons=()),
        created_at=1.0,
    )
    emb2 = FaceEmbedding(
        vector=tuple(_vec(0.98, 0.1, 0, 0)),
        model_id=MODEL_ID_SFACE,
        model_version=MODEL_VERSION_SFACE,
        quality=QualityResult(usable=True, score=0.88, reasons=()),
        created_at=2.0,
    )
    record = identities.enrol([emb, emb2])
    assert record.person_id.startswith("person_")
    assert "Sarah" not in record.person_id
    profile = profiles.upsert(record.person_id, "Sarah", append_note="likes gardening")
    assert profile.name == "Sarah"
    assert identities.get("Sarah") is None
    assert profiles.get(record.person_id) is not None
    assert len(identities.get(record.person_id).embeddings) == 2  # type: ignore[union-attr]


def test_forget_removes_identity_and_profile(tmp_path: Path) -> None:
    """Forgetting a person removes both stores."""
    from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline

    pipeline = FaceIdentityPipeline.__new__(FaceIdentityPipeline)
    pipeline.instance_path = tmp_path
    pipeline.identities = IdentityStore(tmp_path)
    pipeline.profiles = PersonProfileStore(tmp_path)
    emb = FaceEmbedding(
        vector=tuple(_vec(1, 0)),
        model_id=MODEL_ID_SFACE,
        model_version=MODEL_VERSION_SFACE,
        quality=QualityResult(usable=True, score=0.9, reasons=()),
        created_at=1.0,
    )
    record = pipeline.identities.enrol([emb, emb])
    pipeline.profiles.upsert(record.person_id, "Sarah")
    result = pipeline.forget_person(name="Sarah")
    assert result["status"] == "forgotten"
    assert pipeline.identities.get(record.person_id) is None
    assert pipeline.profiles.get(record.person_id) is None


def test_quality_rejects_small_and_blurry() -> None:
    """Small and blurred faces are marked unusable with reasons."""
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    small = _face(width=320, height=240, bbox=(10, 10, 12, 12), confidence=0.9)
    result = assess_face_quality(frame, small)
    assert result.usable is False
    assert "face_too_small" in result.reasons

    # Uniform crop → low Laplacian variance → blurred
    large = _face(width=320, height=240, bbox=(40, 30, 160, 160), confidence=0.95)
    blurred = assess_face_quality(frame, large)
    assert blurred.usable is False
    assert "blurred" in blurred.reasons


def test_head_tracking_guard_restores_prior_on_state() -> None:
    """Prior tracking ON freezes then restores ON; prior OFF stays OFF."""
    robot = MagicMock()
    manager = MovementManager(current_robot=robot)
    manager._head_tracking = True
    with HeadTrackingGuard(manager) as guard:
        assert guard._prior_enabled is True
        assert guard._held_stillness is True
        manager._handle_command("hold_still_for_capture", None, 0.0)
        manager._handle_command("freeze_head_tracking", None, 0.0)
        robot.start_head_tracking.assert_called_with(weight=0.0)
        robot.disable_wobbling.assert_called()
        assert manager._photo_stillness is True
    manager._handle_command("restore_head_tracking", True, 0.0)
    manager._handle_command("release_capture_stillness", None, 0.0)
    assert manager._head_tracking is True
    assert manager._photo_stillness is False
    robot.start_head_tracking.assert_called_with(weight=1.0)
    robot.enable_wobbling.assert_called()

    robot.reset_mock()
    manager._head_tracking = False
    with HeadTrackingGuard(manager) as guard:
        assert guard._prior_enabled is False
        manager._handle_command("hold_still_for_capture", None, 0.0)
        manager._handle_command("freeze_head_tracking", None, 0.0)
    manager._handle_command("restore_head_tracking", False, 0.0)
    manager._handle_command("release_capture_stillness", None, 0.0)
    assert manager._head_tracking is False
    robot.stop_head_tracking.assert_called()
    # Must not enable tracking when it was previously off.
    assert not any(call.kwargs.get("weight") == 1.0 for call in robot.start_head_tracking.call_args_list)


def test_head_tracking_guard_restores_after_exception() -> None:
    """Exceptions during the guarded block still restore tracking and stillness."""
    manager = MagicMock()
    manager.get_head_tracking_enabled.return_value = True
    try:
        with HeadTrackingGuard(manager):
            raise RuntimeError("scan failed")
    except RuntimeError:
        pass
    manager.hold_still_for_capture.assert_called_once()
    manager.restore_head_tracking.assert_called_once_with(True)
    manager.release_capture_stillness.assert_called_once()


def test_collect_photo_embeddings_rejects_multiple_faces() -> None:
    """Multi-face windows do not enrol."""
    pipeline = MagicMock()
    face_a = _face()
    face_b = _face(bbox=(20, 20, 100, 100))
    pipeline.detect.return_value = [face_a, face_b]
    frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
    result = collect_photo_embeddings(pipeline, frames)
    assert result["status"] == "multiple_faces"


def test_face_memory_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feature flags default to off."""
    monkeypatch.delenv("FACE_MEMORY_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", raising=False)
    monkeypatch.delenv("FACE_MEMORY_LIVE_RECOGNITION_ENABLED", raising=False)
    from reachy_mini_conversation_app.face_identity.settings import (
        face_memory_enabled,
        face_memory_photo_enrolment_enabled,
        face_memory_live_recognition_enabled,
    )

    assert face_memory_enabled() is False
    assert face_memory_photo_enrolment_enabled() is False
    assert face_memory_live_recognition_enabled() is False


@pytest.mark.asyncio
async def test_photo_enrol_tool_respects_disabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Photo enrolment tool is a no-op when the feature flag is off."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "false")
    monkeypatch.setenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", "false")
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.tools.photo_enrol_face import PhotoEnrolFace

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    result = await PhotoEnrolFace()(deps, action="scan")
    assert result["error"] == "face_memory_disabled"
    assert result.get("persisted") is False


@pytest.mark.asyncio
async def test_photo_enrol_tool_respects_photo_enrolment_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Face memory on with photo enrolment paused returns the disabled spoken status."""
    monkeypatch.setenv("FACE_MEMORY_ENABLED", "true")
    monkeypatch.setenv("FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", "false")
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.tools.photo_enrol_face import PhotoEnrolFace
    from reachy_mini_conversation_app.face_identity.speech_authority import PHOTO_ENROLMENT_DISABLED_SPOKEN

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), face_memory_service=MagicMock())
    result = await PhotoEnrolFace()(deps, action="scan")
    assert result["status"] == "disabled"
    assert result["spoken"] == PHOTO_ENROLMENT_DISABLED_SPOKEN
    deps.face_memory_service.scan_photo.assert_not_called()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_go_to_sleep_aborts_photo_enrolment() -> None:
    """Sleep wins over an in-progress photo enrolment."""
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.tools.go_to_sleep import GoToSleep

    service = MagicMock()
    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        go_to_sleep=lambda: {"status": "sleeping"},
        face_memory_service=service,
    )
    result = await GoToSleep()(deps)
    assert result["status"] == "sleeping"
    service.abort_photo_enrolment.assert_called_once_with(deps.movement_manager)


def test_no_legacy_lbph_imports() -> None:
    """Broken LBPH/Haar face system must remain absent from the package."""
    import pathlib

    import reachy_mini_conversation_app as pkg

    root = pathlib.Path(pkg.__file__).resolve().parent
    banned = ("LBPHFaceRecognizer", "CascadeClassifier", "family_faces")
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} found in {path}"
