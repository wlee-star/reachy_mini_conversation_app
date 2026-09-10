"""Uploaded-photo management, real quality gates, and file-store failure tests."""

import json
import base64
import threading
from pathlib import Path
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from control_dashboard import people, server
from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    DetectedFace,
    FaceEmbedding,
    FaceLandmarks,
    QualityResult,
)
from reachy_mini_conversation_app.face_identity.models import SFACE_SPEC, YUNET_SPEC, default_models_dir
from reachy_mini_conversation_app.face_identity.embedder import SFaceEmbedder
from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline


@pytest.fixture
def photo() -> dict[str, str]:
    """Provide a small valid image with sufficient sharpness and exposure."""
    frame = np.random.default_rng(42).integers(50, 210, (240, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return {"name": "../photo.png", "type": "image/png", "content": base64.b64encode(encoded).decode()}


@pytest.fixture
def service(tmp_path: Path) -> people.PeopleService:
    """Use real quality checks and stores with deterministic model substitutes."""
    detector = MagicMock()
    detector.detect.return_value = [
        DetectedFace(
            (80, 50, 130, 140),
            0.99,
            FaceLandmarks((110, 95), (165, 95), (140, 120), (115, 150), (165, 150)),
            320,
            240,
        )
    ]
    embedder = MagicMock()
    embedder.embed_face.return_value = FaceEmbedding(
        (1.0,) + (0.0,) * 127,
        MODEL_ID_SFACE,
        MODEL_VERSION_SFACE,
        QualityResult(True, 1.0),
        123.0,
    )
    instance = people.PeopleService(tmp_path)
    instance.pipeline = FaceIdentityPipeline(tmp_path, detector=detector, embedder=embedder)
    return instance


@pytest.mark.parametrize("count", [1, 3, 10])
def test_enrol_uploads_verified_camera_independent(
    service: people.PeopleService, photo: dict[str, str], count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Uploads persist both stores even with all conversational face features off."""
    for flag in ("FACE_MEMORY_ENABLED", "FACE_MEMORY_PHOTO_ENROLMENT_ENABLED", "FACE_MEMORY_LIVE_RECOGNITION_ENABLED"):
        monkeypatch.setenv(flag, "false")
    result = service.execute("enrol", {"name": "Carol", "relationship": "Mum", "photos": [photo] * count})
    assert result["persisted"] is True
    person_id = result["person_id"]
    assert len(service.identities.get(person_id).embeddings) == count
    assert service.profiles.get(person_id).relationship == "Mum"
    assert result["embedding_count"] == count
    assert [row["status"] for row in result["photos"]] == ["accepted"] * count
    assert {path.suffix for path in Path(service.instance_path).rglob("*") if path.is_file()} == {".json"}
    assert "embeddings" not in json.dumps(service.execute("list", {}))


@pytest.mark.parametrize("name", ["", " ", "Reachy", "Ricci", "Richie", "Ritchie", "Rishi"])
def test_invalid_names_never_persist(service: people.PeopleService, photo: dict[str, str], name: str) -> None:
    """Missing names and robot-name variants cannot enrol a person."""
    assert service.execute("enrol", {"name": name, "photos": [photo]})["persisted"] is False
    assert service.identities.list_identities() == []


@pytest.mark.parametrize("faces,status", [(0, "no_face"), (2, "multiple_faces")])
def test_face_count_rejections(service: people.PeopleService, photo: dict[str, str], faces: int, status: str) -> None:
    """Only single-face photos may reach the embedder."""
    detected = service.pipeline.detector.detect.return_value[0]
    service.pipeline.detector.detect.return_value = [detected] * faces
    result = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    assert result["persisted"] is False
    assert result["photos"][0]["status"] == status
    service.pipeline.embedder.embed_face.assert_not_called()


@pytest.mark.parametrize(
    "condition,reason",
    [
        ("blur", "blurred"),
        ("clip", "clipped_at_edge"),
        ("small", "face_too_small"),
        ("confidence", "low_confidence"),
        ("dark", "too_dark"),
        ("bright", "too_bright"),
    ],
)
def test_real_quality_rejections(
    service: people.PeopleService, photo: dict[str, str], condition: str, reason: str
) -> None:
    """Existing quality gates reject poor uploads before any embedding is made."""
    face = service.pipeline.detector.detect.return_value[0]
    if condition in {"blur", "dark", "bright"}:
        frame = np.full((240, 320, 3), {"blur": 120, "dark": 10, "bright": 245}[condition], np.uint8)
        _, encoded = cv2.imencode(".png", frame)
        photo["content"] = base64.b64encode(encoded).decode()
    else:
        bbox = (0, 0, 130, 140) if condition == "clip" else (80, 50, 20, 20) if condition == "small" else face.bbox
        service.pipeline.detector.detect.return_value = [
            DetectedFace(
                bbox,
                0.7 if condition == "confidence" else 0.99,
                face.landmarks,
                320,
                240,
            )
        ]
    result = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    assert result["persisted"] is False
    assert reason in result["photos"][0]["reasons"]
    service.pipeline.embedder.embed_face.assert_not_called()


def test_mixed_uploads_only_persist_accepted(service: people.PeopleService, photo: dict[str, str]) -> None:
    """Rejected samples never enter the gallery while good ones can succeed."""
    result = service.execute("enrol", {"name": "Carol", "photos": [photo, {**photo, "content": "bad"}, photo]})
    assert result["persisted"] is True
    assert result["embedding_count"] == 2
    assert result["photos"][1]["status"] == "invalid_image"


def test_edit_all_details_keeps_identity_bytes(service: people.PeopleService, photo: dict[str, str]) -> None:
    """Profile edits are verified and do not load models or rewrite embeddings."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    identity_bytes = service.identities.path.read_bytes()
    service.pipeline = None
    result = service.execute(
        "edit",
        {
            "person_id": enrolled["person_id"],
            "name": "Caroline",
            "relationship": "Mum",
            "hobbies": "Gardening, cooking",
            "interests": "Jazz",
            "notes": "Likes travel\nTea",
        },
    )
    assert result["persisted"] is True
    profile = service.profiles.get(enrolled["person_id"])
    assert (profile.name, profile.relationship, profile.hobbies, profile.interests, profile.notes) == (
        "Caroline",
        "Mum",
        ["Gardening", "cooking"],
        ["Jazz"],
        ["Likes travel", "Tea"],
    )
    assert service.identities.path.read_bytes() == identity_bytes
    assert service.pipeline is None


def test_add_photos_preserves_existing_samples(service: people.PeopleService, photo: dict[str, str]) -> None:
    """Adding accepted photos appends samples and leaves the profile unchanged."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    profile_bytes = service.profiles.path.read_bytes()
    result = service.execute(
        "add", {"person_id": enrolled["person_id"], "photos": [photo, {**photo, "content": "bad"}]}
    )
    assert result["persisted"] is True
    assert result["embedding_count"] == 2
    assert service.profiles.path.read_bytes() == profile_bytes


def test_model_mismatch_does_not_change_stores(service: people.PeopleService, photo: dict[str, str]) -> None:
    """Incompatible existing model versions cannot silently accept new samples."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    stored = json.loads(service.identities.path.read_text())
    stored["identities"][0]["model_version"] = "old-model"
    service.identities.path.write_text(json.dumps(stored))
    previous = service.identities.path.read_bytes()
    result = service.execute("add", {"person_id": enrolled["person_id"], "photos": [photo]})
    assert result["persisted"] is False
    assert service.identities.path.read_bytes() == previous


@pytest.mark.parametrize("failure", ["identity_write", "profile_write", "verify"])
def test_persistence_failure_rolls_back_new_person(
    service: people.PeopleService, photo: dict[str, str], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Write failures and failed rereads never leave a half-person or success."""
    if failure == "verify":
        monkeypatch.setattr(people, "verify_persisted_enrolment", lambda **kwargs: {"persisted": False})
    else:
        target = service.identities if failure == "identity_write" else service.profiles
        monkeypatch.setattr(
            target,
            "enrol" if failure == "identity_write" else "upsert",
            MagicMock(side_effect=OSError("disk failure")),
        )
    result = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    assert result["persisted"] is False
    assert not service.identities.path.exists()
    assert not service.profiles.path.exists()
    assert not list(Path(service.instance_path).rglob("*.tmp"))


@pytest.mark.parametrize("action", ["edit", "add", "forget"])
def test_failed_existing_mutation_restores_original(
    service: people.PeopleService, photo: dict[str, str], monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """Every failed mutation restores the original identity and profile bytes."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    before = [store.path.read_bytes() for store in (service.identities, service.profiles)]
    if action == "forget":
        monkeypatch.setattr(service.profiles, "delete", MagicMock(side_effect=OSError("disk failure")))
    else:
        monkeypatch.setattr(people, "verify_persisted_enrolment", lambda **kwargs: {"persisted": False})
    result = service.execute(
        action, {"person_id": enrolled["person_id"], "name": "Changed", "photos": [photo], "confirmed": True}
    )
    assert result["persisted"] is False
    assert [store.path.read_bytes() for store in (service.identities, service.profiles)] == before


def test_forget_requires_confirmation_and_verifies_deletion(
    service: people.PeopleService, photo: dict[str, str]
) -> None:
    """Deleting removes both stores' person records only after confirmation."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    body = {"person_id": enrolled["person_id"]}
    assert service.execute("forget", body)["persisted"] is False
    assert service.identities.get(body["person_id"]) is not None
    assert service.execute("forget", {**body, "confirmed": True})["persisted"] is True
    assert service.identities.get(body["person_id"]) is None
    assert service.profiles.get(body["person_id"]) is None
    assert service.execute("forget", {**body, "confirmed": True})["persisted"] is False


@pytest.mark.parametrize("condition", ["mime", "fake_extension", "oversized", "pixels", "corrupt", "base64"])
def test_upload_security(service: people.PeopleService, photo: dict[str, str], condition: str) -> None:
    """File content, MIME, encoded size and decoded dimensions are validated."""
    if condition == "mime":
        photo["type"] = "image/svg+xml"
    elif condition == "fake_extension":
        photo["content"] = base64.b64encode(b"<script>evil</script>").decode()
    elif condition == "oversized":
        photo["content"] = "A" * (12 * 1024 * 1024)
    elif condition == "pixels":
        encoded = bytearray(base64.b64decode(photo["content"]))
        encoded[16:20] = (100_000).to_bytes(4, "big")
        photo["content"] = base64.b64encode(encoded).decode()
    elif condition == "corrupt":
        photo["content"] = photo["content"][:64]
    else:
        photo["content"] = "????"
    assert service.execute("enrol", {"name": "Carol", "photos": [photo]})["persisted"] is False
    assert not service.identities.path.exists()


@pytest.mark.parametrize("count", [0, 11])
def test_photo_count_limit(service: people.PeopleService, photo: dict[str, str], count: int) -> None:
    """Empty and excessive uploads are rejected."""
    assert service.execute("enrol", {"name": "Carol", "photos": [photo] * count})["persisted"] is False


def test_total_upload_limit(service: people.PeopleService, photo: dict[str, str]) -> None:
    """The aggregate payload limit applies before expensive decoding."""
    photo["content"] = "A" * (8 * 1024 * 1024)
    result = service.execute("enrol", {"name": "Carol", "photos": [photo] * 6})
    assert result["persisted"] is False
    service.pipeline.detector.detect.assert_not_called()


def test_jpeg_and_path_traversal_filename(photo: dict[str, str], tmp_path: Path) -> None:
    """A filename never becomes a path; real JPEG bytes decode successfully."""
    frame = people.decode_photo(photo)
    _, encoded = cv2.imencode(".jpg", frame)
    photo.update(type="image/jpeg", content=base64.b64encode(encoded).decode(), name="../../outside.jpg")
    assert people.decode_photo(photo).shape == frame.shape
    assert not list(tmp_path.iterdir())


def test_corrupt_existing_store_is_preserved(service: people.PeopleService, photo: dict[str, str]) -> None:
    """Unreadable memory cannot be mistaken for an empty gallery and overwritten."""
    service.identities.path.parent.mkdir(parents=True)
    service.identities.path.write_text("corrupted")
    assert service.execute("enrol", {"name": "Carol", "photos": [photo]})["persisted"] is False
    assert service.identities.path.read_text() == "corrupted"


def test_concurrent_request_is_bounded(service: people.PeopleService) -> None:
    """Concurrent model work fails promptly rather than queuing images."""
    with people._LOCK:
        assert service.execute("list", {})["persisted"] is False


def test_http_routes_and_origin_guard(
    service: people.PeopleService, photo: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real HTTP upload/list routes reject cross-origin and oversized requests."""
    monkeypatch.setattr(server, "_people", service)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    try:
        connection.request(
            "POST",
            "/api/people/enrol",
            json.dumps({"name": "Carol", "photos": [photo]}),
            {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{httpd.server_port}"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["persisted"] is True
        connection.request("GET", "/api/people", headers={"Origin": f"http://127.0.0.1:{httpd.server_port}"})
        response = connection.getresponse()
        assert len(json.loads(response.read())["people"]) == 1
        connection.request(
            "POST", "/api/people/forget", "{}", {"Content-Type": "application/json", "Origin": "https://evil.test"}
        )
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["persisted"] is False
        connection.request(
            "POST",
            "/api/people/enrol",
            b"",
            {"Content-Type": "application/json", "Content-Length": str(people.MAX_BODY_BYTES + 1)},
        )
        response = connection.getresponse()
        assert response.status == 400
        assert json.loads(response.read()) == {
            "error": "Invalid, oversized or incomplete upload.",
            "persisted": False,
        }
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def test_http_exception_returns_structured_error(
    service: people.PeopleService, photo: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected backend failures must still return JSON instead of dropping the connection."""
    monkeypatch.setattr(server, "_people", service)

    def boom(action: str, body: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(service, "execute", boom)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
    worker = threading.Thread(target=httpd.serve_forever, daemon=True)
    worker.start()
    connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
    try:
        connection.request(
            "POST",
            "/api/people/enrol",
            json.dumps({"name": "Probe", "photos": [photo]}),
            {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{httpd.server_port}"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 500
        assert payload["persisted"] is False
        assert "Face memory could not process" in payload["error"]
        assert service.identities.list_identities() == []
        assert service.profiles.list_profiles() == []
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        worker.join(timeout=2)


def test_combined_base64_budget_rejects_before_model_work(
    service: people.PeopleService, photo: dict[str, str]
) -> None:
    """Client binary totals can look fine while encoded content still exceeds the budget."""
    photo = dict(photo)
    photo["content"] = "A" * (((people.MAX_TOTAL_BYTES + 2) // 3 * 4) + 1)
    result = service.execute("enrol", {"name": "Probe", "photos": [photo]})
    assert result["persisted"] is False
    assert "32 MiB" in str(result.get("error", ""))
    service.pipeline.detector.detect.assert_not_called()
    assert service.identities.list_identities() == []
    assert service.profiles.list_profiles() == []


def test_noop_profile_write_cannot_claim_success(
    service: people.PeopleService, photo: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store returning the old profile cannot authorize an unsaved edit."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    old_profile = service.profiles.get(enrolled["person_id"])
    monkeypatch.setattr(service.profiles, "upsert", lambda *args, **kwargs: old_profile)
    result = service.execute("edit", {"person_id": enrolled["person_id"], "name": "New name"})
    assert result["persisted"] is False
    assert service.profiles.get(enrolled["person_id"]).name == "Carol"


def test_noop_delete_cannot_claim_success(
    service: people.PeopleService, photo: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful return value alone cannot authorize a forget confirmation."""
    enrolled = service.execute("enrol", {"name": "Carol", "photos": [photo]})
    monkeypatch.setattr(service.identities, "delete", lambda person_id: True)
    result = service.execute("forget", {"person_id": enrolled["person_id"], "confirmed": True})
    assert result["persisted"] is False
    assert service.profiles.get(enrolled["person_id"]) is not None


def test_cached_yunet_rejects_nonface_upload(tmp_path: Path, photo: dict[str, str]) -> None:
    """Run the actual installed YuNet model on a generated non-face image."""
    if not all((default_models_dir() / spec.filename).is_file() for spec in (YUNET_SPEC, SFACE_SPEC)):
        pytest.skip("Verified face models are not cached on this machine")
    instance = people.PeopleService(tmp_path)
    result = instance.execute("enrol", {"name": "Test person", "photos": [photo]})
    assert result["persisted"] is False
    assert result["photos"][0]["status"] == "no_face"


def test_cached_sface_embeddings_persist_with_model_metadata(
    service: people.PeopleService, photo: dict[str, str]
) -> None:
    """Exercise actual SFace and persistence with controlled test landmarks."""
    if not (default_models_dir() / SFACE_SPEC.filename).is_file():
        pytest.skip("Verified SFace model is not cached on this machine")
    service.pipeline.embedder = SFaceEmbedder(allow_download=False)
    result = service.execute("enrol", {"name": "Test person", "photos": [photo] * 3})
    assert result["persisted"] is True
    identity = service.identities.get(result["person_id"])
    assert identity.model_id == MODEL_ID_SFACE
    assert identity.model_version == MODEL_VERSION_SFACE
    assert len(identity.embeddings) == 3
    assert all(len(vector) == 128 and np.isfinite(vector).all() for vector in identity.embeddings)
