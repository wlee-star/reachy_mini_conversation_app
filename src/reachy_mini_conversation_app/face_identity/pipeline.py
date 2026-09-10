"""Offline image pipeline: detect → quality → embed → match."""

from __future__ import annotations
import logging
from typing import Any
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.face_identity.types import MatchResult, DetectedFace, FaceEmbedding, QualityResult
from reachy_mini_conversation_app.face_identity.matcher import match_embedding
from reachy_mini_conversation_app.face_identity.quality import assess_face_quality
from reachy_mini_conversation_app.face_identity.detector import YuNetDetector
from reachy_mini_conversation_app.face_identity.embedder import SFaceEmbedder
from reachy_mini_conversation_app.face_identity.profile_store import PersonProfileStore
from reachy_mini_conversation_app.face_identity.identity_store import IdentityStore
from reachy_mini_conversation_app.face_identity.speech_authority import verify_persisted_enrolment


logger = logging.getLogger(__name__)


def load_bgr_image(path: str | Path) -> np.ndarray:
    """Load a JPEG/PNG as BGR; raise on failure."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


class FaceIdentityPipeline:
    """High-level offline enrolment and recognition helpers."""

    def __init__(
        self,
        instance_path: str | Path | None = None,
        *,
        allow_download: bool = False,
        detector: YuNetDetector | None = None,
        embedder: SFaceEmbedder | None = None,
    ) -> None:
        """Construct detector/embedder/stores for one app instance."""
        self.instance_path = instance_path
        self.detector = detector or YuNetDetector(allow_download=allow_download)
        self.embedder = embedder or SFaceEmbedder(allow_download=allow_download)
        self.identities = IdentityStore(instance_path)
        self.profiles = PersonProfileStore(instance_path)

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a frame."""
        return self.detector.detect(frame_bgr)

    def quality_for(self, frame_bgr: np.ndarray, face: DetectedFace) -> QualityResult:
        """Assess quality for one face."""
        return assess_face_quality(frame_bgr, face)

    def embed_single_face_image(self, frame_bgr: np.ndarray) -> tuple[str, FaceEmbedding | None, dict[str, Any]]:
        """Embed exactly one usable face; reject zero/multiple/poor faces."""
        faces = self.detect(frame_bgr)
        meta: dict[str, Any] = {"face_count": len(faces)}
        if not faces:
            return "no_face", None, meta
        if len(faces) > 1:
            return "multiple_faces", None, meta
        face = faces[0]
        quality = self.quality_for(frame_bgr, face)
        meta["quality"] = quality.to_dict()
        if not quality.usable:
            return "unusable", None, meta
        embedding = self.embedder.embed_face(frame_bgr, face, require_usable=True)
        if embedding is None:
            return "unusable", None, meta
        return "ok", embedding, meta

    def enrol_from_images(
        self,
        image_paths: list[str | Path],
        *,
        name: str,
        details: str | None = None,
        person_id: str | None = None,
    ) -> dict[str, Any]:
        """Enrol a person from several images; persist embeddings only after success."""
        accepted: list[FaceEmbedding] = []
        rejected: list[dict[str, Any]] = []
        for path in image_paths:
            frame = load_bgr_image(path)
            status, embedding, meta = self.embed_single_face_image(frame)
            if status != "ok" or embedding is None:
                rejected.append({"path": str(path), "status": status, **meta})
                continue
            accepted.append(embedding)

        if len(accepted) < 2:
            return {
                "status": "rejected",
                "error": "need_at_least_two_usable_faces",
                "accepted": len(accepted),
                "rejected": rejected,
            }

        record = self.identities.enrol(accepted, person_id=person_id)
        profile = self.profiles.upsert(
            record.person_id,
            name,
            append_note=details.strip() if details else None,
        )
        verified = verify_persisted_enrolment(
            identities=self.identities,
            profiles=self.profiles,
            person_id=record.person_id,
            name=profile.name,
            min_embeddings=2,
        )
        if verified.get("persisted") is not True:
            return {**verified, "rejected": rejected}
        return {
            "status": "enrolled",
            "person_id": record.person_id,
            "name": profile.name,
            "embedding_count": len(record.embeddings),
            "persisted": True,
            "rejected": rejected,
        }

    def recognize_image(self, image_path: str | Path) -> dict[str, Any]:
        """Recognize a single local image without logging embeddings."""
        frame = load_bgr_image(image_path)
        return self.recognize_frame(frame)

    def recognize_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """Recognize faces in one BGR frame."""
        faces = self.detect(frame_bgr)
        if not faces:
            return MatchResult(status="no_face", face_count=0).to_dict()
        if len(faces) > 1:
            return MatchResult(status="multiple_faces", face_count=len(faces)).to_dict()

        face = faces[0]
        quality = self.quality_for(frame_bgr, face)
        if not quality.usable:
            return MatchResult(
                status="unusable",
                face_count=1,
                quality=quality,
                reasons=quality.reasons,
            ).to_dict()

        embedding = self.embedder.embed_face(frame_bgr, face, require_usable=True)
        if embedding is None:
            return MatchResult(status="unusable", face_count=1, quality=quality, reasons=("embed_failed",)).to_dict()

        result = match_embedding(
            embedding.vector,
            self.identities.list_identities(),
            model_id=self.embedder.model_id,
            model_version=self.embedder.model_version,
        )
        result = MatchResult(
            status=result.status,
            person_id=result.person_id,
            similarity=result.similarity,
            second_best_similarity=result.second_best_similarity,
            margin=result.margin,
            face_count=1,
            quality=quality,
            reasons=result.reasons,
        )
        payload = result.to_dict()
        if result.person_id:
            profile = self.profiles.get(result.person_id)
            if profile is not None:
                payload["name"] = profile.name
                if profile.relationship:
                    payload["relationship"] = profile.relationship
        return payload

    def forget_person(self, *, person_id: str | None = None, name: str | None = None) -> dict[str, Any]:
        """Delete identity embeddings and profile for one person."""
        profile = None
        if person_id:
            profile = self.profiles.get(person_id)
        elif name:
            profile = self.profiles.get_by_name(name)
            person_id = profile.person_id if profile else None
        if not person_id:
            return {"status": "not_found"}

        removed_identity = self.identities.delete(person_id)
        removed_profile = self.profiles.delete(person_id)
        return {
            "status": "forgotten" if (removed_identity or removed_profile) else "not_found",
            "person_id": person_id,
            "name": profile.name if profile else None,
            "removed_identity": removed_identity,
            "removed_profile": removed_profile,
        }

    def list_remembered(self) -> list[dict[str, Any]]:
        """List remembered people without embeddings."""
        identities = {record.person_id: record for record in self.identities.list_identities()}
        rows: list[dict[str, Any]] = []
        for profile in self.profiles.list_profiles():
            record = identities.get(profile.person_id)
            rows.append(
                {
                    "person_id": profile.person_id,
                    "name": profile.name,
                    "embedding_count": len(record.embeddings) if record else 0,
                    "hobbies": profile.hobbies,
                    "interests": profile.interests,
                    "notes": profile.notes,
                }
            )
        for person_id, record in identities.items():
            if any(row["person_id"] == person_id for row in rows):
                continue
            rows.append(
                {
                    "person_id": person_id,
                    "name": None,
                    "embedding_count": len(record.embeddings),
                    "hobbies": [],
                    "interests": [],
                    "notes": [],
                }
            )
        return rows
