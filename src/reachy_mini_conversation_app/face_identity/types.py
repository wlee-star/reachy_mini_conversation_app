"""Shared face-identity data types."""

from typing import Any
from dataclasses import field, dataclass


IDENTITY_SCHEMA_VERSION = 1
PROFILE_SCHEMA_VERSION = 1
MODEL_ID_YUNET = "yunet"
MODEL_VERSION_YUNET = "face_detection_yunet_2023mar"
MODEL_ID_SFACE = "sface"
MODEL_VERSION_SFACE = "face_recognition_sface_2021dec"


@dataclass(frozen=True)
class FaceLandmarks:
    """Five facial landmarks used for SFace alignment (x, y) pixels."""

    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose: tuple[float, float]
    right_mouth: tuple[float, float]
    left_mouth: tuple[float, float]

    def as_yunet_row(self) -> list[float]:
        """Return landmark coordinates in YuNet/SFace flat order."""
        return [
            self.right_eye[0],
            self.right_eye[1],
            self.left_eye[0],
            self.left_eye[1],
            self.nose[0],
            self.nose[1],
            self.right_mouth[0],
            self.right_mouth[1],
            self.left_mouth[0],
            self.left_mouth[1],
        ]


@dataclass(frozen=True)
class DetectedFace:
    """One face detection; detector does not pick a 'best' face."""

    bbox: tuple[float, float, float, float]  # x, y, w, h
    confidence: float
    landmarks: FaceLandmarks
    frame_width: int
    frame_height: int


@dataclass(frozen=True)
class QualityResult:
    """Structured face-quality gate result."""

    usable: bool
    score: float
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly quality payload."""
        return {"usable": self.usable, "score": self.score, "reasons": list(self.reasons)}


@dataclass(frozen=True)
class FaceEmbedding:
    """Normalized SFace embedding with model provenance."""

    vector: tuple[float, ...]
    model_id: str
    model_version: str
    quality: QualityResult
    created_at: float


@dataclass(frozen=True)
class MatchCandidate:
    """Best score against one enrolled person."""

    person_id: str
    similarity: float


@dataclass(frozen=True)
class MatchResult:
    """Conservative identity match outcome."""

    status: str  # known | unknown | no_face | multiple_faces | unusable | model_mismatch
    person_id: str | None = None
    similarity: float | None = None
    second_best_similarity: float | None = None
    margin: float | None = None
    face_count: int = 0
    quality: QualityResult | None = None
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly match payload without biometric vectors."""
        payload: dict[str, Any] = {
            "status": self.status,
            "person_id": self.person_id,
            "similarity": self.similarity,
            "second_best_similarity": self.second_best_similarity,
            "margin": self.margin,
            "face_count": self.face_count,
            "reasons": list(self.reasons),
        }
        if self.quality is not None:
            payload["quality"] = self.quality.to_dict()
        return payload


@dataclass
class IdentityRecord:
    """Persisted face identity for one person_id."""

    person_id: str
    model_id: str
    model_version: str
    embeddings: list[list[float]] = field(default_factory=list)
    qualities: list[dict[str, Any]] = field(default_factory=list)
    enrolled_at: list[float] = field(default_factory=list)
    schema_version: int = IDENTITY_SCHEMA_VERSION


@dataclass
class PersonProfile:
    """Conversational profile keyed by the same person_id as face identity."""

    person_id: str
    name: str
    hobbies: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    updated_at: float = 0.0
    schema_version: int = PROFILE_SCHEMA_VERSION
    relationship: str = ""
