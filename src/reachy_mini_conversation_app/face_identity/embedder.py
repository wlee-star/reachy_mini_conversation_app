"""SFace alignment and embedding generation."""

from __future__ import annotations
import time
import logging
from typing import Any
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    DetectedFace,
    FaceEmbedding,
    QualityResult,
)
from reachy_mini_conversation_app.face_identity.models import SFACE_SPEC, ensure_model
from reachy_mini_conversation_app.face_identity.quality import assess_face_quality


logger = logging.getLogger(__name__)


class FaceEmbedderUnavailable(RuntimeError):
    """Raised when OpenCV FaceRecognizerSF is unavailable."""


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise FaceEmbedderUnavailable(
            "OpenCV is required for SFace embeddings. Install opencv-python<=5.0 "
            "(do not add OpenCV contrib packages as an app dependency)."
        ) from exc
    if not hasattr(cv2, "FaceRecognizerSF"):
        raise FaceEmbedderUnavailable("cv2.FaceRecognizerSF is missing from this OpenCV build.")
    return cv2


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return an L2-normalized float32 embedding."""
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm <= 1e-12:
        return flat
    return flat / norm


class SFaceEmbedder:
    """Align a YuNet face with SFace and emit a normalized embedding."""

    def __init__(self, model_path: Path | None = None, *, allow_download: bool = False) -> None:
        """Load SFace from a verified local ONNX path."""
        self._cv2 = _require_cv2()
        path = model_path or ensure_model(SFACE_SPEC, allow_download=allow_download)
        self.model_path = Path(path)
        self.model_id = MODEL_ID_SFACE
        self.model_version = MODEL_VERSION_SFACE
        self._recognizer = self._cv2.FaceRecognizerSF.create(str(self.model_path), "")

    def align_crop(self, frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return the SFace-aligned face crop for one detection."""
        face_row = np.asarray(
            [
                face.bbox[0],
                face.bbox[1],
                face.bbox[2],
                face.bbox[3],
                *face.landmarks.as_yunet_row(),
                face.confidence,
            ],
            dtype=np.float32,
        )
        aligned = self._recognizer.alignCrop(frame_bgr, face_row)
        return np.asarray(aligned)

    def embed_aligned(self, aligned_bgr: np.ndarray, quality: QualityResult) -> FaceEmbedding:
        """Feature-extract an already-aligned crop."""
        features = self._recognizer.feature(aligned_bgr)
        vector = l2_normalize(np.asarray(features, dtype=np.float32))
        return FaceEmbedding(
            vector=tuple(float(v) for v in vector.tolist()),
            model_id=self.model_id,
            model_version=self.model_version,
            quality=quality,
            created_at=time.time(),
        )

    def embed_face(
        self,
        frame_bgr: np.ndarray,
        face: DetectedFace,
        *,
        require_usable: bool = True,
    ) -> FaceEmbedding | None:
        """Assess quality, align, and embed one face; return None when rejected."""
        quality = assess_face_quality(frame_bgr, face)
        if require_usable and not quality.usable:
            logger.info("Rejecting face for embedding: %s", ",".join(quality.reasons) or "unusable")
            return None
        aligned = self.align_crop(frame_bgr, face)
        if aligned is None or getattr(aligned, "size", 0) == 0:
            logger.warning("SFace alignCrop returned an empty crop")
            return None
        return self.embed_aligned(aligned, quality)
