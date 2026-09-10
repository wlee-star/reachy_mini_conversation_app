"""PC-side YuNet face detector (OpenCV Zoo ONNX)."""

from __future__ import annotations
import logging
from typing import Any
from pathlib import Path

import numpy as np

from reachy_mini_conversation_app.face_identity.types import DetectedFace, FaceLandmarks
from reachy_mini_conversation_app.face_identity.models import YUNET_SPEC, ensure_model


logger = logging.getLogger(__name__)


class FaceDetectorUnavailable(RuntimeError):
    """Raised when OpenCV FaceDetectorYN is unavailable."""


def _require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise FaceDetectorUnavailable(
            "OpenCV is required for YuNet detection. Install opencv-python<=5.0 "
            "(do not add OpenCV contrib packages as an app dependency)."
        ) from exc
    if not hasattr(cv2, "FaceDetectorYN"):
        raise FaceDetectorUnavailable("cv2.FaceDetectorYN is missing from this OpenCV build.")
    return cv2


class YuNetDetector:
    """Detect zero or more faces; never collapses to a single 'best' face."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        allow_download: bool = False,
    ) -> None:
        """Load YuNet from a verified local ONNX path."""
        self._cv2 = _require_cv2()
        path = model_path or ensure_model(YUNET_SPEC, allow_download=allow_download)
        self.model_path = Path(path)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector = self._cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a BGR uint8 image."""
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be an HxWx3 BGR image")
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _retval, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []

        detected: list[DetectedFace] = []
        for row in faces:
            values = [float(v) for v in row]
            x, y, w, h = values[0], values[1], values[2], values[3]
            confidence = values[14] if len(values) > 14 else values[-1]
            landmarks = FaceLandmarks(
                right_eye=(values[4], values[5]),
                left_eye=(values[6], values[7]),
                nose=(values[8], values[9]),
                right_mouth=(values[10], values[11]),
                left_mouth=(values[12], values[13]),
            )
            detected.append(
                DetectedFace(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    landmarks=landmarks,
                    frame_width=int(width),
                    frame_height=int(height),
                )
            )
        return detected
