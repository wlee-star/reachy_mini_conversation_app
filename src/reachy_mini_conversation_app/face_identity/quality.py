"""Lightweight face-quality gating before embedding."""

from __future__ import annotations
import math

import numpy as np

from reachy_mini_conversation_app.face_identity.types import DetectedFace, QualityResult


# Tunable gates — documented defaults, overridable by callers.
MIN_CONFIDENCE = 0.75
MIN_RELATIVE_AREA = 0.02
MIN_ABSOLUTE_AREA = 40 * 40
MAX_EDGE_CLIP_RATIO = 0.02
MIN_LAPLACIAN_VAR = 45.0
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 220.0
MAX_EYE_ROLL_DEG = 35.0


def _bbox_clips_edge(face: DetectedFace, margin_ratio: float = MAX_EDGE_CLIP_RATIO) -> bool:
    x, y, w, h = face.bbox
    mx = face.frame_width * margin_ratio
    my = face.frame_height * margin_ratio
    return x < mx or y < my or (x + w) > (face.frame_width - mx) or (y + h) > (face.frame_height - my)


def _crop_bgr(frame_bgr: np.ndarray, face: DetectedFace) -> np.ndarray | None:
    x, y, w, h = face.bbox
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(face.frame_width, int(math.ceil(x + w)))
    y1 = min(face.frame_height, int(math.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame_bgr[y0:y1, x0:x1]


def _laplacian_variance(gray: np.ndarray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _eye_roll_degrees(face: DetectedFace) -> float:
    rx, ry = face.landmarks.right_eye
    lx, ly = face.landmarks.left_eye
    return abs(math.degrees(math.atan2(ly - ry, lx - rx)))


def assess_face_quality(frame_bgr: np.ndarray, face: DetectedFace) -> QualityResult:
    """Return structured quality for one detected face."""
    reasons: list[str] = []
    score = 1.0

    if face.confidence < MIN_CONFIDENCE:
        reasons.append("low_confidence")
        score *= max(0.1, face.confidence / MIN_CONFIDENCE)

    area = max(0.0, face.bbox[2]) * max(0.0, face.bbox[3])
    frame_area = max(1, face.frame_width * face.frame_height)
    relative_area = area / frame_area
    if area < MIN_ABSOLUTE_AREA or relative_area < MIN_RELATIVE_AREA:
        reasons.append("face_too_small")
        score *= 0.4

    if _bbox_clips_edge(face):
        reasons.append("clipped_at_edge")
        score *= 0.5

    crop = _crop_bgr(frame_bgr, face)
    if crop is None or crop.size == 0:
        reasons.append("invalid_crop")
        return QualityResult(usable=False, score=0.0, reasons=tuple(reasons))

    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = _laplacian_variance(gray)
    if blur < MIN_LAPLACIAN_VAR:
        reasons.append("blurred")
        score *= min(1.0, blur / MIN_LAPLACIAN_VAR)

    brightness = float(np.mean(gray))
    if brightness < MIN_BRIGHTNESS:
        reasons.append("too_dark")
        score *= 0.6
    elif brightness > MAX_BRIGHTNESS:
        reasons.append("too_bright")
        score *= 0.6

    roll = _eye_roll_degrees(face)
    if roll > MAX_EYE_ROLL_DEG:
        reasons.append("extreme_pose")
        score *= 0.5

    score = float(max(0.0, min(1.0, score)))
    usable = not reasons
    if not usable:
        # Still allow a soft score for ranking frames even when rejected.
        score = min(score, 0.49)
    return QualityResult(usable=usable, score=score, reasons=tuple(reasons))
