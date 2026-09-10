"""Photo-enrolment session: multi-frame scan, consent, head-tracking freeze."""

from __future__ import annotations
import time
import logging
import threading
from typing import Any
from dataclasses import field, dataclass

import numpy as np

from reachy_mini_conversation_app.face_identity.types import FaceEmbedding
from reachy_mini_conversation_app.face_identity.quality import assess_face_quality
from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline


logger = logging.getLogger(__name__)

PHOTO_FRAME_COUNT = 15
PHOTO_FRAME_INTERVAL_S = 0.08
MIN_USABLE_SAMPLES = 3
MAX_KEEP_SAMPLES = 5


@dataclass
class PhotoEnrolmentSession:
    """In-memory photo enrolment state; nothing persists before consent + name."""

    active: bool = False
    phase: str = "idle"  # idle | scanning | await_consent | await_name | await_details | done
    cancel_event: threading.Event = field(default_factory=threading.Event)
    temporary_embeddings: list[FaceEmbedding] = field(default_factory=list)
    tracking_was_enabled: bool | None = None
    last_scan_status: str | None = None
    pending_person_id: str | None = None
    pending_name: str | None = None

    def reset_temporary(self) -> None:
        """Discard temporary embeddings/crops."""
        self.temporary_embeddings.clear()
        self.pending_person_id = None
        self.pending_name = None
        self.last_scan_status = None


class HeadTrackingGuard:
    """Freeze tracking and pause breathing/wobble for photo scan; restore after."""

    _SETTLE_S = 0.15

    def __init__(self, movement_manager: Any) -> None:
        """Bind to a MovementManager-like object."""
        self._movement_manager = movement_manager
        self._prior_enabled: bool | None = None
        self._held_stillness = False

    def __enter__(self) -> HeadTrackingGuard:
        """Capture tracking state, freeze tracking, and hold the body still."""
        holder = getattr(self._movement_manager, "hold_still_for_capture", None)
        if holder is not None:
            try:
                holder()
                self._held_stillness = True
            except Exception as error:
                logger.warning("Failed to hold stillness for photo capture: %s", error)

        getter = getattr(self._movement_manager, "get_head_tracking_enabled", None)
        freezer = getattr(self._movement_manager, "freeze_head_tracking", None)
        if getter is None or freezer is None:
            logger.warning("Movement manager lacks head-tracking freeze helpers; skipping freeze")
            self._prior_enabled = None
        else:
            self._prior_enabled = bool(getter())
            if self._prior_enabled:
                freezer()
        time.sleep(self._SETTLE_S)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Restore tracking state and release capture stillness."""
        restorer = getattr(self._movement_manager, "restore_head_tracking", None)
        if restorer is not None and self._prior_enabled is not None:
            try:
                restorer(self._prior_enabled)
            except Exception as error:
                logger.warning("Failed to restore head tracking after photo scan: %s", error)

        if self._held_stillness:
            releaser = getattr(self._movement_manager, "release_capture_stillness", None)
            if releaser is not None:
                try:
                    releaser()
                except Exception as error:
                    logger.warning("Failed to release photo capture stillness: %s", error)


def collect_photo_embeddings(
    pipeline: FaceIdentityPipeline,
    frames: list[np.ndarray],
) -> dict[str, Any]:
    """Score a rolling window of frames and keep several high-quality embeddings in memory."""
    usable: list[tuple[float, FaceEmbedding]] = []
    multi_face_frames = 0
    no_face_frames = 0
    poor_frames = 0

    for frame in frames:
        faces = pipeline.detect(frame)
        if not faces:
            no_face_frames += 1
            continue
        if len(faces) > 1:
            multi_face_frames += 1
            continue
        face = faces[0]
        quality = assess_face_quality(frame, face)
        if not quality.usable:
            poor_frames += 1
            continue
        embedding = pipeline.embedder.embed_face(frame, face, require_usable=True)
        if embedding is None:
            poor_frames += 1
            continue
        usable.append((quality.score, embedding))

    if multi_face_frames > 0 and not usable:
        return {"status": "multiple_faces", "face_hint": "multiple"}
    if not usable:
        return {
            "status": "no_clear_face",
            "no_face_frames": no_face_frames,
            "poor_frames": poor_frames,
            "multi_face_frames": multi_face_frames,
        }

    usable.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in usable[:MAX_KEEP_SAMPLES]]
    if len(selected) < MIN_USABLE_SAMPLES:
        return {
            "status": "no_clear_face",
            "usable_samples": len(selected),
            "poor_frames": poor_frames,
        }
    return {
        "status": "ready",
        "embedding_count": len(selected),
        "embeddings": selected,
    }


def capture_frames_from_robot(
    reachy_mini: Any,
    *,
    count: int = PHOTO_FRAME_COUNT,
    interval_s: float = PHOTO_FRAME_INTERVAL_S,
    cancel_event: threading.Event | None = None,
) -> list[np.ndarray]:
    """Capture a short frame window via Reachy media.get_frame() (not cv2.VideoCapture)."""
    media = getattr(reachy_mini, "media", None)
    if media is None or getattr(media, "get_frame", None) is None:
        raise RuntimeError("Reachy media.get_frame is unavailable")

    frames: list[np.ndarray] = []
    for _ in range(count):
        if cancel_event is not None and cancel_event.is_set():
            break
        frame = media.get_frame()
        if frame is not None:
            frames.append(np.asarray(frame))
        time.sleep(interval_s)
    return frames
