"""Personal face memory: YuNet detection + SFace embeddings (PC-side)."""

from reachy_mini_conversation_app.face_identity.models import ensure_models
from reachy_mini_conversation_app.face_identity.service import FaceMemoryService
from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline
from reachy_mini_conversation_app.face_identity.settings import (
    face_memory_enabled,
    face_memory_photo_enrolment_enabled,
    face_memory_live_recognition_enabled,
    face_memory_on_demand_recognition_enabled,
)


__all__ = [
    "FaceIdentityPipeline",
    "FaceMemoryService",
    "ensure_models",
    "face_memory_enabled",
    "face_memory_live_recognition_enabled",
    "face_memory_on_demand_recognition_enabled",
    "face_memory_photo_enrolment_enabled",
]
