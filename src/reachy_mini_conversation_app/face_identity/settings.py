"""Face-memory feature flags and match thresholds."""

from __future__ import annotations
import os
import logging


logger = logging.getLogger(__name__)

FACE_MEMORY_ENABLED_ENV = "FACE_MEMORY_ENABLED"
FACE_MEMORY_PHOTO_ENROLMENT_ENABLED_ENV = "FACE_MEMORY_PHOTO_ENROLMENT_ENABLED"
FACE_MEMORY_LIVE_RECOGNITION_ENABLED_ENV = "FACE_MEMORY_LIVE_RECOGNITION_ENABLED"
FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED_ENV = "FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED"
FACE_MEMORY_MATCH_THRESHOLD_ENV = "FACE_MEMORY_MATCH_THRESHOLD"
FACE_MEMORY_MATCH_MARGIN_ENV = "FACE_MEMORY_MATCH_MARGIN"

# OpenCV Zoo SFace cosine threshold recommendation is ~0.363; we stay slightly stricter.
DEFAULT_MATCH_THRESHOLD = 0.40
DEFAULT_MATCH_MARGIN = 0.05


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean value for %s=%r, using default=%s", name, raw, default)
    return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using default %s.", name, raw, default)
        return default


def face_memory_enabled() -> bool:
    """Return whether face-memory features may run (default off)."""
    return _env_flag(FACE_MEMORY_ENABLED_ENV, default=False)


def face_memory_photo_enrolment_enabled() -> bool:
    """Return whether conversational camera photo enrolment may run (default off)."""
    return face_memory_enabled() and _env_flag(FACE_MEMORY_PHOTO_ENROLMENT_ENABLED_ENV, default=False)


def face_memory_live_recognition_enabled() -> bool:
    """Return whether continuous live recognition is allowed (default off)."""
    return face_memory_enabled() and _env_flag(FACE_MEMORY_LIVE_RECOGNITION_ENABLED_ENV, default=False)


def face_memory_on_demand_recognition_enabled() -> bool:
    """Return whether explicit 'who is this?' recognition may run (default on when face memory is on)."""
    return face_memory_enabled() and _env_flag(FACE_MEMORY_ON_DEMAND_RECOGNITION_ENABLED_ENV, default=True)


def match_threshold() -> float:
    """Minimum cosine similarity required to accept a known identity."""
    return _env_float(FACE_MEMORY_MATCH_THRESHOLD_ENV, DEFAULT_MATCH_THRESHOLD)


def match_margin() -> float:
    """Minimum gap between best and second-best identity scores."""
    return _env_float(FACE_MEMORY_MATCH_MARGIN_ENV, DEFAULT_MATCH_MARGIN)
