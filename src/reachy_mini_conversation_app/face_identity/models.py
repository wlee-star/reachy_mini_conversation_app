"""ONNX model cache management for YuNet and SFace."""

from __future__ import annotations
import os
import hashlib
import logging
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass

from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_ID_YUNET,
    MODEL_VERSION_SFACE,
    MODEL_VERSION_YUNET,
)


logger = logging.getLogger(__name__)

# OpenCV Zoo Git LFS oids (sha256 of the real weight files).
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

YUNET_FILENAME = f"{MODEL_VERSION_YUNET}.onnx"
SFACE_FILENAME = f"{MODEL_VERSION_SFACE}.onnx"

# Prefer Hugging Face resolve URLs (real binaries, not Git LFS pointers).
YUNET_URLS = (
    f"https://huggingface.co/opencv/face_detection_yunet/resolve/main/{YUNET_FILENAME}",
    f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_FILENAME}",
)
SFACE_URLS = (
    f"https://huggingface.co/opencv/face_recognition_sface/resolve/main/{SFACE_FILENAME}",
    f"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/{SFACE_FILENAME}",
)


@dataclass(frozen=True)
class ModelSpec:
    """One cached ONNX model with integrity metadata."""

    model_id: str
    model_version: str
    filename: str
    sha256: str
    urls: tuple[str, ...]


YUNET_SPEC = ModelSpec(
    model_id=MODEL_ID_YUNET,
    model_version=MODEL_VERSION_YUNET,
    filename=YUNET_FILENAME,
    sha256=YUNET_SHA256,
    urls=YUNET_URLS,
)
SFACE_SPEC = ModelSpec(
    model_id=MODEL_ID_SFACE,
    model_version=MODEL_VERSION_SFACE,
    filename=SFACE_FILENAME,
    sha256=SFACE_SHA256,
    urls=SFACE_URLS,
)


class ModelError(RuntimeError):
    """Raised when a required face model is missing or corrupt."""


def default_models_dir() -> Path:
    """Return the user cache directory for face ONNX models."""
    override = os.getenv("FACE_MEMORY_MODELS_DIR")
    if override and override.strip():
        return Path(override.strip()).expanduser()

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / "reachy_mini_conversation_app" / "face_models"

    xdg = os.getenv("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "reachy_mini_conversation_app" / "face_models"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def verify_model(path: Path, spec: ModelSpec) -> None:
    """Raise ModelError when the cached file is missing, an LFS pointer, or corrupt."""
    if not path.is_file():
        raise ModelError(f"Missing face model {spec.filename} at {path}")
    if _is_lfs_pointer(path):
        raise ModelError(
            f"Face model {spec.filename} looks like a Git LFS pointer, not weights. "
            f"Re-download with allow_download=True or replace the file at {path}."
        )
    digest = sha256_file(path)
    if digest.lower() != spec.sha256.lower():
        raise ModelError(f"Face model checksum mismatch for {spec.filename}: expected {spec.sha256}, got {digest}")


def _download_model(path: Path, spec: ModelSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".partial")
    last_error: Exception | None = None
    for url in spec.urls:
        try:
            logger.info("Downloading face model %s from %s", spec.filename, url)
            urllib.request.urlretrieve(url, tmp_path)
            verify_model(tmp_path, spec)
            tmp_path.replace(path)
            logger.info("Cached face model %s at %s", spec.filename, path)
            return
        except (OSError, urllib.error.URLError, ModelError) as exc:
            last_error = exc
            logger.warning("Failed to fetch %s from %s: %s", spec.filename, url, exc)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    raise ModelError(f"Could not download face model {spec.filename}: {last_error}")


def ensure_model(spec: ModelSpec, models_dir: Path | None = None, *, allow_download: bool = False) -> Path:
    """Return a verified local model path; download only when explicitly allowed."""
    root = models_dir or default_models_dir()
    path = root / spec.filename
    try:
        verify_model(path, spec)
        return path
    except ModelError as missing:
        if not allow_download:
            raise ModelError(
                f"{missing} Set FACE_MEMORY_MODELS_DIR or call ensure_models(allow_download=True) once."
            ) from missing
        _download_model(path, spec)
        verify_model(path, spec)
        return path


def ensure_models(models_dir: Path | None = None, *, allow_download: bool = False) -> tuple[Path, Path]:
    """Ensure YuNet and SFace models are present and verified."""
    yunet = ensure_model(YUNET_SPEC, models_dir, allow_download=allow_download)
    sface = ensure_model(SFACE_SPEC, models_dir, allow_download=allow_download)
    return yunet, sface
