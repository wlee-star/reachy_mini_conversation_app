"""Camera-independent uploaded-photo enrolment using the existing face pipeline."""

import re
import json
import math
import uuid
import base64
import logging
import threading
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np
import numpy.typing as npt

from reachy_mini_conversation_app.face_identity.types import FaceEmbedding
from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline
from reachy_mini_conversation_app.face_identity.profile_store import PersonProfileStore
from reachy_mini_conversation_app.face_identity.identity_store import IdentityStore
from reachy_mini_conversation_app.face_identity.speech_authority import (
    is_robot_name_variant,
    verify_persisted_enrolment,
)


logger = logging.getLogger(__name__)
MAX_PHOTOS = 10
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
# Binary photo caps stay at 32 MiB; body limit includes base64 (~4/3) plus JSON wrappers.
MAX_BODY_BYTES = 48 * 1024 * 1024
MAX_PIXELS = 16_000_000
_LOCK = threading.Lock()


def decode_photo(photo: object) -> npt.NDArray[np.uint8]:
    """Decode bounded JPEG/PNG content without trusting or using its filename."""
    if not isinstance(photo, dict):
        raise ValueError("Invalid photo")
    encoded = photo.get("content")
    mime = photo.get("type")
    if mime not in {"image/jpeg", "image/png"}:
        raise ValueError("Use JPG or PNG photos")
    if not isinstance(encoded, str) or len(encoded) > (MAX_FILE_BYTES + 2) // 3 * 4:
        raise ValueError("Each photo must be at most 8 MiB")
    content = base64.b64decode(encoded, validate=True)
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Each photo must be at most 8 MiB")
    width = height = 0
    if mime == "image/png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(content) < 33 or content[12:16] != b"IHDR":
            raise ValueError("Corrupted PNG")
        width = int.from_bytes(content[16:20], "big")
        height = int.from_bytes(content[20:24], "big")
    elif mime == "image/jpeg" and content.startswith(b"\xff\xd8"):
        offset = 2
        while offset + 4 <= len(content):
            if content[offset] != 255:
                break
            while offset < len(content) and content[offset] == 255:
                offset += 1
            if offset >= len(content):
                break
            marker = content[offset]
            offset += 1
            if marker in {0xD9, 0xDA}:
                break
            length = int.from_bytes(content[offset : offset + 2], "big")
            if length < 2 or offset + length > len(content):
                break
            if marker in {0xC0, 0xC1, 0xC2} and length >= 8:
                height = int.from_bytes(content[offset + 3 : offset + 5], "big")
                width = int.from_bytes(content[offset + 5 : offset + 7], "big")
                break
            offset += length
    if not width or not height:
        raise ValueError("Unsupported or corrupted image content")
    if width * height > MAX_PIXELS or max(width, height) > 8192:
        raise ValueError("Photo exceeds 16 megapixels or 8192 pixels per side")
    frame = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Corrupted image")
    if frame.shape[0] * frame.shape[1] > MAX_PIXELS:
        raise ValueError("Decoded photo is too large")
    return np.asarray(frame, dtype=np.uint8)


class PeopleService:
    """Serialize dashboard operations and verify both file stores before success."""

    def __init__(self, instance_path: str | Path | None = None) -> None:
        """Use the same default instance storage as the PC conversation app."""
        self.identities = IdentityStore(instance_path)
        self.profiles = PersonProfileStore(instance_path)
        self.instance_path = instance_path
        self.pipeline: FaceIdentityPipeline | None = None
        self.recovery_required = False

    def execute(self, action: str, body: dict[str, object]) -> dict[str, object]:
        """Run an on-demand operation; never access robot media or enable recognition."""
        if not _LOCK.acquire(blocking=False):
            return {"error": "Face memory is busy. Please try again.", "persisted": False}
        snapshots: dict[Path, bytes | None] = {}
        photos: list[dict[str, object]] = []
        stage = "reading memory"
        mutated = False
        try:
            if self.recovery_required and action != "list":
                raise ValueError("Memory recovery is required before further changes")
            for path, key in ((self.identities.path, "identities"), (self.profiles.path, "profiles")):
                original = path.read_bytes() if path.exists() else None
                if original is not None:
                    stored = json.loads(original)
                    if not isinstance(stored, dict) or not isinstance(stored.get(key), list):
                        raise ValueError("Invalid memory store; recover it before editing")
                snapshots[path] = original
            for path, key, count in (
                (self.identities.path, "identities", len(self.identities.list_identities())),
                (self.profiles.path, "profiles", len(self.profiles.list_profiles())),
            ):
                original = snapshots[path]
                if original and len(json.loads(original)[key]) != count:
                    raise ValueError("Memory contains invalid records; recover it before editing")
            if action == "list":
                identities = {record.person_id: record for record in self.identities.list_identities()}
                return {
                    "people": [
                        {
                            **asdict(profile),
                            "embedding_count": len(identities[profile.person_id].embeddings)
                            if profile.person_id in identities
                            else 0,
                        }
                        for profile in self.profiles.list_profiles()
                    ]
                }
            person_id = body.get("person_id")
            if action not in {"enrol", "edit", "add", "forget"}:
                raise ValueError("Unknown operation")
            if action != "enrol":
                if not isinstance(person_id, str) or not re.fullmatch(r"person_[a-zA-Z0-9_-]+", person_id):
                    raise ValueError("Invalid person ID")
                profile = self.profiles.get(person_id)
                identity = self.identities.get(person_id)
                if profile is None or identity is None:
                    raise ValueError("Person not found or memory is incomplete")
            else:
                person_id = "person_" + uuid.uuid4().hex
            assert isinstance(person_id, str)
            if action == "forget":
                if body.get("confirmed") is not True:
                    raise ValueError("Confirm forgetting this person first")
                stage = "forgetting person"
                mutated = True
                self.identities.delete(person_id)
                self.profiles.delete(person_id)
                if self.identities.get(person_id) is not None or self.profiles.get(person_id) is not None:
                    raise OSError("Deletion verification failed")
                return {"status": "forgotten", "person_id": person_id, "persisted": True}
            fields: dict[str, str] = {}
            if action in {"enrol", "edit"}:
                for key in ("name", "relationship", "hobbies", "interests", "notes"):
                    value = body.get(key, "")
                    if not isinstance(value, str) or len(value) > (120 if key == "name" else 4000):
                        raise ValueError("Profile fields are invalid or too long")
                    fields[key] = value.strip()
                if not fields["name"]:
                    raise ValueError("Enter a name")
                if is_robot_name_variant(fields["name"]):
                    raise ValueError("That sounds like Reachy's name. Enter the person's name.")
            accepted: list[FaceEmbedding] = []
            if action in {"enrol", "add"}:
                uploads = body.get("photos")
                if not isinstance(uploads, list) or not 1 <= len(uploads) <= MAX_PHOTOS:
                    raise ValueError("Choose 1–10 photos")
                total = sum(
                    len(item.get("content", ""))
                    for item in uploads
                    if isinstance(item, dict) and isinstance(item.get("content"), str)
                )
                if total > (MAX_TOTAL_BYTES + 2) // 3 * 4:
                    raise ValueError("Combined photos must be at most 32 MiB")
                stage = "loading face models"
                if self.pipeline is None:
                    self.pipeline = FaceIdentityPipeline(self.instance_path, allow_download=False)
                stage = "validating photos"
                for index, upload in enumerate(uploads):
                    try:
                        frame = decode_photo(upload)
                        status, embedding, metadata = self.pipeline.embed_single_face_image(frame)
                        quality = metadata.get("quality", {})
                        reasons = quality.get("reasons", []) if isinstance(quality, dict) else []
                        if embedding is not None:
                            if (
                                len(embedding.vector) != 128
                                or not all(math.isfinite(v) for v in embedding.vector)
                                or not embedding.quality.usable
                                or sum(v * v for v in embedding.vector) < 0.5
                            ):
                                raise ValueError("Invalid face sample")
                            accepted.append(embedding)
                            status = "accepted"
                        photos.append({"index": index, "status": status, "reasons": reasons})
                        del frame
                    except (ValueError, cv2.error) as exc:
                        logger.warning("Uploaded photo %s rejected: %s", index, type(exc).__name__)
                        photos.append(
                            {
                                "index": index,
                                "status": "invalid_image",
                                "message": str(exc) if isinstance(exc, ValueError) else "Could not decode photo",
                            }
                        )
                if not accepted:
                    return {"error": "No usable photos. Check each photo below.", "photos": photos, "persisted": False}
            stage = "saving identity and profile"
            mutated = True
            previous = self.identities.get(person_id)
            expected_count = (len(previous.embeddings) if previous else 0) + len(accepted)
            if accepted:
                self.identities.enrol(accepted, person_id=person_id)
            if fields:
                saved_profile = self.profiles.upsert(
                    person_id,
                    fields["name"],
                    relationship=fields["relationship"],
                    hobbies=fields["hobbies"].split(","),
                    interests=fields["interests"].split(","),
                    notes=fields["notes"].splitlines(),
                )
                if self.profiles.get(person_id) != saved_profile:
                    raise OSError("Profile verification failed")
                if (
                    saved_profile.name != fields["name"]
                    or saved_profile.relationship != fields["relationship"]
                    or saved_profile.hobbies != [item.strip() for item in fields["hobbies"].split(",") if item.strip()]
                    or saved_profile.interests
                    != [item.strip() for item in fields["interests"].split(",") if item.strip()]
                    or saved_profile.notes != [item.strip() for item in fields["notes"].splitlines() if item.strip()]
                ):
                    raise OSError("Requested profile changes were not saved")
            stage = "verifying saved memory"
            profile = self.profiles.get(person_id)
            verified = verify_persisted_enrolment(
                identities=self.identities,
                profiles=self.profiles,
                person_id=person_id,
                name=profile.name if profile else "",
                min_embeddings=expected_count,
            )
            current = self.identities.get(person_id)
            if verified.get("persisted") is not True or current is None or len(current.embeddings) != expected_count:
                raise OSError("Memory verification failed")
            if previous and current.embeddings[: len(previous.embeddings)] != previous.embeddings:
                raise OSError("Existing samples changed")
            if accepted and current.embeddings[-len(accepted) :] != [list(sample.vector) for sample in accepted]:
                raise OSError("New samples could not be verified")
            return {
                "status": "enrolled" if action == "enrol" else "updated",
                "person_id": person_id,
                "name": profile.name if profile else "",
                "embedding_count": expected_count,
                "persisted": True,
                "photos": photos,
            }
        except Exception as exc:
            logger.exception("Dashboard face memory failed while %s", stage)
            rollback_failed = False
            for path, original in snapshots.items():
                if not mutated:
                    break
                try:
                    if original is None:
                        path.unlink(missing_ok=True)
                    elif not path.exists() or path.read_bytes() != original:
                        replacement = path.with_suffix(".rollback.tmp")
                        try:
                            replacement.write_bytes(original)
                            replacement.replace(path)
                        finally:
                            replacement.unlink(missing_ok=True)
                except OSError:
                    rollback_failed = True
                    logger.exception("Could not roll back face store %s", path.name)
            message = str(exc) if isinstance(exc, ValueError) else "Could not finish " + stage + "."
            if rollback_failed:
                self.recovery_required = True
                message += " Memory recovery is required; do not retry until the stores are checked."
            return {"error": message, "stage": stage, "persisted": False, "photos": photos}
        finally:
            _LOCK.release()
