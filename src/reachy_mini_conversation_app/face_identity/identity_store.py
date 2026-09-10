"""Persistent face-identity store (embeddings only; no display names)."""

from __future__ import annotations
import os
import json
import time
import logging
import threading
from pathlib import Path
from collections.abc import Sequence

from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    IDENTITY_SCHEMA_VERSION,
    FaceEmbedding,
    IdentityRecord,
)


logger = logging.getLogger(__name__)

IDENTITY_FILENAME = "face_identities.v1.json"
_STORE_LOCK = threading.Lock()


def identity_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the face-identity JSON path for this app instance."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / "face_memory" / IDENTITY_FILENAME

    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    if os.name == "nt":
        data_root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return data_root / "reachy_mini_conversation_app" / "face_memory" / IDENTITY_FILENAME


def _next_person_id(existing: Sequence[IdentityRecord]) -> str:
    used = set()
    for record in existing:
        if record.person_id.startswith("person_"):
            suffix = record.person_id.removeprefix("person_")
            if suffix.isdigit():
                used.add(int(suffix))
    next_index = 1
    while next_index in used:
        next_index += 1
    return f"person_{next_index:04d}"


def _record_from_json(value: object) -> IdentityRecord | None:
    if not isinstance(value, dict):
        return None
    person_id = value.get("person_id")
    model_id = value.get("model_id")
    model_version = value.get("model_version")
    embeddings = value.get("embeddings")
    if not isinstance(person_id, str) or not person_id:
        return None
    if not isinstance(model_id, str) or not isinstance(model_version, str):
        return None
    if not isinstance(embeddings, list):
        return None
    parsed_embeddings: list[list[float]] = []
    for item in embeddings:
        if not isinstance(item, list) or not item:
            continue
        try:
            parsed_embeddings.append([float(v) for v in item])
        except (TypeError, ValueError):
            continue
    qualities_raw = value.get("qualities")
    enrolled_raw = value.get("enrolled_at")
    qualities = qualities_raw if isinstance(qualities_raw, list) else []
    enrolled_at = enrolled_raw if isinstance(enrolled_raw, list) else []
    schema_version = int(value.get("schema_version") or IDENTITY_SCHEMA_VERSION)
    return IdentityRecord(
        person_id=person_id,
        model_id=model_id,
        model_version=model_version,
        embeddings=parsed_embeddings,
        qualities=[q for q in qualities if isinstance(q, dict)],
        enrolled_at=[float(t) for t in enrolled_at if isinstance(t, (int, float))],
        schema_version=schema_version,
    )


def _read_records(path: Path) -> list[IdentityRecord]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("Failed to read face identity store at %s: %s", path, exc)
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse face identity store at %s: %s", path, exc)
        return []
    if not isinstance(parsed, dict):
        return []
    records_value = parsed.get("identities")
    if not isinstance(records_value, list):
        return []
    records: list[IdentityRecord] = []
    for item in records_value:
        record = _record_from_json(item)
        if record is not None:
            records.append(record)
    return records


def _write_records(path: Path, records: list[IdentityRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "identities": [
            {
                "person_id": record.person_id,
                "model_id": record.model_id,
                "model_version": record.model_version,
                "embeddings": record.embeddings,
                "qualities": record.qualities,
                "enrolled_at": record.enrolled_at,
                "schema_version": record.schema_version,
            }
            for record in records
        ],
    }
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


class IdentityStore:
    """Thread-safe JSON identity gallery."""

    def __init__(self, instance_path: str | Path | None = None) -> None:
        """Bind the store to an app instance path."""
        self.path = identity_path_for_instance(instance_path)

    def list_identities(self) -> list[IdentityRecord]:
        """Return all identity records."""
        with _STORE_LOCK:
            return list(_read_records(self.path))

    def get(self, person_id: str) -> IdentityRecord | None:
        """Return one identity by person_id."""
        for record in self.list_identities():
            if record.person_id == person_id:
                return record
        return None

    def enrol(
        self,
        embeddings: Sequence[FaceEmbedding],
        *,
        person_id: str | None = None,
    ) -> IdentityRecord:
        """Persist several embeddings for a new or existing person_id."""
        if not embeddings:
            raise ValueError("enrol requires at least one embedding")
        model_ids = {item.model_id for item in embeddings}
        model_versions = {item.model_version for item in embeddings}
        if model_ids != {MODEL_ID_SFACE} or model_versions != {MODEL_VERSION_SFACE}:
            raise ValueError("All embeddings must use the active SFace model version")

        with _STORE_LOCK:
            records = _read_records(self.path)
            target_id = person_id or _next_person_id(records)
            existing = next((record for record in records if record.person_id == target_id), None)
            if existing is None:
                existing = IdentityRecord(
                    person_id=target_id,
                    model_id=MODEL_ID_SFACE,
                    model_version=MODEL_VERSION_SFACE,
                )
                records.append(existing)
            elif existing.model_id != MODEL_ID_SFACE or existing.model_version != MODEL_VERSION_SFACE:
                raise ValueError(
                    f"Cannot append embeddings: person {target_id} uses {existing.model_id}/{existing.model_version}"
                )

            for item in embeddings:
                existing.embeddings.append(list(item.vector))
                existing.qualities.append(item.quality.to_dict())
                existing.enrolled_at.append(item.created_at or time.time())

            _write_records(self.path, records)
            return existing

    def delete(self, person_id: str) -> bool:
        """Remove one identity; return whether it existed."""
        with _STORE_LOCK:
            records = _read_records(self.path)
            kept = [record for record in records if record.person_id != person_id]
            if len(kept) == len(records):
                return False
            _write_records(self.path, kept)
            return True

    def clear(self) -> None:
        """Remove all identities."""
        with _STORE_LOCK:
            _write_records(self.path, [])
