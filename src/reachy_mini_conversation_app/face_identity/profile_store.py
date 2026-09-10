"""Person profile store keyed by person_id (names and notes, not embeddings)."""

from __future__ import annotations
import os
import json
import time
import logging
import threading
from pathlib import Path

from reachy_mini_conversation_app.face_identity.types import PROFILE_SCHEMA_VERSION, PersonProfile


logger = logging.getLogger(__name__)

PROFILE_FILENAME = "person_profiles.v1.json"
_STORE_LOCK = threading.Lock()


def profile_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the person-profile JSON path for this app instance."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / "face_memory" / PROFILE_FILENAME

    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    if os.name == "nt":
        data_root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return data_root / "reachy_mini_conversation_app" / "face_memory" / PROFILE_FILENAME


def _profile_from_json(value: object) -> PersonProfile | None:
    if not isinstance(value, dict):
        return None
    person_id = value.get("person_id")
    name = value.get("name")
    if not isinstance(person_id, str) or not person_id:
        return None
    if not isinstance(name, str) or not name.strip():
        return None
    hobbies_raw = value.get("hobbies")
    interests_raw = value.get("interests")
    notes_raw = value.get("notes")
    hobbies = hobbies_raw if isinstance(hobbies_raw, list) else []
    interests = interests_raw if isinstance(interests_raw, list) else []
    notes = notes_raw if isinstance(notes_raw, list) else []
    return PersonProfile(
        person_id=person_id,
        name=name.strip(),
        relationship=str(value.get("relationship") or ""),
        hobbies=[str(item).strip() for item in hobbies if str(item).strip()],
        interests=[str(item).strip() for item in interests if str(item).strip()],
        notes=[str(item).strip() for item in notes if str(item).strip()],
        updated_at=float(value.get("updated_at") or 0.0),
        schema_version=int(value.get("schema_version") or PROFILE_SCHEMA_VERSION),
    )


def _read_profiles(path: Path) -> dict[str, PersonProfile]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        logger.warning("Failed to read person profile store at %s: %s", path, exc)
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse person profile store at %s: %s", path, exc)
        return {}
    if not isinstance(parsed, dict):
        return {}
    profiles_value = parsed.get("profiles")
    if not isinstance(profiles_value, list):
        return {}
    profiles: dict[str, PersonProfile] = {}
    for item in profiles_value:
        profile = _profile_from_json(item)
        if profile is not None:
            profiles[profile.person_id] = profile
    return profiles


def _write_profiles(path: Path, profiles: dict[str, PersonProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiles": [
            {
                "person_id": profile.person_id,
                "name": profile.name,
                "relationship": profile.relationship,
                "hobbies": profile.hobbies,
                "interests": profile.interests,
                "notes": profile.notes,
                "updated_at": profile.updated_at,
                "schema_version": profile.schema_version,
            }
            for profile in profiles.values()
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


class PersonProfileStore:
    """Thread-safe JSON person profiles."""

    def __init__(self, instance_path: str | Path | None = None) -> None:
        """Bind the store to an app instance path."""
        self.path = profile_path_for_instance(instance_path)

    def list_profiles(self) -> list[PersonProfile]:
        """Return all profiles."""
        with _STORE_LOCK:
            return list(_read_profiles(self.path).values())

    def get(self, person_id: str) -> PersonProfile | None:
        """Return one profile by person_id."""
        with _STORE_LOCK:
            return _read_profiles(self.path).get(person_id)

    def get_by_name(self, name: str) -> PersonProfile | None:
        """Return the first profile whose name matches case-insensitively."""
        needle = name.strip().lower()
        if not needle:
            return None
        for profile in self.list_profiles():
            if profile.name.lower() == needle:
                return profile
        return None

    def upsert(
        self,
        person_id: str,
        name: str,
        *,
        hobbies: list[str] | None = None,
        interests: list[str] | None = None,
        notes: list[str] | None = None,
        append_note: str | None = None,
        relationship: str | None = None,
    ) -> PersonProfile:
        """Create or update a person profile."""
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("name must be non-empty")
        with _STORE_LOCK:
            profiles = _read_profiles(self.path)
            existing = profiles.get(person_id)
            profile = existing or PersonProfile(person_id=person_id, name=cleaned_name)
            profile.name = cleaned_name
            if relationship is not None:
                profile.relationship = relationship.strip()
            if hobbies is not None:
                profile.hobbies = [item.strip() for item in hobbies if item.strip()]
            if interests is not None:
                profile.interests = [item.strip() for item in interests if item.strip()]
            if notes is not None:
                profile.notes = [item.strip() for item in notes if item.strip()]
            if append_note and append_note.strip():
                profile.notes.append(append_note.strip())
            profile.updated_at = time.time()
            profiles[person_id] = profile
            _write_profiles(self.path, profiles)
            return profile

    def delete(self, person_id: str) -> bool:
        """Remove one profile; return whether it existed."""
        with _STORE_LOCK:
            profiles = _read_profiles(self.path)
            if person_id not in profiles:
                return False
            del profiles[person_id]
            _write_profiles(self.path, profiles)
            return True

    def clear(self) -> None:
        """Remove all profiles."""
        with _STORE_LOCK:
            _write_profiles(self.path, {})
