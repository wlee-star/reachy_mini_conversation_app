"""Face-memory service facade used by tools."""

from __future__ import annotations
import time
import logging
import threading
from typing import Any
from pathlib import Path
from collections import Counter

from reachy_mini_conversation_app.face_identity.pipeline import FaceIdentityPipeline
from reachy_mini_conversation_app.face_identity.settings import (
    face_memory_enabled,
    face_memory_photo_enrolment_enabled,
    face_memory_live_recognition_enabled,
    face_memory_on_demand_recognition_enabled,
)
from reachy_mini_conversation_app.face_identity.photo_session import (
    HeadTrackingGuard,
    PhotoEnrolmentSession,
    collect_photo_embeddings,
    capture_frames_from_robot,
)
from reachy_mini_conversation_app.face_identity.speech_authority import (
    ON_DEMAND_DISABLED_SPOKEN,
    is_robot_name_variant,
    verify_persisted_enrolment,
    spoken_for_recognition_result,
    face_memory_photo_enrolment_disabled_result,
)


logger = logging.getLogger(__name__)

# Short on-demand window: enough samples for confirmation without a live loop.
ON_DEMAND_FRAME_COUNT = 8
ON_DEMAND_FRAME_INTERVAL_S = 0.1
ON_DEMAND_MIN_AGREEING = 2


class FaceMemoryService:
    """Coordinates offline recognition, photo enrolment, and privacy ops."""

    def __init__(self, instance_path: str | Path | None = None, *, allow_download: bool = False) -> None:
        """Create stores/pipeline lazily bound to an instance path."""
        self.instance_path = instance_path
        self._allow_download = allow_download
        self._pipeline: FaceIdentityPipeline | None = None
        self._pipeline_lock = threading.Lock()
        self.session = PhotoEnrolmentSession()
        self._session_lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """Whether face-memory features are enabled."""
        return face_memory_enabled()

    @property
    def live_enabled(self) -> bool:
        """Whether continuous live recognition is enabled (always false in this phase by default)."""
        return face_memory_live_recognition_enabled()

    @property
    def photo_enrolment_enabled(self) -> bool:
        """Whether conversational held-up-photo enrolment is enabled (default off)."""
        return face_memory_photo_enrolment_enabled()

    @property
    def on_demand_enabled(self) -> bool:
        """Whether explicit user-triggered frame identification is enabled."""
        return face_memory_on_demand_recognition_enabled()

    def pipeline(self) -> FaceIdentityPipeline:
        """Return a lazily constructed identity pipeline."""
        with self._pipeline_lock:
            if self._pipeline is None:
                self._pipeline = FaceIdentityPipeline(
                    self.instance_path,
                    allow_download=self._allow_download,
                )
            return self._pipeline

    def abort_photo_enrolment(self, movement_manager: Any | None = None) -> None:
        """Cancel photo enrolment, discard temps, and restore tracking if needed."""
        with self._session_lock:
            self.session.cancel_event.set()
            prior = self.session.tracking_was_enabled
            self.session.reset_temporary()
            self.session.active = False
            self.session.phase = "idle"
            self.session.cancel_event = threading.Event()
        if movement_manager is not None and prior is not None:
            restorer = getattr(movement_manager, "restore_head_tracking", None)
            if restorer is not None:
                try:
                    restorer(prior)
                except Exception as exc:
                    logger.warning("Failed to restore head tracking after abort: %s", exc)

    def scan_photo(
        self,
        deps: Any,
    ) -> dict[str, Any]:
        """Run a photo scan window with head-tracking freeze/restore."""
        if not self.enabled:
            return {
                "error": "face_memory_disabled",
                "spoken": "Face memory is turned off right now.",
                "persisted": False,
            }
        if not self.photo_enrolment_enabled:
            return face_memory_photo_enrolment_disabled_result()
        if not getattr(deps, "camera_enabled", False):
            return {
                "error": "camera_disabled",
                "spoken": "The camera is disabled, so I cannot look at a photo.",
                "persisted": False,
            }

        with self._session_lock:
            if self.session.active and self.session.phase not in {"idle", "done"}:
                return {"error": "enrolment_busy", "spoken": "I am already working on a photo enrolment."}
            self.session.active = True
            self.session.phase = "scanning"
            self.session.cancel_event.clear()
            self.session.reset_temporary()

        movement_manager = deps.movement_manager
        try:
            with HeadTrackingGuard(movement_manager) as guard:
                self.session.tracking_was_enabled = guard._prior_enabled
                if self.session.cancel_event.is_set():
                    return {"status": "cancelled", "spoken": "Okay, I cancelled the photo enrolment."}
                frames = capture_frames_from_robot(
                    deps.reachy_mini,
                    cancel_event=self.session.cancel_event,
                )
                if self.session.cancel_event.is_set():
                    return {"status": "cancelled", "spoken": "Okay, I cancelled the photo enrolment."}
                if not frames:
                    self.session.phase = "idle"
                    self.session.active = False
                    return {
                        "status": "no_clear_face",
                        "spoken": "I can't see a clear person in the photo. Could you hold it a little closer?",
                    }
                result = collect_photo_embeddings(self.pipeline(), frames)
        except Exception as exc:
            logger.warning("Photo scan failed: %s", exc)
            self.abort_photo_enrolment(movement_manager)
            return {
                "error": f"photo_scan_failed: {type(exc).__name__}",
                "spoken": "I had trouble looking at that photo.",
            }

        with self._session_lock:
            if result.get("status") == "ready":
                self.session.temporary_embeddings = list(result["embeddings"])
                self.session.phase = "await_consent"
                self.session.last_scan_status = "ready"
                return {
                    "status": "awaiting_consent",
                    "persisted": False,
                    "embedding_count": len(self.session.temporary_embeddings),
                    "spoken": "I can see someone. Would you like me to remember who this is?",
                }
            if result.get("status") == "multiple_faces":
                self.session.reset_temporary()
                self.session.active = False
                self.session.phase = "idle"
                return {
                    "status": "multiple_faces",
                    "persisted": False,
                    "spoken": "I can see more than one person. Could you show me just the person you want me to remember?",
                }
            self.session.reset_temporary()
            self.session.active = False
            self.session.phase = "idle"
            return {
                "status": "no_clear_face",
                "persisted": False,
                "spoken": "I can't see a clear person in the photo. Could you hold it a little closer?",
            }

    def consent(self, accepted: bool, movement_manager: Any | None = None) -> dict[str, Any]:
        """Handle yes/no after a successful photo scan."""
        if not self.photo_enrolment_enabled:
            return face_memory_photo_enrolment_disabled_result()
        with self._session_lock:
            if not self.session.active or self.session.phase != "await_consent":
                return {"error": "not_awaiting_consent"}
            if not accepted:
                self.session.reset_temporary()
                self.session.active = False
                self.session.phase = "idle"
                prior = self.session.tracking_was_enabled
                self.session.tracking_was_enabled = None
                if movement_manager is not None and prior is not None:
                    restorer = getattr(movement_manager, "restore_head_tracking", None)
                    if restorer is not None:
                        restorer(prior)
                return {
                    "status": "consent_declined",
                    "persisted": False,
                    "spoken": "Okay, I will not remember that person.",
                }
            self.session.phase = "await_name"
            return {"status": "awaiting_name", "persisted": False, "spoken": "What's their name?"}

    def set_name_and_persist(self, name: str) -> dict[str, Any]:
        """Persist temporary embeddings under a new person_id and profile name."""
        if not self.photo_enrolment_enabled:
            return face_memory_photo_enrolment_disabled_result()
        cleaned = name.strip()
        if not cleaned:
            return {"error": "empty_name", "persisted": False, "spoken": "I need a name to remember them."}
        if is_robot_name_variant(cleaned):
            return {
                "status": "error",
                "persisted": False,
                "spoken": "That sounds like my name, not theirs. What's the person's name?",
                "reason": "robot_name_variant",
            }
        with self._session_lock:
            if not self.session.active or self.session.phase != "await_name":
                return {"error": "not_awaiting_name", "persisted": False}
            embeddings = list(self.session.temporary_embeddings)
            if len(embeddings) < 2:
                self.session.reset_temporary()
                self.session.active = False
                self.session.phase = "idle"
                return {
                    "error": "missing_embeddings",
                    "persisted": False,
                    "spoken": "I lost the photo samples. Please show the photo again.",
                }

        pipeline = self.pipeline()
        try:
            record = pipeline.identities.enrol(embeddings)
            profile = pipeline.profiles.upsert(record.person_id, cleaned)
        except Exception as exc:
            logger.warning("Face enrolment persistence failed: %s", exc)
            with self._session_lock:
                self.session.reset_temporary()
                self.session.active = False
                self.session.phase = "idle"
            return {
                "status": "persistence_failed",
                "persisted": False,
                "spoken": "I could not save that person in face memory.",
            }

        verified = verify_persisted_enrolment(
            identities=pipeline.identities,
            profiles=pipeline.profiles,
            person_id=record.person_id,
            name=profile.name,
            min_embeddings=2,
        )
        if verified.get("persisted") is not True:
            with self._session_lock:
                self.session.reset_temporary()
                self.session.active = False
                self.session.phase = "idle"
            return verified

        with self._session_lock:
            self.session.temporary_embeddings.clear()
            self.session.pending_person_id = record.person_id
            self.session.pending_name = profile.name
            self.session.phase = "await_details"
        return {
            "status": "enrolled",
            "person_id": record.person_id,
            "name": profile.name,
            "embedding_count": len(record.embeddings),
            "persisted": True,
            "spoken": f"I'll remember {profile.name}. What would you like me to remember about {profile.name}?",
        }

    def save_details(self, details: str, movement_manager: Any | None = None) -> dict[str, Any]:
        """Store freeform details into the profile store and finish the session."""
        with self._session_lock:
            if not self.session.active or self.session.phase != "await_details":
                return {"error": "not_awaiting_details"}
            person_id = self.session.pending_person_id
            name = self.session.pending_name
            if not person_id or not name:
                return {"error": "missing_person"}
            note = details.strip()
            if note:
                self.pipeline().profiles.upsert(person_id, name, append_note=note)
            prior = self.session.tracking_was_enabled
            self.session.reset_temporary()
            self.session.active = False
            self.session.phase = "done"
            self.session.tracking_was_enabled = None

        if movement_manager is not None and prior is not None:
            restorer = getattr(movement_manager, "restore_head_tracking", None)
            if restorer is not None:
                restorer(prior)
        profile = self.pipeline().profiles.get(person_id)
        if profile is None:
            return {
                "status": "persistence_failed",
                "persisted": False,
                "person_id": person_id,
                "name": name,
                "spoken": "I could not verify those details were saved.",
            }
        return {
            "status": "complete",
            "person_id": person_id,
            "name": name,
            "persisted": True,
            "spoken": f"Got it. I've saved those details about {name}.",
        }

    def who_is_in_frame(self, deps: Any) -> dict[str, Any]:
        """Identify who is in the current camera using a short on-demand frame window."""
        started = time.perf_counter()
        if not self.enabled:
            disabled: dict[str, Any] = {
                "error": "face_memory_disabled",
                "status": "disabled",
                "spoken": ON_DEMAND_DISABLED_SPOKEN,
            }
            disabled["spoken"] = spoken_for_recognition_result(disabled)
            return disabled
        if not self.on_demand_enabled:
            return {
                "error": "on_demand_recognition_disabled",
                "status": "disabled",
                "spoken": ON_DEMAND_DISABLED_SPOKEN,
            }
        if not getattr(deps, "camera_enabled", False):
            camera_off: dict[str, Any] = {"error": "camera_disabled", "status": "error"}
            camera_off["spoken"] = spoken_for_recognition_result(camera_off)
            return camera_off
        media = getattr(deps.reachy_mini, "media", None)
        if media is None or getattr(media, "get_frame", None) is None:
            unavailable: dict[str, Any] = {"error": "camera_unavailable", "status": "error"}
            unavailable["spoken"] = spoken_for_recognition_result(unavailable)
            return unavailable

        capture_started = time.perf_counter()
        try:
            frames = capture_frames_from_robot(
                deps.reachy_mini,
                count=ON_DEMAND_FRAME_COUNT,
                interval_s=ON_DEMAND_FRAME_INTERVAL_S,
            )
        except Exception as exc:
            logger.warning("[FACE-ID] frame capture failed: %s", exc)
            failed: dict[str, Any] = {"error": f"capture_failed: {type(exc).__name__}", "status": "error"}
            failed["spoken"] = spoken_for_recognition_result(failed)
            return failed
        capture_ms = (time.perf_counter() - capture_started) * 1000
        logger.info("[FACE-ID] frames_captured=%s capture_ms=%.0f", len(frames), capture_ms)

        if not frames:
            empty: dict[str, Any] = {"status": "no_face", "face_count": 0, "usable_frames": 0}
            empty["spoken"] = spoken_for_recognition_result(empty)
            logger.info("[FACE-ID] result=no_face")
            return empty

        recognize_started = time.perf_counter()
        result = self.recognize_frame_window(frames)
        recognize_ms = (time.perf_counter() - recognize_started) * 1000
        result["spoken"] = spoken_for_recognition_result(result)
        result["latency_ms"] = {
            "capture": round(capture_ms, 1),
            "recognize": round(recognize_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        }
        self._log_face_id_result(result)
        return result

    def recognize_frame_window(self, frames: list[Any]) -> dict[str, Any]:
        """Match a short burst of frames; read-only against identity/profile stores."""
        pipeline = self.pipeline()
        per_frame: list[dict[str, Any]] = []
        for frame in frames:
            per_frame.append(pipeline.recognize_frame(frame))
        return aggregate_on_demand_results(per_frame, profiles=pipeline.profiles)

    def _log_face_id_result(self, payload: dict[str, Any]) -> None:
        """Emit concise on-demand recognition logs without biometric vectors."""
        status = payload.get("status")
        usable = payload.get("usable_frames")
        if usable is not None:
            logger.info("[FACE-ID] usable_frames=%s", usable)
        person_id = payload.get("person_id")
        similarity = payload.get("similarity")
        margin = payload.get("margin")
        if person_id and similarity is not None:
            logger.info(
                "[FACE-ID] best=%s similarity=%.2f margin=%s",
                person_id,
                float(similarity),
                f"{float(margin):.2f}" if margin is not None else "n/a",
            )
        name = payload.get("name")
        if status == "known" and name:
            logger.info("[FACE-ID] result=known name=%s", name)
        else:
            logger.info("[FACE-ID] result=%s", status)


def aggregate_on_demand_results(
    per_frame: list[dict[str, Any]],
    *,
    profiles: Any | None = None,
) -> dict[str, Any]:
    """Combine per-frame match payloads into one fail-safe on-demand result."""
    known_votes: Counter[str] = Counter()
    known_samples: dict[str, list[dict[str, Any]]] = {}
    multiple_faces = 0
    no_face = 0
    unusable = 0
    unknown = 0
    ambiguous = 0
    model_mismatch = 0

    for result in per_frame:
        status = str(result.get("status") or "").strip().lower()
        reasons = {str(item) for item in (result.get("reasons") or [])}
        if status == "known" and result.get("person_id"):
            person_id = str(result["person_id"])
            known_votes[person_id] += 1
            known_samples.setdefault(person_id, []).append(result)
        elif status == "multiple_faces":
            multiple_faces += 1
        elif status == "no_face":
            no_face += 1
        elif status == "unusable":
            unusable += 1
        elif status == "model_mismatch":
            model_mismatch += 1
        elif status == "unknown" and "ambiguous_margin" in reasons:
            ambiguous += 1
        elif status in {"unknown", "ambiguous"}:
            unknown += 1
        else:
            unusable += 1

    usable_frames = sum(known_votes.values()) + unknown + ambiguous
    logger.info(
        "[FACE-ID] frame_summary known=%s unknown=%s ambiguous=%s multiple=%s no_face=%s unusable=%s",
        sum(known_votes.values()),
        unknown,
        ambiguous,
        multiple_faces,
        no_face,
        unusable,
    )

    if model_mismatch and not known_votes:
        return {"status": "model_mismatch", "usable_frames": usable_frames, "reasons": ["incompatible_model_version"]}

    if known_votes:
        winner, count = known_votes.most_common(1)[0]
        second_count = known_votes.most_common(2)[1][1] if len(known_votes) > 1 else 0
        if count >= ON_DEMAND_MIN_AGREEING and count > second_count:
            sample = known_samples[winner][0]
            for candidate in known_samples[winner]:
                if (candidate.get("similarity") or 0) >= (sample.get("similarity") or 0):
                    sample = candidate
            payload: dict[str, Any] = {
                "status": "known",
                "person_id": winner,
                "similarity": sample.get("similarity"),
                "second_best_similarity": sample.get("second_best_similarity"),
                "margin": sample.get("margin"),
                "face_count": 1,
                "usable_frames": usable_frames,
                "agreeing_frames": count,
            }
            name = sample.get("name")
            relationship = ""
            if profiles is not None:
                profile = profiles.get(winner)
                if profile is not None:
                    name = profile.name
                    relationship = getattr(profile, "relationship", "") or ""
            if name:
                payload["name"] = name
            if relationship:
                payload["relationship"] = relationship
            return payload
        if len(known_votes) > 1:
            return {
                "status": "ambiguous",
                "usable_frames": usable_frames,
                "reasons": ["conflicting_frame_winners"],
            }

    if multiple_faces > 0 and sum(known_votes.values()) == 0 and multiple_faces >= max(unknown + ambiguous, 1):
        return {"status": "multiple_faces", "usable_frames": usable_frames, "face_count": 2}

    if usable_frames == 0:
        return {
            "status": "no_face" if no_face >= unusable else "unusable",
            "usable_frames": 0,
            "face_count": 0,
        }

    if ambiguous and unknown == 0 and sum(known_votes.values()) == 0:
        return {"status": "ambiguous", "usable_frames": usable_frames, "reasons": ["ambiguous_margin"]}

    return {"status": "unknown", "usable_frames": usable_frames, "reasons": ["below_threshold_or_no_consensus"]}
