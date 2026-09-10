"""Conservative cosine-similarity identity matching for SFace embeddings."""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np

from reachy_mini_conversation_app.face_identity.types import (
    MODEL_ID_SFACE,
    MODEL_VERSION_SFACE,
    MatchResult,
    IdentityRecord,
    MatchCandidate,
)
from reachy_mini_conversation_app.face_identity.embedder import l2_normalize
from reachy_mini_conversation_app.face_identity.settings import match_margin, match_threshold


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity of two embeddings (assumed L2-normalized or not)."""
    va = l2_normalize(np.asarray(a, dtype=np.float32))
    vb = l2_normalize(np.asarray(b, dtype=np.float32))
    if va.shape != vb.shape:
        raise ValueError(f"Embedding shape mismatch: {va.shape} vs {vb.shape}")
    return float(np.dot(va, vb))


def best_similarity_for_person(probe: Sequence[float], embeddings: Sequence[Sequence[float]]) -> float:
    """Return the best cosine similarity against one person's embedding gallery."""
    if not embeddings:
        return -1.0
    return max(cosine_similarity(probe, sample) for sample in embeddings)


def match_embedding(
    probe: Sequence[float],
    identities: Sequence[IdentityRecord],
    *,
    model_id: str = MODEL_ID_SFACE,
    model_version: str = MODEL_VERSION_SFACE,
    threshold: float | None = None,
    margin: float | None = None,
) -> MatchResult:
    """Match a probe embedding with UNKNOWN bias on weak or ambiguous scores."""
    threshold_value = match_threshold() if threshold is None else threshold
    margin_value = match_margin() if margin is None else margin

    compatible = [
        record for record in identities if record.model_id == model_id and record.model_version == model_version
    ]
    incompatible = [
        record for record in identities if record.model_id != model_id or record.model_version != model_version
    ]
    if identities and not compatible and incompatible:
        return MatchResult(
            status="model_mismatch",
            reasons=("incompatible_model_version",),
        )

    candidates: list[MatchCandidate] = []
    for record in compatible:
        if not record.embeddings:
            continue
        score = best_similarity_for_person(probe, record.embeddings)
        candidates.append(MatchCandidate(person_id=record.person_id, similarity=score))

    if not candidates:
        return MatchResult(status="unknown", reasons=("empty_gallery",))

    candidates.sort(key=lambda item: item.similarity, reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    second_score = second.similarity if second is not None else None
    gap = None if second_score is None else best.similarity - second_score

    if best.similarity < threshold_value:
        return MatchResult(
            status="unknown",
            similarity=best.similarity,
            second_best_similarity=second_score,
            margin=gap,
            reasons=("below_threshold",),
        )

    if gap is not None and gap < margin_value:
        return MatchResult(
            status="unknown",
            similarity=best.similarity,
            second_best_similarity=second_score,
            margin=gap,
            reasons=("ambiguous_margin",),
        )

    return MatchResult(
        status="known",
        person_id=best.person_id,
        similarity=best.similarity,
        second_best_similarity=second_score,
        margin=gap if gap is not None else best.similarity,
    )
