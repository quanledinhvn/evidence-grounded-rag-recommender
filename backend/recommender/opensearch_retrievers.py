from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

from backend.profile import PreferenceSignal, UserProfile
from backend.recommender.models import BranchName, BranchResult, ProvenanceAtom
from backend.recommender.retrievers import (
    _active_signals,
    _finalize,
    _signal_text,
    _validate_request,
)

_TEXT_FIELDS = (
    "title",
    "subtitle",
    "description",
    "feature",
    "main_category",
    "category",
    "details",
)


class OpenSearchRetrievalGateway(Protocol):
    def bm25_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_item_ids: Sequence[str] = (),
        explain: bool = False,
    ) -> tuple[dict[str, Any], ...]: ...

    def item_chunks(
        self,
        item_id: str,
        *,
        embedding_model: str,
        limit: int = 1000,
    ) -> tuple[dict[str, Any], ...]: ...

    def semantic_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        excluded_item_ids: Sequence[str] = (),
    ) -> tuple[dict[str, Any], ...]: ...


@dataclass(frozen=True)
class OpenSearchBM25ContributionRetriever:
    gateway: OpenSearchRetrievalGateway
    artifact_version: str
    branch: BranchName = "bm25"

    _cache: dict[tuple[object, ...], tuple[ProvenanceAtom, ...]] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )

    def retrieve(
        self,
        profile: UserProfile,
        active_source_ids: frozenset[str],
        *,
        limit: int,
    ) -> BranchResult:
        _validate_request(profile, active_source_ids, limit)
        signals = _active_signals(profile, active_source_ids)
        contributions: list[ProvenanceAtom] = []

        missing = tuple(
            signal
            for signal in signals
            if _cache_key(profile, signal, limit) not in self._cache
        )
        if len(missing) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(missing))) as executor:
                futures = tuple(
                    executor.submit(self._source_atoms, profile, signal, limit)
                    for signal in missing
                )

                for future in futures:
                    future.result()

        for signal in signals:
            contributions.extend(self._source_atoms(profile, signal, limit))

        return _finalize(self.branch, contributions, limit)

    def _source_atoms(
        self,
        profile: UserProfile,
        signal: PreferenceSignal,
        limit: int,
    ) -> tuple[ProvenanceAtom, ...]:
        key = _cache_key(profile, signal, limit)
        if key in self._cache:
            return self._cache[key]

        query = _bounded_query(_signal_text(signal))
        if not query:
            self._cache[key] = ()

            return ()

        hits = self.gateway.bm25_search(
            query,
            limit=limit,
            excluded_item_ids=tuple(sorted(profile.seen_item_ids)),
        )
        atoms: list[ProvenanceAtom] = []

        for hit in hits:
            candidate_id = _candidate_id(hit)
            score = _positive_score(hit)

            if candidate_id in profile.seen_item_ids or score == 0.0:
                continue

            atoms.append(
                ProvenanceAtom(
                    self.branch,
                    signal.history_id,
                    candidate_id,
                    signal.weight * score,
                    history_span=query,
                    candidate_span=_hit_text(hit),
                    artifact_version=self.artifact_version,
                )
            )

        result = tuple(atoms)
        self._cache[key] = result

        return result


@dataclass(frozen=True)
class OpenSearchSemanticContributionRetriever:
    gateway: OpenSearchRetrievalGateway
    artifact_version: str
    embedding_model: str
    branch: BranchName = "semantic"

    _cache: dict[tuple[object, ...], tuple[ProvenanceAtom, ...]] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )

    def retrieve(
        self,
        profile: UserProfile,
        active_source_ids: frozenset[str],
        *,
        limit: int,
    ) -> BranchResult:
        _validate_request(profile, active_source_ids, limit)
        signals = _active_signals(profile, active_source_ids)
        contributions: list[ProvenanceAtom] = []

        missing = tuple(
            signal
            for signal in signals
            if _cache_key(profile, signal, limit) not in self._cache
        )
        if len(missing) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(missing))) as executor:
                futures = tuple(
                    executor.submit(self._source_atoms, profile, signal, limit)
                    for signal in missing
                )

                for future in futures:
                    future.result()

        for signal in signals:
            contributions.extend(self._source_atoms(profile, signal, limit))

        return _finalize(self.branch, contributions, limit)

    def _source_atoms(
        self,
        profile: UserProfile,
        signal: PreferenceSignal,
        limit: int,
    ) -> tuple[ProvenanceAtom, ...]:
        key = _cache_key(profile, signal, limit)
        if key in self._cache:
            return self._cache[key]

        chunks = self.gateway.item_chunks(
            signal.item_id,
            embedding_model=self.embedding_model,
        )
        if not chunks:
            self._cache[key] = ()

            return ()

        exclusions = tuple(sorted(profile.seen_item_ids))
        best_by_candidate: dict[str, tuple[float, str, str]] = {}

        embedding = _centroid(tuple(_embedding(chunk) for chunk in chunks))
        history_span = _history_span(chunks)

        hits = self.gateway.semantic_search(
            embedding,
            limit=limit,
            excluded_item_ids=exclusions,
        )

        for hit in hits:
            candidate_id = _candidate_id(hit)
            score = _positive_score(hit)

            if candidate_id in profile.seen_item_ids or score == 0.0:
                continue

            current = best_by_candidate.get(candidate_id)
            if current is not None and current[0] >= score:
                continue

            best_by_candidate[candidate_id] = (
                score,
                history_span,
                _hit_text(hit),
            )

        atoms = tuple(
            ProvenanceAtom(
                self.branch,
                signal.history_id,
                candidate_id,
                signal.weight * score,
                history_span=history_span,
                candidate_span=candidate_span,
                artifact_version=self.artifact_version,
            )
            for candidate_id, (
                score,
                history_span,
                candidate_span,
            ) in sorted(best_by_candidate.items())
        )
        self._cache[key] = atoms

        return atoms


def _candidate_id(hit: dict[str, Any]) -> str:
    candidate_id = str(hit.get("item_id") or "").strip()
    if not candidate_id:
        raise ValueError("OpenSearch hit requires a non-empty item_id")

    return candidate_id


def _bounded_query(text: str, max_terms: int = 128) -> str:
    selected: list[str] = []
    seen: set[str] = set()

    for term in text.split():
        normalized = term.casefold()
        if not normalized or normalized in seen:
            continue

        selected.append(term)
        seen.add(normalized)

        if len(selected) >= max_terms:
            break

    return " ".join(selected)


def _cache_key(
    profile: UserProfile,
    signal: PreferenceSignal,
    limit: int,
) -> tuple[object, ...]:
    return (
        profile.artifact_version,
        profile.user_id,
        profile.as_of,
        profile.seen_item_ids,
        signal.history_id,
        limit,
    )


def _positive_score(hit: dict[str, Any]) -> float:
    score = float(hit.get("score") or 0.0)
    if not math.isfinite(score):
        raise ValueError("OpenSearch score must be finite")

    return max(score, 0.0)


def _embedding(chunk: dict[str, Any]) -> tuple[float, ...]:
    raw_embedding = chunk.get("embedding")
    if not isinstance(raw_embedding, Sequence) or isinstance(raw_embedding, str):
        raise ValueError("Indexed history chunk requires an embedding")

    embedding = tuple(float(value) for value in raw_embedding)
    if not embedding or not all(math.isfinite(value) for value in embedding):
        raise ValueError("Indexed history embedding must be finite and non-empty")

    return embedding


def _centroid(vectors: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError("History chunk embeddings must share one dimension")

    means = tuple(
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    )
    norm = math.sqrt(sum(value * value for value in means))

    if norm == 0.0:
        raise ValueError("History chunk centroid must have non-zero norm")

    return tuple(value / norm for value in means)


def _history_span(chunks: Sequence[dict[str, Any]]) -> str:
    spans = tuple(
        str(chunk.get("raw_text") or "").strip()
        for chunk in chunks
        if str(chunk.get("raw_text") or "").strip()
    )

    return " | ".join(spans)[:2000]


def _hit_text(hit: dict[str, Any]) -> str:
    raw_text = str(hit.get("raw_text") or "").strip()
    if raw_text:
        return raw_text

    return " ".join(
        str(hit.get(field) or "").strip()
        for field in _TEXT_FIELDS
        if str(hit.get(field) or "").strip()
    )
