from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from backend.item_knn import ItemKNNBundle
from backend.profile import UserProfile
from backend.recommender.models import BranchName, BranchResult, ProvenanceAtom


class Retriever(Protocol):
    branch: BranchName

    def retrieve(
        self,
        profile: UserProfile,
        active_source_ids: frozenset[str],
        *,
        limit: int,
    ) -> BranchResult: ...


@dataclass(frozen=True)
class FrozenContributionRetriever:
    branch: BranchName
    contributions: Mapping[str, Mapping[str, float]]
    artifact_version: str

    history_spans: Mapping[str, str] | None = None
    candidate_spans: Mapping[str, str] | None = None

    def retrieve(
        self,
        profile: UserProfile,
        active_source_ids: frozenset[str],
        *,
        limit: int,
    ) -> BranchResult:
        if isinstance(limit, bool) or limit <= 0:
            raise ValueError("Retriever limit must be positive")

        known_sources = {signal.history_id for signal in profile.positive_signals}
        if not active_source_ids <= known_sources:
            raise ValueError("Active source IDs must belong to the profile")

        scores: dict[str, float] = {}
        source_values: list[tuple[str, str, float]] = []

        for source_id in sorted(active_source_ids):
            for candidate_id, raw_value in sorted(
                self.contributions.get(source_id, {}).items()
            ):
                value = float(raw_value)
                if value <= 0.0 or candidate_id in profile.seen_item_ids:
                    continue

                scores[candidate_id] = scores.get(candidate_id, 0.0) + value
                source_values.append((source_id, candidate_id, value))

        ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
        pool = tuple(ordered[:limit])
        allowed_candidates = frozenset(pool)

        history_spans = self.history_spans or {}
        candidate_spans = self.candidate_spans or {}
        atoms = tuple(
            ProvenanceAtom(
                branch=self.branch,
                source_id=source_id,
                candidate_id=candidate_id,
                raw_contribution=value,
                history_span=history_spans.get(source_id, ""),
                candidate_span=candidate_spans.get(candidate_id, ""),
                artifact_version=self.artifact_version,
            )
            for source_id, candidate_id, value in source_values
            if candidate_id in allowed_candidates
        )

        pooled_scores = {candidate_id: scores[candidate_id] for candidate_id in pool}

        return BranchResult(self.branch, pooled_scores, atoms, pool)


@dataclass(frozen=True)
class ItemKNNContributionRetriever:
    bundle: ItemKNNBundle
    artifact_version: str
    branch: BranchName = "itemknn"

    def retrieve(
        self,
        profile: UserProfile,
        active_source_ids: frozenset[str],
        *,
        limit: int,
    ) -> BranchResult:
        _validate_request(profile, active_source_ids, limit)
        index_to_item = {
            index: item_id for item_id, index in self.bundle.item_to_index.items()
        }
        contributions: list[ProvenanceAtom] = []

        for signal in _active_signals(profile, active_source_ids):
            history_index = self.bundle.item_to_index.get(signal.item_id)
            if history_index is None:
                continue

            row = self.bundle.similarity.getrow(history_index)
            for candidate_index, similarity in zip(row.indices, row.data):
                candidate_id = index_to_item[int(candidate_index)]
                value = signal.weight * max(float(similarity), 0.0)

                if value <= 0.0 or candidate_id in profile.seen_item_ids:
                    continue

                contributions.append(
                    ProvenanceAtom(
                        self.branch,
                        signal.history_id,
                        candidate_id,
                        value,
                        history_span=_signal_text(signal),
                        artifact_version=self.artifact_version,
                    )
                )

        return _finalize(self.branch, contributions, limit)


@dataclass(frozen=True)
class BM25ContributionRetriever:
    documents: Mapping[str, Mapping[str, str]]
    artifact_version: str
    k1: float = 1.2

    b: float = 0.75
    branch: BranchName = "bm25"

    def retrieve(
        self,
        profile: UserProfile,
        active_source_ids: frozenset[str],
        *,
        limit: int,
    ) -> BranchResult:
        _validate_request(profile, active_source_ids, limit)
        if self.k1 <= 0.0 or not 0.0 <= self.b <= 1.0:
            raise ValueError("BM25 parameters are invalid")

        tokenized = {
            item_id: _tokenize(" ".join(fields.values()))
            for item_id, fields in self.documents.items()
        }
        document_count = len(tokenized)
        average_length = sum(map(len, tokenized.values())) / max(1, document_count)

        source_terms = {
            signal.history_id: frozenset(_tokenize(_signal_text(signal)))
            for signal in _active_signals(profile, active_source_ids)
        }
        term_sources: dict[str, list[str]] = {}

        for source_id, terms in source_terms.items():
            for term in terms:
                term_sources.setdefault(term, []).append(source_id)

        document_frequency = {
            term: sum(term in set(tokens) for tokens in tokenized.values())
            for term in term_sources
        }
        contributions: list[ProvenanceAtom] = []

        for candidate_id, tokens in tokenized.items():
            if candidate_id in profile.seen_item_ids:
                continue

            counts = Counter(tokens)
            candidate_text = " ".join(self.documents[candidate_id].values())

            for term, source_ids in sorted(term_sources.items()):
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue

                idf = math.log(
                    1.0
                    + (document_count - document_frequency[term] + 0.5)
                    / (document_frequency[term] + 0.5)
                )
                length_ratio = len(tokens) / max(average_length, 1.0)
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio)

                term_score = idf * numerator / denominator
                allocation = term_score / len(source_ids)

                for source_id in sorted(source_ids):
                    contributions.append(
                        ProvenanceAtom(
                            self.branch,
                            source_id,
                            candidate_id,
                            allocation,
                            history_span=term,
                            candidate_span=candidate_text,
                            artifact_version=self.artifact_version,
                        )
                    )

        return _finalize(self.branch, contributions, limit)


@dataclass(frozen=True)
class SemanticContributionRetriever:
    embeddings: Mapping[str, Sequence[float]]
    candidate_spans: Mapping[str, str]
    artifact_version: str

    branch: BranchName = "semantic"

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

        for signal in signals:
            history_vector = self.embeddings.get(signal.item_id)
            if history_vector is None:
                continue

            for candidate_id, candidate_vector in self.embeddings.items():
                if candidate_id in profile.seen_item_ids:
                    continue

                similarity = max(_cosine(history_vector, candidate_vector), 0.0)
                value = signal.weight * similarity
                if value <= 0.0:
                    continue

                contributions.append(
                    ProvenanceAtom(
                        self.branch,
                        signal.history_id,
                        candidate_id,
                        value,
                        history_span=_signal_text(signal),
                        candidate_span=self.candidate_spans.get(candidate_id, ""),
                        artifact_version=self.artifact_version,
                    )
                )

        return _finalize(self.branch, contributions, limit)


def _active_signals(
    profile: UserProfile,
    active_source_ids: frozenset[str],
) -> tuple[object, ...]:
    return tuple(
        signal
        for signal in profile.positive_signals
        if signal.history_id in active_source_ids
    )


def _validate_request(
    profile: UserProfile,
    active_source_ids: frozenset[str],
    limit: int,
) -> None:
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("Retriever limit must be positive")

    known_sources = {signal.history_id for signal in profile.positive_signals}
    if not active_source_ids <= known_sources:
        raise ValueError("Active source IDs must belong to the profile")


def _signal_text(signal: object) -> str:
    return " ".join(field.text for field in signal.text_fields)


def _finalize(
    branch: BranchName,
    contributions: list[ProvenanceAtom],
    limit: int,
) -> BranchResult:
    scores: dict[str, float] = {}

    for atom in contributions:
        scores[atom.candidate_id] = (
            scores.get(atom.candidate_id, 0.0) + atom.raw_contribution
        )

    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))
    pool = tuple(ordered[:limit])
    selected = frozenset(pool)

    atoms = tuple(
        sorted(
            (
                atom
                for atom in contributions
                if atom.candidate_id in selected
            ),
            key=lambda atom: (atom.candidate_id, atom.source_id),
        )
    )
    pooled_scores = {item_id: scores[item_id] for item_id in pool}

    return BranchResult(branch, pooled_scores, atoms, pool)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have one shared non-zero dimension")

    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if not all(math.isfinite(value) for value in left_values + right_values):
        raise ValueError("Embedding vectors must contain finite values")

    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left_values, right_values, strict=True))

    return dot_product / (left_norm * right_norm)

