from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Phase = Literal["validation", "test"]
Sentiment = Literal["positive", "neutral", "negative"]
TextFieldName = Literal[
    "title",
    "subtitle",
    "description",
    "feature",
    "main_category",
    "category",
    "details",
]

_DAY_MS = 86_400_000
_SCHEMA_VERSION = "user-profile-v1"
_SPLITS_BY_PHASE: dict[Phase, tuple[str, ...]] = {
    "validation": ("train",),
    "test": ("train", "validation"),
}


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    rating: float

    timestamp: int
    review_id: str
    split: str


@dataclass(frozen=True)
class CatalogItem:
    item_id: str
    source_id: str
    title: str = ""

    subtitle: str = ""
    description: str = ""
    feature: str = ""

    main_category: str = ""
    category: str = ""
    store: str = ""

    details: str = ""


@dataclass(frozen=True)
class ProfileRequest:
    user_id: str
    as_of: int
    phase: Phase

    artifact_version: str


@dataclass(frozen=True)
class ProfilePolicy:
    allowed_splits: tuple[str, ...]
    positive_threshold: float = 4.0
    negative_threshold: float = 2.0

    max_positive_signals: int = 20
    max_negative_signals: int = 20
    recency_half_life_days: float = 90.0


@dataclass(frozen=True)
class ProfileTextField:
    name: TextFieldName
    text: str
    source_id: str


@dataclass(frozen=True)
class PreferenceSignal:
    history_id: str
    item_id: str
    rating: float

    timestamp: int
    sentiment: Sentiment
    weight: float

    text_fields: tuple[ProfileTextField, ...]


@dataclass(frozen=True)
class ProfileDiagnostics:
    valid_interaction_count: int
    duplicate_item_count: int
    positive_count_before_limit: int

    negative_count_before_limit: int
    text_covered_positive_count: int
    missing_catalog_item_count: int

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class UserProfile:
    schema_version: str
    user_id: str
    as_of: int

    phase: Phase
    artifact_version: str
    seen_item_ids: frozenset[str]

    positive_signals: tuple[PreferenceSignal, ...]
    negative_signals: tuple[PreferenceSignal, ...]

    diagnostics: ProfileDiagnostics


class HistoryStore(Protocol):
    artifact_version: str

    def get_before(
        self,
        user_id: str,
        as_of: int,
        allowed_splits: tuple[str, ...],
    ) -> Sequence[Interaction]: ...


class CatalogStore(Protocol):
    artifact_version: str

    def get_many(self, item_ids: frozenset[str]) -> Mapping[str, CatalogItem]: ...


@dataclass(frozen=True)
class InMemoryCatalogStore:
    artifact_version: str
    items: Mapping[str, CatalogItem]

    def get_many(self, item_ids: frozenset[str]) -> Mapping[str, CatalogItem]:
        return {
            item_id: self.items[item_id]
            for item_id in sorted(item_ids)
            if item_id in self.items
        }


class ProfileBuilder:
    def __init__(
        self,
        history_store: HistoryStore,
        catalog_store: CatalogStore,
    ) -> None:
        self._history_store = history_store
        self._catalog_store = catalog_store

    def build(
        self,
        request: ProfileRequest,
        policy: ProfilePolicy,
    ) -> UserProfile:
        self._validate_contract(request, policy)

        rows = self._history_store.get_before(
            request.user_id,
            request.as_of,
            policy.allowed_splits,
        )
        valid_rows = self._valid_rows(rows, request, policy)
        latest_by_item = self._latest_by_item(valid_rows)

        seen_item_ids = frozenset(row.item_id for row in valid_rows)
        catalog = self._catalog_store.get_many(seen_item_ids)
        self._validate_catalog(catalog)

        positive_rows = self._sentiment_rows(
            latest_by_item.values(),
            minimum=policy.positive_threshold,
        )
        negative_rows = self._sentiment_rows(
            latest_by_item.values(),
            maximum=policy.negative_threshold,
        )

        selected_positive = positive_rows[: policy.max_positive_signals]
        selected_negative = negative_rows[: policy.max_negative_signals]

        positive_signals, next_history_index = self._signals(
            selected_positive,
            "positive",
            request,
            policy,
            catalog,
            history_index=1,
        )
        negative_signals, _ = self._signals(
            selected_negative,
            "negative",
            request,
            policy,
            catalog,
            history_index=next_history_index,
        )

        missing_items = tuple(
            item_id
            for item_id in sorted(seen_item_ids)
            if item_id not in catalog
            or not self._catalog_has_text(catalog[item_id])
        )
        warnings = tuple(
            f"Catalog metadata is unavailable for {item_id}."
            for item_id in missing_items
        )

        diagnostics = ProfileDiagnostics(
            valid_interaction_count=len(valid_rows),
            duplicate_item_count=len(valid_rows) - len(latest_by_item),
            positive_count_before_limit=len(positive_rows),
            negative_count_before_limit=len(negative_rows),
            text_covered_positive_count=sum(
                bool(signal.text_fields) for signal in positive_signals
            ),
            missing_catalog_item_count=len(missing_items),
            warnings=warnings,
        )

        return UserProfile(
            schema_version=_SCHEMA_VERSION,
            user_id=request.user_id,
            as_of=request.as_of,
            phase=request.phase,
            artifact_version=request.artifact_version,
            seen_item_ids=seen_item_ids,
            positive_signals=positive_signals,
            negative_signals=negative_signals,
            diagnostics=diagnostics,
        )

    def _validate_contract(
        self,
        request: ProfileRequest,
        policy: ProfilePolicy,
    ) -> None:
        if not isinstance(request.user_id, str) or not request.user_id.strip():
            raise ValueError("user_id must be a non-empty string")

        if isinstance(request.as_of, bool) or request.as_of <= 0:
            raise ValueError("as_of must be a positive integer")

        if request.phase not in _SPLITS_BY_PHASE:
            raise ValueError(f"Unsupported phase: {request.phase}")

        expected_splits = _SPLITS_BY_PHASE[request.phase]
        if policy.allowed_splits != expected_splits:
            message = f"allowed_splits for {request.phase} must be {expected_splits}"
            raise ValueError(message)

        versions = (
            request.artifact_version,
            self._history_store.artifact_version,
            self._catalog_store.artifact_version,
        )
        if not request.artifact_version or len(set(versions)) != 1:
            raise ValueError("artifact_version must match every profile store")

        thresholds = (policy.negative_threshold, policy.positive_threshold)
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("rating thresholds must be finite")

        if not 1.0 <= policy.negative_threshold < policy.positive_threshold <= 5.0:
            raise ValueError("rating thresholds must satisfy 1 <= negative < positive <= 5")

        limits = (policy.max_positive_signals, policy.max_negative_signals)
        if any(isinstance(value, bool) or value <= 0 for value in limits):
            raise ValueError("profile signal limits must be positive integers")

        half_life = policy.recency_half_life_days
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("recency_half_life_days must be positive")

    @staticmethod
    def _valid_rows(
        rows: Sequence[Interaction],
        request: ProfileRequest,
        policy: ProfilePolicy,
    ) -> tuple[Interaction, ...]:
        valid: list[Interaction] = []
        allowed = frozenset(policy.allowed_splits)

        for row in rows:
            if row.user_id != request.user_id or row.split not in allowed:
                continue

            if row.timestamp >= request.as_of:
                continue

            ProfileBuilder._validate_interaction(row)
            valid.append(row)

        return tuple(valid)

    @staticmethod
    def _validate_interaction(row: Interaction) -> None:
        if not isinstance(row.item_id, str) or not row.item_id.strip():
            raise ValueError("History contains an invalid item_id")

        if isinstance(row.timestamp, bool) or row.timestamp <= 0:
            raise ValueError("History contains an invalid timestamp")

        if not math.isfinite(row.rating) or not 1.0 <= row.rating <= 5.0:
            raise ValueError("History contains an invalid rating")

        if not isinstance(row.review_id, str) or not row.review_id:
            raise ValueError("History contains an invalid review_id")

    @staticmethod
    def _latest_by_item(
        rows: tuple[Interaction, ...],
    ) -> dict[str, Interaction]:
        latest: dict[str, Interaction] = {}

        for row in rows:
            current = latest.get(row.item_id)
            row_key = (row.timestamp, row.review_id, row.item_id)

            if current is None:
                latest[row.item_id] = row
                continue

            current_key = (current.timestamp, current.review_id, current.item_id)
            if row_key > current_key:
                latest[row.item_id] = row

        return latest

    @staticmethod
    def _sentiment_rows(
        rows: Sequence[Interaction],
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> tuple[Interaction, ...]:
        selected = (
            row
            for row in rows
            if (minimum is None or row.rating >= minimum)
            and (maximum is None or row.rating <= maximum)
        )

        return tuple(
            sorted(
                selected,
                key=lambda row: (-row.timestamp, row.review_id, row.item_id),
            )
        )

    @staticmethod
    def _validate_catalog(catalog: Mapping[str, CatalogItem]) -> None:
        for item_id, item in catalog.items():
            if item.item_id != item_id:
                raise ValueError(f"Catalog key does not match item_id: {item_id}")

            if ProfileBuilder._catalog_has_text(item) and not item.source_id:
                raise ValueError(f"Catalog item has an empty source_id: {item_id}")

    @staticmethod
    def _catalog_has_text(item: CatalogItem) -> bool:
        return any(
            text.strip()
            for text in (
                item.title,
                item.subtitle,
                item.description,

                item.feature,
                item.main_category,
                item.category,

                item.details,
            )
        )

    @staticmethod
    def _signals(
        rows: tuple[Interaction, ...],
        sentiment: Sentiment,
        request: ProfileRequest,
        policy: ProfilePolicy,
        catalog: Mapping[str, CatalogItem],
        *,
        history_index: int,
    ) -> tuple[tuple[PreferenceSignal, ...], int]:
        signals: list[PreferenceSignal] = []

        for offset, row in enumerate(rows):
            history_id = f"H{history_index + offset}"
            item = catalog.get(row.item_id)

            signals.append(
                PreferenceSignal(
                    history_id=history_id,
                    item_id=row.item_id,
                    rating=row.rating,
                    timestamp=row.timestamp,
                    sentiment=sentiment,
                    weight=ProfileBuilder._weight(row, request, policy),
                    text_fields=ProfileBuilder._text_fields(item),
                )
            )

        next_index = history_index + len(signals)

        return tuple(signals), next_index

    @staticmethod
    def _weight(
        row: Interaction,
        request: ProfileRequest,
        policy: ProfilePolicy,
    ) -> float:
        age_days = max(0.0, (request.as_of - row.timestamp) / _DAY_MS)
        recency = 2.0 ** (-age_days / policy.recency_half_life_days)

        rating_strength = abs(row.rating - 3.0) / 2.0

        return rating_strength * recency

    @staticmethod
    def _text_fields(item: CatalogItem | None) -> tuple[ProfileTextField, ...]:
        if item is None:
            return ()

        values: tuple[tuple[TextFieldName, str], ...] = (
            ("title", item.title),
            ("subtitle", item.subtitle),
            ("description", item.description),
            ("feature", item.feature),
            ("main_category", item.main_category),
            ("category", item.category),
            ("details", item.details),
        )

        return tuple(
            ProfileTextField(
                name=name,
                text=text.strip(),
                source_id=f"{item.source_id}:{name}",
            )
            for name, text in values
            if text.strip()
        )

