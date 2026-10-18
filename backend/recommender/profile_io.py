from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from backend.profile import (
    PreferenceSignal,
    ProfileDiagnostics,
    ProfileTextField,
    UserProfile,
)


def load_profile(path: Path) -> UserProfile:
    """Load a validated user profile produced by recsys-profile."""
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError("Profile JSON root must be an object")

    schema_version = _string(payload, "schema_version")
    if schema_version != "user-profile-v1":
        raise ValueError(f"Unsupported profile schema: {schema_version}")

    phase = _string(payload, "phase")
    if phase not in {"validation", "test"}:
        raise ValueError(f"Unsupported profile phase: {phase}")

    diagnostics_payload = _mapping(payload, "diagnostics")
    diagnostics = ProfileDiagnostics(
        valid_interaction_count=_integer(
            diagnostics_payload,
            "valid_interaction_count",
        ),
        duplicate_item_count=_integer(
            diagnostics_payload,
            "duplicate_item_count",
        ),
        positive_count_before_limit=_integer(
            diagnostics_payload,
            "positive_count_before_limit",
        ),
        negative_count_before_limit=_integer(
            diagnostics_payload,
            "negative_count_before_limit",
        ),
        text_covered_positive_count=_integer(
            diagnostics_payload,
            "text_covered_positive_count",
        ),
        missing_catalog_item_count=_integer(
            diagnostics_payload,
            "missing_catalog_item_count",
        ),
        warnings=tuple(
            str(value) for value in _list(diagnostics_payload, "warnings")
        ),
    )
    positive_signals = _signals(payload, "positive_signals")
    negative_signals = _signals(payload, "negative_signals")

    return UserProfile(
        schema_version=schema_version,
        user_id=_string(payload, "user_id"),
        as_of=_integer(payload, "as_of"),
        phase=phase,
        artifact_version=_string(payload, "artifact_version"),
        seen_item_ids=frozenset(
            _non_empty_string(value, "seen_item_ids")
            for value in _list(payload, "seen_item_ids")
        ),
        positive_signals=positive_signals,
        negative_signals=negative_signals,
        diagnostics=diagnostics,
    )


def _signals(
    payload: dict[str, Any],
    name: str,
) -> tuple[PreferenceSignal, ...]:
    signals: list[PreferenceSignal] = []

    for raw_signal in _list(payload, name):
        if not isinstance(raw_signal, dict):
            raise ValueError(f"{name} entries must be objects")

        sentiment = _string(raw_signal, "sentiment")
        if sentiment not in {"positive", "neutral", "negative"}:
            raise ValueError(f"Unsupported sentiment: {sentiment}")

        fields = _text_fields(raw_signal)
        weight = _finite_float(raw_signal, "weight")

        signals.append(
            PreferenceSignal(
                history_id=_string(raw_signal, "history_id"),
                item_id=_string(raw_signal, "item_id"),
                rating=_finite_float(raw_signal, "rating"),
                timestamp=_integer(raw_signal, "timestamp"),
                sentiment=sentiment,
                weight=weight,
                text_fields=fields,
            )
        )

    history_ids = [signal.history_id for signal in signals]
    if len(history_ids) != len(set(history_ids)):
        raise ValueError("Profile history IDs must be unique")

    return tuple(signals)


def _text_fields(signal: dict[str, Any]) -> tuple[ProfileTextField, ...]:
    fields: list[ProfileTextField] = []

    for raw_field in _list(signal, "text_fields"):
        if not isinstance(raw_field, dict):
            raise ValueError("text_fields entries must be objects")

        fields.append(
            ProfileTextField(
                name=_string(raw_field, "name"),
                text=_string(raw_field, "text"),
                source_id=_string(raw_field, "source_id"),
            )
        )

    return tuple(fields)


def _mapping(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")

    return value


def _list(payload: dict[str, Any], name: str) -> list[Any]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")

    return value


def _string(payload: dict[str, Any], name: str) -> str:
    return _non_empty_string(payload.get(name), name)


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")

    return value.strip()


def _integer(payload: dict[str, Any], name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")

    return value


def _finite_float(payload: dict[str, Any], name: str) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number
