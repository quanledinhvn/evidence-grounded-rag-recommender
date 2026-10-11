from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from backend.profile.builder import (
    PreferenceSignal,
    ProfileBuilder,
    ProfilePolicy,
    ProfileRequest,
    UserProfile,
)
from backend.profile.catalog_store import ParquetCatalogStore
from backend.profile.history_store import ParquetHistoryStore


def main(argv: list[str] | None = None) -> int:
    """Build one user profile from the project Parquet artifacts."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    allowed_splits = ("train",) if args.phase == "validation" else (
        "train",
        "validation",
    )

    history_store = ParquetHistoryStore(args.artifact_version, args.splits)
    catalog_store = ParquetCatalogStore(args.artifact_version, args.catalog)

    profile = ProfileBuilder(history_store, catalog_store).build(
        ProfileRequest(
            user_id=args.user_id,
            as_of=args.as_of,
            phase=args.phase,
            artifact_version=args.artifact_version,
        ),
        ProfilePolicy(
            allowed_splits=allowed_splits,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            max_positive_signals=args.max_positive_signals,
            max_negative_signals=args.max_negative_signals,
            recency_half_life_days=args.recency_half_life_days,
        ),
    )
    payload = _profile_to_dict(profile)

    if args.output is None:
        _print_json(payload)
    else:
        _write_json(args.output, payload)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-profile",
        description="Build a user preference profile from Parquet artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build one user profile")

    build.add_argument("--splits", type=Path, required=True)
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--user-id", required=True)

    build.add_argument("--as-of", type=int, required=True)
    build.add_argument("--phase", choices=("validation", "test"), required=True)
    build.add_argument("--artifact-version", required=True)

    build.add_argument("--positive-threshold", type=float, default=4.0)
    build.add_argument("--negative-threshold", type=float, default=2.0)
    build.add_argument("--max-positive-signals", type=int, default=20)

    build.add_argument("--max-negative-signals", type=int, default=20)
    build.add_argument("--recency-half-life-days", type=float, default=90.0)
    build.add_argument("--output", type=Path)

    return parser


def _profile_to_dict(profile: UserProfile) -> dict[str, Any]:
    return {
        "schema_version": profile.schema_version,
        "user_id": profile.user_id,
        "as_of": profile.as_of,
        "phase": profile.phase,
        "artifact_version": profile.artifact_version,
        "seen_item_ids": sorted(profile.seen_item_ids),
        "positive_signals": [
            _signal_to_dict(signal) for signal in profile.positive_signals
        ],
        "negative_signals": [
            _signal_to_dict(signal) for signal in profile.negative_signals
        ],
        "diagnostics": {
            "valid_interaction_count": profile.diagnostics.valid_interaction_count,
            "duplicate_item_count": profile.diagnostics.duplicate_item_count,
            "positive_count_before_limit": profile.diagnostics.positive_count_before_limit,
            "negative_count_before_limit": profile.diagnostics.negative_count_before_limit,
            "text_covered_positive_count": profile.diagnostics.text_covered_positive_count,
            "missing_catalog_item_count": profile.diagnostics.missing_catalog_item_count,
            "warnings": list(profile.diagnostics.warnings),
        },
    }


def _signal_to_dict(signal: PreferenceSignal) -> dict[str, Any]:
    return {
        "history_id": signal.history_id,
        "item_id": signal.item_id,
        "rating": signal.rating,
        "timestamp": signal.timestamp,
        "sentiment": signal.sentiment,
        "weight": signal.weight,
        "text_fields": [
            {
                "name": field.name,
                "text": field.text,
                "source_id": field.source_id,
            }
            for field in signal.text_fields
        ],
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
