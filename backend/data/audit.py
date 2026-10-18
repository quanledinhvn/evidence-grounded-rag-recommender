from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.data.io import iter_jsonl, source_sha256
from backend.data.validation import (
    is_valid_id,
    metadata_item_id,
    review_date_mismatch,
    review_issues,
    review_item_id,
    review_user_id,
    usable_text_count,
)

_REVIEW_METRICS = (
    "review_rows",
    "unique_valid_user_ids",
    "unique_valid_review_items",
    "invalid_review_user_id",
    "invalid_review_item_id",
    "invalid_review_rating",
    "invalid_review_timestamp",
    "review_date_mismatch",
    "exact_duplicate_groups",
    "exact_duplicate_excess_rows",
    "duplicate_user_item_pairs",
    "duplicate_user_item_excess_interactions",
)

_REVIEW_ERROR_LISTS = (
    "invalid_review_user_id_lines",
    "invalid_review_item_id_lines",
    "invalid_review_rating_lines",
    "invalid_review_timestamp_lines",
    "review_date_mismatch_lines",
    "invalid_review_lines",
)

_METADATA_METRICS = (
    "metadata_rows",
    "invalid_metadata_asin",
    "metadata_without_text",
    "duplicate_metadata_asin_groups",
    "duplicate_metadata_excess_rows",
)

_METADATA_ERROR_LISTS = (
    "invalid_metadata_asin_lines",
    "metadata_without_text_lines",
)


def audit_dataset(
    reviews_path: Path,
    metadata_path: Path,
    *,
    audit_time: datetime | None = None,
) -> dict[str, Any]:
    effective_audit_time = audit_time or datetime.now(UTC)
    if effective_audit_time.tzinfo is None:
        effective_audit_time = effective_audit_time.replace(tzinfo=UTC)

    review_state = _audit_reviews(reviews_path, effective_audit_time)
    metadata_state = _audit_metadata(metadata_path)

    review_items = review_state["valid_items"]
    metadata_items = metadata_state["valid_items"]
    metadata_text_items = metadata_state["text_items"]

    missing_metadata = review_items - metadata_items
    missing_text = (review_items & metadata_items) - metadata_text_items
    unused_metadata = metadata_items - review_items

    metrics = dict(review_state["metrics"])
    metrics.update(metadata_state["metrics"])
    metrics.update(
        {
            "metadata_key_coverage": _coverage(review_items, metadata_items),
            "metadata_text_coverage": _coverage(review_items, metadata_text_items),
            "review_items_missing_metadata": len(missing_metadata),
            "review_items_missing_bm25_text": len(missing_text),
            "unused_metadata_items": len(unused_metadata),
        }
    )

    errors = dict(review_state["errors"])
    errors.update(metadata_state["errors"])
    errors.update(
        {
            "review_items_missing_metadata": sorted(missing_metadata),
            "review_items_missing_bm25_text": sorted(missing_text),
            "unused_metadata_items": sorted(unused_metadata),
        }
    )

    return {
        "schema_version": "d0-audit-v1",
        "audit_time": effective_audit_time.astimezone(UTC).isoformat(),
        "sources": {
            "reviews": _source_details(reviews_path),
            "metadata": _source_details(metadata_path),
        },
        "metrics": metrics,
        "errors": errors,
    }


def write_audit_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _audit_reviews(path: Path, audit_time: datetime) -> dict[str, Any]:
    metrics: Counter[str] = Counter()
    errors: dict[str, list[Any]] = defaultdict(list)

    valid_users: set[str] = set()
    valid_items: set[str] = set()

    canonical_lines: dict[str, list[int]] = defaultdict(list)
    user_item_lines: dict[tuple[str, str], list[int]] = defaultdict(list)

    for line_number, record in iter_jsonl(path):
        metrics["review_rows"] += 1
        issues = review_issues(record, audit_time)

        raw_user_id = review_user_id(record)
        if is_valid_id(raw_user_id):
            valid_users.add(raw_user_id.strip())

        _record_review_errors(errors, line_number, issues)
        for issue in issues:
            metric_name = {
                "user_id": "invalid_review_user_id",
                "item_id": "invalid_review_item_id",
                "rating": "invalid_review_rating",
                "timestamp": "invalid_review_timestamp",
            }[issue]
            metrics[metric_name] += 1

        if review_date_mismatch(record):
            metrics["review_date_mismatch"] += 1
            errors["review_date_mismatch_lines"].append(line_number)

        canonical_hash = _canonical_hash(record)
        canonical_lines[canonical_hash].append(line_number)

        if issues:
            errors["invalid_review_lines"].append(line_number)
            continue

        user_id = str(review_user_id(record)).strip()
        item_id = str(review_item_id(record)).strip()
        valid_items.add(item_id)

        user_item_lines[(user_id, item_id)].append(line_number)

    exact_groups = [lines for lines in canonical_lines.values() if len(lines) > 1]
    pair_groups = [lines for lines in user_item_lines.values() if len(lines) > 1]

    metrics.update(
        {
            "unique_valid_user_ids": len(valid_users),
            "unique_valid_review_items": len(valid_items),
            "exact_duplicate_groups": len(exact_groups),
            "exact_duplicate_excess_rows": sum(len(lines) - 1 for lines in exact_groups),
            "duplicate_user_item_pairs": len(pair_groups),
            "duplicate_user_item_excess_interactions": sum(
                len(lines) - 1 for lines in pair_groups
            ),
        }
    )
    errors["exact_duplicate_groups"] = exact_groups
    errors["duplicate_user_item_groups"] = pair_groups

    for name in _REVIEW_METRICS:
        metrics.setdefault(name, 0)

    for name in _REVIEW_ERROR_LISTS:
        errors.setdefault(name, [])

    return {
        "metrics": dict(metrics),
        "errors": dict(errors),
        "valid_items": valid_items,
    }


def _record_review_errors(
    errors: dict[str, list[Any]],
    line_number: int,
    issues: set[str],
) -> None:
    error_names = {
        "user_id": "invalid_review_user_id_lines",
        "item_id": "invalid_review_item_id_lines",
        "rating": "invalid_review_rating_lines",
        "timestamp": "invalid_review_timestamp_lines",
    }

    for issue, error_name in error_names.items():
        if issue in issues:
            errors[error_name].append(line_number)


def _audit_metadata(path: Path) -> dict[str, Any]:
    metrics: Counter[str] = Counter()
    errors: dict[str, list[Any]] = defaultdict(list)

    valid_items: set[str] = set()
    text_items: set[str] = set()
    asin_lines: dict[str, list[int]] = defaultdict(list)

    for line_number, record in iter_jsonl(path):
        metrics["metadata_rows"] += 1
        asin = metadata_item_id(record)

        if not is_valid_id(asin):
            metrics["invalid_metadata_asin"] += 1
            errors["invalid_metadata_asin_lines"].append(line_number)
            continue

        item_id = asin.strip()
        valid_items.add(item_id)
        asin_lines[item_id].append(line_number)

        if usable_text_count(record) == 0:
            metrics["metadata_without_text"] += 1
            errors["metadata_without_text_lines"].append(line_number)
            continue

        text_items.add(item_id)

    duplicate_groups = [lines for lines in asin_lines.values() if len(lines) > 1]
    metrics["duplicate_metadata_asin_groups"] = len(duplicate_groups)
    metrics["duplicate_metadata_excess_rows"] = sum(
        len(lines) - 1 for lines in duplicate_groups
    )

    errors["duplicate_metadata_asin_groups"] = duplicate_groups

    for name in _METADATA_METRICS:
        metrics.setdefault(name, 0)

    for name in _METADATA_ERROR_LISTS:
        errors.setdefault(name, [])

    return {
        "metrics": dict(metrics),
        "errors": dict(errors),
        "valid_items": valid_items,
        "text_items": text_items,
    }


def _canonical_hash(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coverage(review_items: set[str], metadata_items: set[str]) -> float:
    if not review_items:
        return 0.0

    return len(review_items & metadata_items) / len(review_items)


def _source_details(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": source_sha256(path),
        "size_bytes": path.stat().st_size,
    }
