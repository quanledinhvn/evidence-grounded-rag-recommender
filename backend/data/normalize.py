from __future__ import annotations

import hashlib
import html
import json

import re
from datetime import datetime
from datetime import timezone

from pathlib import Path
from typing import Any, Literal

from backend.data.io import iter_jsonl, source_sha256, write_parquet
from backend.data.validation import (
    is_valid_id,
    metadata_item_id,
    review_item_id,
    review_issues,
    review_rating,
    review_timestamp,
    review_user_id,
    usable_text_count,
)


CatalogPolicy = Literal["all_reviews", "metadata_text"]

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

_INTERACTION_COLUMNS = (
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "review_id",
    "review_text",
    "summary",
    "text",
    "source_line",
)

_CATALOG_COLUMNS = (
    "item_id",
    "title",
    "description",
    "feature",
    "category",
    "metadata_text",
    "source_id",
    "source_line",
)


def clean_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [clean_text(part) for part in value if isinstance(part, str)]
        return " ".join(part for part in parts if part)

    if not isinstance(value, str):
        return ""

    unescaped = html.unescape(value)
    without_tags = _HTML_TAG.sub(" ", unescaped)

    return _WHITESPACE.sub(" ", without_tags).strip()


def normalize_dataset(
    reviews_path: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    catalog_policy: CatalogPolicy = "all_reviews",
    audit_time: datetime | None = None,
) -> dict[str, Any]:
    if catalog_policy not in ("all_reviews", "metadata_text"):
        raise ValueError(f"Unsupported catalog policy: {catalog_policy}")

    review_hash = source_sha256(reviews_path)
    metadata_hash = source_sha256(metadata_path)
    selected_metadata, metadata_stats = _select_metadata(metadata_path, metadata_hash)

    raw_interactions: list[dict[str, Any]] = []
    invalid_review_lines: list[int] = []
    reviewed_items: set[str] = set()

    effective_audit_time = audit_time or datetime.now(timezone.utc)
    if effective_audit_time.tzinfo is None:
        effective_audit_time = effective_audit_time.replace(tzinfo=timezone.utc)
    else:
        effective_audit_time = effective_audit_time.astimezone(timezone.utc)

    for line_number, record in iter_jsonl(reviews_path):
        if review_issues(record, effective_audit_time):
            invalid_review_lines.append(line_number)
            continue

        item_id = str(review_item_id(record)).strip()
        reviewed_items.add(item_id)

        raw_interactions.append(
            _normalize_interaction(record, line_number, review_hash)
        )

    metadata_text_items = {
        item_id
        for item_id, selected in selected_metadata.items()
        if selected["metadata_text"]
    }
    if catalog_policy == "metadata_text":
        catalog_items = reviewed_items & metadata_text_items
    else:
        catalog_items = reviewed_items

    interactions = [
        row for row in raw_interactions if row["item_id"] in catalog_items
    ]
    catalog = [
        _catalog_row(item_id, selected_metadata.get(item_id))
        for item_id in sorted(catalog_items)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    interactions_path = output_dir / "interactions.parquet"
    catalog_path = output_dir / "catalog.parquet"

    write_parquet(
        interactions_path,
        interactions,
        columns=_INTERACTION_COLUMNS,
    )
    write_parquet(catalog_path, catalog, columns=_CATALOG_COLUMNS)

    stats = {
        "schema_version": "d1-normalize-v1",
        "catalog_policy": catalog_policy,
        "audit_time": effective_audit_time.isoformat(),
        "source_hashes": {
            "reviews": review_hash,
            "metadata": metadata_hash,
        },
        "review_rows_read": len(raw_interactions) + len(invalid_review_lines),
        "invalid_review_rows_removed": len(invalid_review_lines),
        "invalid_review_lines": invalid_review_lines,
        "valid_interactions_before_catalog_filter": len(raw_interactions),
        "interactions_written": len(interactions),
        "interactions_removed_by_catalog_policy": len(raw_interactions) - len(interactions),
        "review_catalog_items": len(reviewed_items),
        "catalog_items_written": len(catalog),
        "catalog_items_with_bm25_text": sum(
            bool(row["metadata_text"]) for row in catalog
        ),
        "metadata": metadata_stats,
        "outputs": {
            "interactions": str(interactions_path),
            "catalog": str(catalog_path),
        },
    }
    _write_json(output_dir / "normalization_stats.json", stats)

    return stats


def _select_metadata(
    path: Path,
    metadata_hash: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    selected: dict[str, dict[str, Any]] = {}
    rows_read = 0

    invalid_asin_rows = 0
    duplicate_rows_merged = 0

    for line_number, record in iter_jsonl(path):
        rows_read += 1
        raw_item_id = metadata_item_id(record)
        if not is_valid_id(raw_item_id):
            invalid_asin_rows += 1
            continue

        item_id = raw_item_id.strip()
        candidate = _normalize_metadata(record, line_number, metadata_hash)
        current = selected.get(item_id)

        if current is not None:
            duplicate_rows_merged += 1

        if current is None or candidate["usable_text_fields"] > current["usable_text_fields"]:
            selected[item_id] = candidate

    return selected, {
        "rows_read": rows_read,
        "invalid_asin_rows_removed": invalid_asin_rows,
        "duplicate_rows_merged": duplicate_rows_merged,
        "unique_valid_items": len(selected),
    }


def _normalize_metadata(
    record: dict[str, Any],
    line_number: int,
    metadata_hash: str,
) -> dict[str, Any]:
    title = clean_text(record.get("title"))
    description = clean_text(record.get("description"))

    feature = clean_text(record.get("feature", record.get("features")))
    category = clean_text(record.get("category", record.get("categories")))

    metadata_text = " ".join(
        part for part in (title, description, feature, category) if part
    )

    return {
        "item_id": str(metadata_item_id(record)).strip(),
        "title": title or None,
        "description": description,
        "feature": feature,
        "category": category,
        "metadata_text": metadata_text,
        "source_id": _stable_id(metadata_hash, line_number),
        "source_line": line_number,
        "usable_text_fields": usable_text_count(record),
    }


def _normalize_interaction(
    record: dict[str, Any],
    line_number: int,
    review_hash: str,
) -> dict[str, Any]:
    review_text = clean_text(record.get("reviewText", record.get("text")))
    summary = clean_text(record.get("summary", record.get("title")))
    evidence_text = " ".join(part for part in (summary, review_text) if part)

    return {
        "user_id": str(review_user_id(record)).strip(),
        "item_id": str(review_item_id(record)).strip(),
        "rating": float(review_rating(record)),
        "timestamp": int(review_timestamp(record)),
        "review_id": _stable_id(review_hash, line_number),
        "review_text": review_text,
        "summary": summary,
        "text": evidence_text,
        "source_line": line_number,
    }


def _catalog_row(
    item_id: str,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if metadata is not None:
        return {
            key: value
            for key, value in metadata.items()
            if key != "usable_text_fields"
        }

    return {
        "item_id": item_id,
        "title": None,
        "description": "",
        "feature": "",
        "category": "",
        "metadata_text": "",
        "source_id": None,
        "source_line": None,
    }


def _stable_id(source_hash: str, line_number: int) -> str:
    payload = f"{source_hash}:{line_number}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
