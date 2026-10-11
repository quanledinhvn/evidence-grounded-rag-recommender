from __future__ import annotations

import hashlib
import json
import math

from pathlib import Path
from typing import Any

from backend.data.io import (
    iter_jsonl,
    read_parquet,
    source_sha256,
    write_parquet,
)
from backend.data.normalize import clean_text
from backend.data.validation import metadata_item_id


_CATALOG_COLUMNS = (
    "item_id",
    "title",
    "subtitle",
    "description",
    "feature",
    "main_category",
    "category",
    "store",
    "details",
    "metadata_text",
    "average_rating",
    "rating_number",
    "source_id",
    "source_line",
)

_DETAIL_FIELDS = (
    "Genre",
    "Actors",
    "Starring",
    "Directors",
    "Director",
    "Producers",
    "Writers",
    "Studio",
    "Release date",
    "Initial release date",
    "Run time",
    "Runtime",
    "MPAA rating",
    "Content advisory",
    "Language",
    "Audio languages",
    "Subtitles",
    "Dubbed",
    "Audio Description",
    "Media Format",
    "Format",
    "Aspect Ratio",
)

_COVERAGE_FIELDS = (
    "title",
    "subtitle",
    "description",
    "feature",
    "main_category",
    "category",
    "store",
    "details",
    "metadata_text",
    "average_rating",
    "rating_number",
)


def enrich_catalog(
    catalog_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.resolve() in (catalog_path.resolve(), metadata_path.resolve()):
        raise ValueError("Enriched catalog output must not overwrite an input file")

    catalog_rows = read_parquet(catalog_path)
    catalog_ids = _catalog_ids(catalog_rows)

    metadata_hash = source_sha256(metadata_path)
    selected: dict[str, dict[str, Any]] = {}

    rows_read = 0
    matched_rows = 0
    invalid_id_rows = 0
    duplicate_rows = 0

    for line_number, record in iter_jsonl(metadata_path):
        rows_read += 1
        item_id = _item_id(record)

        if not item_id:
            invalid_id_rows += 1
            continue

        if item_id not in catalog_ids:
            continue

        matched_rows += 1
        candidate = _metadata_row(record, item_id, line_number, metadata_hash)
        current = selected.get(item_id)

        if current is not None:
            duplicate_rows += 1

        if current is None or candidate["text_field_count"] > current["text_field_count"]:
            selected[item_id] = candidate

    enriched_rows = [
        _enriched_row(row, selected.get(str(row["item_id"])))
        for row in catalog_rows
    ]
    write_parquet(output_path, enriched_rows, columns=_CATALOG_COLUMNS)

    items_with_text = sum(bool(row["metadata_text"]) for row in enriched_rows)
    catalog_count = len(catalog_rows)

    stats = {
        "schema_version": "catalog-enrichment-v2",
        "source_hashes": {
            "catalog": source_sha256(catalog_path),
            "metadata": metadata_hash,
        },
        "metadata_rows_read": rows_read,
        "invalid_metadata_id_rows": invalid_id_rows,
        "matched_metadata_rows": matched_rows,
        "duplicate_metadata_rows": duplicate_rows,
        "catalog_items": catalog_count,
        "catalog_items_with_metadata_row": len(selected),
        "catalog_items_with_text": items_with_text,
        "catalog_items_without_text": catalog_count - items_with_text,
        "field_coverage": {
            field: sum(_has_value(row.get(field)) for row in enriched_rows)
            for field in _COVERAGE_FIELDS
        },
        "text_coverage": items_with_text / catalog_count if catalog_count else 0.0,
        "outputs": {
            "catalog": str(output_path),
            "stats": str(output_path.with_suffix(".stats.json")),
        },
    }
    _write_json(output_path.with_suffix(".stats.json"), stats)

    return stats


def _catalog_ids(rows: list[dict[str, Any]]) -> frozenset[str]:
    item_ids: set[str] = set()

    for row in rows:
        item_id = str(row.get("item_id") or "").strip()
        if not item_id:
            raise ValueError("Catalog contains an empty item_id")

        if item_id in item_ids:
            raise ValueError(f"Duplicate catalog item: {item_id}")

        item_ids.add(item_id)

    return frozenset(item_ids)


def _item_id(record: dict[str, Any]) -> str:
    raw_item_id = metadata_item_id(record)

    return raw_item_id.strip() if isinstance(raw_item_id, str) else ""


def _metadata_row(
    record: dict[str, Any],
    item_id: str,
    line_number: int,
    metadata_hash: str,
) -> dict[str, Any]:
    title = clean_text(record.get("title"))
    subtitle = clean_text(record.get("subtitle"))
    description = clean_text(record.get("description"))

    feature = clean_text(record.get("feature", record.get("features")))
    main_category = clean_text(record.get("main_category"))
    category = _category_text(record, main_category)

    store = clean_text(record.get("store"))
    details = _details_text(record.get("details"))

    text_parts = (
        title,
        subtitle,
        description,
        feature,
        main_category,
        category,
        details,
    )
    metadata_text = " ".join(part for part in text_parts if part)

    return {
        "item_id": item_id,
        "title": title or None,
        "subtitle": subtitle or None,
        "description": description,
        "feature": feature,
        "main_category": main_category,
        "category": category,
        "store": store,
        "details": details,
        "metadata_text": metadata_text,
        "average_rating": _optional_float(record.get("average_rating")),
        "rating_number": _optional_int(record.get("rating_number")),
        "source_id": _source_id(metadata_hash, line_number),
        "source_line": line_number,
        "text_field_count": sum(bool(part) for part in text_parts),
    }


def _category_text(record: dict[str, Any], main_category: str) -> str:
    values: list[str] = []

    raw_categories = record.get("category", record.get("categories"))
    if isinstance(raw_categories, list):
        categories = [clean_text(value) for value in raw_categories]
    else:
        categories = [clean_text(raw_categories)]

    for category in categories:
        if category and category != main_category and category not in values:
            values.append(category)

    return " ".join(values)


def _details_text(value: object) -> str:
    if not isinstance(value, dict):
        return ""

    parts: list[str] = []
    for field in _DETAIL_FIELDS:
        detail = _detail_value(value.get(field))

        if detail:
            parts.append(f"{field}: {detail}")

    return " ".join(parts)


def _detail_value(value: object) -> str:
    if isinstance(value, (str, list)):
        return clean_text(value)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)

    return ""


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None

    return result if result >= 0 else None


def _has_value(value: object) -> bool:
    if isinstance(value, str):
        return bool(value)

    return value is not None


def _enriched_row(
    catalog_row: dict[str, Any],
    metadata_row: dict[str, Any] | None,
) -> dict[str, Any]:
    if metadata_row is None:
        return {
            column: catalog_row.get(column)
            for column in _CATALOG_COLUMNS
        }

    return {
        column: metadata_row.get(column)
        for column in _CATALOG_COLUMNS
    }


def _source_id(metadata_hash: str, line_number: int) -> str:
    payload = f"{metadata_hash}:{line_number}".encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
