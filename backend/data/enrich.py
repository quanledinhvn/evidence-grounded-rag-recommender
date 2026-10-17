"""Attach Amazon metadata (title, description, cast, ...) to every catalog item.

Input:  catalog.parquet (item ids) and meta_Movies_and_TV.jsonl (one JSON per product)
Output: catalog_enriched.parquet (one row per catalog item) and <output>.stats.json

Items are matched exactly on parent_asin == item_id. `metadata_text` joins the content
fields into one document that BM25, embeddings and evidence retrieval can all use.
"""

from __future__ import annotations

import html
import re

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from backend.data.io import iter_jsonl, sha256_file, write_json


# Fields that describe the content of an item, in the order they appear in metadata_text.
TEXT_FIELDS = ["title", "subtitle", "description", "feature", "main_category", "category", "details"]

# Keys of the metadata "details" dict that are worth keeping.
DETAIL_KEYS = [
    "Genre", "Actors", "Starring", "Directors", "Director", "Producers", "Writers", "Studio",
    "Release date", "Initial release date", "Run time", "Runtime", "MPAA rating",
    "Content advisory", "Language", "Audio languages", "Subtitles", "Dubbed",
    "Audio Description", "Media Format", "Format", "Aspect Ratio",
]

CATALOG_SCHEMA = pa.schema(
    [(field, pa.string()) for field in TEXT_FIELDS]
    + [
        ("item_id", pa.string()),
        ("store", pa.string()),
        ("metadata_text", pa.string()),
        ("average_rating", pa.float64()),
        ("rating_number", pa.int64()),
        ("source_id", pa.string()),
    ]
)

_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


def enrich_catalog(catalog_path: Path, metadata_path: Path, output_path: Path) -> dict[str, Any]:
    item_ids = pq.read_table(catalog_path, columns=["item_id"])["item_id"].to_pylist()
    wanted = set(item_ids)
    best: dict[str, dict[str, Any]] = {}

    for record in iter_jsonl(metadata_path):
        item_id = record.get("parent_asin") or record.get("asin")
        if item_id not in wanted:
            continue

        row = metadata_row(item_id, record)

        # If an item appears twice, keep the record with more text.
        if item_id not in best or count_text(row) > count_text(best[item_id]):
            best[item_id] = row

    rows = [best.get(item_id) or empty_row(item_id) for item_id in item_ids]
    table = pa.Table.from_pylist(rows, schema=CATALOG_SCHEMA)
    pq.write_table(table, output_path)

    stats = {
        "catalog_items": len(rows),
        "catalog_items_with_metadata": len(best),
        "field_coverage": {
            field: sum(1 for row in rows if row[field]) for field in TEXT_FIELDS + ["metadata_text"]
        },
        "source_hashes": {"catalog": sha256_file(catalog_path), "metadata": sha256_file(metadata_path)},
    }
    write_json(output_path.with_suffix(".stats.json"), stats)

    return stats


def metadata_row(item_id: str, record: dict[str, Any]) -> dict[str, Any]:
    main_category = clean_text(record.get("main_category"))
    row = {
        "item_id": item_id,
        "title": clean_text(record.get("title")),
        "subtitle": clean_text(record.get("subtitle")),
        "description": clean_text(record.get("description")),
        "feature": clean_text(record.get("features", record.get("feature"))),
        "main_category": main_category,
        "category": category_text(record.get("categories", record.get("category")), main_category),
        "details": details_text(record.get("details")),
        "store": clean_text(record.get("store")),
        "average_rating": to_float(record.get("average_rating")),
        "rating_number": to_int(record.get("rating_number")),
        # Stable id that explanations can cite, e.g. "meta:B00271DNP4:description".
        "source_id": f"meta:{item_id}",
    }
    row["metadata_text"] = " ".join(row[field] for field in TEXT_FIELDS if row[field])

    return row


def empty_row(item_id: str) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in TEXT_FIELDS}
    row.update(item_id=item_id, store="", metadata_text="", average_rating=None, rating_number=None, source_id=None)

    return row


def count_text(row: dict[str, Any]) -> int:
    return sum(1 for field in TEXT_FIELDS if row[field])


def clean_text(value: Any) -> str:
    """Plain text: lists are joined, HTML entities/tags removed, whitespace collapsed."""
    if isinstance(value, list):
        return " ".join(part for part in (clean_text(item) for item in value) if part)

    if not isinstance(value, str):
        return ""

    text = _TAG.sub(" ", html.unescape(value))

    return _SPACES.sub(" ", text).strip()


def category_text(categories: Any, main_category: str) -> str:
    """Distinct categories, without repeating main_category."""
    values: list[str] = []

    for category in categories if isinstance(categories, list) else [categories]:
        text = clean_text(category)
        if text and text != main_category and text not in values:
            values.append(text)

    return " ".join(values)


def details_text(details: Any) -> str:
    """'Directors: X Starring: Y ...' from the useful keys of the details dict."""
    if not isinstance(details, dict):
        return ""

    parts = []
    for key in DETAIL_KEYS:
        value = details.get(key)
        text = str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else clean_text(value)

        if text:
            parts.append(f"{key}: {text}")

    return " ".join(parts)


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
