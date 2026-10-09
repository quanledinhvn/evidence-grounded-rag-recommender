from __future__ import annotations

import csv
import json
import math

from pathlib import Path
from typing import Any

from backend.data.io import source_sha256


_SOURCE_COLUMNS = ("user_id", "parent_asin", "rating", "timestamp", "history")
_BATCH_SIZE = 50_000


def import_pre_split_dataset(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Import an already split interaction dataset into project artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = output_dir / "splits.parquet"
    validation_output_path = output_dir / "queries_val.parquet"
    test_output_path = output_dir / "queries_test.parquet"

    catalog_items: set[str] = set()
    users_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    split_counts = {name: 0 for name in users_by_split}

    with _ParquetBuffer(splits_path, _interaction_schema()) as split_writer:
        with _ParquetBuffer(validation_output_path, _query_schema()) as validation_writer:
            with _ParquetBuffer(test_output_path, _query_schema()) as test_writer:
                sources = (
                    ("train", train_path, None),
                    ("validation", validation_path, validation_writer),
                    ("test", test_path, test_writer),
                )

                for split, source_path, query_writer in sources:
                    count = _import_source(
                        source_path,
                        split,
                        split_writer,
                        query_writer,
                        catalog_items,
                        users_by_split[split],
                    )
                    split_counts[split] = count

    catalog_path = output_dir / "catalog.parquet"
    with _ParquetBuffer(catalog_path, _catalog_schema()) as catalog_writer:
        for item_id in sorted(catalog_items):
            catalog_writer.add(_catalog_row(item_id))

    source_hashes = {
        "train": source_sha256(train_path),
        "validation": source_sha256(validation_path),
        "test": source_sha256(test_path),
    }
    stats = {
        "schema_version": "amazon-2023-presplit-v1",
        "source_format": "amazon-reviews-2023-sequential-csv",
        "timestamp_unit": "milliseconds",
        "source_hashes": source_hashes,
        "split_counts": split_counts,
        "query_counts": {
            "validation": split_counts["validation"],
            "test": split_counts["test"],
        },
        "users_by_split": {
            name: len(users)
            for name, users in users_by_split.items()
        },
        "catalog_items": len(catalog_items),
        "catalog_items_with_bm25_text": 0,
        "outputs": {
            "splits": str(splits_path),
            "catalog": str(catalog_path),
            "validation_queries": str(validation_output_path),
            "test_queries": str(test_output_path),
        },
    }
    _write_json(output_dir / "split_stats.json", stats)

    return stats


def _import_source(
    path: Path,
    split: str,
    split_writer: _ParquetBuffer,
    query_writer: _ParquetBuffer | None,
    catalog_items: set[str],
    users: set[str],
) -> int:
    count = 0

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _SOURCE_COLUMNS:
            message = f"Unexpected CSV header in {path}; expected {_SOURCE_COLUMNS!r}"
            raise ValueError(message)

        for line_number, raw in enumerate(reader, start=2):
            row = _normalize_row(raw, path, line_number, split)
            split_writer.add(row)

            catalog_items.add(str(row["item_id"]))
            users.add(str(row["user_id"]))

            if query_writer is not None:
                query_writer.add(_query_row(row))

            count += 1

    return count


def _normalize_row(
    raw: dict[str | None, str | list[str] | None],
    path: Path,
    line_number: int,
    split: str,
) -> dict[str, Any]:
    if None in raw:
        raise ValueError(f"Unexpected extra CSV fields at {path}:{line_number}")

    user_id = str(raw.get("user_id") or "").strip()
    item_id = str(raw.get("parent_asin") or "").strip()
    if not user_id or not item_id:
        raise ValueError(f"Empty user_id or parent_asin at {path}:{line_number}")

    rating = _parse_rating(raw.get("rating"), path, line_number)
    timestamp = _parse_timestamp(raw.get("timestamp"), path, line_number)

    history = str(raw.get("history") or "").split()
    if item_id in history:
        message = f"target item appears in history at {path}:{line_number}"
        raise ValueError(message)

    review_id = f"amazon-2023:{split}:{line_number}"

    return {
        "user_id": user_id,
        "item_id": item_id,
        "rating": rating,
        "timestamp": timestamp,
        "review_id": review_id,
        "review_text": "",
        "summary": "",
        "text": "",
        "source_line": line_number,
        "split": split,
    }


def _parse_rating(value: str | list[str] | None, path: Path, line_number: int) -> float:
    try:
        rating = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rating at {path}:{line_number}") from exc

    if not math.isfinite(rating) or not 1.0 <= rating <= 5.0:
        raise ValueError(f"Invalid rating at {path}:{line_number}: {rating}")

    return rating


def _parse_timestamp(
    value: str | list[str] | None,
    path: Path,
    line_number: int,
) -> int:
    try:
        timestamp = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp at {path}:{line_number}") from exc

    if timestamp <= 0:
        raise ValueError(f"Invalid timestamp at {path}:{line_number}: {timestamp}")

    return timestamp


def _query_row(row: dict[str, Any]) -> dict[str, Any]:
    timestamp = int(row["timestamp"])

    return {
        "user_id": str(row["user_id"]),
        "as_of": timestamp,
        "target_item_id": str(row["item_id"]),
        "target_rating": float(row["rating"]),
        "target_timestamp": timestamp,
        "target_review_id": str(row["review_id"]),
    }


def _catalog_row(item_id: str) -> dict[str, Any]:
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


class _ParquetBuffer:
    def __init__(self, path: Path, schema: Any) -> None:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            message = "Parquet support requires the project dependency 'pyarrow'."
            raise RuntimeError(message) from exc

        path.parent.mkdir(parents=True, exist_ok=True)

        self._schema = schema
        self._writer = parquet.ParquetWriter(path, schema, compression="zstd")
        self._rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(row)

        if len(self._rows) >= _BATCH_SIZE:
            self._flush()

    def close(self) -> None:
        self._flush()
        self._writer.close()

    def __enter__(self) -> _ParquetBuffer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _flush(self) -> None:
        if not self._rows:
            return

        import pyarrow as pa

        table = pa.Table.from_pylist(self._rows, schema=self._schema)
        self._writer.write_table(table)
        self._rows.clear()


def _interaction_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("user_id", pa.string()),
            ("item_id", pa.string()),
            ("rating", pa.float64()),
            ("timestamp", pa.int64()),
            ("review_id", pa.string()),
            ("review_text", pa.string()),
            ("summary", pa.string()),
            ("text", pa.string()),
            ("source_line", pa.int64()),
            ("split", pa.string()),
        ]
    )


def _query_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("user_id", pa.string()),
            ("as_of", pa.int64()),
            ("target_item_id", pa.string()),
            ("target_rating", pa.float64()),
            ("target_timestamp", pa.int64()),
            ("target_review_id", pa.string()),
        ]
    )


def _catalog_schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [
            ("item_id", pa.string()),
            ("title", pa.string()),
            ("description", pa.string()),
            ("feature", pa.string()),
            ("category", pa.string()),
            ("metadata_text", pa.string()),
            ("source_id", pa.string()),
            ("source_line", pa.int64()),
        ]
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
