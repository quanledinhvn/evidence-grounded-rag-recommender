"""Import the three Amazon Reviews 2023 CSV splits into Parquet files.

Input: train / validation / test CSV files with the columns
    user_id, parent_asin, rating, timestamp, history

Output (in output_dir):
    splits.parquet        every interaction, with the split it came from
    queries_val.parquet   one row per validation target (what the evaluator must predict)
    queries_test.parquet  same for test
    catalog.parquet       every item id that appears in any split
    split_stats.json      row/user counts and input file hashes

The splits are kept exactly as published; nothing is re-split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from backend.data.io import sha256_file, write_json


SOURCE_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp", "history"]

COLUMN_TYPES = {
    "user_id": pa.string(),
    "parent_asin": pa.string(),
    "rating": pa.float64(),
    "timestamp": pa.int64(),
    "history": pa.string(),
}


def import_splits(
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {"train": train_path, "validation": validation_path, "test": test_path}
    tables = {split: read_split(path, split) for split, path in sources.items()}

    interactions = pa.concat_tables(list(tables.values()))
    pq.write_table(interactions, output_dir / "splits.parquet")

    pq.write_table(make_queries(tables["validation"]), output_dir / "queries_val.parquet")
    pq.write_table(make_queries(tables["test"]), output_dir / "queries_test.parquet")

    item_ids = sorted(pc.unique(interactions["item_id"]).to_pylist())
    pq.write_table(pa.table({"item_id": item_ids}), output_dir / "catalog.parquet")

    stats = {
        "split_counts": {split: table.num_rows for split, table in tables.items()},
        "users_by_split": {
            split: len(pc.unique(table["user_id"])) for split, table in tables.items()
        },
        "catalog_items": len(item_ids),
        "source_hashes": {split: sha256_file(path) for split, path in sources.items()},
        "timestamp_unit": "milliseconds",
    }
    write_json(output_dir / "split_stats.json", stats)

    return stats


def read_split(path: Path, split: str) -> pa.Table:
    """Read one CSV, check it, and convert it to the project interaction schema."""
    convert = pacsv.ConvertOptions(column_types=COLUMN_TYPES)
    source = pacsv.read_csv(path, convert_options=convert)

    if source.column_names != SOURCE_COLUMNS:
        raise ValueError(f"{path}: expected columns {SOURCE_COLUMNS}, got {source.column_names}")

    check_rows(source, path)

    # review_id points back to the CSV line (line 1 is the header).
    line_numbers = range(2, source.num_rows + 2)
    review_ids = [f"amazon-2023:{split}:{line}" for line in line_numbers]

    return pa.table(
        {
            "user_id": source["user_id"],
            "item_id": source["parent_asin"],
            "rating": source["rating"],
            "timestamp": source["timestamp"],
            "review_id": pa.array(review_ids, pa.string()),
            "split": pa.array([split] * source.num_rows, pa.string()),
        }
    )


def check_rows(source: pa.Table, path: Path) -> None:
    """Reject rows that would silently corrupt training or evaluation."""
    for column in ("user_id", "parent_asin", "rating", "timestamp"):
        if source[column].null_count:
            raise ValueError(f"{path}: empty {column}")

    for column in ("user_id", "parent_asin"):
        if pc.any(pc.equal(pc.utf8_trim_whitespace(source[column]), "")).as_py():
            raise ValueError(f"{path}: empty {column}")

    ratings = source["rating"]
    if pc.min(ratings).as_py() < 1.0 or pc.max(ratings).as_py() > 5.0:
        raise ValueError(f"{path}: rating outside 1..5")

    if pc.min(source["timestamp"]).as_py() <= 0:
        raise ValueError(f"{path}: timestamp must be positive")

    # The target of a row must not already be in that row's history,
    # otherwise the model could "predict" an item the user has already seen.
    items = source["parent_asin"].to_pylist()
    histories = source["history"].to_pylist()

    for line, (item, history) in enumerate(zip(items, histories), start=2):
        if history and item in history.split():
            raise ValueError(f"{path}:{line}: target item appears in its own history")


def make_queries(split_table: pa.Table) -> pa.Table:
    """Evaluation targets: predict item_id for user_id using only data before as_of."""
    return pa.table(
        {
            "user_id": split_table["user_id"],
            "as_of": split_table["timestamp"],
            "target_item_id": split_table["item_id"],
            "target_rating": split_table["rating"],
            "target_timestamp": split_table["timestamp"],
            "target_review_id": split_table["review_id"],
        }
    )
