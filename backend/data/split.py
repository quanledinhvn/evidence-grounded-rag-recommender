from __future__ import annotations

import json

from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.data.io import (
    read_parquet,
    source_sha256,
    write_parquet,
)


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
    "split",
)

_QUERY_COLUMNS = (
    "user_id",
    "as_of",
    "target_item_id",
    "target_rating",
    "target_timestamp",
    "target_review_id",
)


def split_rows(
    rows: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()

    for row in rows:
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        pair = (user_id, item_id)

        if pair in seen_pairs:
            message = f"Duplicate user-item pair after D2: {pair!r}"
            raise ValueError(message)

        seen_pairs.add(pair)
        grouped[user_id].append(row)

    splits: list[dict[str, Any]] = []
    validation_queries: list[dict[str, Any]] = []
    test_queries: list[dict[str, Any]] = []

    for user_id in sorted(grouped):
        history = sorted(
            grouped[user_id],
            key=lambda row: (int(row["timestamp"]), str(row["item_id"])),
        )

        if len(history) < 3:
            splits.extend(_with_split(row, "train") for row in history)
            continue

        train_rows = history[:-2]
        validation_target = history[-2]
        test_target = history[-1]

        splits.extend(_with_split(row, "train") for row in train_rows)
        splits.append(_with_split(validation_target, "validation"))
        splits.append(_with_split(test_target, "test"))

        validation_queries.append(_query_row(validation_target))
        test_queries.append(_query_row(test_target))

    return splits, validation_queries, test_queries


def split_dataset(
    interactions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    rows = read_parquet(interactions_path)
    splits, validation_queries, test_queries = split_rows(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    splits_path = output_dir / "splits.parquet"
    validation_path = output_dir / "queries_val.parquet"
    test_path = output_dir / "queries_test.parquet"

    write_parquet(splits_path, splits, columns=_INTERACTION_COLUMNS)
    write_parquet(
        validation_path,
        validation_queries,
        columns=_QUERY_COLUMNS,
    )
    write_parquet(test_path, test_queries, columns=_QUERY_COLUMNS)

    split_counts = {
        name: sum(row["split"] == name for row in splits)
        for name in ("train", "validation", "test")
    }
    total_users = len({str(row["user_id"]) for row in rows})
    eligible_users = len(validation_queries)

    stats = {
        "schema_version": "d3-time-split-v1",
        "input_sha256": source_sha256(interactions_path),
        "input_rows": len(rows),
        "total_users": total_users,
        "eligible_users": eligible_users,
        "excluded_evaluation_users": total_users - eligible_users,
        "split_counts": split_counts,
        "query_counts": {
            "validation": len(validation_queries),
            "test": len(test_queries),
        },
        "outputs": {
            "splits": str(splits_path),
            "validation_queries": str(validation_path),
            "test_queries": str(test_path),
        },
    }
    _write_json(output_dir / "split_stats.json", stats)

    return stats


def _with_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {**row, "split": split}


def _query_row(target: dict[str, Any]) -> dict[str, Any]:
    timestamp = int(target["timestamp"])

    return {
        "user_id": str(target["user_id"]),
        "as_of": timestamp,
        "target_item_id": str(target["item_id"]),
        "target_rating": float(target["rating"]),
        "target_timestamp": timestamp,
        "target_review_id": str(target["review_id"]),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
