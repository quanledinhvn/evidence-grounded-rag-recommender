from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.data.io import read_parquet, write_parquet

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


def deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    retained: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        key = (str(row["user_id"]), str(row["item_id"]))
        current = retained.get(key)

        candidate_order = (int(row["timestamp"]), int(row["source_line"]))
        if current is None:
            retained[key] = row
            continue

        current_order = (
            int(current["timestamp"]),
            int(current["source_line"]),
        )
        if candidate_order < current_order:
            retained[key] = row

    return sorted(
        retained.values(),
        key=lambda row: (
            str(row["user_id"]),
            int(row["timestamp"]),
            int(row["source_line"]),
            str(row["item_id"]),
        ),
    )


def deduplicate_dataset(
    interactions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    rows = read_parquet(interactions_path)
    deduplicated = deduplicate_rows(rows)

    write_parquet(
        output_path,
        deduplicated,
        columns=_INTERACTION_COLUMNS,
    )

    return {
        "input_rows": len(rows),
        "output_rows": len(deduplicated),
        "removed_duplicate_interactions": len(rows) - len(deduplicated),
        "output": str(output_path),
    }
