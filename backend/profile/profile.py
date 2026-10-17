"""Build a user preference profile for one query (user_id, as_of).

Steps:
1. Load the user's interactions with timestamp < as_of, only from the splits allowed
   in the phase (validation: train; test: train + validation). This prevents leakage.
2. Keep one row per item (the latest one).
3. rating >= 4 -> positive signal, rating <= 2 -> negative signal, rating 3 -> neutral
   (neutral items are only "seen", so they are never recommended again).
4. Newest signals first, at most `max_signals` of each kind.
5. weight = rating strength * recency, where recency halves every `half_life_days`.
6. Attach the catalog text of each item with a source id, so later steps can cite it as [H1], [H2], ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ALLOWED_SPLITS = {"validation": ["train"], "test": ["train", "validation"]}
TEXT_FIELDS = ["title", "subtitle", "description", "feature", "main_category", "category", "details"]

POSITIVE_RATING = 4.0
NEGATIVE_RATING = 2.0
DAY_MS = 86_400_000


def load_history(splits_path: Path, user_id: str, as_of: int, phase: str) -> list[dict[str, Any]]:
    """Interactions of one user strictly before as_of, from the splits allowed in `phase`."""
    filters = [
        ("user_id", "=", user_id),
        ("timestamp", "<", as_of),
        ("split", "in", ALLOWED_SPLITS[phase]),
    ]
    columns = ["user_id", "item_id", "rating", "timestamp", "review_id", "split"]

    return pq.read_table(splits_path, columns=columns, filters=filters).to_pylist()


def load_catalog(catalog_path: Path, item_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Catalog rows of the given items, keyed by item_id."""
    if not item_ids:
        return {}

    table = pq.read_table(catalog_path, filters=[("item_id", "in", sorted(item_ids))])

    return {row["item_id"]: row for row in table.to_pylist()}


def build_profile(
    user_id: str,
    as_of: int,
    history: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    *,
    max_signals: int = 20,
    half_life_days: float = 90.0,
) -> dict[str, Any]:
    # 1. Safety net: never use anything at or after the query time.
    rows = [row for row in history if row["timestamp"] < as_of]

    # 2. One row per item: the latest interaction wins.
    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda row: (row["timestamp"], row["review_id"])):
        latest[row["item_id"]] = row

    # 3 + 4. Split by rating, newest first, keep at most max_signals of each kind.
    newest_first = sorted(latest.values(), key=lambda row: (-row["timestamp"], row["review_id"]))
    positives = [row for row in newest_first if row["rating"] >= POSITIVE_RATING][:max_signals]
    negatives = [row for row in newest_first if row["rating"] <= NEGATIVE_RATING][:max_signals]

    signals = []
    for number, row in enumerate(positives + negatives, start=1):
        signals.append(
            {
                "history_id": f"H{number}",
                "item_id": row["item_id"],
                "rating": row["rating"],
                "timestamp": row["timestamp"],
                "sentiment": "positive" if row["rating"] >= POSITIVE_RATING else "negative",
                "weight": signal_weight(row, as_of, half_life_days),
                "text_fields": text_fields(catalog.get(row["item_id"])),
            }
        )

    missing = sorted(item_id for item_id in latest if not text_fields(catalog.get(item_id)))

    return {
        "user_id": user_id,
        "as_of": as_of,
        "seen_item_ids": sorted(latest),
        "positive_signals": [signal for signal in signals if signal["sentiment"] == "positive"],
        "negative_signals": [signal for signal in signals if signal["sentiment"] == "negative"],
        "items_without_metadata": missing,
    }


def signal_weight(row: dict[str, Any], as_of: int, half_life_days: float) -> float:
    """|rating - 3| / 2 (1.0 for 5 or 1 stars, 0.5 for 4 or 2) times a recency decay."""
    age_days = max(0.0, (as_of - row["timestamp"]) / DAY_MS)
    recency = 2.0 ** (-age_days / half_life_days)

    return abs(row["rating"] - 3.0) / 2.0 * recency


def text_fields(item: dict[str, Any] | None) -> list[dict[str, str]]:
    """Non-empty text fields of a catalog item, each with a citable source id."""
    if item is None:
        return []

    fields = []
    for name in TEXT_FIELDS:
        text = (item.get(name) or "").strip()

        if text:
            # A title borrowed from another edition (backend/data/editions.py) cites that edition.
            source_id = item.get("title_source") if name == "title" else None
            fields.append({"name": name, "text": text, "source_id": source_id or f"{item.get('source_id')}:{name}"})

    return fields
