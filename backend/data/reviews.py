"""Review store: the raw Amazon reviews of catalog items, in SQLite, for evidence retrieval.

Input:  Movies_and_TV.jsonl (one review per line) and the catalog (which items we care about)
Output: reviews.sqlite with one table `reviews`

Only reviews that can serve as evidence are kept:
- the item is in the catalog,
- the text has at least `min_words` words (one-liners like "Five stars" say nothing),
- the review is about the film, not about shipping, the disc or the price.

Two indexes make the two lookups we need fast:
- reviews of an item written before a time  -> evidence [E#] from other people
- the review a user wrote for an item       -> the user's own words for the profile
"""

from __future__ import annotations

import re
import sqlite3

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from backend.data.enrich import clean_text
from backend.data.io import iter_jsonl


# Reviews that mention these are mostly about the product, not the film.
NOT_ABOUT_THE_FILM = re.compile(
    r"\b(ship(ped|ping)?|deliver(y|ed)|arrived|packag(e|ing)|seller|refund|return(ed)?|scratch(ed|es)?"
    r"|region code|won't play|wouldn't play|doesn't play|price|cheap(er)?)\b",
    re.IGNORECASE,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    review_id    TEXT PRIMARY KEY,
    item_id      TEXT NOT NULL,
    user_id      TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    rating       REAL,
    title        TEXT,
    text         TEXT NOT NULL,
    helpful_vote INTEGER
)
"""


def import_reviews(
    reviews_path: Path,
    catalog_path: Path,
    db_path: Path,
    *,
    min_words: int = 10,
    max_chars: int = 2000,
) -> dict[str, Any]:
    items = set(pq.read_table(catalog_path, columns=["item_id"])["item_id"].to_pylist())
    db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(db_path)

    connection.execute(SCHEMA)
    counts = {"read": 0, "not_in_catalog": 0, "too_short": 0, "not_about_film": 0, "kept": 0}
    batch: list[tuple[Any, ...]] = []

    for line, record in enumerate(iter_jsonl(reviews_path), start=1):
        counts["read"] += 1
        reason = skip_reason(record, items, min_words)

        if reason:
            counts[reason] += 1
            continue

        batch.append(review_row(line, record, max_chars))
        counts["kept"] += 1

        if len(batch) >= 50_000:
            insert(connection, batch)

    insert(connection, batch)

    # Indexes are built once at the end: much faster than updating them on every insert.
    connection.execute("CREATE INDEX idx_item_time ON reviews (item_id, timestamp)")
    connection.execute("CREATE INDEX idx_user_item ON reviews (user_id, item_id)")
    connection.commit()

    connection.close()

    return counts


def skip_reason(record: dict[str, Any], items: set[str], min_words: int) -> str:
    """Why a review is not kept ("" if it is kept)."""
    if (record.get("parent_asin") or record.get("asin")) not in items:
        return "not_in_catalog"

    text = clean_text(record.get("text"))
    if len(text.split()) < min_words:
        return "too_short"

    if NOT_ABOUT_THE_FILM.search(text):
        return "not_about_film"

    return ""


def review_row(line: int, record: dict[str, Any], max_chars: int) -> tuple[Any, ...]:
    return (
        f"review:{line}",                                   # line in the raw file, for tracing back
        record.get("parent_asin") or record.get("asin"),
        record.get("user_id"),
        int(record.get("timestamp") or 0),
        record.get("rating"),
        clean_text(record.get("title")),
        clean_text(record.get("text"))[:max_chars],
        int(record.get("helpful_vote") or 0),
    )


def insert(connection: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    connection.executemany("INSERT OR IGNORE INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
    connection.commit()
    batch.clear()


def item_reviews(
    connection: sqlite3.Connection,
    item_id: str,
    before: int,
    exclude_user: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Reviews of an item written strictly before `before` by anyone except `exclude_user`.

    Most helpful first, so the evidence shown to the LLM is the most informative.
    """
    rows = connection.execute(
        "SELECT review_id, item_id, user_id, timestamp, rating, title, text, helpful_vote FROM reviews"
        " WHERE item_id = ? AND timestamp < ? AND user_id != ?"
        " ORDER BY helpful_vote DESC, timestamp DESC LIMIT ?",
        (item_id, before, exclude_user, limit),
    ).fetchall()

    return [as_dict(row) for row in rows]


def user_review(connection: sqlite3.Connection, user_id: str, item_id: str, before: int) -> dict[str, Any] | None:
    """The review `user_id` wrote for `item_id` before `before`, if it was kept."""
    row = connection.execute(
        "SELECT review_id, item_id, user_id, timestamp, rating, title, text, helpful_vote FROM reviews"
        " WHERE user_id = ? AND item_id = ? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1",
        (user_id, item_id, before),
    ).fetchone()

    return as_dict(row) if row else None


def as_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    keys = ["review_id", "item_id", "user_id", "timestamp", "rating", "title", "text", "helpful_vote"]

    return dict(zip(keys, row))
