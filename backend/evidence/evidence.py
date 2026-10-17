"""Evidence package for one query: what the LLM is allowed to say, each piece with a citable id.

For the user (history evidence, ids H1, H2, ...):
    the liked / disliked items of the profile, with their catalog text and the user's own review.
For every candidate item (item evidence, ids E1, E2, ...):
    - metadata: title, genres, directors/cast and description from the catalog;
    - reviews: up to `reviews_per_item` reviews by *other* people written *before* as_of,
      chosen by word overlap with what the user liked, then by helpful votes.

Every piece keeps a `source_id` pointing back to the catalog row or raw review, so a validator
can check that a cited id exists and belongs to the right item.
"""

from __future__ import annotations

import re
import sqlite3

from typing import Any

from backend.data.reviews import item_reviews, user_review


WORD = re.compile(r"[a-z][a-z']+")
METADATA_FIELDS = ["title", "genres", "description"]


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


def shorten(text: str, max_words: int) -> str:
    parts = text.split()

    return text if len(parts) <= max_words else " ".join(parts[:max_words]) + " ..."


def history_evidence(profile: dict[str, Any], connection: sqlite3.Connection | None) -> dict[str, dict[str, Any]]:
    """H# entries: what the user liked or disliked, in their own words when available."""
    evidence = {}

    for signal in profile["positive_signals"] + profile["negative_signals"]:
        text = " ".join(field["text"] for field in signal["text_fields"] if field["name"] in ("title", "details"))
        own = user_review(connection, profile["user_id"], signal["item_id"], profile["as_of"]) if connection else None

        evidence[signal["history_id"]] = {
            "item_id": signal["item_id"],
            "sentiment": signal["sentiment"],
            "rating": signal["rating"],
            "text": shorten(text, 60),
            "own_review": shorten(own["text"], 80) if own else None,
            "source_id": own["review_id"] if own else f"history:{signal['item_id']}",
        }

    return evidence


def item_evidence(
    candidates: list[str],
    catalog: dict[str, dict[str, Any]],
    profile: dict[str, Any],
    connection: sqlite3.Connection | None,
    *,
    reviews_per_item: int = 2,
    max_words: int = 120,
) -> dict[str, dict[str, Any]]:
    """E# entries for every candidate, numbered in candidate order."""
    liked_words = set().union(*(words(field["text"]) for signal in profile["positive_signals"] for field in signal["text_fields"]))
    evidence: dict[str, dict[str, Any]] = {}

    def add(item_id: str, source_type: str, source_id: str, text: str, timestamp: int | None = None) -> None:
        evidence[f"E{len(evidence) + 1}"] = {
            "item_id": item_id,
            "source_type": source_type,
            "source_id": source_id,
            "timestamp": timestamp,
            "text": shorten(text, max_words),
        }

    for item_id in candidates:
        row = catalog.get(item_id) or {}

        for field in METADATA_FIELDS:
            if row.get(field):
                add(item_id, "metadata", f"{row.get('source_id')}:{field}", f"{field}: {row[field]}")

        # item_doc already contains "Directors: ..." and "Cast: ..." parts.
        people = [part for part in (row.get("item_doc") or "").split(". ") if part.startswith(("Directors:", "Cast:"))]
        if people:
            add(item_id, "metadata", f"{row.get('source_id')}:people", ". ".join(people))

        if connection is None:
            continue

        # Other people's reviews written before the query time; most related to the user's taste first.
        reviews = item_reviews(connection, item_id, before=profile["as_of"], exclude_user=profile["user_id"])
        reviews.sort(key=lambda review: (-len(words(review["text"]) & liked_words), -review["helpful_vote"]))

        for review in reviews[:reviews_per_item]:
            add(item_id, "review", review["review_id"], review["text"], review["timestamp"])

    return evidence


def evidence_package(
    profile: dict[str, Any],
    candidates: list[str],
    catalog: dict[str, dict[str, Any]],
    connection: sqlite3.Connection | None,
    **options: Any,
) -> dict[str, Any]:
    return {
        "user_id": profile["user_id"],
        "as_of": profile["as_of"],
        "candidates": candidates,
        "history": history_evidence(profile, connection),
        "evidence": item_evidence(candidates, catalog, profile, connection, **options),
    }
