from __future__ import annotations

from typing import Any

from backend.cf.bundle import CFBundle


def recommend_popularity(
    bundle: CFBundle,
    user_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not 0 <= limit <= 50:
        raise ValueError("limit must be between 0 and 50")

    if limit == 0:
        return []

    excluded = bundle.seen_by_user.get(user_id, set())

    items: list[dict[str, Any]] = []
    for item_id, count in bundle.popularity:
        if item_id in excluded:
            continue

        items.append(
            {"item_id": item_id, "score": float(count), "source": "popularity"}
        )
        if len(items) == limit:
            break

    return items
