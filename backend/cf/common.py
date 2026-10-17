"""Pieces shared by every CF model: the fit data, popularity and top-k selection.

Users and items are referred to by their position (index) in `user_ids` / `item_ids`.
`item_ids` is sorted, so ordering by index is the same as ordering by item id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scipy.sparse import csr_matrix


POSITIVE_RATING = 4.0

# Splits a model may learn from in each phase. "test" is never used for fitting.
FIT_SPLITS = {"validation": ["train"], "test": ["train", "validation"]}


@dataclass
class FitData:
    """Interactions a model learns from, grouped by user and sorted by time.

    The history of user number u is items[indptr[u]:indptr[u + 1]] (oldest first),
    with the matching ratings in ratings[indptr[u]:indptr[u + 1]].
    """

    phase: str
    user_ids: list[str]
    item_ids: list[str]

    indptr: np.ndarray
    items: np.ndarray
    ratings: np.ndarray

    def __post_init__(self) -> None:
        self.user_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.item_index = {item_id: index for index, item_id in enumerate(self.item_ids)}

    def history(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        """(item indices, ratings) of one user, oldest first. Empty for unknown users."""
        user = self.user_index.get(user_id)
        if user is None:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

        start, end = self.indptr[user], self.indptr[user + 1]

        return self.items[start:end], self.ratings[start:end]

    def matrix(self, min_rating: float = 1.0) -> csr_matrix:
        """Binary user x item matrix of the interactions with rating >= min_rating."""
        users = np.repeat(np.arange(len(self.user_ids)), np.diff(self.indptr))
        keep = self.ratings >= min_rating
        ones = np.ones(int(keep.sum()), dtype=np.float32)

        shape = (len(self.user_ids), len(self.item_ids))

        return csr_matrix((ones, (users[keep], self.items[keep])), shape=shape)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            phase=np.array(self.phase),
            user_ids=np.array(self.user_ids, dtype=str),
            item_ids=np.array(self.item_ids, dtype=str),
            indptr=self.indptr,
            items=self.items,
            ratings=self.ratings,
        )

    @classmethod
    def load(cls, path: Path) -> FitData:
        with np.load(path) as arrays:
            return cls(
                phase=str(arrays["phase"]),
                user_ids=arrays["user_ids"].tolist(),
                item_ids=arrays["item_ids"].tolist(),
                indptr=arrays["indptr"],
                items=arrays["items"],
                ratings=arrays["ratings"],
            )


def load_fit_data(splits_path: Path, catalog_path: Path, phase: str) -> FitData:
    """Read the interactions a model may learn from in `phase` and index them."""
    item_ids = sorted(pq.read_table(catalog_path, columns=["item_id"])["item_id"].to_pylist())
    columns = ["user_id", "item_id", "rating", "timestamp"]
    table = pq.read_table(splits_path, columns=columns, filters=[("split", "in", FIT_SPLITS[phase])])

    user_ids = sorted(pc.unique(table["user_id"]).to_pylist())
    users = pc.index_in(table["user_id"], value_set=pa.array(user_ids, pa.string())).to_numpy()
    items = pc.index_in(table["item_id"], value_set=pa.array(item_ids, pa.string()))

    if items.null_count:
        raise ValueError("splits contain items that are not in the catalog")

    items = items.to_numpy()
    if len(np.unique(users.astype(np.int64) * len(item_ids) + items)) != len(items):
        raise ValueError("splits contain the same user-item pair twice")

    # Sort by user, then time, then item: each user's history becomes one ordered block.
    order = np.lexsort((items, table["timestamp"].to_numpy(), users))
    indptr = np.concatenate(([0], np.cumsum(np.bincount(users, minlength=len(user_ids)))))
    ratings = table["rating"].to_numpy()

    return FitData(
        phase=phase,
        user_ids=user_ids,
        item_ids=item_ids,
        indptr=indptr.astype(np.int64),
        items=items[order].astype(np.int32),
        ratings=ratings[order].astype(np.float32),
    )


def top_k(indices: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k highest finite scores, best first; equal scores are ordered by item index."""
    keep = np.isfinite(scores)
    indices, scores = indices[keep], scores[keep]

    if len(scores) > k:
        # Cheap pre-filter: everything at least as good as the k-th best score.
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        keep = scores >= threshold
        indices, scores = indices[keep], scores[keep]

    order = np.lexsort((indices, -scores))[:k]

    return indices[order], scores[order]


class Popularity:
    """Items ranked by their number of positive (rating >= 4) interactions.

    Used on its own as the non-personalised baseline, and by the other models to fill
    the list when they cannot score enough items (e.g. a user without history).
    """

    def __init__(self, data: FitData) -> None:
        self.data = data
        positives = data.items[data.ratings >= POSITIVE_RATING]
        self.counts = np.bincount(positives, minlength=len(data.item_ids))

        # Most positives first; ties by item index.
        self.order = np.lexsort((np.arange(len(self.counts)), -self.counts))

    def fill(self, chosen: list[int], excluded: set[int], limit: int) -> list[int]:
        """Most popular items that are not excluded or chosen, up to `limit` items in total."""
        taken = excluded | set(chosen)
        extra: list[int] = []

        for item in self.order:
            if len(chosen) + len(extra) >= limit:
                break

            if int(item) not in taken:
                extra.append(int(item))

        return extra

    def recommend(self, user_id: str, limit: int = 10, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        excluded = seen_items(self.data, user_id) | item_indices(self.data, exclude)

        return rank_items(self, [], [], excluded, limit, "popularity")

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        return [self.recommend(user_id, limit) for user_id in user_ids]


def seen_items(data: FitData, user_id: str) -> set[int]:
    """Every item the user interacted with in the fit data (any rating)."""
    items, _ = data.history(user_id)

    return set(items.tolist())


def item_indices(data: FitData, item_ids: tuple[str, ...]) -> set[int]:
    return {data.item_index[item_id] for item_id in item_ids if item_id in data.item_index}


def rank_items(
    popularity: Popularity,
    indices: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    excluded: set[int],
    limit: int,
    source: str,
) -> list[dict[str, Any]]:
    """Final list: the model's own top items first, then popular items to reach `limit`."""
    if not 0 <= limit <= 50:
        raise ValueError("limit must be between 0 and 50")

    data = popularity.data
    ranked = [
        {"item_id": data.item_ids[int(item)], "score": float(score), "source": source}
        for item, score in zip(indices, scores)
    ][:limit]

    chosen = [data.item_index[item["item_id"]] for item in ranked]
    for item in popularity.fill(chosen, excluded, limit):
        ranked.append({"item_id": data.item_ids[item], "score": float(popularity.counts[item]), "source": "popularity"})

    return ranked
