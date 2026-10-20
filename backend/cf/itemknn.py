"""ItemKNN: recommend items similar to what the user watched, recent items counting more.

For a user whose history is h_1 ... h_n (oldest to newest):

    score(j) = sum over k of  decay^(n - k) * sim(h_k, j)

- sim is the cosine similarity between two item columns of the user x item matrix,
  i.e. how often the two items are watched by the same people. Each item keeps only its
  `neighbors` most similar items.
- decay^(n - k) gives the newest item weight 1, the one before it `decay`, and so on:
  the next item a user watches is mostly predicted by what they watched last.
- Items with no score are filled from popularity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scipy.sparse import csr_matrix, diags, load_npz, save_npz

from backend.cf.common import FitData, Popularity, item_indices, rank_items, seen_items, top_k
from backend.data.io import read_json, write_json


def recency_weights(length: int, decay: float) -> np.ndarray:
    """Weights for a history ordered oldest to newest: [..., decay^2, decay, 1]."""
    return (decay ** np.arange(length - 1, -1, -1)).astype(np.float32)


def cosine_top_k(matrix: csr_matrix, neighbors: int, batch_size: int = 1000) -> csr_matrix:
    """Item x item cosine similarity, keeping the `neighbors` best per item (no self links)."""
    binary = (matrix > 0).astype(np.float32)
    degree = np.asarray(binary.sum(axis=0)).ravel()
    normalized = (binary @ diags(1.0 / np.sqrt(np.maximum(degree, 1.0)))).tocsr()

    by_item = normalized.T.tocsr()
    item_count = binary.shape[1]
    rows, columns, values = [], [], []

    for start in range(0, item_count, batch_size):
        # Cosine of items start..start+batch_size with every item.
        block = (by_item[start : start + batch_size] @ normalized).tocsr()

        for offset in range(block.shape[0]):
            item = start + offset
            low, high = block.indptr[offset], block.indptr[offset + 1]
            others = block.indices[low:high] != item

            best, scores = top_k(block.indices[low:high][others], block.data[low:high][others], neighbors)
            rows.append(np.full(len(best), item))
            columns.append(best)

            values.append(scores)

    similarity = csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(item_count, item_count),
        dtype=np.float32,
    )

    return similarity


class ItemKNN:
    name = "itemknn"

    def __init__(self, data: FitData, similarity: csr_matrix, config: dict[str, Any]) -> None:
        self.data = data
        self.similarity = similarity
        self.config = config

        self.popularity = Popularity(data)

    @classmethod
    def fit(
        cls,
        data: FitData,
        neighbors: int = 400,
        recency_decay: float = 0.7,
        min_rating: float = 1.0,
    ) -> ItemKNN:
        similarity = cosine_top_k(data.matrix(min_rating), neighbors)
        config = {
            "model": cls.name,
            "phase": data.phase,
            "neighbors": neighbors,
            "recency_decay": recency_decay,
            "min_rating": min_rating,
        }

        return cls(data, similarity, config)

    def recommend(
        self,
        user_id: str,
        limit: int = 10,
        exclude: tuple[str, ...] = (),
        explain: int = 3,
    ) -> list[dict[str, Any]]:
        """Top items for one user; each ItemKNN item says which watched items caused it."""
        used, weights = self._weighted_history(user_id)
        excluded = seen_items(self.data, user_id) | item_indices(self.data, exclude)

        scores = self._scores([used], [weights])
        ranked = self._rank(scores, 0, excluded, limit)

        for item in ranked:
            item["because"] = self._because(used, weights, item, explain)

        return ranked

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        histories = [self._weighted_history(user_id) for user_id in user_ids]
        scores = self._scores([used for used, _ in histories], [weights for _, weights in histories])

        return [
            self._rank(scores, row, seen_items(self.data, user_id), limit)
            for row, user_id in enumerate(user_ids)
        ]

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "config.json", self.config)
        self.data.save(model_dir / "fit_data.npz")

        save_npz(model_dir / "similarity.npz", self.similarity)

    @classmethod
    def load(cls, model_dir: Path) -> ItemKNN:
        data = FitData.load(model_dir / "fit_data.npz")
        similarity = load_npz(model_dir / "similarity.npz").tocsr()

        return cls(data, similarity, read_json(model_dir / "config.json"))

    def _weighted_history(self, user_id: str) -> tuple[np.ndarray, np.ndarray]:
        """History items allowed to vote (rating >= min_rating) and their recency weights."""
        items, ratings = self.data.history(user_id)
        used = items[ratings >= self.config["min_rating"]]

        return used, recency_weights(len(used), self.config["recency_decay"])

    def _scores(self, histories: list[np.ndarray], weights: list[np.ndarray]) -> csr_matrix:
        """One row of item scores per user: weights @ similarity[history items]."""
        rows = np.repeat(np.arange(len(histories)), [len(items) for items in histories])
        shape = (len(histories), len(self.data.item_ids))
        votes = csr_matrix((np.concatenate(weights), (rows, np.concatenate(histories))), shape=shape)

        return (votes @ self.similarity).tocsr()

    def _rank(self, scores: csr_matrix, row: int, excluded: set[int], limit: int) -> list[dict[str, Any]]:
        low, high = scores.indptr[row], scores.indptr[row + 1]
        indices, values = scores.indices[low:high], scores.data[low:high]
        keep = ~np.isin(indices, list(excluded))

        best, best_scores = top_k(indices[keep], values[keep], limit)

        return rank_items(self.popularity, best, best_scores, excluded, limit, self.name)

    def _because(
        self,
        used: np.ndarray,
        weights: np.ndarray,
        item: dict[str, Any],
        explain: int,
    ) -> list[dict[str, Any]]:
        """History items that contributed most to this item's score (evidence [H#])."""
        if item["source"] != self.name or not len(used) or explain <= 0:
            return []

        target = self.data.item_index[item["item_id"]]
        similarities = self.similarity[used][:, target].toarray().ravel()
        contributions = similarities * weights

        order = np.lexsort((used, -contributions))[:explain]

        return [
            {
                "item_id": self.data.item_ids[int(used[k])],
                "similarity": float(similarities[k]),
                "weight": float(weights[k]),
            }
            for k in order
            if contributions[k] > 0
        ]
