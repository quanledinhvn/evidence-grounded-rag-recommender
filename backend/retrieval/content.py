"""Helpers shared by the content retrievers (BM25 and embeddings)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from scipy.sparse import csr_matrix

from backend.cf.common import POSITIVE_RATING, FitData


def load_item_docs(catalog_path: Path, data: FitData) -> list[str]:
    """item_doc of every item, in the same order as data.item_ids ("" when missing)."""
    table = pq.read_table(catalog_path, columns=["item_id", "item_doc"])
    docs = dict(zip(table["item_id"].to_pylist(), table["item_doc"].to_pylist()))

    return [docs.get(item_id) or "" for item_id in data.item_ids]


def liked_history(data: FitData, user_id: str, recency_decay: float, max_items: int) -> tuple[np.ndarray, np.ndarray]:
    """The user's most recent items rated >= 4 and their weights (newest = 1, then decay, decay^2 ...).

    This is the "query" of a content retriever: what the user liked lately.
    """
    items, ratings = data.history(user_id)
    liked = items[ratings >= POSITIVE_RATING][-max_items:]
    weights = (recency_decay ** np.arange(len(liked) - 1, -1, -1)).astype(np.float32)

    return liked, weights


def history_matrix(data: FitData, user_ids: list[str], recency_decay: float, max_items: int) -> csr_matrix:
    """users x items matrix with the recency weight of every liked history item."""
    histories = [liked_history(data, user_id, recency_decay, max_items) for user_id in user_ids]
    rows = np.repeat(np.arange(len(histories)), [len(items) for items, _ in histories])
    items = np.concatenate([items for items, _ in histories])

    weights = np.concatenate([weights for _, weights in histories])

    return csr_matrix((weights, (rows, items)), shape=(len(user_ids), len(data.item_ids)))
