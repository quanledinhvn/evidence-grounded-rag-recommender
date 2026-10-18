"""Offline evaluation: does the model put the user's next item in its top 10 / top 50?

For every query (user_id, target_item_id) the model returns 50 items. Metrics:
- HR@10      : 1 if the target is in the top 10, else 0
- NDCG@10    : 1 / log2(rank + 1) if the target is in the top 10, else 0 (rank 1 -> 1.0)
- Recall@50  : 1 if the target is in the top 50 (the candidate pool for later steps)
All are averaged over queries. `positive_target` repeats them for targets rated >= 4.

With `editions` (item_id -> work_id, see backend/data/editions.py) the report adds `same_film`:
the same metrics when any edition of the target film counts as a hit (DVD vs Blu-ray of one film).
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def evaluate(
    model: Any,
    queries_path: Path,
    *,
    query_limit: int | None = None,
    seed: int = 42,
    batch_size: int = 256,
    editions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """`model` is anything with recommend_batch(user_ids, limit) (ItemKNN, BPR, Popularity)."""
    queries = load_queries(queries_path, query_limit, seed)
    count = len(queries)
    ranks, film_ranks = np.zeros(count), np.zeros(count)        # 0 = target not in the top 50
    fallback = np.zeros(count, dtype=bool)

    recommended: set[str] = set()
    started = perf_counter()

    for start in range(0, count, batch_size):
        batch = queries[start : start + batch_size]
        results = model.recommend_batch([query["user_id"] for query in batch], limit=50)

        for offset, (query, items) in enumerate(zip(batch, results)):
            index = start + offset
            item_ids = [item["item_id"] for item in items]
            recommended.update(item_ids)

            fallback[index] = all(item["source"] == "popularity" for item in items)
            ranks[index] = target_rank(item_ids, query["target_item_id"])

            if editions:
                films = [editions.get(item_id, item_id) for item_id in item_ids]
                film_ranks[index] = target_rank(films, editions.get(query["target_item_id"], query["target_item_id"]))

    elapsed_ms = (perf_counter() - started) * 1000.0
    positive = np.array([(query["target_rating"] or 0.0) >= 4.0 for query in queries], dtype=bool)
    catalog_size = len(model.data.item_ids)

    report = summarize(ranks, fallback)
    report["positive_target"] = summarize(ranks[positive], fallback[positive])
    report["item_coverage"] = len(recommended) / catalog_size

    if editions:
        report["same_film"] = summarize(film_ranks, fallback)
        report["same_film"]["positive_target"] = summarize(film_ranks[positive], fallback[positive])

    report["mean_latency_ms"] = elapsed_ms / max(1, count)
    report["model"] = getattr(model, "config", {"model": "popularity"})

    return report


def load_queries(queries_path: Path, query_limit: int | None = None, seed: int = 42) -> list[dict[str, Any]]:
    """All queries, or a fixed random sample so different models are compared on the same users."""
    present = pq.read_schema(queries_path).names
    columns = [name for name in ["user_id", "as_of", "target_item_id", "target_rating"] if name in present]
    queries = pq.read_table(queries_path, columns=columns).to_pylist()

    if query_limit is not None and query_limit < len(queries):
        picked = sorted(Random(seed).sample(range(len(queries)), query_limit))
        queries = [queries[index] for index in picked]

    return queries


def target_rank(item_ids: list[str], target: str) -> int:
    """1-based rank of the target in the list, 0 when it is missing."""
    return item_ids.index(target) + 1 if target in item_ids else 0


def film_rank(item_ids: list[str], target: str, editions: dict[str, str]) -> int:
    """Rank of the first edition of the target film (any DVD / Blu-ray / Prime Video id counts)."""
    return target_rank([editions.get(item_id, item_id) for item_id in item_ids], editions.get(target, target))


def ranking_metrics(ranks: np.ndarray) -> dict[str, float]:
    """NDCG@10, HR@10, Recall@20, Recall@50 from 1-based ranks (0 = missing)."""
    if not len(ranks):
        return {"ndcg_at_10": 0.0, "hr_at_10": 0.0, "recall_at_20": 0.0, "recall_at_50": 0.0}

    top10 = (ranks >= 1) & (ranks <= 10)

    return {
        "ndcg_at_10": float(np.where(top10, 1.0 / np.log2(np.maximum(ranks, 1) + 1), 0.0).mean()),
        "hr_at_10": float(top10.mean()),
        "recall_at_20": float(((ranks >= 1) & (ranks <= 20)).mean()),
        "recall_at_50": float((ranks >= 1).mean()),
    }


def summarize(ranks: np.ndarray, fallback: np.ndarray) -> dict[str, Any]:
    if not len(ranks):
        return {"queries": 0}

    top10 = (ranks >= 1) & (ranks <= 10)
    ndcg = np.where(top10, 1.0 / np.log2(np.maximum(ranks, 1) + 1), 0.0)

    return {
        "queries": int(len(ranks)),
        "ndcg_at_10": float(ndcg.mean()),
        "hr_at_10": float(top10.mean()),
        "recall_at_50": float((ranks >= 1).mean()),
        "fallback_rate": float(fallback.mean()),
    }
