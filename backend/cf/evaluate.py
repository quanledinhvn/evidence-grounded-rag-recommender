"""Offline evaluation: does the model put the user's next item in its top 10 / top 50?

For every query (user_id, target_item_id) the model returns 50 items. Metrics:
- HR@10      : 1 if the target is in the top 10, else 0
- NDCG@10    : 1 / log2(rank + 1) if the target is in the top 10, else 0 (rank 1 -> 1.0)
- Recall@50  : 1 if the target is in the top 50 (the candidate pool for later steps)
All are averaged over queries. `positive_target` repeats them for targets rated >= 4.
"""

from __future__ import annotations

import math

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
) -> dict[str, Any]:
    """`model` is anything with recommend_batch(user_ids, limit) (ItemKNN, BPR, Popularity)."""
    columns = ["user_id", "target_item_id", "target_rating"]
    queries = pq.read_table(queries_path, columns=columns).to_pylist()

    if query_limit is not None and query_limit < len(queries):
        # Fixed random sample so different models are compared on the same users.
        picked = sorted(Random(seed).sample(range(len(queries)), query_limit))
        queries = [queries[index] for index in picked]

    count = len(queries)
    ndcg, hit10, hit50 = np.zeros(count), np.zeros(count), np.zeros(count)
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

            if query["target_item_id"] in item_ids:
                rank = item_ids.index(query["target_item_id"]) + 1
                hit50[index] = 1.0
                hit10[index] = float(rank <= 10)

                ndcg[index] = 1.0 / math.log2(rank + 1) if rank <= 10 else 0.0

    elapsed_ms = (perf_counter() - started) * 1000.0
    positive = np.array([(query["target_rating"] or 0.0) >= 4.0 for query in queries], dtype=bool)
    catalog_size = len(model.data.item_ids)

    report = summarize(ndcg, hit10, hit50, fallback)
    report["positive_target"] = summarize(ndcg[positive], hit10[positive], hit50[positive], fallback[positive])
    report["item_coverage"] = len(recommended) / catalog_size

    report["mean_latency_ms"] = elapsed_ms / max(1, count)
    report["model"] = getattr(model, "config", {"model": "popularity"})

    return report


def summarize(ndcg: np.ndarray, hit10: np.ndarray, hit50: np.ndarray, fallback: np.ndarray) -> dict[str, Any]:
    if not len(ndcg):
        return {"queries": 0}

    return {
        "queries": int(len(ndcg)),
        "ndcg_at_10": float(ndcg.mean()),
        "hr_at_10": float(hit10.mean()),
        "recall_at_50": float(hit50.mean()),
        "fallback_rate": float(fallback.mean()),
    }
