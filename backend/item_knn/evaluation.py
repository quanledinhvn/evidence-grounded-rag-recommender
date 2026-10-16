from __future__ import annotations

import math

from pathlib import Path
from time import perf_counter
from typing import Any

from backend.data.io import read_parquet
from backend.item_knn.bundle import ItemKNNBundle


def evaluate_item_knn(
    bundle: ItemKNNBundle,
    queries_path: Path,
    *,
    limit: int = 50,
) -> dict[str, Any]:
    if limit != 50:
        raise ValueError("Evaluation limit must be exactly 50")

    queries = read_parquet(queries_path)
    metrics = _evaluate_queries(bundle, queries, limit)
    positive_queries = [
        row for row in queries if float(row.get("target_rating", 0.0)) >= 4.0
    ]

    metrics["positive_target_slice"] = _evaluate_queries(
        bundle,
        positive_queries,
        limit,
    )
    metrics["model_size_bytes"] = bundle.model_size_bytes

    return metrics


def _evaluate_queries(
    bundle: ItemKNNBundle,
    queries: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    hits_at_10 = 0
    reciprocal_discounts = 0.0
    hits_at_50 = 0

    native_queries = 0
    recommended_items: set[str] = set()
    latencies: list[float] = []

    for query in queries:
        user_id = str(query["user_id"])
        if user_id in bundle.positive_users:
            native_queries += 1

        started_at = perf_counter()
        recommendation = bundle.recommend(user_id, limit=limit)
        latencies.append((perf_counter() - started_at) * 1000.0)

        item_ids = [str(item["item_id"]) for item in recommendation["items"]]
        recommended_items.update(item_ids)
        target = str(query["target_item_id"])

        if target not in item_ids:
            continue

        rank = item_ids.index(target) + 1
        if rank <= 50:
            hits_at_50 += 1

        if rank <= 10:
            hits_at_10 += 1
            reciprocal_discounts += 1.0 / math.log2(rank + 1)

    count = len(queries)
    ordered_latencies = sorted(latencies)
    p50_latency = _percentile(ordered_latencies, 0.50)
    p95_latency = _percentile(ordered_latencies, 0.95)

    coverage = native_queries / max(1, count)

    return {
        "queries": count,
        "ndcg_at_10": reciprocal_discounts / max(1, count),
        "hr_at_10": hits_at_10 / max(1, count),
        "candidate_recall_at_50": hits_at_50 / max(1, count),
        "query_coverage": coverage,
        "fallback_rate": 1.0 - coverage,
        "item_coverage": len(recommended_items) / max(1, len(bundle.item_to_index)),
        "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    index = max(0, math.ceil(percentile * len(values)) - 1)

    return values[index]
