from __future__ import annotations

from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

import math
import numpy as np

from backend.cf.baselines import recommend_popularity
from backend.cf.bundle import CFBundle
from backend.data.io import read_parquet


def evaluate_bundle(
    bundle: CFBundle,
    queries_path: Path,
    *,
    limit: int = 50,
    batch_size: int = 256,
) -> dict[str, Any]:
    if limit != 50:
        raise ValueError("Evaluation limit must be exactly 50")

    queries = read_parquet(queries_path)
    result = _evaluate_bpr_batched(
        bundle,
        queries,
        limit,
        batch_size,
        include_positive_slice=True,
    )

    return result


def compare_retrievers(
    bundle: CFBundle,
    queries_path: Path,
    *,
    limit: int = 50,
    batch_size: int = 256,
    include_popularity: bool = True,
    query_limit: int | None = None,
    query_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    if limit != 50:
        raise ValueError("Evaluation limit must be exactly 50")

    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive when provided")

    queries = read_parquet(queries_path)
    if query_limit is not None and query_limit < len(queries):
        indices = sorted(Random(query_seed).sample(range(len(queries)), query_limit))
        queries = [queries[index] for index in indices]

    bpr_metrics = _evaluate_bpr_batched(
        bundle,
        queries,
        limit,
        batch_size,
        include_positive_slice=True,
    )
    comparison: dict[str, dict[str, Any]] = {
        "bpr_mf": bpr_metrics,
    }

    if include_popularity:
        positive_queries = [
            row for row in queries if float(row.get("target_rating", 0.0)) >= 4.0
        ]
        popularity_metrics = _evaluate_ranker(
            queries,
            lambda user_id: recommend_popularity(bundle, user_id, limit=limit),
            len(bundle.item_to_index),
        )
        popularity_metrics["positive_target_slice"] = _evaluate_ranker(
            positive_queries,
            lambda user_id: recommend_popularity(bundle, user_id, limit=limit),
            len(bundle.item_to_index),
        )
        comparison["popularity"] = popularity_metrics

    comparison["bpr_mf"]["model_size_bytes"] = (
        bundle.model.user_factors.nbytes
        + bundle.model.item_factors.nbytes
        + bundle.model.item_bias.nbytes
    )
    comparison["bpr_mf"]["model_artifact_size_bytes"] = bundle.manifest[
        "artifact_sizes"
    ]["bpr_model.npz"]

    return comparison


def _evaluate_ranker(
    queries: list[dict[str, Any]],
    ranker: Any,
    catalog_count: int,
) -> dict[str, Any]:
    hits_at_10 = 0
    reciprocal_discounts = 0.0
    hits_at_50 = 0

    served_queries = 0
    recommended_items: set[str] = set()
    latencies: list[float] = []

    for query in queries:
        started_at = perf_counter()
        items = ranker(str(query["user_id"]))
        latencies.append((perf_counter() - started_at) * 1000.0)

        item_ids = [str(item["item_id"]) for item in items]
        if item_ids:
            served_queries += 1

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
    p50_index = max(0, math.ceil(0.50 * len(ordered_latencies)) - 1)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)

    p50_latency = ordered_latencies[p50_index] if ordered_latencies else 0.0
    p95_latency = ordered_latencies[p95_index] if ordered_latencies else 0.0

    return {
        "queries": count,
        "ndcg_at_10": reciprocal_discounts / max(1, count),
        "hr_at_10": hits_at_10 / max(1, count),
        "candidate_recall_at_50": hits_at_50 / max(1, count),
        "query_coverage": served_queries / max(1, count),
        "item_coverage": len(recommended_items) / max(1, catalog_count),
        "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
    }


def _evaluate_bpr_batched(
    bundle: CFBundle,
    queries: list[dict[str, Any]],
    limit: int,
    batch_size: int,
    *,
    include_positive_slice: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    candidate_ids = sorted(bundle.learnable_items)
    candidate_indices = np.fromiter(
        (bundle.item_to_index[item_id] for item_id in candidate_ids),
        dtype=np.int64,
    )
    candidate_positions = {
        item_id: position for position, item_id in enumerate(candidate_ids)
    }

    candidate_factors = bundle.model.item_factors[candidate_indices]
    candidate_bias = bundle.model.item_bias[candidate_indices]

    hits_at_10 = 0
    reciprocal_discounts = 0.0
    hits_at_50 = 0

    bpr_queries = 0
    recommended_items: set[str] = set()
    latencies: list[float] = []

    positive_queries = 0
    positive_bpr_queries = 0
    positive_hits_at_10 = 0

    positive_discounts = 0.0
    positive_hits_at_50 = 0
    positive_items: set[str] = set()

    positive_latencies: list[float] = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        started_at = perf_counter()

        rankings: list[list[str]] = [[] for _ in batch]
        warm_offsets = [
            offset
            for offset, query in enumerate(batch)
            if str(query["user_id"]) in bundle.positive_users
        ]
        warm_offset_set = set(warm_offsets)
        user_indices = np.fromiter(
            (
                bundle.user_to_index[str(batch[offset]["user_id"])]
                for offset in warm_offsets
            ),
            dtype=np.int64,
        )

        if len(user_indices):
            score_matrix = (
                bundle.model.user_factors[user_indices] @ candidate_factors.T
            )
            score_matrix += candidate_bias

            for row, offset in enumerate(warm_offsets):
                user_id = str(batch[offset]["user_id"])
                seen_positions = [
                    candidate_positions[item_id]
                    for item_id in bundle.seen_by_user.get(user_id, set())
                    if item_id in candidate_positions
                ]
                scores = score_matrix[row]
                scores[seen_positions] = -np.inf

                available_count = len(candidate_ids) - len(seen_positions)
                selected_count = min(limit, available_count)
                if selected_count and len(candidate_ids) > selected_count:
                    selected = np.argpartition(scores, -selected_count)[
                        -selected_count:
                    ]
                    threshold = float(scores[selected].min())
                    selected = np.flatnonzero(scores >= threshold)
                else:
                    selected = np.flatnonzero(np.isfinite(scores))

                candidates = [
                    (candidate_ids[int(index)], float(scores[int(index)]))
                    for index in selected
                ]
                candidates.sort(key=lambda pair: (-pair[1], pair[0]))
                rankings[offset] = [
                    item_id for item_id, _ in candidates[:selected_count]
                ]

            bpr_queries += len(warm_offsets)

        for offset, query in enumerate(batch):
            if offset not in warm_offset_set:
                user_id = str(query["user_id"])
                rankings[offset] = [
                    str(item["item_id"])
                    for item in recommend_popularity(bundle, user_id, limit=limit)
                ]

        elapsed_ms = (perf_counter() - started_at) * 1000.0
        latency_ms = elapsed_ms / max(1, len(batch))
        latencies.extend([latency_ms] * len(batch))

        for query, item_ids in zip(batch, rankings):
            recommended_items.update(item_ids)
            target = str(query["target_item_id"])
            is_positive = float(query.get("target_rating", 0.0)) >= 4.0
            if is_positive:
                positive_queries += 1
                positive_items.update(item_ids)
                positive_latencies.append(latency_ms)

                user_id = str(query["user_id"])
                if user_id in bundle.positive_users:
                    positive_bpr_queries += 1

            if target not in item_ids:
                continue

            rank = item_ids.index(target) + 1
            if rank <= 50:
                hits_at_50 += 1
                if is_positive:
                    positive_hits_at_50 += 1

            if rank <= 10:
                hits_at_10 += 1
                reciprocal_discounts += 1.0 / math.log2(rank + 1)
                if is_positive:
                    positive_hits_at_10 += 1
                    positive_discounts += 1.0 / math.log2(rank + 1)

    ordered_latencies = sorted(latencies)
    p50_index = max(0, math.ceil(0.50 * len(ordered_latencies)) - 1)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latencies)) - 1)

    p50_latency = ordered_latencies[p50_index] if ordered_latencies else 0.0
    p95_latency = ordered_latencies[p95_index] if ordered_latencies else 0.0
    query_count = len(queries)

    query_coverage = bpr_queries / max(1, query_count)

    result = {
        "queries": query_count,
        "ndcg_at_10": reciprocal_discounts / max(1, query_count),
        "hr_at_10": hits_at_10 / max(1, query_count),
        "candidate_recall_at_50": hits_at_50 / max(1, query_count),
        "query_coverage": query_coverage,
        "fallback_rate": 1.0 - query_coverage,
        "item_coverage": len(recommended_items) / max(1, len(bundle.item_to_index)),
        "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
    }

    if not include_positive_slice:
        return result

    ordered_positive_latencies = sorted(positive_latencies)
    positive_p50_index = max(
        0,
        math.ceil(0.50 * len(ordered_positive_latencies)) - 1,
    )
    positive_p95_index = max(
        0,
        math.ceil(0.95 * len(ordered_positive_latencies)) - 1,
    )

    positive_p50 = (
        ordered_positive_latencies[positive_p50_index]
        if ordered_positive_latencies
        else 0.0
    )
    positive_p95 = (
        ordered_positive_latencies[positive_p95_index]
        if ordered_positive_latencies
        else 0.0
    )
    positive_coverage = positive_bpr_queries / max(1, positive_queries)

    result["positive_target_slice"] = {
        "queries": positive_queries,
        "ndcg_at_10": positive_discounts / max(1, positive_queries),
        "hr_at_10": positive_hits_at_10 / max(1, positive_queries),
        "candidate_recall_at_50": positive_hits_at_50 / max(1, positive_queries),
        "query_coverage": positive_coverage,
        "fallback_rate": 1.0 - positive_coverage,
        "item_coverage": len(positive_items) / max(1, len(bundle.item_to_index)),
        "mean_latency_ms": sum(positive_latencies) / max(1, len(positive_latencies)),
        "p50_latency_ms": positive_p50,
        "p95_latency_ms": positive_p95,
    }

    return result
