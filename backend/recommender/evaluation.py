from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from backend.item_knn import ItemKNNBundle
from backend.profile import UserProfile
from backend.recommender.intervention import InterventionHarness
from backend.recommender.pipeline import HybridPipeline
from backend.recommender.selector import PAECSelector


@dataclass(frozen=True)
class EvaluationCase:
    query_id: str
    profile: UserProfile
    target_item_id: str
    target_rating: float


def evaluate_cases(
    pipeline: HybridPipeline,
    bundle: ItemKNNBundle,
    cases: tuple[EvaluationCase, ...],
    *,
    checkpoint_path: Path | None = None,
    paec_limit: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen rankers, checkpoint results, and report progress."""
    if paec_limit < 0:
        raise ValueError("paec_limit must be non-negative")

    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Evaluation query IDs must be unique")

    records = _load_checkpoint(checkpoint_path)
    _validate_checkpoint_contract(records, pipeline)

    completed_ids = {str(record["query_id"]) for record in records}

    total_cases = len(cases)

    for index, case in enumerate(cases, start=1):
        if case.query_id in completed_ids:
            _report_progress(
                progress,
                f"[{index}/{total_cases}] Skipping query {case.query_id} "
                "(already checkpointed)",
            )
            continue

        _report_progress(
            progress,
            f"[{index}/{total_cases}] Processing query {case.query_id}",
        )
        started_at = perf_counter()
        factual = pipeline.run(case.profile)
        latency_ms = (perf_counter() - started_at) * 1000.0

        rankings = {
            "popularity": _popularity_ranking(bundle, case.profile, 50),
            "itemknn": _itemknn_ranking(bundle, case.profile, 50),
            "bm25": _branch_ranking(factual, "bm25", 50),
            "hybrid": tuple(item.item_id for item in factual.ranking[:50]),
        }
        if "semantic" in pipeline.config.weights:
            rankings["semantic"] = _branch_ranking(factual, "semantic", 50)

        paec = None

        if len(records) < paec_limit and factual.ranking:
            selector = PAECSelector(InterventionHarness(pipeline))
            target_item = factual.ranking[0].item_id

            paec = asdict(
                selector.select(case.profile, factual, target_item)
            )

        record = {
            "query_id": case.query_id,
            "target_item_id": case.target_item_id,
            "target_rating": case.target_rating,
            "rankings": {name: list(values) for name, values in rankings.items()},
            "hybrid_latency_ms": latency_ms,
            "paec": paec,
            "artifact_manifest_hash": pipeline.artifact_manifest_hash,
            "config_hash": pipeline.config_hash,
        }
        records.append(record)
        completed_ids.add(case.query_id)

        _append_checkpoint(checkpoint_path, record)
        _report_progress(
            progress,
            f"[{index}/{total_cases}] Completed query {case.query_id} "
            f"in {latency_ms:.1f} ms",
        )

    requested_ids = frozenset(query_ids)
    selected_records = tuple(
        record for record in records if str(record["query_id"]) in requested_ids
    )

    rankers = ("popularity", *pipeline.config.weights, "hybrid")

    return _summarize(selected_records, len(bundle.item_to_index), rankers)


def _report_progress(
    progress: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress is not None:
        progress(message)


def _popularity_ranking(
    bundle: ItemKNNBundle,
    profile: UserProfile,
    limit: int,
) -> tuple[str, ...]:
    return tuple(
        item_id
        for item_id, _ in bundle.popularity
        if item_id not in profile.seen_item_ids
    )[:limit]


def _itemknn_ranking(
    bundle: ItemKNNBundle,
    profile: UserProfile,
    limit: int,
) -> tuple[str, ...]:
    result = bundle.recommend(
        profile.user_id,
        seen_ids=set(profile.seen_item_ids),
        limit=limit,
    )

    return tuple(str(item["item_id"]) for item in result["items"])


def _branch_ranking(
    factual: object,
    branch: str,
    limit: int,
) -> tuple[str, ...]:
    scored = (
        (item.item_id, float(item.branch_scores[branch]))
        for item in factual.ranking
        if float(item.branch_scores[branch]) > 0.0
    )

    return tuple(
        item_id
        for item_id, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))
    )[:limit]


def _load_checkpoint(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue

            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("query_id"):
                raise ValueError(f"Invalid checkpoint record at line {line_number}")

            records.append(value)

    query_ids = [str(record["query_id"]) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Checkpoint contains duplicate query IDs")

    return records


def _append_checkpoint(
    path: Path | None,
    record: dict[str, Any],
) -> None:
    if path is None:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)

    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{serialized}\n")
        stream.flush()

        os.fsync(stream.fileno())


def _validate_checkpoint_contract(
    records: list[dict[str, Any]],
    pipeline: HybridPipeline,
) -> None:
    for record in records:
        if record.get("artifact_manifest_hash") != pipeline.artifact_manifest_hash:
            raise ValueError("Checkpoint artifact manifest hash does not match")

        if record.get("config_hash") != pipeline.config_hash:
            raise ValueError("Checkpoint config hash does not match")


def _summarize(
    records: tuple[dict[str, Any], ...],
    catalog_size: int,
    ranker_names: tuple[str, ...],
) -> dict[str, Any]:
    rankers = {
        ranker: _ranker_metrics(records, ranker, catalog_size)
        for ranker in ranker_names
    }
    positive_records = tuple(
        record for record in records if float(record["target_rating"]) >= 4.0
    )

    for ranker in ranker_names:
        rankers[ranker]["positive_target_slice"] = _ranker_metrics(
            positive_records,
            ranker,
            catalog_size,
        )

    paec_records = [record["paec"] for record in records if record["paec"]]

    return {
        "schema_version": "hybrid-evaluation-v1",
        "completed_query_count": len(records),
        "positive_query_count": len(positive_records),
        "rankers": rankers,
        "paec": {
            "evaluated_query_count": len(paec_records),
            "selected_count": sum(
                record.get("status") == "selected" for record in paec_records
            ),
        },
    }


def _ranker_metrics(
    records: tuple[dict[str, Any], ...],
    ranker: str,
    catalog_size: int,
) -> dict[str, Any]:
    hits_at_10 = 0
    hits_at_50 = 0
    discounts = 0.0

    covered_queries = 0
    recommended_items: set[str] = set()

    for record in records:
        ranking = tuple(str(value) for value in record["rankings"][ranker])
        recommended_items.update(ranking)

        if ranking:
            covered_queries += 1

        target = str(record["target_item_id"])
        if target not in ranking:
            continue

        rank = ranking.index(target) + 1
        hits_at_50 += rank <= 50

        if rank <= 10:
            hits_at_10 += 1
            discounts += 1.0 / math.log2(rank + 1)

    count = len(records)
    result = {
        "queries": count,
        "ndcg_at_10": discounts / max(1, count),
        "hr_at_10": hits_at_10 / max(1, count),
        "candidate_recall_at_50": hits_at_50 / max(1, count),
        "query_coverage": covered_queries / max(1, count),
        "item_coverage": len(recommended_items) / max(1, catalog_size),
    }

    if ranker == "hybrid":
        latencies = [float(record["hybrid_latency_ms"]) for record in records]
        ordered = sorted(latencies)

        result.update(
            {
                "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
                "p50_latency_ms": _percentile(ordered, 0.50),
                "p95_latency_ms": _percentile(ordered, 0.95),
            }
        )

    return result


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    index = max(0, math.ceil(percentile * len(values)) - 1)

    return values[index]
