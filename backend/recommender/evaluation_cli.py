from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from backend.data.io import read_parquet, source_sha256
from backend.item_knn import ItemKNNBundle
from backend.profile import (
    CatalogItem,
    InMemoryCatalogStore,
    InMemoryHistoryStore,
    Interaction,
    ProfileBuilder,
    ProfilePolicy,
    ProfileRequest,
)
from backend.recommender.cli import (
    _artifact_manifest_hash,
    _json_value,
    _load_project_env,
    _read_json,
    write_json,
)
from backend.recommender.evaluation import EvaluationCase, evaluate_cases
from backend.recommender.models import PipelineConfig
from backend.recommender.opensearch_retrievers import (
    OpenSearchBM25ContributionRetriever,
    OpenSearchSemanticContributionRetriever,
)
from backend.recommender.pipeline import HybridPipeline
from backend.recommender.retrievers import ItemKNNContributionRetriever
from backend.search.cli import create_opensearch_client
from backend.search.config import SearchSettings
from backend.search.opensearch import OpenSearchGateway


def sample_queries(
    queries: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Select a deterministic evaluation sample without replacement."""
    if isinstance(sample_size, bool) or sample_size <= 0:
        raise ValueError("sample_size must be positive")

    actual_size = min(sample_size, len(queries))
    generator = random.Random(seed)

    shuffled_indices = list(range(len(queries)))
    generator.shuffle(shuffled_indices)

    indices = sorted(shuffled_indices[:actual_size])

    return tuple(queries[index] for index in indices)


def main(argv: list[str] | None = None) -> int:
    """Build profiles, evaluate frozen rankers and write final artifacts."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _load_project_env()
    settings = SearchSettings.from_env()
    client = create_opensearch_client(settings)

    gateway = OpenSearchGateway(
        client,
        item_index=settings.item_index,
        chunk_index=settings.chunk_index,
        dimensions=settings.embedding_dimensions,
    )
    bundle = ItemKNNBundle.load(args.item_knn_bundle)
    if bundle.phase != "test":
        raise ValueError("Evaluation requires the frozen ItemKNN test bundle")

    search_manifest = _read_json(args.search_manifest)
    artifact_hash = _artifact_manifest_hash(
        args.artifact_version,
        bundle.manifest,
        search_manifest,
    )
    weights = {"itemknn": 0.5, "bm25": 0.5}
    if args.semantic:
        weights = {"itemknn": 1 / 3, "bm25": 1 / 3, "semantic": 1 / 3}

    config = PipelineConfig(
        top_k=10,
        pool_size=50,
        evidence_budget=3,
        weights=weights,
    )
    retrievers = {
        "itemknn": ItemKNNContributionRetriever(bundle, args.artifact_version),
        "bm25": OpenSearchBM25ContributionRetriever(
            gateway,
            args.artifact_version,
        ),
    }
    if args.semantic:
        retrievers["semantic"] = OpenSearchSemanticContributionRetriever(
            gateway,
            args.artifact_version,
            settings.embedding_model,
        )

    pipeline = HybridPipeline(
        retrievers,
        config,
        artifact_manifest_hash=artifact_hash,
        fallback_item_ids=(item_id for item_id, _ in bundle.popularity),
    )

    query_rows = read_parquet(args.queries)
    selected_rows = sample_queries(query_rows, args.sample_size, args.seed)
    cases = _build_cases(
        selected_rows,
        args.splits,
        args.catalog,
        args.artifact_version,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_config = {
        "schema_version": "frozen-hybrid-config-v1",
        "artifact_manifest_hash": artifact_hash,
        "sample_size": len(cases),
        "seed": args.seed,
        "profile_phase": "test",
        "semantic_enabled": args.semantic,
        "pipeline": asdict(config),
    }
    write_json(args.output_dir / "frozen_config.json", frozen_config)

    metrics = evaluate_cases(
        pipeline,
        bundle,
        cases,
        checkpoint_path=args.output_dir / "evaluation.checkpoint.jsonl",
        paec_limit=args.paec_limit,
        progress=lambda message: print(message, flush=True),
    )
    write_json(args.output_dir / "ranking_metrics.json", metrics)
    write_json(args.output_dir / "paec_metrics.json", metrics["paec"])

    _write_metrics_csv(args.output_dir / "ranking_metrics.csv", metrics)
    write_json(args.output_dir / "demo_profile.json", _json_value(cases[0].profile))

    run_manifest = {
        "schema_version": "hybrid-evaluation-run-manifest-v1",
        "artifact_manifest_hash": artifact_hash,
        "completed_query_count": metrics["completed_query_count"],
        "semantic_enabled": args.semantic,
        "input_hashes": {
            "splits": source_sha256(args.splits),
            "catalog": source_sha256(args.catalog),
            "queries": source_sha256(args.queries),
            "search_manifest": source_sha256(args.search_manifest),
            "item_knn_manifest": source_sha256(
                args.item_knn_bundle / "manifest.json"
            ),
        },
        "outputs": {
            "frozen_config": "frozen_config.json",
            "ranking_metrics": "ranking_metrics.json",
            "ranking_metrics_csv": "ranking_metrics.csv",
            "paec_metrics": "paec_metrics.json",
            "checkpoint": "evaluation.checkpoint.jsonl",
        },
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-evaluate",
        description="Evaluate the frozen hybrid pipeline on a fixed sample.",
    )
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)

    parser.add_argument("--item-knn-bundle", type=Path, required=True)
    parser.add_argument("--search-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--artifact-version", default="movies-tv-2023")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--paec-limit", type=int, default=100)
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable semantic retrieval (default: enabled; use --no-semantic to disable)",
    )

    return parser


def _build_cases(
    rows: tuple[dict[str, Any], ...],
    splits_path: Path,
    catalog_path: Path,
    artifact_version: str,
) -> tuple[EvaluationCase, ...]:
    user_ids = sorted({str(row["user_id"]) for row in rows})
    history_rows = read_parquet(
        splits_path,
        filters=[
            ("user_id", "in", user_ids),
            ("split", "in", ["train", "validation"]),
        ],
    )
    interactions = tuple(
        Interaction(
            user_id=str(row["user_id"]),
            item_id=str(row["item_id"]),
            rating=float(row["rating"]),
            timestamp=int(row["timestamp"]),
            review_id=str(row["review_id"]),
            split=str(row["split"]),
        )
        for row in history_rows
    )
    history = InMemoryHistoryStore(artifact_version, interactions)

    item_ids = sorted({interaction.item_id for interaction in interactions})
    catalog_rows = read_parquet(
        catalog_path,
        filters=[("item_id", "in", item_ids)],
    )
    catalog_items = {
        str(row["item_id"]): _catalog_item(row)
        for row in catalog_rows
    }
    catalog = InMemoryCatalogStore(artifact_version, catalog_items)

    builder = ProfileBuilder(history, catalog)
    cases: list[EvaluationCase] = []

    for row in rows:
        user_id = str(row["user_id"])
        as_of = int(row["as_of"])

        profile = builder.build(
            ProfileRequest(user_id, as_of, "test", artifact_version),
            ProfilePolicy(allowed_splits=("train", "validation")),
        )
        query_id = str(row.get("target_review_id") or f"{user_id}:{as_of}")

        cases.append(
            EvaluationCase(
                query_id,
                profile,
                str(row["target_item_id"]),
                float(row["target_rating"]),
            )
        )

    return tuple(cases)


def _catalog_item(row: dict[str, Any]) -> CatalogItem:
    return CatalogItem(
        item_id=str(row["item_id"]),
        source_id=str(row.get("source_id") or "").strip(),
        title=_text(row.get("title")),
        subtitle=_text(row.get("subtitle")),
        description=_text(row.get("description")),
        feature=_text(row.get("feature")),
        main_category=_text(row.get("main_category")),
        category=_text(row.get("category")),
        store=_text(row.get("store")),
        details=_text(row.get("details")),
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    fieldnames = (
        "ranker",
        "queries",
        "ndcg_at_10",
        "hr_at_10",
        "candidate_recall_at_50",
        "query_coverage",
        "item_coverage",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
    )

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()

        for ranker, values in metrics["rankers"].items():
            writer.writerow(
                {
                    name: ranker if name == "ranker" else values.get(name, "")
                    for name in fieldnames
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
