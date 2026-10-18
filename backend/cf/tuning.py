from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.cf.bpr import BPRConfig, config_to_dict
from backend.cf.bundle import train_bundle
from backend.cf.evaluation import compare_retrievers


def select_bpr_config(
    splits_path: Path,
    catalog_path: Path,
    queries_path: Path,
    output_dir: Path,
    candidates: list[BPRConfig],
    *,
    query_limit: int = 50_000,
    batch_size: int = 256,
    query_seed: int | None = None,
) -> tuple[BPRConfig, list[dict[str, Any]]]:
    if not candidates:
        raise ValueError("At least one BPR config candidate is required")

    if query_limit <= 0:
        raise ValueError("query_limit must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    effective_query_seed = candidates[0].seed if query_seed is None else query_seed

    for index, config in enumerate(candidates, start=1):
        candidate_dir = output_dir / "tuning" / f"candidate-{index:04d}"
        bundle = train_bundle(
            splits_path,
            catalog_path,
            candidate_dir,
            phase="validation",
            config=config,
        )

        comparison = compare_retrievers(
            bundle,
            queries_path,
            batch_size=batch_size,
            include_popularity=False,
            query_limit=query_limit,
            query_seed=effective_query_seed,
        )
        metrics = comparison["bpr_mf"]

        record = {
            "candidate": index,
            **config_to_dict(config),
            "ndcg_at_10": float(metrics["ndcg_at_10"]),
            "candidate_recall_at_50": float(metrics["candidate_recall_at_50"]),
            "model_size_bytes": int(metrics["model_size_bytes"]),
            "mean_latency_ms": float(metrics["mean_latency_ms"]),
        }
        records.append(record)

    selected_index = max(
        range(len(records)),
        key=lambda index: (
            records[index]["ndcg_at_10"],
            records[index]["candidate_recall_at_50"],
            -records[index]["model_size_bytes"],
            -records[index]["mean_latency_ms"],
        ),
    )
    for index, record in enumerate(records):
        record["selected"] = "true" if index == selected_index else "false"

    metrics_path = output_dir / "validation_metrics.csv"
    fieldnames = list(records[0])

    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return candidates[selected_index], records
