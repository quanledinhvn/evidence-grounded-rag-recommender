from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv

from backend.item_knn import ItemKNNBundle
from backend.recommender.intervention import InterventionHarness
from backend.recommender.llm_explanation import (
    ExplanationSource,
    generate_llm_explanation,
)
from backend.recommender.models import EvidenceRecord, PipelineConfig
from backend.recommender.opensearch_retrievers import (
    OpenSearchBM25ContributionRetriever,
    OpenSearchRetrievalGateway,
    OpenSearchSemanticContributionRetriever,
)
from backend.recommender.pipeline import HybridPipeline
from backend.recommender.profile_io import load_profile
from backend.recommender.rendering import render_template
from backend.recommender.retrievers import ItemKNNContributionRetriever
from backend.recommender.selector import PAECSelector
from backend.search.cli import create_opensearch_client
from backend.search.config import SearchSettings
from backend.search.opensearch import OpenSearchGateway


def run_recommendation(
    profile_path: Path,
    item_knn_bundle_path: Path,
    search_manifest_path: Path,
    *,
    gateway: OpenSearchRetrievalGateway,
    top_k: int = 10,
    pool_size: int = 50,
    evidence_budget: int = 3,
    include_paec: bool = True,
    include_semantic: bool = True,
    include_llm_explanation: bool = False,
    llm_model: str = "gpt-6-luna",
) -> dict[str, Any]:
    """Run the frozen recommender for one serialized profile."""
    profile = load_profile(profile_path)
    bundle = ItemKNNBundle.load(item_knn_bundle_path)

    if profile.phase != "test" or bundle.phase != "test":
        raise ValueError("The frozen two-day pipeline requires test artifacts")

    search_manifest = _read_json(search_manifest_path)
    artifact_hash = _artifact_manifest_hash(
        profile.artifact_version,
        bundle.manifest,
        search_manifest,
    )
    weights = {"itemknn": 0.5, "bm25": 0.5}
    if include_semantic:
        weights = {
            "itemknn": 1.0 / 3.0,
            "bm25": 1.0 / 3.0,
            "semantic": 1.0 / 3.0,
        }

    config = PipelineConfig(
        top_k=top_k,
        pool_size=pool_size,
        evidence_budget=evidence_budget,
        weights=weights,
    )
    embedding_model = str(search_manifest.get("embedding_model") or "").strip()
    if not embedding_model:
        embedding_model = "text-embedding-3-small"

    retrievers = {
        "itemknn": ItemKNNContributionRetriever(
            bundle,
            profile.artifact_version,
        ),
        "bm25": OpenSearchBM25ContributionRetriever(
            gateway,
            profile.artifact_version,
        ),
    }
    if include_semantic:
        retrievers["semantic"] = OpenSearchSemanticContributionRetriever(
            gateway,
            profile.artifact_version,
            embedding_model,
        )

    pipeline = HybridPipeline(
        retrievers,
        config,
        artifact_manifest_hash=artifact_hash,
        fallback_item_ids=(item_id for item_id, _ in bundle.popularity),
    )
    started_at = perf_counter()
    factual = pipeline.run(profile)

    selection = None
    explanation = None
    llm_explanation = None
    evidence: tuple[EvidenceRecord, ...] = ()

    if include_paec and factual.ranking:
        target_item_id = factual.ranking[0].item_id
        selector = PAECSelector(InterventionHarness(pipeline))

        selection = selector.select(profile, factual, target_item_id)
        evidence = _evidence(profile, factual, selection.certificate)
        explanation = render_template(
            target_item_id,
            selection.certificate,
            evidence,
        )

    if include_llm_explanation and factual.ranking:
        target_item_id = factual.ranking[0].item_id
        llm_explanation = generate_llm_explanation(
            target_item_id,
            selection.certificate if selection is not None else None,
            evidence,
            fallback_sources=_llm_sources(profile, factual),
            candidate_evidence=_candidate_evidence(factual, target_item_id),
            model=llm_model,
        )

    elapsed_ms = (perf_counter() - started_at) * 1000.0
    ranking = tuple(factual.ranking[:top_k])

    return {
        "schema_version": "recommendation-run-v1",
        "status": "ok" if factual.evaluable else "invalid",
        "user_id": profile.user_id,
        "as_of": profile.as_of,
        "phase": profile.phase,
        "config": _json_value(config),
        "semantic_enabled": include_semantic,
        "artifact_manifest_hash": artifact_hash,
        "ranking": _json_value(ranking),
        "candidate_count": len(factual.candidate_ids),
        "invalid_reasons": list(factual.invalid_reasons),
        "selection": _json_value(selection),
        "explanation": _json_value(explanation),
        "llm_explanation": llm_explanation,
        "latency_ms": elapsed_ms,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the frozen two-day recommendation pipeline."""
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
    result = run_recommendation(
        args.profile,
        args.item_knn_bundle,
        args.search_manifest,
        gateway=gateway,
        top_k=args.top_k,
        pool_size=args.pool_size,
        evidence_budget=args.evidence_budget,
        include_paec=not args.skip_paec,
        include_semantic=args.semantic,
        include_llm_explanation=args.llm_explanation,
        llm_model=args.llm_model
        or os.environ.get("OPENAI_EXPLANATION_MODEL", "gpt-6-luna"),
    )

    if args.output is None:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        write_json(args.output, result)

    return 0


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one recommendation result as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-recommend",
        description="Run the frozen ItemKNN-BM25-Semantic pipeline.",
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--item-knn-bundle", type=Path, required=True)
    parser.add_argument("--search-manifest", type=Path, required=True)

    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=50)

    parser.add_argument("--evidence-budget", type=int, default=3)
    parser.add_argument("--skip-paec", action="store_true")
    parser.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable semantic retrieval (default: enabled; use --no-semantic to disable)",
    )
    parser.add_argument(
        "--llm-explanation",
        action="store_true",
        help="Generate a grounded explanation with the OpenAI Responses API",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="OpenAI model used only with --llm-explanation",
    )

    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)

    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return value


def _artifact_manifest_hash(
    artifact_version: str,
    item_knn_manifest: Mapping[str, Any],
    search_manifest: Mapping[str, Any],
) -> str:
    payload = {
        "artifact_version": artifact_version,
        "item_knn": item_knn_manifest,
        "search": search_manifest,
        "retrieval_contract": {
            "bm25_query_max_unique_terms": 128,
            "semantic_policy": "per-source-chunk-centroid-v1",
            "semantic_ann_k_policy": "branch-pool-size",
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evidence(
    profile: object,
    factual: object,
    certificate: object,
) -> tuple[EvidenceRecord, ...]:
    if certificate is None:
        return ()

    coalition = frozenset(certificate.coalition_source_ids)
    records = [
        EvidenceRecord(
            signal.history_id,
            "history",
            " ".join(field.text for field in signal.text_fields),
            signal.history_id,
        )
        for signal in profile.positive_signals
        if signal.history_id in coalition
    ]
    target_atoms = (
        atom
        for atom in factual.atoms
        if atom.candidate_id == certificate.target_item_id
        and atom.source_id in coalition
    )

    for index, atom in enumerate(target_atoms, start=1):
        records.append(
            EvidenceRecord(
                f"P{index}",
                "provenance",
                atom.candidate_span,
                atom.source_id,
            )
        )

    records.append(
        EvidenceRecord(
            "CERT1",
            "certificate",
            "Full-pipeline removal and retention certificate.",
        )
    )

    return tuple(records)


def _llm_sources(
    profile: object,
    factual: object,
) -> tuple[ExplanationSource, ...]:
    if not factual.ranking:
        return ()

    contributions = factual.ranking[0].source_attribution
    signals = {
        signal.history_id: signal
        for signal in profile.positive_signals
        if signal.history_id in contributions
    }
    source_ids = sorted(
        signals,
        key=lambda source_id: (-contributions[source_id], source_id),
    )
    return tuple(
        ExplanationSource(
            history_id=source_id,
            item_id=signals[source_id].item_id,
            text=" ".join(field.text for field in signals[source_id].text_fields),
            contribution=float(contributions[source_id]),
        )
        for source_id in source_ids[:3]
    )


def _candidate_evidence(
    factual: object,
    item_id: str,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            atom.candidate_span
            for atom in factual.atoms
            if atom.candidate_id == item_id and atom.candidate_span
        )
    )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if is_dataclass(value):
        return _json_value(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]

    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _load_project_env() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)

    compatibility_path = project_root / "backend" / ".env"
    if compatibility_path.exists():
        load_dotenv(compatibility_path, override=False)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


if __name__ == "__main__":
    raise SystemExit(main())
