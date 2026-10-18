from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import Any, TextIO
from urllib.parse import urlparse

from dotenv import load_dotenv

from backend.search.chunking import ChunkingPolicy
from backend.search.config import SearchSettings
from backend.search.embeddings import OpenAIEmbeddingProvider
from backend.search.indexing import (
    EmbeddingProgress,
    create_embedding_artifact,
    index_catalog,
    index_embedding_artifact,
    plan_catalog,
)
from backend.search.opensearch import OpenSearchGateway


class _EmbeddingProgressPrinter:
    def __init__(
        self,
        stream: TextIO,
        *,
        every: int = 5,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if every <= 0:
            raise ValueError("Progress interval must be positive")

        self._stream = stream
        self._every = every
        self._clock = clock

        self._started_at = clock()

    def __call__(self, progress: EmbeddingProgress) -> None:
        terminal_phase = progress.phase in {"finalizing", "complete"}
        interval_reached = progress.completed_batches % self._every == 0

        if progress.completed_batches != 1 and not terminal_phase and not interval_reached:
            return

        elapsed = max(0.0, self._clock() - self._started_at)
        percentage = (
            100.0
            if progress.total_batches == 0
            else 100.0 * progress.completed_batches / progress.total_batches
        )

        processed_batches = max(
            0,
            progress.completed_batches - progress.resumed_batch_count,
        )
        remaining_batches = max(
            0,
            progress.total_batches - progress.completed_batches,
        )
        eta = None

        if processed_batches > 0:
            eta = elapsed * remaining_batches / processed_batches

        fields = [
            f"[embed:{progress.phase}]",
            (
                f"{progress.completed_batches}/{progress.total_batches} "
                f"({percentage:.1f}%)"
            ),
            f"items={progress.item_count}",
            f"chunks={progress.chunk_count}",
            f"resumed={progress.resumed_batch_count}",
            f"elapsed={timedelta(seconds=round(elapsed))}",
        ]
        if eta is not None:
            fields.append(f"eta={timedelta(seconds=round(eta))}")

        print(" ".join(fields), file=self._stream, flush=True)


def main(argv: list[str] | None = None) -> int:
    _load_project_env()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan-catalog":
        policy = _chunking_policy(args)
        result = plan_catalog(
            args.catalog,
            chunking_policy=policy,
            catalog_batch_size=args.catalog_batch_size,
        )
        _print_json(
            {
                "status": "planned",
                "item_count": result.item_count,
                "text_covered_item_count": result.text_covered_item_count,

                "chunk_count": result.chunk_count,
                "word_count": result.word_count,
                "catalog_sha256": result.catalog_sha256,
            }
        )

        return 0

    settings = SearchSettings.from_env()

    if args.command == "embed-catalog":
        embedder = _embedding_provider(settings)
        result = create_embedding_artifact(
            args.catalog,
            args.artifact,
            args.manifest,
            embedder,
            chunking_policy=_chunking_policy(args),
            catalog_batch_size=args.catalog_batch_size,
            progress=_EmbeddingProgressPrinter(sys.stderr),
        )
        _print_json(
            {
                "status": "embedded",
                "item_count": result.item_count,
                "chunk_count": result.chunk_count,
                "catalog_sha256": result.catalog_sha256,
                "artifact_sha256": result.artifact_sha256,
                "artifact_path": str(result.artifact_path),
                "manifest_path": str(result.manifest_path),
            }
        )

        return 0

    client = create_opensearch_client(settings)

    if args.command == "health":
        info = client.info()
        version = info.get("version", {}).get("number", "unknown")

        _print_json(
            {
                "status": "ok",
                "cluster_name": info.get("cluster_name", "unknown"),
                "opensearch_version": version,

                "item_index": settings.item_index,
                "chunk_index": settings.chunk_index,
                "embedding_model": settings.embedding_model,

                "embedding_dimensions": settings.embedding_dimensions,
            }
        )

        return 0

    gateway = OpenSearchGateway(
        client,
        item_index=settings.item_index,
        chunk_index=settings.chunk_index,
        dimensions=settings.embedding_dimensions,
    )

    if args.command == "create-indices":
        gateway.ensure_indices()
        _print_json(
            {
                "status": "ready",
                "item_index": gateway.item_index,
                "chunk_index": gateway.chunk_index,
            }
        )

        return 0

    if args.command == "index-artifact":
        result = index_embedding_artifact(
            args.catalog,
            args.artifact,
            args.artifact_manifest,
            args.manifest,
            gateway,
            catalog_batch_size=args.catalog_batch_size,
            artifact_batch_size=args.artifact_batch_size,
            expected_embedding_model=settings.embedding_model,
        )
        _print_json(
            {
                "status": "indexed",
                "item_count": result.item_count,
                "chunk_count": result.chunk_count,
                "catalog_sha256": result.catalog_sha256,
                "artifact_sha256": result.artifact_sha256,
                "manifest_path": str(result.manifest_path),
            }
        )

        return 0

    embedder = _embedding_provider(settings)
    policy = _chunking_policy(args)

    result = index_catalog(
        args.catalog,
        args.manifest,
        gateway,
        embedder,
        chunking_policy=policy,
        catalog_batch_size=args.catalog_batch_size,
    )
    _print_json(
        {
            "status": "indexed",
            "item_count": result.item_count,
            "chunk_count": result.chunk_count,
            "embedded_chunk_count": result.embedded_chunk_count,
            "reused_chunk_count": result.reused_chunk_count,

            "catalog_sha256": result.catalog_sha256,
            "manifest_path": str(result.manifest_path),
        }
    )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-search",
        description="Manage BM25 and semantic retrieval indexes in OpenSearch.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Check the configured OpenSearch cluster")
    subparsers.add_parser(
        "create-indices",
        help="Create versioned item and chunk indexes when absent",
    )

    plan = subparsers.add_parser(
        "plan-catalog",
        help="Measure chunking work without calling external services",
    )
    _add_catalog_arguments(plan)

    index = subparsers.add_parser(
        "index-catalog",
        help="Embed and index one enriched Parquet catalog",
    )
    index.add_argument("--manifest", type=Path, required=True)
    _add_catalog_arguments(index)

    embed = subparsers.add_parser(
        "embed-catalog",
        help="Create a reusable local Parquet embedding artifact",
    )
    embed.add_argument("--artifact", type=Path, required=True)
    embed.add_argument("--manifest", type=Path, required=True)
    _add_catalog_arguments(embed)

    artifact = subparsers.add_parser(
        "index-artifact",
        help="Index a precomputed embedding artifact into OpenSearch",
    )
    artifact.add_argument("--catalog", type=Path, required=True)
    artifact.add_argument("--artifact", type=Path, required=True)
    artifact.add_argument("--artifact-manifest", type=Path, required=True)

    artifact.add_argument("--manifest", type=Path, required=True)
    artifact.add_argument("--catalog-batch-size", type=int, default=500)
    artifact.add_argument("--artifact-batch-size", type=int, default=500)

    return parser


def _add_catalog_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-batch-size", type=int, default=500)

    parser.add_argument("--max-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=24)


def _chunking_policy(args: argparse.Namespace) -> ChunkingPolicy:
    return ChunkingPolicy(
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )


def _embedding_provider(settings: SearchSettings) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        timeout_seconds=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )


def create_opensearch_client(
    settings: SearchSettings,
    environ: Mapping[str, str] | None = None,
) -> Any:
    source = os.environ if environ is None else environ
    username = source.get("OPENSEARCH_USERNAME")
    password = source.get("OPENSEARCH_PASSWORD")

    if bool(username) != bool(password):
        raise ValueError("OPENSEARCH_USERNAME and OPENSEARCH_PASSWORD require both values")

    try:
        from opensearchpy import OpenSearch
    except ImportError as exc:
        message = "OpenSearch commands require the project 'search' dependencies."
        raise RuntimeError(message) from exc

    parsed = urlparse(settings.opensearch_url)
    host = {
        "host": parsed.hostname,
        "port": parsed.port or (443 if parsed.scheme == "https" else 80),
        "scheme": parsed.scheme,
    }
    if parsed.path and parsed.path != "/":
        host["url_prefix"] = parsed.path.strip("/")

    options: dict[str, Any] = {
        "hosts": [host],
        "http_compress": True,
        "verify_certs": settings.verify_certificates,
        "timeout": settings.request_timeout_seconds,
    }
    if username and password:
        options["http_auth"] = (username, password)

    return OpenSearch(**options)


def _load_project_env(dotenv_path: Path | None = None) -> None:
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        (dotenv_path,)
        if dotenv_path is not None
        else (project_root / ".env", project_root / "backend" / ".env")
    )

    for path in paths:
        load_dotenv(dotenv_path=path, override=False)


def _print_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
