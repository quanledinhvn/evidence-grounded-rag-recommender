from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from backend.data.io import source_sha256
from backend.search.chunking import ChunkingPolicy, build_catalog_chunks

_ITEM_FIELDS = (
    "item_id",
    "title",
    "subtitle",

    "description",
    "feature",
    "main_category",

    "category",
    "store",
    "details",

    "metadata_text",
    "average_rating",
    "rating_number",

    "source_id",
    "source_line",
)


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...


class SearchIndexGateway(Protocol):
    item_index: str
    chunk_index: str

    def ensure_indices(self) -> None: ...

    def refresh_indices(self) -> None: ...

    def index_items(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        refresh: bool = False,
    ) -> int: ...

    def index_chunks(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        refresh: bool = False,
    ) -> int: ...

    def reusable_chunk_ids(
        self,
        chunk_hashes: Mapping[str, str],
        *,
        embedding_model: str,
    ) -> frozenset[str]: ...


@dataclass(frozen=True)
class IndexingResult:
    item_count: int
    chunk_count: int
    embedded_chunk_count: int

    reused_chunk_count: int
    catalog_sha256: str

    item_index: str
    chunk_index: str
    embedding_model: str

    embedding_dimensions: int
    manifest_path: Path


@dataclass(frozen=True)
class CatalogPlan:
    item_count: int
    text_covered_item_count: int
    chunk_count: int

    word_count: int
    catalog_sha256: str


def iter_catalog_rows(
    catalog_path: Path,
    *,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], ...]]:
    if batch_size <= 0:
        raise ValueError("Catalog batch size must be positive")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        message = "Catalog indexing requires the project dependency 'pyarrow'."
        raise RuntimeError(message) from exc

    parquet_file = parquet.ParquetFile(catalog_path)
    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        yield tuple(record_batch.to_pylist())


def plan_catalog(
    catalog_path: Path,
    *,
    chunking_policy: ChunkingPolicy = ChunkingPolicy(),
    catalog_batch_size: int = 500,
) -> CatalogPlan:
    if catalog_batch_size <= 0:
        raise ValueError("Catalog batch size must be positive")

    seen_item_ids: set[str] = set()
    item_count = 0
    text_covered_item_count = 0

    chunk_count = 0
    word_count = 0

    for rows in iter_catalog_rows(catalog_path, batch_size=catalog_batch_size):
        for row in rows:
            item_id = _validated_item_id(row, seen_item_ids)
            chunks = build_catalog_chunks(row, chunking_policy)

            seen_item_ids.add(item_id)
            item_count += 1
            text_covered_item_count += bool(chunks)

            chunk_count += len(chunks)
            word_count += sum(len(chunk.text.split()) for chunk in chunks)

    return CatalogPlan(
        item_count=item_count,
        text_covered_item_count=text_covered_item_count,
        chunk_count=chunk_count,
        word_count=word_count,
        catalog_sha256=source_sha256(catalog_path),
    )


def index_catalog(
    catalog_path: Path,
    manifest_path: Path,
    gateway: SearchIndexGateway,
    embedder: EmbeddingProvider,
    *,
    chunking_policy: ChunkingPolicy = ChunkingPolicy(),
    catalog_batch_size: int = 500,
) -> IndexingResult:
    if catalog_batch_size <= 0:
        raise ValueError("Catalog batch size must be positive")

    gateway.ensure_indices()
    seen_item_ids: set[str] = set()

    item_count = 0
    chunk_count = 0
    embedded_chunk_count = 0
    reused_chunk_count = 0

    for rows in iter_catalog_rows(catalog_path, batch_size=catalog_batch_size):
        item_documents: list[dict[str, Any]] = []
        chunks = []

        for row in rows:
            item_id = _validated_item_id(row, seen_item_ids)

            seen_item_ids.add(item_id)
            item_documents.append(_item_document(row))
            chunks.extend(build_catalog_chunks(row, chunking_policy))

        item_count += gateway.index_items(item_documents)
        if not chunks:
            continue

        chunk_count += len(chunks)
        reusable_ids = gateway.reusable_chunk_ids(
            {chunk.chunk_id: chunk.text_sha256 for chunk in chunks},
            embedding_model=embedder.model,
        )
        reused_chunk_count += len(reusable_ids)
        pending_chunks = tuple(
            chunk for chunk in chunks if chunk.chunk_id not in reusable_ids
        )
        if not pending_chunks:
            continue

        vectors = embedder.embed(tuple(chunk.text for chunk in pending_chunks))
        if len(vectors) != len(pending_chunks):
            raise RuntimeError("Embedding provider returned the wrong vector count")

        embedded_chunk_count += len(pending_chunks)

        batch_rows_by_id = rows_by_id(rows)
        chunk_documents = [
            {
                "chunk_id": chunk.chunk_id,
                "item_id": chunk.item_id,
                "field_name": chunk.field_name,

                "ordinal": chunk.ordinal,
                "raw_text": chunk.text,
                "raw_text_hash": chunk.text_sha256,

                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "source_id": str(
                    batch_rows_by_id[chunk.item_id].get("source_id") or ""
                ),

                "embedding_model": embedder.model,
                "embedding": list(vector),
            }
            for chunk, vector in zip(pending_chunks, vectors, strict=True)
        ]
        gateway.index_chunks(chunk_documents)

    gateway.refresh_indices()
    result = IndexingResult(
        item_count=item_count,
        chunk_count=chunk_count,
        embedded_chunk_count=embedded_chunk_count,
        reused_chunk_count=reused_chunk_count,
        catalog_sha256=source_sha256(catalog_path),
        item_index=gateway.item_index,
        chunk_index=gateway.chunk_index,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
        manifest_path=manifest_path,
    )
    _write_manifest(manifest_path, result_manifest(result, chunking_policy))

    return result


def result_manifest(result: IndexingResult, policy: ChunkingPolicy) -> dict[str, Any]:
    values = asdict(result)
    values["manifest_path"] = str(result.manifest_path)

    return {
        "schema_version": "search-index-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        **values,
        "chunking_policy": asdict(policy),
    }


def _item_document(row: Mapping[str, Any]) -> dict[str, Any]:
    document = {
        field: row[field]
        for field in _ITEM_FIELDS
        if field in row and row[field] is not None
    }
    encoded = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    document["metadata_hash"] = hashlib.sha256(encoded).hexdigest()

    return document


def rows_by_id(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    return {str(row["item_id"]): row for row in rows}


def _validated_item_id(
    row: Mapping[str, Any],
    seen_item_ids: set[str],
) -> str:
    item_id = str(row.get("item_id") or "").strip()
    if not item_id:
        raise ValueError("Catalog row must contain a non-empty item_id")

    if item_id in seen_item_ids:
        raise ValueError(f"Duplicate catalog item: {item_id}")

    return item_id


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")

    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
