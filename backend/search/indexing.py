from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from backend.data.io import source_sha256
from backend.search.chunking import CatalogChunk, ChunkingPolicy, build_catalog_chunks

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
    dimensions: int

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


@dataclass(frozen=True)
class EmbeddingArtifactResult:
    item_count: int
    chunk_count: int
    catalog_sha256: str

    artifact_sha256: str
    embedding_model: str
    embedding_dimensions: int

    artifact_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class EmbeddingProgress:
    completed_batches: int
    total_batches: int
    item_count: int

    chunk_count: int
    resumed_batch_count: int
    phase: str


@dataclass(frozen=True)
class ArtifactIndexingResult:
    item_count: int
    chunk_count: int
    catalog_sha256: str

    artifact_sha256: str
    item_index: str
    chunk_index: str

    manifest_path: Path


def create_embedding_artifact(
    catalog_path: Path,
    artifact_path: Path,
    manifest_path: Path,
    embedder: EmbeddingProvider,
    *,
    chunking_policy: ChunkingPolicy = ChunkingPolicy(),
    catalog_batch_size: int = 500,
    progress: Callable[[EmbeddingProgress], None] | None = None,
) -> EmbeddingArtifactResult:
    _require_distinct_paths(catalog_path, artifact_path, manifest_path)

    if catalog_batch_size <= 0:
        raise ValueError("Catalog batch size must be positive")

    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        message = "Embedding artifacts require the project dependency 'pyarrow'."
        raise RuntimeError(message) from exc

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _embedding_artifact_schema(pa, embedder.dimensions)
    catalog_hash = source_sha256(catalog_path)

    parquet_file = parquet.ParquetFile(catalog_path)
    total_items = parquet_file.metadata.num_rows
    total_batches = math.ceil(total_items / catalog_batch_size)

    checkpoint_path = artifact_path.with_name(
        f"{artifact_path.name}.checkpoint"
    )
    _require_distinct_paths(
        catalog_path,
        artifact_path,
        manifest_path,
        checkpoint_path,
    )

    contract = _embedding_checkpoint_contract(
        catalog_hash,
        embedder,
        chunking_policy,
        catalog_batch_size,
    )
    completed_result = _completed_embedding_artifact(
        catalog_hash,
        artifact_path,
        manifest_path,
        embedder,
        chunking_policy,
    )
    if completed_result is not None:
        _emit_embedding_progress(
            progress,
            completed_batches=total_batches,
            total_batches=total_batches,
            item_count=completed_result.item_count,
            chunk_count=completed_result.chunk_count,
            resumed_batch_count=total_batches,
            phase="complete",
        )

        return completed_result

    state = _load_or_create_embedding_checkpoint(checkpoint_path, contract)
    parts = list(state["parts"])

    completed_batches = _checkpoint_integer(state, "completed_batches")
    if completed_batches != len(parts) or completed_batches > total_batches:
        raise ValueError("Embedding checkpoint has an invalid batch count")

    seen_item_ids: set[str] = set()
    item_count = 0
    chunk_count = 0
    resumed_batch_count = 0

    for batch_index, rows in enumerate(
        iter_catalog_rows(catalog_path, batch_size=catalog_batch_size)
    ):
        for row in rows:
            item_id = _validated_item_id(row, seen_item_ids)
            seen_item_ids.add(item_id)

        item_count += len(rows)
        if batch_index < completed_batches:
            part = _validated_checkpoint_part(
                checkpoint_path,
                parts[batch_index],
                batch_index=batch_index,
                expected_item_count=len(rows),
            )
            chunk_count += int(part["chunk_count"])
            resumed_batch_count += 1

            _emit_embedding_progress(
                progress,
                completed_batches=batch_index + 1,
                total_batches=total_batches,
                item_count=item_count,
                chunk_count=chunk_count,
                resumed_batch_count=resumed_batch_count,
                phase="resuming",
            )
            continue

        chunks = [
            chunk
            for row in rows
            for chunk in build_catalog_chunks(row, chunking_policy)
        ]
        vectors = embedder.embed(tuple(chunk.text for chunk in chunks)) if chunks else ()
        documents = _embedded_chunk_documents(rows, chunks, vectors, embedder)

        part_path = checkpoint_path / f"part-{batch_index:06d}.parquet"
        _write_parquet_part(part_path, documents, schema, pa, parquet)

        part = {
            "batch_index": batch_index,
            "file_name": part_path.name,
            "item_count": len(rows),
            "chunk_count": len(documents),
            "sha256": source_sha256(part_path),
        }
        parts.append(part)
        chunk_count += len(documents)

        state = {
            **state,
            "completed_batches": batch_index + 1,
            "item_count": item_count,
            "chunk_count": chunk_count,
            "parts": parts,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        _write_manifest(checkpoint_path / "state.json", state)

        _emit_embedding_progress(
            progress,
            completed_batches=batch_index + 1,
            total_batches=total_batches,
            item_count=item_count,
            chunk_count=chunk_count,
            resumed_batch_count=resumed_batch_count,
            phase="embedding",
        )

    _emit_embedding_progress(
        progress,
        completed_batches=total_batches,
        total_batches=total_batches,
        item_count=item_count,
        chunk_count=chunk_count,
        resumed_batch_count=resumed_batch_count,
        phase="finalizing",
    )
    if source_sha256(catalog_path) != catalog_hash:
        raise ValueError("Catalog changed while creating the embedding artifact")

    _combine_embedding_parts(artifact_path, checkpoint_path, parts, schema, parquet)
    artifact_hash = source_sha256(artifact_path)

    result = EmbeddingArtifactResult(
        item_count=item_count,
        chunk_count=chunk_count,
        catalog_sha256=catalog_hash,
        artifact_sha256=artifact_hash,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
    )
    manifest = {
        "schema_version": "embedding-artifact-v1",
        "created_at": datetime.now(UTC).isoformat(),
        **_serialized_dataclass(result),
        "chunking_policy": asdict(chunking_policy),
    }
    _write_manifest(manifest_path, manifest)
    _cleanup_embedding_checkpoint(checkpoint_path, parts)

    _emit_embedding_progress(
        progress,
        completed_batches=total_batches,
        total_batches=total_batches,
        item_count=item_count,
        chunk_count=chunk_count,
        resumed_batch_count=resumed_batch_count,
        phase="complete",
    )

    return result


def index_embedding_artifact(
    catalog_path: Path,
    artifact_path: Path,
    artifact_manifest_path: Path,
    index_manifest_path: Path,
    gateway: SearchIndexGateway,
    *,
    catalog_batch_size: int = 500,
    artifact_batch_size: int = 500,
    expected_embedding_model: str | None = None,
) -> ArtifactIndexingResult:
    _require_distinct_paths(
        catalog_path,
        artifact_path,
        artifact_manifest_path,
        index_manifest_path,
    )

    if catalog_batch_size <= 0 or artifact_batch_size <= 0:
        raise ValueError("Indexing batch sizes must be positive")

    artifact_manifest = _read_embedding_manifest(artifact_manifest_path)
    catalog_hash = source_sha256(catalog_path)
    artifact_hash = source_sha256(artifact_path)

    if artifact_manifest.get("catalog_sha256") != catalog_hash:
        raise ValueError("Embedding artifact catalog hash does not match the catalog")

    if artifact_manifest.get("artifact_sha256") != artifact_hash:
        raise ValueError("Embedding artifact hash does not match its manifest")

    dimensions = _manifest_integer(artifact_manifest, "embedding_dimensions")
    if dimensions != gateway.dimensions:
        raise ValueError("Embedding artifact dimensions do not match OpenSearch")

    embedding_model = str(artifact_manifest["embedding_model"])
    if (
        expected_embedding_model is not None
        and embedding_model != expected_embedding_model
    ):
        raise ValueError("Embedding artifact model does not match the configured model")

    expected_chunks = _manifest_integer(artifact_manifest, "chunk_count")
    _validate_artifact_rows(
        artifact_path,
        batch_size=artifact_batch_size,
        dimensions=dimensions,
        expected_count=expected_chunks,
        expected_model=embedding_model,
    )

    gateway.ensure_indices()
    seen_item_ids: set[str] = set()
    item_count = 0

    for rows in iter_catalog_rows(catalog_path, batch_size=catalog_batch_size):
        item_documents = []

        for row in rows:
            item_id = _validated_item_id(row, seen_item_ids)
            seen_item_ids.add(item_id)
            item_documents.append(_item_document(row))

        item_count += gateway.index_items(item_documents)

    chunk_count = 0
    for documents in iter_catalog_rows(artifact_path, batch_size=artifact_batch_size):
        chunk_count += gateway.index_chunks(documents)

    gateway.refresh_indices()
    result = ArtifactIndexingResult(
        item_count=item_count,
        chunk_count=chunk_count,
        catalog_sha256=catalog_hash,
        artifact_sha256=artifact_hash,
        item_index=gateway.item_index,
        chunk_index=gateway.chunk_index,
        manifest_path=index_manifest_path,
    )
    manifest = {
        "schema_version": "search-index-manifest-v2",
        "created_at": datetime.now(UTC).isoformat(),
        **_serialized_dataclass(result),
        "embedding_model": embedding_model,
        "embedding_dimensions": dimensions,
        "artifact_manifest_path": str(artifact_manifest_path),
    }
    _write_manifest(index_manifest_path, manifest)

    return result


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


def _embedded_chunk_documents(
    rows: tuple[dict[str, Any], ...],
    chunks: list[CatalogChunk],
    vectors: tuple[tuple[float, ...], ...],
    embedder: EmbeddingProvider,
) -> list[dict[str, Any]]:
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding provider returned the wrong vector count")

    batch_rows_by_id = rows_by_id(rows)
    documents: list[dict[str, Any]] = []

    for chunk, values in zip(chunks, vectors, strict=True):
        vector = [float(value) for value in values]
        if len(vector) != embedder.dimensions:
            raise RuntimeError("Embedding provider returned the wrong dimension")

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding values must be finite")

        documents.append(
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
                "embedding": vector,
            }
        )

    return documents


def _embedding_artifact_schema(pa: Any, dimensions: int) -> Any:
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be positive")

    return pa.schema(
        [
            pa.field("chunk_id", pa.string(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("field_name", pa.string(), nullable=False),
            pa.field("ordinal", pa.int64(), nullable=False),
            pa.field("raw_text", pa.string(), nullable=False),
            pa.field("raw_text_hash", pa.string(), nullable=False),
            pa.field("start_char", pa.int64(), nullable=False),
            pa.field("end_char", pa.int64(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("embedding_model", pa.string(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(pa.float64(), dimensions),
                nullable=False,
            ),
        ]
    )


def _embedding_checkpoint_contract(
    catalog_hash: str,
    embedder: EmbeddingProvider,
    chunking_policy: ChunkingPolicy,
    catalog_batch_size: int,
) -> dict[str, Any]:
    return {
        "catalog_sha256": catalog_hash,
        "embedding_model": embedder.model,
        "embedding_dimensions": embedder.dimensions,
        "catalog_batch_size": catalog_batch_size,
        "chunking_policy": {
            "fields": list(chunking_policy.fields),
            "max_words": chunking_policy.max_words,
            "overlap_words": chunking_policy.overlap_words,
        },
    }


def _completed_embedding_artifact(
    catalog_hash: str,
    artifact_path: Path,
    manifest_path: Path,
    embedder: EmbeddingProvider,
    chunking_policy: ChunkingPolicy,
) -> EmbeddingArtifactResult | None:
    if not artifact_path.exists() and not manifest_path.exists():
        return None

    if not artifact_path.is_file() or not manifest_path.is_file():
        raise ValueError("Embedding artifact and manifest must both exist")

    manifest = _read_embedding_manifest(manifest_path)
    expected_policy = {
        "fields": list(chunking_policy.fields),
        "max_words": chunking_policy.max_words,
        "overlap_words": chunking_policy.overlap_words,
    }

    if manifest.get("catalog_sha256") != catalog_hash:
        raise ValueError("Existing embedding artifact uses a different catalog")

    if manifest.get("embedding_model") != embedder.model:
        raise ValueError("Existing embedding artifact uses a different model")

    if manifest.get("embedding_dimensions") != embedder.dimensions:
        raise ValueError("Existing embedding artifact uses different dimensions")

    if manifest.get("chunking_policy") != expected_policy:
        raise ValueError("Existing embedding artifact uses a different chunking policy")

    artifact_hash = source_sha256(artifact_path)
    if manifest.get("artifact_sha256") != artifact_hash:
        raise ValueError("Existing embedding artifact hash does not match")

    return EmbeddingArtifactResult(
        item_count=_manifest_integer(manifest, "item_count"),
        chunk_count=_manifest_integer(manifest, "chunk_count"),
        catalog_sha256=str(manifest["catalog_sha256"]),
        artifact_sha256=artifact_hash,
        embedding_model=embedder.model,
        embedding_dimensions=embedder.dimensions,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
    )


def _load_or_create_embedding_checkpoint(
    checkpoint_path: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if checkpoint_path.exists() and not checkpoint_path.is_dir():
        raise ValueError("Embedding checkpoint path must be a directory")

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_path / "state.json"

    if not state_path.exists():
        unexpected_files = tuple(checkpoint_path.iterdir())
        if unexpected_files:
            raise ValueError("Embedding checkpoint is missing its state file")

        state = {
            "schema_version": "embedding-checkpoint-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "contract": dict(contract),
            "completed_batches": 0,
            "item_count": 0,
            "chunk_count": 0,
            "parts": [],
        }
        _write_manifest(state_path, state)

        return state

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Embedding checkpoint state is unreadable") from exc

    if not isinstance(state, dict):
        raise ValueError("Embedding checkpoint state must be a JSON object")

    if state.get("schema_version") != "embedding-checkpoint-v1":
        raise ValueError("Unsupported embedding checkpoint version")

    if state.get("contract") != dict(contract):
        raise ValueError("Embedding checkpoint configuration does not match")

    if not isinstance(state.get("parts"), list):
        raise ValueError("Embedding checkpoint has invalid parts")

    return state


def _checkpoint_integer(state: Mapping[str, Any], field: str) -> int:
    value = state.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Embedding checkpoint has invalid {field}")

    return value


def _validated_checkpoint_part(
    checkpoint_path: Path,
    raw_part: Any,
    *,
    batch_index: int,
    expected_item_count: int,
) -> dict[str, Any]:
    if not isinstance(raw_part, dict):
        raise ValueError("Embedding checkpoint has an invalid part")

    if raw_part.get("batch_index") != batch_index:
        raise ValueError("Embedding checkpoint part has the wrong batch index")

    if raw_part.get("item_count") != expected_item_count:
        raise ValueError("Embedding checkpoint part has the wrong item count")

    chunk_count = raw_part.get("chunk_count")
    if isinstance(chunk_count, bool) or not isinstance(chunk_count, int):
        raise ValueError("Embedding checkpoint part has an invalid chunk count")

    file_name = raw_part.get("file_name")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise ValueError("Embedding checkpoint part has an invalid file name")

    part_path = checkpoint_path / file_name
    if not part_path.is_file():
        raise ValueError("Embedding checkpoint part is missing")

    if raw_part.get("sha256") != source_sha256(part_path):
        raise ValueError("Embedding checkpoint part hash does not match")

    return raw_part


def _write_parquet_part(
    path: Path,
    documents: list[dict[str, Any]],
    schema: Any,
    pa: Any,
    parquet: Any,
) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    table = pa.Table.from_pylist(documents, schema=schema)

    try:
        parquet.write_table(table, temporary_path, compression="zstd")
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _combine_embedding_parts(
    artifact_path: Path,
    checkpoint_path: Path,
    parts: list[dict[str, Any]],
    schema: Any,
    parquet: Any,
) -> None:
    temporary_path = artifact_path.with_suffix(f"{artifact_path.suffix}.tmp")
    writer = parquet.ParquetWriter(temporary_path, schema, compression="zstd")

    try:
        for part in parts:
            part_path = checkpoint_path / str(part["file_name"])
            part_file = parquet.ParquetFile(part_path)

            for record_batch in part_file.iter_batches():
                writer.write_batch(record_batch)
    except BaseException:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    temporary_path.replace(artifact_path)


def _cleanup_embedding_checkpoint(
    checkpoint_path: Path,
    parts: list[dict[str, Any]],
) -> None:
    for part in parts:
        part_path = checkpoint_path / str(part["file_name"])
        part_path.unlink(missing_ok=True)

        temporary_path = part_path.with_suffix(f"{part_path.suffix}.tmp")
        temporary_path.unlink(missing_ok=True)

    (checkpoint_path / "state.json").unlink(missing_ok=True)
    (checkpoint_path / "state.json.tmp").unlink(missing_ok=True)

    try:
        checkpoint_path.rmdir()
    except OSError:
        return


def _emit_embedding_progress(
    callback: Callable[[EmbeddingProgress], None] | None,
    *,
    completed_batches: int,
    total_batches: int,
    item_count: int,
    chunk_count: int,
    resumed_batch_count: int,
    phase: str,
) -> None:
    if callback is None:
        return

    callback(
        EmbeddingProgress(
            completed_batches=completed_batches,
            total_batches=total_batches,
            item_count=item_count,
            chunk_count=chunk_count,
            resumed_batch_count=resumed_batch_count,
            phase=phase,
        )
    )


def _read_embedding_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Embedding artifact manifest is unreadable") from exc

    if not isinstance(value, dict):
        raise ValueError("Embedding artifact manifest must be a JSON object")

    if value.get("schema_version") != "embedding-artifact-v1":
        raise ValueError("Unsupported embedding artifact manifest version")

    model = value.get("embedding_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Embedding artifact manifest has an invalid model")

    return value


def _manifest_integer(manifest: Mapping[str, Any], field: str) -> int:
    value = manifest.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Embedding artifact manifest has invalid {field}")

    return value


def _validate_artifact_rows(
    artifact_path: Path,
    *,
    batch_size: int,
    dimensions: int,
    expected_count: int,
    expected_model: str,
) -> None:
    seen_chunk_ids: set[str] = set()
    row_count = 0

    for rows in iter_catalog_rows(artifact_path, batch_size=batch_size):
        for row in rows:
            chunk_id = str(row.get("chunk_id") or "").strip()
            if not chunk_id or chunk_id in seen_chunk_ids:
                raise ValueError("Embedding artifact contains an invalid chunk_id")

            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != dimensions:
                raise ValueError("Embedding artifact contains a wrong vector dimension")

            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("Embedding artifact contains non-finite values")

            if row.get("embedding_model") != expected_model:
                raise ValueError("Embedding artifact row has the wrong model")

            seen_chunk_ids.add(chunk_id)
            row_count += 1

    if row_count != expected_count:
        raise ValueError("Embedding artifact row count does not match its manifest")


def _serialized_dataclass(value: Any) -> dict[str, Any]:
    values = asdict(value)

    for field, field_value in tuple(values.items()):
        if isinstance(field_value, Path):
            values[field] = str(field_value)

    return values


def _require_distinct_paths(*paths: Path) -> None:
    resolved_paths = tuple(path.resolve() for path in paths)
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Catalog, artifact, and manifest paths must be distinct")


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
