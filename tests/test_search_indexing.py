from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path
from typing import Any

from backend.data.io import source_sha256, write_parquet
from backend.search.chunking import ChunkingPolicy
from backend.search.indexing import index_catalog, iter_catalog_rows, plan_catalog


class _FakeEmbedder:
    model = "test-embedding-model"
    dimensions = 3

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.calls.append(texts)

        return tuple((float(index), 1.0, 0.0) for index, _ in enumerate(texts))


class _FakeGateway:
    item_index = "items-v1"
    chunk_index = "chunks-v1"

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

        self.ensure_count = 0
        self.refresh_count = 0

    def ensure_indices(self) -> None:
        self.ensure_count += 1

    def refresh_indices(self) -> None:
        self.refresh_count += 1

    def index_items(
        self,
        documents: Any,
        *,
        refresh: bool = False,
    ) -> int:
        batch = [dict(document) for document in documents]
        self.items.extend(batch)

        return len(batch)

    def index_chunks(
        self,
        documents: Any,
        *,
        refresh: bool = False,
    ) -> int:
        batch = [dict(document) for document in documents]
        self.chunks.extend(batch)

        return len(batch)

    def reusable_chunk_ids(
        self,
        chunk_hashes: dict[str, str],
        *,
        embedding_model: str,
    ) -> frozenset[str]:
        return frozenset(
            chunk["chunk_id"]
            for chunk in self.chunks
            if chunk["chunk_id"] in chunk_hashes
            and chunk["raw_text_hash"] == chunk_hashes[chunk["chunk_id"]]
            and chunk["embedding_model"] == embedding_model
        )


class SearchIndexingTests(unittest.TestCase):
    def test_plans_catalog_without_embedding_or_indexing(self) -> None:
        rows = [
            {
                "item_id": "i1",
                "title": "one two three four five",
                "description": "alpha beta",
            },
            {"item_id": "i2", "title": "", "description": None},
        ]

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.parquet"
            write_parquet(catalog_path, rows)

            result = plan_catalog(
                catalog_path,
                chunking_policy=ChunkingPolicy(
                    max_words=3,
                    overlap_words=1,
                    fields=("title", "description"),
                ),
                catalog_batch_size=1,
            )

            expected_hash = source_sha256(catalog_path)

        self.assertEqual(result.item_count, 2)
        self.assertEqual(result.text_covered_item_count, 1)
        self.assertEqual((result.chunk_count, result.word_count), (3, 8))
        self.assertEqual(result.catalog_sha256, expected_hash)

    def test_streams_parquet_in_bounded_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.parquet"
            write_parquet(
                catalog_path,
                [{"item_id": f"i{index}"} for index in range(5)],
            )

            batches = list(iter_catalog_rows(catalog_path, batch_size=2))

        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        self.assertEqual(batches[0][0]["item_id"], "i0")

    def test_indexes_items_chunks_and_writes_reproducible_manifest(self) -> None:
        rows = [
            {
                "item_id": "i1",
                "title": "First Movie",
                "description": "space travel",
                "source_id": "catalog:i1",
                "source_line": 1,
            },
            {
                "item_id": "i2",
                "title": "Second Movie",
                "description": "",
                "source_id": "catalog:i2",
                "source_line": 2,
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.parquet"
            manifest_path = root / "search-manifest.json"

            write_parquet(catalog_path, rows)
            gateway = _FakeGateway()
            embedder = _FakeEmbedder()

            result = index_catalog(
                catalog_path,
                manifest_path,
                gateway,
                embedder,
                chunking_policy=ChunkingPolicy(
                    max_words=4,
                    overlap_words=1,
                    fields=("title", "description"),
                ),
                catalog_batch_size=1,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual((result.item_count, result.chunk_count), (2, 3))
        self.assertEqual(result.embedded_chunk_count, 3)
        self.assertEqual(result.reused_chunk_count, 0)
        self.assertEqual((gateway.ensure_count, gateway.refresh_count), (1, 1))

        self.assertEqual([item["item_id"] for item in gateway.items], ["i1", "i2"])
        self.assertEqual(
            [chunk["chunk_id"] for chunk in gateway.chunks],
            ["i1:title:0000", "i1:description:0000", "i2:title:0000"],
        )
        self.assertEqual(gateway.chunks[0]["embedding"], [0.0, 1.0, 0.0])
        self.assertEqual(gateway.chunks[0]["embedding_model"], embedder.model)

        self.assertEqual(manifest["schema_version"], "search-index-manifest-v1")
        self.assertEqual(manifest["catalog_sha256"], result.catalog_sha256)
        self.assertEqual(manifest["embedding_dimensions"], 3)

        self.assertEqual(
            manifest["chunking_policy"],
            {
                "fields": ["title", "description"],
                "max_words": 4,
                "overlap_words": 1,
            },
        )

    def test_rejects_duplicate_catalog_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.parquet"
            manifest_path = root / "manifest.json"

            write_parquet(
                catalog_path,
                [
                    {"item_id": "duplicate", "title": "One"},
                    {"item_id": "duplicate", "title": "Two"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "Duplicate catalog item"):
                index_catalog(
                    catalog_path,
                    manifest_path,
                    _FakeGateway(),
                    _FakeEmbedder(),
                    catalog_batch_size=1,
                )

    def test_resumes_without_reembedding_matching_chunks(self) -> None:
        rows = [{"item_id": "i1", "title": "Already embedded"}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.parquet"
            write_parquet(catalog_path, rows)

            gateway = _FakeGateway()
            first_embedder = _FakeEmbedder()
            index_catalog(
                catalog_path,
                root / "first.json",
                gateway,
                first_embedder,
            )

            second_embedder = _FakeEmbedder()
            result = index_catalog(
                catalog_path,
                root / "second.json",
                gateway,
                second_embedder,
            )

        self.assertEqual(result.chunk_count, 1)
        self.assertEqual(result.embedded_chunk_count, 0)
        self.assertEqual(result.reused_chunk_count, 1)
        self.assertEqual(len(first_embedder.calls), 1)
        self.assertEqual(second_embedder.calls, [])


if __name__ == "__main__":
    unittest.main()
