from __future__ import annotations

import hashlib
import math
import unittest

from types import SimpleNamespace
from unittest.mock import patch

from backend.search.chunking import ChunkingPolicy, build_catalog_chunks
from backend.search.config import SearchSettings
from backend.search.embeddings import OpenAIEmbeddingProvider
from backend.search.opensearch import (
    OpenSearchGateway,
    chunk_index_body,
    item_index_body,
)


class SearchSettingsTests(unittest.TestCase):
    def test_loads_and_validates_environment(self) -> None:
        settings = SearchSettings.from_env(
            {
                "OPENSEARCH_URL": "https://search.internal:9200/",
                "OPENSEARCH_ITEM_INDEX": "movies-items-v2",
                "OPENSEARCH_CHUNK_INDEX": "movies-chunks-v2",
                "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
                "OPENAI_EMBEDDING_DIMENSIONS": "384",
                "OPENAI_EMBEDDING_BATCH_SIZE": "64",
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS": "45",
                "OPENAI_EMBEDDING_MAX_RETRIES": "4",
                "OPENSEARCH_REQUEST_TIMEOUT_SECONDS": "12.5",
                "OPENSEARCH_VERIFY_CERTIFICATES": "true",
            }
        )

        self.assertEqual(settings.opensearch_url, "https://search.internal:9200")
        self.assertEqual(settings.item_index, "movies-items-v2")
        self.assertEqual(settings.chunk_index, "movies-chunks-v2")

        self.assertEqual(settings.embedding_dimensions, 384)
        self.assertEqual(settings.embedding_batch_size, 64)
        self.assertEqual(settings.embedding_timeout_seconds, 45.0)
        self.assertEqual(settings.embedding_max_retries, 4)
        self.assertEqual(settings.request_timeout_seconds, 12.5)

        self.assertTrue(settings.verify_certificates)

    def test_rejects_unsafe_or_invalid_settings(self) -> None:
        invalid_environments = (
            {"OPENAI_EMBEDDING_DIMENSIONS": "0"},
            {"OPENAI_EMBEDDING_BATCH_SIZE": "0"},
            {"OPENAI_EMBEDDING_TIMEOUT_SECONDS": "0"},
            {"OPENAI_EMBEDDING_MAX_RETRIES": "-1"},
            {"OPENSEARCH_URL": "search.internal:9200"},
            {"OPENSEARCH_ITEM_INDEX": "UPPERCASE"},
            {"OPENSEARCH_VERIFY_CERTIFICATES": "sometimes"},
        )

        for environment in invalid_environments:
            with self.subTest(environment=environment):
                with self.assertRaises(ValueError):
                    SearchSettings.from_env(environment)


class CatalogChunkingTests(unittest.TestCase):
    def test_builds_deterministic_raw_span_chunks(self) -> None:
        row = {
            "item_id": "movie-1",
            "title": "  A Space Odyssey  ",
            "description": "one two three four five six",
            "category": "Science Fiction",
        }
        policy = ChunkingPolicy(
            max_words=4,
            overlap_words=1,
            fields=("title", "description", "category"),
        )

        chunks = build_catalog_chunks(row, policy)

        self.assertEqual(
            [(chunk.field_name, chunk.ordinal, chunk.text) for chunk in chunks],
            [
                ("title", 0, "A Space Odyssey"),
                ("description", 0, "one two three four"),
                ("description", 1, "four five six"),
                ("category", 0, "Science Fiction"),
            ],
        )
        self.assertEqual(chunks[0].chunk_id, "movie-1:title:0000")
        self.assertEqual((chunks[0].start_char, chunks[0].end_char), (2, 17))

        expected_hash = hashlib.sha256(b"A Space Odyssey").hexdigest()
        self.assertEqual(chunks[0].text_sha256, expected_hash)
        self.assertEqual(chunks, build_catalog_chunks(row, policy))

    def test_rejects_missing_identity_and_invalid_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "item_id"):
            build_catalog_chunks({"description": "text"})

        invalid_policies = (
            ChunkingPolicy(max_words=0),
            ChunkingPolicy(max_words=4, overlap_words=4),
            ChunkingPolicy(fields=("title", "title")),
        )

        for policy in invalid_policies:
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    build_catalog_chunks({"item_id": "i1", "title": "text"}, policy)


class _FakeEmbeddingsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **request: object) -> SimpleNamespace:
        self.calls.append(request)
        texts = request["input"]
        dimensions = int(request["dimensions"])

        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1)] * dimensions)
            for index, _ in enumerate(texts)
        ]

        return SimpleNamespace(data=list(reversed(data)))


class OpenAIEmbeddingProviderTests(unittest.TestCase):
    def test_configures_sdk_timeout_and_retries(self) -> None:
        endpoint = _FakeEmbeddingsEndpoint()
        client = SimpleNamespace(embeddings=endpoint)

        with patch("openai.OpenAI", return_value=client) as factory:
            provider = OpenAIEmbeddingProvider(
                dimensions=2,
                timeout_seconds=45.0,
                max_retries=4,
            )
            provider.embed(("alpha",))

        factory.assert_called_once_with(timeout=45.0, max_retries=4)

    def test_batches_requests_and_preserves_input_order(self) -> None:
        endpoint = _FakeEmbeddingsEndpoint()
        client = SimpleNamespace(embeddings=endpoint)

        provider = OpenAIEmbeddingProvider(
            dimensions=3,
            batch_size=2,
            client=client,
        )
        vectors = provider.embed(("alpha", "beta", "gamma"))

        self.assertEqual(vectors, ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 1.0, 1.0)))
        self.assertEqual([call["input"] for call in endpoint.calls], [["alpha", "beta"], ["gamma"]])

        self.assertEqual(endpoint.calls[0]["model"], "text-embedding-3-small")
        self.assertEqual(endpoint.calls[0]["dimensions"], 3)
        self.assertEqual(endpoint.calls[0]["encoding_format"], "float")

    def test_rejects_empty_text_and_invalid_vectors(self) -> None:
        endpoint = _FakeEmbeddingsEndpoint()
        client = SimpleNamespace(embeddings=endpoint)

        provider = OpenAIEmbeddingProvider(dimensions=2, client=client)
        with self.assertRaisesRegex(ValueError, "empty"):
            provider.embed(("valid", "  "))

        endpoint.create = lambda **_: SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[math.nan, 0.0])]
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            provider.embed(("valid",))


class _FakeIndices:
    def __init__(self) -> None:
        self.existing: set[str] = set()
        self.created: list[tuple[str, dict[str, object]]] = []
        self.bodies: dict[str, dict[str, object]] = {}

    def exists(self, *, index: str) -> bool:
        return index in self.existing

    def create(self, *, index: str, body: dict[str, object]) -> None:
        self.existing.add(index)
        self.created.append((index, body))
        self.bodies[index] = body

    def get_mapping(self, *, index: str) -> dict[str, object]:
        return {index: {"mappings": self.bodies[index]["mappings"]}}


class _FakeOpenSearchClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()
        self.bulk_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []

    def bulk(self, **request: object) -> dict[str, object]:
        self.bulk_calls.append(request)
        return {"errors": False, "items": []}

    def search(self, **request: object) -> dict[str, object]:
        self.search_calls.append(request)
        return {
            "hits": {
                "hits": [
                    {
                        "_id": "i2",
                        "_score": 1.25,
                        "_source": {"item_id": "i2", "title": "Candidate"},
                        "_explanation": {"value": 1.25},
                    }
                ]
            }
        }

    def mget(self, **request: object) -> dict[str, object]:
        return {
            "docs": [
                {
                    "_id": "i1:title:0000",
                    "found": True,
                    "_source": {
                        "raw_text_hash": "matching-hash",
                        "embedding_model": "embedding-v1",
                    },
                },
                {"_id": "i2:title:0000", "found": False},
            ]
        }


class OpenSearchGatewayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeOpenSearchClient()
        self.gateway = OpenSearchGateway(
            self.client,
            item_index="items-v1",
            chunk_index="chunks-v1",
            dimensions=3,
        )

    def test_index_mappings_lock_bm25_and_vector_contracts(self) -> None:
        item_body = item_index_body()
        chunk_body = chunk_index_body(768)

        similarity = item_body["settings"]["index"]["similarity"]["default"]
        self.assertEqual(similarity, {"type": "BM25", "k1": 1.2, "b": 0.75})
        self.assertEqual(item_body["mappings"]["dynamic"], "strict")

        vector = chunk_body["mappings"]["properties"]["embedding"]
        self.assertEqual(vector["dimension"], 768)
        self.assertEqual(vector["method"]["name"], "hnsw")
        self.assertEqual(vector["method"]["space_type"], "cosinesimil")

    def test_creates_missing_indices_idempotently(self) -> None:
        self.gateway.ensure_indices()
        self.gateway.ensure_indices()

        self.assertEqual(
            [index for index, _ in self.client.indices.created],
            ["items-v1", "chunks-v1"],
        )

    def test_rejects_existing_index_with_incompatible_vector_dimension(self) -> None:
        self.client.indices.existing.update(("items-v1", "chunks-v1"))
        self.client.indices.bodies["items-v1"] = item_index_body()
        self.client.indices.bodies["chunks-v1"] = chunk_index_body(2)

        with self.assertRaisesRegex(RuntimeError, "dimension"):
            self.gateway.ensure_indices()

    def test_bulk_indexes_with_stable_document_ids(self) -> None:
        item_count = self.gateway.index_items(
            ({"item_id": "i1", "title": "One"},),
            refresh=True,
        )
        chunk_count = self.gateway.index_chunks(
            (
                {
                    "chunk_id": "i1:title:0000",
                    "item_id": "i1",
                    "embedding": [0.0, 1.0, 0.0],
                },
            )
        )

        self.assertEqual((item_count, chunk_count), (1, 1))
        item_action = self.client.bulk_calls[0]["body"][0]["index"]
        chunk_action = self.client.bulk_calls[1]["body"][0]["index"]

        self.assertEqual(item_action, {"_index": "items-v1", "_id": "i1"})
        self.assertEqual(
            chunk_action,
            {"_index": "chunks-v1", "_id": "i1:title:0000"},
        )

    def test_builds_bm25_and_semantic_queries_with_seen_filter(self) -> None:
        bm25_hits = self.gateway.bm25_search(
            "space adventure",
            limit=10,
            excluded_item_ids=("seen-2", "seen-1"),
            explain=True,
        )
        semantic_hits = self.gateway.semantic_search(
            (0.0, 1.0, 0.0),
            limit=5,
            excluded_item_ids=("seen-1",),
        )

        self.assertEqual(bm25_hits[0]["item_id"], "i2")
        self.assertEqual(bm25_hits[0]["score"], 1.25)
        self.assertIn("explanation", bm25_hits[0])

        bm25_body = self.client.search_calls[0]["body"]
        self.assertTrue(bm25_body["explain"])
        self.assertEqual(bm25_body["size"], 10)
        self.assertEqual(
            bm25_body["query"]["bool"]["must_not"][0]["terms"]["item_id"],
            ["seen-1", "seen-2"],
        )

        semantic_body = self.client.search_calls[1]["body"]
        knn = semantic_body["query"]["knn"]["embedding"]
        self.assertEqual(knn["vector"], [0.0, 1.0, 0.0])
        self.assertEqual(knn["k"], 5)

        self.assertEqual(
            knn["filter"]["bool"]["must_not"][0]["terms"]["item_id"],
            ["seen-1"],
        )

    def test_rejects_wrong_semantic_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimension"):
            self.gateway.semantic_search((0.0, 1.0), limit=5)

    def test_finds_only_reusable_chunks_with_matching_contract(self) -> None:
        reusable = self.gateway.reusable_chunk_ids(
            {
                "i1:title:0000": "matching-hash",
                "i2:title:0000": "missing-hash",
            },
            embedding_model="embedding-v1",
        )

        self.assertEqual(reusable, frozenset({"i1:title:0000"}))


if __name__ == "__main__":
    unittest.main()
