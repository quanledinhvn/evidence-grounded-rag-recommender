from __future__ import annotations

import os
import unittest

from backend.search.cli import create_opensearch_client
from backend.search.config import SearchSettings
from backend.search.opensearch import OpenSearchGateway


_RUN_INTEGRATION = os.environ.get("RUN_OPENSEARCH_INTEGRATION") == "1"


@unittest.skipUnless(_RUN_INTEGRATION, "set RUN_OPENSEARCH_INTEGRATION=1")
class OpenSearchIntegrationTests(unittest.TestCase):
    def test_indexes_and_retrieves_bm25_and_semantic_documents(self) -> None:
        settings = SearchSettings(
            item_index="it4613-smoke-items-v1",
            chunk_index="it4613-smoke-chunks-v1",
            embedding_dimensions=3,
            verify_certificates=False,
        )
        client = create_opensearch_client(settings)
        gateway = OpenSearchGateway(
            client,
            item_index=settings.item_index,
            chunk_index=settings.chunk_index,

            dimensions=settings.embedding_dimensions,
        )

        try:
            gateway.ensure_indices()
            gateway.index_items(
                (
                    {
                        "item_id": "candidate-1",
                        "title": "Deep Space Adventure",
                        "description": "Explorers travel through a distant galaxy.",
                        "metadata_hash": "item-hash",
                    },
                )
            )
            gateway.index_chunks(
                (
                    {
                        "chunk_id": "candidate-1:description:0000",
                        "item_id": "candidate-1",
                        "field_name": "description",

                        "ordinal": 0,
                        "raw_text": "Explorers travel through a distant galaxy.",
                        "raw_text_hash": "chunk-hash",

                        "start_char": 0,
                        "end_char": 42,
                        "source_id": "catalog:candidate-1",

                        "embedding_model": "smoke-model",
                        "embedding": [1.0, 0.0, 0.0],
                    },
                )
            )
            gateway.refresh_indices()

            reusable = gateway.reusable_chunk_ids(
                {"candidate-1:description:0000": "chunk-hash"},
                embedding_model="smoke-model",
            )

            bm25_hits = gateway.bm25_search("space adventure", limit=5, explain=True)
            semantic_hits = gateway.semantic_search((1.0, 0.0, 0.0), limit=5)

            self.assertEqual(bm25_hits[0]["item_id"], "candidate-1")
            self.assertGreater(bm25_hits[0]["score"], 0.0)
            self.assertIn("explanation", bm25_hits[0])

            self.assertEqual(semantic_hits[0]["item_id"], "candidate-1")
            self.assertGreater(semantic_hits[0]["score"], 0.0)
            self.assertEqual(
                reusable,
                frozenset({"candidate-1:description:0000"}),
            )
        finally:
            client.indices.delete(
                index=f"{settings.item_index},{settings.chunk_index}",
                ignore_unavailable=True,
            )


if __name__ == "__main__":
    unittest.main()
