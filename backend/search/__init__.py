"""Production-oriented lexical and semantic retrieval infrastructure."""

from backend.search.chunking import CatalogChunk, ChunkingPolicy, build_catalog_chunks
from backend.search.config import SearchSettings
from backend.search.embeddings import OpenAIEmbeddingProvider
from backend.search.opensearch import OpenSearchGateway

__all__ = [
    "CatalogChunk",
    "ChunkingPolicy",
    "OpenAIEmbeddingProvider",
    "OpenSearchGateway",
    "SearchSettings",
    "build_catalog_chunks",
]
