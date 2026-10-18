from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol


class OpenSearchClient(Protocol):
    indices: Any

    def bulk(self, *, body: Sequence[Mapping[str, Any]], refresh: bool) -> Any: ...

    def mget(self, *, index: str, body: Mapping[str, Any]) -> Any: ...

    def search(self, *, index: str, body: Mapping[str, Any]) -> Any: ...


def item_index_body() -> dict[str, Any]:
    text = {"type": "text"}

    return {
        "settings": {
            "index": {
                "similarity": {
                    "default": {"type": "BM25", "k1": 1.2, "b": 0.75}
                }
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "item_id": {"type": "keyword"},
                "title": text,
                "subtitle": text,

                "description": text,
                "feature": text,
                "main_category": text,

                "category": text,
                "store": text,
                "details": text,

                "metadata_text": text,
                "average_rating": {"type": "float"},
                "rating_number": {"type": "long"},

                "source_id": {"type": "keyword"},
                "source_line": {"type": "long"},
                "metadata_hash": {"type": "keyword"},
            },
        },
    }


def chunk_index_body(dimensions: int) -> dict[str, Any]:
    if dimensions <= 0:
        raise ValueError("Vector dimensions must be positive")

    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "item_id": {"type": "keyword"},
                "field_name": {"type": "keyword"},

                "ordinal": {"type": "integer"},
                "raw_text": {"type": "text", "index": False},
                "raw_text_hash": {"type": "keyword"},

                "start_char": {"type": "integer"},
                "end_char": {"type": "integer"},
                "source_id": {"type": "keyword"},

                "embedding_model": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            },
        },
    }


class OpenSearchGateway:
    def __init__(
        self,
        client: OpenSearchClient,
        *,
        item_index: str,
        chunk_index: str,
        dimensions: int,
    ) -> None:
        self._client = client
        self.item_index = item_index
        self.chunk_index = chunk_index

        self.dimensions = dimensions
        if dimensions <= 0:
            raise ValueError("Vector dimensions must be positive")

    def ensure_indices(self) -> None:
        definitions = (
            (self.item_index, item_index_body()),
            (self.chunk_index, chunk_index_body(self.dimensions)),
        )

        for index, body in definitions:
            if self._client.indices.exists(index=index):
                self._validate_existing_mapping(index, body["mappings"])
                continue

            self._client.indices.create(index=index, body=body)

    def refresh_indices(self) -> None:
        self._client.indices.refresh(
            index=f"{self.item_index},{self.chunk_index}"
        )

    def index_items(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        refresh: bool = False,
    ) -> int:
        return self._bulk_index(
            self.item_index,
            "item_id",
            documents,
            refresh=refresh,
        )

    def index_chunks(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        refresh: bool = False,
    ) -> int:
        return self._bulk_index(
            self.chunk_index,
            "chunk_id",
            documents,
            refresh=refresh,
        )

    def bm25_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_item_ids: Sequence[str] = (),
        explain: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("BM25 query must not be empty")

        _validate_limit(limit)
        exclusions = sorted(set(excluded_item_ids))

        body = {
            "size": limit,
            "explain": explain,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": normalized_query,
                                "fields": [
                                    "title^3",
                                    "subtitle^2",
                                    "category^2",
                                    "main_category^2",
                                    "feature",
                                    "description",
                                    "details",
                                ],
                                "_name": "bm25_branch",
                            }
                        }
                    ],
                    "must_not": _exclusion_clauses(exclusions),
                }
            },
        }
        response = self._client.search(index=self.item_index, body=body)

        return _search_hits(response)

    def semantic_search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        excluded_item_ids: Sequence[str] = (),
    ) -> tuple[dict[str, Any], ...]:
        _validate_limit(limit)
        vector = [float(value) for value in embedding]

        if len(vector) != self.dimensions:
            raise ValueError("Semantic query has the wrong vector dimension")

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Semantic query values must be finite")

        query: dict[str, Any] = {"vector": vector, "k": limit}
        exclusions = sorted(set(excluded_item_ids))

        if exclusions:
            query["filter"] = {
                "bool": {"must_not": _exclusion_clauses(exclusions)}
            }

        body = {"size": limit, "query": {"knn": {"embedding": query}}}
        response = self._client.search(index=self.chunk_index, body=body)

        return _search_hits(response)

    def item_chunks(
        self,
        item_id: str,
        *,
        embedding_model: str,
        limit: int = 1000,
    ) -> tuple[dict[str, Any], ...]:
        """Return indexed semantic chunks for one catalog item."""
        normalized_item_id = item_id.strip()
        normalized_model = embedding_model.strip()

        if not normalized_item_id or not normalized_model:
            raise ValueError("Item ID and embedding model must not be empty")

        _validate_limit(limit)
        body = {
            "size": limit,
            "_source": [
                "chunk_id",
                "item_id",
                "field_name",
                "ordinal",
                "raw_text",
                "embedding_model",
                "embedding",
            ],
            "query": {
                "bool": {
                    "filter": [
                        {"term": {"item_id": normalized_item_id}},
                        {"term": {"embedding_model": normalized_model}},
                    ]
                }
            },
            "sort": [{"field_name": "asc"}, {"ordinal": "asc"}],
        }
        response = self._client.search(index=self.chunk_index, body=body)

        return _search_hits(response)

    def reusable_chunk_ids(
        self,
        chunk_hashes: Mapping[str, str],
        *,
        embedding_model: str,
    ) -> frozenset[str]:
        if not chunk_hashes:
            return frozenset()

        response = self._client.mget(
            index=self.chunk_index,
            body={
                "docs": [
                    {
                        "_id": chunk_id,
                        "_source": ["raw_text_hash", "embedding_model"],
                    }
                    for chunk_id in sorted(chunk_hashes)
                ],
            },
        )
        reusable: set[str] = set()

        for document in response.get("docs", []):
            chunk_id = str(document.get("_id") or "")
            source = document.get("_source") or {}

            if not document.get("found") or chunk_id not in chunk_hashes:
                continue

            if source.get("raw_text_hash") != chunk_hashes[chunk_id]:
                continue

            if source.get("embedding_model") == embedding_model:
                reusable.add(chunk_id)

        return frozenset(reusable)

    def _bulk_index(
        self,
        index: str,
        identity_field: str,
        documents: Iterable[Mapping[str, Any]],
        *,
        refresh: bool,
    ) -> int:
        body: list[Mapping[str, Any]] = []
        count = 0

        for document in documents:
            document_id = str(document.get(identity_field) or "").strip()
            if not document_id:
                raise ValueError(f"Document must contain {identity_field}")

            body.append({"index": {"_index": index, "_id": document_id}})
            body.append(dict(document))
            count += 1

        if not body:
            return 0

        response = self._client.bulk(body=body, refresh=refresh)
        if response.get("errors"):
            raise RuntimeError("OpenSearch bulk indexing reported errors")

        return count

    def _validate_existing_mapping(
        self,
        index: str,
        expected: Mapping[str, Any],
    ) -> None:
        response = self._client.indices.get_mapping(index=index)
        index_definition = response.get(index)

        if index_definition is None and len(response) == 1:
            index_definition = next(iter(response.values()))

        actual = (index_definition or {}).get("mappings", {})
        mismatch = _first_contract_mismatch(expected, actual, path="mappings")
        if mismatch is not None:
            message = f"Existing index {index!r} has incompatible {mismatch}"
            raise RuntimeError(message)


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or limit <= 0:
        raise ValueError("Search limit must be a positive integer")


def _exclusion_clauses(item_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not item_ids:
        return []

    return [{"terms": {"item_id": list(item_ids)}}]


def _search_hits(response: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_hits = response.get("hits", {}).get("hits", [])
    hits: list[dict[str, Any]] = []

    for raw_hit in raw_hits:
        source = dict(raw_hit.get("_source") or {})
        hit = {
            **source,
            "document_id": str(raw_hit.get("_id") or ""),
            "score": float(raw_hit.get("_score") or 0.0),
        }

        if "_explanation" in raw_hit:
            hit["explanation"] = raw_hit["_explanation"]

        hits.append(hit)

    return tuple(hits)


def _first_contract_mismatch(
    expected: Any,
    actual: Any,
    *,
    path: str,
) -> str | None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return path

        for key, expected_value in expected.items():
            if key not in actual:
                return f"{path}.{key}"

            mismatch = _first_contract_mismatch(
                expected_value,
                actual[key],
                path=f"{path}.{key}",
            )
            if mismatch is not None:
                return mismatch

        return None

    return None if expected == actual else path
