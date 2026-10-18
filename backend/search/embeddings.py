from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol


class EmbeddingsClient(Protocol):
    embeddings: Any


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 768,
        batch_size: int = 128,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
        client: EmbeddingsClient | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        self._client = client
        self._validate_configuration()

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        inputs = tuple(texts)
        if not inputs:
            return ()

        if any(not isinstance(value, str) or not value.strip() for value in inputs):
            raise ValueError("Embedding inputs must not contain empty text")

        vectors: list[tuple[float, ...]] = []
        client = self._client_or_default()

        for start in range(0, len(inputs), self.batch_size):
            batch = list(inputs[start : start + self.batch_size])
            response = client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self.dimensions,

                encoding_format="float",
            )
            vectors.extend(self._response_vectors(response, len(batch)))

        return tuple(vectors)

    def _validate_configuration(self) -> None:
        if not self.model.strip():
            raise ValueError("Embedding model must not be empty")

        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")

        if self.batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")

        if self.timeout_seconds <= 0:
            raise ValueError("Embedding timeout must be positive")

        if self.max_retries < 0:
            raise ValueError("Embedding max retries must be non-negative")

    def _client_or_default(self) -> EmbeddingsClient:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            message = "OpenAI embeddings require the project 'search' dependencies."
            raise RuntimeError(message) from exc

        self._client = OpenAI(
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )

        return self._client

    def _response_vectors(
        self,
        response: Any,
        expected_count: int,
    ) -> list[tuple[float, ...]]:
        data = tuple(response.data)
        if len(data) != expected_count:
            raise RuntimeError("Embedding response count does not match the request")

        by_index = {int(item.index): item.embedding for item in data}
        if set(by_index) != set(range(expected_count)):
            raise RuntimeError("Embedding response contains invalid indices")

        return [self._validated_vector(by_index[index]) for index in range(expected_count)]

    def _validated_vector(self, values: Sequence[float]) -> tuple[float, ...]:
        vector = tuple(float(value) for value in values)
        if len(vector) != self.dimensions:
            raise RuntimeError("Embedding response has the wrong dimension")

        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding values must be finite")

        return vector
