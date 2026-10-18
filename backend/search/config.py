from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

_INDEX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(frozen=True)
class SearchSettings:
    opensearch_url: str = "http://localhost:9200"
    item_index: str = "it4613-items-v1"
    chunk_index: str = "it4613-chunks-v1"

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 768
    embedding_batch_size: int = 128

    embedding_timeout_seconds: float = 60.0
    embedding_max_retries: int = 5

    request_timeout_seconds: float = 30.0
    verify_certificates: bool = True

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SearchSettings:
        source = os.environ if environ is None else environ

        settings = cls(
            opensearch_url=source.get(
                "OPENSEARCH_URL",
                cls.opensearch_url,
            ).rstrip("/"),
            item_index=source.get("OPENSEARCH_ITEM_INDEX", cls.item_index),
            chunk_index=source.get("OPENSEARCH_CHUNK_INDEX", cls.chunk_index),

            embedding_model=source.get(
                "OPENAI_EMBEDDING_MODEL",
                cls.embedding_model,
            ),
            embedding_dimensions=_integer(
                source,
                "OPENAI_EMBEDDING_DIMENSIONS",
                cls.embedding_dimensions,
            ),
            embedding_batch_size=_integer(
                source,
                "OPENAI_EMBEDDING_BATCH_SIZE",
                cls.embedding_batch_size,
            ),
            embedding_timeout_seconds=_float(
                source,
                "OPENAI_EMBEDDING_TIMEOUT_SECONDS",
                cls.embedding_timeout_seconds,
            ),
            embedding_max_retries=_integer(
                source,
                "OPENAI_EMBEDDING_MAX_RETRIES",
                cls.embedding_max_retries,
            ),

            request_timeout_seconds=_float(
                source,
                "OPENSEARCH_REQUEST_TIMEOUT_SECONDS",
                cls.request_timeout_seconds,
            ),
            verify_certificates=_boolean(
                source,
                "OPENSEARCH_VERIFY_CERTIFICATES",
                cls.verify_certificates,
            ),
        )
        settings.validate()

        return settings

    def validate(self) -> None:
        parsed_url = urlparse(self.opensearch_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("OPENSEARCH_URL must be an absolute HTTP(S) URL")

        if parsed_url.username or parsed_url.password:
            raise ValueError("OPENSEARCH_URL must not contain credentials")

        for name, value in (
            ("OPENSEARCH_ITEM_INDEX", self.item_index),
            ("OPENSEARCH_CHUNK_INDEX", self.chunk_index),
        ):
            if not _INDEX_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase OpenSearch index name")

        if not self.embedding_model.strip():
            raise ValueError("OPENAI_EMBEDDING_MODEL must not be empty")

        if self.embedding_dimensions <= 0:
            raise ValueError("OPENAI_EMBEDDING_DIMENSIONS must be positive")

        if self.embedding_batch_size <= 0:
            raise ValueError("OPENAI_EMBEDDING_BATCH_SIZE must be positive")

        if self.embedding_timeout_seconds <= 0:
            raise ValueError("OPENAI_EMBEDDING_TIMEOUT_SECONDS must be positive")

        if self.embedding_max_retries < 0:
            raise ValueError("OPENAI_EMBEDDING_MAX_RETRIES must be non-negative")

        if self.request_timeout_seconds <= 0:
            raise ValueError("OPENSEARCH_REQUEST_TIMEOUT_SECONDS must be positive")


def _integer(source: Mapping[str, str], name: str, default: int) -> int:
    raw_value = source.get(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(source: Mapping[str, str], name: str, default: float) -> float:
    raw_value = source.get(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _boolean(source: Mapping[str, str], name: str, default: bool) -> bool:
    raw_value = source.get(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean")
