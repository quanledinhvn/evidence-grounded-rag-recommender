from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingPolicy:
    max_words: int = 160
    overlap_words: int = 24

    fields: tuple[str, ...] = (
        "title",
        "subtitle",
        "description",
        "feature",
        "main_category",
        "category",
        "details",
    )


@dataclass(frozen=True)
class CatalogChunk:
    chunk_id: str
    item_id: str
    field_name: str

    ordinal: int
    text: str
    text_sha256: str

    start_char: int
    end_char: int


def build_catalog_chunks(
    row: Mapping[str, object],
    policy: ChunkingPolicy = ChunkingPolicy(),
) -> tuple[CatalogChunk, ...]:
    item_id = str(row.get("item_id") or "").strip()
    if not item_id:
        raise ValueError("Catalog row must contain a non-empty item_id")

    _validate_policy(policy)
    chunks: list[CatalogChunk] = []

    for field_name in policy.fields:
        raw_value = row.get(field_name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue

        chunks.extend(_field_chunks(item_id, field_name, raw_value, policy))

    return tuple(chunks)


def _validate_policy(policy: ChunkingPolicy) -> None:
    if policy.max_words <= 0:
        raise ValueError("max_words must be positive")

    if policy.overlap_words < 0 or policy.overlap_words >= policy.max_words:
        raise ValueError("overlap_words must be between zero and max_words")

    if len(policy.fields) != len(set(policy.fields)):
        raise ValueError("Chunking fields must be unique")

    if any(not field.strip() for field in policy.fields):
        raise ValueError("Chunking fields must not be empty")


def _field_chunks(
    item_id: str,
    field_name: str,
    raw_text: str,
    policy: ChunkingPolicy,
) -> list[CatalogChunk]:
    tokens = tuple(re.finditer(r"\S+", raw_text))
    if not tokens:
        return []

    step = policy.max_words - policy.overlap_words
    chunks: list[CatalogChunk] = []

    for ordinal, token_start in enumerate(range(0, len(tokens), step)):
        token_end = min(token_start + policy.max_words, len(tokens))
        start_char = tokens[token_start].start()
        end_char = tokens[token_end - 1].end()

        text = raw_text[start_char:end_char]
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        chunks.append(
            CatalogChunk(
                chunk_id=f"{item_id}:{field_name}:{ordinal:04d}",
                item_id=item_id,
                field_name=field_name,
                ordinal=ordinal,

                text=text,
                text_sha256=text_hash,
                start_char=start_char,

                end_char=end_char,
            )
        )
        if token_end == len(tokens):
            break

    return chunks
