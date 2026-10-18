from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                message = f"Invalid JSON at {path}:{line_number}: {exc.msg}"
                raise ValueError(message) from exc

            if not isinstance(record, dict):
                message = f"Expected a JSON object at {path}:{line_number}"
                raise ValueError(message)

            yield line_number, record


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_parquet(
    path: Path,
    *,
    filters: list[tuple[str, str, Any]] | None = None,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        message = "Parquet support requires the project dependency 'pyarrow'."
        raise RuntimeError(message) from exc

    return parquet.read_table(path, filters=filters).to_pylist()


def write_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    columns: tuple[str, ...] = (),
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        message = "Parquet support requires the project dependency 'pyarrow'."
        raise RuntimeError(message) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({column: pa.array([]) for column in columns})

    parquet.write_table(table, path, compression="zstd")
