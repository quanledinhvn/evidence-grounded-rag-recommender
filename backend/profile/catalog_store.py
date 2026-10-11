from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.data.io import read_parquet
from backend.profile.builder import CatalogItem


@dataclass(frozen=True)
class ParquetCatalogStore:
    artifact_version: str
    path: Path

    def get_many(self, item_ids: frozenset[str]) -> Mapping[str, CatalogItem]:
        if not item_ids:
            return {}

        rows = read_parquet(
            self.path,
            filters=[("item_id", "in", sorted(item_ids))],
        )
        items: dict[str, CatalogItem] = {}

        for row in rows:
            item_id = str(row.get("item_id") or "").strip()
            if item_id in items:
                raise ValueError(f"Duplicate catalog item: {item_id}")

            items[item_id] = CatalogItem(
                item_id=item_id,
                source_id=str(row.get("source_id") or "").strip(),
                title=_text(row.get("title")),
                subtitle=_text(row.get("subtitle")),
                description=_text(row.get("description")),

                feature=_text(row.get("feature")),
                main_category=_text(row.get("main_category")),
                category=_text(row.get("category")),

                store=_text(row.get("store")),
                details=_text(row.get("details")),
            )

        return items


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
