from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.data.io import read_parquet
from backend.profile.builder import Interaction


@dataclass(frozen=True)
class InMemoryHistoryStore:
    artifact_version: str
    interactions: tuple[Interaction, ...]

    def get_before(
        self,
        user_id: str,
        as_of: int,
        allowed_splits: tuple[str, ...],
    ) -> Sequence[Interaction]:
        allowed = frozenset(allowed_splits)

        return tuple(
            interaction
            for interaction in self.interactions
            if interaction.user_id == user_id
            and interaction.split in allowed
            and interaction.timestamp < as_of
        )


@dataclass(frozen=True)
class ParquetHistoryStore:
    artifact_version: str
    path: Path

    def get_before(
        self,
        user_id: str,
        as_of: int,
        allowed_splits: tuple[str, ...],
    ) -> Sequence[Interaction]:
        split_filter: tuple[str, str, object]
        if len(allowed_splits) == 1:
            split_filter = ("split", "=", allowed_splits[0])
        else:
            split_filter = ("split", "in", list(allowed_splits))

        rows = read_parquet(
            self.path,
            filters=[
                ("user_id", "=", user_id),
                ("timestamp", "<", as_of),
                split_filter,
            ],
        )

        return tuple(
            Interaction(
                user_id=str(row["user_id"]),
                item_id=str(row["item_id"]),
                rating=float(row["rating"]),
                timestamp=int(row["timestamp"]),
                review_id=str(row["review_id"]),
                split=str(row["split"]),
            )
            for row in rows
        )
