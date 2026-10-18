from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from implicit.nearest_neighbours import (
    CosineRecommender,
    ItemItemRecommender,
    normalize,
)
from scipy.sparse import csr_matrix, load_npz, save_npz

from backend.data.io import read_parquet, source_sha256, write_parquet

Phase = Literal["validation", "test"]


@dataclass(frozen=True)
class ItemKNNConfig:
    neighbors: int = 50
    num_threads: int = 0

    def __post_init__(self) -> None:
        if self.neighbors <= 0:
            raise ValueError("neighbors must be positive")

        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")


@dataclass
class ItemKNNBundle:
    phase: Phase
    config: ItemKNNConfig
    similarity: csr_matrix

    user_items: csr_matrix
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]

    seen_by_user: dict[str, set[str]]
    positive_users: set[str]
    popularity: list[tuple[str, int]]

    manifest: dict[str, Any]

    def recommend(
        self,
        user_id: str,
        *,
        seen_ids: set[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        if not 0 <= limit <= 50:
            raise ValueError("limit must be between 0 and 50")

        excluded = set(self.seen_by_user.get(user_id, set()))
        excluded.update(seen_ids or set())

        if user_id not in self.positive_users:
            return {
                "status": "cold_start",
                "items": self._popular_items(excluded, limit),
            }

        user_index = self.user_to_index[user_id]
        scores = self.user_items[user_index] @ self.similarity
        index_to_item = {
            index: item_id for item_id, index in self.item_to_index.items()
        }

        candidates = [
            (index_to_item[int(index)], float(score))
            for index, score in zip(scores.indices, scores.data)
            if np.isfinite(score)
            and score > 0.0
            and index_to_item[int(index)] not in excluded
        ]
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))

        items = [
            {"item_id": item_id, "score": score, "source": "item_knn"}
            for item_id, score in candidates[:limit]
        ]

        return {"status": "ok", "items": items}

    def _popular_items(
        self,
        excluded: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        return [
            {"item_id": item_id, "score": float(count), "source": "popularity"}
            for item_id, count in self.popularity
            if item_id not in excluded
        ][:limit]

    @property
    def model_size_bytes(self) -> int:
        return sum(
            array.nbytes
            for matrix in (self.similarity, self.user_items)
            for array in (matrix.data, matrix.indices, matrix.indptr)
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_npz(output_dir / "similarity.npz", self.similarity, compressed=True)
        save_npz(output_dir / "user_items.npz", self.user_items, compressed=True)

        _write_mapping(output_dir / "user_mapping.json", self.user_to_index)
        _write_mapping(output_dir / "item_mapping.json", self.item_to_index)
        _write_json(output_dir / "item_knn_config.json", _config_to_dict(self.config))

        history = {
            "positive_users": sorted(self.positive_users),
            "seen_by_user": {
                user_id: sorted(item_ids)
                for user_id, item_ids in sorted(self.seen_by_user.items())
            },
        }
        _write_json(output_dir / "fit_history.json", history)

        popularity_rows = [
            {"item_id": item_id, "positive_count": count, "rank": rank}
            for rank, (item_id, count) in enumerate(self.popularity, start=1)
        ]
        write_parquet(output_dir / "popularity.parquet", popularity_rows)

        artifact_names = (
            "similarity.npz",
            "user_items.npz",
            "user_mapping.json",
            "item_mapping.json",
            "item_knn_config.json",
            "fit_history.json",
            "popularity.parquet",
        )
        self.manifest["artifact_hashes"] = {
            name: source_sha256(output_dir / name) for name in artifact_names
        }
        self.manifest["artifact_sizes"] = {
            name: (output_dir / name).stat().st_size for name in artifact_names
        }

        self.manifest["config_sha256"] = self.manifest["artifact_hashes"][
            "item_knn_config.json"
        ]
        _write_json(output_dir / "manifest.json", self.manifest)

    @classmethod
    def load(cls, bundle_dir: Path) -> ItemKNNBundle:
        manifest = _read_json(bundle_dir / "manifest.json")
        _verify_artifacts(bundle_dir, manifest["artifact_hashes"])

        config_values = _read_json(bundle_dir / "item_knn_config.json")
        config = ItemKNNConfig(
            neighbors=int(config_values["neighbors"]),
            num_threads=int(config_values["num_threads"]),
        )

        history = _read_json(bundle_dir / "fit_history.json")
        popularity_rows = read_parquet(bundle_dir / "popularity.parquet")
        popularity = [
            (str(row["item_id"]), int(row["positive_count"]))
            for row in sorted(popularity_rows, key=lambda row: int(row["rank"]))
        ]

        return cls(
            phase=manifest["phase"],
            config=config,
            similarity=load_npz(bundle_dir / "similarity.npz").tocsr(),
            user_items=load_npz(bundle_dir / "user_items.npz").tocsr(),
            user_to_index=_mapping_forward(
                _read_json(bundle_dir / "user_mapping.json")
            ),
            item_to_index=_mapping_forward(
                _read_json(bundle_dir / "item_mapping.json")
            ),
            seen_by_user={
                user_id: set(item_ids)
                for user_id, item_ids in history["seen_by_user"].items()
            },
            positive_users=set(history["positive_users"]),
            popularity=popularity,
            manifest=manifest,
        )


def train_item_knn_bundle(
    splits_path: Path,
    catalog_path: Path,
    output_dir: Path,
    *,
    phase: Phase,
    config: ItemKNNConfig | None = None,
) -> ItemKNNBundle:
    if phase not in ("validation", "test"):
        raise ValueError(f"Unsupported training phase: {phase}")

    effective_config = config or ItemKNNConfig()
    catalog_ids = _read_catalog_ids(catalog_path)
    fit_splits = ["train"] if phase == "validation" else ["train", "validation"]

    filters = [("split", "=", fit_splits[0])]
    if len(fit_splits) > 1:
        filters = [("split", "in", fit_splits)]

    fit_rows = read_parquet(splits_path, filters=filters)
    _validate_rows(fit_rows, set(catalog_ids))

    user_ids = sorted({str(row["user_id"]) for row in fit_rows})
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(catalog_ids)}

    seen_by_user: dict[str, set[str]] = defaultdict(set)
    positive_users: set[str] = set()
    popularity_counts: Counter[str] = Counter(
        {item_id: 0 for item_id in catalog_ids}
    )

    rows: list[int] = []
    columns: list[int] = []

    for row in sorted(
        fit_rows,
        key=lambda value: (str(value["user_id"]), str(value["item_id"])),
    ):
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])
        seen_by_user[user_id].add(item_id)

        if float(row["rating"]) < 4.0:
            continue

        rows.append(user_to_index[user_id])
        columns.append(item_to_index[item_id])
        positive_users.add(user_id)

        popularity_counts[item_id] += 1

    values = np.ones(len(rows), dtype=np.float32)
    user_items = csr_matrix(
        (values, (rows, columns)),
        shape=(len(user_ids), len(catalog_ids)),
    )

    model = CosineRecommender(
        K=effective_config.neighbors,
        num_threads=effective_config.num_threads,
    )
    normalized_items = normalize(user_items.T).T.tocsr()
    ItemItemRecommender.fit(model, normalized_items, show_progress=False)

    popularity = sorted(
        popularity_counts.items(),
        key=lambda pair: (-pair[1], pair[0]),
    )
    manifest: dict[str, Any] = {
        "schema_version": "item-knn-bundle-v1",
        "phase": phase,
        "fit_splits": fit_splits,
        "input_hashes": {
            "catalog": source_sha256(catalog_path),
            "splits": source_sha256(splits_path),
        },
        "counts": {
            "catalog_items": len(catalog_ids),
            "fit_interactions": len(fit_rows),
            "positive_pairs": len(rows),
            "users": len(user_ids),
        },
        "artifact_hashes": {},
    }

    code_digest = hashlib.sha256()
    for name in ("bundle.py", "evaluation.py", "cli.py"):
        code_digest.update((Path(__file__).parent / name).read_bytes())

    manifest["code_sha256"] = code_digest.hexdigest()
    bundle = ItemKNNBundle(
        phase=phase,
        config=effective_config,
        similarity=model.similarity.tocsr(),
        user_items=user_items,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        seen_by_user=dict(seen_by_user),
        positive_users=positive_users,
        popularity=popularity,
        manifest=manifest,
    )
    bundle.save(output_dir)

    return bundle


def _read_catalog_ids(catalog_path: Path) -> list[str]:
    catalog_ids: list[str] = []
    catalog_seen: set[str] = set()

    for row in read_parquet(catalog_path):
        raw_item_id = row.get("item_id")
        item_id = "" if raw_item_id is None else str(raw_item_id).strip()

        if not item_id:
            raise ValueError("Catalog requires non-empty item_id")

        if item_id in catalog_seen:
            raise ValueError(f"Duplicate catalog item: {item_id}")

        catalog_seen.add(item_id)
        catalog_ids.append(item_id)

    catalog_ids.sort()
    if not catalog_ids:
        raise ValueError("Catalog must contain at least one item")

    return catalog_ids


def _validate_rows(rows: list[dict[str, Any]], catalog_ids: set[str]) -> None:
    pairs: set[tuple[str, str]] = set()

    for row in rows:
        raw_user_id = row.get("user_id")
        raw_item_id = row.get("item_id")

        user_id = "" if raw_user_id is None else str(raw_user_id).strip()
        item_id = "" if raw_item_id is None else str(raw_item_id).strip()

        if not user_id or not item_id:
            raise ValueError("Fit rows require non-empty user_id and item_id")

        if item_id not in catalog_ids:
            raise ValueError(f"Fit item is outside catalog: {item_id}")

        rating = float(row["rating"])
        if not np.isfinite(rating) or not 1.0 <= rating <= 5.0:
            raise ValueError(f"Invalid rating for {(user_id, item_id)!r}: {rating}")

        timestamp = row.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError(f"Invalid timestamp for {(user_id, item_id)!r}: {timestamp}")

        pair = (user_id, item_id)
        if pair in pairs:
            raise ValueError(f"Duplicate user-item pair in fit data: {pair!r}")

        pairs.add(pair)


def _config_to_dict(config: ItemKNNConfig) -> dict[str, Any]:
    return {
        "implementation": "implicit.nearest_neighbours.CosineRecommender",
        "neighbors": config.neighbors,
        "num_threads": config.num_threads,
        "positive_rating_threshold": 4.0,
    }


def _write_mapping(path: Path, forward: dict[str, int]) -> None:
    inverse = [
        item_id for item_id, _ in sorted(forward.items(), key=lambda pair: pair[1])
    ]
    _write_json(path, {"id_to_index": forward, "index_to_id": inverse})


def _mapping_forward(value: dict[str, Any]) -> dict[str, int]:
    return {item_id: int(index) for item_id, index in value["id_to_index"].items()}


def _verify_artifacts(bundle_dir: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        actual = source_sha256(bundle_dir / name)

        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
