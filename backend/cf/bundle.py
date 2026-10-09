from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from backend.cf.bpr import BPRConfig, BPRModel, config_to_dict
from backend.data.io import read_parquet, source_sha256, write_parquet


Phase = Literal["validation", "test"]


@dataclass
class CFBundle:
    phase: Phase
    model: BPRModel
    user_to_index: dict[str, int]

    item_to_index: dict[str, int]
    seen_by_user: dict[str, set[str]]
    positive_by_user: dict[str, set[str]]

    positive_users: set[str]
    learnable_items: set[str]
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
            items = self._popular_items(excluded, limit)

            return {"status": "cold_start", "items": items}

        user_index = self.user_to_index[user_id]
        scores = self.model.score_user(user_index)

        candidate_ids = sorted(self.learnable_items - excluded)
        candidate_indices = np.fromiter(
            (self.item_to_index[item_id] for item_id in candidate_ids),
            dtype=np.int64,
        )
        candidate_scores = scores[candidate_indices]

        if limit and len(candidate_ids) > limit:
            selected = np.argpartition(candidate_scores, -limit)[-limit:]
            threshold = float(candidate_scores[selected].min())
            selected = np.flatnonzero(candidate_scores >= threshold)

            candidates = [
                (candidate_ids[int(index)], float(candidate_scores[int(index)]))
                for index in selected
            ]
        else:
            candidates = list(zip(candidate_ids, map(float, candidate_scores)))

        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        items = [
            {"item_id": item_id, "cf_score": score, "source": "bpr"}
            for item_id, score in candidates[:limit]
        ]

        return {"status": "ok", "items": items}

    def _popular_items(
        self,
        excluded: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        return [
            {"item_id": item_id, "cf_score": float(count), "source": "popularity"}
            for item_id, count in self.popularity
            if item_id not in excluded
        ][:limit]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "bpr_model.npz"
        np.savez_compressed(
            model_path,
            user_factors=self.model.user_factors,
            item_factors=self.model.item_factors,
            item_bias=self.model.item_bias,
        )

        _write_mapping(output_dir / "user_mapping.json", self.user_to_index)
        _write_mapping(output_dir / "item_mapping.json", self.item_to_index)
        _write_json(output_dir / "bpr_config.json", config_to_dict(self.model.config))

        history = {
            "learnable_items": sorted(self.learnable_items),
            "positive_by_user": {
                user_id: sorted(items)
                for user_id, items in sorted(self.positive_by_user.items())
            },
            "positive_users": sorted(self.positive_users),
            "seen_by_user": {
                user_id: sorted(items)
                for user_id, items in sorted(self.seen_by_user.items())
            },
        }
        _write_json(output_dir / "fit_history.json", history)

        popularity_rows = [
            {"item_id": item_id, "positive_count": count, "rank": rank}
            for rank, (item_id, count) in enumerate(self.popularity, start=1)
        ]
        write_parquet(output_dir / "popularity.parquet", popularity_rows)

        artifact_names = (
            "bpr_model.npz",
            "user_mapping.json",
            "item_mapping.json",
            "bpr_config.json",
            "training_metrics.csv",
            "popularity.parquet",
            "fit_history.json",
        )
        self.manifest["artifact_hashes"] = {
            name: source_sha256(output_dir / name)
            for name in artifact_names
        }
        self.manifest["artifact_sizes"] = {
            name: (output_dir / name).stat().st_size
            for name in artifact_names
        }
        self.manifest["config_sha256"] = self.manifest["artifact_hashes"][
            "bpr_config.json"
        ]

        _write_json(output_dir / "manifest.json", self.manifest)

    @classmethod
    def load(cls, bundle_dir: Path) -> CFBundle:
        manifest = _read_json(bundle_dir / "manifest.json")
        _verify_artifacts(bundle_dir, manifest["artifact_hashes"])

        config_values = _read_json(bundle_dir / "bpr_config.json")
        config = BPRConfig(
            factors=int(config_values["factors"]),
            learning_rate=float(config_values["learning_rate"]),
            regularization=float(config_values["regularization"]),
            epochs=int(config_values["epochs"]),
            seed=int(config_values["seed"]),
            num_threads=int(config_values.get("num_threads", 1)),
            verify_negative_samples=bool(
                config_values.get("verify_negative_samples", True)
            ),
        )
        model = BPRModel(config)

        with np.load(bundle_dir / "bpr_model.npz") as arrays:
            model.user_factors = arrays["user_factors"]
            model.item_factors = arrays["item_factors"]
            model.item_bias = arrays["item_bias"]

        user_mapping = _read_json(bundle_dir / "user_mapping.json")
        item_mapping = _read_json(bundle_dir / "item_mapping.json")
        history = _read_json(bundle_dir / "fit_history.json")

        popularity_rows = read_parquet(bundle_dir / "popularity.parquet")
        popularity = [
            (
                str(row["item_id"]),
                int(row.get("positive_count", row.get("interaction_count", 0))),
            )
            for row in sorted(popularity_rows, key=lambda row: int(row["rank"]))
        ]

        return cls(
            phase=manifest["phase"],
            model=model,
            user_to_index=_mapping_forward(user_mapping),
            item_to_index=_mapping_forward(item_mapping),
            seen_by_user={
                user_id: set(items)
                for user_id, items in history["seen_by_user"].items()
            },
            positive_by_user={
                user_id: set(items)
                for user_id, items in history["positive_by_user"].items()
            },
            positive_users=set(history["positive_users"]),
            learnable_items=set(history["learnable_items"]),
            popularity=popularity,
            manifest=manifest,
        )


def train_bundle(
    splits_path: Path,
    catalog_path: Path,
    output_dir: Path,
    *,
    phase: Phase,
    config: BPRConfig | None = None,
) -> CFBundle:
    if phase not in ("validation", "test"):
        raise ValueError(f"Unsupported training phase: {phase}")

    effective_config = config or BPRConfig()
    catalog_rows = read_parquet(catalog_path)

    catalog_ids: list[str] = []
    catalog_seen: set[str] = set()

    for row in catalog_rows:
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
    positive_by_user: dict[str, set[str]] = defaultdict(set)
    positive_pairs: list[tuple[int, int]] = []
    positive_users: set[str] = set()

    learnable_items: set[str] = set()

    popularity_counts: Counter[str] = Counter({item_id: 0 for item_id in catalog_ids})
    ordered_rows = sorted(
        fit_rows,
        key=lambda row: (str(row["user_id"]), str(row["item_id"])),
    )
    for row in ordered_rows:
        user_id = str(row["user_id"])
        item_id = str(row["item_id"])

        seen_by_user[user_id].add(item_id)

        if float(row["rating"]) < 4.0:
            continue

        popularity_counts[item_id] += 1
        positive_users.add(user_id)

        positive_by_user[user_id].add(item_id)
        learnable_items.add(item_id)
        positive_pairs.append((user_to_index[user_id], item_to_index[item_id]))

    model = BPRModel(effective_config)
    training_metrics = model.fit(
        len(user_to_index),
        len(item_to_index),
        positive_pairs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics(output_dir / "training_metrics.csv", training_metrics)

    popularity = sorted(popularity_counts.items(), key=lambda pair: (-pair[1], pair[0]))
    manifest = {
        "schema_version": "cf-bpr-bundle-v2",
        "phase": phase,
        "fit_splits": fit_splits,
        "input_hashes": {
            "catalog": source_sha256(catalog_path),
            "splits": source_sha256(splits_path),
        },
        "counts": {
            "catalog_items": len(catalog_ids),
            "fit_interactions": len(fit_rows),
            "positive_pairs": len(positive_pairs),
            "users": len(user_ids),
        },
        "artifact_hashes": {},
    }
    code_digest = hashlib.sha256()
    for name in ("bpr.py", "bundle.py", "baselines.py", "evaluation.py"):
        code_digest.update((Path(__file__).parent / name).read_bytes())

    manifest["code_sha256"] = code_digest.hexdigest()
    bundle = CFBundle(
        phase=phase,
        model=model,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        seen_by_user=dict(seen_by_user),
        positive_by_user=dict(positive_by_user),
        positive_users=positive_users,
        learnable_items=learnable_items,
        popularity=popularity,
        manifest=manifest,
    )
    bundle.save(output_dir)

    return bundle


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


def _write_metrics(path: Path, metrics: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "epoch",
        "train_auc",
        "correct",
        "skipped",
        "duration_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def _write_mapping(path: Path, forward: dict[str, int]) -> None:
    inverse = [item_id for item_id, _ in sorted(forward.items(), key=lambda pair: pair[1])]
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
