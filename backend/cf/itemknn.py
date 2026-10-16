"""Item-based KNN with cosine similarity and recency-weighted user history.

Score of item j for user u, with history h_1..h_n ordered oldest to newest:

    score(u, j) = sum_k decay^(n - k) * sim(h_k, j)

`sim` is the cosine similarity between item columns of the binary user-item
matrix, pruned to the top `neighbors` items per row. Items without a positive
score are filled from the positive-interaction popularity ranking.
"""

from __future__ import annotations

import hashlib
import math

from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as parquet

from scipy.sparse import csr_matrix, diags, load_npz, save_npz

from backend.cf.bundle import _read_json, _verify_artifacts, _write_json
from backend.data.io import source_sha256, write_parquet


Phase = Literal["validation", "test"]

ITEMKNN_SCHEMA_VERSION = "cf-itemknn-bundle-v1"
_MAX_LIMIT = 50
_POSITIVE_RATING = 4.0

_ARTIFACT_NAMES = (
    "itemknn_config.json",
    "similarity.npz",
    "history.npz",
    "users.parquet",
    "items.parquet",
    "popularity.parquet",
)
_SPLIT_COLUMNS = ("user_id", "item_id", "rating", "timestamp", "split")


@dataclass(frozen=True)
class ItemKNNConfig:
    neighbors: int = 100
    recency_decay: float = 0.5
    min_rating: float = 1.0

    def __post_init__(self) -> None:
        if self.neighbors <= 0:
            raise ValueError("neighbors must be positive")

        if not math.isfinite(self.recency_decay) or not 0.0 < self.recency_decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")

        if not math.isfinite(self.min_rating) or not 1.0 <= self.min_rating <= 5.0:
            raise ValueError("min_rating must be between 1 and 5")


def recency_weights(length: int, decay: float) -> np.ndarray:
    """Weights for a history ordered oldest to newest; the newest item gets 1."""
    exponents = np.arange(length - 1, -1, -1, dtype=np.float64)

    return np.power(decay, exponents).astype(np.float32)


def build_similarity(
    interactions: csr_matrix,
    neighbors: int,
    *,
    batch_size: int = 1024,
) -> csr_matrix:
    """Top-k cosine similarity between item columns, without self links."""
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    binary = csr_matrix(interactions, dtype=np.float32, copy=True)
    binary.eliminate_zeros()
    binary.data[:] = 1.0

    item_count = binary.shape[1]
    degrees = np.asarray(binary.sum(axis=0)).ravel()
    scale = np.zeros(item_count, dtype=np.float32)

    nonzero = degrees > 0
    scale[nonzero] = 1.0 / np.sqrt(degrees[nonzero])

    normalized = (binary @ diags(scale)).tocsr()
    transposed = normalized.T.tocsr()

    row_counts = np.zeros(item_count, dtype=np.int64)
    kept_indices: list[np.ndarray] = []
    kept_values: list[np.ndarray] = []

    for start in range(0, item_count, batch_size):
        block = (transposed[start : start + batch_size] @ normalized).tocsr()

        for offset in range(block.shape[0]):
            item = start + offset
            low, high = block.indptr[offset], block.indptr[offset + 1]

            indices = block.indices[low:high]
            values = block.data[low:high]
            mask = (indices != item) & (values > 0)

            indices = indices[mask]
            values = values[mask]

            if len(values) > neighbors:
                order = np.lexsort((indices, -values))[:neighbors]
                order.sort()

                indices = indices[order]
                values = values[order]

            row_counts[item] = len(indices)
            kept_indices.append(indices.astype(np.int32))
            kept_values.append(values.astype(np.float32))

    indptr = np.concatenate(([0], np.cumsum(row_counts)))
    similarity = csr_matrix(
        (
            np.concatenate(kept_values) if kept_values else np.zeros(0, np.float32),
            np.concatenate(kept_indices) if kept_indices else np.zeros(0, np.int32),
            indptr,
        ),
        shape=(item_count, item_count),
        dtype=np.float32,
    )
    similarity.sort_indices()

    return similarity


@dataclass
class ItemKNNBundle:
    phase: Phase
    config: ItemKNNConfig
    user_ids: list[str]

    item_ids: list[str]
    history_indptr: np.ndarray
    history_items: np.ndarray

    history_ratings: np.ndarray
    similarity: csr_matrix
    popularity: list[tuple[str, int]]

    manifest: dict[str, Any]

    user_to_index: dict[str, int] = field(init=False, repr=False)
    item_to_index: dict[str, int] = field(init=False, repr=False)
    _popularity_order: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.user_to_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.item_to_index = {item_id: index for index, item_id in enumerate(self.item_ids)}
        self._popularity_order = np.fromiter(
            (self.item_to_index[item_id] for item_id, _ in self.popularity),
            dtype=np.int64,
            count=len(self.popularity),
        )

    def recommend(
        self,
        user_id: str,
        *,
        seen_ids: set[str] | None = None,
        limit: int = 50,
        explain: int = 3,
    ) -> dict[str, Any]:
        _check_limit(limit)

        user_index = self.user_to_index.get(user_id)
        seen, contributing, weights = self._history(user_index)

        extra = [self.item_to_index[item_id] for item_id in seen_ids or () if item_id in self.item_to_index]
        excluded = np.union1d(seen, np.asarray(extra, dtype=np.int64))

        scores = self._score_rows([contributing], [weights])
        knn_items, knn_scores, fill_items = self._select(scores, 0, excluded, limit)

        because = self._because(contributing, weights, knn_items, explain)
        items = [
            {
                "item_id": self.item_ids[item],
                "cf_score": float(score),
                "source": "itemknn",
                "because": reasons,
            }
            for item, score, reasons in zip(knn_items, knn_scores, because)
        ]

        counts = dict(self.popularity)
        items.extend(
            {
                "item_id": self.item_ids[item],
                "cf_score": float(counts[self.item_ids[item]]),
                "source": "popularity",
                "because": [],
            }
            for item in fill_items
        )
        status = "ok" if len(contributing) else "cold_start"

        return {"status": status, "items": items}

    def rank_users(
        self,
        user_ids: list[str],
        *,
        limit: int = 50,
    ) -> list[tuple[list[str], int]]:
        """Rank many users at once; returns (ranked item ids, ItemKNN-scored count)."""
        _check_limit(limit)

        histories = [self._history(self.user_to_index.get(user_id)) for user_id in user_ids]
        scores = self._score_rows(
            [contributing for _, contributing, _ in histories],
            [weights for _, _, weights in histories],
        )
        rankings: list[tuple[list[str], int]] = []

        for row, (seen, _, _) in enumerate(histories):
            knn_items, _, fill_items = self._select(scores, row, seen, limit)
            ranked = [self.item_ids[item] for item in [*knn_items, *fill_items]]

            rankings.append((ranked, len(knn_items)))

        return rankings

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "itemknn_config.json", asdict(self.config))
        save_npz(output_dir / "similarity.npz", self.similarity, compressed=True)

        np.savez_compressed(
            output_dir / "history.npz",
            indptr=self.history_indptr,
            items=self.history_items,
            ratings=self.history_ratings,
        )
        _write_ids(output_dir / "users.parquet", "user_id", self.user_ids)
        _write_ids(output_dir / "items.parquet", "item_id", self.item_ids)

        popularity_rows = [
            {"item_id": item_id, "positive_count": count, "rank": rank}
            for rank, (item_id, count) in enumerate(self.popularity, start=1)
        ]
        write_parquet(
            output_dir / "popularity.parquet",
            popularity_rows,
            columns=("item_id", "positive_count", "rank"),
        )

        self.manifest["artifact_hashes"] = {
            name: source_sha256(output_dir / name) for name in _ARTIFACT_NAMES
        }
        self.manifest["artifact_sizes"] = {
            name: (output_dir / name).stat().st_size for name in _ARTIFACT_NAMES
        }
        self.manifest["config_sha256"] = self.manifest["artifact_hashes"]["itemknn_config.json"]

        _write_json(output_dir / "manifest.json", self.manifest)

    @classmethod
    def load(cls, bundle_dir: Path) -> ItemKNNBundle:
        manifest = _read_json(bundle_dir / "manifest.json")
        if manifest.get("schema_version") != ITEMKNN_SCHEMA_VERSION:
            raise ValueError(f"Not an ItemKNN bundle: {bundle_dir}")

        _verify_artifacts(bundle_dir, manifest["artifact_hashes"])
        config = ItemKNNConfig(**_read_json(bundle_dir / "itemknn_config.json"))

        with np.load(bundle_dir / "history.npz") as arrays:
            history_indptr = arrays["indptr"]
            history_items = arrays["items"]
            history_ratings = arrays["ratings"]

        popularity_table = parquet.read_table(bundle_dir / "popularity.parquet")
        popularity_rows = sorted(popularity_table.to_pylist(), key=lambda row: int(row["rank"]))

        return cls(
            phase=manifest["phase"],
            config=config,
            user_ids=_read_ids(bundle_dir / "users.parquet", "user_id"),
            item_ids=_read_ids(bundle_dir / "items.parquet", "item_id"),
            history_indptr=history_indptr,
            history_items=history_items,
            history_ratings=history_ratings,
            similarity=load_npz(bundle_dir / "similarity.npz").tocsr(),
            popularity=[(str(row["item_id"]), int(row["positive_count"])) for row in popularity_rows],
            manifest=manifest,
        )

    def _history(self, user_index: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (all seen items, contributing items, recency weights)."""
        if user_index is None:
            empty = np.zeros(0, dtype=np.int64)

            return empty, empty, np.zeros(0, dtype=np.float32)

        start, end = self.history_indptr[user_index], self.history_indptr[user_index + 1]
        items = self.history_items[start:end].astype(np.int64)
        ratings = self.history_ratings[start:end]

        contributing = items[ratings >= self.config.min_rating]
        weights = recency_weights(len(contributing), self.config.recency_decay)

        return np.sort(items), contributing, weights

    def _score_rows(
        self,
        contributing: list[np.ndarray],
        weights: list[np.ndarray],
    ) -> csr_matrix:
        lengths = np.fromiter((len(items) for items in contributing), dtype=np.int64)
        rows = np.repeat(np.arange(len(contributing)), lengths)
        columns = np.concatenate(contributing) if contributing else np.zeros(0, np.int64)

        values = np.concatenate(weights) if weights else np.zeros(0, np.float32)
        history = csr_matrix(
            (values, (rows, columns)),
            shape=(len(contributing), len(self.item_ids)),
            dtype=np.float32,
        )

        return (history @ self.similarity).tocsr()

    def _select(
        self,
        scores: csr_matrix,
        row: int,
        excluded: np.ndarray,
        limit: int,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        low, high = scores.indptr[row], scores.indptr[row + 1]
        indices = scores.indices[low:high]
        values = scores.data[low:high]

        keep = (values > 0) & ~np.isin(indices, excluded)
        indices = indices[keep]
        values = values[keep]

        order = np.lexsort((indices, -values))[:limit]
        knn_items = indices[order].astype(np.int64)
        knn_scores = values[order]

        blocked = set(excluded.tolist()) | set(knn_items.tolist())
        fill_items: list[int] = []

        for item in self._popularity_order:
            if len(knn_items) + len(fill_items) >= limit:
                break

            if int(item) not in blocked:
                fill_items.append(int(item))

        return knn_items, knn_scores, fill_items

    def _because(
        self,
        contributing: np.ndarray,
        weights: np.ndarray,
        items: np.ndarray,
        explain: int,
    ) -> list[list[dict[str, Any]]]:
        if explain <= 0 or not len(contributing) or not len(items):
            return [[] for _ in items]

        similarities = self.similarity[contributing][:, items].toarray()
        contributions = similarities * weights[:, None]
        reasons: list[list[dict[str, Any]]] = []

        for column in range(len(items)):
            order = np.lexsort((contributing, -contributions[:, column]))
            selected = [row for row in order[:explain] if contributions[row, column] > 0]

            reasons.append(
                [
                    {
                        "item_id": self.item_ids[int(contributing[row])],
                        "similarity": float(similarities[row, column]),
                        "weight": float(weights[row]),
                    }
                    for row in selected
                ]
            )

        return reasons


def train_itemknn_bundle(
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
    item_ids = _load_catalog_ids(catalog_path)
    fit_splits = ["train"] if phase == "validation" else ["train", "validation"]

    user_ids, users, items, ratings, timestamps = _load_fit_rows(
        splits_path,
        item_ids,
        fit_splits,
    )
    order = np.lexsort((items, timestamps, users))
    user_counts = np.bincount(users, minlength=len(user_ids))

    history_indptr = np.concatenate(([0], np.cumsum(user_counts))).astype(np.int64)
    history_items = items[order].astype(np.int32)
    history_ratings = ratings[order].astype(np.float32)

    used = ratings >= effective_config.min_rating
    interactions = csr_matrix(
        (np.ones(int(used.sum()), dtype=np.float32), (users[used], items[used])),
        shape=(len(user_ids), len(item_ids)),
    )
    similarity = build_similarity(interactions, effective_config.neighbors)

    positive_counts = np.bincount(
        items[ratings >= _POSITIVE_RATING],
        minlength=len(item_ids),
    )
    popularity_order = np.lexsort((np.arange(len(item_ids)), -positive_counts))
    popularity = [(item_ids[index], int(positive_counts[index])) for index in popularity_order]

    manifest = {
        "schema_version": ITEMKNN_SCHEMA_VERSION,
        "phase": phase,
        "fit_splits": fit_splits,
        "input_hashes": {
            "catalog": source_sha256(catalog_path),
            "splits": source_sha256(splits_path),
        },
        "counts": {
            "catalog_items": len(item_ids),
            "fit_interactions": int(len(users)),
            "similarity_interactions": int(used.sum()),
            "similarity_links": int(similarity.nnz),
            "users": len(user_ids),
        },
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifact_hashes": {},
    }
    bundle = ItemKNNBundle(
        phase=phase,
        config=effective_config,
        user_ids=user_ids,
        item_ids=item_ids,
        history_indptr=history_indptr,
        history_items=history_items,
        history_ratings=history_ratings,
        similarity=similarity,
        popularity=popularity,
        manifest=manifest,
    )
    bundle.save(output_dir)

    return bundle


def evaluate_itemknn(
    bundle: ItemKNNBundle,
    queries_path: Path,
    *,
    limit: int = 50,
    batch_size: int = 256,
    query_limit: int | None = None,
    query_seed: int = 42,
) -> dict[str, Any]:
    if limit != 50:
        raise ValueError("Evaluation limit must be exactly 50")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive when provided")

    schema = parquet.read_schema(queries_path)
    columns = [name for name in ("user_id", "target_item_id", "target_rating") if name in schema.names]
    queries = parquet.read_table(queries_path, columns=columns).to_pylist()

    if query_limit is not None and query_limit < len(queries):
        indices = sorted(Random(query_seed).sample(range(len(queries)), query_limit))
        queries = [queries[index] for index in indices]

    rankings: list[list[str]] = []
    knn_counts: list[int] = []
    latencies: list[float] = []

    for start in range(0, len(queries), batch_size):
        batch = queries[start : start + batch_size]
        started_at = perf_counter()

        ranked = bundle.rank_users([str(query["user_id"]) for query in batch], limit=limit)
        latency_ms = (perf_counter() - started_at) * 1000.0 / max(1, len(batch))

        rankings.extend(item_ids for item_ids, _ in ranked)
        knn_counts.extend(count for _, count in ranked)
        latencies.extend([latency_ms] * len(batch))

    catalog_count = len(bundle.item_ids)
    report = _ranking_metrics(queries, rankings, knn_counts, latencies, catalog_count)

    positive = [
        index
        for index, query in enumerate(queries)
        if float(query.get("target_rating") or 0.0) >= _POSITIVE_RATING
    ]
    report["positive_target_slice"] = _ranking_metrics(
        [queries[index] for index in positive],
        [rankings[index] for index in positive],
        [knn_counts[index] for index in positive],
        [latencies[index] for index in positive],
        catalog_count,
    )

    report["config"] = asdict(bundle.config)
    report["similarity_links"] = int(bundle.similarity.nnz)
    report["model_size_bytes"] = int(
        bundle.similarity.data.nbytes
        + bundle.similarity.indices.nbytes
        + bundle.similarity.indptr.nbytes
    )

    return report


def _ranking_metrics(
    queries: list[dict[str, Any]],
    rankings: list[list[str]],
    knn_counts: list[int],
    latencies: list[float],
    catalog_count: int,
) -> dict[str, Any]:
    hits_at_10 = 0
    hits_at_50 = 0
    discounts = 0.0

    recommended: set[str] = set()
    fill_shares: list[float] = []

    for query, ranking, knn_count in zip(queries, rankings, knn_counts):
        recommended.update(ranking)
        fill_shares.append((len(ranking) - knn_count) / max(1, len(ranking)))

        target = str(query["target_item_id"])
        if target not in ranking:
            continue

        rank = ranking.index(target) + 1
        hits_at_50 += 1

        if rank <= 10:
            hits_at_10 += 1
            discounts += 1.0 / math.log2(rank + 1)

    count = len(queries)
    served = sum(1 for knn_count in knn_counts if knn_count > 0)
    ordered = sorted(latencies)

    p50 = ordered[max(0, math.ceil(0.50 * len(ordered)) - 1)] if ordered else 0.0
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)] if ordered else 0.0

    return {
        "queries": count,
        "ndcg_at_10": discounts / max(1, count),
        "hr_at_10": hits_at_10 / max(1, count),
        "candidate_recall_at_50": hits_at_50 / max(1, count),
        "query_coverage": served / max(1, count),
        "fallback_rate": 1.0 - served / max(1, count),
        "popularity_fill_rate": sum(fill_shares) / max(1, count),
        "item_coverage": len(recommended) / max(1, catalog_count),
        "mean_latency_ms": sum(latencies) / max(1, len(latencies)),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
    }


def _check_limit(limit: int) -> None:
    if not 0 <= limit <= _MAX_LIMIT:
        raise ValueError(f"limit must be between 0 and {_MAX_LIMIT}")


def _load_catalog_ids(catalog_path: Path) -> list[str]:
    column = parquet.read_table(catalog_path, columns=["item_id"])["item_id"].to_pylist()
    item_ids: list[str] = []
    seen: set[str] = set()

    for raw_item_id in column:
        item_id = "" if raw_item_id is None else str(raw_item_id).strip()
        if not item_id:
            raise ValueError("Catalog requires non-empty item_id")

        if item_id in seen:
            raise ValueError(f"Duplicate catalog item: {item_id}")

        seen.add(item_id)
        item_ids.append(item_id)

    if not item_ids:
        raise ValueError("Catalog must contain at least one item")

    return sorted(item_ids)


def _load_fit_rows(
    splits_path: Path,
    item_ids: list[str],
    fit_splits: list[str],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and validate fit rows column-wise; returns ids and index arrays."""
    schema = parquet.read_schema(splits_path)
    missing = [name for name in _SPLIT_COLUMNS if name not in schema.names]

    if missing:
        raise ValueError(f"Splits file is missing columns: {missing}")

    split_filter = [("split", "in", fit_splits)]
    table = parquet.read_table(splits_path, columns=list(_SPLIT_COLUMNS), filters=split_filter)

    user_column = pc.cast(table["user_id"], pa.string())
    item_column = pc.cast(table["item_id"], pa.string())

    for column in (user_column, item_column):
        blank = pc.equal(pc.utf8_trim_whitespace(column), "")
        if column.null_count or pc.any(blank).as_py():
            raise ValueError("Fit rows require non-empty user_id and item_id")

    ratings = pc.cast(table["rating"], pa.float64()).to_numpy(zero_copy_only=False)
    invalid_rating = ~np.isfinite(ratings) | (ratings < 1.0) | (ratings > 5.0)

    if invalid_rating.any():
        raise ValueError(f"Invalid rating in fit data: {ratings[invalid_rating][0]}")

    timestamp_column = table["timestamp"]
    if not pa.types.is_integer(timestamp_column.type) or timestamp_column.null_count:
        raise ValueError("Invalid timestamp in fit data: expected non-null integers")

    item_index = pc.index_in(item_column, value_set=pa.array(item_ids, pa.string()))
    if item_index.null_count:
        outside = pc.filter(item_column, pc.is_null(item_index))[0].as_py()

        raise ValueError(f"Fit item is outside catalog: {outside}")

    user_ids = sorted(pc.unique(user_column).to_pylist())
    user_index = pc.index_in(user_column, value_set=pa.array(user_ids, pa.string()))

    users = user_index.to_numpy(zero_copy_only=False).astype(np.int64)
    items = item_index.to_numpy(zero_copy_only=False).astype(np.int64)
    keys = users * len(item_ids) + items

    unique_keys, counts = np.unique(keys, return_counts=True)
    if len(unique_keys) != len(keys):
        duplicate = int(unique_keys[np.argmax(counts > 1)])
        pair = (user_ids[duplicate // len(item_ids)], item_ids[duplicate % len(item_ids)])

        raise ValueError(f"Duplicate user-item pair in fit data: {pair!r}")

    timestamps = timestamp_column.to_numpy().astype(np.int64)

    return user_ids, users, items, ratings, timestamps


def _write_ids(path: Path, column: str, values: list[str]) -> None:
    table = pa.table({column: pa.array(values, pa.string())})

    parquet.write_table(table, path, compression="zstd")


def _read_ids(path: Path, column: str) -> list[str]:
    return [str(value) for value in parquet.read_table(path, columns=[column])[column].to_pylist()]
