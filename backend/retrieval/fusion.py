"""Fusion: merge the ranked lists of several retrievers into one list.

Weighted reciprocal rank fusion (RRF). Each retriever ("member") returns its top `depth`
items; an item at rank r in member m earns  weight[m] / (k + r). The scores are summed and
the items sorted. Ranks are used instead of raw scores because every member scores on a
different scale (cosine, BM25, popularity counts...).

Only items a member found itself count for that member: when ItemKNN pads its list with
popular items, those padded items do not earn ItemKNN's weight. The "popularity" member
is the exception — popular items are exactly what it returns.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from backend.cf.common import Popularity, item_indices, rank_items, seen_items
from backend.cf.evaluate import evaluate
from backend.data.io import read_json, write_json


def fuse(lists: dict[str, list[dict[str, Any]]], weights: dict[str, float], k: float) -> list[dict[str, Any]]:
    """Fused candidates, best first (ties by item id): {"item_id", "score", "members"}."""
    fused: dict[str, dict[str, Any]] = {}

    for member, items in lists.items():
        own = [item for item in items if member == "popularity" or item["source"] == member]

        for rank, item in enumerate(own, start=1):
            entry = fused.setdefault(item["item_id"], {"item_id": item["item_id"], "score": 0.0, "members": []})
            entry["score"] += weights.get(member, 0.0) / (k + rank)
            entry["members"].append(member)

    return sorted(fused.values(), key=lambda entry: (-entry["score"], entry["item_id"]))


class Fusion:
    name = "fusion"

    def __init__(self, members: dict[str, Any], weights: dict[str, float], k: float = 60.0, depth: int = 50) -> None:
        """`members` maps a name to a model; a "popularity" member is always added."""
        self.members = dict(members)
        self.data = next(iter(self.members.values())).data
        self.popularity = Popularity(self.data)

        self.members.setdefault("popularity", self.popularity)
        self.config = {"model": self.name, "weights": weights, "k": k, "depth": depth}

    def recommend(self, user_id: str, limit: int = 10, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        excluded = seen_items(self.data, user_id) | item_indices(self.data, exclude)
        lists = {name: model.recommend_batch([user_id], self.config["depth"])[0] for name, model in self.members.items()}

        return self._rank(lists, excluded, limit)

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        lists = {name: model.recommend_batch(user_ids, self.config["depth"]) for name, model in self.members.items()}

        return [
            self._rank({name: member_lists[row] for name, member_lists in lists.items()}, seen_items(self.data, user_id), limit)
            for row, user_id in enumerate(user_ids)
        ]

    def save(self, model_dir: Path, member_dirs: dict[str, str]) -> None:
        """Fusion has nothing to train: it only remembers which member models and weights to use."""
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "config.json", {**self.config, "member_dirs": member_dirs})

    @classmethod
    def load(cls, model_dir: Path) -> Fusion:
        from backend.cf.cli import load_model

        config = read_json(model_dir / "config.json")
        members = {name: load_model(Path(path)) for name, path in config["member_dirs"].items()}

        return cls(members, config["weights"], config["k"], config["depth"])

    def _rank(self, lists: dict[str, list[dict[str, Any]]], excluded: set[int], limit: int) -> list[dict[str, Any]]:
        fused = [entry for entry in fuse(lists, self.config["weights"], self.config["k"]) if entry["score"] > 0][:limit]
        indices = [self.data.item_index[entry["item_id"]] for entry in fused]
        ranked = rank_items(self.popularity, indices, [entry["score"] for entry in fused], excluded, limit, self.name)

        members = {entry["item_id"]: entry["members"] for entry in fused}
        for item in ranked:
            item["members"] = members.get(item["item_id"], ["popularity"])

        return ranked


class CachedLists:
    """Replays lists computed once, so many weight settings can be evaluated quickly."""

    def __init__(self, model: Any, user_ids: list[str], depth: int) -> None:
        self.data = model.data
        self.lists = dict(zip(user_ids, model.recommend_batch(user_ids, depth)))

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        return [self.lists[user_id] for user_id in user_ids]


def tune_weights(
    members: dict[str, Any],
    queries_path: Path,
    grid: dict[str, list[float]],
    *,
    query_limit: int = 20_000,
    k: float = 60.0,
    depth: int = 50,
) -> list[dict[str, Any]]:
    """Evaluate every weight combination of `grid` on a fixed query sample (use validation!)."""
    import pyarrow.parquet as pq
    from random import Random

    user_ids = pq.read_table(queries_path, columns=["user_id"])["user_id"].to_pylist()
    picked = sorted(Random(42).sample(range(len(user_ids)), min(query_limit, len(user_ids))))
    sample = [user_ids[index] for index in picked]

    cached = {name: CachedLists(model, sample, depth) for name, model in members.items()}
    names = list(grid)
    results = []

    for values in product(*(grid[name] for name in names)):
        weights = dict(zip(names, values))
        report = evaluate(Fusion(cached, weights, k, depth), queries_path, query_limit=query_limit)

        results.append({**weights, "ndcg_at_10": report["ndcg_at_10"], "recall_at_50": report["recall_at_50"]})

    return sorted(results, key=lambda row: (-row["ndcg_at_10"], -row["recall_at_50"]))
