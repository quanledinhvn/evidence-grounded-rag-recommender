"""Fusion: merge the ranked lists of several retrievers into one list.

Each retriever ("member") returns its top `depth` items. Only items a member found itself
count for that member (when ItemKNN pads its list with popular items, those padded items do
not earn ItemKNN's weight). Two ways to combine them:

method="rrf" (reciprocal rank fusion)
    item at rank r in member m earns  weight[m] / (k + r). Uses ranks only.

method="score" (normalised score fusion, the default)
    item j earns  weight[m] * score_m(j) / best score of m for this user  in every member,
    plus  weight["popularity"] * (1 - popularity_rank(j) / catalog size).
    Keeps how *confident* each member is (a 0.34 ItemKNN similarity counts more than 0.01),
    and the popularity prior can lift popular items that sit deep in a member's list.

On validation RRF never beat ItemKNN alone; see CLAUDE.md for the numbers.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from backend.cf.common import Popularity, item_indices, rank_items, seen_items
from backend.cf.evaluate import evaluate
from backend.data.io import read_json, write_json


def own_items(member: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if member == "popularity" or item["source"] == member]


def fuse_ranks(lists: dict[str, list[dict[str, Any]]], weights: dict[str, float], k: float) -> dict[str, dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for member, items in lists.items():
        for rank, item in enumerate(own_items(member, items), start=1):
            entry = fused.setdefault(item["item_id"], {"item_id": item["item_id"], "score": 0.0, "members": []})
            entry["score"] += weights.get(member, 0.0) / (k + rank)
            entry["members"].append(member)

    return fused


def fuse_scores(
    lists: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    popularity_prior: dict[str, float],
) -> dict[str, dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for member, items in lists.items():
        own = own_items(member, items)
        best = max((item["score"] for item in own), default=0.0)

        for item in own:
            entry = fused.setdefault(item["item_id"], {"item_id": item["item_id"], "score": 0.0, "members": []})
            entry["members"].append(member)

            if member != "popularity" and best > 0:
                entry["score"] += weights.get(member, 0.0) * max(item["score"], 0.0) / best

    for entry in fused.values():
        entry["score"] += weights.get("popularity", 0.0) * popularity_prior[entry["item_id"]]

    return fused


class Fusion:
    name = "fusion"

    def __init__(
        self,
        members: dict[str, Any],
        weights: dict[str, float],
        k: float = 60.0,
        depth: int = 50,
        method: str = "score",
    ) -> None:
        """`members` maps a name to a model; a "popularity" member is always added."""
        self.members = dict(members)
        self.data = next(iter(self.members.values())).data
        self.popularity = Popularity(self.data)

        self.members.setdefault("popularity", self.popularity)
        self.config = {"model": self.name, "method": method, "weights": weights, "k": k, "depth": depth}

        # Popularity prior in [0, 1]: 1 for the most popular item, close to 0 for the least popular.
        rank = np.empty(len(self.popularity.order))
        rank[self.popularity.order] = np.arange(len(rank))
        self.prior = 1.0 - rank / len(rank)

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

        return cls(members, config["weights"], config["k"], config["depth"], config.get("method", "rrf"))

    def _rank(self, lists: dict[str, list[dict[str, Any]]], excluded: set[int], limit: int) -> list[dict[str, Any]]:
        weights = self.config["weights"]

        if self.config["method"] == "rrf":
            fused = fuse_ranks(lists, weights, self.config["k"])
        else:
            candidates = {item["item_id"] for items in lists.values() for item in items}
            prior = {item_id: float(self.prior[self.data.item_index[item_id]]) for item_id in candidates}
            fused = fuse_scores(lists, weights, prior)

        best = sorted((entry for entry in fused.values() if entry["score"] > 0), key=lambda e: (-e["score"], e["item_id"]))[:limit]
        indices = [self.data.item_index[entry["item_id"]] for entry in best]
        ranked = rank_items(self.popularity, indices, [entry["score"] for entry in best], excluded, limit, self.name)

        members = {entry["item_id"]: entry["members"] for entry in best}
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
    method: str = "score",
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
        report = evaluate(Fusion(cached, weights, k, depth, method), queries_path, query_limit=query_limit)

        results.append({**weights, "ndcg_at_10": report["ndcg_at_10"], "recall_at_50": report["recall_at_50"]})
        print(f"{weights} NDCG@10 {report['ndcg_at_10']:.4f} Recall@50 {report['recall_at_50']:.4f}", flush=True)

    return sorted(results, key=lambda row: (-row["ndcg_at_10"], -row["recall_at_50"]))
