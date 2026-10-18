"""Non-LLM re-rankers of the fused pool (the top 20), to compare with the LLM re-ranker.

Learning-to-rank (LTRReranker)
    A gradient-boosted tree model (scikit-learn HistGradientBoostingClassifier) predicts for every
    (user, pooled item) pair whether the item is the user's next film. Features (FEATURES) are things
    the pipeline already computes: fusion rank and score, each member's own score (divided by its best
    score for the user) and rank, popularity, MiniLM similarity to the last liked item and to the mean
    of liked items, history length. Trained on validation users, never on the users it is tested on.

Cross-encoder (CrossEncoderReranker)
    A pre-trained re-ranking model (BAAI/bge-reranker-base) reads the pair
    ("A viewer who liked: <titles>", <candidate card>) and scores how well they match.
"""

from __future__ import annotations

import pickle

from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.retrieval.content import liked_history


MEMBERS = ["bm25", "itemknn"]
FEATURES = [
    "fusion_rank", "fusion_score",
    "bm25_score", "bm25_rank", "itemknn_score", "itemknn_rank",
    "popularity_prior", "log_popularity", "dense_last", "dense_mean", "history_length",
]


def pool_features(fusion: Any, user_ids: list[str], vectors: np.ndarray, pool: int = 20) -> list[tuple[list[str], np.ndarray]]:
    """(pooled item ids in fusion order, feature matrix) per user."""
    data, depth = fusion.data, fusion.config["depth"]
    results = []

    for user_id, (ranked, lists) in zip(user_ids, fusion.recommend_batch_with_lists(user_ids, pool)):
        # Each member's own items: normalised score and rank (missing: score 0, rank depth + 1).
        own = {}
        for name in MEMBERS:
            items = [item for item in lists.get(name, []) if item["source"] == name]
            best = max((item["score"] for item in items), default=0.0) or 1.0
            own[name] = {item["item_id"]: (item["score"] / best, rank) for rank, item in enumerate(items, 1)}

        liked, weights = liked_history(data, user_id, 0.7, 20)
        last = vectors[liked[-1]].astype(np.float32) if len(liked) else None
        mean = (vectors[liked].astype(np.float32) * weights[:, None]).sum(axis=0) if len(liked) else None
        mean = mean / max(float(np.linalg.norm(mean)), 1e-9) if mean is not None else None
        history_length = len(data.history(user_id)[0])

        rows = []
        for rank, item in enumerate(ranked, 1):
            index = data.item_index[item["item_id"]]
            vector = vectors[index].astype(np.float32)
            member_values = [value for name in MEMBERS for value in own[name].get(item["item_id"], (0.0, depth + 1))]

            rows.append([
                rank, item["score"], *member_values,
                float(fusion.prior[index]), float(np.log1p(fusion.popularity.counts[index])),
                float(vector @ last) if last is not None else 0.0,
                float(vector @ mean) if mean is not None else 0.0,
                history_length,
            ])

        results.append(([item["item_id"] for item in ranked], np.array(rows, dtype=np.float64).reshape(-1, len(FEATURES))))

    return results


class LTRReranker:
    def __init__(self) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LTRReranker:
        self.model.fit(X, y)
        return self

    def rerank(self, items: list[str], X: np.ndarray, k: int = 10) -> list[str]:
        """Items by predicted probability of being the next film; ties keep the fusion order."""
        scores = self.model.predict_proba(X)[:, 1]
        order = sorted(range(len(items)), key=lambda i: (-scores[i], i))

        return [items[i] for i in order[:k]]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self.model))

    @classmethod
    def load(cls, path: Path) -> LTRReranker:
        reranker = cls()
        reranker.model = pickle.loads(path.read_bytes())

        return reranker


def user_query(package: dict[str, Any], max_items: int = 5) -> str:
    """'A viewer who liked: <title and details of the newest liked items>' (the user's own reviews are left out)."""
    liked = [entry["text"] for entry in package["history"].values() if entry["sentiment"] == "positive"][:max_items]

    return "A viewer who liked: " + "; ".join(" ".join(text.split()[:25]) for text in liked)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", predict: Callable[[list[tuple[str, str]]], np.ndarray] | None = None) -> None:
        self.model_name = model_name

        if predict is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, device="cpu", max_length=384)
            predict = lambda pairs: np.asarray(model.predict(pairs, batch_size=20, show_progress_bar=False))

        self.predict = predict

    def rerank(self, package: dict[str, Any], pool: list[str], k: int = 10) -> list[str]:
        from backend.generation.rerank import candidate_card

        query = user_query(package)
        scores = self.predict([(query, candidate_card(package, item_id, with_evidence=True)) for item_id in pool])
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))

        return [pool[i] for i in order[:k]]
