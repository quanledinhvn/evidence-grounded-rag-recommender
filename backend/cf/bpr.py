"""BPR-MF baseline (Bayesian Personalized Ranking with matrix factorisation).

Each user and item gets a vector; score(u, j) = user_vector . item_vector + item_bias.
Training (done by the `implicit` library) pushes items the user rated >= 4 above random
other items. Users without any positive rating, and items nobody rated >= 4, have no
useful vector: those users get popularity, those items are never recommended by BPR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from implicit.cpu.bpr import BayesianPersonalizedRanking

from backend.cf.common import POSITIVE_RATING, FitData, Popularity, item_indices, rank_items, seen_items, top_k
from backend.data.io import read_json, write_json


class BPR:
    name = "bpr"

    def __init__(
        self,
        data: FitData,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        item_bias: np.ndarray,
        config: dict[str, Any],
    ) -> None:
        self.data = data
        self.user_factors = user_factors
        self.item_factors = item_factors

        self.item_bias = item_bias
        self.config = config
        self.popularity = Popularity(data)

        # Items with at least one positive interaction are the only ones BPR learned.
        self.learned_items = self.popularity.counts > 0

    @classmethod
    def fit(
        cls,
        data: FitData,
        factors: int = 64,
        learning_rate: float = 0.01,
        regularization: float = 0.001,
        epochs: int = 60,
        seed: int = 42,
        num_threads: int = 0,
    ) -> BPR:
        model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            iterations=epochs,
            random_state=seed,
            num_threads=num_threads,
        )
        model.fit(data.matrix(min_rating=POSITIVE_RATING), show_progress=False)

        # implicit stores the item bias as the last item-factor column
        # (the matching user column is a constant 1).
        config = {
            "model": cls.name,
            "phase": data.phase,
            "factors": factors,
            "learning_rate": learning_rate,
            "regularization": regularization,
            "epochs": epochs,
            "seed": seed,
        }

        return cls(
            data,
            np.asarray(model.user_factors[:, :-1], dtype=np.float32),
            np.asarray(model.item_factors[:, :-1], dtype=np.float32),
            np.asarray(model.item_factors[:, -1], dtype=np.float32),
            config,
        )

    def recommend(self, user_id: str, limit: int = 10, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        excluded = seen_items(self.data, user_id) | item_indices(self.data, exclude)

        return self._rank(user_id, excluded, limit)

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        # One matrix product for the whole batch is much faster than one per user.
        users = [self.data.user_index.get(user_id, 0) for user_id in user_ids]
        scores = self.user_factors[users] @ self.item_factors.T + self.item_bias

        return [
            self._rank(user_id, seen_items(self.data, user_id), limit, scores[row])
            for row, user_id in enumerate(user_ids)
        ]

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "config.json", self.config)
        self.data.save(model_dir / "fit_data.npz")

        np.savez_compressed(
            model_dir / "factors.npz",
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            item_bias=self.item_bias,
        )

    @classmethod
    def load(cls, model_dir: Path) -> BPR:
        data = FitData.load(model_dir / "fit_data.npz")

        with np.load(model_dir / "factors.npz") as arrays:
            user_factors, item_factors, item_bias = (
                arrays["user_factors"], arrays["item_factors"], arrays["item_bias"]
            )

        return cls(data, user_factors, item_factors, item_bias, read_json(model_dir / "config.json"))

    def _rank(
        self,
        user_id: str,
        excluded: set[int],
        limit: int,
        scores: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        _, ratings = self.data.history(user_id)

        if not (ratings >= POSITIVE_RATING).any():
            return rank_items(self.popularity, [], [], excluded, limit, self.name)

        if scores is None:
            user = self.data.user_index[user_id]
            scores = self.item_factors @ self.user_factors[user] + self.item_bias

        scores = scores.copy()
        scores[~self.learned_items] = -np.inf

        scores[list(excluded)] = -np.inf
        best, best_scores = top_k(np.arange(len(scores)), scores, limit)

        return rank_items(self.popularity, best, best_scores, excluded, limit, self.name)
