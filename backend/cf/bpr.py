from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
from implicit.cpu.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class BPRConfig:
    factors: int = 64
    learning_rate: float = 0.01
    regularization: float = 0.001

    epochs: int = 100
    seed: int = 42
    num_threads: int = 0

    verify_negative_samples: bool = True

    def __post_init__(self) -> None:
        if self.factors <= 0:
            raise ValueError("factors must be positive")

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        if not math.isfinite(self.regularization) or self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

        if self.num_threads < 0:
            raise ValueError("num_threads must be non-negative")


class BPRModel:
    def __init__(self, config: BPRConfig) -> None:
        self.config = config

        self.user_factors = np.empty((0, config.factors), dtype=np.float32)
        self.item_factors = np.empty((0, config.factors), dtype=np.float32)
        self.item_bias = np.empty(0, dtype=np.float32)

    def fit(
        self,
        user_count: int,
        item_count: int,
        positive_pairs: list[tuple[int, int]],
    ) -> list[dict[str, float | int]]:
        _validate_fit_inputs(user_count, item_count, positive_pairs)

        pairs = np.asarray(sorted(positive_pairs), dtype=np.int32)
        if len(pairs):
            rows = pairs[:, 0]
            columns = pairs[:, 1]
        else:
            rows = np.empty(0, dtype=np.int32)
            columns = np.empty(0, dtype=np.int32)

        values = np.ones(len(pairs), dtype=np.float32)
        user_items = csr_matrix(
            (values, (rows, columns)),
            shape=(user_count, item_count),
            dtype=np.float32,
        )

        backend = BayesianPersonalizedRanking(
            factors=self.config.factors,
            learning_rate=self.config.learning_rate,
            regularization=self.config.regularization,
            iterations=self.config.epochs,
            num_threads=self.config.num_threads,
            verify_negative_samples=self.config.verify_negative_samples,
            random_state=self.config.seed,
        )
        callback_rows: list[tuple[int, float, int, int]] = []
        backend.fit(
            user_items,
            show_progress=False,
            callback=lambda *row: callback_rows.append(row),
        )

        if backend.user_factors is None or backend.item_factors is None:
            raise RuntimeError("implicit BPR did not produce factor matrices")

        self.user_factors = np.asarray(
            backend.user_factors[:, :-1],
            dtype=np.float32,
        ).copy()
        self.item_factors = np.asarray(
            backend.item_factors[:, :-1],
            dtype=np.float32,
        ).copy()
        self.item_bias = np.asarray(
            backend.item_factors[:, -1],
            dtype=np.float32,
        ).copy()

        metrics: list[dict[str, float | int]] = []
        for epoch, duration, correct, skipped in callback_rows:
            attempted = len(pairs) - skipped
            train_auc = correct / attempted if attempted else 0.0

            metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_auc": train_auc,
                    "correct": correct,
                    "skipped": skipped,
                    "duration_seconds": duration,
                }
            )

        return metrics

    def score_user(self, user_index: int) -> np.ndarray:
        if user_index < 0 or user_index >= len(self.user_factors):
            raise IndexError(f"Unknown user index: {user_index}")

        return self.item_factors @ self.user_factors[user_index] + self.item_bias


def config_to_dict(config: BPRConfig) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "factors": config.factors,
        "implementation": "implicit.cpu.bpr.BayesianPersonalizedRanking",
        "implicit_version": version("implicit"),
        "item_bias": True,
        "learning_rate": config.learning_rate,
        "negative_sampling": "uniform_non_positive",
        "num_threads": config.num_threads,
        "optimizer": "implicit_bpr_sgd",
        "regularization": config.regularization,
        "seed": config.seed,
        "verify_negative_samples": config.verify_negative_samples,
    }


def _validate_fit_inputs(
    user_count: int,
    item_count: int,
    positive_pairs: list[tuple[int, int]],
) -> None:
    if user_count < 0 or item_count < 0:
        raise ValueError("User and item counts must be non-negative")

    for user_index, item_index in positive_pairs:
        if not 0 <= user_index < user_count:
            raise ValueError(f"User index outside mapping: {user_index}")

        if not 0 <= item_index < item_count:
            raise ValueError(f"Item index outside mapping: {item_index}")
