"""Embedding retrieval: recommend items whose item_doc *means* something similar to recent likes.

A sentence-embedding model turns every item_doc into a vector; similar stories, moods or
genres end up close together even without shared words. The user vector is the
recency-weighted average of the vectors of their recent liked items, and
score(j) = cosine(user vector, item vector).

Item vectors depend only on the metadata, so they are computed once and cached in a .npy file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.cf.common import FitData, Popularity, item_indices, rank_items, seen_items, top_k
from backend.data.io import read_json, write_json
from backend.retrieval.content import history_matrix, load_item_docs


# Small enough for CPU: ~50 documents/s on this machine (bge-small was ~2x slower).
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_documents(docs: list[str], model_name: str = DEFAULT_MODEL, max_length: int = 64) -> np.ndarray:
    """Unit-length vectors for the documents (empty documents get a zero vector).

    Only the first `max_length` tokens are read; item_doc puts title, genres and people first.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = max_length
    vectors = np.zeros((len(docs), model.get_sentence_embedding_dimension()), dtype=np.float32)

    present = [index for index, doc in enumerate(docs) if doc]
    encoded = model.encode([docs[i] for i in present], batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    vectors[present] = encoded

    return vectors


class Dense:
    name = "dense"

    def __init__(self, data: FitData, vectors: np.ndarray, config: dict[str, Any]) -> None:
        self.data = data
        self.vectors = vectors.astype(np.float32)
        self.config = config

        self.popularity = Popularity(data)
        self.has_doc = np.abs(self.vectors).sum(axis=1) > 0

    @classmethod
    def fit(
        cls,
        data: FitData,
        catalog_path: Path,
        vectors_path: Path,
        model_name: str = DEFAULT_MODEL,
        recency_decay: float = 0.7,
        max_history: int = 20,
    ) -> Dense:
        """`vectors_path` caches the item vectors; they are computed only if the file is missing."""
        if vectors_path.exists():
            vectors = np.load(vectors_path)
        else:
            vectors = embed_documents(load_item_docs(catalog_path, data), model_name)
            np.save(vectors_path, vectors)

        config = {
            "model": cls.name,
            "phase": data.phase,
            "embedding_model": model_name,
            "recency_decay": recency_decay,
            "max_history": max_history,
        }

        return cls(data, vectors, config)

    def recommend(self, user_id: str, limit: int = 10, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        excluded = seen_items(self.data, user_id) | item_indices(self.data, exclude)

        return self._rank(self._scores([user_id])[0], excluded, limit)

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        scores = self._scores(user_ids)

        return [self._rank(scores[row], seen_items(self.data, user_id), limit) for row, user_id in enumerate(user_ids)]

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "config.json", self.config)
        self.data.save(model_dir / "fit_data.npz")

        np.save(model_dir / "vectors.npy", self.vectors.astype(np.float16))

    @classmethod
    def load(cls, model_dir: Path) -> Dense:
        data = FitData.load(model_dir / "fit_data.npz")
        vectors = np.load(model_dir / "vectors.npy")

        return cls(data, vectors, read_json(model_dir / "config.json"))

    def _scores(self, user_ids: list[str]) -> np.ndarray:
        """users x items cosine scores; rows of users without liked items stay 0."""
        history = history_matrix(self.data, user_ids, self.config["recency_decay"], self.config["max_history"])
        users = history @ self.vectors
        norms = np.linalg.norm(users, axis=1, keepdims=True)

        users = np.divide(users, norms, out=np.zeros_like(users), where=norms > 0)

        return users @ self.vectors.T

    def _rank(self, scores: np.ndarray, excluded: set[int], limit: int) -> list[dict[str, Any]]:
        if not scores.any():
            return rank_items(self.popularity, [], [], excluded, limit, self.name)

        scores = scores.copy()
        scores[~self.has_doc] = -np.inf
        scores[list(excluded)] = -np.inf

        best, best_scores = top_k(np.arange(len(scores)), scores, limit)

        return rank_items(self.popularity, best, best_scores, excluded, limit, self.name)
