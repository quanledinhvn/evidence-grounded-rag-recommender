"""BM25: recommend items whose item_doc shares rare words with what the user recently liked.

Every item_doc is a document. For a user, the "query" is the words of the item_docs of their
recent liked items (newest weighted most). Score of item j:

    score(j) = sum over words t of  query_weight(t) * bm25(t, j)
    bm25(t, j) = idf(t) * tf(t, j) * (k1 + 1) / (tf(t, j) + k1 * (1 - b + b * len(j) / avg_len))

A rare word shared with a liked item (a director's or an actor's name) gives a high score;
very common words (in more than `max_df` of the documents) are ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scipy.sparse import csr_matrix, diags, load_npz, save_npz
from sklearn.feature_extraction.text import CountVectorizer

from backend.cf.common import FitData, Popularity, item_indices, rank_items, seen_items, top_k
from backend.data.io import read_json, write_json
from backend.retrieval.content import history_matrix, load_item_docs


def inverse_document_frequency(counts: csr_matrix) -> np.ndarray:
    """idf(t) = log(1 + (N - df + 0.5) / (df + 0.5)): high for rare words, ~0 for common ones."""
    counts = counts.tocsr()
    doc_freq = np.bincount(counts.indices, minlength=counts.shape[1])

    return np.log(1.0 + (counts.shape[0] - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)


def keep_top_words(query: csr_matrix, idf: np.ndarray, max_words: int) -> csr_matrix:
    """In every user's query keep only the `max_words` most telling words (weight x idf).

    The liked items' documents contain hundreds of words; most are weak and each one
    matches thousands of items. Keeping the strongest ones (names, rare genres) is
    much faster and loses little.
    """
    query = query.tocsr()
    rows, columns, values = [], [], []

    for row in range(query.shape[0]):
        low, high = query.indptr[row], query.indptr[row + 1]
        words, weights = query.indices[low:high], query.data[low:high]
        best = np.argsort(-(weights * idf[words]), kind="stable")[:max_words]

        rows.append(np.full(len(best), row))
        columns.append(words[best])
        values.append(weights[best])

    return csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=query.shape)


def bm25_weights(counts: csr_matrix, k1: float, b: float) -> csr_matrix:
    """documents x words matrix of BM25 term weights, from raw word counts."""
    counts = counts.tocsr().astype(np.float32)
    idf = inverse_document_frequency(counts)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    length_norm = 1.0 - b + b * lengths / max(lengths.mean(), 1.0)

    # Saturate the term frequency of each (document, word) entry, then multiply by idf.
    weights = counts.copy()
    norm_per_entry = np.repeat(length_norm, np.diff(counts.indptr))
    weights.data = weights.data * (k1 + 1.0) / (weights.data + k1 * norm_per_entry)

    return (weights @ diags(idf)).tocsr()


class BM25:
    name = "bm25"

    def __init__(self, data: FitData, words: csr_matrix, weights: csr_matrix, config: dict[str, Any]) -> None:
        self.data = data
        self.words = words          # items x words, 1 if the word is in the item_doc (query side)
        self.weights = weights      # items x words, BM25 weight (document side)

        self.config = config
        self.popularity = Popularity(data)
        self.weights_by_word = weights.T.tocsr()

        self.idf = inverse_document_frequency(words)

    @classmethod
    def fit(
        cls,
        data: FitData,
        catalog_path: Path,
        k1: float = 1.2,
        b: float = 0.75,
        min_df: int = 2,
        max_df: float = 0.15,
        recency_decay: float = 0.7,
        max_history: int = 20,
        max_query_words: int = 50,
    ) -> BM25:
        docs = load_item_docs(catalog_path, data)
        vectorizer = CountVectorizer(stop_words="english", min_df=min_df, max_df=max_df)
        counts = vectorizer.fit_transform(docs)

        words = (counts > 0).astype(np.float32).tocsr()
        config = {
            "model": cls.name,
            "phase": data.phase,
            "k1": k1,
            "b": b,
            "max_df": max_df,
            "recency_decay": recency_decay,
            "max_history": max_history,
            "max_query_words": max_query_words,
            "vocabulary_size": counts.shape[1],
        }

        return cls(data, words, bm25_weights(counts, k1, b), config)

    def recommend(self, user_id: str, limit: int = 10, exclude: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        excluded = seen_items(self.data, user_id) | item_indices(self.data, exclude)

        return self._rank(self._scores([user_id]), 0, excluded, limit)

    def recommend_batch(self, user_ids: list[str], limit: int = 50) -> list[list[dict[str, Any]]]:
        scores = self._scores(user_ids)

        return [self._rank(scores, row, seen_items(self.data, user_id), limit) for row, user_id in enumerate(user_ids)]

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        write_json(model_dir / "config.json", self.config)
        self.data.save(model_dir / "fit_data.npz")

        save_npz(model_dir / "words.npz", self.words)
        save_npz(model_dir / "weights.npz", self.weights)

    @classmethod
    def load(cls, model_dir: Path) -> BM25:
        data = FitData.load(model_dir / "fit_data.npz")
        words = load_npz(model_dir / "words.npz").tocsr()
        weights = load_npz(model_dir / "weights.npz").tocsr()

        return cls(data, words, weights, read_json(model_dir / "config.json"))

    def _scores(self, user_ids: list[str]) -> csr_matrix:
        """users x items scores: (weighted words of liked items) @ BM25 weights."""
        history = history_matrix(self.data, user_ids, self.config["recency_decay"], self.config["max_history"])
        query = keep_top_words(history @ self.words, self.idf, self.config.get("max_query_words", 10**9))

        return (query @ self.weights_by_word).tocsr()

    def _rank(self, scores: csr_matrix, row: int, excluded: set[int], limit: int) -> list[dict[str, Any]]:
        low, high = scores.indptr[row], scores.indptr[row + 1]
        indices, values = scores.indices[low:high], scores.data[low:high]
        keep = ~np.isin(indices, list(excluded))

        best, best_scores = top_k(indices[keep], values[keep], limit)

        return rank_items(self.popularity, best, best_scores, excluded, limit, self.name)
