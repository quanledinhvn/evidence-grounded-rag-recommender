from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix

from backend.item_knn import ItemKNNBundle, ItemKNNConfig
from backend.profile import (
    PreferenceSignal,
    ProfileDiagnostics,
    ProfileTextField,
    UserProfile,
)
from backend.recommender import (
    BM25ContributionRetriever,
    ItemKNNContributionRetriever,
    OpenSearchBM25ContributionRetriever,
    OpenSearchSemanticContributionRetriever,
    SemanticContributionRetriever,
)


class RecommenderRetrieverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = _profile()

    def test_item_knn_recomputes_weighted_edges_from_active_sources(self) -> None:
        similarity = csr_matrix(
            np.array(
                [
                    [0.0, 0.0, 0.8, 0.2],
                    [0.0, 0.0, 0.1, 0.9],
                    [0.8, 0.1, 0.0, 0.0],
                    [0.2, 0.9, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
        )
        bundle = ItemKNNBundle(
            "validation",
            ItemKNNConfig(neighbors=2),
            similarity,
            csr_matrix((1, 4)),
            {"u1": 0},
            {"h1": 0, "h2": 1, "c1": 2, "c2": 3},
            {"u1": {"h1", "h2"}},
            {"u1"},
            [],
            {},
        )
        retriever = ItemKNNContributionRetriever(bundle, "artifact-v1")

        result = retriever.retrieve(
            self.profile,
            frozenset({"H1"}),
            limit=10,
        )

        self.assertEqual(result.pool, ("c1", "c2"))
        self.assertTrue(math.isclose(result.scores["c1"], 0.8, rel_tol=1e-6))
        self.assertEqual({atom.source_id for atom in result.atoms}, {"H1"})

    def test_bm25_allocates_shared_term_contribution_without_double_counting(self) -> None:
        retriever = BM25ContributionRetriever(
            {
                "h1": {"title": "space future"},
                "h2": {"title": "space drama"},
                "c1": {"title": "space future adventure"},
                "c2": {"title": "drama mystery"},
            },
            "artifact-v1",
        )

        result = retriever.retrieve(
            self.profile,
            frozenset({"H1", "H2"}),
            limit=10,
        )
        c1_atoms = [atom for atom in result.atoms if atom.candidate_id == "c1"]

        self.assertEqual(result.pool[0], "c1")
        self.assertTrue(math.isclose(sum(atom.raw_contribution for atom in c1_atoms), result.scores["c1"]))
        self.assertEqual({atom.source_id for atom in c1_atoms}, {"H1", "H2"})

    def test_semantic_scores_exact_positive_cosine_per_history_source(self) -> None:
        retriever = SemanticContributionRetriever(
            {
                "h1": (1.0, 0.0),
                "h2": (0.0, 1.0),
                "c1": (1.0, 0.0),
                "c2": (0.0, 1.0),
                "negative": (-1.0, 0.0),
            },
            {"c1": "space future", "c2": "drama mystery"},
            "artifact-v1",
        )

        result = retriever.retrieve(
            self.profile,
            frozenset({"H1"}),
            limit=10,
        )

        self.assertEqual(result.pool, ("c1",))
        self.assertTrue(math.isclose(result.scores["c1"], 1.0))
        self.assertEqual(result.atoms[0].candidate_span, "space future")

    def test_opensearch_bm25_keeps_weighted_source_contributions(self) -> None:
        gateway = _FakeRetrievalGateway()
        gateway.bm25_hits = {
            "space future": (
                {"item_id": "c1", "score": 2.0, "title": "Candidate One"},
                {"item_id": "h1", "score": 99.0, "title": "Seen"},
            ),
            "space drama": (
                {"item_id": "c1", "score": 1.0, "title": "Candidate One"},
                {"item_id": "c2", "score": 3.0, "title": "Candidate Two"},
            ),
        }
        retriever = OpenSearchBM25ContributionRetriever(gateway, "artifact-v1")

        result = retriever.retrieve(
            self.profile,
            frozenset({"H1", "H2"}),
            limit=10,
        )

        self.assertEqual(result.pool, ("c1", "c2"))
        self.assertTrue(math.isclose(result.scores["c1"], 2.5))
        self.assertTrue(math.isclose(result.scores["c2"], 1.5))
        self.assertNotIn("h1", result.scores)

        c1_atoms = [atom for atom in result.atoms if atom.candidate_id == "c1"]
        self.assertEqual({atom.source_id for atom in c1_atoms}, {"H1", "H2"})
        self.assertEqual(gateway.bm25_calls[0][1], ("h1", "h2"))

    def test_opensearch_bm25_bounds_long_source_query(self) -> None:
        gateway = _FakeRetrievalGateway()
        signal = replace(
            self.profile.positive_signals[0],
            text_fields=(
                ProfileTextField(
                    "details",
                    " ".join(f"term{index}" for index in range(1500)),
                    "catalog:h1:details",
                ),
            ),
        )
        profile = replace(self.profile, positive_signals=(signal,))
        retriever = OpenSearchBM25ContributionRetriever(gateway, "artifact-v1")

        retriever.retrieve(profile, frozenset({"H1"}), limit=10)

        query = gateway.bm25_calls[0][0]
        self.assertEqual(len(query.split()), 128)
        self.assertEqual(query.split()[0], "term0")
        self.assertEqual(query.split()[-1], "term127")

    def test_opensearch_semantic_uses_one_chunk_centroid_query_per_source(self) -> None:
        gateway = _FakeRetrievalGateway()
        gateway.item_chunk_hits = {
            "h1": (
                {"embedding": [1.0, 0.0], "raw_text": "history chunk one"},
                {"embedding": [0.5, 0.5], "raw_text": "history chunk two"},
            )
        }
        gateway.semantic_hits = {
            (0.9486832980505138, 0.31622776601683794): (
                {"item_id": "c1", "score": 0.9, "raw_text": "candidate one"},
                {"item_id": "c2", "score": 0.7, "raw_text": "candidate two"},
            ),
        }
        retriever = OpenSearchSemanticContributionRetriever(
            gateway,
            "artifact-v1",
            "embedding-v1",
        )

        result = retriever.retrieve(
            self.profile,
            frozenset({"H1"}),
            limit=10,
        )

        self.assertEqual(result.pool, ("c1", "c2"))
        self.assertTrue(math.isclose(result.scores["c1"], 0.9))
        self.assertTrue(math.isclose(result.scores["c2"], 0.7))
        self.assertEqual(len(result.atoms), 2)

        c1_atom = next(atom for atom in result.atoms if atom.candidate_id == "c1")
        self.assertEqual(c1_atom.history_span, "history chunk one | history chunk two")
        self.assertEqual(c1_atom.candidate_span, "candidate one")

        retriever.retrieve(self.profile, frozenset({"H1"}), limit=10)
        self.assertEqual(gateway.item_chunk_calls, 1)
        self.assertEqual(gateway.semantic_calls, 1)


class _FakeRetrievalGateway:
    def __init__(self) -> None:
        self.bm25_hits: dict[str, tuple[dict[str, object], ...]] = {}
        self.item_chunk_hits: dict[str, tuple[dict[str, object], ...]] = {}
        self.semantic_hits: dict[tuple[float, ...], tuple[dict[str, object], ...]] = {}

        self.bm25_calls: list[tuple[str, tuple[str, ...]]] = []
        self.item_chunk_calls = 0
        self.semantic_calls = 0

    def bm25_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_item_ids: tuple[str, ...] = (),
        explain: bool = False,
    ) -> tuple[dict[str, object], ...]:
        self.bm25_calls.append((query, tuple(excluded_item_ids)))

        return self.bm25_hits.get(query, ())[:limit]

    def item_chunks(
        self,
        item_id: str,
        *,
        embedding_model: str,
        limit: int = 1000,
    ) -> tuple[dict[str, object], ...]:
        self.item_chunk_calls += 1

        return self.item_chunk_hits.get(item_id, ())[:limit]

    def semantic_search(
        self,
        embedding: tuple[float, ...] | list[float],
        *,
        limit: int,
        excluded_item_ids: tuple[str, ...] = (),
    ) -> tuple[dict[str, object], ...]:
        self.semantic_calls += 1

        return self.semantic_hits.get(tuple(embedding), ())[:limit]


def _profile() -> UserProfile:
    signals = (
        PreferenceSignal(
            "H1",
            "h1",
            5.0,
            100,
            "positive",
            1.0,
            (ProfileTextField("title", "space future", "catalog:h1:title"),),
        ),
        PreferenceSignal(
            "H2",
            "h2",
            4.0,
            200,
            "positive",
            0.5,
            (ProfileTextField("title", "space drama", "catalog:h2:title"),),
        ),
    )

    return UserProfile(
        "user-profile-v1",
        "u1",
        300,
        "validation",
        "artifact-v1",
        frozenset({"h1", "h2"}),
        signals,
        (),
        ProfileDiagnostics(2, 0, 2, 0, 2, 0, ()),
    )


if __name__ == "__main__":
    unittest.main()
