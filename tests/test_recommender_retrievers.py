from __future__ import annotations

import math
import unittest

import numpy as np

from scipy.sparse import csr_matrix

from backend.item_knn import ItemKNNBundle, ItemKNNConfig
from backend.profile import PreferenceSignal, ProfileDiagnostics, ProfileTextField, UserProfile
from backend.recommender import (
    BM25ContributionRetriever,
    ItemKNNContributionRetriever,
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
