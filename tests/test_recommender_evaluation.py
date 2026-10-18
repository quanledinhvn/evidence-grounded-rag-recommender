from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from backend.data.io import write_parquet
from backend.item_knn import ItemKNNBundle, ItemKNNConfig
from backend.profile import (
    PreferenceSignal,
    ProfileDiagnostics,
    ProfileTextField,
    UserProfile,
)
from backend.recommender import (
    FrozenContributionRetriever,
    HybridPipeline,
    PipelineConfig,
)
from backend.recommender.evaluation import EvaluationCase, evaluate_cases
from backend.recommender.evaluation_cli import _build_cases, sample_queries


class RecommenderEvaluationTests(unittest.TestCase):
    def test_builds_profiles_with_one_history_and_one_catalog_scan(self) -> None:
        split_rows = [
            {
                "user_id": user_id,
                "item_id": item_id,
                "rating": 5.0,
                "timestamp": 100,
                "review_id": f"r-{user_id}",
                "split": "train",
            }
            for user_id, item_id in (("u1", "h1"), ("u2", "h2"))
        ]
        catalog_rows = [
            {"item_id": item_id, "source_id": f"catalog:{item_id}", "title": item_id}
            for item_id in ("h1", "h2")
        ]
        queries = (
            {"user_id": "u1", "as_of": 200, "target_item_id": "c1", "target_rating": 5.0},
            {"user_id": "u2", "as_of": 200, "target_item_id": "c2", "target_rating": 4.0},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits_path = root / "splits.parquet"
            catalog_path = root / "catalog.parquet"

            write_parquet(splits_path, split_rows)
            write_parquet(catalog_path, catalog_rows)

            from backend.recommender import evaluation_cli

            with patch.object(
                evaluation_cli,
                "read_parquet",
                wraps=evaluation_cli.read_parquet,
            ) as reader:
                cases = _build_cases(
                    queries,
                    splits_path,
                    catalog_path,
                    "artifact-v1",
                )

        self.assertEqual(len(cases), 2)
        self.assertEqual(reader.call_count, 2)
        self.assertEqual(cases[0].profile.seen_item_ids, frozenset({"h1"}))

    def test_samples_queries_deterministically_without_replacement(self) -> None:
        queries = [{"query_id": str(index)} for index in range(10)]

        first = sample_queries(queries, 4, 42)
        second = sample_queries(queries, 4, 42)
        larger = sample_queries(queries, 7, 42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len({row["query_id"] for row in first}), 4)
        self.assertLessEqual(
            {row["query_id"] for row in first},
            {row["query_id"] for row in larger},
        )

        with self.assertRaisesRegex(ValueError, "sample_size"):
            sample_queries(queries, 0, 42)

    def test_evaluates_five_frozen_rankers_and_resumes_checkpoint(self) -> None:
        profile = _profile()
        pipeline = _pipeline()
        bundle = _bundle()
        cases = (
            EvaluationCase("q1", profile, "c1", 5.0),
            EvaluationCase("q2", profile, "c2", 4.0),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.jsonl"
            first = evaluate_cases(
                pipeline,
                bundle,
                cases,
                checkpoint_path=checkpoint,
            )
            second = evaluate_cases(
                pipeline,
                bundle,
                cases,
                checkpoint_path=checkpoint,
            )

            lines = checkpoint.read_text(encoding="utf-8").splitlines()

        self.assertEqual(first, second)
        self.assertEqual(len(lines), 2)
        self.assertEqual(first["completed_query_count"], 2)
        self.assertEqual(set(first["rankers"]), {
            "popularity",
            "itemknn",
            "bm25",
            "semantic",
            "hybrid",
        })
        self.assertEqual(first["rankers"]["hybrid"]["hr_at_10"], 1.0)
        self.assertGreater(first["rankers"]["hybrid"]["ndcg_at_10"], 0.0)


def _pipeline() -> HybridPipeline:
    retrievers = {
        "itemknn": FrozenContributionRetriever(
            "itemknn",
            {"H1": {"c1": 3.0, "c2": 1.0}},
            "artifact-v1",
        ),
        "bm25": FrozenContributionRetriever(
            "bm25",
            {"H1": {"c2": 2.0, "c1": 1.0}},
            "artifact-v1",
        ),
        "semantic": FrozenContributionRetriever(
            "semantic",
            {"H1": {"c1": 2.0, "c3": 1.0}},
            "artifact-v1",
        ),
    }
    config = PipelineConfig(
        top_k=2,
        pool_size=3,
        evidence_budget=1,
        weights={"itemknn": 1 / 3, "bm25": 1 / 3, "semantic": 1 / 3},
    )

    return HybridPipeline(
        retrievers,
        config,
        artifact_manifest_hash="manifest-v1",
        fallback_item_ids=("c1", "c2", "c3"),
    )


def _bundle() -> ItemKNNBundle:
    similarity = csr_matrix(
        np.array(
            [
                [0.0, 0.8, 0.2, 0.0],
                [0.8, 0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    return ItemKNNBundle(
        "test",
        ItemKNNConfig(neighbors=3),
        similarity,
        csr_matrix(([1.0], ([0], [0])), shape=(1, 4)),
        {"u1": 0},
        {"h1": 0, "c1": 1, "c2": 2, "c3": 3},
        {"u1": {"h1"}},
        {"u1"},
        [("c1", 3), ("c2", 2), ("c3", 1), ("h1", 1)],
        {"phase": "test", "artifact_hashes": {}},
    )


def _profile() -> UserProfile:
    signal = PreferenceSignal(
        "H1",
        "h1",
        5.0,
        100,
        "positive",
        1.0,
        (ProfileTextField("title", "space", "catalog:h1:title"),),
    )

    return UserProfile(
        "user-profile-v1",
        "u1",
        200,
        "test",
        "artifact-v1",
        frozenset({"h1"}),
        (signal,),
        (),
        ProfileDiagnostics(1, 0, 1, 0, 1, 0, ()),
    )


if __name__ == "__main__":
    unittest.main()
