from __future__ import annotations

import io
import json
import math
import tempfile
import unittest

from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from scipy.sparse import csr_matrix

from backend.cf.cli import main as cf_main
from backend.cf.itemknn import (
    ItemKNNBundle,
    ItemKNNConfig,
    build_similarity,
    evaluate_itemknn,
    recency_weights,
    train_itemknn_bundle,
)
from backend.data.io import write_parquet


class ItemKNNMathTests(unittest.TestCase):
    def test_config_rejects_invalid_parameters(self) -> None:
        invalid_configs = (
            {"neighbors": 0},
            {"recency_decay": 0.0},
            {"recency_decay": 1.5},
            {"recency_decay": float("nan")},
            {"min_rating": 0.5},
            {"min_rating": 6.0},
        )

        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ItemKNNConfig(**values)

    def test_recency_weights_give_the_newest_item_full_weight(self) -> None:
        np.testing.assert_allclose(recency_weights(3, 0.5), [0.25, 0.5, 1.0])
        np.testing.assert_allclose(recency_weights(2, 1.0), [1.0, 1.0])

        self.assertEqual(len(recency_weights(0, 0.5)), 0)

    def test_similarity_is_cosine_without_self_links(self) -> None:
        interactions = csr_matrix(
            np.array(
                [
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        )

        similarity = build_similarity(interactions, neighbors=5).toarray()

        expected = np.array(
            [
                [0.0, 1 / math.sqrt(2), 0.5],
                [1 / math.sqrt(2), 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        )
        np.testing.assert_allclose(similarity, expected, rtol=1e-6)

    def test_similarity_keeps_only_top_neighbors_per_item(self) -> None:
        interactions = csr_matrix(
            np.array(
                [
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
        )

        similarity = build_similarity(interactions, neighbors=1, batch_size=2)

        self.assertEqual(similarity.getnnz(axis=1).tolist(), [1, 1, 1])
        self.assertAlmostEqual(similarity[0, 1], 1 / math.sqrt(2), places=6)
        self.assertEqual(similarity[0, 2], 0.0)


class ItemKNNBundleTests(unittest.TestCase):
    def test_validation_bundle_uses_only_train_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        self.assertEqual(bundle.manifest["fit_splits"], ["train"])
        self.assertEqual(bundle.manifest["schema_version"], "cf-itemknn-bundle-v1")
        self.assertEqual(self._history(bundle, "u1"), ["i1", "i2"])
        self.assertNotIn("u5", bundle.user_ids)

    def test_test_bundle_adds_validation_but_never_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="test")

        self.assertEqual(bundle.manifest["fit_splits"], ["train", "validation"])
        self.assertEqual(self._history(bundle, "u1"), ["i1", "i2", "i3"])
        self.assertNotIn("u5", bundle.user_ids)

    def test_recent_history_dominates_and_popularity_fills_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        result = bundle.recommend("u1", limit=4)
        items = result["items"]

        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["item_id"] for item in items], ["i4", "i3", "i5", "i6"])
        self.assertEqual(
            [item["source"] for item in items],
            ["itemknn", "itemknn", "popularity", "popularity"],
        )
        self.assertAlmostEqual(items[0]["cf_score"], 1 / math.sqrt(2), places=6)
        self.assertAlmostEqual(items[1]["cf_score"], 0.5 / math.sqrt(2), places=6)

    def test_uniform_decay_breaks_score_ties_by_item_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(
                root,
                phase="validation",
                config=ItemKNNConfig(recency_decay=1.0),
            )

        items = bundle.recommend("u1", limit=2)["items"]

        self.assertEqual([item["item_id"] for item in items], ["i3", "i4"])

    def test_recommendation_explains_contributing_history_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        items = bundle.recommend("u1", limit=4)["items"]
        because = items[0]["because"]

        self.assertEqual([entry["item_id"] for entry in because], ["i2"])
        self.assertAlmostEqual(because[0]["similarity"], 1 / math.sqrt(2), places=6)
        self.assertAlmostEqual(because[0]["weight"], 1.0)
        self.assertEqual(items[2]["because"], [])

    def test_seen_ids_and_history_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        items = bundle.recommend("u1", seen_ids={"i4"}, limit=10)["items"]

        self.assertEqual([item["item_id"] for item in items], ["i3", "i5", "i6"])

    def test_unknown_user_gets_popularity_cold_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        result = bundle.recommend("unknown", limit=3)

        self.assertEqual(result["status"], "cold_start")
        self.assertEqual([item["item_id"] for item in result["items"]], ["i1", "i2", "i3"])
        self.assertTrue(all(item["source"] == "popularity" for item in result["items"]))

    def test_min_rating_drops_low_ratings_from_similarity_but_keeps_them_seen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(
                root,
                phase="validation",
                config=ItemKNNConfig(min_rating=4.0),
            )

        result = bundle.recommend("u4", limit=10)
        item_ids = [item["item_id"] for item in result["items"]]

        self.assertNotIn("i6", item_ids)
        self.assertNotIn("i5", item_ids)
        self.assertTrue(all(item["source"] == "popularity" for item in result["items"]))

    def test_recommendation_rejects_limit_above_design_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        with self.assertRaises(ValueError):
            bundle.recommend("u1", limit=51)

    def test_rank_users_matches_single_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")

        rankings = bundle.rank_users(["u1", "unknown"], limit=50)

        self.assertEqual(rankings[0], (["i4", "i3", "i5", "i6"], 2))
        self.assertEqual(rankings[1], (["i1", "i2", "i3", "i5", "i4", "i6"], 0))

    def test_save_and_load_round_trip_with_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")
            loaded = ItemKNNBundle.load(root / "bundle")

            expected = bundle.recommend("u1", limit=4)
            actual = loaded.recommend("u1", limit=4)

            (root / "bundle" / "popularity.parquet").write_bytes(b"tampered")

            with self.assertRaisesRegex(ValueError, "checksum"):
                ItemKNNBundle.load(root / "bundle")

        self.assertEqual(actual, expected)
        self.assertEqual(loaded.config, bundle.config)

    def test_fit_rejects_duplicate_pairs_and_items_outside_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path = root / "splits.parquet"
            catalog_path = root / "catalog.parquet"

            row = {"user_id": "u1", "item_id": "i1", "rating": 5.0, "timestamp": 1, "split": "train"}
            write_parquet(splits_path, [row, {**row, "timestamp": 2}])
            write_parquet(catalog_path, [{"item_id": "i1"}])

            with self.assertRaisesRegex(ValueError, "Duplicate user-item pair"):
                train_itemknn_bundle(splits_path, catalog_path, root / "dup", phase="validation")

            write_parquet(splits_path, [{**row, "item_id": "i9"}])

            with self.assertRaisesRegex(ValueError, "outside catalog"):
                train_itemknn_bundle(splits_path, catalog_path, root / "out", phase="validation")

    def test_evaluation_reports_ranking_metrics_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            bundle = self._train(root, phase="validation")
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [
                    {"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0},
                    {"user_id": "unknown", "target_item_id": "i1", "target_rating": 2.0},
                ],
            )

            report = evaluate_itemknn(bundle, queries_path)

        self.assertEqual(report["queries"], 2)
        self.assertAlmostEqual(report["ndcg_at_10"], (1 / math.log2(3) + 1.0) / 2)
        self.assertEqual(report["hr_at_10"], 1.0)
        self.assertEqual(report["query_coverage"], 0.5)
        self.assertEqual(report["fallback_rate"], 0.5)
        self.assertEqual(report["positive_target_slice"]["queries"], 1)
        self.assertAlmostEqual(report["positive_target_slice"]["ndcg_at_10"], 1 / math.log2(3))

    def test_cli_trains_recommends_and_evaluates_itemknn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = _write_dataset(root)
            bundle_dir = root / "bundle"
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [{"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0}],
            )

            train_status = cf_main(
                [
                    "train-itemknn",
                    "--splits",
                    str(splits_path),
                    "--catalog",
                    str(catalog_path),
                    "--output-dir",
                    str(bundle_dir),
                    "--phase",
                    "validation",
                    "--neighbors",
                    "10",
                ]
            )

            recommend_output = io.StringIO()
            with redirect_stdout(recommend_output):
                cf_main(["recommend", "--bundle", str(bundle_dir), "--user-id", "u1", "--limit", "2"])

            evaluate_output = io.StringIO()
            with redirect_stdout(evaluate_output):
                cf_main(
                    [
                        "evaluate-itemknn",
                        "--bundle",
                        str(bundle_dir),
                        "--queries",
                        str(queries_path),
                        "--output",
                        str(root / "report.json"),
                    ]
                )

            saved_report = json.loads((root / "report.json").read_text(encoding="utf-8"))

        recommendation = json.loads(recommend_output.getvalue())
        report = json.loads(evaluate_output.getvalue())

        self.assertEqual(train_status, 0)
        self.assertEqual([item["item_id"] for item in recommendation["items"]], ["i4", "i3"])
        self.assertEqual(report["hr_at_10"], 1.0)
        self.assertEqual(saved_report, report)

    @staticmethod
    def _train(
        root: Path,
        *,
        phase: str,
        config: ItemKNNConfig | None = None,
    ) -> ItemKNNBundle:
        splits_path, catalog_path = _write_dataset(root)

        return train_itemknn_bundle(
            splits_path,
            catalog_path,
            root / "bundle",
            phase=phase,
            config=config,
        )

    @staticmethod
    def _history(bundle: ItemKNNBundle, user_id: str) -> list[str]:
        user_index = bundle.user_ids.index(user_id)
        start, end = bundle.history_indptr[user_index : user_index + 2]

        return [bundle.item_ids[index] for index in bundle.history_items[start:end]]


def _write_dataset(root: Path) -> tuple[Path, Path]:
    """Tiny fixture.

    Train co-occurrence: i1-i3 (u2), i2-i4 (u3), i1-i2 (u1), i5-i6 (u4).
    u1 watched i1 before i2, so i2 neighbours should dominate under decay.
    """
    splits_path = root / "splits.parquet"
    catalog_path = root / "catalog.parquet"

    rows = [
        {"user_id": "u1", "item_id": "i1", "rating": 5.0, "timestamp": 1, "split": "train"},
        {"user_id": "u1", "item_id": "i2", "rating": 5.0, "timestamp": 2, "split": "train"},
        {"user_id": "u1", "item_id": "i3", "rating": 5.0, "timestamp": 3, "split": "validation"},
        {"user_id": "u1", "item_id": "i4", "rating": 5.0, "timestamp": 4, "split": "test"},
        {"user_id": "u2", "item_id": "i1", "rating": 5.0, "timestamp": 1, "split": "train"},
        {"user_id": "u2", "item_id": "i3", "rating": 4.0, "timestamp": 2, "split": "train"},
        {"user_id": "u3", "item_id": "i2", "rating": 5.0, "timestamp": 1, "split": "train"},
        {"user_id": "u3", "item_id": "i4", "rating": 3.0, "timestamp": 2, "split": "train"},
        {"user_id": "u4", "item_id": "i5", "rating": 5.0, "timestamp": 1, "split": "train"},
        {"user_id": "u4", "item_id": "i6", "rating": 2.0, "timestamp": 2, "split": "train"},
        {"user_id": "u5", "item_id": "i1", "rating": 5.0, "timestamp": 9, "split": "test"},
    ]
    write_parquet(splits_path, rows)
    write_parquet(catalog_path, [{"item_id": f"i{index}"} for index in range(1, 7)])

    return splits_path, catalog_path
