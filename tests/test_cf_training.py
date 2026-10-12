from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest

from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.cf.bpr import BPRConfig, BPRModel, config_to_dict
from backend.cf.baselines import recommend_popularity
from backend.cf.bundle import CFBundle, train_bundle
from backend.cf.cli import main as cf_main
from backend.cf.evaluation import compare_retrievers, evaluate_bundle
from backend.cf.tuning import select_bpr_config
from backend.data.io import read_parquet, write_parquet


class BPRModelTests(unittest.TestCase):
    def test_config_rejects_non_positive_training_parameters(self) -> None:
        invalid_configs = (
            {"factors": 0},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
            {"learning_rate": float("inf")},
            {"regularization": -0.1},
            {"regularization": float("nan")},
            {"epochs": 0},
            {"num_threads": -1},
        )

        for values in invalid_configs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    BPRConfig(**values)

    def test_fit_is_deterministic_and_improves_pairwise_ranking(self) -> None:
        config = BPRConfig(
            factors=4,
            learning_rate=0.05,
            regularization=0.001,
            epochs=100,
            seed=7,
            num_threads=1,
        )
        arguments = (2, 4, [(0, 0), (0, 1), (1, 2)])

        first = BPRModel(config)
        metrics = first.fit(*arguments)

        second = BPRModel(config)
        second.fit(*arguments)

        self.assertEqual(len(metrics), config.epochs)
        np.testing.assert_allclose(first.user_factors, second.user_factors)
        np.testing.assert_allclose(first.item_factors, second.item_factors)
        np.testing.assert_allclose(first.item_bias, second.item_bias)

        scores = first.score_user(0)
        self.assertGreater(scores[0], scores[2])

    def test_fit_reports_implicit_training_metrics(self) -> None:
        config = BPRConfig(factors=2, epochs=2, seed=7, num_threads=1)
        model = BPRModel(config)

        metrics = model.fit(2, 4, [(0, 0), (1, 2)])

        self.assertEqual([row["epoch"] for row in metrics], [1, 2])
        self.assertEqual(
            set(metrics[0]),
            {"epoch", "train_auc", "correct", "skipped", "duration_seconds"},
        )

    def test_config_records_implicit_backend_and_sampling_semantics(self) -> None:
        values = config_to_dict(BPRConfig(num_threads=3))

        self.assertEqual(
            values["implementation"],
            "implicit.cpu.bpr.BayesianPersonalizedRanking",
        )
        self.assertEqual(values["num_threads"], 3)
        self.assertEqual(values["negative_sampling"], "uniform_non_positive")


class CFBundleTests(unittest.TestCase):
    def test_validation_reads_only_the_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            with patch(
                "backend.cf.bundle.read_parquet",
                wraps=read_parquet,
            ) as parquet_reader:
                train_bundle(
                    splits_path,
                    catalog_path,
                    root / "bundle",
                    phase="validation",
                    config=self._config(),
                )

        split_call = next(
            call for call in parquet_reader.call_args_list if call.args[0] == splits_path
        )
        self.assertEqual(split_call.kwargs["filters"], [("split", "=", "train")])

    def test_validation_bundle_uses_only_train_and_filters_all_fit_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )
            recommendation = bundle.recommend("u1", limit=10)

        self.assertEqual(bundle.manifest["fit_splits"], ["train"])
        self.assertEqual(bundle.seen_by_user["u1"], {"i1", "i2"})
        self.assertIn("i3", {item["item_id"] for item in recommendation["items"]})
        self.assertNotIn("i1", {item["item_id"] for item in recommendation["items"]})
        self.assertEqual(recommendation["status"], "ok")

    def test_test_bundle_adds_validation_but_never_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="test",
                config=self._config(),
            )

        self.assertEqual(bundle.manifest["fit_splits"], ["train", "validation"])
        self.assertEqual(bundle.seen_by_user["u1"], {"i1", "i2", "i3"})
        self.assertNotIn("i4", bundle.seen_by_user["u1"])

    def test_cold_start_and_no_positive_user_use_popularity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )

            unknown = bundle.recommend("unknown", limit=2)
            neutral = bundle.recommend("u2", limit=2)

        self.assertEqual(unknown["status"], "cold_start")
        self.assertEqual(neutral["status"], "cold_start")
        self.assertTrue(all(item["source"] == "popularity" for item in unknown["items"]))
        self.assertNotIn("i2", {item["item_id"] for item in neutral["items"]})

    def test_popularity_counts_only_positive_interactions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )

        self.assertEqual(
            bundle.popularity,
            [("i1", 1), ("i3", 1), ("i4", 1), ("i2", 0), ("i5", 0)],
        )

    def test_recommendation_rejects_limit_above_design_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )

            with self.assertRaises(ValueError):
                bundle.recommend("u1", limit=51)

    def test_bundle_round_trip_reproduces_predictions_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            output_dir = root / "bundle"

            bundle = train_bundle(
                splits_path,
                catalog_path,
                output_dir,
                phase="validation",
                config=self._config(),
            )
            expected = bundle.recommend("u1", limit=3)
            loaded = CFBundle.load(output_dir)

            self.assertEqual(expected, loaded.recommend("u1", limit=3))
            self.assertTrue((output_dir / "training_metrics.csv").exists())
            self.assertTrue((output_dir / "popularity.parquet").exists())

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertGreaterEqual(len(manifest["artifact_hashes"]), 5)
            self.assertEqual(
                manifest["config_sha256"],
                manifest["artifact_hashes"]["bpr_config.json"],
            )
            self.assertGreater(manifest["artifact_sizes"]["bpr_model.npz"], 0)

    def test_evaluator_reports_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [
                    {
                        "user_id": "u1",
                        "target_item_id": "i3",
                        "target_rating": 5.0,
                    },
                    {
                        "user_id": "unknown",
                        "target_item_id": "i1",
                        "target_rating": 4.0,
                    },
                ],
            )
            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )
            metrics = evaluate_bundle(bundle, queries_path)

            with self.assertRaisesRegex(ValueError, "exactly 50"):
                evaluate_bundle(bundle, queries_path, limit=10)

        for key in (
            "ndcg_at_10",
            "hr_at_10",
            "candidate_recall_at_50",
            "query_coverage",
            "fallback_rate",
            "item_coverage",
            "mean_latency_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "positive_target_slice",
        ):
            self.assertIn(key, metrics)

    def test_batched_bpr_evaluation_matches_single_user_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [
                    {"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0},
                    {"user_id": "unknown", "target_item_id": "i1", "target_rating": 3.0},
                ],
            )
            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )

            expected = evaluate_bundle(bundle, queries_path)
            comparison = compare_retrievers(
                bundle,
                queries_path,
                batch_size=2,
                include_popularity=False,
            )

        actual = comparison["bpr_mf"]
        metric_names = (
            "queries",
            "ndcg_at_10",
            "hr_at_10",
            "candidate_recall_at_50",
            "query_coverage",
            "fallback_rate",
            "item_coverage",
        )
        for name in metric_names:
            self.assertEqual(actual[name], expected[name])

        for name in metric_names:
            self.assertEqual(
                actual["positive_target_slice"][name],
                expected["positive_target_slice"][name],
            )

    def test_popularity_and_bpr_comparison_have_deterministic_rankings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [{"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0}],
            )
            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )

            popular = recommend_popularity(bundle, "u1")
            comparison = compare_retrievers(bundle, queries_path)

        self.assertNotIn("i1", {item["item_id"] for item in popular})
        self.assertEqual(set(comparison), {"bpr_mf", "popularity"})
        self.assertGreater(comparison["bpr_mf"]["model_artifact_size_bytes"], 0)

    def test_comparison_reports_native_bpr_coverage_before_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [{"user_id": "unknown", "target_item_id": "i1", "target_rating": 5.0}],
            )
            bundle = train_bundle(
                splits_path,
                catalog_path,
                root / "bundle",
                phase="validation",
                config=self._config(),
            )

            comparison = compare_retrievers(bundle, queries_path)

        self.assertEqual(comparison["bpr_mf"]["query_coverage"], 0.0)
        self.assertEqual(comparison["bpr_mf"]["fallback_rate"], 1.0)
        positive = comparison["bpr_mf"]["positive_target_slice"]
        self.assertEqual(positive["query_coverage"], 0.0)
        self.assertEqual(positive["fallback_rate"], 1.0)

    def test_fit_rejects_missing_timestamp_and_duplicate_catalog_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path = root / "splits.parquet"
            catalog_path = root / "catalog.parquet"

            write_parquet(
                splits_path,
                [{"user_id": "u1", "item_id": "i1", "rating": 5.0, "split": "train"}],
            )
            write_parquet(catalog_path, [{"item_id": "i1"}])

            with self.assertRaisesRegex(ValueError, "timestamp"):
                train_bundle(
                    splits_path,
                    catalog_path,
                    root / "missing-timestamp",
                    phase="validation",
                    config=self._config(),
                )

            write_parquet(
                splits_path,
                [
                    {
                        "user_id": "u1",
                        "item_id": "i1",
                        "rating": 5.0,
                        "timestamp": 1,
                        "split": "train",
                    }
                ],
            )
            write_parquet(catalog_path, [{"item_id": "i1"}, {"item_id": "i1"}])

            with self.assertRaisesRegex(ValueError, "Duplicate catalog item"):
                train_bundle(
                    splits_path,
                    catalog_path,
                    root / "duplicate-catalog",
                    phase="validation",
                    config=self._config(),
                )

    def test_fit_rejects_null_user_and_catalog_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path = root / "splits.parquet"
            catalog_path = root / "catalog.parquet"

            write_parquet(catalog_path, [{"item_id": None}])
            write_parquet(
                splits_path,
                [
                    {
                        "user_id": "u1",
                        "item_id": "i1",
                        "rating": 5.0,
                        "timestamp": 1,
                        "split": "train",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "non-empty item_id"):
                train_bundle(
                    splits_path,
                    catalog_path,
                    root / "null-catalog",
                    phase="validation",
                    config=self._config(),
                )

            write_parquet(catalog_path, [{"item_id": "i1"}])
            write_parquet(
                splits_path,
                [
                    {
                        "user_id": None,
                        "item_id": "i1",
                        "rating": 5.0,
                        "timestamp": 1,
                        "split": "train",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "non-empty user_id"):
                train_bundle(
                    splits_path,
                    catalog_path,
                    root / "null-user",
                    phase="validation",
                    config=self._config(),
                )

    def test_pipeline_cli_trains_validation_and_test_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            validation_queries = root / "queries_val.parquet"
            test_queries = root / "queries_test.parquet"

            query = [{"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0}]
            write_parquet(validation_queries, query)
            write_parquet(test_queries, query)

            status = cf_main(
                [
                    "pipeline",
                    "--splits",
                    str(splits_path),
                    "--catalog",
                    str(catalog_path),
                    "--validation-queries",
                    str(validation_queries),
                    "--test-queries",
                    str(test_queries),
                    "--output-dir",
                    str(root / "models"),
                    "--factors",
                    "4",
                    "--epochs",
                    "2",
                ]
            )

            report_exists = (root / "models" / "evaluation.json").exists()
            lock_exists = (root / "models" / "locked_config.json").exists()
            tuning_exists = (root / "models" / "validation_metrics.csv").exists()
            validation_exists = (root / "models" / "validation" / "manifest.json").exists()
            test_exists = (root / "models" / "test" / "manifest.json").exists()

            locked_config = json.loads(
                (root / "models" / "locked_config.json").read_text()
            )

        self.assertEqual(status, 0)
        self.assertTrue(report_exists)
        self.assertTrue(lock_exists)
        self.assertTrue(tuning_exists)
        self.assertTrue(validation_exists)
        self.assertTrue(test_exists)
        self.assertIn("config_sha256", locked_config)
        self.assertEqual(locked_config["tuning_query_limit"], 50_000)
        self.assertEqual(locked_config["evaluation_batch_size"], 256)
        self.assertEqual(
            set(locked_config["input_hashes"]),
            {"catalog", "splits", "validation_queries"},
        )

    def test_recommend_cli_returns_recommendations_for_one_user(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            bundle_dir = root / "bundle"
            output = io.StringIO()

            train_bundle(
                splits_path,
                catalog_path,
                bundle_dir,
                phase="validation",
                config=self._config(),
            )

            with redirect_stdout(output):
                status = cf_main(
                    [
                        "recommend",
                        "--bundle",
                        str(bundle_dir),
                        "--user-id",
                        "u1",
                        "--seen-id",
                        "i3",
                        "--limit",
                        "2",
                    ]
                )

        recommendation = json.loads(output.getvalue())
        item_ids = {item["item_id"] for item in recommendation["items"]}

        self.assertEqual(status, 0)
        self.assertEqual(recommendation["status"], "ok")
        self.assertNotIn("i1", item_ids)
        self.assertNotIn("i2", item_ids)
        self.assertNotIn("i3", item_ids)

    @staticmethod
    def _config() -> BPRConfig:
        return BPRConfig(factors=4, epochs=5, seed=3, num_threads=1)

    @staticmethod
    def _write_dataset(root: Path) -> tuple[Path, Path]:
        splits_path = root / "splits.parquet"
        catalog_path = root / "catalog.parquet"

        rows = [
            {"user_id": "u1", "item_id": "i1", "rating": 5.0, "timestamp": 1, "split": "train"},
            {"user_id": "u1", "item_id": "i2", "rating": 1.0, "timestamp": 2, "split": "train"},
            {"user_id": "u1", "item_id": "i3", "rating": 5.0, "timestamp": 3, "split": "validation"},
            {"user_id": "u1", "item_id": "i4", "rating": 5.0, "timestamp": 4, "split": "test"},
            {"user_id": "u2", "item_id": "i2", "rating": 3.0, "timestamp": 1, "split": "train"},
            {"user_id": "u3", "item_id": "i3", "rating": 5.0, "timestamp": 1, "split": "train"},
            {"user_id": "u3", "item_id": "i4", "rating": 5.0, "timestamp": 2, "split": "train"},
        ]
        write_parquet(splits_path, rows)
        write_parquet(catalog_path, [{"item_id": f"i{index}"} for index in range(1, 6)])

        return splits_path, catalog_path


class TuningTests(unittest.TestCase):
    def test_selects_by_metrics_and_writes_validation_table(self) -> None:
        first = BPRConfig(factors=4, epochs=1, seed=3)
        second = BPRConfig(factors=8, epochs=1, seed=3)
        comparison_options: list[dict[str, object]] = []

        def fake_train(*args: object, config: BPRConfig, **kwargs: object) -> object:
            return SimpleNamespace(model=SimpleNamespace(config=config))

        def fake_compare(
            bundle: object,
            queries_path: Path,
            **kwargs: object,
        ) -> dict[str, object]:
            factors = bundle.model.config.factors
            comparison_options.append(kwargs)

            return {
                "bpr_mf": {
                    "ndcg_at_10": 0.4 if factors == 4 else 0.5,
                    "candidate_recall_at_50": 0.7,
                    "model_size_bytes": factors * 100,
                    "mean_latency_ms": 2.0,
                }
            }

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)

            with patch("backend.cf.tuning.train_bundle", side_effect=fake_train):
                with patch(
                    "backend.cf.tuning.compare_retrievers",
                    side_effect=fake_compare,
                ):
                    selected, records = select_bpr_config(
                        Path("splits.parquet"),
                        Path("catalog.parquet"),
                        Path("queries.parquet"),
                        output_dir,
                        [first, second],
                        query_limit=1,
                        batch_size=2,
                    )

            with (output_dir / "validation_metrics.csv").open(
                encoding="utf-8",
                newline="",
            ) as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(selected, second)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["selected"] for row in rows], ["false", "true"])
        self.assertEqual(
            comparison_options,
            [
                {
                    "batch_size": 2,
                    "include_popularity": False,
                    "query_limit": 1,
                    "query_seed": 3,
                },
                {
                    "batch_size": 2,
                    "include_popularity": False,
                    "query_limit": 1,
                    "query_seed": 3,
                },
            ],
        )

    def test_tuning_rejects_an_empty_candidate_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaises(ValueError):
                select_bpr_config(
                    Path("splits.parquet"),
                    Path("catalog.parquet"),
                    Path("queries.parquet"),
                    Path(temporary_dir),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
