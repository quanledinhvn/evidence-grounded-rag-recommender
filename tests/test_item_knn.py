from __future__ import annotations

import io
import json
import tempfile
import unittest

from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from backend.data.io import read_parquet, write_parquet
from backend.item_knn.bundle import (
    ItemKNNBundle,
    ItemKNNConfig,
    train_item_knn_bundle,
)
from backend.item_knn.cli import main as item_knn_main
from backend.item_knn.evaluation import evaluate_item_knn


class ItemKNNTests(unittest.TestCase):
    def test_config_rejects_invalid_training_parameters(self) -> None:
        for values in ({"neighbors": 0}, {"num_threads": -1}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    ItemKNNConfig(**values)

    def test_validation_uses_train_without_training_bpr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            with patch(
                "backend.item_knn.bundle.read_parquet",
                wraps=read_parquet,
            ) as parquet_reader:
                bundle = train_item_knn_bundle(
                    splits_path,
                    catalog_path,
                    root / "item-knn",
                    phase="validation",
                    config=ItemKNNConfig(neighbors=2, num_threads=1),
                )

        split_call = next(
            call for call in parquet_reader.call_args_list if call.args[0] == splits_path
        )
        self.assertEqual(split_call.kwargs["filters"], [("split", "=", "train")])
        self.assertEqual(bundle.manifest["schema_version"], "item-knn-bundle-v1")

    def test_recommends_neighbors_and_filters_seen_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_item_knn_bundle(
                splits_path,
                catalog_path,
                root / "item-knn",
                phase="validation",
                config=ItemKNNConfig(neighbors=2, num_threads=1),
            )
            recommendation = bundle.recommend("u1", limit=3)

        self.assertEqual(recommendation["status"], "ok")
        self.assertEqual(
            [item["item_id"] for item in recommendation["items"]],
            ["i2"],
        )
        self.assertEqual(recommendation["items"][0]["source"], "item_knn")

    def test_cold_start_uses_own_popularity_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)

            bundle = train_item_knn_bundle(
                splits_path,
                catalog_path,
                root / "item-knn",
                phase="validation",
                config=ItemKNNConfig(neighbors=2, num_threads=1),
            )
            recommendation = bundle.recommend("unknown", limit=2)

        self.assertEqual(recommendation["status"], "cold_start")
        self.assertTrue(
            all(item["source"] == "popularity" for item in recommendation["items"])
        )

    def test_bundle_round_trip_preserves_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            output_dir = root / "item-knn"

            bundle = train_item_knn_bundle(
                splits_path,
                catalog_path,
                output_dir,
                phase="validation",
                config=ItemKNNConfig(neighbors=2, num_threads=1),
            )
            expected = bundle.recommend("u1", limit=3)
            loaded = ItemKNNBundle.load(output_dir)

            manifest = json.loads((output_dir / "manifest.json").read_text())

        self.assertEqual(loaded.recommend("u1", limit=3), expected)
        self.assertGreater(manifest["artifact_sizes"]["similarity.npz"], 0)
        self.assertIn("user_items.npz", manifest["artifact_hashes"])

    def test_evaluator_reports_ranking_and_native_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            queries_path = root / "queries.parquet"

            write_parquet(
                queries_path,
                [
                    {
                        "user_id": "u1",
                        "target_item_id": "i2",
                        "target_rating": 5.0,
                    },
                    {
                        "user_id": "unknown",
                        "target_item_id": "i1",
                        "target_rating": 4.0,
                    },
                ],
            )
            bundle = train_item_knn_bundle(
                splits_path,
                catalog_path,
                root / "item-knn",
                phase="validation",
                config=ItemKNNConfig(neighbors=2, num_threads=1),
            )
            report = evaluate_item_knn(bundle, queries_path)

        self.assertEqual(report["query_coverage"], 0.5)
        self.assertEqual(report["fallback_rate"], 0.5)
        self.assertIn("positive_target_slice", report)

    def test_cli_trains_and_recommends_without_cf_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path, catalog_path = self._write_dataset(root)
            output_dir = root / "item-knn"

            status = item_knn_main(
                [
                    "train",
                    "--splits",
                    str(splits_path),
                    "--catalog",
                    str(catalog_path),
                    "--output-dir",
                    str(output_dir),
                    "--phase",
                    "validation",
                    "--neighbors",
                    "2",
                    "--num-threads",
                    "1",
                ]
            )
            output = io.StringIO()

            with redirect_stdout(output):
                recommend_status = item_knn_main(
                    [
                        "recommend",
                        "--bundle",
                        str(output_dir),
                        "--user-id",
                        "u1",
                        "--limit",
                        "3",
                    ]
                )

        recommendation = json.loads(output.getvalue())

        self.assertEqual(status, 0)
        self.assertEqual(recommend_status, 0)
        self.assertEqual(recommendation["status"], "ok")

    @staticmethod
    def _write_dataset(root: Path) -> tuple[Path, Path]:
        splits_path = root / "splits.parquet"
        catalog_path = root / "catalog.parquet"

        rows = [
            {
                "user_id": "u1",
                "item_id": "i1",
                "rating": 5.0,
                "timestamp": 1,
                "split": "train",
            },
            {
                "user_id": "u1",
                "item_id": "i3",
                "rating": 2.0,
                "timestamp": 2,
                "split": "train",
            },
            {
                "user_id": "u2",
                "item_id": "i1",
                "rating": 5.0,
                "timestamp": 1,
                "split": "train",
            },
            {
                "user_id": "u2",
                "item_id": "i2",
                "rating": 4.0,
                "timestamp": 2,
                "split": "train",
            },
            {
                "user_id": "u3",
                "item_id": "i2",
                "rating": 5.0,
                "timestamp": 1,
                "split": "train",
            },
            {
                "user_id": "u3",
                "item_id": "i3",
                "rating": 5.0,
                "timestamp": 2,
                "split": "train",
            },
            {
                "user_id": "u1",
                "item_id": "i2",
                "rating": 5.0,
                "timestamp": 3,
                "split": "validation",
            },
        ]
        write_parquet(splits_path, rows)
        write_parquet(
            catalog_path,
            [{"item_id": item_id} for item_id in ("i1", "i2", "i3", "i4")],
        )

        return splits_path, catalog_path


if __name__ == "__main__":
    unittest.main()
