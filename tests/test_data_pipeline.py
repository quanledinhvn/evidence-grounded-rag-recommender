from __future__ import annotations

import csv
import json
import tempfile
import unittest

from datetime import datetime, timezone
from pathlib import Path

from backend.data.audit import audit_dataset
from backend.data.cli import main as data_main
from backend.data.deduplicate import deduplicate_dataset, deduplicate_rows
from backend.data.enrich import enrich_catalog

from backend.data.io import read_parquet, write_parquet

from backend.data.normalize import clean_text, normalize_dataset
from backend.data.presplit import import_pre_split_dataset
from backend.data.split import split_dataset, split_rows
from backend.data.validation import (
    has_usable_text,
    is_valid_id,
    review_date_mismatch,
    review_issues,
    usable_text_count,
)


AUDIT_TIME = datetime(2021, 1, 1, tzinfo=timezone.utc)


class ValidationTests(unittest.TestCase):
    def test_ids_require_non_empty_strings(self) -> None:
        self.assertTrue(is_valid_id(" A1 "))

        for value in (None, "", "   ", 12):
            self.assertFalse(is_valid_id(value))

    def test_text_fields_accept_strings_and_lists_of_strings(self) -> None:
        self.assertTrue(has_usable_text(" title "))
        self.assertTrue(has_usable_text(["", " feature "]))

        for value in (None, " ", [], [""], [12], {"title": "x"}):
            self.assertFalse(has_usable_text(value))

        record = {
            "title": "A",
            "description": ["B"],
            "feature": [],
            "category": None,
        }
        self.assertEqual(usable_text_count(record), 2)

    def test_review_validation_and_date_cross_check(self) -> None:
        valid = {
            "reviewerID": "u1",
            "asin": "i1",
            "overall": 5.0,
            "unixReviewTime": 1_577_836_800,
            "reviewTime": "01 1, 2020",
        }
        self.assertEqual(review_issues(valid, AUDIT_TIME), set())
        self.assertFalse(review_date_mismatch(valid))

        invalid = dict(valid, overall=float("nan"), unixReviewTime=True)
        self.assertEqual(review_issues(invalid, AUDIT_TIME), {"rating", "timestamp"})

        self.assertTrue(review_date_mismatch(dict(valid, reviewTime="01 2, 2020")))
        self.assertTrue(review_date_mismatch(dict(valid, reviewTime="not-a-date")))


class NormalizationTests(unittest.TestCase):
    def test_clean_text_unescapes_html_strips_tags_and_collapses_space(self) -> None:
        raw = "  A &amp; B <b>live</b>\n\t now  "
        self.assertEqual(clean_text(raw), "A & B live now")

        self.assertEqual(clean_text([" one ", "<i>two</i>", 3]), "one two")

    def test_deduplicate_keeps_earliest_timestamp_then_source_line(self) -> None:
        rows = [
            {"user_id": "u1", "item_id": "i1", "timestamp": 20, "source_line": 1},
            {"user_id": "u1", "item_id": "i1", "timestamp": 10, "source_line": 4},
            {"user_id": "u1", "item_id": "i1", "timestamp": 10, "source_line": 3},
            {"user_id": "u2", "item_id": "i1", "timestamp": 5, "source_line": 2},
        ]
        actual = deduplicate_rows(rows)

        self.assertEqual(len(actual), 2)
        self.assertEqual(actual[0]["source_line"], 3)
        self.assertEqual(actual[1]["user_id"], "u2")


class SplitTests(unittest.TestCase):
    def test_split_rows_uses_leave_last_two_out_per_eligible_user(self) -> None:
        rows = [
            self._row("u1", "i3", 20, 3),
            self._row("u2", "i5", 30, 5),
            self._row("u1", "i1", 10, 1),
            self._row("u2", "i4", 10, 4),
            self._row("u1", "i2", 20, 2),
        ]

        splits, validation, test = split_rows(rows)
        assigned = {
            (row["user_id"], row["item_id"]): row["split"]
            for row in splits
        }

        self.assertEqual(assigned[("u1", "i1")], "train")
        self.assertEqual(assigned[("u1", "i2")], "validation")
        self.assertEqual(assigned[("u1", "i3")], "test")

        self.assertEqual(assigned[("u2", "i4")], "train")
        self.assertEqual(assigned[("u2", "i5")], "train")

        self.assertEqual(validation[0]["user_id"], "u1")
        self.assertEqual(validation[0]["as_of"], 20)
        self.assertEqual(validation[0]["target_item_id"], "i2")

        self.assertEqual(test[0]["user_id"], "u1")
        self.assertEqual(test[0]["as_of"], 20)
        self.assertEqual(test[0]["target_item_id"], "i3")

    def test_split_rows_rejects_duplicate_user_item_pairs(self) -> None:
        rows = [
            self._row("u1", "i1", 10, 1),
            self._row("u1", "i1", 20, 2),
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate user-item pair"):
            split_rows(rows)

    def test_split_dataset_writes_artifacts_and_stats(self) -> None:
        rows = [
            self._row("u1", "i1", 10, 1),
            self._row("u1", "i2", 20, 2),
            self._row("u1", "i3", 30, 3),
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "interactions_deduplicated.parquet"
            write_parquet(input_path, rows)

            stats = split_dataset(input_path, root)
            splits = read_parquet(root / "splits.parquet")

            validation = read_parquet(root / "queries_val.parquet")
            test = read_parquet(root / "queries_test.parquet")

        self.assertEqual(stats["eligible_users"], 1)
        self.assertEqual(stats["split_counts"]["train"], 1)
        self.assertEqual(len(splits), 3)

        self.assertEqual(len(validation), 1)
        self.assertEqual(len(test), 1)

    def test_split_cli_runs_d3(self) -> None:
        rows = [
            self._row("u1", "i1", 10, 1),
            self._row("u1", "i2", 20, 2),
            self._row("u1", "i3", 30, 3),
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "interactions.parquet"
            output_dir = root / "output"

            write_parquet(input_path, rows)

            status = data_main(
                [
                    "split",
                    "--interactions",
                    str(input_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            stats_exists = (output_dir / "split_stats.json").exists()

        self.assertEqual(status, 0)
        self.assertTrue(stats_exists)

    @staticmethod
    def _row(
        user_id: str,
        item_id: str,
        timestamp: int,
        source_line: int,
    ) -> dict[str, object]:
        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": 5.0,
            "timestamp": timestamp,
            "review_id": f"r{source_line}",
            "review_text": "Review",
            "summary": "Summary",
            "text": "Summary Review",
            "source_line": source_line,
        }


class PreSplitImportTests(unittest.TestCase):
    def test_imports_existing_csv_splits_without_resplitting(self) -> None:
        train_rows = [
            self._source_row("u1", "i1", 5.0, 1_000, ""),
            self._source_row("u1", "i2", 4.0, 2_000, "i1"),
            self._source_row("u2", "i2", 3.0, 1_500, ""),
        ]
        validation_rows = [
            self._source_row("u1", "i3", 5.0, 3_000, "i1 i2"),
        ]
        test_rows = [
            self._source_row("u1", "i4", 4.0, 4_000, "i1 i2 i3"),
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write_csv(root / "train.csv", train_rows)
            self._write_csv(root / "valid.csv", validation_rows)
            self._write_csv(root / "test.csv", test_rows)

            stats = import_pre_split_dataset(
                root / "train.csv",
                root / "valid.csv",
                root / "test.csv",
                root / "output",
            )
            splits = read_parquet(root / "output" / "splits.parquet")
            catalog = read_parquet(root / "output" / "catalog.parquet")

            validation = read_parquet(root / "output" / "queries_val.parquet")
            test = read_parquet(root / "output" / "queries_test.parquet")

        assigned = {(row["user_id"], row["item_id"]): row["split"] for row in splits}
        self.assertEqual(assigned[("u1", "i3")], "validation")
        self.assertEqual(assigned[("u1", "i4")], "test")

        self.assertEqual(stats["split_counts"], {"train": 3, "validation": 1, "test": 1})
        self.assertEqual({row["item_id"] for row in catalog}, {"i1", "i2", "i3", "i4"})

        self.assertEqual(validation[0]["target_item_id"], "i3")
        self.assertEqual(validation[0]["as_of"], 3_000)
        self.assertEqual(test[0]["target_item_id"], "i4")

    def test_rejects_invalid_header_and_history_containing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write_csv(
                root / "train.csv",
                [self._source_row("u1", "i1", 5.0, 1_000, "i1")],
            )
            self._write_csv(root / "valid.csv", [])
            self._write_csv(root / "test.csv", [])

            with self.assertRaisesRegex(ValueError, "target item appears in history"):
                import_pre_split_dataset(
                    root / "train.csv",
                    root / "valid.csv",
                    root / "test.csv",
                    root / "output",
                )

            invalid_path = root / "invalid.csv"
            invalid_path.write_text("user_id,parent_asin,rating\nu1,i1,5\n")

            with self.assertRaisesRegex(ValueError, "CSV header"):
                import_pre_split_dataset(
                    invalid_path,
                    root / "valid.csv",
                    root / "test.csv",
                    root / "other-output",
                )

    def test_cli_imports_pre_split_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source_row = self._source_row("u1", "i1", 5.0, 1_000, "")

            for name in ("train", "valid", "test"):
                self._write_csv(root / f"{name}.csv", [source_row])

            status = data_main(
                [
                    "import-splits",
                    "--train",
                    str(root / "train.csv"),
                    "--validation",
                    str(root / "valid.csv"),
                    "--test",
                    str(root / "test.csv"),
                    "--output-dir",
                    str(root / "output"),
                ]
            )

            stats_exists = (root / "output" / "split_stats.json").exists()

        self.assertEqual(status, 0)
        self.assertTrue(stats_exists)

    @staticmethod
    def _source_row(
        user_id: str,
        item_id: str,
        rating: float,
        timestamp: int,
        history: str,
    ) -> dict[str, object]:
        return {
            "user_id": user_id,
            "parent_asin": item_id,
            "rating": rating,
            "timestamp": timestamp,
            "history": history,
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        fieldnames = ["user_id", "parent_asin", "rating", "timestamp", "history"]

        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class CatalogEnrichmentTests(unittest.TestCase):
    def test_enriches_only_catalog_ids_and_preserves_missing_items(self) -> None:
        catalog = [
            self._catalog_row("i1"),
            self._catalog_row("i2"),
            self._catalog_row("i3"),
        ]
        metadata = [
            {"parent_asin": "outside", "title": "Not selected"},
            {"parent_asin": "i1", "title": "Short title"},
            {
                "parent_asin": "i1",
                "title": "Better title",
                "subtitle": "Collector edition",
                "main_category": "Drama",
                "description": ["Mystery story"],
                "features": ["IMDb 8.0"],
                "categories": ["Drama", "Mystery"],
                "store": "Example Studio",
                "details": {
                    "Actors": ["Actor One", "Actor Two"],
                    "Directors": ["Director One"],
                    "Product Dimensions": "1 x 2 inches",
                },
                "average_rating": 4.8,
                "rating_number": 125,
            },
            {"parent_asin": "i2", "title": ""},
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            catalog_path = root / "catalog.parquet"
            metadata_path = root / "meta.jsonl"
            output_path = root / "catalog_enriched.parquet"

            write_parquet(catalog_path, catalog)
            self._write_jsonl(metadata_path, metadata)

            stats = enrich_catalog(catalog_path, metadata_path, output_path)
            enriched = read_parquet(output_path)

        by_id = {row["item_id"]: row for row in enriched}
        self.assertEqual(set(by_id), {"i1", "i2", "i3"})
        self.assertEqual(by_id["i1"]["title"], "Better title")

        self.assertEqual(by_id["i1"]["main_category"], "Drama")
        self.assertEqual(by_id["i1"]["category"], "Mystery")
        self.assertIn("Mystery story", by_id["i1"]["metadata_text"])
        self.assertIn("Actors: Actor One Actor Two", by_id["i1"]["details"])

        self.assertNotIn("Product Dimensions", by_id["i1"]["details"])
        self.assertNotIn("Example Studio", by_id["i1"]["metadata_text"])
        self.assertNotIn("4.8", by_id["i1"]["metadata_text"])
        self.assertEqual(by_id["i1"]["average_rating"], 4.8)

        self.assertEqual(by_id["i1"]["rating_number"], 125)
        self.assertEqual(by_id["i3"]["metadata_text"], "")

        self.assertEqual(stats["catalog_items"], 3)
        self.assertEqual(stats["matched_metadata_rows"], 3)
        self.assertEqual(stats["duplicate_metadata_rows"], 1)

        self.assertEqual(stats["catalog_items_with_text"], 1)
        self.assertEqual(stats["catalog_items_without_text"], 2)
        self.assertEqual(stats["field_coverage"]["details"], 1)
        self.assertEqual(stats["field_coverage"]["subtitle"], 1)

    def test_cli_enriches_catalog_and_writes_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            catalog_path = root / "catalog.parquet"
            metadata_path = root / "meta.jsonl"
            output_path = root / "catalog_enriched.parquet"

            write_parquet(catalog_path, [self._catalog_row("i1")])
            self._write_jsonl(
                metadata_path,
                [{"parent_asin": "i1", "title": "Movie"}],
            )

            status = data_main(
                [
                    "enrich-catalog",
                    "--catalog",
                    str(catalog_path),
                    "--metadata",
                    str(metadata_path),
                    "--output",
                    str(output_path),
                ]
            )

            output_exists = output_path.exists()
            stats_exists = output_path.with_suffix(".stats.json").exists()

        self.assertEqual(status, 0)
        self.assertTrue(output_exists)
        self.assertTrue(stats_exists)

    @staticmethod
    def _catalog_row(item_id: str) -> dict[str, object]:
        return {
            "item_id": item_id,
            "title": None,
            "subtitle": None,
            "description": "",
            "feature": "",
            "main_category": "",
            "category": "",
            "store": "",
            "details": "",
            "metadata_text": "",
            "average_rating": None,
            "rating_number": None,
            "source_id": None,
            "source_line": None,
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


class AuditTests(unittest.TestCase):
    def test_audit_reports_required_metrics_and_error_lines(self) -> None:
        reviews = [
            self._review("u1", "i1", 5, 1_577_836_800, "01 1, 2020"),
            self._review("u1", "i1", 5, 1_577_836_800, "01 1, 2020"),
            self._review("u1", "i1", 4, 1_546_300_800, "01 1, 2019"),
            self._review(" ", "i3", 5, 1_577_836_800, "01 1, 2020"),
            self._review("u3", None, 5, 1_577_836_800, "01 1, 2020"),
            self._review("u4", "i4", 6, 1_577_836_800, "01 1, 2020"),
            self._review("u5", "i5", 4, 1_640_995_200, "01 1, 2022"),
            self._review("u2", "i2", 3, 1_577_836_800, "01 2, 2020"),
        ]
        metadata = [
            {"asin": "i1", "title": "Title"},
            {"asin": "i1", "description": ["Description"], "feature": ["Feature"]},
            {"asin": "i2", "title": " "},
            {"asin": None, "title": "Invalid key"},
            {"asin": "i3", "title": "Unused"},
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reviews_path = root / "reviews.jsonl"
            metadata_path = root / "metadata.jsonl"

            self._write_jsonl(reviews_path, reviews)
            self._write_jsonl(metadata_path, metadata)
            report = audit_dataset(
                reviews_path,
                metadata_path,
                audit_time=AUDIT_TIME,
            )

        metrics = report["metrics"]
        self.assertEqual(metrics["review_rows"], 8)
        self.assertEqual(metrics["unique_valid_user_ids"], 5)

        self.assertEqual(metrics["unique_valid_review_items"], 2)
        self.assertEqual(metrics["invalid_review_user_id"], 1)
        self.assertEqual(metrics["invalid_review_item_id"], 1)

        self.assertEqual(metrics["invalid_review_rating"], 1)
        self.assertEqual(metrics["invalid_review_timestamp"], 1)
        self.assertEqual(metrics["review_date_mismatch"], 1)

        self.assertEqual(metrics["exact_duplicate_groups"], 1)
        self.assertEqual(metrics["exact_duplicate_excess_rows"], 1)
        self.assertEqual(metrics["duplicate_user_item_pairs"], 1)

        self.assertEqual(metrics["duplicate_user_item_excess_interactions"], 2)
        self.assertEqual(metrics["metadata_rows"], 5)
        self.assertEqual(metrics["invalid_metadata_asin"], 1)

        self.assertEqual(metrics["metadata_without_text"], 1)
        self.assertEqual(metrics["duplicate_metadata_asin_groups"], 1)
        self.assertEqual(metrics["metadata_key_coverage"], 1.0)

        self.assertEqual(metrics["metadata_text_coverage"], 0.5)
        self.assertEqual(metrics["review_items_missing_metadata"], 0)
        self.assertEqual(metrics["review_items_missing_bm25_text"], 1)

        self.assertEqual(metrics["unused_metadata_items"], 1)
        self.assertEqual(report["errors"]["invalid_review_user_id_lines"], [4])
        self.assertEqual(len(report["sources"]["reviews"]["sha256"]), 64)

    def test_audit_includes_zero_valued_required_metrics(self) -> None:
        reviews = [self._review("u1", "i1", 5, 1_577_836_800, "01 1, 2020")]
        metadata = [{"asin": "i1", "title": "Title"}]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reviews_path = root / "reviews.jsonl"
            metadata_path = root / "metadata.jsonl"

            self._write_jsonl(reviews_path, reviews)
            self._write_jsonl(metadata_path, metadata)
            report = audit_dataset(
                reviews_path,
                metadata_path,
                audit_time=AUDIT_TIME,
            )

        self.assertEqual(report["metrics"]["invalid_review_rating"], 0)
        self.assertEqual(report["metrics"]["invalid_metadata_asin"], 0)
        self.assertEqual(report["metrics"]["review_date_mismatch"], 0)

    @staticmethod
    def _review(
        user_id: object,
        item_id: object,
        rating: object,
        timestamp: object,
        review_time: str,
    ) -> dict[str, object]:
        return {
            "reviewerID": user_id,
            "asin": item_id,
            "overall": rating,
            "unixReviewTime": timestamp,
            "reviewTime": review_time,
            "reviewText": "Useful review",
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


class PipelineIntegrationTests(unittest.TestCase):
    def test_normalize_supports_amazon_2023_schema(self) -> None:
        reviews = [
            {
                "user_id": "u1",
                "asin": "child-i1",
                "parent_asin": "i1",
                "rating": 5.0,
                "timestamp": 1_577_836_800_123,
                "title": "Great album",
                "text": "Played it all week",
            }
        ]
        metadata = [
            {
                "parent_asin": "i1",
                "title": "Album",
                "description": ["Description"],
                "features": ["Acoustic"],
                "categories": ["Digital Music"],
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reviews_path = root / "reviews.jsonl"
            metadata_path = root / "metadata.jsonl"

            output_dir = root / "output"

            self._write_jsonl(reviews_path, reviews)
            self._write_jsonl(metadata_path, metadata)
            stats = normalize_dataset(reviews_path, metadata_path, output_dir)

            interactions = read_parquet(output_dir / "interactions.parquet")
            catalog = read_parquet(output_dir / "catalog.parquet")

        self.assertEqual(stats["invalid_review_rows_removed"], 0)
        self.assertEqual(interactions[0]["user_id"], "u1")
        self.assertEqual(interactions[0]["item_id"], "i1")

        self.assertEqual(interactions[0]["timestamp"], 1_577_836_800)
        self.assertEqual(interactions[0]["summary"], "Great album")
        self.assertEqual(interactions[0]["review_text"], "Played it all week")

        self.assertEqual(catalog[0]["item_id"], "i1")
        self.assertEqual(
            catalog[0]["metadata_text"],
            "Album Description Acoustic Digital Music",
        )

    def test_normalize_and_deduplicate_write_expected_parquet(self) -> None:
        reviews = [
            self._review("u1", "i1", 5, 20, "<b>Later</b>"),
            self._review("u1", "i1", 4, 10, " Earlier &amp; better "),
            self._review("u1", "i2", 3, 15, "No metadata"),
            self._review("", "i3", 5, 12, "Invalid user"),
        ]
        metadata = [
            {"asin": "i1", "title": "Short"},
            {"asin": "i1", "description": ["Rich"], "feature": ["Acoustic"]},
            {"asin": "i2", "title": " "},
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reviews_path = root / "reviews.jsonl"
            metadata_path = root / "metadata.jsonl"

            output_dir = root / "all"

            self._write_jsonl(reviews_path, reviews)
            self._write_jsonl(metadata_path, metadata)
            stats = normalize_dataset(reviews_path, metadata_path, output_dir)

            interactions = read_parquet(output_dir / "interactions.parquet")
            catalog = read_parquet(output_dir / "catalog.parquet")

            dedup_stats = deduplicate_dataset(
                output_dir / "interactions.parquet",
                output_dir / "interactions_deduplicated.parquet",
            )
            deduplicated = read_parquet(
                output_dir / "interactions_deduplicated.parquet"
            )

            intersection_dir = root / "intersection"
            intersection_stats = normalize_dataset(
                reviews_path,
                metadata_path,
                intersection_dir,
                catalog_policy="metadata_text",
            )

        self.assertEqual(stats["invalid_review_rows_removed"], 1)
        self.assertEqual(len(interactions), 3)
        self.assertEqual(len(catalog), 2)

        self.assertEqual(catalog[0]["source_line"], 2)
        self.assertEqual(catalog[0]["metadata_text"], "Rich Acoustic")
        self.assertEqual(dedup_stats["removed_duplicate_interactions"], 1)

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual(deduplicated[0]["timestamp"], 10)
        self.assertEqual(deduplicated[0]["review_text"], "Earlier & better")

        self.assertEqual(intersection_stats["catalog_items_written"], 1)
        self.assertEqual(intersection_stats["interactions_written"], 2)

    def test_normalize_uses_configured_audit_time_and_writes_empty_schema(self) -> None:
        reviews = [self._review("u1", "i1", 5, 20, "Future review")]
        metadata = [{"asin": "i1", "title": "Title"}]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reviews_path = root / "reviews.jsonl"
            metadata_path = root / "metadata.jsonl"

            output_dir = root / "output"

            self._write_jsonl(reviews_path, reviews)
            self._write_jsonl(metadata_path, metadata)
            stats = normalize_dataset(
                reviews_path,
                metadata_path,
                output_dir,
                audit_time=datetime.fromtimestamp(15, timezone.utc),
            )

            table_rows = read_parquet(output_dir / "interactions.parquet")
            dedup_stats = deduplicate_dataset(
                output_dir / "interactions.parquet",
                output_dir / "interactions_deduplicated.parquet",
            )

        self.assertEqual(stats["invalid_review_rows_removed"], 1)
        self.assertEqual(table_rows, [])
        self.assertEqual(dedup_stats["output_rows"], 0)

    @staticmethod
    def _review(
        user_id: str,
        item_id: str,
        rating: int,
        timestamp: int,
        text: str,
    ) -> dict[str, object]:
        return {
            "reviewerID": user_id,
            "asin": item_id,
            "overall": rating,
            "unixReviewTime": timestamp,
            "reviewText": text,
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    unittest.main()
