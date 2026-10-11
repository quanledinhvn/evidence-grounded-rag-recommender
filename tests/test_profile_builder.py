from __future__ import annotations

import math
import tempfile
import unittest

import json

from pathlib import Path

from backend.profile import (
    CatalogItem,
    InMemoryCatalogStore,
    InMemoryHistoryStore,
    Interaction,
    ParquetCatalogStore,
    ParquetHistoryStore,
    ProfileBuilder,
    ProfilePolicy,
    ProfileRequest,
)
from backend.profile.cli import main as profile_cli_main
from backend.data.io import write_parquet


DAY_MS = 86_400_000


class ProfileBuilderTests(unittest.TestCase):
    def test_filters_user_split_and_timestamp_before_building_seen(self) -> None:
        interactions = (
            self._interaction("u1", "i1", 5.0, 100, "r1", "train"),
            self._interaction("u1", "i2", 3.0, 200, "r2", "train"),
            self._interaction("u1", "i3", 5.0, 300, "r3", "validation"),

            self._interaction("u1", "i4", 5.0, 500, "r4", "train"),
            self._interaction("u1", "i5", 5.0, 50, "r5", "test"),
            self._interaction("u2", "i6", 5.0, 50, "r6", "train"),
        )
        builder = self._builder(interactions)

        profile = builder.build(
            self._request(as_of=500, phase="validation"),
            ProfilePolicy(allowed_splits=("train",)),
        )

        self.assertEqual(profile.seen_item_ids, frozenset({"i1", "i2"}))
        self.assertEqual(self._signal_ids(profile.positive_signals), ["i1"])
        self.assertEqual(profile.diagnostics.valid_interaction_count, 2)

    def test_latest_duplicate_controls_sentiment_but_seen_remains_complete(self) -> None:
        interactions = (
            self._interaction("u1", "i1", 5.0, 100, "r1", "train"),
            self._interaction("u1", "i1", 1.0, 200, "r2", "train"),
            self._interaction("u1", "i2", 3.0, 300, "r3", "train"),
        )
        builder = self._builder(interactions)

        profile = builder.build(
            self._request(as_of=400),
            ProfilePolicy(allowed_splits=("train",)),
        )

        self.assertEqual(profile.seen_item_ids, frozenset({"i1", "i2"}))
        self.assertEqual(profile.positive_signals, ())
        self.assertEqual(self._signal_ids(profile.negative_signals), ["i1"])

        self.assertEqual(profile.diagnostics.valid_interaction_count, 3)
        self.assertEqual(profile.diagnostics.duplicate_item_count, 1)

    def test_limits_signals_after_deterministic_recency_ordering(self) -> None:
        interactions = (
            self._interaction("u1", "i1", 5.0, 100, "r1", "train"),
            self._interaction("u1", "i2", 5.0, 300, "r2", "train"),
            self._interaction("u1", "i3", 4.0, 200, "r3", "train"),

            self._interaction("u1", "i4", 1.0, 250, "r4", "train"),
            self._interaction("u1", "i5", 2.0, 150, "r5", "train"),
        )
        builder = self._builder(interactions)
        policy = ProfilePolicy(
            allowed_splits=("train",),
            max_positive_signals=2,
            max_negative_signals=1,
        )

        profile = builder.build(self._request(as_of=400), policy)

        self.assertEqual(self._signal_ids(profile.positive_signals), ["i2", "i3"])
        self.assertEqual(self._signal_ids(profile.negative_signals), ["i4"])
        self.assertEqual(len(profile.seen_item_ids), 5)

        self.assertEqual(profile.diagnostics.positive_count_before_limit, 3)
        self.assertEqual(profile.diagnostics.negative_count_before_limit, 2)

    def test_computes_rating_and_half_life_weight(self) -> None:
        as_of = 10 * DAY_MS
        interactions = (
            self._interaction(
                "u1",
                "i1",
                5.0,
                as_of - DAY_MS,
                "r1",
                "train",
            ),
            self._interaction(
                "u1",
                "i2",
                4.0,
                as_of - DAY_MS,
                "r2",
                "train",
            ),
        )
        builder = self._builder(interactions)
        policy = ProfilePolicy(
            allowed_splits=("train",),
            recency_half_life_days=1.0,
        )

        profile = builder.build(self._request(as_of=as_of), policy)
        weights = {
            signal.item_id: signal.weight
            for signal in profile.positive_signals
        }

        self.assertTrue(math.isclose(weights["i1"], 0.5))
        self.assertTrue(math.isclose(weights["i2"], 0.25))

    def test_attaches_catalog_fields_and_reports_coverage(self) -> None:
        interactions = (
            self._interaction("u1", "i1", 5.0, 100, "r1", "train"),
            self._interaction("u1", "i2", 1.0, 90, "r2", "train"),
            self._interaction("u1", "missing", 4.0, 80, "r3", "train"),
        )
        catalog = {
            "i1": CatalogItem(
                item_id="i1",
                source_id="catalog:i1",
                title="Hidden Witness",
                subtitle="Collector Edition",
                main_category="Movies & TV",
                category="Mystery",
                store="Example Studio",
                details="Actors: Actor One",
            ),
            "i2": CatalogItem(
                item_id="i2",
                source_id="catalog:i2",
                description="Reality comedy",
            ),
        }
        builder = self._builder(interactions, catalog)

        profile = builder.build(
            self._request(as_of=200),
            ProfilePolicy(allowed_splits=("train",)),
        )

        first = profile.positive_signals[0]
        self.assertEqual(
            [field.name for field in first.text_fields],
            ["title", "subtitle", "main_category", "category", "details"],
        )
        self.assertEqual(first.text_fields[0].source_id, "catalog:i1:title")

        self.assertEqual(profile.diagnostics.text_covered_positive_count, 1)

        self.assertEqual(profile.diagnostics.missing_catalog_item_count, 1)
        self.assertEqual(
            profile.diagnostics.warnings,
            ("Catalog metadata is unavailable for missing.",),
        )

    def test_unknown_user_returns_an_empty_profile(self) -> None:
        builder = self._builder(())

        profile = builder.build(
            self._request(as_of=200),
            ProfilePolicy(allowed_splits=("train",)),
        )

        self.assertEqual(profile.seen_item_ids, frozenset())
        self.assertEqual(profile.positive_signals, ())
        self.assertEqual(profile.negative_signals, ())

        self.assertEqual(profile.diagnostics.valid_interaction_count, 0)

    def test_rejects_phase_policy_and_artifact_mismatches(self) -> None:
        builder = self._builder(())

        with self.assertRaisesRegex(ValueError, "allowed_splits"):
            builder.build(
                self._request(phase="validation"),
                ProfilePolicy(allowed_splits=("train", "validation")),
            )

        with self.assertRaisesRegex(ValueError, "artifact_version"):
            builder.build(
                self._request(artifact_version="other"),
                ProfilePolicy(allowed_splits=("train",)),
            )

    def test_rejects_invalid_request_and_policy_values(self) -> None:
        builder = self._builder(())
        valid_request = self._request()

        invalid_requests = (
            ProfileRequest("", 500, "validation", "test-v1"),
            ProfileRequest("u1", 0, "validation", "test-v1"),
        )
        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(ValueError):
                    builder.build(
                        request,
                        ProfilePolicy(allowed_splits=("train",)),
                    )

        invalid_policies = (
            ProfilePolicy(("train",), max_positive_signals=0),
            ProfilePolicy(("train",), max_negative_signals=-1),
            ProfilePolicy(("train",), recency_half_life_days=0.0),
            ProfilePolicy(("train",), positive_threshold=2.0),
        )
        for policy in invalid_policies:
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError):
                    builder.build(valid_request, policy)

    def test_builds_from_project_parquet_artifacts(self) -> None:
        interaction_rows = [
            self._row("u1", "i1", 5.0, 100, "r1", "train"),
            self._row("u1", "i2", 4.0, 200, "r2", "validation"),
            self._row("u1", "i3", 5.0, 300, "r3", "test"),

            self._row("u1", "i4", 5.0, 400, "r4", "train"),
            self._row("u1", "i6", 3.0, 50, "r6", "train"),
            self._row("u2", "i5", 5.0, 50, "r5", "train"),
        ]
        catalog_rows = [
            self._catalog_row("i1", "First", "Mystery"),
            self._catalog_row("i2", "Second", "Drama"),
            self._catalog_row("i6", "", "", source_id=None),
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path = root / "splits.parquet"
            catalog_path = root / "catalog.parquet"

            write_parquet(splits_path, interaction_rows)
            write_parquet(catalog_path, catalog_rows)

            builder = ProfileBuilder(
                ParquetHistoryStore("test-v1", splits_path),
                ParquetCatalogStore("test-v1", catalog_path),
            )
            profile = builder.build(
                self._request(as_of=400, phase="test"),
                ProfilePolicy(allowed_splits=("train", "validation")),
            )

        self.assertEqual(profile.seen_item_ids, frozenset({"i1", "i2", "i6"}))
        self.assertEqual(self._signal_ids(profile.positive_signals), ["i2", "i1"])
        self.assertEqual(profile.positive_signals[0].text_fields[0].text, "Second")

        self.assertEqual(profile.diagnostics.missing_catalog_item_count, 1)
        self.assertEqual(
            profile.diagnostics.warnings,
            ("Catalog metadata is unavailable for i6.",),
        )

    def test_cli_builds_a_json_profile_from_parquet_artifacts(self) -> None:
        interaction_rows = [
            self._row("u1", "i1", 5.0, 100, "r1", "train"),
            self._row("u1", "i2", 1.0, 200, "r2", "validation"),
        ]
        catalog_rows = [self._catalog_row("i1", "First", "Mystery")]

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            splits_path = root / "splits.parquet"
            catalog_path = root / "catalog.parquet"
            output_path = root / "profile.json"

            write_parquet(splits_path, interaction_rows)
            write_parquet(catalog_path, catalog_rows)
            result = profile_cli_main(
                [
                    "build",
                    "--splits",
                    str(splits_path),
                    "--catalog",
                    str(catalog_path),
                    "--user-id",
                    "u1",
                    "--as-of",
                    "300",
                    "--phase",
                    "test",
                    "--artifact-version",
                    "test-v1",
                    "--output",
                    str(output_path),
                ]
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result, 0)
        self.assertEqual(payload["user_id"], "u1")
        self.assertEqual(payload["seen_item_ids"], ["i1", "i2"])
        self.assertEqual(payload["positive_signals"][0]["item_id"], "i1")
        self.assertEqual(payload["negative_signals"][0]["item_id"], "i2")

    def _builder(
        self,
        interactions: tuple[Interaction, ...],
        catalog: dict[str, CatalogItem] | None = None,
    ) -> ProfileBuilder:
        history_store = InMemoryHistoryStore("test-v1", interactions)
        catalog_store = InMemoryCatalogStore("test-v1", catalog or {})

        return ProfileBuilder(history_store, catalog_store)

    def _request(
        self,
        *,
        as_of: int = 500,
        phase: str = "validation",
        artifact_version: str = "test-v1",
    ) -> ProfileRequest:
        return ProfileRequest(
            user_id="u1",
            as_of=as_of,
            phase=phase,  # type: ignore[arg-type]
            artifact_version=artifact_version,
        )

    @staticmethod
    def _interaction(
        user_id: str,
        item_id: str,
        rating: float,
        timestamp: int,
        review_id: str,
        split: str,
    ) -> Interaction:
        return Interaction(
            user_id=user_id,
            item_id=item_id,
            rating=rating,
            timestamp=timestamp,
            review_id=review_id,
            split=split,
        )

    @staticmethod
    def _row(
        user_id: str,
        item_id: str,
        rating: float,
        timestamp: int,
        review_id: str,
        split: str,
    ) -> dict[str, object]:
        return {
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "timestamp": timestamp,
            "review_id": review_id,
            "split": split,
        }

    @staticmethod
    def _catalog_row(
        item_id: str,
        title: str,
        category: str,
        *,
        source_id: str | None = "generated",
    ) -> dict[str, object]:
        return {
            "item_id": item_id,
            "source_id": source_id and f"catalog:{item_id}",
            "title": title,
            "category": category,
            "feature": "",
            "description": "",
        }

    @staticmethod
    def _signal_ids(signals: tuple[object, ...]) -> list[str]:
        return [str(getattr(signal, "item_id")) for signal in signals]


if __name__ == "__main__":
    unittest.main()
