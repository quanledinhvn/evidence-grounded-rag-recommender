from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from backend.item_knn import ItemKNNBundle, ItemKNNConfig
from backend.recommender.cli import _build_parser, run_recommendation, write_json
from backend.recommender.profile_io import load_profile


class RecommenderProfileIOTests(unittest.TestCase):
    def test_loads_profile_cli_json_without_losing_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(_profile_payload()), encoding="utf-8")

            profile = load_profile(path)

        self.assertEqual(profile.user_id, "u1")
        self.assertEqual(profile.seen_item_ids, frozenset({"h1"}))
        self.assertEqual(profile.positive_signals[0].history_id, "H1")
        self.assertEqual(profile.positive_signals[0].text_fields[0].source_id, "catalog:h1:title")

    def test_rejects_profile_with_mismatched_schema(self) -> None:
        payload = _profile_payload()
        payload["schema_version"] = "unknown"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema"):
                load_profile(path)


class RecommenderCLITests(unittest.TestCase):
    def test_semantic_cli_switch_defaults_on_and_can_be_disabled(self) -> None:
        required = [
            "--profile", "profile.json",
            "--item-knn-bundle", "bundle",
            "--search-manifest", "manifest.json",
        ]

        self.assertTrue(_build_parser().parse_args(required).semantic)
        self.assertFalse(
            _build_parser().parse_args([*required, "--no-semantic"]).semantic
        )

    def test_llm_explanation_switch_is_opt_in(self) -> None:
        required = [
            "--profile", "profile.json",
            "--item-knn-bundle", "bundle",
            "--search-manifest", "manifest.json",
        ]

        self.assertFalse(_build_parser().parse_args(required).llm_explanation)
        self.assertTrue(
            _build_parser().parse_args([*required, "--llm-explanation"])
            .llm_explanation
        )

    def test_runs_frozen_three_branch_pipeline_without_seen_items(self) -> None:
        bundle = _bundle()
        gateway = _Gateway()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            manifest_path = root / "search-manifest.json"

            profile_path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"schema_version": "search-index-manifest-v2"}),
                encoding="utf-8",
            )

            with patch.object(ItemKNNBundle, "load", return_value=bundle):
                result = run_recommendation(
                    profile_path,
                    root / "bundle",
                    manifest_path,
                    gateway=gateway,
                    top_k=2,
                    pool_size=3,
                    include_paec=False,
                )

        ranking = result["ranking"]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(ranking), 2)
        self.assertNotIn("h1", {item["item_id"] for item in ranking})

        self.assertEqual(result["config"]["weights"], {
            "itemknn": 1.0 / 3.0,
            "bm25": 1.0 / 3.0,
            "semantic": 1.0 / 3.0,
        })
        self.assertTrue(result["semantic_enabled"])
        self.assertTrue(result["artifact_manifest_hash"])

    def test_can_disable_semantic_retrieval(self) -> None:
        bundle = _bundle()
        gateway = _Gateway()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            manifest_path = root / "search-manifest.json"

            profile_path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"schema_version": "search-index-manifest-v2"}),
                encoding="utf-8",
            )

            with patch.object(ItemKNNBundle, "load", return_value=bundle):
                result = run_recommendation(
                    profile_path,
                    root / "bundle",
                    manifest_path,
                    gateway=gateway,
                    top_k=2,
                    pool_size=3,
                    include_paec=False,
                    include_semantic=False,
                )

        self.assertEqual(result["config"]["weights"], {
            "itemknn": 0.5,
            "bm25": 0.5,
        })
        self.assertFalse(result["semantic_enabled"])
        self.assertEqual(gateway.semantic_calls, 0)

        for item in result["ranking"]:
            self.assertNotIn("semantic", item["branch_scores"])

    def test_passes_paec_certificate_to_llm_explanation(self) -> None:
        bundle = _bundle()
        gateway = _Gateway()
        llm_result = {
            "status": "generated",
            "text": "Grounded explanation.",
            "citations": ["H1"],
            "model": "test-model",
            "reason": None,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "profile.json"
            manifest_path = root / "search-manifest.json"

            profile_path.write_text(json.dumps(_profile_payload()), encoding="utf-8")
            manifest_path.write_text(
                json.dumps({"schema_version": "search-index-manifest-v2"}),
                encoding="utf-8",
            )

            with patch.object(ItemKNNBundle, "load", return_value=bundle):
                with patch(
                    "backend.recommender.cli.generate_llm_explanation",
                    return_value=llm_result,
                ) as generator:
                    result = run_recommendation(
                        profile_path,
                        root / "bundle",
                        manifest_path,
                        gateway=gateway,
                        top_k=2,
                        pool_size=3,
                        include_paec=False,
                        include_llm_explanation=True,
                        llm_model="test-model",
                    )

        self.assertEqual(result["llm_explanation"], llm_result)
        self.assertEqual(
            generator.call_args.args[0],
            result["ranking"][0]["item_id"],
        )
        self.assertIsNone(generator.call_args.args[1])
        self.assertEqual(generator.call_args.args[2], ())
        self.assertEqual(
            generator.call_args.kwargs["fallback_sources"][0].history_id,
            "H1",
        )
        self.assertEqual(generator.call_args.kwargs["model"], "test-model")

    def test_writes_deterministic_json_and_creates_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json(path, {"b": 2, "a": 1})

            content = path.read_text(encoding="utf-8")

        self.assertEqual(content, '{\n  "a": 1,\n  "b": 2\n}\n')


class _Gateway:
    def __init__(self) -> None:
        self.semantic_calls = 0

    def bm25_search(
        self,
        query: str,
        *,
        limit: int,
        excluded_item_ids: tuple[str, ...] = (),
        explain: bool = False,
    ) -> tuple[dict[str, object], ...]:
        return (
            {"item_id": "c2", "score": 2.0, "title": "Space sequel"},
        )

    def item_chunks(
        self,
        item_id: str,
        *,
        embedding_model: str,
        limit: int = 1000,
    ) -> tuple[dict[str, object], ...]:
        return ({"embedding": [1.0, 0.0], "raw_text": "space future"},)

    def semantic_search(
        self,
        embedding: tuple[float, ...],
        *,
        limit: int,
        excluded_item_ids: tuple[str, ...] = (),
    ) -> tuple[dict[str, object], ...]:
        self.semantic_calls += 1

        return (
            {"item_id": "c3", "score": 1.5, "raw_text": "future world"},
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
        csr_matrix((1, 4)),
        {"u1": 0},
        {"h1": 0, "c1": 1, "c2": 2, "c3": 3},
        {"u1": {"h1"}},
        {"u1"},
        [("c1", 3), ("c2", 2), ("c3", 1), ("h1", 1)],
        {"phase": "test", "artifact_hashes": {"similarity.npz": "abc"}},
    )


def _profile_payload() -> dict[str, object]:
    return {
        "schema_version": "user-profile-v1",
        "user_id": "u1",
        "as_of": 300,
        "phase": "test",
        "artifact_version": "artifact-v1",
        "seen_item_ids": ["h1"],
        "positive_signals": [
            {
                "history_id": "H1",
                "item_id": "h1",
                "rating": 5.0,
                "timestamp": 100,
                "sentiment": "positive",
                "weight": 1.0,
                "text_fields": [
                    {
                        "name": "title",
                        "text": "space future",
                        "source_id": "catalog:h1:title",
                    }
                ],
            }
        ],
        "negative_signals": [],
        "diagnostics": {
            "valid_interaction_count": 1,
            "duplicate_item_count": 0,
            "positive_count_before_limit": 1,
            "negative_count_before_limit": 0,
            "text_covered_positive_count": 1,
            "missing_catalog_item_count": 0,
            "warnings": [],
        },
    }


if __name__ == "__main__":
    unittest.main()
