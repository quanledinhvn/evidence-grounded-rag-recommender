from __future__ import annotations

import json
import os
import tempfile
import unittest

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.search.cli import _load_project_env, create_opensearch_client, main
from backend.search.config import SearchSettings


class _FakeIndices:
    def refresh(self, *, index: str) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.indices = _FakeIndices()

    def info(self) -> dict[str, object]:
        return {
            "cluster_name": "it4613-search",
            "version": {"number": "3.9.0"},
        }


class _FakeGateway:
    item_index = "items-v1"
    chunk_index = "chunks-v1"

    def __init__(self, *_: object, **__: object) -> None:
        self.ensure_count = 0

    def ensure_indices(self) -> None:
        self.ensure_count += 1


class SearchCliTests(unittest.TestCase):
    def test_loads_dotenv_without_overriding_exported_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "OPENAI_API_KEY=from-file\n"
                "OPENSEARCH_ITEM_INDEX=from-file-items\n",
                encoding="utf-8",
            )

            environ = {
                "OPENAI_API_KEY": "from-shell",
            }
            with patch.dict(os.environ, environ, clear=True):
                _load_project_env(dotenv_path)

                self.assertEqual(os.environ["OPENAI_API_KEY"], "from-shell")
                self.assertEqual(
                    os.environ["OPENSEARCH_ITEM_INDEX"],
                    "from-file-items",
                )

    def test_plan_catalog_does_not_create_external_clients(self) -> None:
        result = SimpleNamespace(
            item_count=2,
            text_covered_item_count=1,
            chunk_count=3,
            word_count=8,
            catalog_sha256="abc",
        )
        output = StringIO()

        with patch("backend.search.cli.plan_catalog", return_value=result) as plan:
            with patch("backend.search.cli.create_opensearch_client") as create_client:
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "plan-catalog",
                            "--catalog",
                            "catalog.parquet",
                            "--max-words",
                            "80",
                            "--overlap-words",
                            "10",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        create_client.assert_not_called()

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["word_count"], 8)

        policy = plan.call_args.kwargs["chunking_policy"]
        self.assertEqual((policy.max_words, policy.overlap_words), (80, 10))

    def test_health_reports_cluster_and_contract(self) -> None:
        output = StringIO()

        with patch("backend.search.cli.create_opensearch_client", return_value=_FakeClient()):
            with redirect_stdout(output):
                exit_code = main(["health"])

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["cluster_name"], "it4613-search")

        self.assertEqual(payload["opensearch_version"], "3.9.0")
        self.assertEqual(payload["embedding_dimensions"], 768)

    def test_create_indices_uses_configured_gateway(self) -> None:
        gateway = _FakeGateway()
        output = StringIO()

        with patch("backend.search.cli.create_opensearch_client", return_value=_FakeClient()):
            with patch("backend.search.cli.OpenSearchGateway", return_value=gateway):
                with redirect_stdout(output):
                    exit_code = main(["create-indices"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(gateway.ensure_count, 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "ready")

    def test_index_catalog_wires_manifest_and_chunking_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.parquet"
            manifest = root / "manifest.json"
            catalog.touch()

            result = SimpleNamespace(
                item_count=2,
                chunk_count=3,
                embedded_chunk_count=2,
                reused_chunk_count=1,
                catalog_sha256="abc",
                manifest_path=manifest,
            )
            output = StringIO()

            with patch("backend.search.cli.create_opensearch_client", return_value=_FakeClient()):
                with patch("backend.search.cli.OpenSearchGateway", return_value=_FakeGateway()):
                    with patch("backend.search.cli.index_catalog", return_value=result) as index:
                        with redirect_stdout(output):
                            exit_code = main(
                                [
                                    "index-catalog",
                                    "--catalog",
                                    str(catalog),
                                    "--manifest",
                                    str(manifest),
                                    "--max-words",
                                    "80",
                                    "--overlap-words",
                                    "10",
                                ]
                            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["chunk_count"], 3)
        self.assertEqual(payload["embedded_chunk_count"], 2)
        self.assertEqual(payload["reused_chunk_count"], 1)

        policy = index.call_args.kwargs["chunking_policy"]
        self.assertEqual((policy.max_words, policy.overlap_words), (80, 10))

    def test_requires_complete_basic_auth_pair(self) -> None:
        settings = SearchSettings()

        with self.assertRaisesRegex(ValueError, "both"):
            create_opensearch_client(
                settings,
                {"OPENSEARCH_USERNAME": "admin"},
            )


if __name__ == "__main__":
    unittest.main()
