from __future__ import annotations

import json
import unittest

from backend.recommender.llm_explanation import (
    ExplanationSource,
    generate_llm_explanation,
)
from backend.recommender.models import DecisionCertificate, EvidenceRecord


class LLMExplanationTests(unittest.TestCase):
    def test_generates_grounded_explanation_with_returned_citations(self) -> None:
        client = _Client(
            {
                "explanation": "Recommended because it matches your interest in space stories.",
                "citations": ["H1", "CERT1"],
            }
        )

        result = generate_llm_explanation(
            "c1",
            _certificate(),
            _evidence(),
            model="test-model",
            client=client,
        )

        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["mode"], "paec_certified")
        self.assertEqual(result["citations"], ["H1", "CERT1"])
        self.assertEqual(result["model"], "test-model")
        self.assertFalse(client.request["store"])
        self.assertEqual(client.request["model"], "test-model")
        self.assertIn(
            "never by opaque internal IDs",
            client.request["input"][0]["content"],
        )

    def test_abstains_without_any_grounded_evidence(self) -> None:
        result = generate_llm_explanation(
            "c1",
            None,
            (),
            model="test-model",
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["reason"], "no_grounded_history_evidence")

    def test_generates_retrieval_grounded_explanation_without_paec(self) -> None:
        client = _Client(
            {
                "explanation": "Recommended because it matches your space interest.",
                "citations": ["H1"],
            }
        )

        result = generate_llm_explanation(
            "c1",
            None,
            (),
            fallback_sources=(
                ExplanationSource("H1", "h1", "Space adventure", 0.8),
            ),
            candidate_evidence=("Space sequel",),
            model="test-model",
            client=client,
        )

        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["mode"], "retrieval_grounded")
        self.assertEqual(result["citations"], ["H1"])

    def test_falls_back_when_certificate_is_not_dual_valid(self) -> None:
        result = generate_llm_explanation(
            "c1",
            _certificate(dual_valid=False),
            _evidence(),
            fallback_sources=(
                ExplanationSource("H1", "h1", "Space adventure", 0.8),
            ),
            model="test-model",
            client=_Client(
                {
                    "explanation": "Grounded retrieval explanation.",
                    "citations": ["H1"],
                }
            ),
        )

        self.assertEqual(result["status"], "generated")
        self.assertEqual(result["mode"], "retrieval_grounded")

    def test_rejects_citations_not_present_in_the_evidence(self) -> None:
        client = _Client(
            {
                "explanation": "Unsupported explanation.",
                "citations": ["H1", "unknown"],
            }
        )

        result = generate_llm_explanation(
            "c1",
            _certificate(),
            _evidence(),
            model="test-model",
            client=client,
        )

        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["reason"], "invalid_citations")


def _certificate(*, dual_valid: bool = True) -> DecisionCertificate:
    return DecisionCertificate(
        coalition_source_ids=("H1",),
        source_count=1,
        target_item_id="c1",
        membership={"removal": False, "retention": True},
        margin={"removal": -0.1, "retention": 0.1},
        boundary_items={"removal": "c2", "retention": "c2"},
        candidate_delta={"removal": 0, "retention": 0},
        run_hashes={"removal": "r1", "retention": "r2"},
        artifact_manifest_hash="manifest-v1",
        config_hash="config-v1",
        removal_effect=0.4,
        retention_loss=0.1,
        strict_removal_valid=dual_valid,
        retention_valid=dual_valid,
        dual_valid=dual_valid,
        evaluable=True,
        invalid_reasons=(),
    )


def _evidence() -> tuple[EvidenceRecord, ...]:
    return (
        EvidenceRecord("H1", "history", "Space adventure", "H1"),
        EvidenceRecord("P1", "provenance", "Space sequel", "H1"),
        EvidenceRecord(
            "CERT1",
            "certificate",
            "Full-pipeline removal and retention certificate.",
        ),
    )


class _Client:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.request: dict[str, object] = {}
        self.responses = self

    def create(self, **kwargs: object) -> _Response:
        self.request = kwargs
        return _Response(json.dumps(self.payload))


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


if __name__ == "__main__":
    unittest.main()
