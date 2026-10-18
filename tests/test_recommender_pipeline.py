from __future__ import annotations

import math
import unittest
from dataclasses import replace

from backend.profile import (
    PreferenceSignal,
    ProfileDiagnostics,
    ProfileTextField,
    UserProfile,
)
from backend.recommender import (
    EvidenceRecord,
    FrozenContributionRetriever,
    HybridPipeline,
    InterventionHarness,
    PAECSelector,
    PipelineConfig,
    render_template,
    signed_margin,
    verify_explanation,
)


class RecommenderPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = self._profile()
        self.config = PipelineConfig(
            top_k=2,
            pool_size=10,
            evidence_budget=2,
            weights={"itemknn": 0.4, "bm25": 0.3, "semantic": 0.3},
            retention_tolerance=1.0,
        )
        contributions = {
            "H1": {"target": 10.0, "b": 2.0, "fallback": 0.1},
            "H2": {"target": 1.0, "b": 8.0, "c": 7.0},
        }
        retrievers = {
            branch: FrozenContributionRetriever(
                branch,
                contributions,
                "artifact-v1",
                history_spans={"H1": "science fiction", "H2": "future drama"},
                candidate_spans={"target": "science fiction adventure"},
            )
            for branch in ("itemknn", "bm25", "semantic")
        }
        self.pipeline = HybridPipeline(
            retrievers,
            self.config,
            artifact_manifest_hash="manifest-v1",
            fallback_item_ids=("fallback", "d", "e"),
        )

    def test_fuses_three_branches_and_preserves_score_identity(self) -> None:
        run = self.pipeline.run(self.profile)

        target = self._item(run, "target")
        atom_sum = sum(
            atom.fused_contribution
            for atom in run.atoms
            if atom.candidate_id == "target"
        )

        self.assertTrue(run.evaluable)
        self.assertTrue(math.isclose(target.fused_score, atom_sum, abs_tol=1e-12))
        self.assertTrue(math.isclose(sum(target.source_attribution.values()), atom_sum))

    def test_rerun_is_deterministic_and_uses_item_id_tie_break(self) -> None:
        first = self.pipeline.run(self.profile)
        second = self.pipeline.run(self.profile)

        self.assertEqual(first, second)
        self.assertEqual(first.run_hash, second.run_hash)
        self.assertEqual(first.ranking[0].item_id, "target")

    def test_source_closure_removes_every_descendant_atom(self) -> None:
        run = self.pipeline.run(
            self.profile,
            active_source_ids=frozenset({"H2"}),
            mode="removal",
        )

        self.assertTrue(run.atoms)
        self.assertEqual({atom.source_id for atom in run.atoms}, {"H2"})
        self.assertNotIn("fallback", run.candidate_ids)

    def test_intervention_keeps_factual_seen_ids(self) -> None:
        factual = self.pipeline.run(self.profile)
        harness = InterventionHarness(self.pipeline)

        certificate = harness.evaluate(self.profile, factual, "target", ("H1",))
        removal = self.pipeline.run(
            self.profile,
            active_source_ids=frozenset({"H2"}),
            mode="removal",
        )

        self.assertEqual(removal.seen_item_ids, factual.seen_item_ids)
        self.assertTrue(certificate.evaluable)

    def test_full_pipeline_regenerates_candidate_union(self) -> None:
        factual = self.pipeline.run(self.profile)
        removal = self.pipeline.run(
            self.profile,
            active_source_ids=frozenset({"H2"}),
            mode="removal",
        )

        self.assertIn("fallback", factual.candidate_ids)
        self.assertNotIn("fallback", removal.candidate_ids)
        self.assertNotEqual(factual.candidate_ids, removal.candidate_ids)

    def test_short_candidate_state_is_non_evaluable(self) -> None:
        contributions = {"H1": {"target": 10.0, "b": 2.0}}
        retrievers = {
            branch: FrozenContributionRetriever(
                branch,
                contributions,
                "artifact-v1",
            )
            for branch in ("itemknn", "bm25", "semantic")
        }
        pipeline = HybridPipeline(
            retrievers,
            self.config,
            artifact_manifest_hash="manifest-v1",
        )
        run = pipeline.run(
            self.profile,
            active_source_ids=frozenset({"H1"}),
            mode="retention",
        )

        self.assertFalse(run.evaluable)
        self.assertIn("fewer_than_top_k_plus_one_candidates", run.invalid_reasons)

    def test_empty_query_is_non_evaluable_even_when_fallback_is_large_enough(self) -> None:
        run = self.pipeline.run(
            self.profile,
            active_source_ids=frozenset(),
            mode="removal",
        )

        self.assertGreaterEqual(len(run.candidate_ids), self.config.top_k + 1)
        self.assertFalse(run.evaluable)
        self.assertIn("empty_positive_history", run.invalid_reasons)

    def test_signed_margin_uses_current_boundary(self) -> None:
        factual = self.pipeline.run(self.profile)
        removal = self.pipeline.run(
            self.profile,
            active_source_ids=frozenset({"H2"}),
            mode="removal",
        )

        factual_margin, factual_boundary, factual_member = signed_margin(
            factual,
            "target",
            2,
        )
        removal_margin, removal_boundary, removal_member = signed_margin(
            removal,
            "target",
            2,
        )

        self.assertTrue(factual_member)
        self.assertFalse(removal_member)
        self.assertGreater(factual_margin, 0.0)

        self.assertLess(removal_margin, 0.0)
        self.assertEqual(factual_boundary, "c")
        self.assertEqual(removal_boundary, "c")

    def test_certificate_records_dual_validity_hashes_and_deltas(self) -> None:
        factual = self.pipeline.run(self.profile)
        certificate = InterventionHarness(self.pipeline).evaluate(
            self.profile,
            factual,
            "target",
            ("H1",),
        )

        self.assertTrue(certificate.strict_removal_valid)
        self.assertTrue(certificate.retention_valid)
        self.assertTrue(certificate.dual_valid)

        self.assertEqual(certificate.run_hashes["factual"], factual.run_hash)
        self.assertGreater(certificate.candidate_delta["removal_symmetric_difference"], 0)
        self.assertEqual(certificate.artifact_manifest_hash, "manifest-v1")

    def test_certificate_rejects_artifact_or_config_mismatch(self) -> None:
        foreign = HybridPipeline(
            self.pipeline.retrievers,
            self.config,
            artifact_manifest_hash="other-manifest",
            fallback_item_ids=("fallback", "d", "e"),
        ).run(self.profile)
        certificate = InterventionHarness(self.pipeline).evaluate(
            self.profile,
            foreign,
            "target",
            ("H1",),
        )

        self.assertFalse(certificate.evaluable)
        self.assertIn("artifact_manifest_mismatch", certificate.invalid_reasons)

    def test_exact_selector_finds_smallest_supported_dual_valid_coalition(self) -> None:
        factual = self.pipeline.run(self.profile)
        selector = PAECSelector(
            InterventionHarness(self.pipeline),
            support_gate=lambda coalition: coalition == ("H1",),
        )

        selection = selector.select(self.profile, factual, "target")

        self.assertEqual(selection.status, "selected")
        self.assertEqual(selection.coalition_source_ids, ("H1",))
        self.assertEqual(selection.evaluated_coalitions, 2)
        self.assertTrue(selection.exact)
        self.assertTrue(selection.certificate and selection.certificate.dual_valid)

    def test_exact_selector_stops_after_first_valid_coalition_size(self) -> None:
        extra_signals = tuple(
            PreferenceSignal(
                f"H{index}",
                f"seen-{index}",
                4.0,
                200 + index,
                "positive",
                0.1,
                (),
            )
            for index in range(3, 5)
        )
        profile = replace(
            self.profile,
            positive_signals=self.profile.positive_signals + extra_signals,
        )
        config = replace(self.config, evidence_budget=3)

        pipeline = HybridPipeline(
            self.pipeline.retrievers,
            config,
            artifact_manifest_hash="manifest-v1",
            fallback_item_ids=("fallback", "d", "e"),
        )
        selector = PAECSelector(
            InterventionHarness(pipeline),
            support_gate=lambda coalition: coalition == ("H1", "H2"),
        )

        selection = selector.select(profile, pipeline.run(profile), "target")

        self.assertEqual(selection.coalition_source_ids, ("H1", "H2"))
        self.assertEqual(selection.evaluated_coalitions, 10)

    def test_long_history_uses_bounded_attribution_beam(self) -> None:
        extra_signals = tuple(
            PreferenceSignal(
                f"H{index}",
                f"seen-{index}",
                4.0,
                200 + index,
                "positive",
                0.1,
                (),
            )
            for index in range(3, 8)
        )
        profile = replace(
            self.profile,
            positive_signals=self.profile.positive_signals + extra_signals,
        )
        config = replace(self.config, exact_history_limit=2, beam_width=2)
        pipeline = HybridPipeline(
            self.pipeline.retrievers,
            config,
            artifact_manifest_hash="manifest-v1",
            fallback_item_ids=("fallback", "d", "e"),
        )

        factual = pipeline.run(profile)
        selection = PAECSelector(InterventionHarness(pipeline)).select(
            profile,
            factual,
            "target",
        )

        self.assertFalse(selection.exact)
        self.assertEqual(selection.coalition_source_ids, ("H1",))
        self.assertEqual(selection.evaluated_coalitions, 2)

    def test_support_failure_abstains_without_changing_ranking(self) -> None:
        factual = self.pipeline.run(self.profile)
        selector = PAECSelector(
            InterventionHarness(self.pipeline),
            support_gate=lambda coalition: False,
        )
        selection = selector.select(self.profile, factual, "target")

        explanation = render_template("target", selection.certificate, ())

        self.assertEqual(selection.status, "not_found")
        self.assertEqual(explanation.explanation_status, "abstained")
        self.assertEqual(self.pipeline.run(self.profile), factual)

    def test_template_and_verifier_use_locked_coalition_and_citations(self) -> None:
        factual = self.pipeline.run(self.profile)
        certificate = InterventionHarness(self.pipeline).evaluate(
            self.profile,
            factual,
            "target",
            ("H1",),
        )
        evidence = (
            EvidenceRecord("H1", "history", "Rated a science-fiction film", "H1"),
            EvidenceRecord("C1", "candidate", "Science-fiction adventure"),
            EvidenceRecord("Z1", "certificate", "Dual-valid pipeline rerun"),
        )

        explanation = render_template("target", certificate, evidence)

        self.assertEqual(explanation.explanation_status, "verified")
        self.assertIn("H1", explanation.citations)
        self.assertTrue(verify_explanation(explanation, certificate, evidence))

    @staticmethod
    def _item(run: object, item_id: str) -> object:
        return next(item for item in run.ranking if item.item_id == item_id)

    @staticmethod
    def _profile() -> UserProfile:
        signals = (
            PreferenceSignal(
                "H1",
                "seen-a",
                5.0,
                100,
                "positive",
                1.0,
                (ProfileTextField("title", "science fiction", "catalog:a:title"),),
            ),
            PreferenceSignal(
                "H2",
                "seen-b",
                4.0,
                200,
                "positive",
                0.8,
                (ProfileTextField("title", "future drama", "catalog:b:title"),),
            ),
        )
        diagnostics = ProfileDiagnostics(2, 0, 2, 0, 2, 0, ())

        return UserProfile(
            "user-profile-v1",
            "u1",
            300,
            "validation",
            "artifact-v1",
            frozenset({"seen-a", "seen-b"}),
            signals,
            (),
            diagnostics,
        )


if __name__ == "__main__":
    unittest.main()
