from __future__ import annotations

import unittest

from tests.test_recommender_pipeline import RecommenderPipelineTests

from backend.recommender import (
    InterventionHarness,
    audit_attribution_intervention,
    select_baseline,
)


class RecommenderAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        fixture = RecommenderPipelineTests()
        fixture.setUp()

        self.profile = fixture.profile
        self.pipeline = fixture.pipeline

    def test_baselines_share_target_and_raw_source_budget(self) -> None:
        factual = self.pipeline.run(self.profile)
        harness = InterventionHarness(self.pipeline)

        for name in (
            "contribution",
            "recency",
            "semantic",
            "random",
            "leave_one_out",
        ):
            with self.subTest(name=name):
                coalition = select_baseline(
                    name,
                    self.profile,
                    factual,
                    "target",
                    budget=1,
                    harness=harness,
                    seed=7,
                )

                self.assertEqual(len(coalition), 1)
                self.assertIn(coalition[0], {"H1", "H2"})

    def test_gap_audit_reports_attribution_and_full_pipeline_effect(self) -> None:
        factual = self.pipeline.run(self.profile)
        audit = audit_attribution_intervention(
            self.profile,
            factual,
            "target",
            InterventionHarness(self.pipeline),
            top_q=1,
        )

        self.assertEqual(len(audit.sources), 2)
        self.assertEqual(audit.sources[0].source_id, "H1")
        self.assertGreater(audit.sources[0].attribution, audit.sources[1].attribution)

        self.assertGreaterEqual(audit.top_q_overlap, 0.0)
        self.assertLessEqual(audit.top_q_overlap, 1.0)


if __name__ == "__main__":
    unittest.main()
