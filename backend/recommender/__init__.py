"""Pipeline-aware recommendation and evidence coalition evaluation."""

from backend.recommender.audit import (
    AttributionInterventionAudit,
    SourceInfluence,
    audit_attribution_intervention,
    select_baseline,
)
from backend.recommender.intervention import InterventionHarness, signed_margin
from backend.recommender.models import (
    BranchResult,
    DecisionCertificate,
    EvidenceRecord,
    Explanation,
    PipelineConfig,
    ProvenanceAtom,
    RankedItem,
    RankingRun,
    SelectionResult,
)
from backend.recommender.opensearch_retrievers import (
    OpenSearchBM25ContributionRetriever,
    OpenSearchSemanticContributionRetriever,
)
from backend.recommender.pipeline import HybridPipeline
from backend.recommender.rendering import render_template, verify_explanation
from backend.recommender.retrievers import (
    BM25ContributionRetriever,
    FrozenContributionRetriever,
    ItemKNNContributionRetriever,
    Retriever,
    SemanticContributionRetriever,
)
from backend.recommender.selector import PAECSelector

__all__ = [
    "BranchResult",
    "BM25ContributionRetriever",
    "AttributionInterventionAudit",
    "DecisionCertificate",
    "EvidenceRecord",
    "Explanation",
    "FrozenContributionRetriever",
    "HybridPipeline",
    "InterventionHarness",
    "ItemKNNContributionRetriever",
    "OpenSearchBM25ContributionRetriever",
    "OpenSearchSemanticContributionRetriever",
    "PAECSelector",
    "PipelineConfig",
    "ProvenanceAtom",
    "RankedItem",
    "RankingRun",
    "Retriever",
    "SelectionResult",
    "SemanticContributionRetriever",
    "SourceInfluence",
    "audit_attribution_intervention",
    "render_template",
    "signed_margin",
    "select_baseline",
    "verify_explanation",
]
