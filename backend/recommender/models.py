from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BranchName = Literal["itemknn", "bm25", "semantic"]
RunMode = Literal["factual", "removal", "retention"]


@dataclass(frozen=True)
class PipelineConfig:
    top_k: int
    pool_size: int
    evidence_budget: int

    weights: dict[BranchName, float]
    epsilon: float = 1e-12
    identity_tolerance: float = 1e-9

    retention_tolerance: float = 0.05
    exact_history_limit: int = 12
    beam_width: int = 16

    def __post_init__(self) -> None:
        integer_values = (
            self.top_k,
            self.pool_size,
            self.evidence_budget,
            self.exact_history_limit,
            self.beam_width,
        )
        if any(isinstance(value, bool) or value <= 0 for value in integer_values):
            raise ValueError("Pipeline integer settings must be positive")

        required = {"itemknn", "bm25"}
        allowed = required | {"semantic"}
        configured = set(self.weights)

        if not required <= configured or not configured <= allowed:
            raise ValueError(
                "Fusion weights must define itemknn and bm25, with optional semantic"
            )

        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("Fusion weights must be non-negative")

        if abs(sum(self.weights.values()) - 1.0) > 1e-12:
            raise ValueError("Fusion weights must sum to one")

        if self.epsilon <= 0.0 or self.identity_tolerance < 0.0:
            raise ValueError("Pipeline tolerances are invalid")

        if self.retention_tolerance < 0.0:
            raise ValueError("Retention tolerance must be non-negative")


@dataclass(frozen=True)
class ProvenanceAtom:
    branch: BranchName
    source_id: str
    candidate_id: str

    raw_contribution: float
    fused_contribution: float = 0.0
    history_span: str = ""

    candidate_span: str = ""
    artifact_version: str = ""


@dataclass(frozen=True)
class BranchResult:
    branch: BranchName
    scores: dict[str, float]
    atoms: tuple[ProvenanceAtom, ...]

    pool: tuple[str, ...]


@dataclass(frozen=True)
class RankedItem:
    item_id: str
    rank: int
    fused_score: float

    branch_scores: dict[BranchName, float]
    source_attribution: dict[str, float]


@dataclass(frozen=True)
class RankingRun:
    mode: RunMode
    active_source_ids: frozenset[str]
    seen_item_ids: frozenset[str]

    candidate_ids: tuple[str, ...]
    ranking: tuple[RankedItem, ...]
    atoms: tuple[ProvenanceAtom, ...]

    artifact_manifest_hash: str
    config_hash: str
    run_hash: str

    evaluable: bool
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class DecisionCertificate:
    coalition_source_ids: tuple[str, ...]
    source_count: int
    target_item_id: str

    membership: dict[str, bool]
    margin: dict[str, float]
    boundary_items: dict[str, str | None]

    candidate_delta: dict[str, int]
    run_hashes: dict[str, str]
    artifact_manifest_hash: str

    config_hash: str
    removal_effect: float
    retention_loss: float

    strict_removal_valid: bool
    retention_valid: bool
    dual_valid: bool

    evaluable: bool
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class SelectionResult:
    status: Literal["selected", "not_found"]
    coalition_source_ids: tuple[str, ...]
    certificate: DecisionCertificate | None

    evaluated_coalitions: int
    exact: bool


@dataclass(frozen=True)
class EvidenceRecord:
    citation_id: str
    kind: Literal["history", "candidate", "provenance", "semantic", "certificate"]
    text: str

    source_id: str = ""


@dataclass(frozen=True)
class Explanation:
    item_id: str
    explanation_status: Literal["verified", "mechanistic", "abstained"]
    text: str

    citations: tuple[str, ...]
    abstain_reason: str | None

