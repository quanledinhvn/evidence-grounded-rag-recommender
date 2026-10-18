from __future__ import annotations

from backend.profile import UserProfile
from backend.recommender.models import DecisionCertificate, RankingRun
from backend.recommender.pipeline import HybridPipeline


class InterventionHarness:
    def __init__(self, pipeline: HybridPipeline) -> None:
        self.pipeline = pipeline

    def evaluate(
        self,
        profile: UserProfile,
        factual_run: RankingRun,
        target_item_id: str,
        coalition_source_ids: tuple[str, ...],
    ) -> DecisionCertificate:
        all_sources = frozenset(
            signal.history_id for signal in profile.positive_signals
        )
        coalition = tuple(sorted(set(coalition_source_ids)))
        coalition_set = frozenset(coalition)
        if not coalition or not coalition_set <= all_sources:
            raise ValueError("Coalition must contain known profile source IDs")

        removal = self.pipeline.run(
            profile,
            active_source_ids=all_sources - coalition_set,
            mode="removal",
        )
        retention = self.pipeline.run(
            profile,
            active_source_ids=coalition_set,
            mode="retention",
        )

        top_k = self.pipeline.config.top_k
        factual_margin, factual_boundary, factual_member = signed_margin(
            factual_run,
            target_item_id,
            top_k,
        )
        removal_margin, removal_boundary, removal_member = signed_margin(
            removal,
            target_item_id,
            top_k,
        )
        retention_margin, retention_boundary, retention_member = signed_margin(
            retention,
            target_item_id,
            top_k,
        )

        invalid_reasons = list(factual_run.invalid_reasons)
        invalid_reasons.extend(removal.invalid_reasons)
        invalid_reasons.extend(retention.invalid_reasons)

        if factual_run.artifact_manifest_hash != self.pipeline.artifact_manifest_hash:
            invalid_reasons.append("artifact_manifest_mismatch")

        if factual_run.config_hash != self.pipeline.config_hash:
            invalid_reasons.append("config_hash_mismatch")

        if factual_run.seen_item_ids != removal.seen_item_ids:
            invalid_reasons.append("seen_item_ids_changed")

        if factual_run.seen_item_ids != retention.seen_item_ids:
            invalid_reasons.append("seen_item_ids_changed")

        evaluable = not invalid_reasons
        removal_effect = factual_margin - removal_margin
        retention_loss = abs(factual_margin - retention_margin)

        strict_removal = evaluable and factual_member and not removal_member
        retention_valid = (
            evaluable
            and retention_member
            and retention_loss <= self.pipeline.config.retention_tolerance
        )

        return DecisionCertificate(
            coalition_source_ids=coalition,
            source_count=len(coalition),
            target_item_id=target_item_id,
            membership={
                "factual": factual_member,
                "removal": removal_member,
                "retention": retention_member,
            },
            margin={
                "factual": factual_margin,
                "removal": removal_margin,
                "retention": retention_margin,
            },
            boundary_items={
                "factual": factual_boundary,
                "removal": removal_boundary,
                "retention": retention_boundary,
            },
            candidate_delta={
                "removal_symmetric_difference": len(
                    set(factual_run.candidate_ids) ^ set(removal.candidate_ids)
                ),
                "retention_symmetric_difference": len(
                    set(factual_run.candidate_ids) ^ set(retention.candidate_ids)
                ),
            },
            run_hashes={
                "factual": factual_run.run_hash,
                "removal": removal.run_hash,
                "retention": retention.run_hash,
            },
            artifact_manifest_hash=factual_run.artifact_manifest_hash,
            config_hash=factual_run.config_hash,
            removal_effect=removal_effect,
            retention_loss=retention_loss,
            strict_removal_valid=strict_removal,
            retention_valid=retention_valid,
            dual_valid=strict_removal and retention_valid,
            evaluable=evaluable,
            invalid_reasons=tuple(dict.fromkeys(invalid_reasons)),
        )


def signed_margin(
    run: RankingRun,
    target_item_id: str,
    top_k: int,
) -> tuple[float, str | None, bool]:
    if isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be positive")

    by_id = {item.item_id: item for item in run.ranking}
    target = by_id.get(target_item_id)
    member = target is not None and target.rank <= top_k

    if member:
        boundary = run.ranking[top_k] if len(run.ranking) > top_k else None
    else:
        boundary = run.ranking[top_k - 1] if len(run.ranking) >= top_k else None

    target_score = 0.0 if target is None else target.fused_score
    boundary_score = 0.0 if boundary is None else boundary.fused_score
    boundary_id = None if boundary is None else boundary.item_id

    return target_score - boundary_score, boundary_id, member

