from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace

from backend.profile import UserProfile
from backend.recommender.models import (
    PipelineConfig,
    ProvenanceAtom,
    RankedItem,
    RankingRun,
    RunMode,
)
from backend.recommender.retrievers import Retriever


class HybridPipeline:
    def __init__(
        self,
        retrievers: Mapping[str, Retriever],
        config: PipelineConfig,
        *,
        artifact_manifest_hash: str,
        fallback_item_ids: Iterable[str] = (),
    ) -> None:
        self.retrievers = dict(retrievers)
        self.config = config
        self.artifact_manifest_hash = artifact_manifest_hash
        self.fallback_item_ids = tuple(fallback_item_ids)

        if not artifact_manifest_hash.strip():
            raise ValueError("Artifact manifest hash must not be empty")

        if set(self.retrievers) != set(config.weights):
            raise ValueError("Retrievers must match configured fusion branches")

        if len(set(self.fallback_item_ids)) != len(self.fallback_item_ids):
            raise ValueError("Fallback item IDs must be unique")

        self.config_hash = _stable_hash(asdict(config))

    def run(
        self,
        profile: UserProfile,
        *,
        active_source_ids: frozenset[str] | None = None,
        mode: RunMode = "factual",
    ) -> RankingRun:
        profile_sources = frozenset(
            signal.history_id for signal in profile.positive_signals
        )
        active_sources = profile_sources if active_source_ids is None else active_source_ids
        if not active_sources <= profile_sources:
            raise ValueError("Active source IDs must belong to the profile")

        branch_results = {
            branch: retriever.retrieve(
                profile,
                active_sources,
                limit=self.config.pool_size,
            )
            for branch, retriever in sorted(self.retrievers.items())
        }
        candidate_ids = set().union(
            *(result.pool for result in branch_results.values())
        )
        candidate_ids.difference_update(profile.seen_item_ids)
        self._apply_fallback(candidate_ids, profile.seen_item_ids)

        ordered_candidates = tuple(sorted(candidate_ids))
        denominators = {
            branch: self.config.epsilon
            + sum(max(result.scores.get(item_id, 0.0), 0.0) for item_id in candidate_ids)
            for branch, result in branch_results.items()
        }
        fused_atoms = self._normalized_atoms(branch_results, denominators)
        ranked_items = self._rank_items(
            ordered_candidates,
            branch_results,
            denominators,
            fused_atoms,
        )

        invalid_reasons = self._invalid_reasons(
            profile,
            active_sources,
            branch_results,
            ranked_items,
            fused_atoms,
        )
        payload = {
            "mode": mode,
            "active_source_ids": sorted(active_sources),
            "seen_item_ids": sorted(profile.seen_item_ids),
            "ranking": [asdict(item) for item in ranked_items],
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "config_hash": self.config_hash,
        }

        return RankingRun(
            mode=mode,
            active_source_ids=active_sources,
            seen_item_ids=profile.seen_item_ids,
            candidate_ids=ordered_candidates,
            ranking=ranked_items,
            atoms=fused_atoms,
            artifact_manifest_hash=self.artifact_manifest_hash,
            config_hash=self.config_hash,
            run_hash=_stable_hash(payload),
            evaluable=not invalid_reasons,
            invalid_reasons=tuple(invalid_reasons),
        )

    def _apply_fallback(
        self,
        candidate_ids: set[str],
        seen_item_ids: frozenset[str],
    ) -> None:
        required = self.config.top_k + 1

        for item_id in self.fallback_item_ids:
            if len(candidate_ids) >= required:
                break

            if item_id not in seen_item_ids:
                candidate_ids.add(item_id)

    def _normalized_atoms(
        self,
        branch_results: Mapping[str, object],
        denominators: Mapping[str, float],
    ) -> tuple[ProvenanceAtom, ...]:
        atoms: list[ProvenanceAtom] = []

        for branch, result in branch_results.items():
            weight = self.config.weights[branch]
            denominator = denominators[branch]

            atoms.extend(
                replace(
                    atom,
                    fused_contribution=(
                        weight * max(atom.raw_contribution, 0.0) / denominator
                    ),
                )
                for atom in result.atoms
            )

        return tuple(
            sorted(atoms, key=lambda atom: (atom.candidate_id, atom.branch, atom.source_id))
        )

    def _rank_items(
        self,
        candidate_ids: tuple[str, ...],
        branch_results: Mapping[str, object],
        denominators: Mapping[str, float],
        atoms: tuple[ProvenanceAtom, ...],
    ) -> tuple[RankedItem, ...]:
        values: list[RankedItem] = []

        for item_id in candidate_ids:
            branch_scores = {
                branch: max(result.scores.get(item_id, 0.0), 0.0)
                for branch, result in branch_results.items()
            }
            fused_score = sum(
                self.config.weights[branch] * score / denominators[branch]
                for branch, score in branch_scores.items()
            )
            source_attribution: dict[str, float] = {}

            for atom in atoms:
                if atom.candidate_id != item_id:
                    continue

                source_attribution[atom.source_id] = (
                    source_attribution.get(atom.source_id, 0.0)
                    + atom.fused_contribution
                )

            values.append(
                RankedItem(
                    item_id=item_id,
                    rank=0,
                    fused_score=fused_score,
                    branch_scores=branch_scores,
                    source_attribution=source_attribution,
                )
            )

        values.sort(key=lambda item: (-item.fused_score, item.item_id))

        return tuple(replace(item, rank=index) for index, item in enumerate(values, 1))

    def _invalid_reasons(
        self,
        profile: UserProfile,
        active_sources: frozenset[str],
        branch_results: Mapping[str, object],
        ranking: tuple[RankedItem, ...],
        atoms: tuple[ProvenanceAtom, ...],
    ) -> list[str]:
        reasons: list[str] = []

        if not active_sources:
            reasons.append("empty_positive_history")

        if len(ranking) < self.config.top_k + 1:
            reasons.append("fewer_than_top_k_plus_one_candidates")

        if any(atom.source_id not in active_sources for atom in atoms):
            reasons.append("source_closure_violation")

        for result in branch_results.values():
            if any(atom.artifact_version != profile.artifact_version for atom in result.atoms):
                reasons.append("artifact_version_mismatch")
                break

        atom_totals: dict[str, float] = {}
        for atom in atoms:
            atom_totals[atom.candidate_id] = (
                atom_totals.get(atom.candidate_id, 0.0) + atom.fused_contribution
            )

        for item in ranking:
            residual = abs(item.fused_score - atom_totals.get(item.item_id, 0.0))
            if residual > self.config.identity_tolerance:
                reasons.append("score_identity_violation")
                break

        return reasons


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()

