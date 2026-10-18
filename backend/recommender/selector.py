from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

from backend.profile import UserProfile
from backend.recommender.intervention import InterventionHarness
from backend.recommender.models import RankingRun, SelectionResult


class PAECSelector:
    def __init__(
        self,
        harness: InterventionHarness,
        *,
        support_gate: Callable[[tuple[str, ...]], bool] | None = None,
    ) -> None:
        self.harness = harness
        self.support_gate = support_gate or (lambda coalition: True)

    def select(
        self,
        profile: UserProfile,
        factual_run: RankingRun,
        target_item_id: str,
    ) -> SelectionResult:
        source_ids = tuple(
            sorted(signal.history_id for signal in profile.positive_signals)
        )
        config = self.harness.pipeline.config
        exact = len(source_ids) <= config.exact_history_limit

        candidates = self._coalitions(
            source_ids,
            config.evidence_budget,
            exact,
            factual_run,
            target_item_id,
        )
        evaluated = 0
        valid = []

        for coalition in candidates:
            certificate = self.harness.evaluate(
                profile,
                factual_run,
                target_item_id,
                coalition,
            )
            evaluated += 1

            if certificate.dual_valid and self.support_gate(coalition):
                valid.append(certificate)

        if not valid:
            return SelectionResult("not_found", (), None, evaluated, exact)

        valid.sort(
            key=lambda certificate: (
                certificate.source_count,
                -certificate.removal_effect,
                certificate.retention_loss,
                certificate.coalition_source_ids,
            )
        )
        selected = valid[0]

        return SelectionResult(
            "selected",
            selected.coalition_source_ids,
            selected,
            evaluated,
            exact,
        )

    def _coalitions(
        self,
        source_ids: tuple[str, ...],
        budget: int,
        exact: bool,
        factual_run: RankingRun,
        target_item_id: str,
    ) -> tuple[tuple[str, ...], ...]:
        if exact:
            return tuple(
                coalition
                for size in range(1, min(budget, len(source_ids)) + 1)
                for coalition in combinations(source_ids, size)
            )

        target = next(
            item for item in factual_run.ranking if item.item_id == target_item_id
        )
        attribution = target.source_attribution
        beam: tuple[tuple[str, ...], ...] = ((),)
        selected: list[tuple[str, ...]] = []

        for _ in range(min(budget, len(source_ids))):
            expanded = {
                tuple(sorted((*coalition, source_id)))
                for coalition in beam
                for source_id in source_ids
                if source_id not in coalition
            }
            ordered = sorted(
                expanded,
                key=lambda coalition: (
                    -sum(attribution.get(source_id, 0.0) for source_id in coalition),
                    coalition,
                ),
            )
            beam = tuple(ordered[: self.harness.pipeline.config.beam_width])
            selected.extend(beam)

        return tuple(dict.fromkeys(selected))

