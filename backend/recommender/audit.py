from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

from backend.profile import UserProfile
from backend.recommender.intervention import InterventionHarness
from backend.recommender.models import RankingRun

BaselineName = Literal["contribution", "recency", "semantic", "random", "leave_one_out"]


@dataclass(frozen=True)
class SourceInfluence:
    source_id: str
    attribution: float
    removal_effect: float

    evaluable: bool
    candidate_delta: int


@dataclass(frozen=True)
class AttributionInterventionAudit:
    target_item_id: str
    sources: tuple[SourceInfluence, ...]
    spearman_correlation: float

    top_q_overlap: float
    non_evaluable_rate: float


def select_baseline(
    name: BaselineName,
    profile: UserProfile,
    factual_run: RankingRun,
    target_item_id: str,
    *,
    budget: int,
    harness: InterventionHarness | None = None,
    seed: int = 0,
) -> tuple[str, ...]:
    if isinstance(budget, bool) or budget <= 0:
        raise ValueError("Baseline budget must be positive")

    signals = {signal.history_id: signal for signal in profile.positive_signals}
    source_ids = tuple(sorted(signals))
    if not source_ids:
        return ()

    target = next(
        (item for item in factual_run.ranking if item.item_id == target_item_id),
        None,
    )
    if target is None:
        raise ValueError("Baseline target must belong to the factual ranking")

    if name == "contribution":
        ordered = sorted(
            source_ids,
            key=lambda source_id: (
                -target.source_attribution.get(source_id, 0.0),
                source_id,
            ),
        )
    elif name == "recency":
        ordered = sorted(
            source_ids,
            key=lambda source_id: (-signals[source_id].timestamp, source_id),
        )
    elif name == "semantic":
        semantic_values = {
            source_id: sum(
                atom.fused_contribution
                for atom in factual_run.atoms
                if atom.candidate_id == target_item_id
                and atom.branch == "semantic"
                and atom.source_id == source_id
            )
            for source_id in source_ids
        }
        ordered = sorted(
            source_ids,
            key=lambda source_id: (-semantic_values[source_id], source_id),
        )
    elif name == "random":
        ordered = list(source_ids)
        random.Random(seed).shuffle(ordered)
    elif name == "leave_one_out":
        if harness is None:
            raise ValueError("leave_one_out baseline requires an intervention harness")

        effects = {
            source_id: harness.evaluate(
                profile,
                factual_run,
                target_item_id,
                (source_id,),
            ).removal_effect
            for source_id in source_ids
        }
        ordered = sorted(
            source_ids,
            key=lambda source_id: (-effects[source_id], source_id),
        )
    else:
        raise ValueError(f"Unsupported baseline: {name}")

    return tuple(ordered[: min(budget, len(ordered))])


def audit_attribution_intervention(
    profile: UserProfile,
    factual_run: RankingRun,
    target_item_id: str,
    harness: InterventionHarness,
    *,
    top_q: int = 3,
) -> AttributionInterventionAudit:
    if isinstance(top_q, bool) or top_q <= 0:
        raise ValueError("top_q must be positive")

    target = next(
        (item for item in factual_run.ranking if item.item_id == target_item_id),
        None,
    )
    if target is None:
        raise ValueError("Audit target must belong to the factual ranking")

    influences: list[SourceInfluence] = []
    for source_id in sorted(signal.history_id for signal in profile.positive_signals):
        certificate = harness.evaluate(
            profile,
            factual_run,
            target_item_id,
            (source_id,),
        )
        influences.append(
            SourceInfluence(
                source_id=source_id,
                attribution=target.source_attribution.get(source_id, 0.0),
                removal_effect=certificate.removal_effect,
                evaluable=certificate.evaluable,
                candidate_delta=certificate.candidate_delta[
                    "removal_symmetric_difference"
                ],
            )
        )

    attribution_values = [value.attribution for value in influences]
    effect_values = [value.removal_effect for value in influences]
    correlation = _spearman(attribution_values, effect_values)

    limit = min(top_q, len(influences))
    attribution_top = {
        value.source_id
        for value in sorted(
            influences,
            key=lambda value: (-value.attribution, value.source_id),
        )[:limit]
    }
    effect_top = {
        value.source_id
        for value in sorted(
            influences,
            key=lambda value: (-value.removal_effect, value.source_id),
        )[:limit]
    }
    overlap = len(attribution_top & effect_top) / max(1, limit)
    non_evaluable = sum(not value.evaluable for value in influences) / max(
        1,
        len(influences),
    )

    return AttributionInterventionAudit(
        target_item_id,
        tuple(influences),
        correlation,
        overlap,
        non_evaluable,
    )


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0

    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)

    covariance = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0

    return covariance / (left_scale * right_scale)


def _ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    start = 0

    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1

        average_rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = average_rank

        start = end

    return ranks
