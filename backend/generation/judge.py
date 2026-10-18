"""LLM judge: is each claim supported by the sources of its item and the user's history?

Labels (fixed rubric, one per claim):
- supported   : every factual statement is stated in, or directly implied by, the sources;
- partial     : part of it is supported, part is not;
- unsupported : it states something the sources do not say, or contradicts them.

The judge sees *all* sources of the item, not only the cited ones, so explanations with and
without citations are graded the same way. A stronger model than the generator is used
(llm.JUDGE_MODEL); 30-50 judged claims are also checked by hand.
"""

from __future__ import annotations

import json

from typing import Any

from backend.generation.sources import DATA_RULE, history_lines, item_sources, shorten


LABELS = {"supported", "partial", "unsupported"}

SCHEMA = {
    "type": "OBJECT",
    "properties": {"labels": {"type": "ARRAY", "items": {"type": "STRING", "enum": sorted(LABELS)}}},
    "required": ["labels"],
}


def judge_prompt(package: dict[str, Any], explained: dict[str, list[dict[str, Any]]]) -> str:
    blocks, number = [], 0

    for n, (item_id, claims) in enumerate(explained.items(), 1):
        blocks += [f"Item {n} sources:", *[f"- {shorten(entry['text'], 80)}" for _, entry in item_sources(package, item_id)]]
        blocks.append(f"Claims about item {n}:")

        for claim in claims:
            number += 1
            blocks.append(f"{number}. {claim['text']}")

        blocks.append("")

    return "\n".join(
        [
            "You check explanations of movie recommendations against their sources.",
            DATA_RULE,
            "",
            "The user's history (true facts about the user):",
            "<data>",
            *history_lines(package, with_ids=False),
            "</data>",
            "",
            "<data>",
            *blocks,
            "</data>",
            "",
            "Label every numbered claim, judging it only against the user's history and the sources of its own item:",
            '- "supported": every factual statement is stated in or directly implied by those sources;',
            '- "partial": some of it is supported and some is not;',
            '- "unsupported": it says something the sources do not say, or contradicts them.',
            "Opinions attributed to reviewers count as supported only if a review says so.",
            f'Answer only JSON: {{"labels": [...]}} with exactly {number} labels, in claim order.',
        ]
    )


def parse_judge(text: str | None, count: int) -> list[str]:
    try:
        labels = json.loads(text)["labels"]
        assert isinstance(labels, list)
    except (TypeError, ValueError, KeyError, AssertionError):
        labels = []

    labels = [label if label in LABELS else "unlabeled" for label in labels[:count]]

    return labels + ["unlabeled"] * (count - len(labels))
