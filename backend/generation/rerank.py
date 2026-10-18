"""Q5: the LLM re-ranks the fused top 20 into a top 10.

The prompt holds the user's history (H lines) and one short card per candidate: title,
genres, people, a few words of description and of one review. Candidates are listed in fusion
order and called C1..C20, so the model can only answer with codes we gave it.

Answer: {"item_ids": ["C4", "C1", ...]}. Following docs/architecture.md (Q5):
- exactly k known, distinct codes            -> "llm_valid";
- fewer than k, all known and distinct       -> missing places filled in fusion order, "llm_completed";
- unknown or repeated code, more than k, bad JSON, timeout -> fusion top k, "hybrid_fallback".
"""

from __future__ import annotations

import json

from typing import Any

from backend.generation.sources import DATA_RULE, history_lines, item_name, item_sources, shorten


# Gemini returns exactly this shape (without it the model sometimes answers a bare list).
SCHEMA = {"type": "OBJECT", "properties": {"item_ids": {"type": "ARRAY", "items": {"type": "STRING"}}}, "required": ["item_ids"]}


def candidate_card(package: dict[str, Any], item_id: str, with_evidence: bool) -> str:
    if not with_evidence:
        return item_name(package, item_id)

    parts, review = [], ""

    for _, entry in item_sources(package, item_id):
        if entry["source_type"] == "review":
            review = review or f'review: "{shorten(entry["text"], 35)}"'
        else:
            parts.append(shorten(entry["text"], 30))

    return " | ".join(parts + ([review] if review else [])) or "(no information)"


def rerank_prompt(package: dict[str, Any], candidates: list[str], k: int = 10, with_evidence: bool = True) -> str:
    k = min(k, len(candidates))
    cards = [f"C{n}: {candidate_card(package, item_id, with_evidence)}" for n, item_id in enumerate(candidates, 1)]

    return "\n".join(
        [
            "You recommend movies and TV shows. Predict which items this user will watch next.",
            DATA_RULE,
            "",
            "Items the user rated before (most recent and strongest first):",
            "<data>",
            *history_lines(package, with_ids=False),
            "</data>",
            "",
            f"Candidates, listed in the order of a collaborative-filtering retriever (usually a strong signal):",
            "<data>",
            *cards,
            "</data>",
            "",
            f"Choose the {k} candidates the user is most likely to watch next, best first.",
            f'Answer only JSON: {{"item_ids": ["C..", ...]}} with exactly {k} distinct codes from the list above.',
        ]
    )


def parse_rerank(text: str | None, candidates: list[str], k: int = 10) -> tuple[list[str], str]:
    k = min(k, len(candidates))
    fallback = candidates[:k]
    codes = {f"C{n}": item_id for n, item_id in enumerate(candidates, 1)}

    try:
        chosen = json.loads(text)["item_ids"]
    except (TypeError, ValueError, KeyError):
        return fallback, "hybrid_fallback"

    if not isinstance(chosen, list) or not chosen or len(chosen) > k or len(set(map(str, chosen))) < len(chosen):
        return fallback, "hybrid_fallback"

    if any(code not in codes for code in chosen):
        return fallback, "hybrid_fallback"

    ranking = [codes[code] for code in chosen]
    if len(ranking) == k:
        return ranking, "llm_valid"

    return ranking + [item for item in candidates if item not in ranking][: k - len(ranking)], "llm_completed"


def llm_rerank(llm: Any, package: dict[str, Any], candidates: list[str], k: int = 10, with_evidence: bool = True) -> dict[str, Any]:
    answer = llm.generate(rerank_prompt(package, candidates, k, with_evidence), max_output_tokens=512, schema=SCHEMA)
    ranking, status = parse_rerank(answer["text"], candidates, k)

    return {"ranking": ranking, "status": status, **{key: answer[key] for key in ("input_tokens", "output_tokens", "latency_ms", "cache_hit", "error")}}
