"""Q6: the LLM explains why the user may like each of the top 3 items.

Two modes over exactly the same information (run G compares them):
- cited: every source has an id ([E#] item evidence, [H#] user history) and every claim must
  list the ids that support it; the validator (validator.py) then checks each claim;
- free:  the same sources without ids; the model writes up to 2 sentences per item.

Answer (docs/architecture.md, Q6):
    {"items": [{"item_id": "C1", "claims": [{"text": "...", "evidence_ids": ["E2"], "history_ids": ["H1"]}]}]}
At most 2 claims per item; unknown item codes are ignored; bad JSON -> None ("generation_failed").
"""

from __future__ import annotations

import json
import re

from typing import Any

from backend.generation.sources import DATA_RULE, history_lines, item_sources, shorten


MAX_CLAIMS = 2

# The model sometimes writes ids inside the text ("... Billy Bob Thornton [E104, E105]").
INLINE_IDS = re.compile(r"\s*\[((?:[EH]\d+\s*,?\s*)+)\]")

STRINGS = {"type": "ARRAY", "items": {"type": "STRING"}}


def schema(cited: bool) -> dict[str, Any]:
    claim = {"text": {"type": "STRING"}, **({"evidence_ids": STRINGS, "history_ids": STRINGS} if cited else {})}
    item = {"type": "OBJECT", "properties": {"item_id": {"type": "STRING"},
                                             "claims": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": claim, "required": ["text"]}}},
            "required": ["item_id", "claims"]}

    return {"type": "OBJECT", "properties": {"items": {"type": "ARRAY", "items": item}}, "required": ["items"]}

CITED_RULES = [
    "For each item write at most 2 short claims explaining why this user may like it.",
    "Every claim must be supported by the sources: facts about the item cite its evidence ids (E#),",
    "statements about the user's taste cite history ids (H#); a claim linking both cites both.",
    "Cite only evidence listed under that same item. Use no fact that is not in the sources.",
    "Reviews are opinions of other viewers: say so (for example 'a reviewer says ...').",
    "Do not put ids inside the text. If nothing supports a reason, return an empty claims list.",
    'Answer only JSON: {"items": [{"item_id": "C1", "claims": [{"text": "...", "evidence_ids": ["E1"], "history_ids": ["H1"]}]}]}',
]

FREE_RULES = [
    "For each item write at most 2 short sentences explaining why this user may like it.",
    'Answer only JSON: {"items": [{"item_id": "C1", "claims": [{"text": "first sentence"}, {"text": "second sentence"}]}]}',
]


def explain_prompt(package: dict[str, Any], items: list[str], cited: bool = True) -> str:
    blocks = []

    for n, item_id in enumerate(items, 1):
        lines = [f"[{eid}] {shorten(entry['text'], 80)}" if cited else f"- {shorten(entry['text'], 80)}"
                 for eid, entry in item_sources(package, item_id)]
        blocks += [f"Item C{n}:", *(lines or ["(no information)"]), ""]

    return "\n".join(
        [
            "You explain movie and TV recommendations to a user.",
            DATA_RULE,
            "",
            "The user's history:",
            "<data>",
            *history_lines(package, with_ids=cited),
            "</data>",
            "",
            "Recommended items and their sources:",
            "<data>",
            *blocks,
            "</data>",
            "",
            *(CITED_RULES if cited else FREE_RULES),
        ]
    )


def parse_explanations(text: str | None, items: list[str], cited: bool = True) -> dict[str, list[dict[str, Any]]] | None:
    """item_id -> claims ([] for an item the model skipped); None when the answer is not the expected JSON."""
    codes = {f"C{n}": item_id for n, item_id in enumerate(items, 1)}

    try:
        answer = json.loads(text)["items"]
        assert isinstance(answer, list)
    except (TypeError, ValueError, KeyError, AssertionError):
        return None

    explained: dict[str, list[dict[str, Any]]] = {item_id: [] for item_id in items}

    for entry in answer:
        if not isinstance(entry, dict) or entry.get("item_id") not in codes:
            continue

        for claim in (entry.get("claims") or [])[:MAX_CLAIMS]:
            if isinstance(claim, dict) and isinstance(claim.get("text"), str) and claim["text"].strip():
                explained[codes[entry["item_id"]]].append(clean_claim(claim, cited))

    return explained


def clean_claim(claim: dict[str, Any], cited: bool) -> dict[str, Any]:
    """Ids written inside the text are removed from it and, in cited mode, added to the id lists."""
    inline = [code.strip() for group in INLINE_IDS.findall(claim["text"]) for code in group.split(",") if code.strip()]
    text = INLINE_IDS.sub("", claim["text"]).strip()

    evidence = [str(i) for i in claim.get("evidence_ids") or []] if cited else []
    history = [str(i) for i in claim.get("history_ids") or []] if cited else []

    if cited:
        evidence += [code for code in inline if code.startswith("E") and code not in evidence]
        history += [code for code in inline if code.startswith("H") and code not in history]

    return {"text": text, "evidence_ids": evidence, "history_ids": history}


def llm_explain(llm: Any, package: dict[str, Any], items: list[str], cited: bool = True) -> dict[str, Any]:
    answer = llm.generate(explain_prompt(package, items, cited), max_output_tokens=768, schema=schema(cited))
    explained = parse_explanations(answer["text"], items, cited)

    return {
        "items": explained or {},
        "status": "ok" if explained is not None else "generation_failed",
        **{key: answer[key] for key in ("input_tokens", "output_tokens", "latency_ms", "cache_hit", "error")},
    }
