"""Checks every claim of a cited explanation against the evidence package.

A claim is kept only when it has no problem. Problems, cheapest checks first:
- no_source               : no E# and no H# at all;
- unknown_evidence        : an E# that is not in the package;
- evidence_of_other_item  : an E# that belongs to another candidate;
- unknown_history         : an H# that is not in the user's profile;
- name_not_in_sources     : a name in the claim ("Steven Spielberg") that none of the cited sources contains;
- low_similarity          : (with an encoder) the claim is not close in meaning to any cited source.

Valid ids only prove that the claim points at real sources of the right item; the last two checks
are a cheap test of whether the sources actually say what the claim says. The LLM judge
(judge.py) measures how well these checks work.
"""

from __future__ import annotations

import re

from typing import Any, Callable

import numpy as np

from backend.generation.sources import item_name


# Two or more capitalised words in a row: names of people and titles ("Christopher Nolan").
NAME = re.compile(r"\b[A-Z][^\W\d_]+(?:\s+[A-Z][^\W\d_.]*\.?)*\s+[A-Z][^\W\d_]+")

# Capitalised only because they start a sentence or a phrase, not part of a name.
NOT_NAME_START = {
    "a", "an", "the", "like", "if", "as", "this", "its", "it", "you", "your", "both", "since", "with", "for",
    "fans", "reviewers", "viewers", "another", "and", "but", "also", "in", "from", "one", "while", "given",
}


def names(text: str) -> list[str]:
    found = []

    for match in NAME.finditer(text):
        words = match.group(0).split()
        while words and words[0].lower() in NOT_NAME_START:
            words = words[1:]

        if len(words) >= 2:
            found.append(" ".join(words))

    return found


def source_texts(claim: dict[str, Any], package: dict[str, Any]) -> list[str]:
    texts = [package["evidence"][eid]["text"] for eid in claim["evidence_ids"] if eid in package["evidence"]]

    for hid in claim["history_ids"]:
        entry = package["history"].get(hid)
        if entry:
            texts.append(entry["text"] + " " + (entry.get("own_review") or ""))

    return texts


def claim_problems(
    claim: dict[str, Any],
    item_id: str,
    package: dict[str, Any],
    encode: Callable[[list[str]], np.ndarray] | None = None,
    min_similarity: float = 0.3,
) -> list[str]:
    evidence, history = package["evidence"], package["history"]

    if not claim["evidence_ids"] and not claim["history_ids"]:
        return ["no_source"]

    problems = []
    if any(eid not in evidence for eid in claim["evidence_ids"]):
        problems.append("unknown_evidence")
    if any(eid in evidence and evidence[eid]["item_id"] != item_id for eid in claim["evidence_ids"]):
        problems.append("evidence_of_other_item")
    if any(hid not in history for hid in claim["history_ids"]):
        problems.append("unknown_history")

    if problems:
        return problems

    # The explained item's own title may always be named, even when its title source is not cited.
    texts = source_texts(claim, package)
    joined = " ".join(texts + [item_name(package, item_id)]).lower()

    if any(name.lower() not in joined for name in names(claim["text"])):
        problems.append("name_not_in_sources")

    if encode is not None and max_similarity(claim, package, encode) < min_similarity:
        problems.append("low_similarity")

    return problems


def max_similarity(claim: dict[str, Any], package: dict[str, Any], encode: Callable[[list[str]], np.ndarray]) -> float:
    """Highest cosine between the claim and one of its cited sources (0 when nothing is cited)."""
    texts = source_texts(claim, package)
    if not texts:
        return 0.0

    vectors = encode([claim["text"], *texts])
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)

    return float((vectors[1:] @ vectors[0]).max())


def kept(claim: dict[str, Any], min_similarity: float) -> bool:
    """A checked claim survives when it has no problem and is close enough to its sources."""
    return not claim["problems"] and claim["similarity"] >= min_similarity


def render(claims: list[dict[str, Any]]) -> str:
    """'Claim one [E2][H1]. Claim two [E3].' - citations come from the checked ids, not from the model's text."""
    sentences = []

    for claim in claims:
        ids = "".join(f"[{i}]" for i in claim["evidence_ids"] + claim["history_ids"])
        sentences.append(f"{claim['text'].rstrip(' .')} {ids}.")

    return " ".join(sentences)


def sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Callable[[list[str]], np.ndarray]:
    """MiniLM (the same model as the Dense retriever), loaded once."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = 128

    return lambda texts: model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
