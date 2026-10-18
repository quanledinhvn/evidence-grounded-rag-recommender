from __future__ import annotations

import json

import numpy as np

from backend.generation.explain import explain_prompt, llm_explain, parse_explanations
from backend.generation.judge import judge_prompt, parse_judge
from backend.generation.llm import Gemini
from backend.generation.rerank import llm_rerank, parse_rerank, rerank_prompt
from backend.generation.validator import claim_problems, render


PACKAGE = {
    "user_id": "REAL-USER-ID",
    "as_of": 100,
    "candidates": ["B0INTERST", "B0TITANIC", "B0NOTITLE"],
    "history": {
        "H1": {"item_id": "B0INCEPTI", "sentiment": "positive", "rating": 5.0, "text": "Inception Directors: Christopher Nolan",
               "own_review": "Loved the dream layers", "source_id": "review:5"},
    },
    "evidence": {
        "E1": {"item_id": "B0INTERST", "source_type": "metadata", "source_id": "meta:B0INTERST:title", "text": "title: Interstellar"},
        "E2": {"item_id": "B0INTERST", "source_type": "metadata", "source_id": "meta:B0INTERST:people",
               "text": "Directors: Christopher Nolan. Cast: Matthew McConaughey"},
        "E3": {"item_id": "B0INTERST", "source_type": "review", "source_id": "review:1", "text": "The time travel plot and the score are stunning"},
        "E4": {"item_id": "B0TITANIC", "source_type": "metadata", "source_id": "meta:B0TITANIC:title", "text": "title: Titanic"},
        "E5": {"item_id": "B0TITANIC", "source_type": "metadata", "source_id": "meta:B0TITANIC:people",
               "text": "Directors: James Cameron. Cast: Leonardo DiCaprio"},
        "E6": {"item_id": "B0NOTITLE", "source_type": "metadata", "source_id": "meta:B0NOTITLE:people",
               "text": "Directors: Greta Gerwig. Cast: Saoirse Ronan"},
    },
}
CANDIDATES = PACKAGE["candidates"]


class FakeLLM:
    """Returns canned texts in order and remembers the prompts."""

    def __init__(self, *texts):
        self.texts, self.prompts = list(texts), []

    def generate(self, prompt, max_output_tokens=512, schema=None):
        self.prompts.append(prompt)
        self.schemas = getattr(self, "schemas", []) + [schema]
        text = self.texts.pop(0)

        return {"text": text, "input_tokens": 10, "output_tokens": 5, "latency_ms": 1.0, "cache_hit": False, "error": None if text else "timeout"}


# ---------- Q5 rerank ----------

def test_rerank_prompt_uses_short_codes_and_never_sends_ids():
    prompt = rerank_prompt(PACKAGE, CANDIDATES, k=2)

    assert "C1" in prompt and "Interstellar" in prompt and "Christopher Nolan" in prompt
    assert "B0INTERST" not in prompt and "REAL-USER-ID" not in prompt
    assert "Loved the dream layers" in prompt                          # the user's own review is part of the profile


def test_parse_rerank_valid_partial_and_fallback():
    assert parse_rerank('{"item_ids": ["C3", "C1"]}', CANDIDATES, k=2) == (["B0NOTITLE", "B0INTERST"], "llm_valid")
    assert parse_rerank('{"item_ids": ["C2"]}', CANDIDATES, k=2) == (["B0TITANIC", "B0INTERST"], "llm_completed")

    for bad in ['{"item_ids": ["C9", "C1"]}', '{"item_ids": ["C1", "C1"]}', '{"item_ids": ["C1", "C2", "C3"]}', "not json", None]:
        assert parse_rerank(bad, CANDIDATES, k=2) == (["B0INTERST", "B0TITANIC"], "hybrid_fallback")


def test_llm_rerank_keeps_usage_and_falls_back_on_error():
    result = llm_rerank(FakeLLM('{"item_ids": ["C2", "C1"]}'), PACKAGE, CANDIDATES, k=2)
    assert result["ranking"] == ["B0TITANIC", "B0INTERST"] and result["status"] == "llm_valid"
    assert result["input_tokens"] == 10

    assert llm_rerank(FakeLLM(None), PACKAGE, CANDIDATES, k=2)["status"] == "hybrid_fallback"


# ---------- Q6 explain ----------

def test_explain_prompt_cited_shows_source_ids_free_does_not():
    cited = explain_prompt(PACKAGE, ["B0INTERST", "B0TITANIC"], cited=True)
    free = explain_prompt(PACKAGE, ["B0INTERST", "B0TITANIC"], cited=False)

    assert "[E2]" in cited and "[H1]" in cited and "E5" in cited
    assert "E6" not in cited                                           # only the explained items' sources
    assert "[E2]" not in free and "Christopher Nolan" in free
    assert "B0INTERST" not in cited + free


def test_parse_explanations_maps_codes_and_keeps_two_claims():
    text = json.dumps({"items": [
        {"item_id": "C1", "claims": [{"text": "a", "evidence_ids": ["E2"], "history_ids": ["H1"]},
                                     {"text": "b", "evidence_ids": ["E3"]}, {"text": "c", "evidence_ids": []}]},
        {"item_id": "C7", "claims": [{"text": "unknown item"}]},
    ]})

    explained = parse_explanations(text, ["B0INTERST", "B0TITANIC"])

    assert [claim["text"] for claim in explained["B0INTERST"]] == ["a", "b"]
    assert explained["B0INTERST"][1]["history_ids"] == []
    assert explained["B0TITANIC"] == []
    assert parse_explanations("{", ["B0INTERST"]) is None


def test_llm_explain_returns_claims_and_usage():
    result = llm_explain(FakeLLM('{"items": [{"item_id": "C1", "claims": [{"text": "x"}]}]}'), PACKAGE, ["B0INTERST"], cited=False)

    assert result["items"]["B0INTERST"][0]["text"] == "x"
    assert result["status"] == "ok" and result["output_tokens"] == 5


# ---------- validator ----------

def claim(text, evidence=(), history=()):
    return {"text": text, "evidence_ids": list(evidence), "history_ids": list(history)}


def test_validator_accepts_a_grounded_claim_and_names_each_problem():
    ok = claim("Directed by Christopher Nolan, like Inception which you loved", ["E2"], ["H1"])
    assert claim_problems(ok, "B0INTERST", PACKAGE) == []

    assert claim_problems(claim("A great film"), "B0INTERST", PACKAGE) == ["no_source"]
    assert claim_problems(claim("x", ["E9"]), "B0INTERST", PACKAGE) == ["unknown_evidence"]
    assert claim_problems(claim("x", ["E5"]), "B0INTERST", PACKAGE) == ["evidence_of_other_item"]
    assert claim_problems(claim("x", [], ["H7"]), "B0INTERST", PACKAGE) == ["unknown_history"]
    assert claim_problems(claim("Directed by Steven Spielberg", ["E2"]), "B0INTERST", PACKAGE) == ["name_not_in_sources"]


def test_validator_flags_low_similarity_when_an_encoder_is_given():
    def encode(texts):          # claim and E3 orthogonal, claim and E2 identical direction
        return np.array([[1.0, 0.0] if "score" in t or "Score" in t else [0.0, 1.0] for t in texts])

    assert claim_problems(claim("Reviewers praise the score", ["E3"]), "B0INTERST", PACKAGE, encode) == []
    assert claim_problems(claim("It is long", ["E3"]), "B0INTERST", PACKAGE, encode) == ["low_similarity"]


def test_render_appends_checked_citations():
    claims = [claim("Directed by Christopher Nolan.", ["E2"], ["H1"]), claim("Reviewers praise the score", ["E3"])]

    assert render(claims) == "Directed by Christopher Nolan [E2][H1]. Reviewers praise the score [E3]."


# ---------- judge ----------

def test_judge_prompt_numbers_claims_and_parse_fills_missing_labels():
    explained = {"B0INTERST": [claim("Nolan film"), claim("Has dragons")], "B0TITANIC": [claim("Stars DiCaprio")]}
    prompt = judge_prompt(PACKAGE, explained)

    assert "1. Nolan film" in prompt and "3. Stars DiCaprio" in prompt and "Leonardo DiCaprio" in prompt
    assert parse_judge('{"labels": ["supported", "unsupported"]}', 3) == ["supported", "unsupported", "unlabeled"]
    assert parse_judge('{"labels": ["maybe", "partial", "supported"]}', 3) == ["unlabeled", "partial", "supported"]
    assert parse_judge("oops", 2) == ["unlabeled", "unlabeled"]


# ---------- Gemini client ----------

def test_gemini_caches_responses_on_disk(tmp_path, monkeypatch):
    calls = []

    def fake_post(self, body):
        calls.append(body)
        return {"candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}}

    monkeypatch.setattr(Gemini, "_post", fake_post)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    first = Gemini("m", tmp_path / "cache.jsonl").generate("hello")
    second = Gemini("m", tmp_path / "cache.jsonl").generate("hello")      # new client: cache read from disk

    assert first["text"] == second["text"] == '{"ok": true}'
    assert (first["input_tokens"], first["output_tokens"], first["cache_hit"]) == (7, 3, False)
    assert second["cache_hit"] is True and len(calls) == 1
    assert "test-key" not in (tmp_path / "cache.jsonl").read_text()


def test_gemini_does_not_cache_a_cut_answer_and_sends_the_thinking_level(tmp_path, monkeypatch):
    bodies = []

    def fake_post(self, body):
        bodies.append(body)
        return {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": '{"labels": ["supp'}]}}]}

    monkeypatch.setattr(Gemini, "_post", fake_post)

    llm = Gemini("m", tmp_path / "cache.jsonl", thinking_level="low")
    result = llm.generate("hello")

    assert result["error"] == "max_tokens" and result["text"] is None
    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert not (tmp_path / "cache.jsonl").exists()


def test_rerank_and_explain_send_a_json_schema():
    llm = FakeLLM('{"item_ids": ["C1", "C2"]}', '{"items": []}')
    llm_rerank(llm, PACKAGE, CANDIDATES, k=2)
    llm_explain(llm, PACKAGE, ["B0INTERST"], cited=True)

    assert llm.schemas[0]["properties"]["item_ids"]["type"] == "ARRAY"
    assert "claims" in str(llm.schemas[1])


def test_parse_explanations_moves_inline_citations_into_the_id_lists():
    text = json.dumps({"items": [{"item_id": "C1", "claims": [
        {"text": "Directed by Christopher Nolan [E2, E3] like Inception [H1].", "evidence_ids": ["E2"], "history_ids": []}]}]})

    claim = parse_explanations(text, ["B0INTERST"])["B0INTERST"][0]

    assert claim["text"] == "Directed by Christopher Nolan like Inception."
    assert claim["evidence_ids"] == ["E2", "E3"] and claim["history_ids"] == ["H1"]


def test_validator_allows_the_title_of_the_explained_item():
    title_claim = claim("The Grand Interstellar Voyage is not it, but Interstellar Stars Matthew McConaughey", ["E2"])
    assert claim_problems(title_claim, "B0INTERST", PACKAGE) == ["name_not_in_sources"]      # an invented title is still caught

    assert claim_problems(claim("Like Inception, it is directed by Christopher Nolan", ["E2"]), "B0INTERST", PACKAGE) == []
    package = {**PACKAGE, "evidence": {**PACKAGE["evidence"], "E1": {**PACKAGE["evidence"]["E1"], "text": "title: American Hustle"}}}
    assert claim_problems(claim("American Hustle stars Matthew McConaughey", ["E2"]), "B0INTERST", package) == []
