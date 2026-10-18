"""End-to-end recommendation for one user (the demo), following docs/architecture.md Q1-Q7.

    python -m backend.service --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test

profile (Q1) -> fusion with editions, pool of 20 (Q2-Q3) -> evidence package (Q4)
-> LLM re-rank to 10 (Q5) -> LLM explains the top 3 with citations (Q6) -> validator keeps the
claims it can check and renders them with their source ids (Q7).

A user without any rated item gets the fusion / popularity list and no LLM call ("cold_start").
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time

from pathlib import Path
from typing import Any, Callable

from backend.cf.cli import load_model
from backend.data.editions import borrow_titles, load_editions
from backend.evidence.evidence import evidence_package
from backend.generation.explain import llm_explain
from backend.generation.llm import CHEAP_MODEL, Gemini
from backend.generation.rerank import llm_rerank
from backend.generation.sources import item_name
from backend.generation.validator import claim_problems, render, sentence_encoder
from backend.profile.profile import build_profile, load_catalog, load_history


FUSION = {"validation": "runs/validation/fusion-editions", "test": "runs/fusion-editions-movies-tv-2023/test"}
POOL, TOP_K, EXPLAINED = 20, 10, 3


def recommend(
    user_id: str,
    as_of: int,
    phase: str,
    fusion: Any,
    artifacts: Path,
    editions: dict[str, str],
    llm: Any,
    encode: Callable | None = None,
    min_similarity: float = 0.3,
) -> dict[str, Any]:
    started = time.perf_counter()
    catalog_path = artifacts / "catalog_enriched.parquet"

    # Q1 profile (untitled items borrow the title of another edition).
    history = load_history(artifacts / "splits.parquet", user_id, as_of, phase)
    history_catalog = load_catalog(catalog_path, {row["item_id"] for row in history})
    borrow_titles(history_catalog, editions, catalog_path)
    profile = build_profile(user_id, as_of, history, history_catalog)

    # Q2-Q4 pool and evidence.
    pool = [item["item_id"] for item in fusion.recommend(user_id, POOL, tuple(profile["seen_item_ids"]))]
    pool_catalog = load_catalog(catalog_path, set(pool))
    borrow_titles(pool_catalog, editions, catalog_path)
    package = evidence_package(profile, pool, pool_catalog, sqlite3.connect(artifacts / "reviews.sqlite"))

    trace: dict[str, Any] = {"model": getattr(llm, "model", None), "retrieval_ms": (time.perf_counter() - started) * 1000.0, "llm_calls": []}

    if not profile["positive_signals"] and not profile["negative_signals"]:
        ranking, rerank_status, explained = pool[:TOP_K], "skipped", {}
    else:
        # Q5 re-rank, Q6 explain.
        reranked = llm_rerank(llm, package, pool, TOP_K)
        ranking, rerank_status = reranked["ranking"], reranked["status"]

        explanation = llm_explain(llm, package, ranking[:EXPLAINED], cited=True)
        explained = explanation["items"] if explanation["status"] == "ok" else None

        for stage, call in (("rerank", reranked), ("explain", explanation)):
            trace["llm_calls"].append({"stage": stage, **{k: call[k] for k in ("input_tokens", "output_tokens", "latency_ms", "cache_hit", "error")}})

    recommendations = []
    for rank, item_id in enumerate(ranking, 1):
        entry = {"rank": rank, "item_id": item_id, "title": item_name(package, item_id), "explanation": None,
                 "explanation_status": "not_requested", "evidence_ids": [], "history_ids": []}

        if rank <= EXPLAINED and rerank_status != "skipped":
            if explained is None:
                entry["explanation_status"] = "generation_failed"
            else:
                # Q7: keep only claims the validator accepts; render citations from the checked ids.
                valid = [c for c in explained.get(item_id, []) if not claim_problems(c, item_id, package, encode, min_similarity)]
                entry["explanation"] = render(valid) or None
                entry["explanation_status"] = "citations_valid" if valid else "evidence_unavailable"
                entry["evidence_ids"] = sorted({i for c in valid for i in c["evidence_ids"]}, key=lambda i: int(i[1:]))
                entry["history_ids"] = sorted({i for c in valid for i in c["history_ids"]}, key=lambda i: int(i[1:]))

        recommendations.append(entry)

    cited_e = {i for r in recommendations for i in r["evidence_ids"]}
    cited_h = {i for r in recommendations for i in r["history_ids"]}
    trace["total_ms"] = (time.perf_counter() - started) * 1000.0

    return {
        "user_id": user_id,
        "as_of": as_of,
        "status": "cold_start" if rerank_status == "skipped" else "ok",
        "rerank_status": rerank_status,
        "recommendations": recommendations,
        "evidence": {eid: package["evidence"][eid] for eid in package["evidence"] if eid in cited_e},
        "history_sources": {hid: package["history"][hid] for hid in package["history"] if hid in cited_h},
        "trace": trace,
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"User {result['user_id']} at {result['as_of']} - rerank: {result['rerank_status']}")

    for entry in result["recommendations"]:
        print(f"{entry['rank']:2d}. {entry['title']}")
        if entry["explanation"]:
            print(f"    {entry['explanation']}")
        elif entry["explanation_status"] != "not_requested":
            print(f"    ({entry['explanation_status']})")

    print("\nSources:")
    for hid, source in result["history_sources"].items():
        print(f"  [{hid}] rated {source['rating']:.0f}/5: {source['text'][:100]}")
    for eid, source in result["evidence"].items():
        print(f"  [{eid}] {source['source_id']}: {source['text'][:140]}")

    calls = result["trace"]["llm_calls"]
    tokens = sum((c["input_tokens"] or 0) + (c["output_tokens"] or 0) for c in calls)
    print(f"\n{len(calls)} LLM calls, {tokens} tokens, {result['trace']['total_ms'] / 1000:.1f} s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-demo")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--as-of", type=int, required=True)
    parser.add_argument("--phase", choices=["validation", "test"], default="test")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/movies-tv-2023"))
    parser.add_argument("--model", default=CHEAP_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("runs/llm/demo_cache.jsonl"))
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--json", action="store_true", help="print the full response as JSON")
    args = parser.parse_args(argv)

    result = recommend(
        args.user_id, args.as_of, args.phase, load_model(Path(FUSION[args.phase])), args.artifacts_dir,
        load_editions(args.artifacts_dir / "editions.parquet"), Gemini(args.model, args.cache), sentence_encoder(), args.min_similarity,
    )

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=float))
    else:
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
