"""Runs F (LLM re-ranking) and G (explanations with / without citations) on a fixed user sample.

    python -m backend.generation.experiment --phase test --sample 200 --in-pool 200 --output-dir runs/llm/test-200

Steps, each saved in --output-dir (a re-run reuses them; every LLM call is cached in llm_cache.jsonl):
1. packages.jsonl  for each sampled query: fusion top 50 (for metrics), the re-rank pool (top 20)
                   and the evidence package of the pool ([H#] history, [E#] item evidence);
2. rerank.jsonl    F: LLM top 10 from the pool with evidence cards; F0: the same from titles only;
3. explain.jsonl   G: explanations of the F top 3, "free" and "cited"; every cited claim checked by the validator;
4. judge.jsonl     judge labels (supported / partial / unsupported) for every free and cited claim;
5. summary.json    all metrics; manual_check.csv: 40 judged claims to check by hand.

Samples (docs/architecture.md, section 6):
- main: `--sample` queries drawn with seed 42, never chosen by whether the retriever found the target;
- in-pool: the first `--in-pool` queries of a 5,000-query draw whose target film is in the pool. Re-ranking
  can only help when the target is in the pool, so F is also reported on this group, separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
from typing import Any, Callable

import numpy as np

from backend.cf.cli import load_model
from backend.cf.evaluate import film_rank, load_queries, ranking_metrics
from backend.data.editions import borrow_titles, load_editions
from backend.data.io import write_json
from backend.evidence.evidence import evidence_package
from backend.generation.explain import llm_explain
from backend.generation.judge import SCHEMA as JUDGE_SCHEMA, judge_prompt, parse_judge
from backend.generation.llm import CHEAP_MODEL, JUDGE_MODEL, Gemini
from backend.generation.rerank import llm_rerank
from backend.generation.sources import item_name, item_sources
from backend.generation.validator import claim_problems, max_similarity, sentence_encoder
from backend.profile.profile import build_profile, load_catalog, load_history


QUERIES = {"validation": "queries_val.parquet", "test": "queries_test.parquet"}
FUSION = {"validation": "runs/validation/fusion-editions", "test": "runs/fusion-editions-movies-tv-2023/test"}
POOL, TOP_K, EXPLAINED = 20, 10, 3
ID_PROBLEMS = {"no_source", "unknown_evidence", "evidence_of_other_item", "unknown_history"}


# ---------- 1. sample and evidence packages ----------

def draw_queries(queries_path: Path, fusion: Any, editions: dict[str, str], sample: int, in_pool: int, seed: int) -> dict[str, Any]:
    main = load_queries(queries_path, sample, seed)
    if not in_pool:
        return {"main": main, "in_pool": [], "scanned": 0, "pool_recall": None}

    scan = load_queries(queries_path, 5_000, seed)
    found = []

    for start in range(0, len(scan), 256):
        batch = scan[start : start + 256]
        pools = fusion.recommend_batch([query["user_id"] for query in batch], limit=POOL)
        found += [q for q, items in zip(batch, pools) if film_rank([i["item_id"] for i in items], q["target_item_id"], editions)]

    return {"main": main, "in_pool": found[:in_pool], "scanned": len(scan), "pool_recall": len(found) / len(scan)}


def build_packages(queries: list[dict[str, Any]], fusion: Any, artifacts: Path, phase: str, editions: dict[str, str], path: Path) -> dict[str, dict]:
    """user_id -> {"query", "top50", "pool", "package"}; appended to `path` one by one, so a stopped run resumes."""
    records = {}
    if path.exists():
        records = {r["query"]["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    todo = [query for query in queries if query["user_id"] not in records]
    connection = sqlite3.connect(artifacts / "reviews.sqlite")
    catalog_path = artifacts / "catalog_enriched.parquet"

    for start in range(0, len(todo), 64):
        batch = todo[start : start + 64]
        tops = fusion.recommend_batch([query["user_id"] for query in batch], limit=50)

        for query, items in zip(batch, tops):
            history = load_history(artifacts / "splits.parquet", query["user_id"], query["as_of"], phase)
            history_catalog = load_catalog(catalog_path, {row["item_id"] for row in history})
            borrow_titles(history_catalog, editions, catalog_path)
            profile = build_profile(query["user_id"], query["as_of"], history, history_catalog)

            top50 = [item["item_id"] for item in items]
            pool_catalog = load_catalog(catalog_path, set(top50[:POOL]))
            borrow_titles(pool_catalog, editions, catalog_path)

            record = {"query": query, "top50": top50, "pool": top50[:POOL], "package": evidence_package(profile, top50[:POOL], pool_catalog, connection)}
            records[query["user_id"]] = record

            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=float) + "\n")

        print(f"packages {len(records)}/{len(queries)}", flush=True)

    return records


# ---------- 2-4. LLM steps ----------

def run_rerank(records: list[dict[str, Any]], llm: Gemini, workers: int) -> dict[str, dict]:
    def one(record):
        package, pool = record["package"], record["pool"]
        return record["query"]["user_id"], {"F": llm_rerank(llm, package, pool, TOP_K, True), "F0": llm_rerank(llm, package, pool, TOP_K, False)}

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, records))


def run_explain(records: list[dict[str, Any]], reranks: dict[str, dict], llm: Gemini, encode: Callable, workers: int) -> dict[str, dict]:
    def one(record):
        top = reranks[record["query"]["user_id"]]["F"]["ranking"][:EXPLAINED]
        return record["query"]["user_id"], {"items": top, "free": llm_explain(llm, record["package"], top, cited=False),
                                            "cited": llm_explain(llm, record["package"], top, cited=True)}

    with ThreadPoolExecutor(workers) as pool:
        explains = dict(pool.map(one, records))

    # Validator in the main thread (the sentence encoder is not shared between threads).
    for record in records:
        package = record["package"]

        for item_id, claims in explains[record["query"]["user_id"]]["cited"]["items"].items():
            for claim in claims:
                claim["problems"] = claim_problems(claim, item_id, package)
                claim["similarity"] = max_similarity(claim, package, encode) if not ID_PROBLEMS & set(claim["problems"]) else 0.0

    return explains


def run_judge(records: list[dict[str, Any]], explains: dict[str, dict], judge: Gemini, workers: int) -> dict[str, dict]:
    def one(record):
        user = record["query"]["user_id"]
        usage = {}

        for mode in ("free", "cited"):
            explained = explains[user][mode]["items"]
            claims = [claim for item_claims in explained.values() for claim in item_claims]
            if not claims:
                continue

            answer = judge.generate(judge_prompt(record["package"], explained), max_output_tokens=4096, schema=JUDGE_SCHEMA)
            for claim, label in zip(claims, parse_judge(answer["text"], len(claims))):
                claim["label"] = label

            usage[mode] = {key: answer[key] for key in ("input_tokens", "output_tokens", "latency_ms", "cache_hit", "error")}

        return user, usage

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, records))


# ---------- 5. metrics ----------

def rerank_summary(records: list[dict[str, Any]], reranks: dict[str, dict], editions: dict[str, str]) -> dict[str, Any]:
    def ranks(lists):
        return np.array([film_rank(items, r["query"]["target_item_id"], editions) for r, items in zip(records, lists)])

    users = [r["query"]["user_id"] for r in records]
    positive = np.array([(r["query"]["target_rating"] or 0) >= 4 for r in records])
    summary = {"users": len(records)}

    for name, lists in [("fusion", [r["top50"] for r in records]),
                        ("F", [reranks[u]["F"]["ranking"] for u in users]),
                        ("F0", [reranks[u]["F0"]["ranking"] for u in users])]:
        model_ranks = ranks(lists)
        summary[name] = {**ranking_metrics(model_ranks), "positive_target": ranking_metrics(model_ranks[positive])}

    for name in ("F", "F0"):
        summary[name]["status"] = dict(Counter(reranks[u][name]["status"] for u in users))
        summary[name]["usage"] = usage_summary([reranks[u][name] for u in users])

    summary["positive_target_users"] = int(positive.sum())

    return summary


def explain_summary(explains: dict[str, dict], judges: dict[str, dict], min_similarity: float) -> dict[str, Any]:
    users = list(explains)
    requested = sum(len(explains[u]["items"]) for u in users)
    summary: dict[str, Any] = {"users": len(users), "items_requested": requested}

    free = [c for u in users for claims in explains[u]["free"]["items"].values() for c in claims]
    cited = [c for u in users for claims in explains[u]["cited"]["items"].values() for c in claims]

    def kept(claim, threshold):
        return not claim["problems"] and claim["similarity"] >= threshold

    def covered(mode, threshold=None):
        return sum(1 for u in users for claims in explains[u][mode]["items"].values()
                   if any(threshold is None or kept(c, threshold) for c in claims))

    summary["free"] = {**label_summary(free), "items_explained": covered("free")}
    summary["cited_raw"] = {**label_summary(cited), "items_explained": covered("cited")}
    summary["cited_validated"] = {**label_summary([c for c in cited if kept(c, min_similarity)]), "items_explained": covered("cited", min_similarity)}
    summary["cited_validated"]["min_similarity"] = min_similarity

    ids = [(eid, c) for c in cited for eid in c["evidence_ids"] + c["history_ids"]]
    summary["citations"] = {"total": len(ids), "claims_with_id_problems": sum(1 for c in cited if ID_PROBLEMS & set(c["problems"]))}
    summary["problems"] = dict(Counter(p for c in cited for p in c["problems"]))
    summary["removed_by_validator"] = label_summary([c for c in cited if not kept(c, min_similarity)])

    summary["similarity_thresholds"] = {
        str(t): {**label_summary([c for c in cited if kept(c, t)]), "items_explained": covered("cited", t)}
        for t in (0.0, 0.2, 0.3, 0.4, 0.5)
    }
    summary["generation_failed"] = {mode: sum(1 for u in users if explains[u][mode]["status"] != "ok") for mode in ("free", "cited")}
    summary["usage"] = {mode: usage_summary([explains[u][mode] for u in users]) for mode in ("free", "cited")}
    summary["judge_usage"] = usage_summary([j[mode] for j in judges.values() for mode in j])

    return summary


def label_summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(claim.get("label", "unlabeled") for claim in claims)
    labeled = sum(labels[name] for name in ("supported", "partial", "unsupported"))

    return {
        "claims": len(claims),
        "labels": dict(labels),
        "supported_rate": labels["supported"] / labeled if labeled else None,
        "unsupported_rate": labels["unsupported"] / labeled if labeled else None,
    }


def usage_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    fresh = [c for c in calls if not c.get("cache_hit") and c.get("latency_ms") is not None]
    latency = np.array([c["latency_ms"] for c in fresh]) if fresh else np.zeros(0)

    return {
        "calls": len(calls),
        "errors": sum(1 for c in calls if c.get("error")),
        "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
        "latency_ms_p50": float(np.percentile(latency, 50)) if len(latency) else None,
        "latency_ms_p95": float(np.percentile(latency, 95)) if len(latency) else None,
    }


def manual_check(records: dict[str, dict], explains: dict[str, dict], path: Path, count: int = 40, seed: int = 42) -> None:
    """A fixed random set of judged claims, with their sources, for a person to label."""
    rows = []

    for user, explained in explains.items():
        package = records[user]["package"]

        for mode in ("free", "cited"):
            for item_id, claims in explained[mode]["items"].items():
                for claim in claims:
                    sources = [f"[{eid}] {entry['text']}" for eid, entry in item_sources(package, item_id)]
                    rows.append({"mode": mode, "item": item_name(package, item_id), "claim": claim["text"],
                                 "cited": " ".join(claim["evidence_ids"] + claim["history_ids"]),
                                 "problems": " ".join(claim.get("problems", [])), "judge_label": claim.get("label", "unlabeled"),
                                 "manual_label": "", "sources": " || ".join(sources)})

    picked = Random(seed).sample(rows, min(count, len(rows)))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["claim"])
        writer.writeheader()
        writer.writerows(picked)


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.generation.experiment")
    parser.add_argument("--phase", choices=["validation", "test"], required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/movies-tv-2023"))
    parser.add_argument("--model-dir", type=Path, help="fusion with editions (default: the one of the phase)")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--in-pool", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=CHEAP_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", default="packages,rerank,explain,judge", help="stop early, e.g. 'packages'")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out, steps = args.output_dir, set(args.steps.split(","))
    out.mkdir(parents=True, exist_ok=True)
    editions = load_editions(args.artifacts_dir / "editions.parquet")
    fusion = load_model(args.model_dir or Path(FUSION[args.phase]))

    drawn = draw_queries(args.artifacts_dir / QUERIES[args.phase], fusion, editions, args.sample, args.in_pool, args.seed)
    records = build_packages(drawn["main"] + drawn["in_pool"], fusion, args.artifacts_dir, args.phase, editions, out / "packages.jsonl")
    main_records = [records[q["user_id"]] for q in drawn["main"]]
    pool_records = [records[q["user_id"]] for q in drawn["in_pool"]]

    summary: dict[str, Any] = {"phase": args.phase, "model": args.model, "judge_model": args.judge_model,
                               "sample": args.sample, "seed": args.seed, "pool_recall_at_20_of_5000": drawn["pool_recall"]}

    if "rerank" in steps:
        llm = Gemini(args.model, out / "llm_cache.jsonl")
        reranks = run_rerank(main_records + pool_records, llm, args.workers)
        write_json(out / "rerank.json", reranks)

        summary["F_main"] = rerank_summary(main_records, reranks, editions)
        if pool_records:
            summary["F_in_pool"] = rerank_summary(pool_records, reranks, editions)

        if "explain" in steps:
            explains = run_explain(main_records, reranks, llm, sentence_encoder(), args.workers)

            # The judge "thinks" a little before labelling: longer timeout and a large output budget.
            judge = Gemini(args.judge_model, out / "judge_cache.jsonl", timeout=60.0, thinking_level="low")
            judges = run_judge(main_records, explains, judge, args.workers) if "judge" in steps else {}

            write_json(out / "explain.json", explains)
            summary["G"] = explain_summary(explains, judges, args.min_similarity)
            manual_check(records, explains, out / "manual_check.csv")

    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
