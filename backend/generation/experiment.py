"""Run G: explanations of the fusion top 3, with and without citations, on a fixed user sample.

    python -m backend.generation.experiment --phase test --sample 200 --output-dir runs/llm/test-200
    python -m backend.generation.experiment --phase test --sample 5 --with-target 5 --output-dir runs/llm/smoke-10 \
        --report docs/results/smoke_test_10_record.md

The ranking is the fusion order (no re-ranking). Steps, saved in --output-dir (a re-run reuses them;
every LLM call is cached in llm_cache.jsonl / judge_cache.jsonl):
1. packages.jsonl  for each sampled query: profile summary, fusion top 10, evidence package of the top 10;
2. explain.json    the LLM explains the top 3 twice: "free" (no ids) and "cited" ([E#]/[H#] required);
                   every cited claim is checked by the validator;
3. judge           a stronger LLM labels every claim supported / partial / unsupported;
4. summary.json    ranking and explanation metrics; manual_check.csv: 40 judged claims to check by hand;
   --report        a Markdown report with every step of every sampled query.

Samples (docs/architecture.md, section 6):
- random:       `--sample` queries drawn with seed 42, never chosen by whether the system found the target;
- with target:  the first `--with-target` queries of a 5,000-query draw whose target film is in the
                fusion top 10 (to look at explanations of correct recommendations). Reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random
from typing import Any, Callable

import numpy as np

from backend.cf.cli import load_model
from backend.data.editions import load_editions
from backend.data.io import write_json
from backend.eval.evaluate import film_rank, load_queries, ranking_metrics
from backend.generation.explain import llm_explain
from backend.generation.judge import SCHEMA as JUDGE_SCHEMA, judge_prompt, parse_judge
from backend.generation.llm import CHEAP_MODEL, JUDGE_MODEL, Gemini
from backend.generation.report import write_report
from backend.generation.sources import item_name, item_sources
from backend.generation.validator import claim_problems, kept, max_similarity, sentence_encoder
from backend.service import EXPLAINED, FUSION, QUERIES, TOP_K, build_package


ID_PROBLEMS = {"no_source", "unknown_evidence", "evidence_of_other_item", "unknown_history"}


# ---------- 1. sample and evidence packages ----------

def draw_queries(queries_path: Path, fusion: Any, editions: dict[str, str], sample: int, with_target: int, seed: int) -> dict[str, Any]:
    random_queries = load_queries(queries_path, sample, seed)
    drawn = {"random": random_queries, "with_target": [], "scanned": 0, "hit_rate_at_10": None}
    if not with_target:
        return drawn

    scan = load_queries(queries_path, 5_000, seed)
    taken = {query["user_id"] for query in random_queries}
    hits = []

    for start in range(0, len(scan), 256):
        batch = scan[start : start + 256]
        tops = fusion.recommend_batch([query["user_id"] for query in batch], limit=TOP_K)
        hits += [q for q, items in zip(batch, tops) if film_rank([i["item_id"] for i in items], q["target_item_id"], editions)]

    drawn.update(with_target=[q for q in hits if q["user_id"] not in taken][:with_target], scanned=len(scan), hit_rate_at_10=len(hits) / len(scan))

    return drawn


def build_packages(drawn: dict[str, Any], fusion: Any, artifacts: Path, phase: str, editions: dict[str, str], path: Path) -> list[dict]:
    """One record per sampled query; appended to `path` one by one, so a stopped run resumes."""
    done = {}
    if path.exists():
        done = {r["query"]["user_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}

    records = []
    for group in ("random", "with_target"):
        for query in drawn[group]:
            if query["user_id"] not in done:
                package, ranked, profile = build_package(query["user_id"], query["as_of"], phase, fusion, artifacts, editions)
                summary = {"seen": len(profile["seen_item_ids"]), "positive": len(profile["positive_signals"]),
                           "negative": len(profile["negative_signals"]), "without_metadata": len(profile["items_without_metadata"])}

                done[query["user_id"]] = {"query": query, "group": group, "profile": summary, "ranked": ranked, "package": package}
                with path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(done[query["user_id"]], ensure_ascii=False, default=float) + "\n")

            records.append(done[query["user_id"]])

    return records


# ---------- 2-3. explanations, validator, judge ----------

def run_explain(records: list[dict[str, Any]], llm: Gemini, encode: Callable, workers: int) -> dict[str, dict]:
    def one(record):
        top = record["package"]["candidates"][:EXPLAINED]
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
        user, usage = record["query"]["user_id"], {}

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


# ---------- 4. metrics ----------

def ranking_summary(records: list[dict[str, Any]], editions: dict[str, str]) -> dict[str, Any]:
    ranks = np.array([film_rank([i["item_id"] for i in r["ranked"]], r["query"]["target_item_id"], editions) for r in records])
    metrics = ranking_metrics(ranks)

    return {"users": len(records), "ndcg_at_10": metrics["ndcg_at_10"], "hr_at_10": metrics["hr_at_10"]}


def explain_summary(explains: dict[str, dict], judges: dict[str, dict], min_similarity: float) -> dict[str, Any]:
    users = list(explains)
    summary: dict[str, Any] = {"users": len(users), "items_requested": sum(len(explains[u]["items"]) for u in users)}

    free = [c for u in users for claims in explains[u]["free"]["items"].values() for c in claims]
    cited = [c for u in users for claims in explains[u]["cited"]["items"].values() for c in claims]

    def covered(mode, threshold=None):
        return sum(1 for u in users for claims in explains[u][mode]["items"].values() if any(threshold is None or kept(c, threshold) for c in claims))

    summary["free"] = {**label_summary(free), "items_explained": covered("free")}
    summary["cited_raw"] = {**label_summary(cited), "items_explained": covered("cited")}
    summary["cited_validated"] = {**label_summary([c for c in cited if kept(c, min_similarity)]), "items_explained": covered("cited", min_similarity),
                                  "min_similarity": min_similarity}
    summary["removed_by_validator"] = label_summary([c for c in cited if not kept(c, min_similarity)])
    summary["problems"] = dict(Counter(p for c in cited for p in c["problems"]))
    summary["similarity_thresholds"] = {
        str(t): {**label_summary([c for c in cited if kept(c, t)]), "items_explained": covered("cited", t)} for t in (0.0, 0.2, 0.3, 0.4, 0.5)
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


def manual_check(records: list[dict], explains: dict[str, dict], path: Path, count: int = 40, seed: int = 42) -> None:
    """A fixed random set of judged claims, with their sources, for a person to label."""
    rows = []
    packages = {r["query"]["user_id"]: r["package"] for r in records}

    for user, explained in explains.items():
        for mode in ("free", "cited"):
            for item_id, claims in explained[mode]["items"].items():
                for claim in claims:
                    sources = [f"[{eid}] {entry['text']}" for eid, entry in item_sources(packages[user], item_id)]
                    rows.append({"mode": mode, "item": item_name(packages[user], item_id), "claim": claim["text"],
                                 "cited": " ".join(claim["evidence_ids"] + claim["history_ids"]), "problems": " ".join(claim.get("problems", [])),
                                 "judge_label": claim.get("label", "unlabeled"), "manual_label": "", "sources": " || ".join(sources)})

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["mode", "item", "claim", "cited", "problems", "judge_label", "manual_label", "sources"])
        writer.writeheader()
        writer.writerows(Random(seed).sample(rows, min(count, len(rows))))


# ---------- CLI ----------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-llm")
    parser.add_argument("--phase", choices=["validation", "test"], required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/movies-tv-2023"))
    parser.add_argument("--model-dir", type=Path, help="fusion with editions (default: the one of the phase)")
    parser.add_argument("--sample", type=int, default=200, help="random queries")
    parser.add_argument("--with-target", type=int, default=0, help="extra queries whose target is in the fusion top 10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=CHEAP_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--min-similarity", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, help="also write a Markdown report of every step of every query")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    editions = load_editions(args.artifacts_dir / "editions.parquet")
    fusion = load_model(args.model_dir or Path(FUSION[args.phase]))

    drawn = draw_queries(args.artifacts_dir / QUERIES[args.phase], fusion, editions, args.sample, args.with_target, args.seed)
    records = build_packages(drawn, fusion, args.artifacts_dir, args.phase, editions, out / "packages.jsonl")

    explains = run_explain(records, Gemini(args.model, out / "llm_cache.jsonl"), sentence_encoder(), args.workers)

    # The judge "thinks" a little before labelling: longer timeout and a large output budget.
    judge = Gemini(args.judge_model, out / "judge_cache.jsonl", timeout=60.0, thinking_level="low")
    judges = run_judge(records, explains, judge, args.workers)
    write_json(out / "explain.json", explains)

    summary = {
        "phase": args.phase, "model": args.model, "judge_model": args.judge_model, "seed": args.seed,
        "hit_rate_at_10_of_scan": drawn["hit_rate_at_10"],
        "ranking": {group: ranking_summary([r for r in records if r["group"] == group], editions) for group in ("random", "with_target")},
        "G": explain_summary(explains, judges, args.min_similarity),
    }
    write_json(out / "summary.json", summary)
    manual_check(records, explains, out / "manual_check.csv")

    if args.report:
        write_report(args.report, args, drawn, records, explains, summary, fusion, editions)

    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
