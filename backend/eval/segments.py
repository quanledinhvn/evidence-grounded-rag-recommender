"""Metrics by segment: which users and targets does each retriever help?

Segments use only what is known before the query (fit data and catalog), never the test split:
- history: how many items the user saw before as_of: 1-2, 3-5, 6-20, >20;
- popularity: rank of the target item in Popularity: head (top 1% of the catalog),
  torso (next 9%), tail (the remaining 90%);
- title: the target item has a title in the catalog, or not (many Prime Video items).

With `editions` a hit is any edition of the target film ("same film"). Recall@20 is reported too:
20 is the pool the LLM re-ranks.

    python -m backend.eval.segments --member itemknn=runs/itemknn-movies-tv-2023/test ...
        --queries artifacts/movies-tv-2023/queries_test.parquet --catalog ... --editions ... --query-limit 50000
"""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from backend.cf.common import Popularity
from backend.data.editions import load_editions
from backend.data.io import write_json
from backend.eval.evaluate import film_rank, load_queries, ranking_metrics


HISTORY_BINS = [(2, "1-2"), (5, "3-5"), (20, "6-20")]


def history_segment(length: int) -> str:
    for upper, name in HISTORY_BINS:
        if length <= upper:
            return name

    return ">20"


def popularity_segment(rank: int, catalog_size: int) -> str:
    """rank is 0-based (0 = most popular item)."""
    if rank < 0.01 * catalog_size:
        return "head"

    return "torso" if rank < 0.10 * catalog_size else "tail"


def segment_report(
    models: dict[str, Any],
    queries: list[dict[str, Any]],
    titled: set[str],
    editions: dict[str, str] | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    editions = editions or {}
    data = next(iter(models.values())).data

    # Popularity rank of every item (0 = most popular), from the fit data only.
    order = Popularity(data).order
    popularity_rank = {data.item_ids[item]: rank for rank, item in enumerate(order)}

    users = [query["user_id"] for query in queries]
    ranks = {name: film_ranks(model, users, queries, editions, batch_size) for name, model in models.items()}

    labels = {
        "history": [history_segment(len(data.history(user)[0])) for user in users],
        "popularity": [popularity_segment(popularity_rank.get(q["target_item_id"], len(order)), len(order)) for q in queries],
        "title": ["with title" if q["target_item_id"] in titled else "without title" for q in queries],
    }

    report: dict[str, Any] = {"all": summarize_models(ranks, np.ones(len(queries), dtype=bool))}

    for segment, values in labels.items():
        values = np.array(values)
        report[segment] = {name: summarize_models(ranks, values == name) for name in sorted(set(values))}

    return report


def film_ranks(model: Any, users: list[str], queries: list[dict[str, Any]], editions: dict[str, str], batch_size: int) -> np.ndarray:
    """1-based rank of the target film in the model's top 50 (0 = missing)."""
    ranks = np.zeros(len(users))

    for start in range(0, len(users), batch_size):
        results = model.recommend_batch(users[start : start + batch_size], limit=50)

        for offset, items in enumerate(results):
            query = queries[start + offset]
            ranks[start + offset] = film_rank([item["item_id"] for item in items], query["target_item_id"], editions)

    return ranks


def summarize_models(ranks: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {"queries": int(mask.sum())}

    for name, model_ranks in ranks.items():
        summary[name] = ranking_metrics(model_ranks[mask])

    return summary


def main(argv: list[str] | None = None) -> int:
    from backend.cf.cli import load_model, pairs

    parser = argparse.ArgumentParser(prog="python -m backend.eval.segments")
    parser.add_argument("--member", action="append", required=True, help="name=model_dir, repeatable")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--editions", type=Path)
    parser.add_argument("--query-limit", type=int, default=50_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    models = {name: load_model(Path(path)) for name, path in pairs(args.member).items()}
    queries = load_queries(args.queries, args.query_limit)

    catalog = pq.read_table(args.catalog, columns=["item_id", "title"]).to_pylist()
    titled = {row["item_id"] for row in catalog if row["title"]}

    report = segment_report(models, queries, titled, load_editions(args.editions) if args.editions else None)
    print(json.dumps(report, indent=2))

    if args.output:
        write_json(args.output, report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
