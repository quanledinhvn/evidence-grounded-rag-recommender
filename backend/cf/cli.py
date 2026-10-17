"""recsys-cf: train, evaluate and query every retriever (ItemKNN, BPR, BM25, Dense, Popularity)."""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from backend.cf.bpr import BPR
from backend.cf.common import Popularity, load_fit_data
from backend.cf.evaluate import evaluate
from backend.cf.itemknn import ItemKNN
from backend.data.editions import load_editions
from backend.data.io import read_json, write_json
from backend.retrieval.bm25 import BM25
from backend.retrieval.dense import Dense
from backend.retrieval.fusion import Fusion, tune_weights


MODELS = {"itemknn": ItemKNN, "bpr": BPR, "bm25": BM25, "dense": Dense}


def load_model(model_dir: Path) -> Any:
    """Open a saved model; config.json says which kind it is."""
    kind = read_json(model_dir / "config.json")["model"]

    return Fusion.load(model_dir) if kind == "fusion" else MODELS[kind].load(model_dir)


def pairs(values: list[str]) -> dict[str, str]:
    """["a=1", "b=2"] -> {"a": "1", "b": "2"}"""
    return dict(value.split("=", 1) for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-cf")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="fit a model and save it to --output-dir")
    train.add_argument("--model", choices=sorted(MODELS), required=True)
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--catalog", type=Path, required=True, help="catalog_enriched.parquet for bm25/dense")
    train.add_argument("--phase", choices=["validation", "test"], required=True)
    train.add_argument("--output-dir", type=Path, required=True)

    train.add_argument("--neighbors", type=int, default=400, help="itemknn")
    train.add_argument("--recency-decay", type=float, default=0.7, help="itemknn, bm25, dense")
    train.add_argument("--min-rating", type=float, default=1.0, help="itemknn")
    train.add_argument("--factors", type=int, default=64, help="bpr")
    train.add_argument("--epochs", type=int, default=60, help="bpr")
    train.add_argument("--learning-rate", type=float, default=0.01, help="bpr")
    train.add_argument("--regularization", type=float, default=0.001, help="bpr")
    train.add_argument("--seed", type=int, default=42, help="bpr")
    train.add_argument("--vectors", type=Path, help="dense: item vector cache (default: next to the catalog)")

    evaluation = commands.add_parser("evaluate", help="score a saved model on a queries file")
    evaluation.add_argument("--model-dir", type=Path, required=True)
    evaluation.add_argument("--queries", type=Path, required=True)
    evaluation.add_argument("--popularity", action="store_true", help="evaluate the popularity baseline instead")
    evaluation.add_argument("--query-limit", type=int, help="random sample of this many queries")
    evaluation.add_argument("--output", type=Path, help="also write the report to this JSON file")
    evaluation.add_argument("--editions", type=Path, help="editions.parquet: also report same-film metrics")

    recommend = commands.add_parser("recommend", help="top items for one user")
    recommend.add_argument("--model-dir", type=Path, required=True)
    recommend.add_argument("--user-id", required=True)
    recommend.add_argument("--limit", type=int, default=10)
    recommend.add_argument("--exclude", nargs="*", default=[], help="item ids to leave out")

    fuse = commands.add_parser("fuse", help="save a fusion of trained models (nothing to train)")
    fuse.add_argument("--member", action="append", required=True, help="name=model_dir, repeatable")
    fuse.add_argument("--weight", action="append", required=True, help="name=weight, repeatable (incl. popularity)")
    fuse.add_argument("--method", choices=["score", "rrf"], default="score")
    fuse.add_argument("--depth", type=int, default=200, help="items asked from each member")
    fuse.add_argument("--output-dir", type=Path, required=True)
    fuse.add_argument("--editions", type=Path, help="editions.parquet: one edition per film, skip seen films")

    tune = commands.add_parser("tune-fusion", help="grid-search fusion weights on validation queries")
    tune.add_argument("--member", action="append", required=True, help="name=model_dir, repeatable")
    tune.add_argument("--grid", action="append", required=True, help="name=w1,w2,... (incl. popularity)")
    tune.add_argument("--queries", type=Path, required=True)
    tune.add_argument("--query-limit", type=int, default=20_000)
    tune.add_argument("--method", choices=["score", "rrf"], default="score")
    tune.add_argument("--depth", type=int, default=200, help="items asked from each member")
    tune.add_argument("--output", type=Path, help="also write the table to this JSON file")

    args = parser.parse_args(argv)

    if args.command == "train":
        data = load_fit_data(args.splits, args.catalog, args.phase)

        if args.model == "itemknn":
            model = ItemKNN.fit(data, args.neighbors, args.recency_decay, args.min_rating)
        elif args.model == "bpr":
            model = BPR.fit(data, args.factors, args.learning_rate, args.regularization, args.epochs, args.seed)
        elif args.model == "bm25":
            model = BM25.fit(data, args.catalog, recency_decay=args.recency_decay)
        else:
            vectors = args.vectors or args.catalog.parent / "item_vectors.npy"
            model = Dense.fit(data, args.catalog, vectors, recency_decay=args.recency_decay)

        model.save(args.output_dir)
        result: Any = model.config

    elif args.command == "evaluate":
        model = load_model(args.model_dir)
        model = Popularity(model.data) if args.popularity else model
        editions = load_editions(args.editions) if args.editions else None
        result = evaluate(model, args.queries, query_limit=args.query_limit, editions=editions)

        if args.output:
            write_json(args.output, result)

    elif args.command == "recommend":
        model = load_model(args.model_dir)
        items = model.recommend(args.user_id, args.limit, tuple(args.exclude))
        result = {"user_id": args.user_id, "items": items}

    elif args.command == "fuse":
        member_dirs = pairs(args.member)
        members = {name: load_model(Path(path)) for name, path in member_dirs.items()}
        weights = {name: float(value) for name, value in pairs(args.weight).items()}

        editions = load_editions(args.editions) if args.editions else None

        fusion = Fusion(members, weights, depth=args.depth, method=args.method, editions=editions)
        fusion.save(args.output_dir, member_dirs, args.editions)
        result = {**fusion.config, "member_dirs": member_dirs, "editions": str(args.editions or "")}

    else:
        members = {name: load_model(Path(path)) for name, path in pairs(args.member).items()}
        members["popularity"] = Popularity(next(iter(members.values())).data)
        grid = {name: [float(v) for v in values.split(",")] for name, values in pairs(args.grid).items()}

        result = tune_weights(members, args.queries, grid, query_limit=args.query_limit, depth=args.depth, method=args.method)

        if args.output:
            write_json(args.output, result)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
