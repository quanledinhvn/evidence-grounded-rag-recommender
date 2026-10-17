"""recsys-cf: train, evaluate and query the CF models (ItemKNN, BPR, Popularity)."""

from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any

from backend.cf.bpr import BPR
from backend.cf.common import Popularity, load_fit_data
from backend.cf.evaluate import evaluate
from backend.cf.itemknn import ItemKNN
from backend.data.io import read_json, write_json


MODELS = {"itemknn": ItemKNN, "bpr": BPR}


def load_model(model_dir: Path) -> Any:
    """Open a saved model; config.json says which kind it is."""
    return MODELS[read_json(model_dir / "config.json")["model"]].load(model_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-cf")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="fit a model and save it to --output-dir")
    train.add_argument("--model", choices=sorted(MODELS), required=True)
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--catalog", type=Path, required=True)
    train.add_argument("--phase", choices=["validation", "test"], required=True)
    train.add_argument("--output-dir", type=Path, required=True)

    train.add_argument("--neighbors", type=int, default=400, help="itemknn")
    train.add_argument("--recency-decay", type=float, default=0.7, help="itemknn")
    train.add_argument("--min-rating", type=float, default=1.0, help="itemknn")
    train.add_argument("--factors", type=int, default=64, help="bpr")
    train.add_argument("--epochs", type=int, default=60, help="bpr")
    train.add_argument("--learning-rate", type=float, default=0.01, help="bpr")
    train.add_argument("--regularization", type=float, default=0.001, help="bpr")
    train.add_argument("--seed", type=int, default=42, help="bpr")

    evaluation = commands.add_parser("evaluate", help="score a saved model on a queries file")
    evaluation.add_argument("--model-dir", type=Path, required=True)
    evaluation.add_argument("--queries", type=Path, required=True)
    evaluation.add_argument("--popularity", action="store_true", help="evaluate the popularity baseline instead")
    evaluation.add_argument("--query-limit", type=int, help="random sample of this many queries")
    evaluation.add_argument("--output", type=Path, help="also write the report to this JSON file")

    recommend = commands.add_parser("recommend", help="top items for one user")
    recommend.add_argument("--model-dir", type=Path, required=True)
    recommend.add_argument("--user-id", required=True)
    recommend.add_argument("--limit", type=int, default=10)
    recommend.add_argument("--exclude", nargs="*", default=[], help="item ids to leave out")

    args = parser.parse_args(argv)

    if args.command == "train":
        data = load_fit_data(args.splits, args.catalog, args.phase)

        if args.model == "itemknn":
            model = ItemKNN.fit(data, args.neighbors, args.recency_decay, args.min_rating)
        else:
            model = BPR.fit(data, args.factors, args.learning_rate, args.regularization, args.epochs, args.seed)

        model.save(args.output_dir)
        result: Any = model.config

    elif args.command == "evaluate":
        model = load_model(args.model_dir)
        model = Popularity(model.data) if args.popularity else model
        result = evaluate(model, args.queries, query_limit=args.query_limit)

        if args.output:
            write_json(args.output, result)

    else:
        model = load_model(args.model_dir)
        items = model.recommend(args.user_id, args.limit, tuple(args.exclude))
        result = {"user_id": args.user_id, "items": items}

    print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
