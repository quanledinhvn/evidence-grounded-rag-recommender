from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.item_knn.bundle import (
    ItemKNNBundle,
    ItemKNNConfig,
    train_item_knn_bundle,
)
from backend.item_knn.evaluation import evaluate_item_knn


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        bundle = train_item_knn_bundle(
            args.splits,
            args.catalog,
            args.output_dir,
            phase=args.phase,
            config=ItemKNNConfig(
                neighbors=args.neighbors,
                num_threads=args.num_threads,
            ),
        )
        _print_json(bundle.manifest)

        return 0

    bundle = ItemKNNBundle.load(args.bundle)

    if args.command == "recommend":
        recommendation = bundle.recommend(
            args.user_id,
            seen_ids=set(args.seen_id),
            limit=args.limit,
        )
        _print_json(recommendation)

        return 0

    _print_json(evaluate_item_knn(bundle, args.queries, limit=args.limit))

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-item-knn",
        description="Train, evaluate, and serve standalone Item-KNN bundles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one Item-KNN bundle")
    train.add_argument("--splits", type=Path, required=True)
    train.add_argument("--catalog", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--phase", choices=("validation", "test"), required=True)
    train.add_argument("--neighbors", type=int, default=50)
    train.add_argument("--num-threads", type=int, default=0)

    recommend = subparsers.add_parser(
        "recommend",
        help="Recommend from a saved Item-KNN bundle",
    )
    recommend.add_argument("--bundle", type=Path, required=True)
    recommend.add_argument("--user-id", required=True)
    recommend.add_argument("--seen-id", action="append", default=[])
    recommend.add_argument("--limit", type=int, default=50)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved Item-KNN bundle",
    )
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--queries", type=Path, required=True)
    evaluate.add_argument("--limit", type=int, choices=(50,), default=50)

    return parser


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
