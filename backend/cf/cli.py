from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from backend.cf.bpr import BPRConfig, config_to_dict
from backend.cf.bundle import CFBundle, train_bundle
from backend.cf.evaluation import compare_retrievers, evaluate_bundle
from backend.cf.tuning import select_bpr_config
from backend.data.io import source_sha256


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        bundle = train_bundle(
            args.splits,
            args.catalog,
            args.output_dir,
            phase=args.phase,
            config=_config_from_args(args),
        )
        _print_json(bundle.manifest)

        return 0

    if args.command == "evaluate":
        bundle = CFBundle.load(args.bundle)
        report = evaluate_bundle(
            bundle,
            args.queries,
            limit=args.limit,
            batch_size=args.batch_size,
        )
        _print_json(report)

        return 0

    if args.command == "recommend":
        _print_json(_run_recommendation(args))

        return 0

    report = _run_pipeline(args)
    _print_json(report)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-cf",
        description="Train and evaluate CPU-optimized BPR-MF bundles.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train one validation or test bundle")
    _add_training_paths(train)
    train.add_argument("--phase", choices=("validation", "test"), required=True)
    _add_config_options(train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved CF bundle")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--queries", type=Path, required=True)
    evaluate.add_argument("--limit", type=int, choices=(50,), default=50)
    evaluate.add_argument("--batch-size", type=int, default=256)

    recommend = subparsers.add_parser(
        "recommend",
        help="Return CF recommendations for one user from a saved bundle",
    )
    recommend.add_argument("--bundle", type=Path, required=True)
    recommend.add_argument("--user-id", required=True)
    recommend.add_argument("--seen-id", action="append", default=[])
    recommend.add_argument("--limit", type=int, default=50)

    pipeline = subparsers.add_parser(
        "pipeline",
        help="Train validation, lock config, refit test, and compare baselines",
    )
    _add_training_paths(pipeline)
    pipeline.add_argument("--validation-queries", type=Path, required=True)
    pipeline.add_argument("--test-queries", type=Path, required=True)
    _add_config_options(pipeline)
    pipeline.add_argument("--factor-candidates", nargs="+", type=int)
    pipeline.add_argument("--learning-rate-candidates", nargs="+", type=float)
    pipeline.add_argument("--regularization-candidates", nargs="+", type=float)
    pipeline.add_argument("--epoch-candidates", nargs="+", type=int)
    pipeline.add_argument("--tuning-query-limit", type=int, default=50_000)
    pipeline.add_argument("--evaluation-batch-size", type=int, default=256)

    return parser


def _add_training_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def _add_config_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.001)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument(
        "--no-verify-negative-samples",
        action="store_false",
        dest="verify_negative_samples",
    )


def _config_from_args(args: argparse.Namespace) -> BPRConfig:
    return BPRConfig(
        factors=args.factors,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        epochs=args.epochs,
        seed=args.seed,
        num_threads=args.num_threads,
        verify_negative_samples=args.verify_negative_samples,
    )


def _run_recommendation(args: argparse.Namespace) -> dict[str, Any]:
    bundle = CFBundle.load(args.bundle)

    return bundle.recommend(
        args.user_id,
        seen_ids=set(args.seen_id),
        limit=args.limit,
    )


def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    base_config = _config_from_args(args)
    validation_dir = args.output_dir / "validation"
    test_dir = args.output_dir / "test"

    candidate_values = product(
        args.factor_candidates or [base_config.factors],
        args.learning_rate_candidates or [base_config.learning_rate],
        args.regularization_candidates or [base_config.regularization],
        args.epoch_candidates or [base_config.epochs],
    )
    candidates = [
        BPRConfig(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            epochs=epochs,
            seed=base_config.seed,
            num_threads=base_config.num_threads,
            verify_negative_samples=base_config.verify_negative_samples,
        )
        for factors, learning_rate, regularization, epochs in candidate_values
    ]
    candidates = list(dict.fromkeys(candidates))

    config, tuning_records = select_bpr_config(
        args.splits,
        args.catalog,
        args.validation_queries,
        args.output_dir,
        candidates,
        query_limit=args.tuning_query_limit,
        batch_size=args.evaluation_batch_size,
        query_seed=base_config.seed,
    )
    validation_bundle = train_bundle(
        args.splits,
        args.catalog,
        validation_dir,
        phase="validation",
        config=config,
    )
    validation_report = compare_retrievers(
        validation_bundle,
        args.validation_queries,
        batch_size=args.evaluation_batch_size,
    )

    locked_config = {
        "schema_version": "cf-locked-config-v2",
        "config_sha256": validation_bundle.manifest["config_sha256"],
        "input_hashes": {
            "catalog": source_sha256(args.catalog),
            "splits": source_sha256(args.splits),
            "validation_queries": source_sha256(args.validation_queries),
        },
        "selection_metric": "ndcg_at_10",
        "tuning_query_limit": args.tuning_query_limit,
        "tuning_query_seed": base_config.seed,
        "evaluation_batch_size": args.evaluation_batch_size,
        "tie_breakers": [
            "candidate_recall_at_50_desc",
            "model_size_bytes_asc",
            "mean_latency_ms_asc",
        ],
        "bpr": config_to_dict(config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "locked_config.json", locked_config)

    test_bundle = train_bundle(
        args.splits,
        args.catalog,
        test_dir,
        phase="test",
        config=config,
    )
    test_report = compare_retrievers(
        test_bundle,
        args.test_queries,
        batch_size=args.evaluation_batch_size,
    )

    report = {
        "schema_version": "cf-evaluation-v2",
        "selection_protocol": "config_locked_after_validation_before_test_refit",
        "locked_config": locked_config,
        "tuning_candidates": tuning_records,
        "validation": validation_report,
        "test": test_report,
    }
    _write_json(args.output_dir / "evaluation.json", report)

    return report


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
