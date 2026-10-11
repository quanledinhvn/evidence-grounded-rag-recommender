from __future__ import annotations

import argparse
import json
from datetime import datetime

from pathlib import Path
from typing import Any

from backend.data.audit import audit_dataset, write_audit_report
from backend.data.deduplicate import deduplicate_dataset
from backend.data.enrich import enrich_catalog
from backend.data.normalize import normalize_dataset
from backend.data.presplit import import_pre_split_dataset

from backend.data.split import split_dataset


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "audit":
        report = audit_dataset(
            args.reviews,
            args.metadata,
            audit_time=args.audit_time,
        )
        write_audit_report(report, args.output)
        _print_summary(report["metrics"])

        return 0

    if args.command == "normalize":
        stats = normalize_dataset(
            args.reviews,
            args.metadata,
            args.output_dir,
            catalog_policy=args.catalog_policy,
            audit_time=args.audit_time,
        )
        _print_summary(stats)

        return 0

    if args.command == "deduplicate":
        stats = deduplicate_dataset(args.interactions, args.output)
        _write_json(args.output.with_suffix(".stats.json"), stats)
        _print_summary(stats)

        return 0

    if args.command == "split":
        stats = split_dataset(args.interactions, args.output_dir)
        _print_summary(stats)

        return 0

    if args.command == "import-splits":
        stats = import_pre_split_dataset(
            args.train,
            args.validation,
            args.test,
            args.output_dir,
        )
        _print_summary(stats)

        return 0

    if args.command == "enrich-catalog":
        stats = enrich_catalog(
            args.catalog,
            args.metadata,
            args.output,
        )
        _print_summary(stats)

        return 0

    stats = _prepare(args)
    _print_summary(stats)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-data",
        description="Prepare raw data or import existing dataset splits.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="D0: audit raw JSONL files")
    _add_raw_paths(audit)
    audit.add_argument("--output", type=Path, required=True)

    audit.add_argument("--audit-time", type=_parse_datetime)

    normalize = subparsers.add_parser(
        "normalize",
        help="D1: normalize valid reviews and catalog metadata",
    )
    _add_raw_paths(normalize)
    _add_normalize_options(normalize)

    normalize.add_argument("--audit-time", type=_parse_datetime)

    deduplicate = subparsers.add_parser(
        "deduplicate",
        help="D2: keep the earliest interaction for every user-item pair",
    )
    deduplicate.add_argument("--interactions", type=Path, required=True)
    deduplicate.add_argument("--output", type=Path, required=True)

    split = subparsers.add_parser(
        "split",
        help="D3: create per-user time-based train/validation/test splits",
    )
    split.add_argument("--interactions", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)

    import_splits = subparsers.add_parser(
        "import-splits",
        help="Import existing Amazon 2023 train/validation/test CSV files",
    )
    import_splits.add_argument("--train", type=Path, required=True)
    import_splits.add_argument("--validation", type=Path, required=True)
    import_splits.add_argument("--test", type=Path, required=True)

    import_splits.add_argument("--output-dir", type=Path, required=True)

    enrich = subparsers.add_parser(
        "enrich-catalog",
        help="Join item metadata into an imported catalog by parent_asin",
    )
    enrich.add_argument("--catalog", type=Path, required=True)
    enrich.add_argument("--metadata", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Run D0 through D3 in sequence",
    )
    _add_raw_paths(prepare)
    _add_normalize_options(prepare)

    prepare.add_argument("--audit-time", type=_parse_datetime)

    return parser


def _add_raw_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)


def _add_normalize_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog-policy",
        choices=("all_reviews", "metadata_text"),
        default="all_reviews",
    )


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = audit_dataset(
        args.reviews,
        args.metadata,
        audit_time=args.audit_time,
    )
    write_audit_report(report, args.output_dir / "data_quality.json")

    normalize_stats = normalize_dataset(
        args.reviews,
        args.metadata,
        args.output_dir,
        catalog_policy=args.catalog_policy,
        audit_time=args.audit_time,
    )

    deduplicated_path = args.output_dir / "interactions_deduplicated.parquet"
    deduplicate_stats = deduplicate_dataset(
        args.output_dir / "interactions.parquet",
        deduplicated_path,
    )
    _write_json(args.output_dir / "deduplication_stats.json", deduplicate_stats)

    split_stats = split_dataset(deduplicated_path, args.output_dir)

    return {
        "d0": report["metrics"],
        "d1": normalize_stats,
        "d2": deduplicate_stats,
        "d3": split_stats,
    }


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an ISO-8601 datetime") from exc


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _print_summary(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
