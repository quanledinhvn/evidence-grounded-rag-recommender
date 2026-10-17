"""recsys-data: import the CSV splits and enrich the catalog with metadata."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from backend.data.enrich import enrich_catalog
from backend.data.importer import import_splits, reorganize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-data")
    commands = parser.add_subparsers(dest="command", required=True)

    splits = commands.add_parser("import-splits", help="CSV splits -> Parquet artifacts")
    splits.add_argument("--train", type=Path, required=True)
    splits.add_argument("--validation", type=Path, required=True)
    splits.add_argument("--test", type=Path, required=True)
    splits.add_argument("--output-dir", type=Path, required=True)

    enrich = commands.add_parser("enrich-catalog", help="Join item metadata into the catalog")
    enrich.add_argument("--catalog", type=Path, required=True)
    enrich.add_argument("--metadata", type=Path, required=True)
    enrich.add_argument("--output", type=Path, required=True)

    tidy = commands.add_parser("reorganize", help="Sort artifacts made by older versions for fast lookups")
    tidy.add_argument("--artifacts-dir", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "import-splits":
        stats = import_splits(args.train, args.validation, args.test, args.output_dir)
    elif args.command == "reorganize":
        stats = reorganize(args.artifacts_dir)
    else:
        stats = enrich_catalog(args.catalog, args.metadata, args.output)

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
