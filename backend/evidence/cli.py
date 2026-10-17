"""recsys-evidence: candidates + evidence package for one user, ready to give to an LLM.

Pipeline for one query (user_id, as_of):
    profile (backend/profile) -> top candidates from a saved model (e.g. the fusion)
    -> evidence package (backend/evidence/evidence.py)
"""

from __future__ import annotations

import argparse
import json
import sqlite3

from pathlib import Path

from backend.cf.cli import load_model
from backend.data.io import write_json
from backend.evidence.evidence import evidence_package
from backend.profile.profile import build_profile, load_catalog, load_history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-evidence")
    parser.add_argument("--model-dir", type=Path, required=True, help="model that proposes candidates")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True, help="catalog_enriched.parquet")
    parser.add_argument("--reviews", type=Path, help="reviews.sqlite (optional)")

    parser.add_argument("--user-id", required=True)
    parser.add_argument("--as-of", type=int, required=True)
    parser.add_argument("--phase", choices=["validation", "test"], required=True)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)

    history = load_history(args.splits, args.user_id, args.as_of, args.phase)
    profile = build_profile(args.user_id, args.as_of, history, load_catalog(args.catalog, {r["item_id"] for r in history}))

    # The profile's seen items are excluded too, so nothing the user already watched is proposed.
    items = load_model(args.model_dir).recommend(args.user_id, args.candidates, tuple(profile["seen_item_ids"]))
    candidates = [item["item_id"] for item in items]

    connection = sqlite3.connect(args.reviews) if args.reviews else None
    package = evidence_package(profile, candidates, load_catalog(args.catalog, set(candidates)), connection)
    package["retrieval"] = items

    if args.output:
        write_json(args.output, package)
    else:
        print(json.dumps(package, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
