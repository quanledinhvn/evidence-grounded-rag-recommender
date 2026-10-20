"""recsys-profile: build the preference profile of one user at one point in time."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

from backend.data.io import write_json
from backend.profile.profile import build_profile, load_catalog, load_history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recsys-profile")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--user-id", required=True)

    parser.add_argument("--as-of", type=int, required=True, help="query time, Unix milliseconds")
    parser.add_argument("--phase", choices=["validation", "test"], required=True)
    parser.add_argument("--output", type=Path, help="write the profile here instead of printing it")

    args = parser.parse_args(argv)

    history = load_history(args.splits, args.user_id, args.as_of, args.phase)
    catalog = load_catalog(args.catalog, {row["item_id"] for row in history})
    profile = build_profile(args.user_id, args.as_of, history, catalog)

    if args.output:
        write_json(args.output, profile)
    else:
        print(json.dumps(profile, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
