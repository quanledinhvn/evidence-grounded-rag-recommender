from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.profile.cli import main as profile_main
from backend.profile.profile import DAY_MS, build_profile, load_catalog, load_history, signal_weight


AS_OF = 100 * DAY_MS


def row(item_id, rating, day, split="train", review="r"):
    return {
        "user_id": "u1",
        "item_id": item_id,
        "rating": rating,
        "timestamp": day * DAY_MS,
        "review_id": f"{review}-{item_id}-{day}",
        "split": split,
    }


def write_splits(tmp_path):
    rows = [
        row("i1", 5.0, 10),
        row("i2", 2.0, 20),
        row("i3", 4.0, 30, split="validation"),
        row("i4", 5.0, 100),                 # at as_of: must be excluded
        row("i5", 5.0, 50, split="test"),    # test split: never used
        {**row("i6", 5.0, 5), "user_id": "u2"},
    ]
    path = tmp_path / "splits.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    return path


def test_history_respects_phase_and_as_of(tmp_path):
    splits = write_splits(tmp_path)

    validation = load_history(splits, "u1", AS_OF, "validation")
    test = load_history(splits, "u1", AS_OF, "test")

    assert sorted(r["item_id"] for r in validation) == ["i1", "i2"]
    assert sorted(r["item_id"] for r in test) == ["i1", "i2", "i3"]


def test_profile_splits_signals_by_rating_newest_first():
    history = [row("i1", 5.0, 10), row("i2", 2.0, 20), row("i3", 4.0, 30), row("i7", 3.0, 40)]
    catalog = {"i1": {"item_id": "i1", "title": "Inception", "details": "", "source_id": "meta:i1"}}

    profile = build_profile("u1", AS_OF, history, catalog)

    assert profile["seen_item_ids"] == ["i1", "i2", "i3", "i7"]          # neutral i7 is still seen
    assert [s["item_id"] for s in profile["positive_signals"]] == ["i3", "i1"]
    assert [s["history_id"] for s in profile["positive_signals"]] == ["H1", "H2"]
    assert [s["history_id"] for s in profile["negative_signals"]] == ["H3"]
    assert profile["positive_signals"][1]["text_fields"] == [
        {"name": "title", "text": "Inception", "source_id": "meta:i1:title"}
    ]
    assert profile["items_without_metadata"] == ["i2", "i3", "i7"]


def test_latest_interaction_wins_and_future_rows_are_ignored():
    history = [row("i1", 5.0, 10), row("i1", 1.0, 20), row("i2", 5.0, 100)]

    profile = build_profile("u1", AS_OF, history, {})

    assert profile["positive_signals"] == []
    assert [s["item_id"] for s in profile["negative_signals"]] == ["i1"]
    assert "i2" not in profile["seen_item_ids"]


def test_weight_combines_rating_strength_and_half_life():
    assert signal_weight(row("i1", 5.0, 100), AS_OF, 90.0) == pytest.approx(1.0)
    assert signal_weight(row("i1", 4.0, 10), AS_OF, 90.0) == pytest.approx(0.25)   # 0.5 strength, 90 days old


def test_cli_prints_profile(tmp_path, capsys):
    splits = write_splits(tmp_path)
    pq.write_table(pa.Table.from_pylist([{"item_id": "i1", "title": "Inception", "source_id": "meta:i1"}]),
                   tmp_path / "catalog.parquet")

    profile_main(["--splits", str(splits), "--catalog", str(tmp_path / "catalog.parquet"),
                  "--user-id", "u1", "--as-of", str(AS_OF), "--phase", "test"])
    profile = json.loads(capsys.readouterr().out)

    assert profile["seen_item_ids"] == ["i1", "i2", "i3"]
    assert load_catalog(tmp_path / "catalog.parquet", set()) == {}
