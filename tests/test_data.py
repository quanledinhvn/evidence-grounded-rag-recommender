from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.data.cli import main as data_main
from backend.data.enrich import clean_text, enrich_catalog
from backend.data.importer import import_splits


HEADER = "user_id,parent_asin,rating,timestamp,history\n"


def write_csvs(tmp_path, train_rows, validation_rows, test_rows):
    paths = []
    for name, rows in (("train", train_rows), ("valid", validation_rows), ("test", test_rows)):
        path = tmp_path / f"{name}.csv"
        path.write_text(HEADER + "".join(row + "\n" for row in rows), encoding="utf-8")

        paths.append(path)

    return paths


def test_import_keeps_splits_and_builds_queries_and_catalog(tmp_path):
    train, valid, test = write_csvs(
        tmp_path,
        ["u1,i1,5.0,100,", "u1,i2,4.0,200,i1"],
        ["u1,i3,3.0,300,i1 i2"],
        ["u1,i4,5.0,400,i1 i2 i3"],
    )

    stats = import_splits(train, valid, test, tmp_path / "out")

    splits = pq.read_table(tmp_path / "out" / "splits.parquet").to_pylist()
    queries = pq.read_table(tmp_path / "out" / "queries_test.parquet").to_pylist()
    catalog = pq.read_table(tmp_path / "out" / "catalog.parquet")["item_id"].to_pylist()

    assert [row["split"] for row in splits] == ["train", "train", "validation", "test"]
    assert splits[1]["review_id"] == "amazon-2023:train:3"
    assert queries == [
        {
            "user_id": "u1",
            "as_of": 400,
            "target_item_id": "i4",
            "target_rating": 5.0,
            "target_timestamp": 400,
            "target_review_id": "amazon-2023:test:2",
        }
    ]
    assert catalog == ["i1", "i2", "i3", "i4"]
    assert stats["split_counts"] == {"train": 2, "validation": 1, "test": 1}


@pytest.mark.parametrize(
    "bad_row, message",
    [
        ("u1,i1,7.0,100,", "rating"),
        ("u1,i1,5.0,0,", "timestamp"),
        ("u1,i1,5.0,100,i1", "history"),
    ],
)
def test_import_rejects_invalid_rows(tmp_path, bad_row, message):
    train, valid, test = write_csvs(tmp_path, [bad_row], ["u1,i2,5.0,200,"], ["u1,i3,5.0,300,"])

    with pytest.raises(ValueError, match=message):
        import_splits(train, valid, test, tmp_path / "out")


def test_enrich_joins_metadata_by_parent_asin(tmp_path):
    pq.write_table(pa.table({"item_id": ["i1", "i2"]}), tmp_path / "catalog.parquet")
    records = [
        {
            "parent_asin": "i1",
            "title": "Tom &amp; Jerry <b>Show</b>",
            "main_category": "Movies & TV",
            "categories": ["Movies & TV", "Kids", "Kids"],
            "details": {"Directors": "Hanna", "Unused": "x"},
            "average_rating": 4.5,
            "rating_number": 10,
        },
        {"parent_asin": "i1", "title": "shorter duplicate"},
        {"parent_asin": "i9", "title": "not in catalog"},
    ]
    (tmp_path / "meta.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    stats = enrich_catalog(tmp_path / "catalog.parquet", tmp_path / "meta.jsonl", tmp_path / "enriched.parquet")
    rows = {row["item_id"]: row for row in pq.read_table(tmp_path / "enriched.parquet").to_pylist()}

    assert rows["i1"]["title"] == "Tom & Jerry Show"
    assert rows["i1"]["category"] == "Kids"
    assert rows["i1"]["details"] == "Directors: Hanna"
    assert rows["i1"]["metadata_text"] == "Tom & Jerry Show Movies & TV Kids Directors: Hanna"
    assert rows["i1"]["source_id"] == "meta:i1"
    assert rows["i2"]["metadata_text"] == ""
    assert stats["catalog_items_with_metadata"] == 1


def test_clean_text_handles_lists_and_html():
    assert clean_text(["a<br/>b", None, "  c  "]) == "a b c"
    assert clean_text(None) == ""


def test_cli_runs_both_commands(tmp_path, capsys):
    train, valid, test = write_csvs(tmp_path, ["u1,i1,5.0,100,"], ["u1,i2,5.0,200,i1"], ["u1,i3,5.0,300,i1 i2"])
    (tmp_path / "meta.jsonl").write_text(json.dumps({"parent_asin": "i1", "title": "T"}), encoding="utf-8")

    out = tmp_path / "out"
    assert data_main(["import-splits", "--train", str(train), "--validation", str(valid),
                      "--test", str(test), "--output-dir", str(out)]) == 0
    assert data_main(["enrich-catalog", "--catalog", str(out / "catalog.parquet"),
                      "--metadata", str(tmp_path / "meta.jsonl"), "--output", str(out / "enriched.parquet")]) == 0

    assert (out / "enriched.stats.json").exists()
