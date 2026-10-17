from __future__ import annotations

import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from backend.data.reviews import import_reviews, item_reviews, user_review


LONG = "a clever and moving story with great acting from the whole cast"   # 12 words


def write_inputs(tmp_path):
    reviews = [
        {"parent_asin": "i1", "user_id": "u1", "timestamp": 100, "rating": 5.0, "title": "Wow", "text": LONG, "helpful_vote": 1},
        {"parent_asin": "i1", "user_id": "u2", "timestamp": 200, "rating": 4.0, "title": "", "text": LONG + " again", "helpful_vote": 9},
        {"parent_asin": "i1", "user_id": "u3", "timestamp": 300, "rating": 5.0, "title": "", "text": LONG, "helpful_vote": 0},
        {"parent_asin": "i1", "user_id": "u4", "timestamp": 50, "rating": 5.0, "title": "", "text": "Five stars", "helpful_vote": 0},
        {"parent_asin": "i1", "user_id": "u5", "timestamp": 60, "rating": 1.0, "title": "", "text": LONG + " but the disc arrived scratched", "helpful_vote": 0},
        {"parent_asin": "i9", "user_id": "u1", "timestamp": 70, "rating": 5.0, "title": "", "text": LONG, "helpful_vote": 0},
    ]
    (tmp_path / "reviews.jsonl").write_text("\n".join(json.dumps(r) for r in reviews), encoding="utf-8")
    pq.write_table(pa.table({"item_id": ["i1", "i2"]}), tmp_path / "catalog.parquet")

    return tmp_path / "reviews.jsonl", tmp_path / "catalog.parquet"


def test_import_keeps_only_useful_reviews_of_catalog_items(tmp_path):
    reviews, catalog = write_inputs(tmp_path)

    counts = import_reviews(reviews, catalog, tmp_path / "reviews.sqlite")

    assert counts == {"read": 6, "not_in_catalog": 1, "too_short": 1, "not_about_film": 1, "kept": 3}


def test_item_reviews_respect_time_and_exclude_the_user(tmp_path):
    reviews, catalog = write_inputs(tmp_path)
    import_reviews(reviews, catalog, tmp_path / "reviews.sqlite")
    connection = sqlite3.connect(tmp_path / "reviews.sqlite")

    found = item_reviews(connection, "i1", before=300, exclude_user="u1")

    assert [review["user_id"] for review in found] == ["u2"]          # u1 excluded, u3 is not before 300
    assert user_review(connection, "u1", "i1", before=300)["title"] == "Wow"
    assert user_review(connection, "u1", "i1", before=100) is None   # strictly before
