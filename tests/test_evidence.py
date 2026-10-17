from __future__ import annotations

import sqlite3

from backend.data.reviews import SCHEMA
from backend.evidence.evidence import evidence_package


def make_reviews(tmp_path):
    connection = sqlite3.connect(tmp_path / "reviews.sqlite")
    connection.execute(SCHEMA)
    rows = [
        ("review:1", "i2", "u9", 50, 5.0, "", "Nolan bends time again, a clever puzzle", 1),
        ("review:2", "i2", "u8", 60, 5.0, "", "Great soundtrack and visuals", 7),
        ("review:3", "i2", "u7", 500, 5.0, "", "Written after the query time, must not appear", 99),
        ("review:4", "i2", "u1", 40, 5.0, "", "The user's own review of the candidate", 50),
        ("review:5", "i1", "u1", 10, 5.0, "", "Loved the dream layers", 0),
    ]
    connection.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    return connection


PROFILE = {
    "user_id": "u1",
    "as_of": 100,
    "positive_signals": [
        {
            "history_id": "H1",
            "item_id": "i1",
            "rating": 5.0,
            "sentiment": "positive",
            "text_fields": [{"name": "title", "text": "Inception, a Nolan puzzle", "source_id": "meta:i1:title"}],
        }
    ],
    "negative_signals": [],
}

CATALOG = {
    "i2": {
        "title": "Interstellar",
        "genres": "Science Fiction",
        "description": "",
        "item_doc": "Interstellar. Genres: Science Fiction. Directors: Christopher Nolan. Cast: Matthew McConaughey",
        "source_id": "meta:i2",
    }
}


def test_history_evidence_uses_the_users_own_review(tmp_path):
    package = evidence_package(PROFILE, ["i2"], CATALOG, make_reviews(tmp_path))

    assert package["history"]["H1"]["own_review"] == "Loved the dream layers"
    assert package["history"]["H1"]["source_id"] == "review:5"


def test_item_evidence_has_metadata_and_earlier_reviews_by_others(tmp_path):
    package = evidence_package(PROFILE, ["i2"], CATALOG, make_reviews(tmp_path))
    evidence = package["evidence"]

    metadata = [e["source_id"] for e in evidence.values() if e["source_type"] == "metadata"]
    reviews = [e["source_id"] for e in evidence.values() if e["source_type"] == "review"]

    assert metadata == ["meta:i2:title", "meta:i2:genres", "meta:i2:people"]
    assert reviews == ["review:1", "review:2"]     # review:1 shares "nolan"/"puzzle" with the liked item
    assert list(evidence) == ["E1", "E2", "E3", "E4", "E5"]


def test_works_without_a_review_store():
    package = evidence_package(PROFILE, ["i2"], CATALOG, None)

    assert all(e["source_type"] == "metadata" for e in package["evidence"].values())
    assert package["history"]["H1"]["own_review"] is None
