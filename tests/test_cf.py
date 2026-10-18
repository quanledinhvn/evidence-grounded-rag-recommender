from __future__ import annotations

import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scipy.sparse import csr_matrix

from backend.cf.bpr import BPR
from backend.cf.cli import main as cf_main
from backend.cf.common import Popularity, load_fit_data, top_k
from backend.cf.evaluate import evaluate
from backend.cf.itemknn import ItemKNN, cosine_top_k, recency_weights


def write_dataset(tmp_path):
    """Tiny dataset.

    Co-watched in train: i1-i3 (u2), i2-i4 (u3), i1-i2 (u1), i5-i6 (u4).
    u1 watched i1 before i2, so with recency weighting i2's neighbour i4 beats i1's neighbour i3.
    """
    rows = [
        ("u1", "i1", 5.0, 1, "train"),
        ("u1", "i2", 5.0, 2, "train"),
        ("u1", "i3", 5.0, 3, "validation"),
        ("u1", "i4", 5.0, 4, "test"),
        ("u2", "i1", 5.0, 1, "train"),
        ("u2", "i3", 4.0, 2, "train"),
        ("u3", "i2", 5.0, 1, "train"),
        ("u3", "i4", 3.0, 2, "train"),
        ("u4", "i5", 5.0, 1, "train"),
        ("u4", "i6", 2.0, 2, "train"),
        ("u5", "i1", 5.0, 9, "test"),
    ]
    columns = ["user_id", "item_id", "rating", "timestamp", "split"]
    pq.write_table(pa.Table.from_pylist([dict(zip(columns, r)) for r in rows]), tmp_path / "splits.parquet")
    pq.write_table(pa.table({"item_id": [f"i{n}" for n in range(1, 7)]}), tmp_path / "catalog.parquet")

    return tmp_path / "splits.parquet", tmp_path / "catalog.parquet"


def ids(items):
    return [item["item_id"] for item in items]


# ---------- fit data, popularity, top-k ----------

def test_fit_data_uses_only_allowed_splits_in_time_order(tmp_path):
    splits, catalog = write_dataset(tmp_path)

    validation = load_fit_data(splits, catalog, "validation")
    test = load_fit_data(splits, catalog, "test")

    history = lambda data, user: [data.item_ids[i] for i in data.history(user)[0]]
    assert history(validation, "u1") == ["i1", "i2"]
    assert history(test, "u1") == ["i1", "i2", "i3"]
    assert "u5" not in test.user_ids                    # only has a test row


def test_fit_data_rejects_items_outside_catalog(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    pq.write_table(pa.table({"item_id": ["i1"]}), catalog)

    with pytest.raises(ValueError, match="catalog"):
        load_fit_data(splits, catalog, "validation")


def test_popularity_counts_positive_ratings_and_skips_seen(tmp_path):
    data = load_fit_data(*write_dataset(tmp_path), "validation")
    popularity = Popularity(data)

    assert [data.item_ids[i] for i in popularity.order] == ["i1", "i2", "i3", "i5", "i4", "i6"]
    assert ids(popularity.recommend("u1", limit=3)) == ["i3", "i5", "i4"]


def test_top_k_breaks_ties_by_index():
    best, scores = top_k(np.array([5, 3, 9, 1]), np.array([0.5, 0.9, 0.5, -np.inf]), 2)

    assert best.tolist() == [3, 5]
    assert scores.tolist() == [0.9, 0.5]


# ---------- ItemKNN ----------

def test_recency_weights():
    assert recency_weights(3, 0.5).tolist() == [0.25, 0.5, 1.0]


def test_cosine_similarity_and_pruning():
    matrix = csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32))

    full = cosine_top_k(matrix, neighbors=5).toarray()
    pruned = cosine_top_k(matrix, neighbors=1, batch_size=2).toarray()

    expected = np.array([[0, 1 / math.sqrt(2), 0.5], [1 / math.sqrt(2), 0, 0], [0.5, 0, 0]])
    np.testing.assert_allclose(full, expected, rtol=1e-6)
    assert (pruned > 0).sum(axis=1).tolist() == [1, 1, 1]


def test_itemknn_ranks_recent_neighbours_first_then_popularity(tmp_path):
    data = load_fit_data(*write_dataset(tmp_path), "validation")
    model = ItemKNN.fit(data, neighbors=10, recency_decay=0.5)

    items = model.recommend("u1", limit=4)

    assert ids(items) == ["i4", "i3", "i5", "i6"]
    assert [item["source"] for item in items] == ["itemknn", "itemknn", "popularity", "popularity"]
    assert items[0]["score"] == pytest.approx(1 / math.sqrt(2))          # i2 (weight 1) -> i4
    assert items[1]["score"] == pytest.approx(0.5 / math.sqrt(2))        # i1 (weight 0.5) -> i3
    assert items[0]["because"] == [{"item_id": "i2", "similarity": pytest.approx(1 / math.sqrt(2)), "weight": 1.0}]


def test_itemknn_exclude_unknown_user_and_batch(tmp_path):
    data = load_fit_data(*write_dataset(tmp_path), "validation")
    model = ItemKNN.fit(data, neighbors=10, recency_decay=0.5)

    assert ids(model.recommend("u1", limit=10, exclude=("i4",))) == ["i3", "i5", "i6"]
    assert ids(model.recommend("unknown", limit=2)) == ["i1", "i2"]
    assert [ids(r) for r in model.recommend_batch(["u1", "unknown"], limit=2)] == [["i4", "i3"], ["i1", "i2"]]

    with pytest.raises(ValueError):
        model.recommend("u1", limit=501)


def test_itemknn_save_and_load(tmp_path):
    data = load_fit_data(*write_dataset(tmp_path), "validation")
    model = ItemKNN.fit(data, neighbors=10)
    model.save(tmp_path / "model")

    loaded = ItemKNN.load(tmp_path / "model")

    assert loaded.recommend("u1") == model.recommend("u1")
    assert loaded.config == model.config


# ---------- BPR ----------

def test_bpr_recommends_only_unseen_learned_items(tmp_path):
    data = load_fit_data(*write_dataset(tmp_path), "validation")
    model = BPR.fit(data, factors=4, epochs=20, num_threads=1)
    model.save(tmp_path / "bpr")

    items = BPR.load(tmp_path / "bpr").recommend("u1", limit=4)
    bpr_items = [item["item_id"] for item in items if item["source"] == "bpr"]

    assert not {"i1", "i2"} & set(ids(items))          # seen
    assert not {"i4", "i6"} & set(bpr_items)           # never rated >= 4, so not learned
    assert {item["source"] for item in model.recommend("unknown", limit=3)} == {"popularity"}


# ---------- evaluation and CLI ----------

def test_evaluate_reports_ranking_metrics(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    model = ItemKNN.fit(load_fit_data(splits, catalog, "validation"), neighbors=10)
    queries = tmp_path / "queries.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0},       # ItemKNN rank 2
                {"user_id": "unknown", "target_item_id": "i1", "target_rating": 2.0},  # popularity rank 1
            ]
        ),
        queries,
    )

    report = evaluate(model, queries)

    assert report["ndcg_at_10"] == pytest.approx((1 / math.log2(3) + 1.0) / 2)
    assert report["hr_at_10"] == 1.0
    assert report["fallback_rate"] == 0.5
    assert report["positive_target"]["queries"] == 1
    assert report["positive_target"]["ndcg_at_10"] == pytest.approx(1 / math.log2(3))


def test_evaluate_also_counts_another_edition_of_the_target_film(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    model = ItemKNN.fit(load_fit_data(splits, catalog, "validation"), neighbors=10)
    queries = tmp_path / "queries.parquet"
    pq.write_table(pa.Table.from_pylist([{"user_id": "u1", "target_item_id": "i5", "target_rating": 5.0}]), queries)

    ranked = ids(model.recommend("u1", limit=50))                  # i4, i3, then popular items
    report = evaluate(model, queries, editions={"i5": "i3", "i3": "i3"})

    assert report["ndcg_at_10"] == pytest.approx(1 / math.log2(ranked.index("i5") + 2))
    assert report["same_film"]["ndcg_at_10"] == pytest.approx(1 / math.log2(3))   # i3 at rank 2 is the same film


def test_cli_train_recommend_evaluate(tmp_path, capsys):
    splits, catalog = write_dataset(tmp_path)
    model_dir = tmp_path / "model"
    queries = tmp_path / "queries.parquet"
    pq.write_table(pa.Table.from_pylist([{"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0}]), queries)

    cf_main(["train", "--model", "itemknn", "--splits", str(splits), "--catalog", str(catalog),
             "--phase", "validation", "--output-dir", str(model_dir)])
    capsys.readouterr()

    cf_main(["recommend", "--model-dir", str(model_dir), "--user-id", "u1", "--limit", "2"])
    assert ids(json.loads(capsys.readouterr().out)["items"]) == ["i4", "i3"]

    cf_main(["evaluate", "--model-dir", str(model_dir), "--queries", str(queries), "--popularity",
             "--output", str(tmp_path / "report.json")])
    report = json.loads(capsys.readouterr().out)
    assert report["fallback_rate"] == 1.0
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report


def test_segments_split_queries_by_history_popularity_and_title(tmp_path):
    from backend.cf.segments import history_segment, popularity_segment, segment_report

    assert [history_segment(n) for n in (1, 2, 3, 6, 21)] == ["1-2", "1-2", "3-5", "6-20", ">20"]
    assert [popularity_segment(r, 1000) for r in (0, 9, 10, 99, 100)] == ["head", "head", "torso", "torso", "tail"]

    splits, catalog = write_dataset(tmp_path)
    model = ItemKNN.fit(load_fit_data(splits, catalog, "validation"), neighbors=10)
    queries = [
        {"user_id": "u1", "target_item_id": "i3", "target_rating": 5.0},   # ItemKNN rank 2, 2 items seen
        {"user_id": "u4", "target_item_id": "i1", "target_rating": 5.0},
    ]

    report = segment_report({"itemknn": model}, queries, titled={"i3"})

    assert report["all"]["queries"] == 2
    assert report["history"]["1-2"]["queries"] == 2
    assert report["title"]["with title"]["itemknn"]["ndcg_at_10"] == pytest.approx(1 / math.log2(3))
    assert report["title"]["without title"]["queries"] == 1
