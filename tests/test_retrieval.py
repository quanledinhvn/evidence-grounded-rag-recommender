from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.cf.cli import load_model
from backend.cf.common import load_fit_data
from backend.cf.itemknn import ItemKNN
from backend.retrieval.bm25 import BM25
from backend.retrieval.content import liked_history
from backend.retrieval.dense import Dense
from backend.retrieval.fusion import Fusion, fuse


DOCS = {
    "i1": "Inception. Directors: Christopher Nolan. Cast: Leonardo DiCaprio",
    "i2": "Interstellar. Directors: Christopher Nolan. Cast: Matthew McConaughey",
    "i3": "Titanic. Directors: James Cameron. Cast: Leonardo DiCaprio",
    "i4": "Toy Story. Genres: Animation",
    "i5": "Finding Nemo. Genres: Animation",
    "i6": "",
}


def write_dataset(tmp_path):
    """u1 liked Inception and disliked Toy Story; u3 has no liked item."""
    rows = [
        ("u1", "i1", 5.0, 1, "train"),
        ("u1", "i4", 2.0, 2, "train"),
        ("u2", "i4", 5.0, 1, "train"),
        ("u2", "i5", 5.0, 2, "train"),
        ("u3", "i3", 2.0, 1, "train"),
    ]
    columns = ["user_id", "item_id", "rating", "timestamp", "split"]
    pq.write_table(pa.Table.from_pylist([dict(zip(columns, r)) for r in rows]), tmp_path / "splits.parquet")
    pq.write_table(pa.table({"item_id": list(DOCS), "item_doc": list(DOCS.values())}), tmp_path / "catalog.parquet")

    return tmp_path / "splits.parquet", tmp_path / "catalog.parquet"


def ids(items):
    return [item["item_id"] for item in items]


def test_liked_history_keeps_recent_likes_with_recency_weights(tmp_path):
    data = load_fit_data(*write_dataset(tmp_path), "validation")

    items, weights = liked_history(data, "u2", recency_decay=0.5, max_items=20)

    assert [data.item_ids[i] for i in items] == ["i4", "i5"]
    assert weights.tolist() == [0.5, 1.0]


def test_bm25_finds_items_sharing_rare_words(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    model = BM25.fit(load_fit_data(splits, catalog, "validation"), catalog, min_df=1, max_df=1.0)

    items = model.recommend("u1", limit=4)

    assert set(ids(items)[:2]) == {"i2", "i3"}                  # same director / same actor as Inception
    assert [item["source"] for item in items[:2]] == ["bm25", "bm25"]
    assert "i4" not in ids(items)                                # seen (even if disliked)
    assert {item["source"] for item in model.recommend("u3", limit=3)} == {"popularity"}


def test_bm25_save_and_load(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    model = BM25.fit(load_fit_data(splits, catalog, "validation"), catalog, min_df=1, max_df=1.0)
    model.save(tmp_path / "bm25")

    assert BM25.load(tmp_path / "bm25").recommend("u1") == model.recommend("u1")


def test_dense_ranks_by_cosine_to_liked_items(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    vectors = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.7, 0.7], [-1, 0], [0, 0]], dtype=np.float32)
    np.save(tmp_path / "vectors.npy", vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9))

    model = Dense.fit(load_fit_data(splits, catalog, "validation"), catalog, tmp_path / "vectors.npy")
    model.save(tmp_path / "dense")

    items = Dense.load(tmp_path / "dense").recommend("u1", limit=4)

    assert ids(items) == ["i2", "i3", "i5", "i6"]                # i6 has no document: only via popularity
    assert [item["source"] for item in items] == ["dense", "dense", "dense", "popularity"]
    assert {item["source"] for item in model.recommend("u3", limit=3)} == {"popularity"}

def test_fuse_sums_weighted_reciprocal_ranks():
    lists = {
        "itemknn": [
            {"item_id": "a", "source": "itemknn"},
            {"item_id": "b", "source": "itemknn"},
            {"item_id": "p", "source": "popularity"},     # padding: earns nothing for itemknn
        ],
        "bm25": [{"item_id": "b", "source": "bm25"}],
        "popularity": [{"item_id": "p", "source": "popularity"}],
    }

    fused = fuse(lists, {"itemknn": 1.0, "bm25": 1.0, "popularity": 0.5}, k=1.0)

    assert [entry["item_id"] for entry in fused] == ["b", "a", "p"]
    assert fused[0]["score"] == pytest.approx(1 / 3 + 1 / 2)    # rank 2 in itemknn, rank 1 in bm25
    assert fused[0]["members"] == ["itemknn", "bm25"]
    assert fused[2]["score"] == pytest.approx(0.5 / 2)


def test_fusion_model_combines_members_and_reloads(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    data = load_fit_data(splits, catalog, "validation")
    ItemKNN.fit(data, neighbors=10).save(tmp_path / "itemknn")
    BM25.fit(data, catalog, min_df=1, max_df=1.0).save(tmp_path / "bm25")

    member_dirs = {"itemknn": str(tmp_path / "itemknn"), "bm25": str(tmp_path / "bm25")}
    members = {name: load_model(Path(path)) for name, path in member_dirs.items()}
    fusion = Fusion(members, {"itemknn": 1.0, "bm25": 1.0, "popularity": 0.1})
    fusion.save(tmp_path / "fusion", member_dirs)

    items = load_model(tmp_path / "fusion").recommend("u1", limit=3)

    assert items == fusion.recommend("u1", limit=3)
    assert {"i2", "i3"} <= set(ids(items))
    assert all("members" in item for item in items)
