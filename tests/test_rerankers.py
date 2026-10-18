from __future__ import annotations

import numpy as np

from backend.cf.common import load_fit_data
from backend.cf.itemknn import ItemKNN
from backend.retrieval.bm25 import BM25
from backend.retrieval.fusion import Fusion
from backend.retrieval.rerankers import FEATURES, CrossEncoderReranker, LTRReranker, pool_features, user_query

from tests.test_generation import PACKAGE
from tests.test_retrieval import write_dataset


def test_pool_features_has_one_row_per_pooled_item(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    data = load_fit_data(splits, catalog, "validation")
    fusion = Fusion({"itemknn": ItemKNN.fit(data, neighbors=10), "bm25": BM25.fit(data, catalog, min_df=1, max_df=1.0)},
                    {"itemknn": 1.0, "bm25": 1.0, "popularity": 0.1}, depth=10)
    vectors = np.eye(6, dtype=np.float32)

    (items, X), = pool_features(fusion, ["u1"], vectors, pool=3)

    assert items == [item["item_id"] for item in fusion.recommend("u1", 3)]
    assert X.shape == (3, len(FEATURES))
    assert X[:, FEATURES.index("fusion_rank")].tolist() == [1, 2, 3]
    assert X[:, FEATURES.index("history_length")].tolist() == [2, 2, 2]


def test_ltr_learns_which_feature_marks_the_target(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((400, len(FEATURES)))
    y = (X[:, 3] > 0.8).astype(int)                    # the 4th feature decides

    model = LTRReranker().fit(X, y)
    items = ["a", "b", "c"]
    pool_X = np.zeros((3, len(FEATURES)))
    pool_X[:, 3] = [0.1, 0.95, 0.9]                    # a is below the learnt threshold, b and c above

    assert model.rerank(items, pool_X, k=2) == ["b", "c"]

    model.save(tmp_path / "ltr.pkl")
    assert LTRReranker.load(tmp_path / "ltr.pkl").rerank(items, pool_X, k=2) == ["b", "c"]


def test_user_query_lists_liked_items():
    query = user_query(PACKAGE)

    assert "Inception" in query and "Loved the dream layers" not in query


def test_cross_encoder_orders_the_pool_by_pair_score():
    def predict(pairs):                                # higher score for the Nolan film
        return np.array([1.0 if "Nolan" in doc else 0.0 for _, doc in pairs])

    reranker = CrossEncoderReranker(predict=predict)

    assert reranker.rerank(PACKAGE, ["B0TITANIC", "B0INTERST", "B0NOTITLE"], k=2) == ["B0INTERST", "B0TITANIC"]
