from __future__ import annotations

import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from backend.cf.common import load_fit_data
from backend.cf.itemknn import ItemKNN
from backend.data.reviews import SCHEMA
from backend.retrieval.bm25 import BM25
from backend.retrieval.fusion import Fusion
from backend.service import recommend

from tests.test_retrieval import write_dataset


class FakeLLM:
    def __init__(self, *texts):
        self.texts = list(texts)

    def generate(self, prompt, max_output_tokens=512, schema=None):
        return {"text": self.texts.pop(0), "input_tokens": 100, "output_tokens": 10, "latency_ms": 5.0, "cache_hit": False, "error": None}


def setup(tmp_path):
    splits, catalog = write_dataset(tmp_path)
    table = pq.read_table(splits)
    pq.write_table(table.append_column("review_id", pa.array([f"r{n}" for n in range(table.num_rows)])), splits)

    data = load_fit_data(splits, catalog, "validation")
    fusion = Fusion({"itemknn": ItemKNN.fit(data, neighbors=10), "bm25": BM25.fit(data, catalog, min_df=1, max_df=1.0)},
                    {"itemknn": 1.0, "bm25": 1.0, "popularity": 0.1}, depth=10)

    (tmp_path / "catalog_enriched.parquet").write_bytes(catalog.read_bytes())
    connection = sqlite3.connect(tmp_path / "reviews.sqlite")
    connection.execute(SCHEMA)
    connection.commit()

    return fusion


def test_recommend_keeps_fusion_order_and_only_valid_claims(tmp_path):
    fusion = setup(tmp_path)

    # Fusion order for u1 is i5, i2, i3, i6; E1 is the cast of i2 (C2 in the explanation prompt).
    explanation = {"items": [
        {"item_id": "C1", "claims": [{"text": "Invented source", "evidence_ids": ["E999"], "history_ids": []}]},
        {"item_id": "C2", "claims": [{"text": "It has the same kind of cast", "evidence_ids": ["E1"], "history_ids": []}]},
    ]}
    result = recommend("u1", 100, "validation", fusion, tmp_path, {}, FakeLLM(json.dumps(explanation)))
    first, second, third = result["recommendations"][:3]

    assert [entry["item_id"] for entry in result["recommendations"]] == [item["item_id"] for item in fusion.recommend("u1", 10)]
    assert result["status"] == "ok"
    assert first["explanation"] is None and first["explanation_status"] == "evidence_unavailable"
    assert second["explanation"] == "It has the same kind of cast [E1]." and second["explanation_status"] == "citations_valid"
    assert third["explanation_status"] == "evidence_unavailable"                                 # the LLM gave no claim
    assert list(result["evidence"]) == ["E1"] and result["evidence"]["E1"]["item_id"] == second["item_id"]
    assert [call["stage"] for call in result["trace"]["llm_calls"]] == ["explain"]
    assert {"i1", "i4"}.isdisjoint(entry["item_id"] for entry in result["recommendations"])      # seen items


def test_user_without_history_gets_no_llm_call(tmp_path):
    result = recommend("stranger", 100, "validation", setup(tmp_path), tmp_path, {}, FakeLLM())

    assert result["status"] == "cold_start"
    assert all(entry["explanation_status"] == "not_requested" for entry in result["recommendations"])
