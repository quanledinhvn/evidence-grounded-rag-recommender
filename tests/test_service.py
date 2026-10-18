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


def test_recommend_reranks_explains_and_keeps_only_valid_claims(tmp_path):
    fusion = setup(tmp_path)
    explanation = {"items": [
        {"item_id": "C1", "claims": [{"text": "It has the same kind of cast", "evidence_ids": ["E1"], "history_ids": []}]},
        {"item_id": "C2", "claims": [{"text": "Invented source", "evidence_ids": ["E999"], "history_ids": []}]},
    ]}
    # Pool (fusion order) is i5, i2, i3, i6; E1 is the cast of i2, so the LLM puts C2 (= i2) first.
    llm = FakeLLM('{"item_ids": ["C2", "C1"]}', json.dumps(explanation))

    result = recommend("u1", 100, "validation", fusion, tmp_path, {}, llm)
    first, second = result["recommendations"][:2]

    assert result["status"] == "ok" and result["rerank_status"] == "llm_completed"
    assert first["explanation"] == "It has the same kind of cast [E1]." and first["explanation_status"] == "citations_valid"
    assert second["explanation"] is None and second["explanation_status"] == "evidence_unavailable"
    assert list(result["evidence"]) == ["E1"] and result["evidence"]["E1"]["item_id"] == first["item_id"]
    assert [call["stage"] for call in result["trace"]["llm_calls"]] == ["rerank", "explain"]
    assert {"i1", "i4"}.isdisjoint(entry["item_id"] for entry in result["recommendations"])      # seen items


def test_user_without_history_gets_no_llm_call(tmp_path):
    result = recommend("stranger", 100, "validation", setup(tmp_path), tmp_path, {}, FakeLLM())

    assert result["status"] == "cold_start" and result["rerank_status"] == "skipped"
    assert all(entry["explanation_status"] == "not_requested" for entry in result["recommendations"])
