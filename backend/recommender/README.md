# Pipeline-aware recommender and PAEC

Package này hiện thực lõi nghiên cứu trong `research/systemdessign.md`:

- ba retriever ItemKNN, BM25 và semantic trả cả score lẫn provenance theo raw history source;
- candidate union, chuẩn hóa trong query và fusion tất định;
- full-pipeline removal/retention giữ nguyên factual `seen_item_ids`;
- signed top-K margin, evaluable-state gate và decision certificate;
- PAEC level-wise exact search cho history ngắn, bounded beam search cho history dài;
- contribution, recency, semantic, random và leave-one-out baselines;
- attribution–intervention gap audit;
- deterministic template renderer, citation verifier và abstention.

## Luồng sử dụng

```python
from backend.recommender import (
    HybridPipeline,
    InterventionHarness,
    PAECSelector,
    PipelineConfig,
)

config = PipelineConfig(
    top_k=10,
    pool_size=100,
    evidence_budget=3,
    weights={"itemknn": 0.4, "bm25": 0.3, "semantic": 0.3},
)

pipeline = HybridPipeline(
    {
        "itemknn": itemknn_retriever,
        "bm25": bm25_retriever,
        "semantic": semantic_retriever,
    },
    config,
    artifact_manifest_hash="sha256-of-locked-artifact-manifest",
    fallback_item_ids=locked_popularity_order,
)

factual = pipeline.run(profile)
harness = InterventionHarness(pipeline)
selection = PAECSelector(harness, support_gate=raw_support_gate).select(
    profile,
    factual,
    factual.ranking[0].item_id,
)
```

`FrozenContributionRetriever` dành cho fixture và replay artifact. Ba adapter triển
khai dùng cho experiment là:

- `ItemKNNContributionRetriever`: đọc trực tiếp similarity matrix đã khóa và tính
  lại từng edge từ history source đang active;
- `BM25ContributionRetriever`: BM25 tất định, phân bổ contribution của term dùng
  chung sao cho không bị đếm lặp;
- `SemanticContributionRetriever`: cosine được tính lại chính xác trên embedding
  đã khóa, contribution cộng theo từng history–candidate pair.

OpenSearch trong `infra/opensearch/compose.yaml` phục vụ indexing và candidate
generation quy mô lớn. Score/provenance cuối vẫn phải được tính lại tất định bằng
adapter tương ứng trước khi fusion. Không dùng ANN score trực tiếp làm exact
contribution.

## Bất biến được thực thi

- Source can thiệp phải thuộc positive history của profile.
- Mọi atom trong rerun phải thuộc tập source còn active.
- Fused score phải bằng tổng normalized source contribution trong tolerance.
- Rerun giữ nguyên `seen_item_ids` của factual profile.
- Empty positive history hoặc ít hơn `top_k + 1` candidate sau fallback là
  non-evaluable.
- Certificate lệch artifact manifest hoặc config hash không thể dual-valid.
- Renderer chỉ phát biểu dual validity khi certificate evaluable và bundle có đủ
  history citation cùng certificate citation; trường hợp khác sẽ abstain.

## Kiểm thử

```bash
.venv/bin/pytest -q tests/test_recommender_pipeline.py
.venv/bin/pytest -q tests/test_recommender_retrievers.py
.venv/bin/pytest -q tests/test_recommender_audit.py
```

Các test bao phủ trace identity, deterministic rerun, source closure, seen-item
invariance, candidate regeneration, evaluable gate, certificate integrity,
selector parity, renderer isolation, fallback/abstention và level-wise exact
coalition search.

Selector đánh giá coalition theo kích thước tăng dần. Nó đánh giá hết một tầng để
giữ tie-break theo removal effect và retention loss, nhưng dừng trước tầng lớn hơn
ngay khi tầng hiện tại có coalition dual-valid qua support gate. Quy tắc này vẫn
cho nghiệm exact theo objective ưu tiên coalition nhỏ hơn; nếu không có coalition
hợp lệ, selector tiếp tục đến hết evidence budget rồi abstain.
