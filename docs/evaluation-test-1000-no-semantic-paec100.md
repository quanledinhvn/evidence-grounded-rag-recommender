# Báo cáo đánh giá: 1.000 query, không semantic, PAEC 100 query

## Phạm vi chạy

| Thuộc tính | Giá trị |
|---|---|
| Artifact version | `movies-tv-2023` |
| Query input | `artifacts/movies-tv-2023/queries_test_1000.parquet` |
| Query hoàn tất | 1.000 |
| Query positive, target rating từ 4 trở lên | 814 |
| Seed lấy mẫu | 29 |
| Semantic retrieval | Tắt |
| Ranker hybrid | ItemKNN 0,5 + BM25 0,5 |
| Top-K | 10 |
| Candidate pool mỗi branch | 50 |
| PAEC limit | 100 query |
| PAEC hoàn tất | 100 query |
| LLM explanation | Không chạy trong evaluator |

Nguồn kết quả: [ranking metrics](../artifacts/eval-test-1000-no-semantic-paec100/ranking_metrics.json), [PAEC metrics](../artifacts/eval-test-1000-no-semantic-paec100/paec_metrics.json), [run manifest](../artifacts/eval-test-1000-no-semantic-paec100/run_manifest.json) và [checkpoint](../artifacts/eval-test-1000-no-semantic-paec100/evaluation.checkpoint.jsonl).

## Ranking metrics trên toàn bộ 1.000 query

| Ranker | NDCG@10 | HR@10 | Candidate Recall@50 | Query coverage | Item coverage |
|---|---:|---:|---:|---:|---:|
| Popularity cold-start | 1,40% | 2,40% | 5,20% | 100,00% | 0,03% |
| BM25 | 1,09% | 2,30% | 6,90% | 99,30% | 11,01% |
| ItemKNN | 4,36% | 7,10% | 10,90% | 100,00% | 15,35% |
| Hybrid ItemKNN + BM25 | **5,20%** | **7,90%** | **10,90%** | **100,00%** | 14,00% |

## Ranking metrics trên 814 positive query

| Ranker | NDCG@10 | HR@10 | Candidate Recall@50 | Query coverage | Item coverage |
|---|---:|---:|---:|---:|---:|
| Popularity cold-start | 1,69% | 2,83% | 5,77% | 100,00% | 0,03% |
| BM25 | 1,29% | 2,70% | 8,35% | 99,51% | 9,50% |
| ItemKNN | 5,23% | 8,48% | 12,78% | 100,00% | 13,20% |
| Hybrid ItemKNN + BM25 | **6,26%** | **9,58%** | **13,02%** | **100,00%** | 11,99% |

## Hybrid latency

Latency chỉ được đo cho full hybrid pipeline; evaluator không ghi latency độc lập cho Popularity, ItemKNN và BM25.

| Tập query | Mean | P50 | P95 |
|---|---:|---:|---:|
| Toàn bộ 1.000 query | 83,71 ms | 53,12 ms | 268,18 ms |
| Positive slice, 814 query | 85,58 ms | 54,24 ms | 269,12 ms |

## Kiểm định Hybrid so với các baseline

HR@10 dùng exact McNemar test trên paired hit/miss. Khoảng tin cậy cho chênh lệch HR@10 và NDCG@10 được tính bằng paired bootstrap 20.000 lần, seed `20261019`.

| Baseline | Hybrid-only hit@10 | Baseline-only hit@10 | Delta HR@10 | 95% CI HR@10 | p, McNemar | Delta NDCG@10 | 95% CI NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | 71 | 16 | +5,50 điểm % | +3,70 đến +7,30 | 1,96e-09 | +3,79 điểm % | +2,56 đến +5,09 |
| BM25 | 60 | 4 | +5,60 điểm % | +4,10 đến +7,20 | 7,36e-14 | +4,10 điểm % | +3,08 đến +5,18 |
| ItemKNN | 17 | 9 | +0,80 điểm % | -0,20 đến +1,80 | 0,169 | +0,84 điểm % | +0,18 đến +1,52 |

Diễn giải:

- Hybrid tốt hơn Popularity và BM25 ở HR@10 với ý nghĩa thống kê.
- Hybrid cải thiện NDCG@10 so với cả ba baseline; riêng so với ItemKNN, bằng chứng mạnh ở vị trí xếp hạng nhưng chưa đủ mạnh để kết luận tăng HR@10.
- Hybrid và ItemKNN cùng Candidate Recall@50 là 10,90%; cải thiện Hybrid hiện đến từ fusion/rerank candidate, không phải mở rộng pool candidate.

## PAEC statistics

| Chỉ số | Giá trị |
|---|---:|
| PAEC query được đánh giá | 100 |
| Certificate được chọn | 91 |
| Không tìm được certificate trong budget | 9 |
| Selection rate | 91,00% |
| Exact 95% CI của selection rate | 83,60% đến 95,80% |
| Selector exact | 76 query |
| Selector beam/bounded | 24 query |
| Coalition size trung bình, chỉ selected | 1,18 |
| Evaluated coalitions trung bình | 15,51 |
| Evaluated coalitions nhỏ nhất/lớn nhất | 1 / 231 |

### Phân bố coalition size của 91 PAEC certificate được chọn

| Coalition size | Số query | Tỷ lệ trên selected query |
|---:|---:|---:|
| 1 history source | 80 | 87,91% |
| 2 history source | 6 | 6,59% |
| 3 history source | 5 | 5,49% |

PAEC selection rate đo tỷ lệ selector tìm được coalition dual-valid trong 100 query được chạy. Nó không phải ranking metric và không thể được diễn giải như cải thiện NDCG hoặc HR.

## Kết luận thực nghiệm

ItemKNN là retriever mạnh nhất khi xét từng branch. BM25 đơn lẻ có độ phủ item cao hơn Popularity nhưng chất lượng top-10 thấp. Fusion ItemKNN + BM25 tạo ranking tốt nhất: NDCG@10 tăng rõ ràng so với ItemKNN, đồng thời PAEC tìm được certificate cho 91% query được kiểm tra. Với kết quả hiện tại, nên mô tả Hybrid là cải thiện vị trí của target trong candidate pool hơn là cải thiện candidate recall so với ItemKNN.
