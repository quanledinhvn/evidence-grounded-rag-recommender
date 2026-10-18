# Kế hoạch hoàn thiện pipeline trong hai ngày

## 1. Mục tiêu đã khóa

Trong thời hạn hai ngày, hệ thống phải chạy được luồng sau trên artifact thật:

```text
profile test
  -> ItemKNN test bundle + OpenSearch BM25 + OpenSearch semantic
  -> candidate union và query-local fusion
  -> top-10 item chưa tương tác
  -> PAEC removal/retention cho top-1
  -> kết quả JSON có provenance, certificate và manifest hash
```

ItemKNN test bundle tại
`runs/item-knn-movies-tv-2023/test/20261015T173128Z/bundle` được xem là đã
khóa. Bundle này fit trên `train + validation`, đúng với pha test. Vì không tạo
validation bundle mới, trọng số fusion được đặt trước và tuyệt đối không thay
đổi sau khi đọc metric test.

## 2. Cấu hình thí nghiệm cố định

| Thuộc tính | Giá trị |
|---|---:|
| Phase | `test` |
| Top-K | 10 |
| Pool mỗi retriever | 50 |
| ItemKNN weight | $1/3$ |
| BM25 weight | $1/3$ |
| Semantic weight | $1/3$ |
| PAEC evidence budget | 3 |
| Positive rating threshold | 4 |
| Seed lấy mẫu | 42 |
| Ranking sample mục tiêu | 1.000 query |
| PAEC sample mục tiêu | 100 query hợp lệ |

Mọi output phải lưu config, source hash, artifact manifest hash và thời gian
chạy. Test target chỉ được ghép vào sau inference để tính metric.

Semantic ANN dùng `k=50` ở mức chunk, đúng bằng pool budget và không overfetch.
Sau khi gom theo item, semantic pool có thể nhỏ hơn 50 nếu nhiều chunk cùng
thuộc một item; query coverage và candidate recall phải phản ánh trade-off này.

## 3. Đường găng triển khai

### Ngày 1 — Serving pipeline

1. Đóng mốc embedding/index hiện tại bằng unit test và kiểm tra document count.
2. Thêm gateway đọc các chunk và vector đã index của một history item.
3. Thêm BM25 adapter chạy một query cho từng raw history source.
4. Thêm semantic adapter lấy centroid của các chunk thuộc từng history item,
   chạy một ANN query cho mỗi raw source, gom chunk hit về item và giữ
   source-level contribution. Quyết định dùng centroid được khóa để sample
   1.000 query có thể hoàn tất trong deadline; đây không phải best-chunk-pair
   scorer trong thiết kế đầy đủ.
5. Thêm loader cho profile JSON và CLI end-to-end dùng ItemKNN test bundle.
6. Chạy smoke test với `artifacts/movies-tv-2023/profile_example.json`.

Điều kiện hoàn thành ngày 1:

- output có đúng tối đa 10 item và không chứa `seen_item_ids`;
- ba nhánh đều tạo candidate hoặc ghi rõ nhánh không áp dụng;
- cùng input/config cho cùng thứ tự kết quả;
- PAEC giữ nguyên factual `seen_item_ids` trong removal và retention;
- không gọi lại API embedding khi vector history đã có trong OpenSearch.

### Ngày 2 — Đánh giá và đóng gói

1. Thêm evaluator theo batch có resume/checkpoint.
2. Chạy năm ranker: Popularity, ItemKNN, BM25, Semantic và hybrid ba nhánh.
3. Báo NDCG@10, HR@10, Candidate Recall@50, query/item coverage và latency.
4. Chạy PAEC cho tối đa 100 query hợp lệ; báo dual-valid rate, coalition size,
   số rerun và latency.
5. Sinh deterministic explanation; abstain nếu không có certificate dual-valid.
6. Ghi toàn bộ kết quả vào `artifacts/final/` cùng một manifest tái lập.

## 4. Phạm vi chủ động loại bỏ

Không thực hiện trong mốc hai ngày:

- huấn luyện ItemKNN validation bundle hoặc tune weight bằng test;
- nhập review text Movies and TV;
- LLM renderer, semantic verifier bằng LLM hoặc repair loop;
- FastAPI, giao diện web hoặc deployment production;
- toàn bộ ablation R0–R7 và full test set 657.203 query.

Kết quả vì vậy là thí nghiệm test với cấu hình đặt trước, không phải kết quả
tuning tối ưu. Explanation chỉ dựa trên metadata và trace cơ học, không được mô
tả là review-grounded.

## 5. Artifact bàn giao

```text
artifacts/final/
  frozen_config.json
  ranking_metrics.json
  ranking_metrics.csv
  paec_metrics.json
  demo_profile.json
  demo_recommendation.json
  run_manifest.json
```

Nếu thời gian chạy không đủ, sample đã hoàn tất vẫn được giữ cùng checkpoint và
`completed_query_count`; không được báo metric của sample dở dang như full test.
