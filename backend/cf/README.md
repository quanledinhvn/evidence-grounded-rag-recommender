# CF Training

Package này huấn luyện BPR-MF từ artifact D3, tạo popularity fallback và đánh
giá BPR song song với Popularity.

## Chạy production trên AWS

```bash
./infra/skypilot/run.sh --region ap-southeast-2 --mode check
./infra/skypilot/run.sh --region ap-southeast-2 --mode full
```

Production training/evaluation luôn chạy qua SkyPilot trên AWS. Local chỉ chạy
unit test nhỏ. Full task tune bốn candidate trên mẫu cố định 50.000 validation
query, dùng batch 256, rồi đánh giá model thắng trên full validation và test.

## Lấy gợi ý cho một user

```bash
uv run recsys-cf recommend \
  --bundle artifacts/cf-v1/test \
  --user-id u17 \
  --limit 10
```

Dùng `--seen-id <item_id>` nhiều lần để loại thêm các item khỏi kết quả.

Nếu không truyền các tùy chọn `--*-candidates`, pipeline đánh giá đúng một cấu
hình lấy từ `--factors`, `--learning-rate`, `--regularization` và `--epochs`.
Các candidate và quyết định chọn được lưu trong `validation_metrics.csv`;
bundle trung gian nằm trong thư mục `tuning/`.

Training dùng `implicit.cpu.bpr.BayesianPersonalizedRanking` với CSR chỉ chứa
rating từ 4 trở lên. `--num-threads 0` dùng mọi CPU để ưu tiên tốc độ; đặt
`--num-threads 1` cho các lần chạy cần tái lập factor bit-for-bit. Mặc định
backend xác minh negative sample không phải positive; chỉ tắt bằng
`--no-verify-negative-samples` khi chủ động chấp nhận trade-off chất lượng.

Mỗi bundle có model NumPy, mapping hai chiều, lịch sử fit, popularity ranking,
config, metric theo epoch và manifest chứa hash đầu vào, code, config, kích
thước và checksum từng artifact. Popularity đếm positive interaction
$rating \ge 4$; item chưa có positive vẫn nằm cuối ranking với count bằng 0 để
cold-start có thể trả đủ số lượng khi catalog cho phép.

Evaluator luôn lấy pool 50 để nhãn Candidate Recall@50 có nghĩa cố định. Báo
cáo gồm NDCG@10, HR@10, Candidate Recall@50, query/item coverage, fallback
rate, mean/p50/p95 latency và kích thước model trên RAM lẫn trên đĩa.

## ItemKNN

`backend/cf/itemknn.py` cài ItemKNN cosine với lịch sử có trọng số theo độ gần
đây. Với lịch sử $h_1, \dots, h_n$ của user, xếp từ cũ đến mới:

$$
score(u, j) = \sum_{k=1}^{n} \gamma^{\,n-k} \cdot sim(h_k, j)
$$

trong đó $sim$ là cosine giữa hai cột item của ma trận user–item nhị phân, mỗi
item chỉ giữ `--neighbors` hàng xóm có độ tương đồng cao nhất, và $\gamma$ là
`--recency-decay` ($\gamma = 1$ tắt trọng số độ gần đây). Item mới nhất có trọng
số 1. `--min-rating` chọn tương tác nào được đưa vào ma trận tương đồng và vào
tổng điểm; mọi tương tác vẫn bị loại khỏi kết quả như item đã xem. Khi không đủ
item có điểm dương, kết quả được bổ sung theo popularity giống bundle BPR.

Pha `validation` chỉ fit trên `train`; pha `test` fit trên `train` và
`validation`, không bao giờ đọc `test`.

```bash
uv run recsys-cf train-itemknn \
  --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet \
  --output-dir runs/itemknn-movies-tv-2023/test \
  --phase test --neighbors 400 --recency-decay 0.7

uv run recsys-cf evaluate-itemknn \
  --bundle runs/itemknn-movies-tv-2023/test \
  --queries artifacts/movies-tv-2023/queries_test.parquet \
  --output runs/itemknn-movies-tv-2023/test_evaluation.json

uv run recsys-cf recommend --bundle runs/itemknn-movies-tv-2023/test --user-id <USER_ID> --limit 10
```

`recommend` tự nhận loại bundle qua `schema_version` trong manifest. Mỗi item
ItemKNN có trường `because` liệt kê tối đa ba item trong lịch sử đóng góp nhiều
nhất vào điểm, kèm `similarity` và `weight`; đây là nguồn bằng chứng `[H#]` cho
bước giải thích.

Bundle gồm `similarity.npz` (ma trận top-k dạng CSR), `history.npz` (lịch sử
từng user theo thời gian), `users.parquet`, `items.parquet`,
`popularity.parquet`, `itemknn_config.json` và `manifest.json` có checksum từng
artifact. Báo cáo đánh giá có cùng các chỉ số với BPR, thêm
`popularity_fill_rate` là tỉ lệ vị trí trong top 50 phải bổ sung bằng
popularity.
