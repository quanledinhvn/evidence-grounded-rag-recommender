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
