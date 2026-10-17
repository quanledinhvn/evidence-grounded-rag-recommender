# backend/cf — collaborative filtering

| File | Việc |
|---|---|
| `common.py` | `FitData` (lịch sử từng user theo thời gian), `load_fit_data`, `Popularity`, `top_k`, `rank_items` |
| `itemknn.py` | ItemKNN có trọng số độ gần đây — nhánh CF chính |
| `bpr.py` | BPR-MF (thư viện `implicit`) — baseline |
| `evaluate.py` | NDCG@10, HR@10, Recall@50 trên file query |
| `cli.py` | Lệnh `recsys-cf train / evaluate / recommend` |

Mọi model có cùng giao diện: `fit(data, ...)`, `recommend(user_id, limit, exclude)`, `recommend_batch(user_ids, limit)`, `save(dir)`, `load(dir)`. Kết quả là danh sách `{"item_id", "score", "source"}`; `source` cho biết item đến từ model hay được bổ sung bằng popularity.

## Pha validation và test

`load_fit_data(splits, catalog, phase)` chỉ đọc các split được phép: pha `validation` học từ `train`; pha `test` học từ `train` + `validation`. Không model nào đọc `test`. Chọn tham số trên validation rồi mới chạy test.

## ItemKNN

Với lịch sử $h_1, \dots, h_n$ của user, xếp từ cũ đến mới:

$$
score(j) = \sum_{k=1}^{n} \gamma^{\,n-k} \cdot sim(h_k, j)
$$

$sim$ là cosine giữa hai cột item của ma trận user–item (hai phim được cùng nhiều người xem thì giống nhau); mỗi item chỉ giữ `--neighbors` hàng xóm gần nhất. $\gamma$ là `--recency-decay`: phim xem gần nhất có trọng số 1, phim trước đó $\gamma$, rồi $\gamma^2$… Cấu hình chọn trên validation: 400 hàng xóm, $\gamma = 0{,}7$. Mỗi gợi ý của ItemKNN có trường `because` liệt kê các phim trong lịch sử góp điểm nhiều nhất — đây là bằng chứng `[H#]` cho bước giải thích.

## BPR-MF

Học vector cho user và item từ các tương tác rating ≥ 4; điểm = tích vô hướng + bias của item. User không có rating ≥ 4 nhận popularity; item chưa ai chấm ≥ 4 không được BPR gợi ý. Mặc định 64 factors, 60 epoch, learning rate 0,01, regularization 0,001, seed 42.

## Lệnh

```bash
# Train (pha test) và lưu model
uv run recsys-cf train --model itemknn --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet --phase test \
  --output-dir runs/itemknn-movies-tv-2023/test
uv run recsys-cf train --model bpr ... --output-dir runs/bpr-movies-tv-2023/test

# Đánh giá trên toàn bộ test (thêm --query-limit 20000 để chạy nhanh trên mẫu)
uv run recsys-cf evaluate --model-dir runs/itemknn-movies-tv-2023/test \
  --queries artifacts/movies-tv-2023/queries_test.parquet \
  --output runs/itemknn-movies-tv-2023/test_evaluation.json
uv run recsys-cf evaluate --model-dir runs/itemknn-movies-tv-2023/test \
  --queries artifacts/movies-tv-2023/queries_test.parquet --popularity   # baseline Popularity

# Gợi ý cho một user
uv run recsys-cf recommend --model-dir runs/itemknn-movies-tv-2023/test --user-id <USER_ID> --limit 10
```

Thư mục model chứa `config.json` (loại model và tham số), `fit_data.npz` (lịch sử dùng để học) và `similarity.npz` (ItemKNN) hoặc `factors.npz` (BPR).

## Kiểm thử

```bash
uv run pytest tests/test_cf.py -q
```
