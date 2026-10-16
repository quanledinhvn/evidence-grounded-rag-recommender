# Standalone Item-KNN

Package này huấn luyện, lưu, đánh giá và phục vụ Item-KNN độc lập. Nó không
import bundle, model hay pipeline từ `backend/cf`.

Item-KNN chỉ dùng interaction có rating từ 4 trở lên để tính cosine similarity.
Toàn bộ interaction trong fit split vẫn được lưu làm lịch sử `seen` để loại
khỏi kết quả. User lạ hoặc user không có positive interaction dùng Popularity
fallback của chính bundle Item-KNN.

## Train

Validation bundle chỉ fit split `train`:

```bash
uv run recsys-item-knn train \
  --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet \
  --output-dir artifacts/item-knn/validation \
  --phase validation \
  --neighbors 50 \
  --num-threads 0
```

Test bundle fit cả `train` và `validation`:

```bash
uv run recsys-item-knn train \
  --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet \
  --output-dir artifacts/item-knn/test \
  --phase test \
  --neighbors 50 \
  --num-threads 0
```

Bundle lưu `similarity.npz`, `user_items.npz`, mappings, fit history,
popularity, config và manifest có checksum cho từng artifact.

## Evaluate

```bash
uv run recsys-item-knn evaluate \
  --bundle artifacts/item-knn/validation \
  --queries artifacts/movies-tv-2023/queries_val.parquet
```

```bash
uv run recsys-item-knn evaluate \
  --bundle artifacts/item-knn/test \
  --queries artifacts/movies-tv-2023/queries_test.parquet
```

## Recommend

```bash
uv run recsys-item-knn recommend \
  --bundle artifacts/item-knn/test \
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q \
  --limit 10
```

Dùng `--seen-id <item_id>` nhiều lần để loại thêm item khỏi kết quả.
