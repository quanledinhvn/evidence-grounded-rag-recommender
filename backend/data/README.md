# backend/data — chuẩn bị dữ liệu

| File | Việc |
|---|---|
| `importer.py` | Đọc 3 CSV split (train/valid/test), kiểm tra, ghi Parquet |
| `enrich.py` | Nối metadata (title, mô tả, đạo diễn, diễn viên…) vào catalog theo `parent_asin` |
| `io.py` | Hàm đọc/ghi JSON, JSONL, hash file |
| `cli.py` | Lệnh `recsys-data` |

## 1. Nhập split

```bash
uv run recsys-data import-splits \
  --train dataset/2023/Movies_and_TV.train.csv \
  --validation dataset/2023/Movies_and_TV.valid.csv \
  --test dataset/2023/Movies_and_TV.test.csv \
  --output-dir artifacts/movies-tv-2023
```

Mỗi CSV có cột `user_id,parent_asin,rating,timestamp,history`. Split giữ nguyên như nguồn công bố, không chia lại. `parent_asin` đổi tên thành `item_id`; timestamp giữ đơn vị mili-giây.

Kiểm tra trước khi ghi: ID không rỗng, rating trong 1–5, timestamp dương, item đích không nằm trong chính `history` của dòng đó.

Đầu ra:

- `splits.parquet`: mọi tương tác, cột `user_id, item_id, rating, timestamp, review_id, split`; `review_id = amazon-2023:<split>:<dòng CSV>`;
- `queries_val.parquet`, `queries_test.parquet`: mỗi dòng là một câu hỏi đánh giá `user_id, as_of, target_item_id, target_rating, …` với `as_of` = thời điểm của item đích;
- `catalog.parquet`: mọi `item_id` xuất hiện trong ba split;
- `split_stats.json`: số dòng, số user, hash file nguồn.

## 2. Nối metadata

```bash
uv run recsys-data enrich-catalog \
  --catalog artifacts/movies-tv-2023/catalog.parquet \
  --metadata dataset/2023/meta_Movies_and_TV.jsonl \
  --output artifacts/movies-tv-2023/catalog_enriched.parquet
```

Mỗi item có các trường nội dung `title, subtitle, description, feature, main_category, category, details` (HTML đã được làm sạch; `details` chỉ giữ các khóa hữu ích như `Directors`, `Starring`, `Genre`), cộng `store, average_rating, rating_number`. `metadata_text` ghép các trường nội dung thành một văn bản dùng chung cho BM25, embedding và bằng chứng. `source_id = meta:<item_id>` để giải thích trích dẫn được.

## Kiểm thử

```bash
uv run pytest tests/test_data.py -q
```
