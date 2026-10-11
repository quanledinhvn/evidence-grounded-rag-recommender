# Pipeline dữ liệu Movies and TV 2023

Nguồn dữ liệu chính của dự án là ba split tuần tự đã có sẵn:

- `dataset/2023/Movies_and_TV.train.csv`
- `dataset/2023/Movies_and_TV.valid.csv`
- `dataset/2023/Movies_and_TV.test.csv`

Mỗi file có schema `user_id,parent_asin,rating,timestamp,history`. Pipeline không chia lại dữ liệu; nhãn `train`, `validation` và `test` được giữ nguyên từ file nguồn. `parent_asin` được chuẩn hóa thành `item_id`, còn timestamp Unix mili-giây được giữ nguyên để không làm mất thứ tự sự kiện.

## Cài đặt

```bash
uv sync --extra dev
```

## Nhập ba split có sẵn

```bash
uv run recsys-data import-splits \
  --train dataset/2023/Movies_and_TV.train.csv \
  --validation dataset/2023/Movies_and_TV.valid.csv \
  --test dataset/2023/Movies_and_TV.test.csv \
  --output-dir artifacts/movies-tv-2023
```

Lệnh này đọc theo batch để phù hợp với file train lớn và tạo:

- `splits.parquet`: toàn bộ tương tác, giữ nguyên split nguồn;
- `queries_val.parquet` và `queries_test.parquet`: target dành riêng cho evaluator;
- `catalog.parquet`: catalog ID hợp nhất từ cả ba split;
- `split_stats.json`: số dòng, số user, hash nguồn và đường dẫn artifact.

Importer kiểm tra header, ID rỗng, rating ngoài khoảng 1–5, timestamp không dương và trường hợp target xuất hiện trong chính `history`. Vì nguồn CSV không chứa review text hoặc metadata phim, `catalog.parquet` ban đầu chỉ có ID và `catalog_items_with_bm25_text` bằng 0.

## Metadata và RAG

Metadata Movies and TV đã được giải nén tại `dataset/2023/meta_Movies_and_TV.jsonl`. Enrich catalog bằng phép nối chính xác `parent_asin` → `item_id`:

```bash
uv run recsys-data enrich-catalog \
  --catalog artifacts/movies-tv-2023/catalog.parquet \
  --metadata dataset/2023/meta_Movies_and_TV.jsonl \
  --output artifacts/movies-tv-2023/catalog_enriched.parquet
```

Kết quả schema `catalog-enrichment-v2` hiện khớp đủ 197.943 item và được audit tại `catalog_enriched.stats.json`. Catalog giữ các field nội dung `title`, `subtitle`, `description`, `feature`, `main_category`, `category`, `details`; giữ `store`, `average_rating` và `rating_number` như thuộc tính phụ; đồng thời tạo `metadata_text` làm canonical document cho text index. `store` và hai thống kê rating không được ghép vào `metadata_text`.

Dùng `catalog_enriched.parquet` cho Profile Builder, BM25, semantic search và metadata evidence. Review text Movies and TV vẫn chưa được import, vì vậy chưa có review evidence. Không dùng `meta_Digital_Music.jsonl` cho catalog phim.

Các subcommand `audit`, `normalize`, `deduplicate`, `split` và `prepare` vẫn được giữ để hỗ trợ nguồn JSONL thô khác; chúng không phải đường xử lý mặc định của ba CSV hiện tại.

## Kiểm thử

```bash
uv run pytest -q
```
