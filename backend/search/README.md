# OpenSearch retrieval infrastructure

Module này cung cấp hai index tách biệt:

- `items-v1`: một document cho mỗi item, dùng BM25;
- `chunks-v1`: một document cho mỗi metadata chunk, dùng HNSW cosine.

Việc tách index ngăn chunking semantic làm thay đổi document frequency và
document length của BM25. Parquet vẫn là nguồn dữ liệu chuẩn; OpenSearch là
derived artifact có thể dựng lại.

Encoder mặc định là OpenAI `text-embedding-3-small` với 768 chiều. Đây là lựa
chọn đã khóa cho index `v1`, không phải ràng buộc kiến trúc: indexer phụ thuộc
vào interface `EmbeddingProvider`, nên có thể thêm Gemini provider sau. Không
được trộn vector từ OpenAI và Gemini trong cùng index; khi đổi provider, model,
dimension hoặc chunking phải tạo index version mới và re-index toàn bộ.

## Cài dependency

```bash
uv sync --extra search --extra dev
```

Không ghi API key vào repository hoặc file Compose. CLI tự động đọc `.env` ở
project root; biến đã export trong shell luôn được ưu tiên. File
`backend/.env` cũng được đọc như đường dẫn tương thích với thiết lập cũ.

Sao chép `.env.example` thành `.env` và điền key:

```bash
cp .env.example .env
```

Hoặc cấp key qua secret manager của môi trường chạy hay export tạm trong shell:

```bash
export OPENAI_API_KEY='...'
```

## Chạy OpenSearch local

Cấu hình Compose chỉ dành cho development: security plugin bị tắt và cổng chỉ
bind vào loopback.

```bash
docker compose -f infra/opensearch/compose.yaml up -d
docker compose -f infra/opensearch/compose.yaml ps
uv run --extra search recsys-search health
```

Tắt container nhưng giữ index:

```bash
docker compose -f infra/opensearch/compose.yaml down
```

Không chạy `down -v` trừ khi chủ động muốn xóa toàn bộ index local.

## Tạo embedding local và nạp index

Tên index phải có version bất biến. Khi thay mapping, encoder hoặc chunking,
tạo tên mới thay vì sửa index đang dùng.

```bash
export OPENSEARCH_ITEM_INDEX='it4613-items-v1'
export OPENSEARCH_CHUNK_INDEX='it4613-chunks-v1'
export OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
export OPENAI_EMBEDDING_DIMENSIONS='768'

uv run --extra search recsys-search create-indices

uv run --extra search recsys-search plan-catalog \
  --catalog artifacts/movies-tv-2023/catalog_enriched.parquet

uv run --extra search recsys-search embed-catalog \
  --catalog artifacts/movies-tv-2023/catalog_enriched.parquet \
  --artifact artifacts/movies-tv-2023/search-embeddings-v1.parquet \
  --manifest artifacts/movies-tv-2023/search-embeddings-manifest-v1.json
```

`embed-catalog` gọi OpenAI theo batch nhưng không kết nối OpenSearch. Vector và
metadata chunk được ghi vào một Parquet nén Zstandard ở local. Manifest đi kèm
khóa hash của catalog và artifact, model, dimension và chunking policy để có thể
kiểm tra lại trước khi index.

Trong lúc chạy, CLI in tiến độ, thời gian đã chạy và ETA ra stderr; JSON kết quả
cuối vẫn được in riêng ra stdout. Mỗi catalog batch được commit nguyên tử vào thư
mục `<artifact>.checkpoint`. Nếu API hết retry, mất mạng hoặc tiến trình dừng,
chạy lại đúng lệnh sẽ xác minh và tái sử dụng các batch đã hoàn tất. Checkpoint
chỉ được dọn sau khi artifact và manifest cuối đã ghi thành công.

Checkpoint khóa theo hash catalog, model, dimension, chunking policy và batch
size. Nếu một trong các giá trị này thay đổi, dùng một đường dẫn artifact mới
hoặc chủ động dọn checkpoint cũ; CLI sẽ không trộn vector từ hai cấu hình.

Khi OpenSearch sẵn sàng, nạp artifact đã tạo mà không gọi lại embedding API:

```bash
uv run --extra search recsys-search index-artifact \
  --catalog artifacts/movies-tv-2023/catalog_enriched.parquet \
  --artifact artifacts/movies-tv-2023/search-embeddings-v1.parquet \
  --artifact-manifest artifacts/movies-tv-2023/search-embeddings-manifest-v1.json \
  --manifest artifacts/movies-tv-2023/search-index-manifest-v1.json
```

`plan-catalog` không gọi OpenSearch hoặc OpenAI. Lệnh này kiểm tra ID trùng,
đếm item có text, số chunk và tổng số từ sau overlap để dự báo khối lượng trước
khi dùng API trả phí.

`index-artifact` xác minh hash catalog, hash artifact, model, dimension, số dòng
và từng vector trước khi tạo hoặc cập nhật index. Sau đó lệnh đọc cả catalog và
artifact theo batch, dùng document ID tất định và bulk upsert. Vì vậy có thể
chạy lại an toàn. Manifest index chỉ được ghi sau khi cả hai index đã nạp và
refresh thành công.

Lệnh `index-catalog` cũ vẫn được giữ cho workflow một bước. Lệnh này gọi
embedding API và OpenSearch trong cùng một lần chạy; dùng hai lệnh phía trên khi
cần tạo embedding trước rồi push ở thời điểm khác.

## Cấu hình

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `OPENSEARCH_URL` | `http://localhost:9200` | Endpoint OpenSearch |
| `OPENSEARCH_ITEM_INDEX` | `it4613-items-v1` | Index BM25 |
| `OPENSEARCH_CHUNK_INDEX` | `it4613-chunks-v1` | Index semantic chunks |
| `OPENSEARCH_USERNAME` | trống | Basic-auth username tùy chọn |
| `OPENSEARCH_PASSWORD` | trống | Basic-auth password tùy chọn |
| `OPENSEARCH_VERIFY_CERTIFICATES` | `true` | Xác minh TLS certificate |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Encoder đã khóa |
| `OPENAI_EMBEDDING_DIMENSIONS` | `768` | Số chiều vector |
| `OPENAI_EMBEDDING_BATCH_SIZE` | `128` | Số text mỗi API request |
| `OPENAI_EMBEDDING_TIMEOUT_SECONDS` | `60` | Timeout mỗi API request |
| `OPENAI_EMBEDDING_MAX_RETRIES` | `5` | Số retry do OpenAI SDK thực hiện |

Không đặt credential trong `OPENSEARCH_URL`; CLI chủ động từ chối URL chứa
username hoặc password.

## Production

Không dùng Compose development làm production. Production phải có TLS,
authentication, snapshot, ít nhất một replica khi yêu cầu availability, giới
hạn network và monitoring disk/heap. Dùng một index version mới, nạp đầy đủ,
kiểm tra manifest rồi mới chuyển cấu hình backend sang version mới.

Rollback không xóa index mới: đặt lại `OPENSEARCH_ITEM_INDEX` và
`OPENSEARCH_CHUNK_INDEX` về version trước, sau đó restart backend và chạy
`recsys-search health`. Khi kết quả đã ổn định mới dọn index lỗi theo quy trình
retention riêng.
