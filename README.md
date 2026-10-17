# Evidence-grounded RAG Recommender

Hệ gợi ý phim và chương trình truyền hình có giải thích dựa trên bằng chứng: hệ thống trả 10 gợi ý cho một người dùng và giải thích ba gợi ý đầu, mỗi lý do trích dẫn về metadata phim, review của người khác hoặc lịch sử của chính người dùng. Dữ liệu: Amazon Reviews 2023 – Movies & TV. Đồ án học phần Hệ gợi ý, Thạc sĩ Khoa học máy tính, HUST.

**Trạng thái:** đã có pipeline dữ liệu, hồ sơ người dùng và nhánh CF (ItemKNN, BPR). BM25, embedding, fusion, bằng chứng và LLM chưa làm. Chi tiết tiến độ ở mục *Tiến độ* trong [CLAUDE.md](CLAUDE.md).

## Đọc tài liệu theo thứ tự

| # | Tài liệu | Nội dung |
|---|---|---|
| 1 | [docs/proposal.md](docs/proposal.md) | Đề xuất đề tài: bài toán, câu hỏi nghiên cứu, phạm vi |
| 2 | [docs/overview.md](docs/overview.md) | Tổng quan dễ hiểu: 6 bước của hệ thống, dữ liệu, thí nghiệm, kết quả CF hiện tại |
| 3 | [docs/architecture.md](docs/architecture.md) | Đặc tả chi tiết: hợp đồng từng bước, schema, chống rò rỉ, API, thiết kế thí nghiệm |
| 4 | [docs/data/](docs/data/) | [contract.md](docs/data/contract.md): hợp đồng dữ liệu đang dùng; [raw-dataset-survey.md](docs/data/raw-dataset-survey.md): khảo sát dữ liệu thô |
| 5 | [docs/cf/](docs/cf/) | [guide.md](docs/cf/guide.md): CF cho người mới; [bpr-design.md](docs/cf/bpr-design.md): thiết kế BPR-MF |
| 6 | [docs/profile/design.md](docs/profile/design.md) | Thiết kế Profile Builder |

Hướng dẫn chạy từng module nằm cạnh code: [backend/data/README.md](backend/data/README.md), [backend/cf/README.md](backend/cf/README.md). Hướng dẫn chạy test từng phần đã cài đặt (input, output, ví dụ) ở mục *Hướng dẫn chạy test* trong [CLAUDE.md](CLAUDE.md). Biên bản họp ở [docs/meetings/](docs/meetings/), tài liệu tham khảo ở [docs/references/](docs/references/).

## Cấu trúc thư mục

```text
backend/
  data/        # nhập 3 CSV split, nối metadata vào catalog (CLI recsys-data)
  profile/     # hồ sơ người dùng từ lịch sử trước as_of (CLI recsys-profile)
  cf/          # ItemKNN (nhánh chính), BPR-MF, Popularity, đánh giá (CLI recsys-cf)
tests/         # pytest, dữ liệu giả rất nhỏ, không cần dữ liệu thật
docs/          # tài liệu (xem bảng trên)
dataset/       # dữ liệu gốc, không đưa lên Git
artifacts/     # dữ liệu đã xử lý, không đưa lên Git
runs/          # model và kết quả từng lần chạy, không đưa lên Git
```

## Cài đặt và chạy

Yêu cầu Python 3.11 trở lên và [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run pytest -q
```

Dữ liệu đặt tại `dataset/` (tải từ <https://amazon-reviews-2023.github.io/>):

```text
dataset/2023/Movies_and_TV.{train,valid,test}.csv   # split tuần tự 5-core
dataset/2023/meta_Movies_and_TV.jsonl               # metadata item
dataset/Movies_and_TV.jsonl                         # review thô, dùng cho kho bằng chứng
```

Luồng chạy chính:

```bash
# 1. Nhập split và enrich catalog -> artifacts/movies-tv-2023/
uv run recsys-data import-splits --train dataset/2023/Movies_and_TV.train.csv \
  --validation dataset/2023/Movies_and_TV.valid.csv --test dataset/2023/Movies_and_TV.test.csv \
  --output-dir artifacts/movies-tv-2023
uv run recsys-data enrich-catalog --catalog artifacts/movies-tv-2023/catalog.parquet \
  --metadata dataset/2023/meta_Movies_and_TV.jsonl \
  --output artifacts/movies-tv-2023/catalog_enriched.parquet

# 2. ItemKNN pha test (mặc định 400 hàng xóm, decay 0,7) và đánh giá trên toàn bộ test
uv run recsys-cf train --model itemknn --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet --phase test \
  --output-dir runs/itemknn-movies-tv-2023/test
uv run recsys-cf evaluate --model-dir runs/itemknn-movies-tv-2023/test \
  --queries artifacts/movies-tv-2023/queries_test.parquet \
  --output runs/itemknn-movies-tv-2023/test_evaluation.json

# 3. Gợi ý và hồ sơ cho một người dùng
uv run recsys-cf recommend --model-dir runs/itemknn-movies-tv-2023/test --user-id <USER_ID> --limit 10
uv run recsys-profile --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog_enriched.parquet --user-id <USER_ID> \
  --as-of <AS_OF_MS> --phase test
```

Trên Windows, nếu `implicit` báo `DLL load failed ... filename too long`, hãy đặt virtualenv ở đường dẫn ngắn hoặc bật LongPaths.

## Quy ước

Quy ước viết code, tài liệu và TDD nằm trong [AGENTS.md](AGENTS.md). Quy tắc dữ liệu quan trọng nhất: mọi thông tin dùng để gợi ý phải có $timestamp < as\_of$; pha validation chỉ fit trên `train`, pha test fit trên `train` + `validation` và không đọc `test`.
