# CLAUDE.md

Hệ gợi ý phim có giải thích dựa trên bằng chứng (CF + BM25 + embedding → fusion → LLM rerank + giải thích bắt buộc trích dẫn) trên Amazon Reviews 2023 – Movies & TV. Đồ án học phần Hệ gợi ý, Thạc sĩ KHMT, HUST. Tổng quan: [README.md](README.md), [docs/overview.md](docs/overview.md); đặc tả: [docs/architecture.md](docs/architecture.md).

## Quy tắc bắt buộc

@AGENTS.md

## Môi trường

- Python ≥ 3.11, quản lý bằng `uv` (`uv sync --extra dev`, `uv run pytest -q`). Nếu máy không có `uv`: tạo venv rồi `pip install -e ".[dev]"`.
- CLI: `recsys-data` (backend/data), `recsys-cf` (backend/cf), `recsys-profile` (backend/profile).
- Windows: `implicit` lỗi `DLL load failed ... filename too long` khi venv nằm ở đường dẫn dài; dùng venv ở đường dẫn ngắn.
- Dự án làm ở local trên nhánh `f/quan`: commit bình thường, **không push** lên remote.
- `dataset/`, `artifacts/`, `runs/` không đưa lên Git (đã có trong `.gitignore`).

## Dữ liệu và artifact

| Đường dẫn | Nội dung |
|---|---|
| `artifacts/movies-tv-2023/splits.parquet` | 7.441.129 tương tác (train 6.126.723, validation 657.203, test 657.203); cột text rỗng |
| `artifacts/movies-tv-2023/queries_{val,test}.parquet` | 657.203 query mỗi pha, mỗi user một target, `as_of` = timestamp target |
| `artifacts/movies-tv-2023/catalog_enriched.parquet` | 197.943 item, đủ metadata; 82.840 item Prime Video không có title |
| `runs/cf-movies-tv-2023-full/20261009T135249Z/` | Bundle BPR pha test (train trên AWS) và `evaluation.json` |
| `runs/itemknn-movies-tv-2023/` | Bundle ItemKNN pha test, `test_evaluation.json`, `validation_metrics.csv` |
| `dataset/Movies_and_TV.jsonl`, `dataset/meta_Movies_and_TV.jsonl` | Review thô (17,3 triệu) và metadata thô, chưa import |

Đặc điểm cần nhớ: trung vị 5 tương tác train mỗi user; split không có review text; một phim có nhiều `parent_asin` (DVD, Blu-ray, Prime Video).

## Bất biến

- Mọi dữ liệu dùng cho một query phải có `timestamp < as_of`.
- Pha `validation` chỉ fit trên `train`; pha `test` fit trên `train` + `validation`, không đọc `test`. Chọn cấu hình trên validation, test chỉ dùng để báo cáo.
- Pool đánh giá cố định 50 (NDCG@10, HR@10, Candidate Recall@50); `recommend` giới hạn `limit ≤ 50`.
- Item index trong bundle theo thứ tự `item_id` tăng dần; phá hòa theo `item_id`.
- Bundle có `manifest.json` với checksum từng artifact; `load` phải kiểm tra checksum.

## Tiến độ (cập nhật 2026-10-16)

| Bước | Trạng thái |
|---|---|
| Dữ liệu 5-core: import split, enrich catalog | Xong |
| ① Profile Builder (`backend/profile`) | Xong phần lõi; chậm (~3,2 s/user do quét toàn bộ parquet) |
| ② CF: ItemKNN (`backend/cf/itemknn.py`) | Xong: cài đặt, 18 test, CLI, tune, đánh giá full test |
| ② CF: BPR-MF | Xong, giữ làm baseline |
| ② BM25, embedding | Chưa làm |
| ③ Fusion | Chưa làm; đã có số liệu ban đầu về prior popularity |
| ④ Kho bằng chứng (review) | Chưa làm |
| ⑤ LLM rerank + giải thích, ⑥ validator | Chưa làm |
| Service, API, chạy run A–G, báo cáo | Chưa làm |

Kết quả trên toàn bộ 657.203 query test:

| Phương pháp | NDCG@10 | HR@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity | 0,0091 | 0,0171 | 0,045 |
| BPR-MF (64 factors, 60 epoch) | 0,0132 | 0,0228 | 0,045 |
| ItemKNN (400 hàng xóm, decay 0,7) | 0,0491 | 0,0768 | 0,119 |

Phát hiện dùng cho bước tiếp theo (đo trên 20.000 query validation, seed 42):

- Trọng số theo độ gần đây là thành phần quan trọng của ItemKNN và được giữ trong cấu hình chốt ($\gamma = 0{,}7$). Thí nghiệm ablation: nếu không dùng nó ($\gamma = 1$, mọi item trong lịch sử ngang nhau) thì NDCG@10 giảm khoảng 17% (0,0586 → 0,0489 với 400 hàng xóm).
- ItemKNN + $\lambda \cdot$ prior popularity theo thứ hạng: $\lambda = 0{,}3$ cho NDCG 0,0585 → 0,0688 và Recall@50 0,134 → 0,179. Recent popularity 30 ngày đứng riêng đạt khoảng 2,5 lần BPR.
- BPR yếu vì bỏ qua thứ tự thời gian, chưa hội tụ (AUC còn tăng ở epoch 60) và dữ liệu thưa.

## Việc tiếp theo

1. Tăng tốc profile store (sắp xếp theo `user_id` hoặc SQLite) và thời gian nạp bundle.
2. Xây `item_doc` từ `catalog_enriched` (title, mô tả, thể loại đã lọc, đạo diễn, diễn viên; mỗi trường có `source_id`); item thiếu title dùng đạo diễn và diễn viên.
3. Nhánh BM25 (`bm25s`) và embedding (`bge-small`), mỗi nhánh xuất pool top-100 và đánh giá riêng.
4. Fusion ItemKNN + BM25 + embedding + popularity (toàn cục và 30 ngày), tune trên validation, khóa cấu hình.
5. Import review thô vào SQLite theo `(item_id, ts)`, lọc review ngắn hoặc nói về giao hàng; đưa review của chính user vào profile.
6. Chọn bằng chứng, LLM rerank và giải thích, validator; chạy run A–G.

Vấn đề mở không chặn tiến độ: `preprocess.py` và các file mà [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md) nhắc tới không có trong repo; cần hỏi người viết khảo sát nếu dùng Dense-Works.

## Quyết định đang chờ

- Có dùng tập Dense-Works (gộp phiên bản, 20-core; xem [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md)) cho giai đoạn LLM không. Cần chốt trước bước 6: với Recall@50 = 0,119, chỉ khoảng 24/200 user có target trong pool.
- LLM provider, ngân sách token, hạn nộp.
