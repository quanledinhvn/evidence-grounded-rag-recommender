# Evidence-grounded RAG Recommender

Hệ gợi ý phim và chương trình truyền hình **có giải thích dựa trên bằng chứng**: với một người dùng tại một thời điểm, hệ thống trả 10 gợi ý và giải thích 3 gợi ý đầu; mỗi câu giải thích trích dẫn nguồn có thật — metadata phim `[E#]`, review của người khác `[E#]` hoặc lịch sử của chính người dùng `[H#]` — và câu nào không kiểm tra được nguồn thì bị loại. Dữ liệu: Amazon Reviews 2023 – Movies & TV. Đồ án học phần Hệ gợi ý, Thạc sĩ Khoa học máy tính, HUST.

```text
lịch sử user ─► ① hồ sơ ─► ② ItemKNN + BM25 + popularity ─► ③ fusion, gom phiên bản ─► top 10
                                                                                          │
           ⑥ validator ◄── ⑤ LLM (Gemini) giải thích top 3, bắt buộc trích dẫn ◄── ④ bằng chứng
                │
                ▼
   "You previously enjoyed Harry Potter and the Chamber of Secrets ... [E1][E2][H1]."
```

Thứ tự gợi ý là thứ tự của fusion: xếp lại bằng LLM, cross-encoder hay learning-to-rank đều đã thử và không tốt hơn, nên LLM chỉ dùng để giải thích.

## Kết quả chính

Mẫu cố định 50.000 query test (seed 42), "cùng phim" = gợi ý bất kỳ phiên bản (DVD, Blu-ray, Prime Video) của phim đúng:

| Phương pháp | NDCG@10 | HR@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity (toàn bộ test, đúng mã) | 0,91% | 1,71% | 4,5% |
| ItemKNN (cùng phim) | 5,51% | 8,72% | 13,6% |
| **Fusion + gom phiên bản — hệ thống cuối (cùng phim)** | **6,46%** | **10,40%** | **16,6%** |

Giải thích (smoke test 10 user, giám khảo LLM): bắt buộc trích dẫn + validator có **88%** câu được nguồn xác nhận, **3%** câu sai; để LLM viết tự do: 72% và 17%. Chi tiết: [docs/results/smoke_test_10_record.md](docs/results/smoke_test_10_record.md); toàn bộ số liệu và các kết quả âm: [docs/results/results.md](docs/results/results.md).

## Cấu trúc thư mục

```text
backend/
  data/         # nhập split, nối metadata, gom phiên bản, kho review    (recsys-data)
  profile/      # ① hồ sơ người dùng trước thời điểm gợi ý                (recsys-profile)
  cf/           # ② ItemKNN, BPR-MF, Popularity; CLI chung cho mọi model (recsys-cf)
  retrieval/    # ② BM25, embedding; ③ fusion
  eval/         # đánh giá: NDCG@10, HR@10, Recall@50, metric cùng phim, phân tích theo nhóm
  evidence/     # ④ gói bằng chứng [H#]/[E#]                              (recsys-evidence)
  generation/   # ⑤ Gemini giải thích, ⑥ validator, giám khảo, thí nghiệm (recsys-llm)
  service.py    # ①→⑥ cho một user                                         (recsys-demo)
tests/          # 66 test pytest, dữ liệu giả rất nhỏ, LLM giả — không cần dữ liệu thật hay API key
docs/           # đề xuất, tổng quan, kiến trúc, thiết kế từng module, results/ (báo cáo kết quả)
dataset/        # dữ liệu gốc            — tự tải, không đưa lên Git
artifacts/      # dữ liệu đã xử lý       — sinh bằng lệnh, không đưa lên Git
runs/           # model, kết quả, cache LLM — sinh bằng lệnh, không đưa lên Git
.env.example    # mẫu file .env chứa GEMINI_API_KEY
```

## 1. Cài đặt

Yêu cầu Python ≥ 3.11. Cách 1, dùng [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

Cách 2, không có `uv`:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell; Git Bash: source .venv/Scripts/activate; Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Các lệnh dưới viết theo `uv run ...`; nếu dùng venv thì bỏ `uv run` (ví dụ `recsys-cf ...` hoặc `python -m backend.cf.cli ...`).

## 2. Kiểm tra nhanh (không cần dữ liệu, ~20 giây)

```bash
uv run pytest -q
```

Kết quả mong đợi: `66 passed`. Các test dùng dữ liệu giả vài dòng và một LLM giả, kiểm tra: chống rò rỉ thời gian, các model, fusion và gom phiên bản, metric, gói bằng chứng, prompt không gửi ID thật, validator, cache Gemini và luồng end-to-end.

## 3. Chuẩn bị dữ liệu

Tải từ <https://amazon-reviews-2023.github.io/> và đặt vào `dataset/`:

```text
dataset/2023/Movies_and_TV.train.csv   # split tuần tự 5-core (benchmark), cột user_id,parent_asin,rating,timestamp,history
dataset/2023/Movies_and_TV.valid.csv
dataset/2023/Movies_and_TV.test.csv
dataset/meta_Movies_and_TV.jsonl       # metadata item (~1,3 GB)
dataset/Movies_and_TV.jsonl            # review thô (~8,4 GB), dùng làm kho bằng chứng
```

Dựng artifact (một lần, tổng ~1,5 giờ, phần lớn là kho review):

```bash
A=artifacts/movies-tv-2023
uv run recsys-data import-splits --train dataset/2023/Movies_and_TV.train.csv \
  --validation dataset/2023/Movies_and_TV.valid.csv --test dataset/2023/Movies_and_TV.test.csv --output-dir $A
uv run recsys-data enrich-catalog --catalog $A/catalog.parquet --metadata dataset/meta_Movies_and_TV.jsonl \
  --output $A/catalog_enriched.parquet                                                   # ~80 giây
uv run recsys-data group-editions --catalog $A/catalog_enriched.parquet --output $A/editions.parquet   # ~15 giây
uv run recsys-data import-reviews --reviews dataset/Movies_and_TV.jsonl --catalog $A/catalog.parquet \
  --output $A/reviews.sqlite                                                             # ~60 phút, 4,7 GB
```

Kết quả: `splits.parquet` (7,4 triệu tương tác), `queries_{val,test}.parquet` (657.203 query mỗi pha), `catalog_enriched.parquet` (197.943 item), `editions.parquet` (133.694 phim), `reviews.sqlite` (9,25 triệu review).

## 4. Train và đánh giá hệ thống gợi ý (pha test)

```bash
# Hai nhánh của fusion (~5 và ~3 phút)
uv run recsys-cf train --model itemknn --splits $A/splits.parquet --catalog $A/catalog_enriched.parquet \
  --phase test --output-dir runs/itemknn-movies-tv-2023/test
uv run recsys-cf train --model bm25 --splits $A/splits.parquet --catalog $A/catalog_enriched.parquet \
  --phase test --output-dir runs/bm25-movies-tv-2023/test

# Fusion với trọng số đã chọn trên validation, có gom phiên bản (không cần train)
uv run recsys-cf fuse --member itemknn=runs/itemknn-movies-tv-2023/test --member bm25=runs/bm25-movies-tv-2023/test \
  --weight itemknn=1 --weight bm25=0.1 --weight popularity=1 --method score --depth 200 \
  --editions $A/editions.parquet --output-dir runs/fusion-editions-movies-tv-2023/test

# Đánh giá trên mẫu 50.000 query (~7 phút); bỏ --query-limit để chạy toàn bộ test (~1,5 giờ)
uv run recsys-cf evaluate --model-dir runs/fusion-editions-movies-tv-2023/test --queries $A/queries_test.parquet \
  --query-limit 50000 --editions $A/editions.parquet --output runs/fusion-editions-movies-tv-2023/test_evaluation_50k.json
```

Mong đợi trong file JSON: `same_film.ndcg_at_10` ≈ 0,0646, `same_film.hr_at_10` ≈ 0,104, `same_film.recall_at_50` ≈ 0,166. Baseline khác: `--popularity` (Popularity), `--model bpr` (BPR-MF), `--model dense` (embedding, ~65 phút lần đầu để tính vector). Pha validation (chọn tham số) dùng `--phase validation` và `queries_val.parquet`.

## 5. Gợi ý + giải thích cho một người dùng

Cần API key Gemini (<https://aistudio.google.com/apikey>):

```bash
cp .env.example .env          # rồi sửa GEMINI_API_KEY=... trong .env (file này không bao giờ được commit)
uv run recsys-demo --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test
```

`--as-of` là thời điểm gợi ý (mili-giây): mọi lịch sử và review dùng cho gợi ý đều trước thời điểm này. Kết quả mong đợi (~20 giây, phần lớn là nạp model; 1 lời gọi Gemini ~1.900 token):

```text
 1. HP & thePrisoner of Azkaban SE(2D/BD)  [bm25+itemknn]
    You previously enjoyed the Harry Potter and the Chamber of Secrets movie, and this is another installment
    in the franchise [E1][E2][H1]. A reviewer mentions enjoying the well-crafted story and time travel elements [E4].
 2. Harry Potter and the Goblet of Fire ...
 ...
Sources:
  [H1] rated 5/5: Harry Potter and the Chamber of Secrets (Widescreen Edition) ...
  [E2] meta:B00271DNP4:people: Directors: Alfonso Cuarón. Cast: Daniel Radcliffe, Rupert Grint, Emma Watson
```

Thêm `--json` để xem toàn bộ response (bằng chứng, trạng thái từng câu, token, độ trễ); `--model gemini-3.8-flash` để dùng model mạnh hơn (mặc định `gemini-3.5-flash-lite`, rẻ nhất). Mọi lời gọi được cache trong `runs/llm/`, chạy lại cùng user không tốn token. Muốn thử user khác: lấy `user_id` và `as_of` từ `queries_test.parquet`.

Chỉ xem phần không cần LLM:

```bash
uv run recsys-profile --splits $A/splits.parquet --catalog $A/catalog_enriched.parquet \
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test          # hồ sơ
uv run recsys-evidence --model-dir runs/fusion-editions-movies-tv-2023/test --splits $A/splits.parquet \
  --catalog $A/catalog_enriched.parquet --reviews $A/reviews.sqlite --editions $A/editions.parquet \
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test --candidates 3   # gói bằng chứng
```

## 6. Thí nghiệm giải thích (smoke test và run G)

```bash
# 5 user ngẫu nhiên + 5 user có phim đúng trong top 10; ~4 phút, ~0,1 USD; ghi báo cáo từng bước
uv run recsys-llm --phase test --sample 5 --with-target 5 --output-dir runs/llm/smoke-10 \
  --report docs/results/smoke_test_10_record.md
```

Mỗi user: LLM giải thích top 3 hai lần (tự do / bắt buộc trích dẫn), validator kiểm từng câu, giám khảo `gemini-3.5-flash` chấm *supported / partial / unsupported*. Kết quả: `summary.json` (metric), `explain.json` (từng câu, lỗi, nhãn), `manual_check.csv` (40 câu để chấm tay) và báo cáo Markdown. Thí nghiệm chính: `--sample 200` (~45 phút, ~2 USD với giá trả phí; bậc miễn phí của Gemini dùng được nhưng có giới hạn tốc độ). Lưu ý: chạy lại với `--report` sẽ ghi đè cả phần đánh giá viết tay trong báo cáo.

Phân tích theo nhóm người dùng / phim (~15 phút): `uv run python -m backend.eval.segments --help`.

## 7. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| Windows: `implicit` báo `DLL load failed ... filename too long` | Đặt virtualenv ở đường dẫn ngắn (ví dụ `C:\venv\recsys`) hoặc bật LongPaths |
| `RuntimeError: Set GEMINI_API_KEY` | Tạo `.env` từ `.env.example` hoặc đặt biến môi trường `GEMINI_API_KEY` |
| Gemini `http_429` / chậm | Hết hạn mức (đặc biệt bậc miễn phí); client tự chờ và thử lại 3 lần; giảm `--workers` |
| `max_tokens` | Câu trả lời bị cắt, không được cache; chạy lại hoặc tăng ngân sách output |
| Dựng gói bằng chứng chậm (~9 giây/user) | Đọc review từ `reviews.sqlite` 4,7 GB; đặt `artifacts/` trên ổ cục bộ, không đồng bộ đám mây |
| `ModuleNotFoundError: backend` | Chạy lệnh từ thư mục gốc repo, hoặc cài bằng `pip install -e .` |

## Hạn chế đã biết

- **Gom phiên bản dựa trên luật** (title chuẩn hóa / đạo diễn + diễn viên chính và số lượt rating gần bằng nhau). Bản được bán riêng có số lượt rating riêng thì không gộp được: ví dụ demo ở mục 5 vẫn có hai bản *Order of the Phoenix* (hạng 4, 5) và một bản *Sorcerer's Stone (Ultimate Edition)* (hạng 7). Ngược lại, vài series dùng chung số lượt rating cho nhiều mùa nên các mùa bị gộp.
- 82.840 item (chủ yếu Prime Video) không có title; 25.806 trong số đó mượn được title từ phiên bản khác, phần còn lại chỉ hiển thị bằng tên đạo diễn/diễn viên.
- Giám khảo là LLM: tỉ lệ "supported" cần đối chiếu với chấm tay (`manual_check.csv`). Lọc theo độ giống trong validator chưa được hiệu chỉnh.
- ItemKNN học cả từ tương tác của người khác xảy ra sau thời điểm gợi ý (đánh giá leave-last-out chuẩn của tập dữ liệu), nên số liệu có thể cao hơn một chút so với thiết lập hoàn toàn theo thời gian.

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [docs/proposal.md](docs/proposal.md) | Đề xuất đề tài: bài toán, câu hỏi nghiên cứu, phạm vi |
| [docs/overview.md](docs/overview.md) | Tổng quan dễ hiểu: các bước, dữ liệu, thí nghiệm, khung báo cáo |
| [docs/architecture.md](docs/architecture.md) | Đặc tả: hợp đồng từng bước, schema, chống rò rỉ, thiết kế thí nghiệm (đầu file ghi chỗ khác thiết kế) |
| [docs/data/](docs/data/), [docs/cf/](docs/cf/), [docs/profile/](docs/profile/) | Hợp đồng dữ liệu, khảo sát dữ liệu thô, CF cho người mới, thiết kế BPR-MF và hồ sơ người dùng |
| [docs/results/smoke_test_10_record.md](docs/results/smoke_test_10_record.md) | Smoke test 10 record: dữ liệu, từng bước, đánh giá |
| [docs/guide.md](docs/guide.md) | Hướng dẫn chạy từng bước kèm input/output/lý do/ví dụ trên dữ liệu thật |
| [docs/results/results.md](docs/results/results.md) | Toàn bộ kết quả, kể cả kết quả âm; tiến độ |

Quy tắc dữ liệu quan trọng nhất: mọi thông tin dùng để gợi ý phải có $timestamp < as\_of$; pha validation chỉ fit trên `train`, pha test fit trên `train` + `validation` và không đọc `test`.
