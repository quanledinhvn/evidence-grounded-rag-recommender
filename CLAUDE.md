# CLAUDE.md

Hệ gợi ý phim có giải thích dựa trên bằng chứng (CF + BM25 + embedding → fusion có gom phiên bản → bằng chứng → LLM giải thích bắt buộc trích dẫn + validator; không xếp lại bằng LLM) trên Amazon Reviews 2023 – Movies & TV. Đồ án học phần Hệ gợi ý, Thạc sĩ KHMT, HUST. Tổng quan: [README.md](README.md), [docs/overview.md](docs/overview.md); đặc tả: [docs/architecture.md](docs/architecture.md).

Thứ tự file này:

1. **Thông tin đồ án** (cố định): quy tắc → môi trường → bản đồ code → dữ liệu và artifact → bất biến → hướng dẫn chạy test.
2. **Tiến độ dự án** (thay đổi theo thời gian): tiến độ và kết quả → việc tiếp theo → quyết định đang chờ → nhật ký phiên làm việc.

## Quy tắc bắt buộc

@AGENTS.md

- Code phải **đơn giản, dễ đọc, dễ theo dõi**: đây là đồ án, không phải sản phẩm. Ưu tiên hàm/class ngắn, comment giải thích "vì sao"; không thêm checksum, Protocol, store trừu tượng, hạ tầng cloud nếu không được yêu cầu.

## Môi trường

- Python ≥ 3.11, quản lý bằng `uv` (`uv sync --extra dev`, `uv run pytest -q`). Nếu máy không có `uv`: tạo venv rồi `pip install -e ".[dev]"`.
- Máy hiện tại không có `uv`; venv đã cài sẵn ở `%LOCALAPPDATA%\Temp\fq\.venv` (gồm cả `scikit-learn`, `sentence-transformers`, `torch` CPU). Chạy từ gốc repo, luôn dùng `-m` để Python lấy `backend/` của repo:
  `& "$env:LOCALAPPDATA\Temp\fq\.venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider` và `... -m backend.cf.cli ...`.
- Windows: `implicit` lỗi `DLL load failed ... filename too long` khi venv nằm ở đường dẫn dài; dùng venv ở đường dẫn ngắn. PowerShell 5.1 làm hỏng dấu `"` trong tham số truyền cho chương trình ngoài: commit bằng `git commit -F <file>`.
- CPU máy này chậm: embedding ~50 tài liệu/giây; tránh chạy hai việc nặng cùng lúc.
- Dự án làm ở local trên nhánh `f/quan`: commit bình thường, **không push**. `dataset/`, `artifacts/`, `runs/` không đưa lên Git.

## Bản đồ code

| File | Việc |
|---|---|
| `backend/data/importer.py` | 3 CSV split → `splits.parquet` (sắp theo user), `queries_{val,test}.parquet`, `catalog.parquet`; `reorganize` cho artifact cũ |
| `backend/data/enrich.py` | Nối metadata vào catalog → `catalog_enriched.parquet`, có `genres` và `item_doc` |
| `backend/data/reviews.py` | Review thô → `reviews.sqlite`; `item_reviews`, `user_review` |
| `backend/data/editions.py` | Gom các phiên bản (DVD, Blu-ray, Prime Video) của cùng một phim → `editions.parquet`; `borrow_titles` cho item không có title |
| `backend/profile/profile.py` | `load_history`, `load_catalog`, `build_profile` |
| `backend/cf/common.py` | `FitData`, `load_fit_data`, `Popularity`, `top_k`, `rank_items` |
| `backend/cf/itemknn.py`, `backend/cf/bpr.py` | Hai model CF |
| `backend/retrieval/bm25.py`, `backend/retrieval/dense.py` | Hai nhánh nội dung trên `item_doc` |
| `backend/retrieval/fusion.py` | Trộn các nhánh (theo điểm hoặc RRF), tune trọng số; với `editions`: một phiên bản mỗi phim, bỏ mọi phiên bản của phim đã xem |
| `backend/cf/evaluate.py` | NDCG@10, HR@10, Recall@50 cho mọi model; với `editions` thêm metric "cùng phim" (`same_film`) |
| `backend/cf/cli.py` | `recsys-cf train / evaluate / recommend / fuse / tune-fusion` cho mọi model |
| `backend/cf/segments.py` | Metric theo nhóm: độ dài lịch sử, độ phổ biến của target, target có title |
| `backend/generation/llm.py` | Client Gemini: JSON schema, cache trên đĩa, mức "thinking"; key đọc từ `GEMINI_API_KEY` / `.env` |
| `backend/generation/sources.py`, `explain.py` | Hiển thị bằng chứng cho LLM (mã C#/H#/E#, không gửi ID thật); LLM giải thích top 3, chế độ "tự do" và "bắt buộc trích dẫn" |
| `backend/generation/validator.py`, `judge.py` | Kiểm từng câu giải thích (nguồn, đúng phim, tên người, độ giống); LLM giám khảo chấm supported/partial/unsupported |
| `backend/generation/experiment.py` | Thí nghiệm giải thích (run G) + báo cáo Markdown từng record (`recsys-llm`) |
| `backend/service.py` | Luồng end-to-end cho một user: hồ sơ → fusion top 10 → bằng chứng → giải thích → validator (`recsys-demo`) |
| `backend/evidence/evidence.py`, `backend/evidence/cli.py` | Gói bằng chứng [H#]/[E#] cho một query (`recsys-evidence`); bỏ review chỉ nói về sách, bỏ tên giữ chỗ "Various" |
| `tests/` | 66 test, dữ liệu giả rất nhỏ, LLM giả (không gọi API) |

Mọi model (ItemKNN, BPR, BM25, Dense, Fusion, Popularity) có cùng giao diện `recommend(user_id, limit, exclude)` và `recommend_batch(user_ids, limit)`, trả `{"item_id", "score", "source"}`.

## Dữ liệu và artifact

| Đường dẫn | Nội dung |
|---|---|
| `artifacts/movies-tv-2023/splits.parquet` | 7.441.129 tương tác (train 6.126.723, validation 657.203, test 657.203), sắp theo user |
| `artifacts/movies-tv-2023/queries_{val,test}.parquet` | 657.203 query mỗi pha, mỗi user một target, `as_of` = timestamp target |
| `artifacts/movies-tv-2023/catalog_enriched.parquet` | 197.943 item; `item_doc` phủ 195.993; 82.840 item Prime Video không có title |
| `artifacts/movies-tv-2023/reviews.sqlite` | 9.249.801 review dùng làm bằng chứng (4,7 GB) |
| `artifacts/movies-tv-2023/editions.parquet` | `(item_id, work_id)`: 197.943 item → 133.694 phim; `work_id` = `item_id` nhỏ nhất của phim |
| `artifacts/movies-tv-2023/item_vectors.npy` | Vector embedding của mọi item (MiniLM-L6, 384 chiều), dùng chung cho mọi pha |
| `runs/validation/{itemknn,bm25,dense}/` | Model pha validation (fit trên train) và kết quả trên 20k query validation; `fusion-grid.json` |
| `runs/validation/{fusion,fusion-editions}/`, `*-eval10k.json` | Fusion pha validation không/có gom phiên bản và kết quả trên 10k query validation (cả metric cùng phim) |
| `runs/{itemknn,bpr}-movies-tv-2023/test/` | Model pha test |
| `runs/{itemknn,bpr,popularity,bm25,fusion}-movies-tv-2023/test_evaluation.json` | Kết quả trên toàn bộ test |
| `runs/dense-movies-tv-2023/test_evaluation_50k.json` | Kết quả Dense trên mẫu 50.000 query test (seed 42) |
| `runs/{bm25,dense,fusion}-movies-tv-2023/test/` | Model pha test; fusion chỉ có `config.json` trỏ tới ItemKNN và BM25 pha test |
| `runs/fusion-editions-movies-tv-2023/test/` | **Hệ thống cuối**: fusion pha test có gom phiên bản (`config.json` trỏ tới `editions.parquet`) |
| `runs/{itemknn,fusion,fusion-editions}-movies-tv-2023/test_evaluation_50k.json` | Kết quả trên mẫu 50k test, có cả metric cùng phim |
| `runs/examples/evidence_example.json` | Gói bằng chứng mẫu cho user Harry Potter (hệ thống cuối) |
| `runs/examples/recommend_5_users_editions.{json,txt}` | Gợi ý + bằng chứng cho 5 query test cố định (seed 42); `recommend_5_users.json` là bản trước khi gom phiên bản |
| `runs/llm/smoke-10/` | Smoke test 10 record (test): gói bằng chứng, câu trả lời LLM, nhãn giám khảo, cache; báo cáo ở `docs/smoke_test_10_record.md` |
| `runs/llm/dev-10/`, `runs/llm/dev-inpool-50/`, `runs/rerank/` | Smoke test trên validation: thử LLM xếp lại, learning-to-rank, cross-encoder (đã bỏ khỏi đề tài, giữ số liệu) |
| `runs/analysis/segments_test50k.json` | Phân tích theo nhóm trên mẫu 50k test |
| `.env` | `GEMINI_API_KEY=...` — chỉ ở máy local, git-ignore |
| `runs/cf-movies-tv-2023-full/20261009T135249Z/` | Lượt BPR cũ trên AWS; chỉ còn giá trị tham khảo |
| `dataset/Movies_and_TV.jsonl`, `dataset/meta_Movies_and_TV.jsonl` | Review thô (17,3 triệu) và metadata thô |

Đặc điểm cần nhớ: trung vị 5 tương tác train mỗi user; split không có review text; một phim có nhiều `parent_asin` (DVD, Blu-ray, Prime Video) — ~93 nghìn item có ít nhất một phiên bản khác, xem bước 10.

## Bất biến

- Mọi dữ liệu dùng cho một query phải có `timestamp < as_of` (lịch sử, review làm bằng chứng).
- Pha `validation` chỉ fit trên `train`; pha `test` fit trên `train` + `validation`, không đọc `test`. Chọn cấu hình trên validation, test chỉ dùng để báo cáo.
- Bằng chứng review không bao giờ lấy review của chính user đang được gợi ý.
- Đánh giá luôn lấy 50 item mỗi query; `recommend` giới hạn `limit ≤ 50`.
- Item được đánh số theo `item_id` tăng dần; điểm bằng nhau thì phá hòa theo `item_id`.
- Hệ thống cuối không gợi ý lại một phim user đã xem dưới phiên bản khác, và mỗi phim chỉ xuất hiện một lần trong danh sách.

## Quy ước báo cáo

- Metric viết dạng **%**: NDCG@10 4,91%, HR@10 7,68%, Recall@50 11,9% (dấu phẩy thập phân). Mức thay đổi tương đối ghi rõ là tương đối, ví dụ "+14% so với ItemKNN".
- Trọng số fusion viết dạng **% tổng 100%**, ví dụ ItemKNN 47,6%, BM25 4,8%, popularity 47,6%. Chỉ tỉ lệ giữa các trọng số ảnh hưởng thứ hạng; CLI nhận giá trị tương đối (`--weight itemknn=1 --weight bm25=0.1 --weight popularity=1` tương đương 47,6% / 4,8% / 47,6%). % trọng số **không** phải tỉ lệ ảnh hưởng thực tế lên thứ hạng; muốn đo đóng góp thì dùng ablation.
- Mỗi bảng ghi rõ tập và cỡ mẫu (toàn bộ test, mẫu 50k test, mẫu 10k/20k validation).
- Ghi rõ loại metric: **đúng mã** (target đúng `item_id`, metric gốc) hay **cùng phim** (bất kỳ phiên bản nào của phim target, cần `--editions`).

## Hướng dẫn chạy test các phần đã implement

Ví dụ xuyên suốt là một user thật: `AGXVBIUFLFGMVLATYXHJYL4A5Q7Q`. Lịch sử của user này có 7 phim ở train (2006–2017), một phim ở validation là **Harry Potter phần 2 — Chamber of Secrets** (`B002DQDV6Y`, bản Prime Video không có title, xem lúc `1530556502815`; trước 17/10 ghi nhầm là phần 1, xác định lại nhờ số lượt rating 64.870 trùng các bản DVD/Blu-ray của phần 2), và target test là **Harry Potter phần 3** (`B00271DNP4`, `as_of = 1530556520803`, tức 18 giây sau).

Trong các lệnh dưới, `python` là `& "$env:LOCALAPPDATA\Temp\fq\.venv\Scripts\python.exe"` (hoặc `uv run python`), `$a = "artifacts/movies-tv-2023"`, chạy từ gốc repo.

### 0. Unit test (không cần dữ liệu thật, ~15 giây)

```powershell
python -m pytest -q -p no:cacheprovider                         # toàn bộ 66 test
python -m pytest -q -p no:cacheprovider tests/test_retrieval.py  # một file
```

| File | Kiểm tra gì |
|---|---|
| `tests/test_data.py` | Import giữ nguyên split, `review_id`, query, catalog; từ chối dữ liệu hỏng; `reorganize` sắp theo user; enrich làm sạch HTML, `genres` bỏ nhãn không phải thể loại, `item_doc` |
| `tests/test_profile.py` | Pha validation không thấy split validation; tương tác tại `as_of` bị loại; thích/ghét; tương tác mới nhất thắng; công thức trọng số; title mượn trích đúng phiên bản nguồn |
| `tests/test_cf.py` | Pha fit đúng split; Popularity; cosine và cắt hàng xóm; ItemKNN ưu tiên phim vừa xem, `because`; BPR không gợi ý phim đã xem; metric, metric cùng phim; chia nhóm (lịch sử, độ phổ biến, title); CLI |
| `tests/test_retrieval.py` | Lịch sử "đã thích" có trọng số; BM25 tìm phim chung đạo diễn/diễn viên; Dense xếp theo cosine; công thức RRF; Fusion lưu/nạp lại; Fusion một phiên bản mỗi phim, bỏ phiên bản của phim đã xem |
| `tests/test_editions.py` | Chuẩn hóa title giữ "Season 1", xử lý "X, The:", "Part One", năm trong ngoặc; khóa người bỏ "Various"; gom phiên bản cần chung title/người **và** số lượt rating gần nhau (cùng title ≤ 5%, chỉ cùng người ≤ 2%); không gộp bản làm lại, mùa khác, phần tiếp theo, bộ sưu tập; mượn title |
| `tests/test_reviews.py` | Chỉ giữ review hữu ích của item trong catalog; lấy review trước `as_of`, loại review của chính user |
| `tests/test_evidence.py` | [H#] dùng review của chính user; [E#] gồm metadata và review của người khác trước `as_of`; bỏ review chỉ nói về sách; tên viết tắt ("Michael E. Satrazemis") không bị cắt; bỏ "Various"; title mượn trích đúng phiên bản nguồn |
| `tests/test_generation.py` | Prompt không chứa `user_id`/mã Amazon; bản trích dẫn có [E#]/[H#], bản tự do không; parse JSON, tối đa 2 câu, tách mã viết trong câu; validator gọi đúng tên từng lỗi, cho phép tên chính phim; renderer; giám khảo; cache Gemini không lưu key, không cache câu trả lời bị cắt |
| `tests/test_service.py` | Luồng end-to-end giữ thứ tự fusion, chỉ giữ câu hợp lệ, một lời gọi LLM; user không có lịch sử → `cold_start`, không gọi LLM |

Kết quả mong đợi: `66 passed`.

### 1. Nhập split — `backend/data/importer.py`

```powershell
python -m backend.data.cli import-splits --train dataset/2023/Movies_and_TV.train.csv `
  --validation dataset/2023/Movies_and_TV.valid.csv --test dataset/2023/Movies_and_TV.test.csv `
  --output-dir artifacts/movies-tv-2023
python -m backend.data.cli reorganize --artifacts-dir artifacts/movies-tv-2023   # chỉ cho artifact cũ
```

- **Input:** 3 CSV, cột `user_id,parent_asin,rating,timestamp,history`. Ví dụ một dòng train: `AGXVBIUFLFGMVLATYXHJYL4A5Q7Q,B0091VXC54,5.0,1443550066000,B0002J58ME`.
- **Các bước và lý do:**
  1. Kiểm tra ID không rỗng, rating 1–5, timestamp dương: dữ liệu hỏng sẽ làm sai model mà không báo lỗi.
  2. Kiểm tra target không nằm trong `history` của chính dòng đó: nếu có, model "đoán" đúng một phim user đã xem, metric bị thổi phồng.
  3. Đổi `parent_asin` thành `item_id`, gắn nhãn split, tạo `review_id = amazon-2023:<split>:<dòng CSV>` để truy ngược về file gốc.
  4. Tách target validation/test ra file query riêng: evaluator đọc đáp án, model thì không.
  5. Ghi `splits.parquet` đã sắp theo user, nhóm 20.000 dòng: Parquet nhớ min/max user của từng nhóm nên đọc lịch sử một user chỉ chạm 1–2 nhóm (hồ sơ một user ~6 s → ~0,3 s).
- **Output:** `splits.parquet`, `queries_val.parquet`, `queries_test.parquet`, `catalog.parquet`, `split_stats.json`.
- **Ví dụ:** dòng trên thành `{user_id: AGXV…, item_id: B0091VXC54, rating: 5.0, timestamp: 1443550066000, review_id: amazon-2023:train:3, split: train}`. Query test của user: `{as_of: 1530556520803, target_item_id: B00271DNP4, target_rating: 5.0}`.

### 2. Nối metadata — `backend/data/enrich.py`

```powershell
python -m backend.data.cli enrich-catalog --catalog $a/catalog.parquet `
  --metadata dataset/meta_Movies_and_TV.jsonl --output $a/catalog_enriched.parquet
```

- **Input:** `catalog.parquet` (danh sách `item_id`) và file metadata JSONL (một sản phẩm mỗi dòng).
- **Các bước và lý do:**
  1. Chỉ giữ bản ghi có `parent_asin` thuộc catalog, nối **chính xác** theo ID: không ghép gần đúng theo tên để tránh gán nhầm phim.
  2. Làm sạch HTML; `details` chỉ giữ khóa hữu ích (`Directors`, `Starring`, `Genre`…).
  3. `genres`: nhãn thể loại, bỏ nhãn định dạng/hãng phim/khuyến mãi (Blu-ray, "All … Titles", "Warner Home Video", "Today's Deals"): nhãn này không nói gì về nội dung và làm nhiễu tìm kiếm.
  4. `item_doc` = `Title. Genres. Directors. Cast. Description`, phần quan trọng đứng trước: văn bản chung cho BM25, embedding và bằng chứng; phim không có title vẫn được mô tả bằng đạo diễn và diễn viên.
  5. `source_id = meta:<item_id>` để lời giải thích trích dẫn được nguồn.
- **Output:** `catalog_enriched.parquet` và `catalog_enriched.stats.json` (độ phủ từng trường).
- **Ví dụ:** `B00271DNP4` → `title` rỗng (Prime Video), `item_doc = "Directors: Alfonso Cuarón. Cast: Daniel Radcliffe, Rupert Grint, Emma Watson"`. Mất khoảng 80 giây.

### 3. Hồ sơ người dùng — `backend/profile/profile.py`

```powershell
python -m backend.profile.cli --splits $a/splits.parquet --catalog $a/catalog_enriched.parquet `
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test
```

- **Input:** user, thời điểm `as_of` (mili-giây), pha (`validation` hoặc `test`).
- **Các bước và lý do:**
  1. Đọc tương tác có `timestamp < as_of` từ split được phép (test: train + validation): **chống rò rỉ** tương lai và test.
  2. Mỗi phim giữ tương tác mới nhất: ý kiến gần nhất của user là đáng tin nhất.
  3. Rating ≥ 4 là thích, ≤ 2 là không thích, 3 chỉ tính là "đã xem": tín hiệu trung tính không nên kéo truy vấn về phía nào.
  4. Trọng số = $|rating - 3|/2 \times 2^{-\text{tuổi}/90\text{ ngày}}$: sở thích mạnh và gần đây quan trọng hơn.
  5. Gắn text từng trường kèm `source_id`, đánh số `H1, H2, …`: để bước giải thích trích dẫn được lịch sử.
- **Output:** JSON gồm `seen_item_ids`, `positive_signals`, `negative_signals`, `items_without_metadata`.
- **Ví dụ:** 8 phim đã xem (7 train + Harry Potter 2 ở validation; Harry Potter 3 **không** có vì xảy ra đúng lúc `as_of`). `H1 = B002DQDV6Y`, weight 0,99999 (xem 18 giây trước); `H2 = B01LWY8995`, weight 0,0186 (xem 517 ngày trước). Mất ~0,3 giây.

### 4. Train model — `recsys-cf train`

```powershell
python -m backend.cf.cli train --model itemknn --splits $a/splits.parquet --catalog $a/catalog_enriched.parquet `
  --phase test --output-dir runs/itemknn-movies-tv-2023/test
# --model bpr | bm25 | dense ; bm25 và dense cần catalog_enriched (đọc item_doc)
```

- **Input:** `splits.parquet`, catalog, pha, tham số (mặc định ItemKNN 400 hàng xóm, $\gamma = 0{,}7$; BPR 64 factors, 60 epoch; BM25 $k_1 = 1{,}2$, $b = 0{,}75$; Dense MiniLM-L6, 64 token).
- **Các bước và lý do:**
  1. `load_fit_data` chỉ đọc split được phép của pha (test: train + validation, **không** test) và xếp lịch sử từng user theo thời gian: thứ tự cần cho trọng số độ gần đây.
  2. ItemKNN: cosine giữa các phim từ ma trận user × phim, mỗi phim giữ 400 hàng xóm. BPR: `implicit` học vector user/phim từ rating ≥ 4.
  3. BM25: đếm từ trong `item_doc`, bỏ stopword và từ xuất hiện ở hơn 15% phim (như "cast", "directors"); truy vấn của user là các từ trong `item_doc` của những phim họ **thích** gần đây (mới nhất trọng số 1, rồi 0,7, 0,49…). Từ hiếm trùng nhau (tên đạo diễn, diễn viên) cho điểm cao.
  4. Dense: biến `item_doc` thành vector 384 chiều (chỉ tính một lần, lưu `item_vectors.npy`); vector user = trung bình có trọng số của vector các phim thích gần đây; điểm = cosine. Bắt được phim giống về ý nghĩa dù không chung từ.
- **Output:** thư mục model gồm `config.json`, `fit_data.npz` và `similarity.npz` / `factors.npz` / `words.npz + weights.npz` / `vectors.npy`.
- **Thời gian:** ItemKNN ~5 phút, BPR ~4 phút, BM25 ~3 phút, Dense ~65 phút lần đầu (tính vector), vài phút các lần sau.

### 5. Gợi ý cho một user — `recsys-cf recommend`

```powershell
python -m backend.cf.cli recommend --model-dir runs/itemknn-movies-tv-2023/test `
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --limit 3
```

- **Input:** thư mục model, user, số item; `--exclude` để loại thêm phim.
- **Các bước và lý do:** chấm điểm → bỏ phim đã xem → lấy top → thiếu thì bổ sung popularity để luôn trả đủ danh sách.
- **Output:** danh sách `{item_id, score, source}`; ItemKNN thêm `because` (phim trong lịch sử góp điểm nhiều nhất), Fusion thêm `members` (nhánh nào tìm ra phim đó).
- **Ví dụ:** ItemKNN xếp hạng 1 **`B00271DNP4` (Harry Potter 3) — đúng target test**, `because = [{item_id: B002DQDV6Y, similarity: 0,341, weight: 1,0}]`: user vừa xem Harry Potter 2, và nhiều người xem phần 2 cũng xem phần 3. Mất ~15 giây (chủ yếu nạp model).

### 6. Đánh giá — `recsys-cf evaluate`

```powershell
python -m backend.cf.cli evaluate --model-dir runs/itemknn-movies-tv-2023/test `
  --queries $a/queries_test.parquet --output runs/itemknn-movies-tv-2023/test_evaluation.json
# --popularity: baseline Popularity; --query-limit 20000: mẫu cố định (seed 42) để chạy nhanh
# --editions $a/editions.parquet: báo thêm metric "cùng phim" (same_film), xem bước 10
```

- **Input:** model, file query (`user_id`, `target_item_id`, `target_rating`).
- **Các bước và lý do:** với mỗi query lấy 50 item, tìm hạng của target; HR@10 = có trong top 10; NDCG@10 = $1/\log_2(rank+1)$ (đúng ở hạng cao được thưởng nhiều hơn); Recall@50 = có trong 50 ứng viên (cái mà các bước sau còn có thể xếp lại). Báo thêm riêng cho target rating ≥ 4 (`positive_target`).
- **Output:** JSON gồm `ndcg_at_10`, `hr_at_10`, `recall_at_50`, `fallback_rate`, `item_coverage`, `mean_latency_ms`, `positive_target`; với `--editions` thêm `same_film` (cùng các metric đó, nhưng target trúng khi danh sách có **bất kỳ phiên bản nào** của phim target — DVD và Blu-ray của một phim là cùng một đáp án).
- **Thời gian trên toàn bộ test:** ItemKNN ~10 phút, Popularity ~2 phút, BM25/Dense/BPR lâu hơn nhiều (BPR ~80 phút) vì phải chấm hàng trăm nghìn phim cho mỗi user; dùng `--query-limit 20000` khi chỉ cần số nhanh.

### 7. Fusion — `recsys-cf tune-fusion`, `recsys-cf fuse`

```powershell
# Tìm trọng số trên VALIDATION (model pha validation)
python -m backend.cf.cli tune-fusion --method score --depth 200 --member itemknn=runs/validation/itemknn `
  --member bm25=runs/validation/bm25 --member dense=runs/validation/dense --grid itemknn=1 --grid bm25=0,0.1,0.3 `
  --grid dense=0,0.1,0.3 --grid popularity=0,0.25,0.5,1,2 --queries $a/queries_val.parquet --query-limit 10000 `
  --output runs/validation/fusion-score-grid.json
# Ghép model pha TEST với trọng số đã chọn; --editions = hệ thống cuối (bước 10)
python -m backend.cf.cli fuse --member itemknn=runs/itemknn-movies-tv-2023/test --member bm25=runs/bm25-movies-tv-2023/test `
  --weight itemknn=1 --weight bm25=0.1 --weight popularity=1 --method score --depth 200 `
  --editions $a/editions.parquet --output-dir runs/fusion-editions-movies-tv-2023/test
```

- **Input:** các model đã train (cùng pha), trọng số từng nhánh.
- **Các bước và lý do:**
  1. Mỗi nhánh trả 200 ứng viên; chỉ tính những phim nhánh đó **tự tìm ra** (phim nó bổ sung bằng popularity không được tính).
  2. `method score` (mặc định): điểm của nhánh chia cho điểm cao nhất của nhánh đó với user, nhân trọng số, cộng lại, rồi cộng $w_{pop} \cdot (1 - \text{hạng popularity}/N)$. Giữ được **độ tự tin** của từng nhánh và cho phép phim phổ biến nằm sâu trong danh sách được đẩy lên.
  3. `method rrf`: chỉ dùng hạng, $w_m / (60 + r)$. Đã thử và không vượt ItemKNN, giữ lại để so sánh.
  4. `tune-fusion` tính danh sách của từng nhánh một lần rồi thử mọi tổ hợp trọng số trên cùng một mẫu query validation.
  5. Với `--editions`: duyệt danh sách đã trộn từ trên xuống, bỏ phim user đã xem **dưới bất kỳ phiên bản nào** và bỏ phiên bản thứ hai trở đi của một phim đã có trong danh sách — người dùng không cần thấy "Cast Away" ba lần, càng không cần thấy lại phim vừa xem.
- **Output:** bảng mọi tổ hợp (tốt nhất trước); thư mục fusion chỉ chứa `config.json` (nhánh nào, trọng số, method, depth). Kết quả gợi ý có thêm `members`: nhánh nào đã tìm ra phim đó.
- **Ví dụ:** trên 10k query validation, ItemKNN đứng riêng NDCG@10 6,03% → fusion (ItemKNN 47,6%, BM25 4,8%, popularity 47,6%) 6,96%. Trên toàn bộ test: 4,91% → 5,60%.
- **Trọng số:** CLI nhận giá trị tương đối (1 / 0,1 / 1); trong báo cáo viết dạng % tổng 100% (47,6% / 4,8% / 47,6%).

### 8. Kho review — `recsys-data import-reviews`

```powershell
python -m backend.data.cli import-reviews --reviews dataset/Movies_and_TV.jsonl --catalog $a/catalog.parquet --output $a/reviews.sqlite
```

- **Input:** 17,3 triệu review thô.
- **Các bước và lý do:** chỉ giữ review của phim trong catalog, ≥ 10 từ (câu như "Five stars" không có thông tin), không nói về giao hàng/đĩa/giá (không phải ý kiến về phim); ghi vào SQLite với index `(item_id, timestamp)` để lấy "review của phim X viết trước thời điểm T" và `(user_id, item_id)` để lấy review của chính user.
- **Output:** `reviews.sqlite`: giữ 9.249.801; bỏ 1.535.240 ngoài catalog, 5.917.406 quá ngắn, 625.867 không nói về phim. Mất ~60 phút.

### 9. Gói bằng chứng — `recsys-evidence`

```powershell
python -m backend.evidence.cli --model-dir runs/fusion-editions-movies-tv-2023/test --splits $a/splits.parquet `
  --catalog $a/catalog_enriched.parquet --reviews $a/reviews.sqlite --editions $a/editions.parquet `
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test --candidates 3 `
  --output runs/examples/evidence_example.json
```

- **Input:** model đề xuất ứng viên (hệ thống cuối: fusion có gom phiên bản), user, `as_of`, kho review, `editions.parquet` (tùy chọn).
- **Các bước và lý do:**
  1. Dựng hồ sơ (bước 3) và lấy top ứng viên, loại mọi phim đã xem (mọi phiên bản).
  2. Với `--editions`: item không có title (Prime Video) **mượn title** của một phiên bản khác cùng phim, `source_id` trỏ về đúng phiên bản cho title (`meta:B01KW24RCC:title`) — không có title thì LLM không biết đó là phim gì; trỏ đúng nguồn để validator kiểm tra được. Áp dụng cho cả [H#] và [E#]; 25.806 / 82.842 item không có title được lợi.
  3. [H#]: phim user thích/không thích, kèm review của chính user nếu có — để LLM nói "vì bạn đã thích X".
  4. [E#] cho mỗi ứng viên: title, thể loại, đạo diễn/diễn viên, mô tả, và tối đa 2 review của **người khác** viết **trước** `as_of`, ưu tiên review có nhiều từ trùng với phim user thích rồi review được vote "helpful" nhiều. Mỗi mục có `source_id` để validator kiểm tra.
  5. Làm sạch: bỏ review **chỉ nói về sách** (có "book/novel/read" mà không có "movie/film/watch/actor/scene…"; một số review sách bị xếp nhầm vào phim); bỏ dòng người giữ chỗ ("Directors: Various. Cast: Various"); tách `item_doc` không cắt tên viết tắt ("Michael E. Satrazemis", "Jr.").
- **Output:** JSON `{candidates, history: {H1: …}, evidence: {E1: …}, retrieval}`.
- **Ví dụ** (`runs/examples/evidence_example.json`): ứng viên = Harry Potter 3 (`B00271DNP4`, **đúng target**), Harry Potter 4, Harry Potter 1 — cả ba do ItemKNN và BM25 cùng tìm ra. H1 = `B002DQDV6Y` "Harry Potter and the Chamber of Secrets (Widescreen Edition)" (title mượn). E1 = `meta:B01KW24RCC:title` "HP & the Prisoner of Azkaban SE"; E2 = `meta:B00271DNP4:people` "Directors: Alfonso Cuarón. Cast: Daniel Radcliffe, Rupert Grint, Emma Watson"; E3 = review của một phụ huynh (con xem 3 lần trong 2 ngày). Mất ~25 giây cho cả lệnh (nạp model); riêng bằng chứng của 3 ứng viên < 2 giây.

### 10. Gom phiên bản của cùng một phim — `recsys-data group-editions`

```powershell
python -m backend.data.cli group-editions --catalog $a/catalog_enriched.parquet --output $a/editions.parquet
```

- **Vấn đề:** một phim có nhiều `item_id` (DVD, Blu-ray, VHS, nhiều bản Prime Video). Không gom thì (a) user đã xem "Cast Away" bản Prime vẫn được gợi ý "Cast Away" bản DVD; (b) một danh sách có ba bản "Cast Away"; (c) gợi ý bản DVD khi target là bản Blu-ray bị tính là **trượt**.
- **Input:** `catalog_enriched.parquet` (cột `title`, `item_doc`, `rating_number`).
- **Các bước và lý do:**
  1. Hai item là **ứng viên** cùng phim nếu chung title chuẩn hóa (bỏ nhãn định dạng trong ngoặc như "[Blu-ray]", "(Two-Disc Special Edition)" và năm trong ngoặc "(2010)", nhưng giữ "(Season 1)"; "Twilight Saga, The: …" → "the twilight saga …"; "Part One" → "part 1") hoặc chung đạo diễn đầu + diễn viên đầu (Prime Video thường chỉ có tên người).
  2. Chỉ gộp khi **số lượt rating gần bằng nhau**: Amazon gộp rating mọi định dạng của một phim (Cast Away: DVD 16.753, ba bản Prime 16.747 / 16.755 / 16.756), còn bản làm lại và các mùa khác nhau có số lượt khác hẳn (A Christmas Carol 1951 ~9.900, bản 1984 ~13.700; Under the Dome 20.010 / 12.169). Ngưỡng: **cùng title lệch ≤ 5%** (số lượt được ghi ở các thời điểm khác nhau: Eclipse 20.129 / 20.797, lệch 3,3%), **chỉ cùng người lệch ≤ 2%** (tín hiệu yếu hơn).
  3. Chặn gộp dây chuyền: hai nhóm đã có title thì title phải trùng (không nối "Spider-Man 2" với "Spider-Man 3" qua một bản Prime không title); bỏ tên giữ chỗ "Various"/"None" (3.494 item "Various|Various" từng nối American Beauty với Chinatown); bộ sưu tập ("8-Film Collection", "Complete Series", box set) không gộp với gì.
  4. `work_id` = `item_id` nhỏ nhất của nhóm (union-find).
- **Output:** `editions.parquet` `(item_id, work_id)`: 197.943 item → 133.694 phim; 92.681 item có ít nhất một phiên bản khác. Mất ~15 giây.
- **Dùng ở:** `fuse --editions` (một phiên bản mỗi phim, bỏ mọi phiên bản của phim đã xem), `evaluate --editions` (metric cùng phim), `recsys-evidence --editions` (mượn title).
- **Ví dụ:** `B00003CXRP` (Cast Away DVD), `B0026NZR98`, `B0044WUIHA`, `B0026O475M` (Prime) → một phim; `B0152ZY7KQ` và `B015301ZQ4` (cùng một mùa The Walking Dead) → một phim; 7 bản *Eclipse* và các bản *Breaking Dawn Part 1* (kể cả "Twilight Saga, The: …", "Part One") → mỗi phim một nhóm; Harry Potter 1–4, Breaking Dawn 1/2, Aladdin 1992/2019 tách riêng.
- **Hạn chế đã biết:** gộp thiếu khi title khác cách viết ("Harry Potter and Sorcerer's Stone (Special Edition/2 Disc/BD" không gộp với "… the Sorcerer's Stone (4K)"); một series có thể dùng chung số lượt rating cho nhiều mùa nên các mùa bị gộp (nhóm lớn nhất: "Downton Abbey", 24 item) — ảnh hưởng nhỏ: chỉ 0,12% target test là phiên bản khác của phim user đã xem.

### 11. Giải thích có trích dẫn và demo — `recsys-demo` (`backend/service.py`)

```powershell
python -m backend.service --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test   # thêm --json để xem toàn bộ
```

- **Cần:** API key Gemini trong biến môi trường `GEMINI_API_KEY` hoặc file `.env` ở gốc repo (`GEMINI_API_KEY=...`, đã git-ignore, **không bao giờ commit**). Mọi lời gọi được cache ở `runs/llm/demo_cache.jsonl`; chạy lại cùng user không tốn token.
- **Input:** user, `as_of`, pha; model mặc định `gemini-3.5-flash-lite` (`--model` để nâng cấp, ví dụ `gemini-3.5-flash`).
- **Các bước và lý do:**
  1. Hồ sơ (bước 3) + **fusion có gom phiên bản, top 10 giữ nguyên thứ tự fusion** — không xếp lại: LLM, cross-encoder và learning-to-rank đều đã thử và không vượt thứ tự fusion (xem mục Tiến độ).
  2. Gói bằng chứng cho top 10 (bước 9), title mượn cho phim không có title.
  3. **Một lời gọi LLM** giải thích top 3 (`backend/generation/explain.py`): prompt chỉ chứa mã C1–C3, [H#], [E#] và nội dung nguồn — **không gửi `user_id` hay mã Amazon**; nội dung review nằm trong `<data>` và model được dặn không làm theo lệnh trong đó. Đầu ra ép đúng JSON schema `{"items": [{"item_id": "C1", "claims": [{"text", "evidence_ids", "history_ids"}]}]}`, tối đa 2 câu/phim; mã [E#] model lỡ viết trong câu được tách ra và đưa vào danh sách nguồn.
  4. **Validator** (`backend/generation/validator.py`) kiểm từng câu: có nguồn; E# tồn tại và **thuộc đúng phim** đang giải thích; H# thuộc hồ sơ; tên người/tựa (2+ từ viết hoa) có trong nguồn được dẫn hoặc là tên chính phim đó; độ giống MiniLM giữa câu và nguồn ≥ ngưỡng (mặc định 0,3, chưa hiệu chỉnh). Câu lỗi bị bỏ; phim không còn câu nào → `evidence_unavailable`.
  5. Ghép câu hợp lệ kèm trích dẫn từ ID đã kiểm tra (renderer tự chèn `[E2][H1]`, không dùng chữ của model).
- **Output:** top 10 kèm `found_by` (nhánh fusion), top 3 kèm lời giải thích + nguồn được dẫn; `trace` có token, độ trễ. User không có phim nào đã chấm điểm → `cold_start`, không gọi LLM.
- **Ví dụ** (user Harry Potter): 1. *Harry Potter 3* (đúng target) — "You previously enjoyed the Harry Potter and the Chamber of Secrets movie, and this is another installment in the franchise [E1][E2][H1]. A reviewer mentions enjoying the well-crafted story and time travel elements [E4]." Một lời gọi ~2.300 token, ~20 s tổng (phần lớn nạp model).

### 12. Thí nghiệm giải thích (run G) và smoke test — `recsys-llm` (`backend/generation/experiment.py`)

```powershell
# Smoke test 10 record, kèm báo cáo chi tiết từng bước
python -m backend.generation.experiment --phase test --sample 5 --with-target 5 --output-dir runs/llm/smoke-10 --report docs/smoke_test_10_record.md
# Thí nghiệm chính (chưa chạy): 200 user test ngẫu nhiên
python -m backend.generation.experiment --phase test --sample 200 --output-dir runs/llm/test-200
```

- **Mẫu:** `--sample` query ngẫu nhiên (`Random(42)`, không chọn theo việc hệ thống trúng target) + `--with-target` query đầu tiên trong 5.000 query ngẫu nhiên có target trong top 10 (để xem giải thích của gợi ý đúng; báo riêng).
- **Các bước:** gói bằng chứng (lưu `packages.jsonl`, chạy lại thì dùng tiếp) → LLM giải thích top 3 hai lần: **tự do** (cùng thông tin, không có mã nguồn) và **bắt buộc trích dẫn** → validator cho bản trích dẫn → **giám khảo** `gemini-3.5-flash` (thinking low, JSON schema) chấm từng câu *supported / partial / unsupported* so với toàn bộ nguồn của phim và lịch sử → `summary.json`, `explain.json`, `manual_check.csv` (40 câu để chấm tay), và `--report` (Markdown từng record).
- **Chỉ số:** NDCG@10/HR@10 cùng phim của từng nhóm; số câu, số phim có giải thích (độ phủ), tỉ lệ supported/unsupported cho tự do, trích dẫn trước và sau validator; lý do validator loại; bảng ngưỡng độ giống 0–0,5; token, độ trễ p50/p95.
- **Chi phí:** khoảng 1.600 + 300 token mỗi lời gọi giải thích; giám khảo ~3.200 + 1.200 token mỗi user.

### 13. Phân tích theo nhóm — `python -m backend.cf.segments`

```powershell
python -m backend.cf.segments --member itemknn=runs/itemknn-movies-tv-2023/test --member bm25=runs/bm25-movies-tv-2023/test `
  --member fusion=runs/fusion-editions-movies-tv-2023/test --queries $a/queries_test.parquet --catalog $a/catalog_enriched.parquet `
  --editions $a/editions.parquet --query-limit 50000 --output runs/analysis/segments_test50k.json
```

- Chia query theo **độ dài lịch sử** (3–5, 6–20, > 20; tập 5-core nên không có 1–2), **độ phổ biến của target** (head = top 1% catalog, torso = 9% tiếp, tail = 90% còn lại, tính từ dữ liệu fit) và **target có title hay không**; báo NDCG@10, HR@10, Recall@20, Recall@50 cùng phim cho từng model. Mất ~15 phút trên mẫu 50k.

---

# Tiến độ dự án

Các mục dưới đây thay đổi theo thời gian; phần trên là thông tin cố định của đồ án.

## Tiến độ (cập nhật 2026-10-18)

| Hạng mục | Tác dụng trong đồ án | Trạng thái |
|---|---|---|
| Nhập 3 split | Biến CSV gốc thành bảng chuẩn và **tách target đánh giá ra file query riêng**, để model không bao giờ thấy đáp án. Là nền cho mọi bước sau. | Xong; sắp theo user để đọc nhanh |
| Nối metadata, `genres`, `item_doc` | Gắn nội dung phim vào từng item và tạo một văn bản mô tả chung. Không có bước này thì không làm được BM25, embedding, bằng chứng [E#]. | Xong |
| Gom phiên bản (`editions.parquet`) | Biết item nào là cùng một phim (DVD/Blu-ray/Prime): không gợi ý lại phim đã xem dưới bản khác, không lặp một phim trong danh sách, đo đúng "trúng phim", cho phim Prime mượn title. | Xong 17/10, sửa 18/10 (Twilight); 133.694 phim từ 197.943 item |
| ① Hồ sơ người dùng | Tóm tắt sở thích của user **tại thời điểm gợi ý**; đầu vào cho bằng chứng [H#]. | Xong, ~0,3 s/user |
| ② ItemKNN | Nhánh CF chính: "người xem A thường xem B", ưu tiên phim vừa xem; `because` cho sẵn bằng chứng. | Xong, đã tune và đánh giá |
| ② BPR-MF | Baseline học máy cổ điển (run B). | Xong |
| Popularity | Mốc thấp nhất (run A) và nhánh trong fusion. | Xong |
| ② BM25 | Phim chung **từ hiếm** (đạo diễn, diễn viên, thể loại) với phim user thích (run C). | Xong; test NDCG@10 1,35% |
| ② Embedding (Dense) | Phim giống về **ý nghĩa** dù khác từ (run D). | Xong; test NDCG@10 1,16% (mẫu 50k); không dùng trong fusion |
| ③ Fusion | Trộn các nhánh thành 50 ứng viên; kỳ vọng tốt hơn nhánh tốt nhất (run E). | Xong; test NDCG@10 5,60% (+14% so với ItemKNN); **hệ thống cuối** = fusion + gom phiên bản |
| Đánh giá | Đo khách quan, cùng một giao thức cho mọi run A–G. | Xong |
| Kho review | 9,25 triệu review làm nguồn bằng chứng ý kiến. | Xong |
| ④ Gói bằng chứng | Đưa cho LLM chỉ những gì có nguồn: [H#] lịch sử, [E#] metadata và review trước `as_of`. | Xong; đã chạy trên 6 user thật (ví dụ bên dưới), < 2 s cho 3 ứng viên |
| Xếp lại top 20 (LLM, cross-encoder, learning-to-rank) | Thử xem có cách nào xếp lại pool của fusion tốt hơn không (run F). | **Bỏ khỏi đề tài 18/10**: không cách nào vượt thứ tự fusion (kết quả âm, bảng bên dưới) |
| ⑤ LLM giải thích top 3 có trích dẫn, ⑥ validator, giám khảo | Lý do kèm [E#]/[H#] cho từng gợi ý; validator loại câu không kiểm được; LLM giám khảo đo tỉ lệ câu đúng nguồn. **Đóng góp chính.** | Xong code 18/10 (Gemini `gemini-3.5-flash-lite`, giám khảo `gemini-3.5-flash`); smoke test 10 record; chưa chạy 200 user |
| Demo end-to-end (`recsys-demo`) | Một lệnh: hồ sơ → fusion → bằng chứng → giải thích → validator. | Xong 18/10 |
| Run G 200 user, chấm tay, báo cáo, slide | Số liệu chính cho câu hỏi "giải thích bắt buộc trích dẫn có đáng tin hơn"; viết báo cáo. | Chưa làm |

Kết quả trên toàn bộ 657.203 query test:

| Phương pháp | NDCG@10 | HR@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity (A) | 0,91% | 1,71% | 4,5% |
| Dense — MiniLM-L6 (D), **mẫu 50k test** | 1,16% | 2,26% | 4,3% |
| BPR-MF, 64 factors, 60 epoch (B) | 1,31% | 2,27% | 4,5% |
| BM25 — 50 từ truy vấn (C) | 1,35% | 2,73% | 6,2% |
| ItemKNN, 400 hàng xóm, $\gamma = 0{,}7$ (B) | 4,91% | 7,68% | 11,9% |
| **Fusion theo điểm** — ItemKNN 47,6%, BM25 4,8%, popularity 47,6% (E) | **5,60%** | **8,99%** | **14,2%** |

Mọi dòng là toàn bộ 657.203 query test, trừ Dense (mẫu cố định 50.000 query, seed 42; sai số NDCG@10 ~±0,1%). Nhánh nội dung đứng riêng chỉ nhỉnh hơn Popularity; Dense còn có Recall@50 thấp hơn Popularity (4,3% so với 4,5%).

Fusion so với ItemKNN trên toàn bộ test: NDCG@10 +14%, HR@10 +17%, Recall@50 +19% (tương đối) — khớp mức tăng trên validation (+15% NDCG, +19% Recall@50), nên cấu hình tune trên validation đứng vững. Đánh đổi: độ phủ catalog trong top 50 giảm từ 99,8% (ItemKNN) xuống 92,6% vì prior popularity kéo gợi ý về phim phổ biến.

**Gom phiên bản (17/10, số chạy lại 18/10 sau khi sửa gom Twilight)** — mẫu 50.000 query test (seed 42), model pha test; "đúng mã" = target đúng `item_id` (metric gốc), "cùng phim" = bất kỳ phiên bản nào của phim target (bước 10):

| Phương pháp | NDCG@10 đúng mã | HR@10 đúng mã | Recall@50 đúng mã | NDCG@10 cùng phim | HR@10 cùng phim | Recall@50 cùng phim |
|---|---:|---:|---:|---:|---:|---:|
| ItemKNN | 4,91% | 7,72% | 11,9% | 5,51% | 8,72% | 13,6% |
| Fusion (E) | 5,63% | 9,02% | 14,2% | 6,33% | 10,18% | 16,2% |
| **Fusion + gom phiên bản (hệ thống cuối)** | 5,57% | 8,81% | 13,6% | **6,46%** | **10,40%** | **16,6%** |

- **Metric gốc đánh giá thấp mọi phương pháp ~12%** (fusion 5,63% → 6,33% khi tính cùng phim): gợi ý bản DVD khi target là bản Blu-ray vẫn là gợi ý đúng. Từ nay báo cả hai; metric cùng phim là chính cho hệ thống cuối.
- Gom phiên bản trong fusion: **+2,0% NDCG@10, +2,2% HR@10, +2,2% Recall@50** (tương đối, metric cùng phim) — chỗ của các bản trùng được nhường cho phim khác. Target rating ≥ 4: 7,20% → 7,35%. Theo metric đúng mã thì giảm nhẹ (5,63% → 5,57%) vì chỉ giữ một bản mỗi phim, không nhất thiết đúng bản target — hệ quả mong muốn, không phải lỗi.
- Validation (mẫu 10k, cùng phim): fusion 7,64% / 11,91% / 17,9% → có gom 7,69% / 12,04% / 18,2% — cùng chiều với test.
- Giá phải trả: loại mọi phiên bản của phim đã xem làm mất các target "mua lại" (target là bản khác của phim đã xem) — 0,12% query test.
- Độ phủ catalog (mẫu 50k): 69,2% → 69,4%.
- Lần đầu (ngưỡng 2% cho mọi khóa) cho 6,43% / 10,38% / 16,6%; sửa 18/10 (cùng title cho lệch 5%, chuẩn hóa "X, The:", "Part One", năm) chỉ đổi kết quả ở chữ số thứ hai nhưng bỏ được lỗi gợi ý lại phim đã xem thấy trong smoke test.

**Phân tích theo nhóm (18/10)** — mẫu 50k test, metric cùng phim, NDCG@10 / Recall@20 / Recall@50 (`runs/analysis/segments_test50k.json`):

| Nhóm | Số query | ItemKNN | BM25 | Fusion + gom phiên bản |
|---|---:|---|---|---|
| Tất cả | 50.000 | 5,51% / 10,8% / 13,6% | 1,57% / 4,6% / 6,7% | 6,46% / 13,0% / 16,6% |
| Lịch sử 3–5 phim | 20.533 | 6,36% / 12,1% / 15,0% | 1,81% / 5,4% / 7,8% | 7,31% / 14,3% / 18,2% |
| Lịch sử 6–20 | 25.276 | 5,17% / 10,3% / 13,2% | 1,46% / 4,2% / 6,1% | 6,10% / 12,7% / 16,2% |
| Lịch sử > 20 | 4.191 | 3,44% / 6,7% / 8,5% | 1,06% / 3,1% / 4,6% | 4,49% / 8,7% / 10,7% |
| Target phổ biến (top 1%) | 13.150 | 9,56% / 20,6% / 26,6% | 1,44% / 5,4% / 8,7% | 11,20% / 25,2% / 34,2% |
| Target trung bình (9% tiếp) | 15.619 | 5,34% / 9,4% / 11,4% | 1,71% / 4,7% / 6,5% | 6,67% / 12,3% / 14,5% |
| Target hiếm (90% còn lại) | 21.231 | 3,13% / 5,7% / 7,1% | 1,55% / 4,0% / 5,5% | 3,37% / 6,0% / 7,2% |
| Target có title | 21.115 | 5,90% / 10,4% / 12,6% | 2,11% / 5,6% / 7,8% | 6,80% / 12,4% / 14,7% |
| Target không có title | 28.885 | 5,23% / 11,0% / 14,3% | 1,17% / 3,9% / 5,8% | 6,21% / 13,5% / 18,0% |

- Tập 5-core nên không có user 1–2 phim. User có lịch sử dài lại khó đoán hơn (gu đa dạng, target cuối cùng khó).
- Fusion giúp nhiều nhất với **phim phổ biến** (Recall@50 26,6% → 34,2%, do prior popularity), gần như không giúp **phim hiếm** (7,1% → 7,2%). Phim hiếm là chỗ BM25 tương đối mạnh nhất (gần một nửa ItemKNN, so với ~1/7 ở phim phổ biến) nhưng vẫn thua ItemKNN → không có nhóm nào nhánh nội dung thắng nhánh CF.
- BM25 tốt gấp đôi khi target có title (2,11% so với 1,17%): `item_doc` của phim không title chỉ có tên người.
- **Recall@20 = 13,0%**: chỉ khoảng 1/8 user có phim đúng trong 20 ứng viên đầu — trần của mọi bước xếp lại sau fusion.

**Xếp lại pool top 20 — kết quả âm, bỏ khỏi đề tài (18/10).** Smoke test trên validation, 50 user có target trong pool (50 user đầu của mẫu 5.000 query, seed 42), metric cùng phim; mã đã xóa khỏi repo, số liệu ở `runs/llm/dev-inpool-50/`, `runs/rerank/`:

| Cách xếp lại | NDCG@10 | HR@10 | P(tốt hơn fusion)* | Chi phí |
|---|---:|---:|---:|---|
| **Giữ thứ tự fusion** | **53,7%** | **86%** | — | 0 |
| Learning-to-rank (HistGradientBoosting, 11 đặc trưng, train 29.747 user validation khác) | 51,9% | 80% | 16% | vài phút CPU |
| LLM `gemini-3.5-flash-lite`, có thẻ bằng chứng | 48,7% | 76% | 15% | ~2.500 token/user |
| LLM, chỉ title | 44,6% | 72% | 2% | ~1.500 token/user |
| Cross-encoder `BAAI/bge-reranker-base` | 31,3% | 58% | 0% | 15,8 s/user (CPU) |

\* bootstrap 2.000 lần trên 50 user. Learning-to-rank trên 690 user có target trong pool: 54,2% so với 53,7% của fusion; trên cả 5.000 user 7,47% so với 7,41% — ngang nhau. Permutation importance: bỏ "hạng fusion" làm NDCG giảm 13,7 điểm, mọi đặc trưng khác < 1 điểm → thứ tự fusion đã gần tối ưu với các tín hiệu có sẵn. Cross-encoder và LLM chỉ đọc nội dung nên không thấy tín hiệu "người xem phim A xem tiếp phim B". Điểm đáng ghi: thẻ bằng chứng giúp LLM xếp tốt hơn (48,7% so với 44,6%). LLM rerank bị fallback 1/50 lần (lặp mã C4), dù đã ép JSON schema.

**Smoke test 10 record, hệ thống cuối (18/10)** — chi tiết từng bước ở [docs/smoke_test_10_record.md](docs/smoke_test_10_record.md); 5 user test ngẫu nhiên (1/5 trúng) + 5 user có target trong top 10:

| Giải thích (30 phim) | Số câu | Phim có giải thích | supported | unsupported |
|---|---:|---:|---:|---:|
| Tự do (cùng thông tin, không mã nguồn) | 60 | 30 | 72% | 17% |
| Bắt buộc trích dẫn, sau validator | 33 | 27 | 88% | 3% |

- Bắt buộc trích dẫn giảm câu sai rõ rệt, đổi lại nói ít hơn và từ chối 3/30 phim (`evidence_unavailable`). Câu sai của bản tự do chủ yếu là gán sở thích không có trong lịch sử.
- Validator: kiểm ID/phim/tên không bắt lỗi nào lần này; lọc độ giống (0,3) loại nhầm 1 câu đúng và giữ 1 câu sai → chưa hữu ích, cần hiệu chỉnh hoặc bỏ.
- Vấn đề định tính: câu đúng nguồn nhưng không phải lý do (review kể đọc sách Tolkien hồi nhỏ); trích ý khen nhỏ trong review chê; dẫn thừa nguồn (2,1 nguồn/câu).
- Chi phí hệ thống cuối: 1 lời gọi LLM/user, ~1.600 token vào + ~300 ra, p50 1,6 s.
- Smoke test tìm ra lỗi gom phiên bản (fan Twilight được gợi ý lại *Eclipse* và *Breaking Dawn Part 1* đã xem) → đã sửa `editions.py`.

**Các smoke test LLM trước đó (validation, 18/10)** tìm và sửa: giám khảo `gemini-3.5-flash` dùng hết ngân sách output cho "thinking" → JSON bị cắt (nay thinking low, 4.096 token, không cache câu trả lời bị cắt); model trả mảng thay vì object (nay ép `responseSchema`); model chèn `[E#]` vào câu (nay tách ra); validator bắt nhầm tên chính phim đang giải thích (nay cho phép).

**Ví dụ gợi ý + bằng chứng (hệ thống cuối, 17/10)** — 5 query test cố định (`Random(42).sample(…, 5)`) và user Harry Potter; đầy đủ ở `runs/examples/recommend_5_users_editions.{txt,json}` và `runs/examples/evidence_example.json`:

| User | Lịch sử (gần nhất) | Target (rating) | Top 3 hệ thống cuối | Kết quả |
|---|---|---|---|---|
| `AGXV…Q7Q` | Harry Potter 2 (xem 18 giây trước target), Mr. Church… | Harry Potter 3 (5★) | Harry Potter 3, Harry Potter 4, Harry Potter 1 | **Trúng hạng 1** |
| `AHKI…GJN3Q` | Heavy Metal, Akira, Ice Age, A Bug's Life (33 phim) | Three Amigos (5★) | Krull, Fire and Ice, Ice Age: Collision Course | Trượt; top 10 đúng gu anime/fantasy (Ghost in the Shell, Paprika, Vampire Hunter D), target là phim hài |
| `AFUC…D7P4Q` | Groundhog Day, Best Exotic Marigold Hotel, Elf | The Peanut Butter Falcon (3★) | It Happened on 5th Avenue, Holiday Inn, The Hundred-Foot Journey | Trượt; đúng gu phim nhẹ nhàng, Giáng sinh |
| `AFAG…TBWQ` | The Walking Dead, Frozen, Pixels (4 phim) | Một mùa The Walking Dead (5★), bản Prime | Wings, Ant-Man, **cùng mùa The Walking Dead (bản khác)** | Trượt theo đúng mã, **trúng hạng 3 theo cùng phim** |
| `AFM7…Z6A` | Forrest Gump, Sling Blade, O Brother Where Art Thou? | Battleship (3★) | Cast Away, Fargo, National Lampoon's Christmas Vacation | Trượt; gợi ý hợp lý |
| `AEOU…LHA` | Thích Cast Away (4★), ghét 3 phim | The Jackal (**1★**) | The People vs. Larry Flynt, Detachment, Ikiru | Trượt; top 10 trước khi gom có **4 mục là Cast Away (3 bản) và bản trùng Polar Express**, nay đã hết |

Bằng chứng mẫu (user Harry Potter, top 3):

| ID | Nguồn (`source_id`) | Nội dung |
|---|---|---|
| H1 | `history:B002DQDV6Y` | 5★ "Harry Potter and the Chamber of Secrets (Widescreen Edition)" — title mượn từ bản DVD |
| E1 | `meta:B01KW24RCC:title` | "HP & the Prisoner of Azkaban SE" — title mượn cho bản Prime `B00271DNP4` |
| E2 | `meta:B00271DNP4:people` | "Directors: Alfonso Cuarón. Cast: Daniel Radcliffe, Rupert Grint, Emma Watson" |
| E3 | `review:5340230` | Review của một phụ huynh: con xem 3 lần trong 2 ngày, "the best of the Potter movies she'd seen yet" |
| E5 | `meta:B000E6EK38:title` | "Harry Potter and the Goblet of Fire" (ứng viên 2) |
| E9 | `meta:B075W1L85B:title` | "Harry Potter and the Sorcerer's Stone" (ứng viên 3) |

Sửa được nhờ chạy ví dụ (so với lần chạy đầu): review nói về sách ("this book was full of surprises") bị loại; "Directors: Greg Nicotero, Michael E. Cast:…" bị cắt tên → đủ tên; dòng "Directors: Various. Cast: Various" của Fargo bị loại; `B002DQLGHU` mượn nhầm title "Harry Potter: 8-Film Collection" → chặn gộp bộ sưu tập, nay đúng "Sorcerer's Stone"; [H#] của phim Prime nay có title.

Vấn đề bằng chứng còn lại: review nói về dịch vụ chứ không về phim vẫn lọt ("Used a 24 hour rental… No issues with play", "will not purchase… until CC is included"); hiếm hơn là review xếp nhầm phim (một review về video sửa mặt bếp nằm ở The Walking Dead); 57.036 item vẫn không có title (không phiên bản nào có title) — bằng chứng chỉ có tên người.

Kết quả trên 20.000 query validation (seed 42), model pha validation:

| Nhánh (run) | NDCG@10 | HR@10 | Recall@50 | Thời gian/user |
|---|---:|---:|---:|---:|
| Popularity (A) | 1,11% | 2,14% | 5,2% | 0,3 ms |
| Dense — MiniLM-L6, 64 token (D) | 1,40% | 2,86% | 5,4% | 8,4 ms |
| BM25 — 50 từ truy vấn (C) | 1,65% | 3,44% | 7,6% | 3,4 ms |
| ItemKNN (B) | 5,86% | 8,95% | 13,4% | ~3 ms |
| Fusion RRF, tốt nhất trong 64 tổ hợp | 5,86% | 8,95% | 13,4% | — |
| **Fusion theo điểm (E)**: ItemKNN 47,6%, BM25 4,8%, popularity 47,6% — mẫu 10k | **6,96%** | — | **16,0%** | — |

Đứng riêng, hai nhánh nội dung chỉ nhỉnh hơn Popularity; ItemKNN vượt trội vì "phim xem tiếp" phụ thuộc mạnh vào hành vi đám đông và thứ tự thời gian. Giá trị của BM25/Dense nằm ở chỗ **bổ sung** phim mà ItemKNN bỏ sót, đo bằng fusion.

**Fusion RRF (kết quả âm):** grid 64 tổ hợp trọng số (bm25, dense, popularity ∈ {0; 0,25; 0,5; 1}, itemknn = 1) — tổ hợp tốt nhất là ItemKNN đứng riêng; thêm nhánh nào cũng làm NDCG@10 giảm (bm25 0,25 → 5,74%; dense 0,25 → 5,66%; popularity 0,25 → 5,44%) và Recall@50 gần như không đổi. Hai lý do: RRF chỉ dùng hạng nên bỏ mất độ tự tin của ItemKNN; mỗi nhánh chỉ góp top 50 nên phim mới hầu như không vào được top 50 (hạng 1 của BM25 ở trọng số 0,25 được 0,25/61 < 1/110 của hạng 50 ItemKNN). Ngược lại, cộng prior popularity vào **điểm** ItemKNN trên toàn bộ ứng viên từng tăng NDCG 18% → thử fusion theo điểm, danh sách sâu hơn.

**Fusion theo điểm (cấu hình chốt):** điểm mỗi nhánh chia cho điểm cao nhất của nhánh đó với user, cộng $w_{pop} \cdot (1 - \text{hạng popularity}/N)$, mỗi nhánh góp 200 ứng viên. Grid 45 tổ hợp trên 10k query validation: ItemKNN đứng riêng 6,03% / 13,5% → + popularity 1,0: 6,93% / 16,1% → + BM25 0,1: **6,96% / 16,0%** (chọn). Gần như toàn bộ mức tăng (+15% NDCG, +19% Recall@50) đến từ prior popularity; BM25/Dense chỉ thêm ~0,03%, nhỏ hơn sai số mẫu. Kết luận cho báo cáo: với bài toán "phim xem tiếp", nhánh nội dung hầu như không tăng độ chính xác; giá trị của chúng là bằng chứng và giải thích.

**Embedding xếp lại trong pool (kết quả âm, 17/10):** thay vì tìm trên toàn catalog bằng vector trung bình, cộng $w \cdot \cos(\text{phim vừa thích}, j)$ (hoặc max có trọng số trên 3 phim thích gần nhất) vào điểm fusion của các ứng viên. Cùng mẫu 10k validation: mốc 6,96% / 16,04%; tốt nhất $w = 0{,}25$ (phim vừa thích) 6,99% / 15,94% — tăng 0,03%, nhỏ hơn sai số mẫu (~±0,2%), Recall@50 giảm; $w \ge 0{,}5$ giảm rõ. Không đưa vào fusion. Lý do: ItemKNN đã nắm tín hiệu "phim vừa xem"; `item_doc` của nhiều phim chỉ có tên người. Ví dụ user mẫu: embedding trung bình đặt Harry Potter 3 ở hạng 5.298, dùng riêng phim vừa xem → hạng 37, trong khi ItemKNN xếp hạng 1. → Embedding dùng cho **bằng chứng và kiểm tra lời giải thích** ở bước LLM.

BM25: chỉ giữ 50 từ mạnh nhất (trọng số × idf) trong truy vấn thay vì toàn bộ ~380 từ: nhanh hơn 5 lần (25 → 4,9 ms/user), NDCG@10 1,56% → 1,65%, Recall@50 8,0% → 7,6%.

**Quy ước đánh giá trên test (quyết định 17/10):**

- Các run nằm trong hệ thống cuối hoặc là baseline chính (Popularity, BPR, ItemKNN, BM25, Fusion) được đánh giá trên **toàn bộ** 657.203 query test.
- Dense **không** nằm trong fusion cuối (trọng số 0 trên validation), nên chỉ được đánh giá trên **mẫu cố định 50.000 query test** (`--query-limit 50000`, `Random(42)` như mọi mẫu khác), kết quả ở `runs/dense-movies-tv-2023/test_evaluation_50k.json`. Lý do: đánh giá toàn bộ mất ~90 phút trên máy này (mỗi user chấm cả 198k phim), trong khi mẫu 50k mất ~7 phút và sai số NDCG@10 chỉ ~±0,1% — đủ để báo cáo một nhánh yếu. Trong bảng kết quả phải ghi rõ "mẫu 50k" cạnh dòng Dense; không so sánh trực tiếp nó với các dòng toàn bộ test ở mức chênh lệch nhỏ hơn ~0,2%.
- Thử nghiệm và tune luôn làm trên **validation** (mẫu 10k/20k); test chỉ chạy với cấu hình đã chốt.

Test thấp hơn validation khoảng 13–16% với **mọi** phương pháp (target test là tương tác cuối cùng, khó hơn), nên không phải dấu hiệu tune quá khớp.

Phát hiện khác (validation):

- Trọng số độ gần đây được giữ ($\gamma = 0{,}7$). Ablation không dùng nó ($\gamma = 1$) làm NDCG@10 của ItemKNN giảm khoảng 17%.
- BPR yếu vì bỏ qua thứ tự thời gian, chưa hội tụ (AUC còn tăng ở epoch 60) và dữ liệu thưa.
- Embedding chạy CPU chậm: bge-small (~29 tài liệu/s) được thay bằng MiniLM-L6 với 64 token (~51 tài liệu/s); ghi là hạn chế trong báo cáo.

## Việc tiếp theo (cập nhật 18/10)

Đã quyết: **bỏ bước xếp lại** (giữ thứ tự fusion); LLM chỉ làm giải thích có trích dẫn; dùng Gemini, mức tiết kiệm (`gemini-3.5-flash-lite` để sinh, `gemini-3.5-flash` làm giám khảo), nâng model sau khi hoàn thiện.

1. **Hiệu chỉnh trên validation** (~50 user, ~0,5 triệu token): chốt ngưỡng độ giống của validator (hoặc bỏ bước này) và chỉnh prompt giải thích — chỉ nêu lý do liên quan sở thích của user, không trích ý trái với ý chính của review, mỗi câu dẫn tối đa 2–3 nguồn. Khóa prompt trước khi chạy test.
2. **Run G trên 200 user test ngẫu nhiên** (~2 triệu token): tự do vs. bắt buộc trích dẫn trước/sau validator; báo supported/unsupported, độ phủ, token, độ trễ.
3. **Chấm tay 30–50 câu** (`manual_check.csv`) để đo độ tin của giám khảo — cần người làm.
4. (Tùy chọn) Chạy lại phần giải thích bằng model mạnh hơn (`--model gemini-3.5-flash`) trên cùng mẫu để so sánh chi phí/chất lượng.
5. Cập nhật `docs/overview.md`, `docs/architecture.md` theo thiết kế cuối (không rerank); viết báo cáo và slide (khung ở [docs/overview.md](docs/overview.md) mục 7): bảng A–E, run G, các kết quả âm (RRF, embedding xếp lại, xếp lại pool), phát hiện về phiên bản, phân tích theo nhóm, ví dụ từ smoke test.

Vấn đề mở không chặn tiến độ: `preprocess.py` và các file mà [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md) nhắc tới không có trong repo; không dùng Dense-Works (giữ 5-core, gom phiên bản thay cho "Works").

## Quyết định đang chờ

- Đồng ý chạy bước 1–2 ở trên (~2,5 triệu token Gemini).
- Mẫu báo cáo / slide của trường (nếu có) và hạn nộp.

## Nhật ký phiên làm việc (13–18/10/2026)

- Kiểm tra nhánh `f/quan`; chạy thử luồng profile → CF trên dữ liệu thật, không rò rỉ.
- So sánh BPR với baseline đơn giản → BPR yếu; cài ItemKNN, tune trên validation, đánh giá full test.
- Phát hiện và sửa lỗi số liệu: bảng ItemKNN đầu tiên bị lẫn prior popularity do lỗi phá hòa trong script thử; số đúng là 4,91% NDCG@10.
- Sắp xếp lại tài liệu theo module, thêm `README.md`, `CLAUDE.md`, `docs/overview.md`, xóa tài liệu nháp/lỗi thời.
- **Đơn giản hóa toàn bộ backend** (~3.700 → ~950 dòng); chạy lại trên dữ liệu thật, ItemKNN và Popularity khớp kết quả cũ; BPR train local.
- 17/10: tăng tốc profile (~6 s → ~0,3 s); thêm `genres`, `item_doc`; cài BM25, Dense, Fusion, kho review (9,25 triệu review), gói bằng chứng; chạy pha validation. Thử embedding xếp lại trong pool: không cải thiện (+0,03%). Đánh giá Dense trên test chuyển sang mẫu 50k (đánh giá toàn bộ ~90 phút, không ảnh hưởng quyết định nào). Kết quả test: Fusion 5,60% > ItemKNN 4,91% > BM25 1,35% > BPR 1,31% > Dense 1,16% (50k) > Popularity 0,91% NDCG@10; fusion +14% NDCG, +19% Recall@50 so với ItemKNN. Embedding bge-small quá chậm trên CPU này → đổi sang MiniLM-L6 (66 phút cho cả catalog). BM25 giới hạn 50 từ truy vấn (nhanh hơn 5 lần). Fusion RRF không vượt ItemKNN → thêm fusion theo điểm + prior popularity (+15% NDCG). Kết quả validation: ItemKNN 5,86% ≫ BM25 1,65% > Dense 1,40% > Popularity 1,11% NDCG@10.
- 17/10 (tiếp): chạy gợi ý cho 5 query test → phát hiện một phim có nhiều phiên bản làm hệ thống gợi ý lại phim đã xem (Cast Away 3 bản) và làm metric tính trượt khi gợi ý đúng phim khác bản. Thêm `backend/data/editions.py` (gom theo title/người + số lượt rating lệch ≤ 2%, chặn gộp dây chuyền, "Various", bộ sưu tập), `fuse --editions`, `evaluate --editions` (metric cùng phim), mượn title cho phim Prime trong bằng chứng; lọc review nói về sách, sửa cắt tên viết tắt, bỏ "Various". Kết quả mẫu 50k test (cùng phim): fusion 6,31% → 6,43% NDCG@10, 16,2% → 16,6% Recall@50; metric đúng mã đánh giá thấp ~12%. Phát hiện user ví dụ xem Harry Potter **2** (không phải 1) trước target. 49 test.
- 18/10: phân tích theo nhóm (`backend/cf/segments.py`). Cài phần LLM với Gemini (`backend/generation/`: client có cache và JSON schema, giải thích tự do / bắt buộc trích dẫn, validator, giám khảo, thí nghiệm) và demo `backend/service.py`; API key ở `.env` (git-ignore). Smoke test validation tìm và sửa 4 lỗi (thinking làm cắt JSON, sai schema, mã trong câu, validator bắt nhầm tên phim). Thử 3 cách xếp lại pool top 20 (LLM, learning-to-rank, cross-encoder) — không cách nào vượt thứ tự fusion → **bỏ bước xếp lại khỏi đề tài**, xóa mã, giữ số liệu. Smoke test 10 record trên test với báo cáo `docs/smoke_test_10_record.md`: bắt buộc trích dẫn 88% câu đúng nguồn so với 72% khi tự do; tìm ra và sửa lỗi gom phiên bản (Twilight). Chạy lại đánh giá 50k với `editions.parquet` mới: hệ thống cuối 6,46% NDCG@10 cùng phim. 66 test.
