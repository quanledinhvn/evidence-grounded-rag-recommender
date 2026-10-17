# CLAUDE.md

Hệ gợi ý phim có giải thích dựa trên bằng chứng (CF + BM25 + embedding → fusion → LLM rerank + giải thích bắt buộc trích dẫn) trên Amazon Reviews 2023 – Movies & TV. Đồ án học phần Hệ gợi ý, Thạc sĩ KHMT, HUST. Tổng quan: [README.md](README.md), [docs/overview.md](docs/overview.md); đặc tả: [docs/architecture.md](docs/architecture.md).

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
| `backend/profile/profile.py` | `load_history`, `load_catalog`, `build_profile` |
| `backend/cf/common.py` | `FitData`, `load_fit_data`, `Popularity`, `top_k`, `rank_items` |
| `backend/cf/itemknn.py`, `backend/cf/bpr.py` | Hai model CF |
| `backend/retrieval/bm25.py`, `backend/retrieval/dense.py` | Hai nhánh nội dung trên `item_doc` |
| `backend/retrieval/fusion.py` | Trộn các nhánh (RRF có trọng số), tune trọng số |
| `backend/cf/evaluate.py` | NDCG@10, HR@10, Recall@50 cho mọi model |
| `backend/cf/cli.py` | `recsys-cf train / evaluate / recommend / fuse / tune-fusion` cho mọi model |
| `backend/evidence/evidence.py`, `backend/evidence/cli.py` | Gói bằng chứng [H#]/[E#] cho một query (`recsys-evidence`) |
| `tests/` | 36 test, dữ liệu giả rất nhỏ |

Mọi model (ItemKNN, BPR, BM25, Dense, Fusion, Popularity) có cùng giao diện `recommend(user_id, limit, exclude)` và `recommend_batch(user_ids, limit)`, trả `{"item_id", "score", "source"}`.

## Dữ liệu và artifact

| Đường dẫn | Nội dung |
|---|---|
| `artifacts/movies-tv-2023/splits.parquet` | 7.441.129 tương tác (train 6.126.723, validation 657.203, test 657.203), sắp theo user |
| `artifacts/movies-tv-2023/queries_{val,test}.parquet` | 657.203 query mỗi pha, mỗi user một target, `as_of` = timestamp target |
| `artifacts/movies-tv-2023/catalog_enriched.parquet` | 197.943 item; `item_doc` phủ 195.993; 82.840 item Prime Video không có title |
| `artifacts/movies-tv-2023/reviews.sqlite` | 9.249.801 review dùng làm bằng chứng (4,7 GB) |
| `artifacts/movies-tv-2023/item_vectors.npy` | Vector embedding của mọi item (MiniLM-L6, 384 chiều), dùng chung cho mọi pha |
| `runs/validation/{itemknn,bm25,dense}/` | Model pha validation (fit trên train) và kết quả trên 20k query validation; `fusion-grid.json` |
| `runs/{itemknn,bpr}-movies-tv-2023/test/` | Model pha test |
| `runs/{itemknn,bpr,popularity}-movies-tv-2023/test_evaluation.json` | Kết quả trên toàn bộ test |
| `runs/examples/evidence_example.json` | Gói bằng chứng mẫu cho user ví dụ |
| `runs/cf-movies-tv-2023-full/20261009T135249Z/` | Lượt BPR cũ trên AWS; chỉ còn giá trị tham khảo |
| `dataset/Movies_and_TV.jsonl`, `dataset/meta_Movies_and_TV.jsonl` | Review thô (17,3 triệu) và metadata thô |

Đặc điểm cần nhớ: trung vị 5 tương tác train mỗi user; split không có review text; một phim có nhiều `parent_asin` (DVD, Blu-ray, Prime Video).

## Bất biến

- Mọi dữ liệu dùng cho một query phải có `timestamp < as_of` (lịch sử, review làm bằng chứng).
- Pha `validation` chỉ fit trên `train`; pha `test` fit trên `train` + `validation`, không đọc `test`. Chọn cấu hình trên validation, test chỉ dùng để báo cáo.
- Bằng chứng review không bao giờ lấy review của chính user đang được gợi ý.
- Đánh giá luôn lấy 50 item mỗi query; `recommend` giới hạn `limit ≤ 50`.
- Item được đánh số theo `item_id` tăng dần; điểm bằng nhau thì phá hòa theo `item_id`.

## Hướng dẫn chạy test các phần đã implement

Ví dụ xuyên suốt là một user thật: `AGXVBIUFLFGMVLATYXHJYL4A5Q7Q`. Lịch sử của user này có 7 phim ở train (2006–2017), một phim ở validation là **Harry Potter phần 1** (`B002DQDV6Y`, xem lúc `1530556502815`), và target test là **Harry Potter phần 3** (`B00271DNP4`, `as_of = 1530556520803`, tức 18 giây sau).

Trong các lệnh dưới, `python` là `& "$env:LOCALAPPDATA\Temp\fq\.venv\Scripts\python.exe"` (hoặc `uv run python`), `$a = "artifacts/movies-tv-2023"`, chạy từ gốc repo.

### 0. Unit test (không cần dữ liệu thật, ~15 giây)

```powershell
python -m pytest -q -p no:cacheprovider                         # toàn bộ 36 test
python -m pytest -q -p no:cacheprovider tests/test_retrieval.py  # một file
```

| File | Kiểm tra gì |
|---|---|
| `tests/test_data.py` | Import giữ nguyên split, `review_id`, query, catalog; từ chối dữ liệu hỏng; `reorganize` sắp theo user; enrich làm sạch HTML, `genres` bỏ nhãn không phải thể loại, `item_doc` |
| `tests/test_profile.py` | Pha validation không thấy split validation; tương tác tại `as_of` bị loại; thích/ghét; tương tác mới nhất thắng; công thức trọng số |
| `tests/test_cf.py` | Pha fit đúng split; Popularity; cosine và cắt hàng xóm; ItemKNN ưu tiên phim vừa xem, `because`; BPR không gợi ý phim đã xem; metric; CLI |
| `tests/test_retrieval.py` | Lịch sử "đã thích" có trọng số; BM25 tìm phim chung đạo diễn/diễn viên; Dense xếp theo cosine; công thức RRF; Fusion lưu/nạp lại |
| `tests/test_reviews.py` | Chỉ giữ review hữu ích của item trong catalog; lấy review trước `as_of`, loại review của chính user |
| `tests/test_evidence.py` | [H#] dùng review của chính user; [E#] gồm metadata và review của người khác trước `as_of` |

Kết quả mong đợi: `36 passed`.

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
- **Ví dụ:** 8 phim đã xem (7 train + Harry Potter 1 ở validation; Harry Potter 3 **không** có vì xảy ra đúng lúc `as_of`). `H1 = B002DQDV6Y`, weight 0,99999 (xem 18 giây trước); `H2 = B01LWY8995`, weight 0,0186 (xem 517 ngày trước). Mất ~0,3 giây.

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
- **Ví dụ:** ItemKNN xếp hạng 1 **`B00271DNP4` (Harry Potter 3) — đúng target test**, `because = [{item_id: B002DQDV6Y, similarity: 0,341, weight: 1,0}]`: user vừa xem Harry Potter 1, và nhiều người xem phần 1 cũng xem phần 3. Mất ~15 giây (chủ yếu nạp model).

### 6. Đánh giá — `recsys-cf evaluate`

```powershell
python -m backend.cf.cli evaluate --model-dir runs/itemknn-movies-tv-2023/test `
  --queries $a/queries_test.parquet --output runs/itemknn-movies-tv-2023/test_evaluation.json
# --popularity: baseline Popularity; --query-limit 20000: mẫu cố định (seed 42) để chạy nhanh
```

- **Input:** model, file query (`user_id`, `target_item_id`, `target_rating`).
- **Các bước và lý do:** với mỗi query lấy 50 item, tìm hạng của target; HR@10 = có trong top 10; NDCG@10 = $1/\log_2(rank+1)$ (đúng ở hạng cao được thưởng nhiều hơn); Recall@50 = có trong 50 ứng viên (cái mà các bước sau còn có thể xếp lại). Báo thêm riêng cho target rating ≥ 4 (`positive_target`).
- **Output:** JSON gồm `ndcg_at_10`, `hr_at_10`, `recall_at_50`, `fallback_rate`, `item_coverage`, `mean_latency_ms`, `positive_target`.
- **Thời gian trên toàn bộ test:** ItemKNN ~10 phút, Popularity ~2 phút, BM25/Dense/BPR lâu hơn nhiều (BPR ~80 phút) vì phải chấm hàng trăm nghìn phim cho mỗi user; dùng `--query-limit 20000` khi chỉ cần số nhanh.

### 7. Fusion — `recsys-cf tune-fusion`, `recsys-cf fuse`

```powershell
# Tìm trọng số trên VALIDATION (model pha validation)
python -m backend.cf.cli tune-fusion --member itemknn=runs/validation/itemknn --member bm25=runs/validation/bm25 `
  --member dense=runs/validation/dense --grid itemknn=1 --grid bm25=0,0.25,0.5,1 --grid dense=0,0.25,0.5,1 `
  --grid popularity=0,0.25,0.5,1 --queries $a/queries_val.parquet --output runs/validation/fusion-grid.json
# Ghép model pha TEST với trọng số đã chọn
python -m backend.cf.cli fuse --member itemknn=runs/itemknn-movies-tv-2023/test --member bm25=... --member dense=... `
  --weight itemknn=1 --weight bm25=<w> --weight dense=<w> --weight popularity=<w> --output-dir runs/fusion-movies-tv-2023/test
```

- **Các bước và lý do:** mỗi nhánh trả top 50; phim ở hạng $r$ của nhánh $m$ được $w_m / (60 + r)$; cộng lại rồi xếp. Dùng **hạng** thay vì điểm vì mỗi nhánh chấm theo thang khác nhau (cosine, BM25, số lượt). Popularity là một nhánh riêng; phim mà một nhánh chỉ bổ sung bằng popularity không được tính cho nhánh đó. `tune-fusion` tính danh sách của từng nhánh một lần rồi thử mọi tổ hợp trọng số trên cùng 20k query validation.
- **Output:** `fusion-grid.json` (mọi tổ hợp, tốt nhất trước); thư mục fusion chỉ chứa `config.json` (nhánh nào, trọng số bao nhiêu).

### 8. Kho review — `recsys-data import-reviews`

```powershell
python -m backend.data.cli import-reviews --reviews dataset/Movies_and_TV.jsonl --catalog $a/catalog.parquet --output $a/reviews.sqlite
```

- **Input:** 17,3 triệu review thô.
- **Các bước và lý do:** chỉ giữ review của phim trong catalog, ≥ 10 từ (câu như "Five stars" không có thông tin), không nói về giao hàng/đĩa/giá (không phải ý kiến về phim); ghi vào SQLite với index `(item_id, timestamp)` để lấy "review của phim X viết trước thời điểm T" và `(user_id, item_id)` để lấy review của chính user.
- **Output:** `reviews.sqlite`: giữ 9.249.801; bỏ 1.535.240 ngoài catalog, 5.917.406 quá ngắn, 625.867 không nói về phim. Mất ~60 phút.

### 9. Gói bằng chứng — `recsys-evidence`

```powershell
python -m backend.evidence.cli --model-dir runs/itemknn-movies-tv-2023/test --splits $a/splits.parquet `
  --catalog $a/catalog_enriched.parquet --reviews $a/reviews.sqlite `
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --as-of 1530556520803 --phase test --candidates 3
```

- **Input:** model đề xuất ứng viên, user, `as_of`, kho review.
- **Các bước và lý do:**
  1. Dựng hồ sơ (bước 3) và lấy top ứng viên, loại mọi phim đã xem.
  2. [H#]: phim user thích/không thích, kèm review của chính user nếu có — để LLM nói "vì bạn đã thích X".
  3. [E#] cho mỗi ứng viên: title, thể loại, đạo diễn/diễn viên, mô tả, và tối đa 2 review của **người khác** viết **trước** `as_of`, ưu tiên review có nhiều từ trùng với phim user thích rồi review được vote "helpful" nhiều. Mỗi mục có `source_id` để validator kiểm tra.
- **Output:** JSON `{candidates, history: {H1: …}, evidence: {E1: …}, retrieval}`.
- **Ví dụ** (`runs/examples/evidence_example.json`): E1 = `meta:B00271DNP4:people` "Directors: Alfonso Cuarón. Cast: Daniel Radcliffe, Rupert Grint, Emma Watson"; E2, E3 = review của người khác về Harry Potter 3 viết trước 2018. Mất ~55 giây (truy vấn review của phim rất nhiều review).

---

# Tiến độ dự án

Các mục dưới đây thay đổi theo thời gian; phần trên là thông tin cố định của đồ án.

## Tiến độ (cập nhật 2026-10-17)

| Hạng mục | Tác dụng trong đồ án | Trạng thái |
|---|---|---|
| Nhập 3 split | Biến CSV gốc thành bảng chuẩn và **tách target đánh giá ra file query riêng**, để model không bao giờ thấy đáp án. Là nền cho mọi bước sau. | Xong; sắp theo user để đọc nhanh |
| Nối metadata, `genres`, `item_doc` | Gắn nội dung phim vào từng item và tạo một văn bản mô tả chung. Không có bước này thì không làm được BM25, embedding, bằng chứng [E#]. | Xong |
| ① Hồ sơ người dùng | Tóm tắt sở thích của user **tại thời điểm gợi ý**; đầu vào cho bằng chứng [H#]. | Xong, ~0,3 s/user |
| ② ItemKNN | Nhánh CF chính: "người xem A thường xem B", ưu tiên phim vừa xem; `because` cho sẵn bằng chứng. | Xong, đã tune và đánh giá |
| ② BPR-MF | Baseline học máy cổ điển (run B). | Xong |
| Popularity | Mốc thấp nhất (run A) và nhánh trong fusion. | Xong |
| ② BM25 | Phim chung **từ hiếm** (đạo diễn, diễn viên, thể loại) với phim user thích (run C). | Xong; validation NDCG@10 0,0165 |
| ② Embedding (Dense) | Phim giống về **ý nghĩa** dù khác từ (run D). | Xong; validation NDCG@10 0,0140 (vector đã tính, `item_vectors.npy`) |
| ③ Fusion | Trộn các nhánh thành 50 ứng viên; kỳ vọng tốt hơn nhánh tốt nhất (run E). | Xong; fusion theo điểm +15% NDCG trên validation; đang đánh giá trên test |
| Đánh giá | Đo khách quan, cùng một giao thức cho mọi run A–G. | Xong |
| Kho review | 9,25 triệu review làm nguồn bằng chứng ý kiến. | Xong |
| ④ Gói bằng chứng | Đưa cho LLM chỉ những gì có nguồn: [H#] lịch sử, [E#] metadata và review trước `as_of`. | Xong, đã chạy thử |
| ⑤ LLM rerank + giải thích, ⑥ validator | Xếp lại top 20 → top 10 và giải thích top 3 bắt buộc trích dẫn; validator loại câu không có nguồn. Đóng góp chính của đề tài. | Chưa làm — chờ chọn LLM |
| Service, API, run A–G, báo cáo | Ghép mọi bước thành một luồng, chạy bảng thí nghiệm và viết báo cáo. | Chưa làm |

Kết quả trên toàn bộ 657.203 query test:

| Phương pháp | NDCG@10 | HR@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity | 0,0091 | 0,0171 | 0,045 |
| BPR-MF (64 factors, 60 epoch) | 0,0131 | 0,0227 | 0,045 |
| ItemKNN (400 hàng xóm, $\gamma = 0{,}7$) | 0,0491 | 0,0768 | 0,119 |

Kết quả trên 20.000 query validation (seed 42), model pha validation:

| Nhánh (run) | NDCG@10 | HR@10 | Recall@50 | Thời gian/user |
|---|---:|---:|---:|---:|
| Popularity (A) | 0,0111 | 0,0214 | 0,052 | 0,3 ms |
| Dense — MiniLM-L6, 64 token (D) | 0,0140 | 0,0286 | 0,054 | 8,4 ms |
| BM25 — 50 từ truy vấn (C) | 0,0165 | 0,0344 | 0,076 | 3,4 ms |
| ItemKNN (B) | 0,0586 | 0,0895 | 0,134 | ~3 ms |
| Fusion RRF, tốt nhất trong 64 tổ hợp | 0,0586 | 0,0895 | 0,134 | — |
| **Fusion theo điểm (E)**: itemknn 1, bm25 0,1, popularity 1,0 — mẫu 10k | **0,0696** | — | **0,160** | — |

Đứng riêng, hai nhánh nội dung chỉ nhỉnh hơn Popularity; ItemKNN vượt trội vì "phim xem tiếp" phụ thuộc mạnh vào hành vi đám đông và thứ tự thời gian. Giá trị của BM25/Dense nằm ở chỗ **bổ sung** phim mà ItemKNN bỏ sót, đo bằng fusion.

**Fusion RRF (kết quả âm):** grid 64 tổ hợp trọng số (bm25, dense, popularity ∈ {0; 0,25; 0,5; 1}, itemknn = 1) — tổ hợp tốt nhất là ItemKNN đứng riêng; thêm nhánh nào cũng làm NDCG@10 giảm (bm25 0,25 → 0,0574; dense 0,25 → 0,0566; popularity 0,25 → 0,0544) và Recall@50 gần như không đổi. Hai lý do: RRF chỉ dùng hạng nên bỏ mất độ tự tin của ItemKNN; mỗi nhánh chỉ góp top 50 nên phim mới hầu như không vào được top 50 (hạng 1 của BM25 ở trọng số 0,25 được 0,25/61 < 1/110 của hạng 50 ItemKNN). Ngược lại, cộng prior popularity vào **điểm** ItemKNN trên toàn bộ ứng viên từng tăng NDCG 18% → thử fusion theo điểm, danh sách sâu hơn.

**Fusion theo điểm (cấu hình chốt):** điểm mỗi nhánh chia cho điểm cao nhất của nhánh đó với user, cộng $w_{pop} \cdot (1 - \text{hạng popularity}/N)$, mỗi nhánh góp 200 ứng viên. Grid 45 tổ hợp trên 10k query validation: ItemKNN đứng riêng 0,0603 / 0,135 → + popularity 1,0: 0,0693 / 0,161 → + BM25 0,1: **0,0696 / 0,160** (chọn). Gần như toàn bộ mức tăng (+15% NDCG, +19% Recall@50) đến từ prior popularity; BM25/Dense chỉ thêm ~0,0003, nhỏ hơn sai số mẫu. Kết luận cho báo cáo: với bài toán "phim xem tiếp", nhánh nội dung hầu như không tăng độ chính xác; giá trị của chúng là bằng chứng và giải thích.

BM25: chỉ giữ 50 từ mạnh nhất (trọng số × idf) trong truy vấn thay vì toàn bộ ~380 từ: nhanh hơn 5 lần (25 → 4,9 ms/user), NDCG@10 0,0156 → 0,0165, Recall@50 0,080 → 0,076.

Test thấp hơn validation khoảng 13–16% với **mọi** phương pháp (target test là tương tác cuối cùng, khó hơn), nên không phải dấu hiệu tune quá khớp.

Phát hiện khác (validation):

- Trọng số độ gần đây được giữ ($\gamma = 0{,}7$). Ablation không dùng nó ($\gamma = 1$) làm NDCG@10 của ItemKNN giảm khoảng 17%.
- BPR yếu vì bỏ qua thứ tự thời gian, chưa hội tụ (AUC còn tăng ở epoch 60) và dữ liệu thưa.
- Embedding chạy CPU chậm: bge-small (~29 tài liệu/s) được thay bằng MiniLM-L6 với 64 token (~51 tài liệu/s); ghi là hạn chế trong báo cáo.

## Việc tiếp theo

1. Đánh giá trên toàn bộ test: fusion (itemknn 1, bm25 0,1, popularity 1,0, method score, depth 200), BM25, Dense — đang chạy nền (run C, D, E).
2. Chọn LLM provider; cài LLM rerank top 20 → top 10 và giải thích top 3 bằng JSON có trích dẫn (Q5, Q6 trong [docs/architecture.md](docs/architecture.md)).
3. Validator: ID nguồn tồn tại và đúng phim, tên người/thể loại được nhắc có trong bằng chứng; fallback về thứ hạng fusion khi LLM lỗi.
4. Chạy run F, G trên mẫu user cố định; chấm citation và claim support; viết báo cáo.

Vấn đề mở không chặn tiến độ: `preprocess.py` và các file mà [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md) nhắc tới không có trong repo; cần hỏi người viết khảo sát nếu dùng Dense-Works.

## Quyết định đang chờ

- LLM provider, ngân sách token, hạn nộp (cần trước bước 2).
- Có dùng tập Dense-Works (gộp phiên bản, 20-core; xem [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md)) cho giai đoạn LLM không: với Recall@50 hiện tại, chỉ một phần nhỏ trong 200 user có target trong pool.

## Nhật ký phiên làm việc (13–17/10/2026)

- Kiểm tra nhánh `f/quan`; chạy thử luồng profile → CF trên dữ liệu thật, không rò rỉ.
- So sánh BPR với baseline đơn giản → BPR yếu; cài ItemKNN, tune trên validation, đánh giá full test.
- Phát hiện và sửa lỗi số liệu: bảng ItemKNN đầu tiên bị lẫn prior popularity do lỗi phá hòa trong script thử; số đúng là 0,0491 NDCG@10.
- Sắp xếp lại tài liệu theo module, thêm `README.md`, `CLAUDE.md`, `docs/overview.md`, xóa tài liệu nháp/lỗi thời.
- **Đơn giản hóa toàn bộ backend** (~3.700 → ~950 dòng); chạy lại trên dữ liệu thật, ItemKNN và Popularity khớp kết quả cũ; BPR train local.
- 17/10: tăng tốc profile (~6 s → ~0,3 s); thêm `genres`, `item_doc`; cài BM25, Dense, Fusion, kho review (9,25 triệu review), gói bằng chứng; chạy pha validation. Embedding bge-small quá chậm trên CPU này → đổi sang MiniLM-L6 (66 phút cho cả catalog). BM25 giới hạn 50 từ truy vấn (nhanh hơn 5 lần). Fusion RRF không vượt ItemKNN → thêm fusion theo điểm + prior popularity (+15% NDCG). Kết quả validation: ItemKNN 0,0586 ≫ BM25 0,0165 > Dense 0,0140 > Popularity 0,0111 NDCG@10.
