# Hướng dẫn chạy từng bước

Mỗi bước ghi rõ lệnh, input, output, các bước xử lý và lý do, kèm ví dụ trên dữ liệu thật. Cài đặt và cách chạy nhanh: [README.md](../README.md). Kết quả: [results/results.md](results/results.md).

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
| `runs/llm/smoke-10/` | Smoke test 10 record (test): gói bằng chứng, câu trả lời LLM, nhãn giám khảo, cache; báo cáo ở `docs/results/smoke_test_10_record.md` |
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

## Các bước

Ví dụ xuyên suốt là một user thật: `AGXVBIUFLFGMVLATYXHJYL4A5Q7Q`. Lịch sử của user này có 7 phim ở train (2006–2017), một phim ở validation là **Harry Potter phần 2 — Chamber of Secrets** (`B002DQDV6Y`, bản Prime Video không có title, xem lúc `1530556502815`; xác định nhờ số lượt rating 64.870 trùng các bản DVD/Blu-ray của phần 2), và target test là **Harry Potter phần 3** (`B00271DNP4`, `as_of = 1530556520803`, tức 18 giây sau).

Trong các lệnh dưới (PowerShell; bash dùng `\` thay cho dấu `` ` `` khi xuống dòng), `python` là Python của môi trường đã cài dự án (`uv run python` hoặc `.venv`, xem [README.md](../README.md)), `$a = "artifacts/movies-tv-2023"`, chạy từ gốc repo.

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
  1. Hồ sơ (bước 3) + **fusion có gom phiên bản, top 10 giữ nguyên thứ tự fusion** — không xếp lại: LLM, cross-encoder và learning-to-rank đều đã thử và không vượt thứ tự fusion (xem [results/results.md](results/results.md)).
  2. Gói bằng chứng cho top 10 (bước 9), title mượn cho phim không có title.
  3. **Một lời gọi LLM** giải thích top 3 (`backend/generation/explain.py`): prompt chỉ chứa mã C1–C3, [H#], [E#] và nội dung nguồn — **không gửi `user_id` hay mã Amazon**; nội dung review nằm trong `<data>` và model được dặn không làm theo lệnh trong đó. Đầu ra ép đúng JSON schema `{"items": [{"item_id": "C1", "claims": [{"text", "evidence_ids", "history_ids"}]}]}`, tối đa 2 câu/phim; mã [E#] model lỡ viết trong câu được tách ra và đưa vào danh sách nguồn.
  4. **Validator** (`backend/generation/validator.py`) kiểm từng câu: có nguồn; E# tồn tại và **thuộc đúng phim** đang giải thích; H# thuộc hồ sơ; tên người/tựa (2+ từ viết hoa) có trong nguồn được dẫn hoặc là tên chính phim đó; độ giống MiniLM giữa câu và nguồn ≥ ngưỡng (mặc định 0,3, chưa hiệu chỉnh). Câu lỗi bị bỏ; phim không còn câu nào → `evidence_unavailable`.
  5. Ghép câu hợp lệ kèm trích dẫn từ ID đã kiểm tra (renderer tự chèn `[E2][H1]`, không dùng chữ của model).
- **Output:** top 10 kèm `found_by` (nhánh fusion), top 3 kèm lời giải thích + nguồn được dẫn; `trace` có token, độ trễ. User không có phim nào đã chấm điểm → `cold_start`, không gọi LLM.
- **Ví dụ** (user Harry Potter): 1. *Harry Potter 3* (đúng target) — "You previously enjoyed the Harry Potter and the Chamber of Secrets movie, and this is another installment in the franchise [E1][E2][H1]. A reviewer mentions enjoying the well-crafted story and time travel elements [E4]." Một lời gọi ~2.300 token, ~20 s tổng (phần lớn nạp model).

### 12. Thí nghiệm giải thích (run G) và smoke test — `recsys-llm` (`backend/generation/experiment.py`)

```powershell
# Smoke test 10 record, kèm báo cáo chi tiết từng bước
python -m backend.generation.experiment --phase test --sample 5 --with-target 5 --output-dir runs/llm/smoke-10 --report docs/results/smoke_test_10_record.md
# Thí nghiệm chính (chưa chạy): 200 user test ngẫu nhiên
python -m backend.generation.experiment --phase test --sample 200 --output-dir runs/llm/test-200
```

- **Mẫu:** `--sample` query ngẫu nhiên (`Random(42)`, không chọn theo việc hệ thống trúng target) + `--with-target` query đầu tiên trong 5.000 query ngẫu nhiên có target trong top 10 (để xem giải thích của gợi ý đúng; báo riêng).
- **Các bước:** gói bằng chứng (lưu `packages.jsonl`, chạy lại thì dùng tiếp) → LLM giải thích top 3 hai lần: **tự do** (cùng thông tin, không có mã nguồn) và **bắt buộc trích dẫn** → validator cho bản trích dẫn → **giám khảo** `gemini-3.5-flash` (thinking low, JSON schema) chấm từng câu *supported / partial / unsupported* so với toàn bộ nguồn của phim và lịch sử → `summary.json`, `explain.json`, `manual_check.csv` (40 câu để chấm tay), và `--report` (Markdown từng record).
- **Chỉ số:** NDCG@10/HR@10 cùng phim của từng nhóm; số câu, số phim có giải thích (độ phủ), tỉ lệ supported/unsupported cho tự do, trích dẫn trước và sau validator; lý do validator loại; bảng ngưỡng độ giống 0–0,5; token, độ trễ p50/p95.
- **Chi phí:** khoảng 1.600 + 300 token mỗi lời gọi giải thích; giám khảo ~3.200 + 1.200 token mỗi user.

### 13. Phân tích theo nhóm — `python -m backend.eval.segments`

```powershell
python -m backend.eval.segments --member itemknn=runs/itemknn-movies-tv-2023/test --member bm25=runs/bm25-movies-tv-2023/test `
  --member fusion=runs/fusion-editions-movies-tv-2023/test --queries $a/queries_test.parquet --catalog $a/catalog_enriched.parquet `
  --editions $a/editions.parquet --query-limit 50000 --output runs/analysis/segments_test50k.json
```

- Chia query theo **độ dài lịch sử** (3–5, 6–20, > 20; tập 5-core nên không có 1–2), **độ phổ biến của target** (head = top 1% catalog, torso = 9% tiếp, tail = 90% còn lại, tính từ dữ liệu fit) và **target có title hay không**; báo NDCG@10, HR@10, Recall@20, Recall@50 cùng phim cho từng model. Mất ~15 phút trên mẫu 50k.
