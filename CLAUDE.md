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
- Máy hiện tại không có `uv`; venv đã cài sẵn ở `%LOCALAPPDATA%\Temp\fq\.venv`. Chạy từ gốc repo, luôn dùng `-m` để Python lấy `backend/` của repo:
  `& "$env:LOCALAPPDATA\Temp\fq\.venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider` và `... -m backend.cf.cli ...`.
- Windows: `implicit` lỗi `DLL load failed ... filename too long` khi venv nằm ở đường dẫn dài; dùng venv ở đường dẫn ngắn.
- Dự án làm ở local trên nhánh `f/quan`: commit bình thường, **không push**. `dataset/`, `artifacts/`, `runs/` không đưa lên Git.

## Bản đồ code

| File | Việc |
|---|---|
| `backend/data/importer.py` | 3 CSV split → `splits.parquet`, `queries_{val,test}.parquet`, `catalog.parquet` |
| `backend/data/enrich.py` | Nối metadata vào catalog → `catalog_enriched.parquet` |
| `backend/profile/profile.py` | `load_history`, `load_catalog`, `build_profile` |
| `backend/cf/common.py` | `FitData`, `load_fit_data`, `Popularity`, `top_k`, `rank_items` |
| `backend/cf/itemknn.py`, `backend/cf/bpr.py` | Hai model CF, cùng giao diện `fit / recommend / recommend_batch / save / load` |
| `backend/cf/evaluate.py` | NDCG@10, HR@10, Recall@50 |
| `tests/test_data.py`, `tests/test_profile.py`, `tests/test_cf.py` | 24 test, dữ liệu giả rất nhỏ |

## Dữ liệu và artifact

| Đường dẫn | Nội dung |
|---|---|
| `artifacts/movies-tv-2023/splits.parquet` | 7.441.129 tương tác (train 6.126.723, validation 657.203, test 657.203) |
| `artifacts/movies-tv-2023/queries_{val,test}.parquet` | 657.203 query mỗi pha, mỗi user một target, `as_of` = timestamp target |
| `artifacts/movies-tv-2023/catalog_enriched.parquet` | 197.943 item, đủ metadata; 82.840 item Prime Video không có title |
| `runs/{itemknn,bpr}-movies-tv-2023/test/` | Model pha test (`config.json`, `fit_data.npz`, `similarity.npz` hoặc `factors.npz`) |
| `runs/{itemknn,bpr,popularity}-movies-tv-2023/test_evaluation.json` | Kết quả trên toàn bộ test |
| `runs/itemknn-movies-tv-2023/validation_metrics.csv` | Bảng tune ItemKNN trên validation |
| `runs/cf-movies-tv-2023-full/20261009T135249Z/` | Lượt BPR cũ trên AWS; chỉ còn giá trị tham khảo (`evaluation.json`), code mới không đọc được bundle này |
| `dataset/Movies_and_TV.jsonl`, `dataset/meta_Movies_and_TV.jsonl` | Review thô (17,3 triệu) và metadata thô, chưa import |

Đặc điểm cần nhớ: trung vị 5 tương tác train mỗi user; split không có review text; một phim có nhiều `parent_asin` (DVD, Blu-ray, Prime Video).

## Bất biến

- Mọi dữ liệu dùng cho một query phải có `timestamp < as_of`.
- Pha `validation` chỉ fit trên `train`; pha `test` fit trên `train` + `validation`, không đọc `test`. Chọn cấu hình trên validation, test chỉ dùng để báo cáo.
- Đánh giá luôn lấy 50 item mỗi query; `recommend` giới hạn `limit ≤ 50`.
- Item được đánh số theo `item_id` tăng dần; điểm bằng nhau thì phá hòa theo `item_id`.

## Hướng dẫn chạy test các phần đã implement

Ví dụ xuyên suốt là một user thật: `AGXVBIUFLFGMVLATYXHJYL4A5Q7Q`. Lịch sử của user này có 7 phim ở train (2006–2017), một phim ở validation là **Harry Potter phần 1** (`B002DQDV6Y`, xem lúc `1530556502815`), và target test là **Harry Potter phần 3** (`B00271DNP4`, `as_of = 1530556520803`, tức 18 giây sau).

Trong các lệnh dưới, `python` là `& "$env:LOCALAPPDATA\Temp\fq\.venv\Scripts\python.exe"` (hoặc `uv run python`), chạy từ gốc repo.

### 0. Unit test (không cần dữ liệu thật, ~15 giây)

```powershell
python -m pytest -q -p no:cacheprovider                 # toàn bộ 24 test
python -m pytest -q -p no:cacheprovider tests/test_cf.py # chỉ phần CF
```

| File | Kiểm tra gì |
|---|---|
| `tests/test_data.py` | Import giữ nguyên split, `review_id`, file query, catalog; từ chối rating ngoài 1–5, timestamp ≤ 0, target nằm trong history; enrich làm sạch HTML, chọn bản metadata nhiều text hơn |
| `tests/test_profile.py` | Pha validation không thấy split validation; tương tác tại `as_of` bị loại; phân loại thích/ghét; tương tác mới nhất thắng; công thức trọng số |
| `tests/test_cf.py` | Pha fit đúng split; Popularity; cosine và cắt hàng xóm; ItemKNN ưu tiên phim vừa xem, trường `because`; BPR không gợi ý phim đã xem; metric tính đúng; CLI |

Kết quả mong đợi: `24 passed`.

### 1. Nhập split — `backend/data/importer.py`

```powershell
python -m backend.data.cli import-splits --train dataset/2023/Movies_and_TV.train.csv `
  --validation dataset/2023/Movies_and_TV.valid.csv --test dataset/2023/Movies_and_TV.test.csv `
  --output-dir artifacts/movies-tv-2023
```

- **Input:** 3 CSV, cột `user_id,parent_asin,rating,timestamp,history`. Ví dụ một dòng train: `AGXVBIUFLFGMVLATYXHJYL4A5Q7Q,B0091VXC54,5.0,1443550066000,B0002J58ME`.
- **Các bước và lý do:**
  1. Kiểm tra ID không rỗng, rating 1–5, timestamp dương: dữ liệu hỏng sẽ làm sai model mà không báo lỗi.
  2. Kiểm tra target không nằm trong `history` của chính dòng đó: nếu có, model "đoán" đúng một phim user đã xem, metric bị thổi phồng.
  3. Đổi `parent_asin` thành `item_id`, gắn nhãn split, tạo `review_id = amazon-2023:<split>:<dòng CSV>` để truy ngược về file gốc.
  4. Tách target validation/test ra file query riêng: evaluator đọc đáp án, model thì không.
- **Output:** `splits.parquet`, `queries_val.parquet`, `queries_test.parquet`, `catalog.parquet`, `split_stats.json`.
- **Ví dụ:** dòng trên thành `{user_id: AGXV…, item_id: B0091VXC54, rating: 5.0, timestamp: 1443550066000, review_id: amazon-2023:train:3, split: train}`. Query test của user: `{as_of: 1530556520803, target_item_id: B00271DNP4, target_rating: 5.0}`.
- **Kiểm tra nhanh:** `split_stats.json` phải có train 6.126.723, validation 657.203, test 657.203 dòng.

### 2. Nối metadata — `backend/data/enrich.py`

```powershell
python -m backend.data.cli enrich-catalog --catalog artifacts/movies-tv-2023/catalog.parquet `
  --metadata dataset/2023/meta_Movies_and_TV.jsonl --output artifacts/movies-tv-2023/catalog_enriched.parquet
```

- **Input:** `catalog.parquet` (danh sách `item_id`) và file metadata JSONL (một sản phẩm mỗi dòng).
- **Các bước và lý do:**
  1. Chỉ giữ bản ghi có `parent_asin` thuộc catalog, nối **chính xác** theo ID: không ghép gần đúng theo tên để tránh gán nhầm phim.
  2. Làm sạch HTML, gộp danh sách thành chuỗi; `details` chỉ giữ khóa hữu ích (`Directors`, `Starring`, `Genre`…).
  3. Ghép các trường nội dung thành `metadata_text`: một văn bản dùng chung cho BM25, embedding và bằng chứng, để ba bước này nhìn cùng một nội dung.
  4. Gắn `source_id` để lời giải thích trích dẫn được nguồn.
- **Output:** `catalog_enriched.parquet` (mỗi item một dòng) và `catalog_enriched.stats.json` (độ phủ từng trường).
- **Ví dụ:** `B00271DNP4` → `main_category: Prime Video`, `details: Starring: Daniel Radcliffe Rupert Grint Emma Watson Directors: Alfonso Cuarón …`, `title` rỗng (đúng với 42% item Prime Video).
- **Kiểm tra nhanh:** stats có `catalog_items_with_metadata = 197943`.

### 3. Hồ sơ người dùng — `backend/profile/profile.py`

```powershell
python -m backend.profile.cli --splits artifacts/movies-tv-2023/splits.parquet `
  --catalog artifacts/movies-tv-2023/catalog_enriched.parquet `
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
- **Ví dụ:** 8 phim đã xem (7 train + Harry Potter 1 ở validation; Harry Potter 3 **không** có vì xảy ra đúng lúc `as_of`). `H1 = B002DQDV6Y`, weight 0,99999 (xem 18 giây trước); `H2 = B01LWY8995`, weight 0,0186 (xem 517 ngày trước).

### 4. Train model CF — `backend/cf/itemknn.py`, `backend/cf/bpr.py`

```powershell
python -m backend.cf.cli train --model itemknn --splits artifacts/movies-tv-2023/splits.parquet `
  --catalog artifacts/movies-tv-2023/catalog.parquet --phase test --output-dir runs/itemknn-movies-tv-2023/test
python -m backend.cf.cli train --model bpr ... --output-dir runs/bpr-movies-tv-2023/test
```

- **Input:** `splits.parquet`, `catalog.parquet`, pha, tham số (mặc định ItemKNN 400 hàng xóm, $\gamma = 0{,}7$; BPR 64 factors, 60 epoch).
- **Các bước và lý do:**
  1. `load_fit_data` chỉ đọc split được phép của pha (test: train + validation, **không** test) và xếp lịch sử từng user theo thời gian: thứ tự cần cho trọng số độ gần đây.
  2. ItemKNN: tính cosine giữa các phim từ ma trận user × phim, mỗi phim giữ 400 hàng xóm gần nhất. Cắt bớt để model gọn (~250 MB) mà gần như không mất độ chính xác (đã đo trên validation).
  3. BPR: thư viện `implicit` học vector user/phim từ tương tác rating ≥ 4.
- **Output:** thư mục model gồm `config.json`, `fit_data.npz`, `similarity.npz` (ItemKNN) hoặc `factors.npz` (BPR).
- **Thời gian:** ItemKNN khoảng 5 phút, BPR khoảng 4 phút trên máy hiện tại. BPR train đa luồng nên số liệu có thể lệch rất nhỏ giữa các lần chạy (local 0,0131 so với AWS cũ 0,0132 NDCG@10).

### 5. Gợi ý cho một user

```powershell
python -m backend.cf.cli recommend --model-dir runs/itemknn-movies-tv-2023/test `
  --user-id AGXVBIUFLFGMVLATYXHJYL4A5Q7Q --limit 3
```

- **Input:** thư mục model, user, số item; `--exclude` để loại thêm phim.
- **Các bước và lý do:** cộng điểm hàng xóm của các phim đã xem (phim vừa xem nặng ký nhất) → bỏ phim đã xem → lấy top → thiếu thì bổ sung popularity để luôn trả đủ danh sách.
- **Output:** danh sách `{item_id, score, source, because}`; `because` là các phim trong lịch sử góp điểm nhiều nhất.
- **Ví dụ:** hạng 1 là **`B00271DNP4` (Harry Potter 3) — đúng target test**, `because = [{item_id: B002DQDV6Y, similarity: 0,341, weight: 1,0}]`: vì user vừa xem Harry Potter 1, và nhiều người xem phần 1 cũng xem phần 3.
- Lệnh mất khoảng 15 giây, chủ yếu để nạp model.

### 6. Đánh giá — `backend/cf/evaluate.py`

```powershell
python -m backend.cf.cli evaluate --model-dir runs/itemknn-movies-tv-2023/test `
  --queries artifacts/movies-tv-2023/queries_test.parquet --output runs/itemknn-movies-tv-2023/test_evaluation.json
# thêm --popularity để đánh giá baseline Popularity; --query-limit 20000 để chạy nhanh trên mẫu
```

- **Input:** model, file query (`user_id`, `target_item_id`, `target_rating`).
- **Các bước và lý do:** với mỗi query lấy 50 item, tìm hạng của target; HR@10 = có trong top 10; NDCG@10 = $1/\log_2(rank+1)$ (đúng hạng cao được thưởng nhiều hơn); Recall@50 = có trong 50 ứng viên (cái mà các bước sau còn có thể xếp lại). Báo thêm riêng cho target rating ≥ 4 (`positive_target`) vì mục tiêu là gợi ý phim user *thích*.
- **Output:** JSON gồm `ndcg_at_10`, `hr_at_10`, `recall_at_50`, `fallback_rate`, `item_coverage`, `mean_latency_ms`, `positive_target`.
- **Ví dụ:** ItemKNN trên toàn bộ test: `ndcg_at_10 = 0.0491`, `hr_at_10 = 0.0768`, `recall_at_50 = 0.1194`. Mất khoảng 10 phút (ItemKNN), 2 phút (Popularity), 80 phút (BPR, vì phải chấm cả 198k item cho mỗi user); dùng `--query-limit 20000` khi chỉ cần số nhanh.

---

# Tiến độ dự án

Các mục dưới đây thay đổi theo thời gian; phần trên là thông tin cố định của đồ án.

## Tiến độ (cập nhật 2026-10-17)

| Hạng mục | Tác dụng trong đồ án | Trạng thái |
|---|---|---|
| Nhập 3 split (`recsys-data import-splits`) | Biến CSV gốc thành bảng chuẩn và **tách target đánh giá ra file query riêng**, để model không bao giờ thấy đáp án. Là nền cho mọi bước sau. | Xong |
| Nối metadata (`recsys-data enrich-catalog`) | Gắn nội dung phim (title, mô tả, đạo diễn, diễn viên…) vào từng item. Không có bước này thì không làm được BM25, embedding, bằng chứng [E#] và lời giải thích. | Xong |
| ① Hồ sơ người dùng (`recsys-profile`) | Tóm tắt sở thích của user **tại thời điểm gợi ý**: phim thích/không thích, trọng số theo độ gần đây, text kèm mã nguồn. Là đầu vào truy vấn cho BM25/embedding và là nguồn bằng chứng [H#]. | Xong phần lõi; chậm (~6 s/user do quét toàn bộ parquet) |
| ② ItemKNN | Nhánh CF chính: tìm ứng viên từ hành vi đám đông ("người xem A thường xem B"), ưu tiên phim vừa xem. Trường `because` cho sẵn bằng chứng "vì bạn đã xem X". | Xong, đã tune và đánh giá |
| ② BPR-MF | Baseline học máy cổ điển; chứng minh vì sao chọn ItemKNN làm nhánh chính (run B). | Xong, chạy local |
| Popularity | Mốc thấp nhất không cá nhân hóa (run A) và phần bổ sung khi model không đủ ứng viên. | Xong |
| Đánh giá (`recsys-cf evaluate`) | Đo khách quan, cùng một giao thức cho mọi run A–G, để so sánh công bằng. | Xong |
| ② BM25, embedding | Thêm phim giống về **nội dung** (cùng đạo diễn, cùng thể loại, cốt truyện tương tự); giúp phim mới hoặc ít người xem mà CF không thấy. | Chưa làm |
| ③ Fusion | Trộn các nhánh thành một danh sách 50 ứng viên; kỳ vọng tốt hơn nhánh tốt nhất. | Chưa làm; đã có số liệu ban đầu về prior popularity |
| ④ Kho bằng chứng | Lấy đoạn metadata và review của người khác (trước `as_of`) cho top 20, để LLM chỉ được nói điều có nguồn. | Chưa làm |
| ⑤ LLM rerank + giải thích, ⑥ validator | Xếp lại top 20 → top 10 và giải thích top 3 bắt buộc trích dẫn; validator loại câu không có nguồn. Đây là đóng góp chính của đề tài. | Chưa làm |
| Service, API, run A–G, báo cáo | Ghép mọi bước thành một luồng, chạy bảng thí nghiệm và viết báo cáo. | Chưa làm |

Kết quả trên toàn bộ 657.203 query test (chạy lại bằng code đã đơn giản hóa ngày 17/10, khớp kết quả cũ):

| Phương pháp | NDCG@10 | HR@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity | 0,0091 | 0,0171 | 0,045 |
| BPR-MF (64 factors, 60 epoch) | 0,0131 | 0,0227 | 0,045 |
| ItemKNN (400 hàng xóm, $\gamma = 0{,}7$) | 0,0491 | 0,0768 | 0,119 |

Test thấp hơn validation khoảng 13–16% với **mọi** phương pháp (target test là tương tác cuối cùng, khó hơn), nên không phải dấu hiệu ItemKNN bị tune quá khớp.

Phát hiện dùng cho bước tiếp theo (20.000 query validation, seed 42):

- Trọng số độ gần đây được giữ ($\gamma = 0{,}7$). Ablation không dùng nó ($\gamma = 1$) làm NDCG@10 giảm khoảng 17% (0,0586 → 0,0489).
- ItemKNN + $\lambda \cdot$ prior popularity theo thứ hạng: $\lambda = 0{,}3$ cho NDCG 0,0585 → 0,0688 và Recall@50 0,134 → 0,179. Recent popularity 30 ngày đứng riêng đạt khoảng 2,5 lần BPR.
- BPR yếu vì bỏ qua thứ tự thời gian, chưa hội tụ (AUC còn tăng ở epoch 60) và dữ liệu thưa.

## Việc tiếp theo

1. Tăng tốc đọc lịch sử cho profile (sắp `splits.parquet` theo `user_id` hoặc dùng SQLite).
2. Xây `item_doc` từ `catalog_enriched` (title, mô tả, thể loại đã lọc, đạo diễn, diễn viên; mỗi trường có `source_id`); item thiếu title dùng đạo diễn và diễn viên.
3. Nhánh BM25 (`bm25s`) và embedding (`bge-small`), mỗi nhánh xuất pool top-100 và đánh giá riêng.
4. Fusion ItemKNN + BM25 + embedding + popularity (toàn cục và 30 ngày), tune trên validation, khóa cấu hình.
5. Import review thô vào SQLite theo `(item_id, ts)`, lọc review ngắn hoặc nói về giao hàng; đưa review của chính user vào profile.
6. Chọn bằng chứng, LLM rerank và giải thích, validator; chạy run A–G.

Vấn đề mở không chặn tiến độ: `preprocess.py` và các file mà [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md) nhắc tới không có trong repo; cần hỏi người viết khảo sát nếu dùng Dense-Works.

## Quyết định đang chờ

- Có dùng tập Dense-Works (gộp phiên bản, 20-core; xem [docs/data/raw-dataset-survey.md](docs/data/raw-dataset-survey.md)) cho giai đoạn LLM không. Cần chốt trước bước 6: với Recall@50 = 0,119, chỉ khoảng 24/200 user có target trong pool.
- LLM provider, ngân sách token, hạn nộp.

## Nhật ký phiên làm việc (13–17/10/2026)

- Kiểm tra nhánh `f/quan`; chạy thử luồng profile → CF trên dữ liệu thật, không rò rỉ.
- So sánh BPR với baseline đơn giản → BPR yếu; cài ItemKNN, tune trên validation, đánh giá full test.
- Phát hiện và sửa lỗi số liệu: bảng ItemKNN đầu tiên bị lẫn prior popularity do lỗi phá hòa trong script thử; số đúng là 0,0491 NDCG@10.
- Sắp xếp lại tài liệu theo module, thêm `README.md`, `CLAUDE.md`, `docs/overview.md`, xóa tài liệu nháp/lỗi thời.
- **Đơn giản hóa toàn bộ backend** theo yêu cầu (code đồ án, không phải sản phẩm): ~3.700 → ~950 dòng (không tính dòng trống); bỏ pipeline JSONL thô, AWS/SkyPilot, tuning grid, checksum, Protocol; profile thành 3 hàm; CF thành `common / itemknn / bpr / evaluate`. Chạy lại trên dữ liệu thật: ItemKNN và Popularity khớp kết quả cũ tới nhiều chữ số; BPR train local.
