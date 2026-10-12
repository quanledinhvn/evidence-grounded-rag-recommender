# D0 — Hợp đồng và kiểm tra dữ liệu Movies and TV 2023

## 1. Nguồn dữ liệu đang dùng

Dự án dùng ba file CSV trong `dataset/2023/` và tôn trọng cách chia tập có sẵn:

| File | Split | Số dòng dữ liệu |
|---|---|---:|
| `Movies_and_TV.train.csv` | train | 6.126.723 |
| `Movies_and_TV.valid.csv` | validation | 657.203 |
| `Movies_and_TV.test.csv` | test | 657.203 |
| **Tổng** |  | **7.441.129** |

Các số trên không tính dòng header. Không dùng hai file `Digital_Music.jsonl` và `meta_Digital_Music.jsonl` cho thí nghiệm Movies and TV.

## 2. Schema CSV

| Trường | Kiểu sau import | Ý nghĩa |
|---|---|---|
| `user_id` | string | ID người dùng |
| `parent_asin` | string | ID sản phẩm Movies and TV; được ánh xạ thành `item_id` |
| `rating` | float | Điểm đánh giá từ 1 đến 5 |
| `timestamp` | integer | Unix timestamp theo mili-giây |
| `history` | string | Danh sách `parent_asin` trước target, phân cách bằng khoảng trắng |

Mỗi dòng là một target tuần tự. Trong file train, các dòng của cùng user biểu diễn các bước lịch sử tăng dần. File validation và test chứa target held-out cùng lịch sử đã có trước target.

## 3. Quy tắc import và kiểm tra

Lệnh `recsys-data import-splits` thực hiện các kiểm tra sau:

- header phải đúng thứ tự năm trường đã công bố;
- `user_id` và `parent_asin` phải là chuỗi không rỗng;
- `rating` phải hữu hạn và nằm trong đoạn từ 1 đến 5;
- `timestamp` phải là số nguyên dương;
- target `parent_asin` không được xuất hiện trong chính trường `history`;
- split đầu vào được giữ nguyên, không chạy lại leave-last-two-out.

Importer ánh xạ `parent_asin` thành `item_id`, tạo `review_id` tổng hợp ổn định theo split và số dòng, rồi xuất Parquet theo batch. `history` nguồn không được dùng như nhãn; lịch sử phục vụ model được dựng từ các interaction thuộc split được phép.

## 4. Artifact đầu ra

| Artifact | Nội dung |
|---|---|
| `splits.parquet` | Interaction của cả ba split |
| `queries_val.parquet` | Target validation tách khỏi service |
| `queries_test.parquet` | Target test tách khỏi service |
| `catalog.parquet` | Hợp các `parent_asin` xuất hiện làm target |
| `split_stats.json` | Hash nguồn, số dòng, số user và số item catalog |

Schema interaction gồm `user_id`, `item_id`, `rating`, `timestamp`, `review_id`, các trường text rỗng, `source_line` và `split`. Timestamp vẫn dùng đơn vị mili-giây trong toàn bộ artifact Movies and TV.

## 5. Giới hạn dữ liệu văn bản

Ba CSV không tự chứa title, mô tả, thể loại, diễn viên, đạo diễn hoặc nội dung review. Metadata Movies and TV đã được import riêng và nối chính xác bằng `parent_asin`, tạo `artifacts/movies-tv-2023/catalog_enriched.parquet` với coverage 197.943/197.943 item. Vì vậy:

- có thể huấn luyện và đánh giá BPR-MF cùng Popularity;
- có thể xây BM25, semantic search và metadata evidence từ catalog enriched;
- chưa có review evidence vì review text Movies and TV chưa được import;
- tuyệt đối không nối với metadata Digital Music chỉ vì cùng dùng ASIN.

Các baseline text phải dùng cùng `catalog_enriched.parquet`; item universe và ba split không thay đổi sau enrichment.
