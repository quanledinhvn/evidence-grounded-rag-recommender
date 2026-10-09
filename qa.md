╔══════════════════════════════════════════════════════════════════╗
║                     1. OFFLINE PIPELINE                          ║
║              Chuẩn bị dữ liệu và huấn luyện model                ║
╚══════════════════════════════════════════════════════════════════╝

     Movies_and_TV.train.csv · valid.csv · test.csv
     (user_id, parent_asin, rating, timestamp, history)
                               │
                               ▼
                  ┌────────────────────────┐
                  │ D0. DATA QUALITY AUDIT │
                  │                        │
                  │ • Kiểm tra ID          │
                  │ • Kiểm tra timestamp   │
                  │ • Kiểm tra history     │
                  │ • Giữ hash nguồn       │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ D1. IMPORT SPLITS      │
                  │                        │
                  │ user_id giữ nguyên     │
                  │ parent_asin → item_id  │
                  │ timestamp mili-giây    │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ D2. BUILD QUERIES      │
                  │                        │
                  │ Target validation/test │
                  │ chỉ evaluator được đọc │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │ D3. PRESERVE SPLITS    │
                  │                        │
                  │ Không chia lại dữ liệu │
                  │ Train / Validate / Test│
                  │ theo file nguồn        │
                  └──────┬───────────┬─────┘
                         │           │
              ┌──────────┘           └─────────────┐
              ▼                                    ▼
  ┌─────────────────────────┐        ┌─────────────────────────┐
  │ CF TRAINING             │        │ TEXT INDEX              │
  │                         │        │                         │
  │ • Popularity            │        │ • Cần metadata phim     │
  │ • BPR-MF                │        │ • Không dùng meta sai   │
  └────────────┬────────────┘        └────────────┬────────────┘
               │                                  │
               └──────────────┬───────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ EVIDENCE/HISTORY STORE  │
                 │                         │
                 │ • User history          │
                 │ • Metadata khi bổ sung  │
                 │ • Review khi bổ sung    │
                 │ • SQLite/Parquet        │
                 └────────────┬────────────┘
                              │
                              ▼
              ┌──────────────────────────────────┐
              │         ARTIFACT BUNDLE          │
              │                                  │
              │ • CF model                       │
              │ • BM25 index                     │
              │ • User/item mappings             │
              │ • Catalog                        │
              │ • Evidence/history store         │
              │ • Config + manifest + checksum   │
              └────────────────┬─────────────────┘
                               │
                               │ Nạp khi server khởi động
                               ▼


╔══════════════════════════════════════════════════════════════════╗
║                       2. RUNTIME PIPELINE                        ║
║                     Xử lý một yêu cầu gợi ý                      ║
╚══════════════════════════════════════════════════════════════════╝

             ┌───────────────────────────────────┐
             │ Client / Swagger / CLI / Evaluator│
             └─────────────────┬─────────────────┘
                               │
                               │ user_id, as_of, config_id
                               ▼
                    ┌───────────────────────┐
                    │ FASTAPI / CLI         │
                    │                       │
                    │ Kiểm tra request      │
                    └───────────┬───────────┘
                                ▼
             ┌────────────────────────────────────┐
             │ RECOMMENDATION SERVICE             │
             │                                    │
             │ Điều phối toàn bộ Q0 → Q7          │
             └──────────────────┬─────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ PROFILE BUILDER       │
                    │                       │
                    │ • Lịch sử trước as_of │
                    │ • Item thích: rating≥4│
                    │ • Item ghét: rating≤2 │
                    │ • Toàn bộ seen items  │
                    └───────────┬───────────┘
                                │
               ┌────────────────┴─────────────────┐
               ▼                                  ▼
    ┌────────────────────────┐       ┌────────────────────────┐
    │ CF RETRIEVER           │       │ BM25 RETRIEVER         │
    │                        │       │                        │
    │ BPR-MF                 │       │ Tìm theo metadata      │
    │                        │       │ và từ khóa sở thích    │
    │ Trả top-50             │       │ Trả top-50             │
    └────────────┬───────────┘       └────────────┬───────────┘
                 │                                │
                 └──────────────┬─────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ FILTER + FUSION       │
                    │                       │
                    │ • Loại item đã xem    │
                    │ • Gộp CF và BM25      │
                    │ • Tính hybrid score   │
                    │ • Giữ top-50          │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ RERANK POOL           │
                    │                       │
                    │ Chọn top-20 ứng viên  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ EVIDENCE RETRIEVAL    │
                    │                       │
                    │ Mỗi item lấy:         │
                    │ • Metadata            │
                    │ • Tối đa 2 review     │
                    │ • Chỉ trước as_of     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ LLM RERANK            │
                    │                       │
                    │ Sắp top-20 thành      │
                    │ top-10                │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ RANK VALIDATOR        │
                    │                       │
                    │ • JSON đúng schema?   │
                    │ • ID thuộc top-20?    │
                    │ • Có ID trùng không?  │
                    └───────┬────────┬──────┘
                            │        │
                       Hợp lệ        Lỗi
                            │        │
                            │        ▼
                            │  ┌──────────────────┐
                            │  │ HYBRID FALLBACK  │
                            │  │ Dùng hybrid top-K│
                            │  └────────┬─────────┘
                            │           │
                            └─────┬─────┘
                                  ▼
                    ┌───────────────────────┐
                    │ LLM EXPLANATION       │
                    │                       │
                    │ Giải thích top-3      │
                    │ kèm [H] và [E]        │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ CITATION VALIDATOR    │
                    │                       │
                    │ • Nguồn tồn tại?      │
                    │ • Nguồn đúng item?    │
                    │ • History hợp lệ?     │
                    └───────────┬───────────┘
                                ▼
              ┌──────────────────────────────────┐
              │           RESPONSE JSON          │
              │                                  │
              │ • Top-10 sản phẩm                │
              │ • Giải thích top-3               │
              │ • Evidence                       │
              │ • Status/warnings                │
              │ • Latency/token usage            │
              └──────────────────────────────────┘
