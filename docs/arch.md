```text
                     KIẾN TRÚC HỆ GỢI Ý RAG–LLM
                             
┌──────────────────────────────────────────────────────────────┐
│        AMAZON DIGITAL MUSIC 5-CORE                           │
│  user_id · item_id · rating · timestamp · review · metadata  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     TIỀN XỬ LÝ DỮ LIỆU                       │
│  • Làm sạch review                                           │
│  • Sắp xếp tương tác theo thời gian                          │
│  • Train / Validation / Test                                 │
│  • Loại dữ liệu có nguy cơ rò rỉ                             │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
                ▼                              ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│  DỮ LIỆU TƯƠNG TÁC        │      │  DỮ LIỆU VĂN BẢN          │
│                           │      │                           │
│  user ─ rating ─ item     │      │  • Metadata sản phẩm     │
│  lịch sử người dùng       │      │  • Review hợp lệ         │
└─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │
              ▼                                  ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│ COLLABORATIVE RETRIEVER   │      │  TEXT RETRIEVER – BM25   │
│                           │      │                           │
│ BPR-MF hoặc ItemKNN       │      │ Truy xuất theo hồ sơ     │
│                           │      │ sở thích dạng văn bản    │
│ → ứng viên + CF score     │      │ → ứng viên + BM25 score  │
└─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │
              └────────────────┬─────────────────┘
                               ▼
                 ┌───────────────────────────────┐
                 │       HYBRID RETRIEVAL        │
                 │                               │
                 │ Chuẩn hóa và kết hợp điểm:    │
                 │                               │
                 │ score = α·CF + (1-α)·BM25     │
                 │                               │
                 │       → Top-50 ứng viên       │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │      RETRIEVE EVIDENCE        │
                 │                               │
                 │ Với mỗi ứng viên, lấy:        │
                 │ • Metadata liên quan          │
                 │ • Các đoạn review phù hợp     │
                 │ • Gán mã [E1], [E2], ...      │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │         LLM RERANKER          │
                 │                               │
                 │ Đầu vào:                      │
                 │ • Hồ sơ người dùng            │
                 │ • Top-20 ứng viên             │
                 │ • Các đoạn evidence           │
                 │                               │
                 │ Đầu ra JSON: Top-10           │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  SINH GIẢI THÍCH CÓ CĂN CỨ   │
                 │                               │
                 │ Giải thích cho Top-3          │
                 │ kèm trích dẫn [E1], [E2]      │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │       KIỂM TRA ĐẦU RA         │
                 │                               │
                 │ • JSON đúng schema?           │
                 │ • Item thuộc candidate set?   │
                 │ • Mã evidence tồn tại?        │
                 │ • Có nhận định thiếu căn cứ?  │
                 └───────────────┬───────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────┐
│                         KẾT QUẢ                              │
│                                                              │
│  Top-10 sản phẩm âm nhạc được gợi ý                          │
│  Top-3 có giải thích và trích dẫn bằng chứng                 │
│                                                              │
│  Ví dụ:                                                      │
│  1. Album A – phù hợp với sở thích jazz nhẹ [E1][E3]         │
│  2. Album B – tương tự các album từng đánh giá cao [E2]      │
└──────────────────────────────────────────────────────────────┘
```

Luồng đánh giá:

```text
                         TẬP TEST
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
       Baseline CF     Hybrid CF+BM25    RAG–LLM đầy đủ
           │                │                │
           └────────────────┼────────────────┘
                            ▼
            ┌────────────────────────────────┐
            │ Recall@10 · NDCG@10 · HR@10   │
            │ Candidate Recall@20/50         │
            │ Citation validity              │
            │ Supported-claim rate           │
            │ Latency · Token usage          │
            └────────────────────────────────┘
```

Ý tưởng cốt lõi là CF và BM25 chịu trách nhiệm **tìm sản phẩm**, còn LLM chỉ **sắp xếp lại và giải thích**. Cách phân chia này ngăn LLM tự tạo ra sản phẩm không tồn tại và giữ chi phí trong phạm vi dự án ba tuần.