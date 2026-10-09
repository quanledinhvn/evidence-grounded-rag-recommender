╔════════════════════════════════════════════════════════════════════╗
║                  1. OFFLINE EXPERIMENT PIPELINE                   ║
║              Huấn luyện và lựa chọn cơ chế LLM Gate               ║
╚════════════════════════════════════════════════════════════════════╝

                    ┌─────────────────────────┐
                    │ VALIDATION QUERIES      │
                    │                         │
                    │ • user_id               │
                    │ • as_of                 │
                    │ • target_item_id        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ HYBRID RECOMMENDER      │
                    │                         │
                    │ • BPR-MF                │
                    │ • BM25                  │
                    │ • Rank fusion           │
                    │ • Hybrid top-20         │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
       ┌───────────────────────┐     ┌────────────────────────┐
       │ KHÔNG DÙNG LLM       │     │ LUÔN DÙNG LLM         │
       │                       │     │                        │
       │ Giữ hybrid top-10    │     │ LLM rerank top-20     │
       │                       │     │ thành top-10           │
       └───────────┬───────────┘     └────────────┬───────────┘
                   │                              │
                   ▼                              ▼
       ┌───────────────────────┐     ┌────────────────────────┐
       │ BASELINE RESULT       │     │ LLM RESULT             │
       │                       │     │                        │
       │ • NDCG@10             │     │ • NDCG@10              │
       │ • HR@10               │     │ • HR@10                │
       │ • Latency             │     │ • Latency              │
       │ • Token = 0           │     │ • Token usage          │
       └───────────┬───────────┘     └────────────┬───────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │ TẠO NHÃN CHO GATE        │
                    │                           │
                    │ LLM hữu ích nếu:          │
                    │ • Cải thiện ranking       │
                    │ • Chi phí chấp nhận được │
                    │                           │
                    │ label = CALL / SKIP       │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ GATE FEATURE BUILDER      │
                    │                           │
                    │ • Độ dài user history    │
                    │ • Candidate popularity   │
                    │ • CF score margin        │
                    │ • CF–BM25 disagreement   │
                    │ • Hybrid score entropy   │
                    │ • Evidence coverage      │
                    │ • Long-tail ratio        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
       ┌────────────────────────┐   ┌──────────────────────────┐
       │ HEURISTIC GATE         │   │ LEARNED GATE             │
       │                        │   │                          │
       │ Luật cố định:          │   │ Logistic Regression /    │
       │ • History ngắn         │   │ Decision Tree / XGBoost  │
       │ • Nhiều long-tail      │   │                          │
       │ • CF không chắc chắn   │   │ P(LLM hữu ích | feature) │
       └────────────┬───────────┘   └─────────────┬────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                    ┌────────────────────────────┐
                    │ LOCKED GATE CONFIG         │
                    │                            │
                    │ • Gate type/model          │
                    │ • Decision threshold       │
                    │ • Feature normalization    │
                    │ • Prompt version           │
                    │ • LLM model                │
                    │ • Token/latency budget     │
                    └──────────────┬─────────────┘
                                   │
                                   │ Chỉ dùng test sau khi khóa
                                   ▼


╔════════════════════════════════════════════════════════════════════╗
║                     2. RUNTIME PIPELINE                           ║
║                Quyết định có gọi LLM hay không                    ║
╚════════════════════════════════════════════════════════════════════╝

              ┌───────────────────────────────────┐
              │ Client / CLI / Offline Evaluator  │
              └─────────────────┬─────────────────┘
                                │
                                │ user_id, as_of, config_id
                                ▼
                    ┌────────────────────────┐
                    │ RECOMMENDATION SERVICE │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ PROFILE BUILDER        │
                    │                        │
                    │ • User history         │
                    │ • Positive items       │
                    │ • Negative items       │
                    │ • Seen items           │
                    └────────────┬───────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                ▼
      ┌───────────────────────┐        ┌───────────────────────┐
      │ CF RETRIEVER          │        │ BM25 RETRIEVER        │
      │                       │        │                       │
      │ BPR-MF                │        │ Metadata text         │
      │ Top-50                │        │ Top-50                │
      └───────────┬───────────┘        └───────────┬───────────┘
                  │                                │
                  └───────────────┬────────────────┘
                                  ▼
                    ┌────────────────────────────┐
                    │ FILTER + HYBRID FUSION     │
                    │                            │
                    │ • Loại item đã xem         │
                    │ • Chuẩn hóa rank           │
                    │ • Tính hybrid score        │
                    │ • Giữ rerank pool top-20   │
                    └──────────────┬─────────────┘
                                   ▼
                    ┌────────────────────────────┐
                    │ GATE FEATURE BUILDER       │
                    │                            │
                    │ User features:             │
                    │ • History length           │
                    │                            │
                    │ Candidate features:        │
                    │ • Popularity / long-tail   │
                    │ • Evidence coverage        │
                    │                            │
                    │ Ranking features:          │
                    │ • Score margin             │
                    │ • Score entropy            │
                    │ • CF–BM25 disagreement     │
                    └──────────────┬─────────────┘
                                   ▼
                    ┌────────────────────────────┐
                    │ SELECTIVE LLM GATE         │
                    │                            │
                    │ Dự đoán:                   │
                    │ P(LLM cải thiện ranking)   │
                    └──────────┬───────┬─────────┘
                               │       │
                P < threshold  │       │ P ≥ threshold
                               │       │
                               ▼       ▼
             ┌────────────────────┐  ┌─────────────────────────┐
             │ SKIP LLM           │  │ EVIDENCE RETRIEVAL      │
             │                    │  │                         │
             │ Trả hybrid top-10  │  │ • Metadata              │
             │                    │  │ • Review hợp lệ         │
             │ rerank_status =    │  │ • Chỉ trước as_of       │
             │ gate_skipped       │  │ • Evidence ID           │
             └─────────┬──────────┘  └────────────┬────────────┘
                       │                          ▼
                       │             ┌─────────────────────────┐
                       │             │ SCORE-AWARE PROMPT      │
                       │             │                         │
                       │             │ Mỗi candidate gồm:      │
                       │             │ • CF score/rank         │
                       │             │ • BM25 score/rank       │
                       │             │ • Hybrid score/rank     │
                       │             │ • Evidence              │
                       │             └────────────┬────────────┘
                       │                          ▼
                       │             ┌─────────────────────────┐
                       │             │ LLM RERANK              │
                       │             │                         │
                       │             │ Sắp top-20 thành top-10 │
                       │             └────────────┬────────────┘
                       │                          ▼
                       │             ┌─────────────────────────┐
                       │             │ RANK VALIDATOR          │
                       │             │                         │
                       │             │ • JSON đúng schema      │
                       │             │ • ID thuộc top-20       │
                       │             │ • Không trùng ID        │
                       │             └───────┬────────┬────────┘
                       │                     │        │
                       │                  Hợp lệ      Lỗi
                       │                     │        │
                       │                     │        ▼
                       │                     │  ┌────────────────┐
                       │                     │  │ HYBRID FALLBACK│
                       │                     │  └───────┬────────┘
                       │                     └────┬─────┘
                       │                          ▼
                       │             ┌─────────────────────────┐
                       │             │ CITATION-ENFORCED       │
                       │             │ EXPLANATION             │
                       │             │                         │
                       │             │ Giải thích top-3        │
                       │             │ bằng [H] và [E]         │
                       │             └────────────┬────────────┘
                       │                          │
                       └──────────────┬───────────┘
                                      ▼
                     ┌────────────────────────────────┐
                     │ RESPONSE + EXECUTION TRACE     │
                     │                                │
                     │ • Top-10                       │
                     │ • Gate decision                │
                     │ • Gate probability             │
                     │ • Gate reason/features         │
                     │ • LLM called: true/false       │
                     │ • Rerank status                │
                     │ • Token usage                  │
                     │ • Latency từng bước            │
                     │ • Explanation + citations      │
                     └────────────────────────────────┘


╔════════════════════════════════════════════════════════════════════╗
║                      3. TEST EVALUATION                            ║
╚════════════════════════════════════════════════════════════════════╝

                    ┌─────────────────────────┐
                    │ LOCKED TEST QUERIES     │
                    └────────────┬────────────┘
                                 ▼
       ┌─────────────────────────────────────────────────────┐
       │ CHẠY CÙNG MỘT TẬP QUERY                            │
       │                                                     │
       │ A. Hybrid — không LLM                              │
       │ B. Always LLM                                      │
       │ C. Heuristic gate                                  │
       │ D. Learned gate                                    │
       │ E. Learned gate, không truyền retrieval scores     │
       │ F. Learned gate, có retrieval scores               │
       └─────────────────────────┬───────────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ QUALITY METRICS         │
                    │                         │
                    │ • NDCG@10               │
                    │ • HR@10                 │
                    │ • Candidate Recall@50   │
                    │ • Item coverage         │
                    │ • Long-tail NDCG@10     │
                    │ • Short-history NDCG@10 │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ EFFICIENCY METRICS      │
                    │                         │
                    │ • LLM call rate         │
                    │ • Token/query           │
                    │ • Cost/query            │
                    │ • Mean latency          │
                    │ • p50/p95 latency       │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ GATE METRICS            │
                    │                         │
                    │ • Precision / Recall    │
                    │ • F1                    │
                    │ • False-skip rate       │
                    │ • False-call rate       │
                    │ • Quality–cost curve    │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ TRUST METRICS           │
                    │                         │
                    │ • Citation coverage     │
                    │ • Citation validity     │
                    │ • Semantic support      │
                    │ • Invalid claim rate    │
                    └─────────────────────────┘
