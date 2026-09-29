# 5 ĐỀ TÀI RAG–LLM CHO HỌC PHẦN HỆ GỢI Ý IT4613

HUST hiện liệt kê IT4613 với tên học phần **Hệ gợi ý – Recommender System**. Vì vậy, các đề tài dưới đây lấy bài toán gợi ý làm trung tâm; RAG và LLM là công cụ cải tiến chứ không biến đề tài thành một chatbot hỏi–đáp thuần túy. [Thư viện HUST](https://library.hust.edu.vn/vi/node/779)

## 1. Tiêu chí thiết kế đề tài

Mỗi đề tài nên có:

- Bài toán gợi ý cụ thể: top-\(K\), tuần tự, hội thoại, cold-start hoặc giải thích gợi ý.
- Baseline truyền thống và baseline dựa trên LLM.
- Thành phần retrieval được mô tả rõ: truy xuất người dùng tương tự, item, review, knowledge graph hoặc lịch sử tương tác.
- Ít nhất một cơ chế cải tiến có thể kiểm chứng bằng ablation study.
- Dataset công khai và quy trình chia tập dữ liệu tái lập được.
- Đánh giá đồng thời độ chính xác, độ đa dạng, độ trung thực, chi phí và độ trễ khi phù hợp.

## 2. Danh sách 5 đề tài

| # | Đề tài nghiên cứu | Hướng đóng góp thuật toán | Dataset/benchmark công khai | Bài báo tham khảo gần và sát |
|---:|---|---|---|---|
| 1 | **Evidence-grounded RAG–LLM cho gợi ý cá nhân hóa có giải thích** | Xây dựng hybrid retriever kết hợp collaborative filtering, metadata và review; LLM vừa xếp hạng top-\(K\), vừa sinh giải thích có trích dẫn bằng chứng. Có thể đề xuất evidence-aware reranking hoặc cơ chế kiểm tra xem từng nhận định trong lời giải thích có được hỗ trợ bởi review đã truy xuất hay không. | Amazon Reviews, Yelp Open Dataset, MovieLens kết hợp TMDb. | [LLMRec: Benchmarking Large Language Models on Recommendation Task](https://arxiv.org/abs/2308.12241): đánh giá LLM trên dự đoán rating, gợi ý tuần tự, gợi ý trực tiếp và sinh giải thích. |
| 2 | **Knowledge-graph RAG cho gợi ý cold-start người dùng và sản phẩm mới** | Tạo graph gồm user–item–attribute–entity; sử dụng graph retrieval hoặc multi-hop retrieval để tìm item và bằng chứng liên quan trước khi LLM rerank. Có thể đề xuất adaptive hop selection, confidence-aware fusion hoặc cơ chế chống hallucination khi metadata thưa. | MovieLens 1M/20M + DBpedia/Wikidata; Amazon Reviews; Yelp Open Dataset. | [ColdRAG: Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation](https://arxiv.org/abs/2505.20773): dùng knowledge graph động và suy luận nhiều bước để gợi ý cold-start. [KALM4Rec](https://arxiv.org/abs/2405.19612): retrieval và LLM reranking cho người dùng cold-start. |
| 3 | **Conversational RAG–LLM với bộ nhớ sở thích động** | Trích xuất và cập nhật trạng thái sở thích qua từng lượt hội thoại; truy xuất collaborative evidence và item metadata; phát hiện xung đột giữa sở thích cũ và mới; lựa chọn hỏi thêm hay đưa ra gợi ý. Có thể tối ưu đồng thời chất lượng gợi ý và số lượt hội thoại. | ReDial, Reddit-v2, INSPIRED, OpenDialKG. | [CRAG: Collaborative Retrieval for LLM-based Conversational Recommender Systems](https://arxiv.org/abs/2502.14137): kết hợp collaborative filtering với LLM trên ReDial và Reddit-v2. [G-CRS](https://arxiv.org/abs/2503.06430): graph retrieval, Personalized PageRank và LLM cho gợi ý hội thoại. |
| 4 | **Temporal RAG–LLM cho gợi ý tuần tự và sở thích thay đổi theo thời gian** | Truy xuất có trọng số thời gian từ lịch sử gần, lịch sử dài hạn và người dùng tương tự; LLM rerank theo ngữ cảnh phiên hiện tại. Có thể đề xuất change-point detection, adaptive time window hoặc cơ chế cân bằng sở thích ngắn hạn–dài hạn. | Amazon Reviews theo timestamp, MovieLens 20M/25M, MIND, KuaiRec. | [LIKR: LLM’s Intuition-aware Knowledge Graph Reasoning for Cold-start Sequential Recommendation](https://arxiv.org/abs/2412.12464): kết hợp tri thức LLM, knowledge graph và thông tin thời gian cho gợi ý tuần tự cold-start. |
| 5 | **Budget-aware adaptive RAG cho hệ gợi ý LLM thời gian thực** | Với mỗi yêu cầu, hệ thống tự quyết định có cần retrieval hay không, dùng retriever nào, lấy bao nhiêu item và có cần gọi LLM lớn hay không. Tối ưu đa mục tiêu giữa NDCG/Recall, latency, số token và chi phí; có thể dùng contextual bandit, early exit hoặc confidence-based routing. | Criteo, Avazu, Amazon Reviews, MovieLens, MIND. | [The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit](https://arxiv.org/abs/2501.02173): kết hợp graph retrieval và early exit để cân bằng độ chính xác với hiệu năng. |

## 3. Baseline và thước đo đề xuất

| Thành phần | Lựa chọn phù hợp |
|---|---|
| Baseline truyền thống | Popularity, ItemKNN, BPR-MF, LightGCN |
| Gợi ý tuần tự | GRU4Rec, SASRec, BERT4Rec |
| Retrieval | BM25, dense retrieval, LightGCN retrieval, graph retrieval |
| LLM baseline | Zero-shot LLM, few-shot LLM, LLM có retrieval nhưng không có cơ chế cải tiến |
| Độ chính xác | Recall@\(K\), NDCG@\(K\), Hit Rate@\(K\), MRR |
| Beyond-accuracy | Coverage, diversity, novelty, serendipity, popularity bias |
| Chất lượng RAG | Recall của retriever, evidence precision, faithfulness, hallucinated-item rate |
| Hiệu năng | Latency, throughput, số token, chi phí trung bình mỗi truy vấn |
| Hội thoại | Success rate, số lượt hội thoại, preference-consistency |
| Thực nghiệm | Ablation, sensitivity, nhiều seed và kiểm định thống kê |

Đặc biệt, không nên chỉ dùng BLEU hoặc ROUGE để đánh giá phần giải thích. Nghiên cứu LLMRec cho thấy các metric này có thể không phản ánh đúng chất lượng giải thích do LLM sinh ra; nên bổ sung faithfulness, kiểm tra bằng chứng và đánh giá con người hoặc LLM-as-a-judge có kiểm chuẩn. [LLMRec](https://arxiv.org/abs/2308.12241)

## 4. Yêu cầu triển khai theo hai pha

**Pha 1 – baseline bắt buộc**

- Tiền xử lý một dataset công khai.
- Cài đặt ít nhất một baseline truyền thống.
- Xây dựng RAG cơ bản: retrieval → tạo prompt → LLM reranking hoặc sinh giải thích.
- Báo cáo Recall@\(K\), NDCG@\(K\), latency và chi phí.
- Công bố cấu hình, prompt, seed và cách chia train/validation/test.

**Pha 2 – đóng góp nghiên cứu**

Chọn ít nhất một hướng:

- Hybrid hoặc graph-based retrieval.
- Retrieval thích nghi theo từng truy vấn.
- Personalization và preference memory.
- Gợi ý cold-start.
- Temporal hoặc sequential RAG.
- Faithfulness và giảm hallucination.
- Multi-objective reranking.
- Learning-to-retrieve hoặc reinforcement learning.
- Tối ưu chi phí, token và latency.
- Diversity, fairness hoặc giảm popularity bias.

## 5. Đề tài nên ưu tiên

Nếu cần một đề tài cân bằng giữa tính khả thi và khả năng viết thành bài nghiên cứu, mình đề xuất **Đề tài 1: Evidence-grounded RAG–LLM cho gợi ý cá nhân hóa có giải thích**.

Đề tài này có ưu điểm:

- Dễ xây dựng baseline từ MovieLens, Amazon hoặc Yelp.
- Có thể chạy bằng LLM mã nguồn mở cỡ nhỏ.
- Kết hợp được các nội dung cốt lõi của môn Hệ gợi ý: collaborative filtering, ranking, personalization và evaluation.
- Có đóng góp nghiên cứu rõ: chất lượng retrieval, reranking và độ trung thực của giải thích.
- Không phụ thuộc quá nhiều vào GPU nếu đóng băng LLM và chỉ thực hiện inference.