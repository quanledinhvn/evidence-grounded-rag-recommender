# ĐỀ XUẤT ĐỀ TÀI

## Evidence-grounded RAG–LLM cho gợi ý âm nhạc cá nhân hóa có giải thích

### 1. Bài toán và mục tiêu

Các nền tảng âm nhạc cần chọn một số ít sản phẩm phù hợp từ danh mục lớn. Lọc cộng tác (collaborative filtering – CF) học được sở thích từ lịch sử đánh giá nhưng khó giải thích kết quả; mô hình ngôn ngữ lớn (LLM) giải thích tự nhiên nhưng có thể sinh thông tin không có trong dữ liệu.

Đề tài xây dựng hệ gợi ý âm nhạc theo mô hình **retrieval-augmented generation (RAG)**, kết hợp CF với metadata và review. Với lịch sử đánh giá của một người dùng, hệ thống trả về 10 sản phẩm chưa tương tác và giải thích có dẫn chứng cho ba sản phẩm đầu.

Mục tiêu là xây dựng pipeline tái lập gồm truy xuất, reranking và sinh giải thích; đánh giá tác động của hybrid retrieval lên Recall/NDCG; và kiểm tra liệu prompt bắt buộc trích dẫn có giảm nhận định thiếu căn cứ hay không.

Câu hỏi nghiên cứu là: **hybrid retrieval có cải thiện chất lượng top-\(K\), và evidence-grounded prompting có tạo giải thích trung thực hơn RAG thông thường trong ngân sách tính toán giới hạn hay không?**

### 2. Dữ liệu và phạm vi

Đề tài sử dụng **Amazon Review Data (2018) – Digital Music 5-core**, gồm khoảng 169.781 review cùng rating, timestamp và mã người dùng/sản phẩm. Mỗi người dùng và sản phẩm có ít nhất năm tương tác, đủ cho CF nhưng vẫn vừa sức trong ba tuần.

Với mỗi người dùng, dữ liệu được sắp theo thời gian: tương tác cuối dùng làm test, tương tác liền trước làm validation và phần còn lại dùng để huấn luyện. Hồ sơ người dùng và kho bằng chứng chỉ sử dụng dữ liệu có trước thời điểm cần dự đoán. Review do chính người dùng viết cho item test không được đưa vào prompt để tránh rò rỉ dữ liệu.

Đề tài không fine-tune LLM, xây giao diện web hoàn chỉnh hay đánh giá với người dùng thật. LLM chỉ xử lý tập ứng viên nhỏ và không được sinh item ngoài danh mục truy xuất.

### 3. Phương pháp dự kiến

Hệ thống gồm bốn bước:

1. **Tạo hồ sơ sở thích:** tổng hợp các sản phẩm người dùng đánh giá cao và từ khóa quan trọng trong metadata/review hợp lệ.
2. **Truy xuất ứng viên:** mô hình CF (ưu tiên BPR-MF, ItemKNN là phương án dự phòng) lấy top-50 sản phẩm chưa tương tác. BM25 truy xuất thêm các sản phẩm phù hợp với hồ sơ văn bản. Hai điểm số được chuẩn hóa rồi kết hợp theo trọng số \(\alpha\).
3. **LLM reranking:** LLM nhận hồ sơ, top-20 ứng viên và các đoạn bằng chứng liên quan, sau đó sắp xếp thành top-10. Prompt yêu cầu mô hình chỉ chọn item trong đầu vào và xuất JSON theo schema cố định.
4. **Giải thích có căn cứ:** với ba item đầu, LLM tạo giải thích ngắn kèm mã bằng chứng như `[E1]`. Bộ hậu xử lý phát hiện sai schema, item ngoài danh mục hoặc mã trích dẫn không tồn tại.

Các cấu hình thực nghiệm gồm Popularity, CF, BM25, CF + BM25, CF + BM25 + LLM và mô hình đầy đủ có trích dẫn. Ablation study lần lượt bỏ tín hiệu văn bản và ràng buộc bằng chứng để xác định đóng góp của từng thành phần.

### 4. Đầu vào và đầu ra

**Đầu vào:** mã người dùng; lịch sử tương tác gồm mã sản phẩm, rating và timestamp; metadata sản phẩm; review trong tập huấn luyện; các tham số top-\(K\), số ứng viên và trọng số \(\alpha\).

**Đầu ra:** danh sách top-10 sản phẩm và thứ hạng; giải thích cho ba sản phẩm đầu; các mã bằng chứng được trích dẫn; latency và số token. Kết quả được lưu ở JSON/CSV để đánh giá và tái lập thí nghiệm.

### 5. Phương pháp đánh giá

Chất lượng gợi ý được đo bằng Recall@10, NDCG@10 và Hit Rate@10. Candidate Recall@20/50 được báo cáo riêng để phân biệt lỗi retrieval với lỗi reranking. Baseline không dùng LLM chạy trên toàn bộ test; các cấu hình dùng LLM chạy trên cùng một mẫu cố định khoảng 200 người dùng, với seed được công bố, nhằm giới hạn thời gian và chi phí.

Chất lượng giải thích được đo bằng tỷ lệ trích dẫn hợp lệ, item ngoài danh mục và nhận định được bằng chứng hỗ trợ. Chỉ số cuối dùng prompt chấm cố định và kiểm tra thủ công 30–50 kết quả. Hệ thống cũng báo cáo latency và số token.

### 6. Kế hoạch ba tuần

| Thời gian | Công việc và kết quả |
|---|---|
| Tuần 1 | Làm sạch và chia dữ liệu; cài đặt Popularity, ItemKNN/BPR-MF; hoàn thành script Recall/NDCG. |
| Tuần 2 | Xây BM25, hybrid retrieval và kho bằng chứng; chạy so sánh và ablation cho các retriever. |
| Tuần 3 | Tích hợp LLM reranking và giải thích; kiểm tra schema/trích dẫn; chạy thí nghiệm, phân tích và hoàn thiện báo cáo/demo. |

Mốc tối thiểu là pipeline CF + BM25 + giải thích có trích dẫn và bảng so sánh baseline. Nếu tài nguyên hạn chế, số người dùng đánh giá bằng LLM sẽ giảm nhưng giao thức và seed không thay đổi. Dense retrieval hoặc tối ưu sâu trọng số \(\alpha\) chỉ là phần mở rộng.

### 7. Kết quả kỳ vọng

Sản phẩm cuối gồm mã nguồn, cấu hình tái lập, các baseline, mô hình hybrid RAG–LLM, bảng so sánh độ chính xác–độ trung thực–chi phí và demo trả về top-10 kèm giải thích có dẫn chứng. Đóng góp chính là kiểm chứng tác động của hybrid retrieval và ràng buộc bằng chứng trong một hệ gợi ý nhỏ gọn.

### Tài liệu tham khảo

1. Liu, J. và cộng sự. *LLMRec: Benchmarking Large Language Models on Recommendation Task*, 2023. <https://arxiv.org/abs/2308.12241>
2. Ni, J. và cộng sự. *Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects*, EMNLP 2019.
3. Amazon Review Data (2018), Digital Music 5-core. <https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/>
