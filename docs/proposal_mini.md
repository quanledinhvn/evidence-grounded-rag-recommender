# ĐỀ XUẤT ĐỀ TÀI

## Hệ gợi ý phim và chương trình truyền hình có giải thích bằng RAG và LLM

### 1. Mô tả bài toán thực tế

Các nền tảng phim và chương trình truyền hình có danh mục rất lớn. Người dùng thường khó tìm được nội dung phù hợp nếu chỉ tìm kiếm bằng tên hoặc thể loại. Vì vậy, hệ thống cần dựa vào lịch sử đánh giá để đưa ra danh sách gợi ý riêng cho từng người.

Tuy nhiên, một danh sách gợi ý đơn thuần chưa giúp người dùng hiểu vì sao sản phẩm đó phù hợp. Nếu dùng mô hình ngôn ngữ lớn để tạo lời giải thích, mô hình có thể đưa ra thông tin không có trong dữ liệu. Đề tài giải quyết hai yêu cầu chính: tìm đúng sản phẩm phù hợp và giải thích dựa trên bằng chứng có sẵn.

Trong một kịch bản sử dụng, người dùng đã đánh giá một số phim hoặc chương trình. Hệ thống phân tích lịch sử này, tìm các sản phẩm chưa được người dùng tương tác và trả về 10 gợi ý. Ba gợi ý đầu tiên có lời giải thích ngắn kèm nguồn thông tin liên quan.

Mục đích của đề tài là xây dựng một hệ gợi ý Movies and TV nhỏ gọn, dễ chạy thử và có khả năng tạo kết quả rõ ràng, đáng tin cậy.

### 2. Dữ liệu và phạm vi

Đề tài sử dụng ba split Amazon Reviews 2023, nhóm Movies and TV, trong `dataset/2023/`. Dữ liệu hiện có gồm mã người dùng, `parent_asin`, rating, timestamp, lịch sử item và đã được chia train, validation, test.

Ba CSV chưa chứa metadata hoặc review text. Do đó phần CF có thể chạy ngay; BM25 và lời giải thích có căn cứ cần chờ metadata/review Movies and TV tương thích được bổ sung và kiểm tra độ phủ.

Hệ thống chỉ gợi ý các sản phẩm có trong dữ liệu và chưa được người dùng tương tác. Các nhận xét dùng làm bằng chứng phải có trước thời điểm cần đưa ra gợi ý để hạn chế việc sử dụng thông tin không phù hợp.

Trong phạm vi đề tài, nhóm không huấn luyện lại LLM, không xây dựng giao diện web hoàn chỉnh và không khảo sát người dùng thật. Trọng tâm là xây dựng quy trình gợi ý và tạo lời giải thích có căn cứ.

### 3. Phương pháp thực hiện

Quy trình của hệ thống gồm bốn bước.

1. Tạo hồ sơ sở thích từ những sản phẩm người dùng đã đánh giá cao và thông tin trong các nhận xét liên quan.

2. Dùng CF để tìm sản phẩm dựa trên lịch sử tương tác. Đồng thời, dùng BM25 để tìm sản phẩm có nội dung phù hợp với sở thích của người dùng. Kết quả từ hai phương pháp được kết hợp thành một danh sách ứng viên.

3. Cung cấp hồ sơ người dùng, danh sách ứng viên và thông tin liên quan cho LLM. LLM sắp xếp lại các ứng viên và chọn ra 10 sản phẩm phù hợp nhất.

4. Tạo lời giải thích cho ba sản phẩm đầu tiên. Mỗi lời giải thích phải dựa trên thông tin sản phẩm hoặc nhận xét đã được hệ thống tìm thấy.

### 4. Đầu vào và đầu ra

Đầu vào của hệ thống gồm mã người dùng, lịch sử đánh giá, thông tin sản phẩm và các nhận xét hợp lệ.

Đầu ra là danh sách 10 phim hoặc chương trình được gợi ý. Khi có kho bằng chứng hợp lệ, ba sản phẩm đầu có lời giải thích ngắn và thông tin làm bằng chứng. Kết quả được lưu dưới dạng JSON hoặc CSV để thuận tiện cho việc kiểm tra và viết báo cáo.

### 5. Kết quả kỳ vọng

Đề tài dự kiến tạo ra một hệ thống có thể kết hợp lịch sử tương tác và nội dung văn bản để gợi ý phim và chương trình truyền hình. Hệ thống không chỉ đưa ra sản phẩm mà còn giải thích lý do lựa chọn dựa trên dữ liệu đã tìm thấy.

Sản phẩm cuối gồm mã nguồn, dữ liệu đã xử lý, cấu hình chạy thử và một bản minh họa kết quả gợi ý. Kết quả của đề tài có thể làm cơ sở để phát triển hệ gợi ý rõ ràng và đáng tin cậy hơn.

### Tài liệu tham khảo

1. Liu, J. và cộng sự. *LLMRec: Benchmarking Large Language Models on Recommendation Task*, 2023. <https://arxiv.org/abs/2308.12241>

2. Amazon Reviews 2023, Movies and TV. <https://amazon-reviews-2023.github.io/main.html>

### Phụ lục: Giải thích thuật ngữ

* RAG: Retrieval Augmented Generation

* LLM: Large Language Model

* CF: Collaborative Filtering

* BM25: Best Matching 25
