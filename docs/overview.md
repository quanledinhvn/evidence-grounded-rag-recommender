# Tổng quan đồ án: hệ gợi ý phim có giải thích dựa trên bằng chứng

> Tài liệu này mô tả cấu trúc tổng thể của đồ án ở mức dễ hiểu. Chi tiết hợp đồng từng module, schema và quy tắc chống rò rỉ nằm ở [architecture.md](architecture.md).

## 1. Đồ án làm gì

Gợi ý phim cho một người dùng **và giải thích vì sao gợi ý**, trong đó mỗi lý do đều trỏ về một bằng chứng có thật: metadata phim, review của người khác hoặc lịch sử của chính người dùng. LLM không được đưa ra lý do không có nguồn.

Có thể hình dung hệ thống như một nhân viên cửa hàng DVD làm ba việc:

1. **Nhớ khách:** "Anh này thích phim của Nolan, không thích phim kinh dị."
2. **Lục kệ:** tìm khoảng 50 phim có thể hợp.
3. **Tư vấn:** chọn 10 phim và nói "phim này cùng đạo diễn với Inception mà anh thích [H1], nhiều người khen cốt truyện nhiều tầng [E2]".

## 2. Sáu bước của hệ thống

```text
 ① Hồ sơ người dùng  →  ② Tìm ứng viên (3 nhánh)  →  ③ Trộn kết quả
                                                          │
 ⑥ Kiểm tra trích dẫn ←  ⑤ LLM xếp hạng + giải thích  ←  ④ Gom bằng chứng
```

| Bước | Làm gì | Ví dụ | Trạng thái |
|---|---|---|---|
| **① Hồ sơ** | Từ lịch sử *trước thời điểm gợi ý*, lấy phim đã thích hoặc không thích (phim gần đây có trọng số cao hơn) và review của chính người dùng | "Thích: Inception, Interstellar. Không thích: Saw" | Đã có (`backend/profile`), cần tăng tốc |
| **② Tìm ứng viên** | Ba nhánh độc lập, mỗi nhánh trả khoảng 100 phim | | |
| · CF | Dựa trên hành vi đám đông: "người xem Inception thường xem tiếp…" | Tenet | Đã có: ItemKNN là nhánh chính, BPR là baseline (mục 6) |
| · BM25 | Khớp **đúng từ**: cùng đạo diễn, diễn viên, thể loại | Phim khác của Nolan | Chưa có |
| · Embedding | Khớp **ý nghĩa**: cốt truyện, không khí tương tự dù khác từ ngữ | Arrival | Chưa có |
| **③ Trộn** | Gộp ba danh sách: phim được nhiều nhánh xếp cao thì lên đầu; cộng thêm tín hiệu phim đang được xem nhiều gần đây. Lấy top 50, rồi top 20 cho LLM | | Chưa có |
| **④ Bằng chứng** | Với mỗi phim trong top 20: đoạn metadata **[E]**, review của người khác viết *trước* thời điểm gợi ý **[E]**, lịch sử của người dùng **[H]** | [E2] "Plot twists keep you thinking…" | Chưa có |
| **⑤ LLM** | Xếp lại 20 phim thành top 10, rồi giải thích top 3. Chỉ được nói điều có trong bằng chứng, kèm mã [E#] hoặc [H#] | | Chưa có |
| **⑥ Kiểm tra** | Kiểm tra tự động: mã trích dẫn có tồn tại, đúng phim, tên người và thể loại được nhắc có xuất hiện trong bằng chứng. Câu sai bị loại; LLM lỗi thì dùng kết quả bước ③ | | Chưa có |

**Quy tắc xuyên suốt:** mọi dữ liệu dùng để gợi ý phải xảy ra trước thời điểm gợi ý ($timestamp < as\_of$). Không dùng thông tin từ tương lai.

## 3. Dữ liệu: hai tầng

| Tầng | Dùng cho | Lý do |
|---|---|---|
| **Dense-Works** (chính) | Toàn bộ sáu bước, kể cả LLM | Phim có tên và mô tả đầy đủ, người dùng có lịch sử dài, đủ nhẹ để chạy trên máy cá nhân |
| **5-core Amazon** (phụ, đã có trong `artifacts/movies-tv-2023`) | Chỉ bước ② và ③ | Chứng minh hệ thống chạy được ở quy mô lớn (657k người dùng) và trên split chuẩn |
| **Review thô** (`dataset/Movies_and_TV.jsonl`) | Kho bằng chứng ở bước ④ và review của người dùng ở bước ① | Hai tầng trên không có nội dung review |

**Dense-Works** là tên gọi tắt cho tập con được đề xuất trong khảo sát dữ liệu [data/raw-dataset-survey.md](data/raw-dataset-survey.md):

- **Works:** gộp các phiên bản của cùng một phim (DVD, Blu-ray, bản kỷ niệm, Prime Video) thành một *tác phẩm*, theo khóa title đã chuẩn hóa cộng đạo diễn đầu tiên. Ví dụ *The Big Lebowski* có 14 mã sản phẩm.
- **Dense:** lọc 20-core, tức chỉ giữ người dùng và tác phẩm có ít nhất 20 tương tác.
- Chỉ giữ item có title.

| | 5-core hiện tại | Dense-Works (theo khảo sát) |
|---|---|---|
| Đơn vị item | Mã sản phẩm (`parent_asin`) | Tác phẩm |
| Quy mô | 7,4 triệu tương tác, 657k người dùng, 198k item | Khoảng 787k tương tác, 18k người dùng, 13k tác phẩm |
| Lịch sử train mỗi người dùng (trung vị) | 5 | Khoảng 30 |
| Item có title | 58% | 100% |
| Gợi ý trùng phiên bản | Có | Không |
| Nguồn | Split có sẵn của Amazon Reviews 2023 | Tự xử lý từ `dataset/*.jsonl`, chia leave-last-out |

Số liệu của Dense-Works lấy từ khảo sát và cần kiểm chứng lại khi viết lại script tiền xử lý.

**Hạn chế cần nêu trong báo cáo:** Dense-Works chỉ chứa người dùng tích cực (ít nhất 20 review), trong khi 58% người dùng gốc chỉ có một review; việc gộp phiên bản dùng luật heuristic nên có thể gộp nhầm (các mùa của cùng một chương trình TV) hoặc bỏ sót.

## 4. Thí nghiệm

Mỗi run thêm một thành phần để đo đóng góp của thành phần đó.

| Run | Thành phần | Câu hỏi trả lời |
|---|---|---|
| A | Popularity (toàn cục và gần đây) | Mốc thấp nhất |
| B | CF (ItemKNN; BPR để so sánh) | Hành vi đám đông giúp được bao nhiêu? |
| C | BM25 | Khớp từ khóa metadata giúp được bao nhiêu? |
| D | Embedding | Khớp ngữ nghĩa có hơn khớp từ khóa? |
| E | Trộn B + C + D, kèm các bản bỏ từng nhánh | Kết hợp có tốt hơn nhánh tốt nhất? |
| F | E + LLM xếp hạng lại | LLM có xếp hạng tốt hơn công thức trộn? |
| G | F + giải thích bắt buộc trích dẫn và kiểm tra | Giải thích có đáng tin hơn so với để LLM nói tự do? |

Run A–E chạy trên toàn bộ tập test của cả hai tầng; F và G chạy trên một mẫu người dùng cố định của Dense-Works vì tốn chi phí LLM.

Thước đo:

- **Gợi ý có trúng không:** NDCG@10, HR@10, Recall@50.
- **Giải thích có trung thực không:** tỉ lệ trích dẫn đúng, tỉ lệ câu được bằng chứng ủng hộ (chấm tự động cộng chấm tay 30–50 mẫu), tỉ lệ từ chối giải thích.
- **Vận hành:** số token, độ trễ, tỉ lệ fallback.

## 5. Tổ chức code

```text
backend/
  data/        # đọc, làm sạch, chia tập, gộp phiên bản, import review   (đã có một phần)
  profile/     # ① hồ sơ người dùng                                        (đã có)
  cf/          # ② CF: BPR (đã có), ItemKNN, recent popularity
  retrieval/   # ② bm25.py, dense.py   ③ fusion.py
  evidence/    # ④ kho review (SQLite) và chọn bằng chứng
  generation/  # ⑤ gọi LLM, prompt   ⑥ validator
  eval/        # chấm điểm chung cho mọi run A–G
  service.py   # nối ①→⑥ thành một luồng; CLI và API dùng chung
configs/       # cấu hình từng tầng dữ liệu, trọng số trộn, ngân sách LLM
artifacts/     # dữ liệu đã xử lý (không đưa lên Git)
runs/          # kết quả từng lần chạy (không đưa lên Git)
```

## 6. Hiện trạng CF và điều chỉnh

Kết quả trên **toàn bộ 657.203 query test** của tầng 5-core:

| Phương pháp | NDCG@10 | HR@10 | Recall@50 | Nguồn |
|---|---:|---:|---:|---|
| Popularity toàn cục | 0,0091 | 0,0171 | 0,045 | `runs/cf-movies-tv-2023-full/20261009T135249Z/evaluation.json` |
| BPR-MF (64 factors, 60 epoch) | 0,0132 | 0,0228 | 0,045 | như trên |
| **ItemKNN** (400 hàng xóm, decay 0,7) | **0,0491** | **0,0768** | **0,119** | `runs/itemknn-movies-tv-2023/test_evaluation.json` |

ItemKNN (`backend/cf/itemknn.py`, xem [backend/cf/README.md](../backend/cf/README.md)) tính cosine giữa các item và cộng điểm theo lịch sử người dùng, item càng gần đây trọng số càng lớn: $\gamma^{k}$ với $k$ là vị trí tính từ tương tác mới nhất. Số hàng xóm và $\gamma$ được chọn trên 20.000 query validation (`runs/itemknn-movies-tv-2023/validation_metrics.csv`):

| Hàng xóm \ $\gamma$ | 0,3 | 0,5 | 0,7 | 1,0 (không dùng độ gần đây) |
|---|---:|---:|---:|---:|
| 100 | 0,0568 | 0,0585 | 0,0581 | 0,0482 |
| 400 | 0,0569 | 0,0586 | **0,0586** | 0,0489 |
| 1600 | 0,0572 | 0,0590 | 0,0591 | 0,0493 |

Giá trị trong bảng là NDCG@10. Cấu hình 400/0,7 có Recall@50 = 0,134, gần bằng 1600/0,7 (0,138) nhưng ma trận tương đồng nhỏ hơn 2,4 lần. Bỏ trọng số độ gần đây làm NDCG giảm khoảng 17%.

Hai tín hiệu bổ sung, đo trên 20.000 query (seed 42):

- **Recent popularity 30 ngày** (chỉ đếm tương tác trước thời điểm gợi ý) đạt NDCG@10 0,032 trên mẫu test, gấp khoảng 2,5 lần BPR, dù không cá nhân hóa.
- **ItemKNN cộng prior popularity** trên validation, với điểm cuối $score + \lambda \cdot prior$ và $prior \in [0, 1]$ theo thứ hạng popularity:

| $\lambda$ | 0 | 0,03 | 0,1 | 0,3 |
|---|---:|---:|---:|---:|
| Prior popularity toàn cục: NDCG@10 / Recall@50 | 0,0585 / 0,134 | 0,0623 / 0,151 | 0,0669 / 0,171 | **0,0688 / 0,179** |
| Prior popularity 30 ngày: NDCG@10 / Recall@50 | 0,0585 / 0,134 | 0,0619 / 0,150 | 0,0653 / 0,165 | 0,0662 / 0,175 |

Việc trộn này thuộc bước ③ (fusion) và cần tune lại cùng BM25 và embedding.

Nhận xét:

- ItemKNN gấp 3,7 lần BPR về NDCG@10 và 2,6 lần về Recall@50.
- BPR hiện tại chỉ thắng Popularity toàn cục; Recall@50 bằng Popularity nên gần như không đóng góp thêm ứng viên cho bước trộn.
- Dữ liệu phụ thuộc mạnh vào thời gian và thứ tự: test là tương tác *kế tiếp*, người dùng hay xem tiếp mùa sau, cùng loạt phim hoặc phim đang được xem nhiều. BPR coi lịch sử là tập hợp không thứ tự nên bỏ lỡ tín hiệu này.
- BPR chưa hội tụ: train AUC vẫn tăng ở epoch 60 và cấu hình thắng nằm ở biên của grid.

Điều chỉnh:

- Dùng **ItemKNN có trọng số theo độ gần đây** làm nhánh CF chính (đã cài đặt); giữ BPR làm baseline. Mỗi gợi ý của ItemKNN kèm trường `because` liệt kê item trong lịch sử đóng góp điểm, là bằng chứng dạng "vì bạn đã xem X" cho bước giải thích.
- Đưa popularity (toàn cục và 30 ngày) vào bước trộn như một nhánh riêng và tune trọng số trên validation.
- Nếu tiếp tục dùng BPR: train lâu hơn (200–500 epoch), learning rate 0,05–0,1, 128 factors; thử thêm ALS. Trên Dense-Works có thể thử EASE và SASRec.

**Lưu ý đánh giá:** ItemKNN và BPR đều học từ tương tác của người dùng khác xảy ra sau thời điểm gợi ý. Đây là giới hạn đã nêu trong [architecture.md](architecture.md) mục 3.2, nên kết quả của hai phương pháp này có thể cao hơn một chút so với thiết lập hoàn toàn theo thời gian.

## 7. Khung báo cáo

1. **Giới thiệu:** vấn đề LLM đưa ra lý do không có thật trong hệ gợi ý; mục tiêu đồ án.
2. **Cơ sở lý thuyết:** CF (ItemKNN, BPR), BM25, embedding, fusion (RRF), RAG, citation.
3. **Dữ liệu:** Amazon Reviews 2023 Movies & TV, cách gộp phiên bản, hai tầng dữ liệu, quy tắc chống rò rỉ.
4. **Kiến trúc hệ thống:** sáu bước ở mục 2.
5. **Thực nghiệm:** run A–G, bảng kết quả, ablation.
6. **Phân tích:** ví dụ giải thích tốt và xấu, khi nào LLM giúp được, chi phí.
7. **Hạn chế và hướng phát triển:** thiên lệch về người dùng tích cực, giới hạn đánh giá theo thời gian; hướng mở rộng gồm giải thích phản thực tế ("nếu bạn không chấm thấp phim X thì phim Y đã ở top 1") và cổng chọn lọc chỉ gọi LLM khi cần (người dùng mới, item đuôi dài) để giảm token và độ trễ.

## 8. Thứ tự triển khai

1. **Nền dữ liệu:** tạo Dense-Works, import review, xây tài liệu mô tả item dùng chung cho BM25, embedding và bằng chứng.
2. **Bước ② và ③:** ItemKNN, BM25, embedding, trộn; tune trên validation rồi khóa cấu hình.
3. **Bước ④:** kho review và chọn bằng chứng.
4. **Bước ⑤ và ⑥:** LLM và validator.
5. **Chạy A–G và viết báo cáo.**

Mỗi giai đoạn đều cho kết quả đo được, nên nếu thiếu thời gian vẫn có một đồ án hoàn chỉnh ở mốc A–E.
