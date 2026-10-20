# CF Training — BPR-MF

## 1. Mục tiêu và phạm vi

CF Training huấn luyện mô hình lọc cộng tác từ lịch sử tương tác user–item. Mô hình chính là **Bayesian Personalized Ranking Matrix Factorization (BPR-MF)**, dùng để xếp hạng các item mà user chưa tương tác.

Bước này chỉ chịu trách nhiệm:

- học biểu diễn ẩn của user và item từ tập dữ liệu được phép sử dụng;
- chấm điểm và trả về tối đa 50 item chưa thấy cho mỗi user;
- xuất model, ánh xạ ID và cấu hình huấn luyện thành artifact có thể tái lập.

Bước này không sử dụng metadata hoặc nội dung review, không gọi BM25 hay LLM, và không tạo lời giải thích. Popularity được xây riêng để làm baseline và fallback cho user không có tín hiệu BPR hợp lệ.

---

## 2. Đầu vào và đầu ra

### 2.1. Đầu vào

CF Training nhận artifact do `import-splits` tạo từ Movies and TV 2023 với schema tối thiểu:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `user_id` | string | ID user đã chuẩn hóa |
| `item_id` | string | ID item đã chuẩn hóa từ `parent_asin` |
| `rating` | float | Rating hữu hạn trong khoảng từ 1 đến 5 |
| `timestamp` | integer | Thời điểm tương tác |
| `split` | string | Một trong `train`, `validation`, `test` |

Dữ liệu phải thỏa mãn:

- mỗi cặp `(user_id, item_id)` trong tập fit phải là duy nhất;
- mọi ID đều không rỗng và thuộc catalog đã chốt;
- tập validation và test không được đọc trong lúc fit bundle validation;
- positive pair được sắp trước khi tạo CSR; với `num_threads=1`, cùng seed và cấu
  hình phải cho cùng factor. Chế độ nhiều thread ưu tiên tốc độ và không cam kết
  tái lập bit-for-bit vì các cập nhật SGD chạy đồng thời.

### 2.2. Đầu ra

CF Training xuất tối thiểu:

| Artifact | Nội dung |
|---|---|
| `bpr_model.npz` | Ma trận embedding user, embedding item và item bias nếu implementation sử dụng |
| `user_mapping.json` | Ánh xạ hai chiều giữa `user_id` và chỉ số nội bộ |
| `item_mapping.json` | Ánh xạ hai chiều giữa `item_id` và chỉ số nội bộ |
| `bpr_config.json` | Siêu tham số, seed, số epoch và quy tắc sampling |
| `training_metrics.csv` | Train AUC, số mẫu đúng/bị bỏ qua và thời gian theo epoch |
| `popularity.parquet` | Thứ hạng item theo số positive để làm baseline và fallback; item count 0 nằm cuối |
| `manifest.json` | Phiên bản schema, phase, hash dữ liệu, catalog, config và code |

Artifact chỉ hợp lệ khi mapping, model và catalog có hash khớp với cùng một manifest.

---

## 3. Tín hiệu học

### 3.1. Tương tác tích cực

Với user $u$, tập item tích cực là:

$$
P_u = \{i \mid (u,i) \text{ thuộc tập fit và } rating_{ui} \ge 4\}.
$$

Rating bằng 3 là trung tính. Rating nhỏ hơn hoặc bằng 2 là tín hiệu không thích. Hai nhóm này không được coi là positive khi huấn luyện BPR.

Mọi item đã xuất hiện trong lịch sử hợp lệ của user vẫn thuộc tập `seen_ids`, kể cả item có rating thấp. Vì vậy hệ thống không được gợi ý lại các item này ở runtime.

### 3.2. Negative sampling

Với mỗi positive pair $(u,i)$, backend `implicit` lấy mẫu một item $j$ không
thuộc tập positive của user trong tập đang fit:

$$
j \in C \setminus P_u,
$$

trong đó $C$ là catalog đã chốt. Item có rating không vượt quá 3 không phải
positive và có thể được lấy làm negative. Dù vậy, mọi item đã tương tác vẫn bị
lọc khỏi kết quả recommendation bằng `seen_by_user`.

Không được đọc validation hoặc test để loại target khỏi tập negative. Việc dùng held-out target khi sampling, kể cả chỉ để tránh lấy target làm negative, cũng làm cho quá trình train phụ thuộc vào dữ liệu tương lai.

Sampling phải dùng seed được ghi trong config. Phiên bản đầu sử dụng uniform negative sampling; popularity-based hoặc hard-negative sampling chỉ được thêm như một cấu hình thí nghiệm riêng.

### 3.3. User và item không học được embedding

- User không có positive trong tập fit không được dùng để cập nhật BPR và sử dụng popularity fallback khi truy vấn.
- Item không xuất hiện trong positive training data không có điểm BPR đáng tin cậy.
- Không gán điểm CF giả cho user hoặc item không có embedding.
- Các item này vẫn có thể được lấy từ BM25 hoặc popularity nếu thỏa mãn điều kiện của nhánh tương ứng.

---

## 4. Mô hình BPR-MF

### 4.1. Hàm chấm điểm

Mỗi user $u$ có vector ẩn $p_u \in \mathbb{R}^d$ và mỗi item $i$ có vector ẩn $q_i \in \mathbb{R}^d$. Điểm CF cơ bản là:

$$
\hat{x}_{ui} = p_u^Tq_i + b_i,
$$

trong đó $b_i$ là item bias. Nếu thư viện được chọn không sử dụng item bias, quyết định này phải được ghi trong `bpr_config.json` và áp dụng nhất quán giữa train và inference.

BPR không diễn giải $\hat{x}_{ui}$ là rating dự đoán. Điểm chỉ có ý nghĩa để so sánh thứ hạng các item của cùng một user.

### 4.2. Pairwise ranking

Với bộ ba $(u,i,j)$ gồm positive item $i$ và sampled negative item $j$, độ chênh điểm là:

$$
\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}.
$$

Mục tiêu là làm cho positive item có điểm cao hơn negative item. Loss tối thiểu hóa là:

$$
\mathcal{L}_{BPR}
= -\sum_{(u,i,j)} \log \sigma(\hat{x}_{uij})
+ \lambda \lVert \Theta \rVert_2^2,
$$

trong đó $\sigma$ là sigmoid, $\Theta$ là toàn bộ tham số và $\lambda$ là hệ số regularization.

### 4.3. Khởi tạo và cập nhật

- `implicit.cpu.bpr.BayesianPersonalizedRanking` nhận CSR user–item nhị phân chỉ
  chứa positive interactions.
- Embedding và negative sampling dùng seed đã khai báo.
- `verify_negative_samples=true` ngăn positive item được dùng làm negative.
- Backend Cython/OpenMP chạy trên toàn bộ CPU khi `num_threads=0`; đặt bằng 1
  cho thí nghiệm cần tái lập bit-for-bit.
- Implementation, phiên bản thư viện, optimizer, learning rate và quy tắc
  regularization được lưu trong config.

---

## 5. Hai pha huấn luyện

### 5.1. Pha validation

Bundle validation chỉ được fit bằng split `train`:

1. Xây mapping từ catalog và dữ liệu train.
2. Tạo positive pairs từ các dòng train có rating từ 4 trở lên.
3. Fit BPR-MF và popularity trên train.
4. Sinh recommendation cho validation mà không đưa validation target vào service.
5. Evaluator ghép prediction với target sau inference.

Pha này dùng để chọn siêu tham số BPR và cấu hình retrieval. Tuning dùng một
mẫu cố định 50.000 validation query với seed được khóa. Model thắng được báo
cáo lại trên toàn bộ validation. Không được chọn epoch hoặc siêu tham số dựa
trên kết quả test.

Implementation ghi toàn bộ candidate validation vào `validation_metrics.csv`.
Cấu hình thắng được ghi vào `locked_config.json`, kèm hash của config, split,
catalog và validation queries, trước khi bắt đầu fit bundle test.

### 5.2. Pha test

Sau khi khóa cấu hình bằng validation:

1. Gộp train và validation thành tập fit mới.
2. Fit lại BPR-MF và popularity từ đầu với cấu hình đã khóa.
3. Không tiếp tục tuning bằng test.
4. Test target chỉ được evaluator đọc sau inference.

Model test phải nằm trong bundle khác model validation và có phase/hash riêng trong manifest.

---

## 6. Cấu hình huấn luyện

Cấu hình khởi đầu đề xuất:

| Tham số | Giá trị khởi đầu | Không gian validation đề xuất |
|---|---:|---|
| `factors` | 64 | 32, 64 |
| `learning_rate` | 0.01 | Giữ cố định |
| `regularization` | 0.001 | Giữ cố định |
| `epochs` | 60 | 30, 60 |
| `negative_sampling` | Uniform non-positive | Giữ cố định |
| `num_threads` | 0 (mọi core) | 1 khi cần tái lập bit-for-bit |
| `verify_negative_samples` | true | Giữ true trong thí nghiệm chính |
| `seed` | 42 | Cố định |

Grid chính có bốn candidate từ tích của hai giá trị `factors` và hai giá trị
`epochs`. Giữ nguyên learning rate, regularization, sampling và seed để giới
hạn chi phí và số biến thí nghiệm.

Checkpoint được chọn theo NDCG@10 trên validation. Khi hai cấu hình có NDCG@10 bằng nhau trong sai số báo cáo, ưu tiên lần lượt Candidate Recall@50 cao hơn, model nhỏ hơn và thời gian truy vấn thấp hơn.

CLI biểu diễn checkpoint bằng các giá trị trong `--epoch-candidates`. Mỗi giá
trị epoch được fit lại từ cùng seed, vì vậy model tại epoch đó tương đương
checkpoint tương ứng trong cùng cấu hình optimizer và sampling.

Evaluator cache candidate matrix/index và chấm user theo batch 256. Full metric
và positive-target slice được tích lũy trong cùng một lượt BPR. Popularity
không chạy trong vòng tuning; baseline này chỉ được đánh giá một lần trên full
validation và một lần trên full test.

---

## 7. Hợp đồng inference

Với một `user_id`, CF retriever thực hiện:

1. Kiểm tra user có mapping và embedding hợp lệ.
2. Chấm điểm các item có embedding thuộc catalog hiện hành.
3. Loại toàn bộ `seen_ids` trước khi chọn top-N.
4. Sắp điểm giảm dần; nếu bằng điểm, phá hòa bằng `item_id` tăng dần.
5. Trả tối đa 50 cặp `(item_id, cf_score)`.

Các điều kiện bổ sung:

- `cf_score` là điểm model gốc, không phải xác suất hay rating dự đoán;
- một item chỉ xuất hiện tối đa một lần;
- mọi item trả về phải thuộc catalog và chưa xuất hiện trong history hợp lệ;
- nếu còn ít hơn 50 item hợp lệ thì trả toàn bộ số item còn lại;
- user lạ hoặc không có positive history chuyển sang popularity fallback và được gắn trạng thái `cold_start`.

Điểm BPR chỉ được dùng để sắp xếp trong nhánh CF. Khi kết hợp với BM25, hệ thống dùng rank fusion đã quy định; không cộng trực tiếp điểm BPR với điểm BM25 khác thang đo.

---

## 8. Đánh giá

### 8.1. Chỉ số chính

| Chỉ số | Mục đích |
|---|---|
| NDCG@10 | Đo target xuất hiện cao hay thấp trong top 10 |
| HR@10 | Đo tỷ lệ query có target trong top 10 |
| Candidate Recall@50 | Đo khả năng CF đưa target vào candidate pool |
| Coverage | Đo tỷ lệ query và item mà CF có thể phục vụ |
| Fallback rate | Đo tỷ lệ query phải chuyển từ BPR sang popularity |
| Latency | Đo mean, p50 và p95 thời gian chấm điểm và lấy top 50 |
| Model size | Đo dung lượng artifact trên đĩa và RAM khi load |

Với mỗi user chỉ có một target, HR@10 bằng Recall@10. Báo kết quả trên toàn bộ target theo giao thức chính và báo thêm một lát cắt target có rating từ 4 trở lên để phản ánh đúng tín hiệu mà BPR được huấn luyện.

Evaluator luôn truy xuất pool 50. Giá trị nhỏ hơn không được chấp nhận vì sẽ
làm chỉ số Candidate Recall@50 bị gắn nhãn sai cutoff.

### 8.2. Baseline bắt buộc

BPR-MF phải được so sánh ít nhất với:

- Popularity, để đo lợi ích của cá nhân hóa và làm fallback cold-start.

BM25 và hybrid được đánh giá ở bước retrieval riêng. Không dùng kết quả LLM để kết luận chất lượng riêng của BPR-MF.

---

## 9. Kiểm tra bắt buộc

### 9.1. Dữ liệu và leakage

- Fit bundle validation chỉ đọc train interactions.
- Fit bundle test chỉ đọc train và validation sau khi config đã khóa.
- Negative sampler không đọc held-out interactions.
- Test target không xuất hiện trong training metrics, checkpoint selection hoặc early stopping.
- Hash split và catalog trong model phải khớp manifest.

### 9.2. Tính đúng của model

- Với cùng input, config và seed, mapping và prediction phải tái lập.
- Positive và sampled negative của cùng bộ ba phải khác nhau.
- Sampled negative không thuộc lịch sử của user trong tập fit.
- User không có positive không được tạo recommendation BPR giả.
- Item không có embedding không được nhận `cf_score` giả.

### 9.3. Tính đúng của recommendation

- Không trả item đã thấy, kể cả item có rating thấp.
- Không trả item ngoài catalog hoặc ID không tồn tại trong mapping.
- Không có item trùng trong top-N.
- Kết quả được sắp giảm dần theo score và phá hòa xác định.
- Top-N vẫn đúng khi catalog còn ít hơn N item hợp lệ.

---

## 10. Điều kiện hoàn thành

CF Training được coi là hoàn thành khi:

- tạo được bundle validation từ train mà không đọc validation/test target;
- BPR-MF trả được top 50 item chưa thấy cho user hợp lệ;
- popularity xử lý được user cold-start;
- evaluator xuất NDCG@10, HR@10, Candidate Recall@50, coverage và latency;
- có bảng so sánh BPR-MF với Popularity;
- cấu hình được khóa trên validation rồi refit thành bundle test;
- artifact có manifest/hash và có thể load lại để tái tạo cùng prediction trong sai số số thực cho phép.
