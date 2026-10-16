# Hướng dẫn Collaborative Filtering cho người mới

Tài liệu có hai phần. **Phần I** giải thích lý thuyết và ví dụ mà chưa yêu cầu người đọc biết cấu trúc code. **Phần II** mới ánh xạ từng khái niệm vào file, hàm và dòng code của dự án.

> Số dòng khớp với code tại thời điểm viết. Khi code thay đổi, hãy tìm tên hàm được ghi kèm thay vì chỉ dựa vào số dòng.
>
> Hướng dẫn này chỉ nói về BPR-MF và Popularity. ItemKNN, nhánh CF chính hiện nay, được mô tả trong [backend/cf/README.md](../../backend/cf/README.md#itemknn); kết quả so sánh nằm ở mục 6 của [overview.md](../overview.md).

# Phần I Lý thuyết nền tảng

## 1. Collaborative Filtering giải quyết bài toán gì

Collaborative Filtering, viết tắt là CF, gợi ý dựa trên mẫu tương tác của nhiều người dùng. Nếu hai người từng thích các item giống nhau, sở thích còn lại của họ có thể liên quan. CF không cần biết nội dung hoặc thể loại của item để học mẫu này.

Hãy hình dung dữ liệu như một bảng:

| User | M1 | M2 | M3 | M4 |
|---|---:|---:|---:|---:|
| An | 5 | 2 | ? | ? |
| Bình | 4 | ? | 5 | ? |
| Chi | ? | ? | ? | 3 |

Dấu `?` là tương tác chưa quan sát. Hệ gợi ý cần xếp hạng các item chưa quan sát, không nhất thiết phải đoán chính xác rating.

Trong dự án này:

- rating từ 4 trở lên là **positive**, tức tín hiệu thích dùng để học;
- mọi item đã tương tác là **seen**, kể cả rating thấp;
- item đã seen không được gợi ý lại;
- user không có positive không có đủ tín hiệu để dùng BPR.

Với An, M1 là positive; M2 không phải positive; nhưng cả M1 và M2 đều là seen. Candidate hợp lệ chỉ còn M3 và M4.

## 2. Matrix Factorization

Matrix Factorization, viết tắt là MF, thay mỗi user và item bằng một vector ngắn.

```text
An = [0.9, 0.2]
M3 = [0.8, 0.1]
M4 = [0.1, 0.9]
```

Vector của An gần M3 hơn M4, nên M3 nhận điểm cao hơn. Các chiều ẩn được học từ dữ liệu; không nên tự gán rằng một chiều chắc chắn là “hành động” hay “hài”.

Điểm của item $i$ đối với user $u$ là:

$$
\hat{x}_{ui} = p_u^T q_i + b_i
$$

**Phát biểu bằng lời:** điểm bằng độ phù hợp giữa vector user và vector item, cộng thiên lệch riêng của item.

Các ký hiệu:

- $u$ là user đang xét;
- $i$ là item đang xét;
- $p_u$ là vector ẩn của user $u$;
- $q_i$ là vector ẩn của item $i$;
- $p_u^Tq_i$ là tích vô hướng của hai vector;
- $b_i$ là item bias;
- $\hat{x}_{ui}$ là điểm xếp hạng, không phải rating hay xác suất.

Điểm 2.1 không có nghĩa user sẽ đánh giá 2.1 sao. Điểm chỉ dùng để so sánh các item của cùng user.

## 3. BPR học thứ tự

Bayesian Personalized Ranking, viết tắt là BPR, không cố dự đoán rating. Mỗi
mẫu học gồm user $u$, positive item $i$ và một item không thuộc tập positive
$j$.

$$
\hat{x}_{uij} = \hat{x}_{ui} - \hat{x}_{uj}
$$

**Phát biểu bằng lời:** lấy điểm positive trừ điểm negative. Mô hình muốn hiệu này dương và càng lớn càng tốt.

Trong đó $i$ là item positive, $j$ là sampled negative và $\hat{x}_{uij}$ là khoảng cách điểm giữa chúng.

Với An, một mẫu có thể là `(An, M1, M3)`. Mô hình được sửa để M1 đứng trên M3. Lần lấy mẫu khác có thể dùng M4.

### Loss

$$
\operatorname{loss}
= \log\left(1 + e^{-\hat{x}_{uij}}\right)
+ \lambda \lVert \Theta \rVert_2^2
$$

**Phát biểu bằng lời:** loss phạt khi positive chưa đứng đủ cao hơn negative, đồng thời phạt các tham số quá lớn.

- $\lambda$ là hệ số regularization;
- $\Theta$ là các vector và bias tham gia cập nhật;
- $\lVert\Theta\rVert_2^2$ là tổng bình phương các tham số đó.

Độ lớn gradient tương đương:

$$
g = \frac{1}{1 + e^{\hat{x}_{uij}}}
$$

**Phát biểu bằng lời:** nếu thứ tự đang sai thì $g$ lớn và bước sửa mạnh; nếu thứ tự đã đúng với khoảng cách lớn thì $g$ nhỏ.

Một bước học kéo user và positive gần nhau, đẩy negative ra xa, rồi dùng regularization để giữ tham số không tăng mất kiểm soát.

## 4. Negative sampling

BPR không so sánh một positive với toàn bộ catalog ở mỗi bước. Backend
`implicit` lấy ngẫu nhiên một item không thuộc tập positive:

$$
n_{available} = n_{catalog} - n_{positive}
$$

**Phát biểu bằng lời:** số negative khả dụng bằng số item trong catalog trừ số
item positive của user.

- $n_{catalog}$ là số item trong catalog;
- $n_{positive}$ là số item user đã rating từ 4 trở lên;
- $n_{available}=0$ nghĩa là không còn negative hợp lệ.

Rating thấp vẫn thuộc seen để lọc ở runtime, nhưng không thuộc ma trận positive
và có thể được backend lấy làm negative. “Negative” ở đây nghĩa là non-positive,
không nhất thiết khẳng định user ghét item đó.

## 5. Recommendation và cold start

Sau khi train, hệ thống:

1. chấm điểm các item có thể học bằng BPR;
2. loại mọi item đã seen;
3. xếp điểm giảm dần;
4. lấy tối đa 50 item.

| Item | Điểm BPR | An đã thấy | Được xét |
|---|---:|---|---|
| M1 | 2.10 | Có | Không |
| M2 | 1.80 | Có, dù rating thấp | Không |
| M3 | 1.35 | Không | Có |
| M4 | 0.90 | Không | Có |

Kết quả là M3 rồi M4. M1 không được gợi ý lại dù có điểm cao nhất.

User lạ hoặc user không có positive không có factor BPR đáng tin cậy. Hệ thống dùng Popularity và gắn trạng thái `cold_start` thay vì tạo điểm BPR giả.

## 6. Baseline

### Popularity

Popularity xếp item theo số interaction positive. Nó không cá nhân hóa, nhưng là mốc so sánh và fallback hợp lý cho cold start. Item cùng count được phá hòa bằng `item_id`.

## 7. Metric đánh giá

Mỗi query có một target bị giữ lại. Retriever sinh ranking mà không nhìn target; evaluator mới kiểm tra target nằm ở đâu.

### NDCG@10

Vì mỗi query chỉ có một target:

$$
\operatorname{NDCG@10}
= \frac{1}{Q}\sum_{q=1}^{Q}
\frac{\mathbb{1}[r_q \le 10]}{\log_2(r_q + 1)}
$$

**Phát biểu bằng lời:** target ngoài top 10 nhận 0; target trong top 10 nhận điểm giảm dần theo vị trí; sau đó lấy trung bình.

- $Q$ là số query;
- $r_q$ là thứ hạng bắt đầu từ 1 của target;
- $\mathbb{1}$ bằng 1 khi điều kiện đúng, ngược lại bằng 0;
- $\log_2$ là log cơ số 2.

Target ở hạng 1 đóng góp 1; hạng 3 đóng góp 0.5; ngoài top 10 đóng góp 0.

### HR@10

$$
\operatorname{HR@10}
= \frac{\text{số query có target trong top 10}}{Q}
$$

**Phát biểu bằng lời:** HR@10 chỉ hỏi target có xuất hiện trong top 10 không; hạng 1 và hạng 10 đều được tính như nhau.

### Candidate Recall@50

$$
\operatorname{CandidateRecall@50}
= \frac{\text{số query có target trong top 50}}{Q}
$$

**Phát biểu bằng lời:** metric đo khả năng retriever đưa target vào pool 50 item cho các bước sau.

### Coverage và latency

$$
\operatorname{ItemCoverage}
= \frac{\text{số item khác nhau từng được trả}}{\text{số item trong catalog}}
$$

**Phát biểu bằng lời:** item coverage cho biết recommendation phủ bao nhiêu phần catalog.

Query coverage của BPR là tỷ lệ query có đủ tín hiệu positive để BPR phục vụ. Fallback rate bằng một trừ query coverage.

Latency gồm mean, p50 và p95. p95 quan trọng vì nó biểu diễn phần đuôi chậm, không chỉ tốc độ trung bình.

## 8. Validation và test

```text
train
  -> fit các candidate
  -> đo bằng validation queries
  -> chọn và khóa cấu hình

train + validation
  -> fit lại từ đầu bằng cấu hình đã khóa
  -> đo một lần bằng test queries
```

Nếu test metric được dùng để chọn số factors, learning rate hoặc epochs, test không còn là phép đo khách quan.

# Phần II Từ lý thuyết đến code

## 9. Bản đồ module

| File | Vai trò |
|---|---|
| `backend/cf/cli.py` | Nhận lệnh, tạo grid, khóa cấu hình và chạy hai phase. |
| `backend/cf/tuning.py` | Train từng candidate và chọn candidate tốt nhất. |
| `backend/cf/bundle.py` | Chuẩn bị dữ liệu, lưu/nạp bundle và recommend. |
| `backend/cf/bpr.py` | Adapter CSR cho BPR CPU của `implicit`. |
| `backend/cf/baselines.py` | Cài đặt Popularity. |
| `backend/cf/evaluation.py` | Chạy ranker và tính metric. |

```text
cli._run_pipeline
  -> tuning.select_bpr_config
     -> bundle.train_bundle
        -> BPRModel.fit
           -> implicit.cpu.bpr.BayesianPersonalizedRanking.fit
     -> evaluation.compare_retrievers
  -> ghi locked_config.json
  -> train_bundle phase=test
  -> compare_retrievers trên test
```

Vị trí điều phối: `backend/cf/cli.py`, dòng 106–187.

## 10. Chuẩn bị dữ liệu trong `bundle.py`

### Catalog và split

- Dòng 218–239: từ chối `item_id` rỗng/trùng, sắp ID và kiểm tra catalog không rỗng.
- Dòng 241–247: phase `validation` chỉ đọc `train`; phase `test` đọc `train + validation`.
- Test chống leakage: `tests/test_cf_training.py`, dòng 95–153.

### Kiểm tra interaction

Hàm `_validate_rows`, `backend/cf/bundle.py`, dòng 335–362:

| Trường | Điều kiện | Dòng |
|---|---|---:|
| `user_id` | Không rỗng | 341–345 |
| `item_id` | Không rỗng và thuộc catalog | 341–348 |
| `rating` | Hữu hạn, từ 1 đến 5 | 350–352 |
| `timestamp` | Số nguyên, không phải `bool` | 354–356 |
| cặp user–item | Không trùng | 358–362 |

### Cấu trúc huấn luyện

| Biến | Dòng | Liên hệ lý thuyết |
|---|---:|---|
| `user_to_index` | 249–250 | Hàng của $p_u$. |
| `item_to_index` | 251 | Hàng của $q_i$. |
| `seen_by_user` | 253, 269 | Mọi item đã thấy. |
| `positive_by_user` | 254, 277 | Item rating từ 4. |
| `positive_pairs` | 255, 279 | Các cặp $(u,i)$ cho BPR. |
| `positive_users` | 256, 275 | User dùng được BPR. |
| `learnable_items` | 258, 278 | Item được chấm bằng BPR. |
| `popularity_counts` | 260, 274 | Số positive của item. |

Dòng 261–264 sắp interaction để kết quả tái lập. Dòng 285–291 gọi `BPRModel.fit`.

## 11. Đọc `bpr.py`

`BPRModel.fit` sắp positive pair, dựng CSR `user × item` kiểu `float32` và gọi
`implicit.cpu.bpr.BayesianPersonalizedRanking`. Backend Cython/OpenMP thực hiện
sampling và SGD; Python không còn lặp qua từng update.

Backend trả factor có thêm một chiều bias. Adapter tách chiều cuối của item
factor thành ba artifact tương thích với inference hiện tại:

```text
user_factors: user_count × factors
item_factors: item_count × factors
item_bias:    item_count
```

Callback mỗi epoch được chuyển thành `train_auc`, `correct`, `skipped` và
`duration_seconds`. `num_threads=0` dùng mọi core khả dụng; dùng
`num_threads=1` khi kiểm thử tính tái lập.

## 12. Recommendation và bundle

### `CFBundle.recommend`

Hàm nằm tại `backend/cf/bundle.py`, dòng 37–83:

1. dòng 44–45 kiểm tra `limit` từ 0 đến 50;
2. dòng 47–48 hợp nhất history với `seen_ids` truyền thêm;
3. dòng 50–53 chuyển user không có positive sang Popularity;
4. dòng 55–63 chấm điểm và lấy candidate chưa thấy;
5. dòng 65–75 dùng `argpartition` lấy vùng top $N$;
6. dòng 77 phá hòa bằng `item_id`;
7. dòng 78–83 tạo response.

`argpartition` tránh sort toàn catalog. Code giữ mọi item bằng ngưỡng, sort chính xác rồi cắt lại để tie ổn định.

### Artifact

| Artifact | Dòng ghi | Nội dung |
|---|---:|---|
| `bpr_model.npz` | 98–104 | User factor, item factor và bias. |
| `user_mapping.json` | 106 | Mapping user. |
| `item_mapping.json` | 107 | Mapping item. |
| `bpr_config.json` | 108 | Cấu hình train. |
| `fit_history.json` | 110–122 | Seen, positive và learnable item. |
| `popularity.parquet` | 124–128 | Ranking Popularity. |
| `manifest.json` | 130–151 | Hash, kích thước và phase. |

`CFBundle.load`, dòng 153–204, kiểm tra checksum trước khi dựng model. `_verify_artifacts` nằm tại dòng 382–386.

## 13. Baseline trong code

### Popularity

- `backend/cf/bundle.py`, dòng 260: khởi tạo count 0 cho toàn catalog.
- Dòng 271–275: chỉ tăng count khi rating từ 4.
- Dòng 296: sort theo count giảm dần rồi `item_id`.
- `backend/cf/baselines.py`, dòng 11–26: loại seen và cắt theo limit.

## 14. Metric trong code

`_evaluate_ranker` nằm tại `backend/cf/evaluation.py`, dòng 112–166.

| Việc | Dòng |
|---|---:|
| Gọi ranker và đo latency | 125–128 |
| Thu thập item coverage | 130–134 |
| Tìm target và rank | 135–140 |
| Candidate Recall@50 | 141–142, 160 |
| HR@10 | 144–145, 159 |
| Discount NDCG@10 | 146, 158 |
| p50 và p95 | 149–165 |

Evaluator bắt buộc `limit == 50`. BPR cache candidate matrix/index, chấm theo
batch và tích lũy `positive_target_slice` trong cùng lượt với full metrics.

`compare_retrievers` chỉ chạy BPR trong vòng tuning. Popularity được thêm vào
comparison cho full validation và full test. Latency BPR là latency trung bình
khấu hao theo từng query trong một batch.

## 15. Tuning và khóa cấu hình

`select_bpr_config` train từng candidate, đánh giá trên mẫu validation cố định
50.000 query rồi lưu NDCG@10, Candidate Recall@50, model size và latency.

Thứ tự chọn tại dòng 49–57:

1. NDCG@10 cao hơn;
2. Candidate Recall@50 cao hơn;
3. model nhỏ hơn;
4. latency thấp hơn.

Candidate thắng được đánh `selected = true`; bảng được ghi tại dòng 58–67.

`backend/cf/cli.py`, dòng 111–128, dùng `itertools.product` tạo mọi tổ hợp factors, learning rate, regularization và epochs.

Thứ tự khóa cấu hình:

- dòng 130–147: tune rồi đánh giá validation;
- dòng 149–166: ghi `locked_config.json`;
- dòng 168–174: fit test bundle bằng config đã khóa;
- dòng 175–185: đánh giá test và ghi `evaluation.json`.

## 16. Chạy chương trình

```bash
uv run recsys-cf pipeline \
  --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet \
  --validation-queries artifacts/movies-tv-2023/queries_val.parquet \
  --test-queries artifacts/movies-tv-2023/queries_test.parquet \
  --output-dir artifacts/cf-v1
```

Mặc định là 64 factors, learning rate 0.01, regularization 0.001, 100 epochs và seed 42. Các option nằm tại `backend/cf/cli.py`, dòng 63–92.

| Lệnh | Khi dùng | Code |
|---|---|---|
| `train` | Tạo một validation hoặc test bundle. | `cli.py`, dòng 21–31 |
| `evaluate` | Đánh giá bundle đã lưu. | `cli.py`, dòng 33–38 |
| `pipeline` | Tune, lock, refit test và so baseline. | `cli.py`, dòng 40–43 và 106–187 |

## 17. Debug theo triệu chứng

| Triệu chứng | Kiểm tra trước | Vị trí |
|---|---|---|
| Mọi user đều `cold_start` | `positive_users` có rỗng vì rating hoặc split không? | `bundle.py`, 241–279 |
| Có item đã xem trong top N | `seen_by_user` và `seen_ids` có đúng không? | `bundle.py`, 47–58 |
| Train AUC không tăng | Có positive pair và non-positive candidate không? | `bpr.py` |
| Không tái lập | Có đang dùng `num_threads=1`, cùng seed, mapping và config không? | `bpr.py`; `bundle.py` |
| Recall@50 thấp | Target có thuộc catalog và learnable pool không? | `bundle.py`, 58–63 |
| Load lỗi checksum | Artifact nào bị sửa sau khi save? | `bundle.py`, 139–156, 382–386 |

## 18. Test quan trọng

| Hành vi | Test |
|---|---|
| Backend, metrics và factor tái lập ở một thread | `tests/test_cf_training.py`, `BPRModelTests` |
| Fit tái lập và cải thiện thứ tự | dòng 52–74 |
| Validation chỉ đọc train | dòng 95–137 |
| Test bundle không đọc test | dòng 138–153 |
| Cold start dùng Popularity | dòng 155–174 |
| Save/load giữ prediction và hash | dòng 210–236 |
| Evaluator có đủ metric | dòng 238–283 |
| Pipeline tạo lock và hai bundle | dòng 434–485 |
| Tuning chọn candidate tốt hơn | dòng 511–556 |

## Kết luận

```text
Lý thuyết
positive + sampled negative
  -> BPR học thứ tự
  -> loại seen
  -> lấy top 50
  -> đo ranking

Code
train_bundle
  -> BPRModel.fit
  -> CFBundle.recommend
  -> compare_retrievers
  -> select_bpr_config
  -> refit test
```

Đọc hết Phần I trước sẽ giúp Phần II trở thành việc đối chiếu khái niệm với code, thay vì phải học lý thuyết trong lúc lần theo từng hàm.
