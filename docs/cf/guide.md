# Hướng dẫn Collaborative Filtering cho người mới

Tài liệu có hai phần. **Phần I** giải thích lý thuyết và ví dụ mà chưa yêu cầu người đọc biết cấu trúc code. **Phần II** mới ánh xạ từng khái niệm vào file, hàm và dòng code của dự án.

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

Fallback rate là tỷ lệ query mà model không chấm được item nào (user không có tín hiệu) nên toàn bộ danh sách là popularity. Latency là thời gian trung bình mỗi query.

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
| `backend/cf/common.py` | `load_fit_data` đọc split được phép theo pha; `FitData` giữ lịch sử từng user theo thời gian; `Popularity`; `top_k`; `rank_items` ghép kết quả model với phần bổ sung popularity. |
| `backend/cf/bpr.py` | Class `BPR` bọc `implicit.cpu.bpr.BayesianPersonalizedRanking`. |
| `backend/cf/itemknn.py` | Class `ItemKNN`, nhánh CF chính (xem [backend/cf/README.md](../../backend/cf/README.md#itemknn)). |
| `backend/eval/evaluate.py` | `evaluate(model, queries)` tính NDCG@10, HR@10, Recall@50. |
| `backend/cf/cli.py` | Lệnh `recsys-cf train / evaluate / recommend`. |

```text
recsys-cf train --model bpr
  -> load_fit_data(splits, catalog, phase)      # chỉ train (validation) hoặc train+validation (test)
  -> BPR.fit(data, ...)                          # implicit học factor từ tương tác rating >= 4
  -> model.save(output_dir)

recsys-cf evaluate --model-dir ...
  -> BPR.load(...)
  -> evaluate(model, queries)                    # model.recommend_batch(user_ids, 50) theo từng lô
```

## 10. Dữ liệu fit: `load_fit_data` và `FitData`

- Pha `validation` chỉ đọc `train`; pha `test` đọc `train + validation` (hằng `FIT_SPLITS`). Đây là chỗ chặn rò rỉ test vào model.
- User và item được đánh số theo thứ tự ID tăng dần; vì vậy "phá hòa theo index" cũng là phá hòa theo `item_id`.
- Tương tác được sắp theo (user, thời gian, item). Lịch sử của user số `u` là đoạn `items[indptr[u]:indptr[u+1]]`, cũ trước mới sau.
- Hai kiểm tra duy nhất: item phải thuộc catalog, và một cặp user–item không xuất hiện hai lần.
- `FitData.matrix(min_rating)` dựng ma trận user × item nhị phân; BPR dùng `min_rating = 4` (chỉ positive), ItemKNN mặc định dùng mọi tương tác.

## 11. `BPR`

- `BPR.fit` gọi `implicit`; thư viện thực hiện negative sampling và SGD bằng Cython/OpenMP.
- `implicit` lưu bias của item ở cột cuối của item factor (cột tương ứng phía user luôn bằng 1). Code tách ra thành `user_factors`, `item_factors`, `item_bias`, nên $score(u, i) = p_u \cdot q_i + b_i$ đúng như Phần I.
- `learned_items`: item có ít nhất một positive. Item còn lại không có tín hiệu học nên bị gán $-\infty$.
- `num_threads=0` dùng mọi core; test dùng `num_threads=1`.

## 12. Gợi ý: `_rank`, `top_k`, `rank_items`

1. User không có rating ≥ 4 → danh sách popularity (cold start).
2. Tính điểm mọi item; gán $-\infty$ cho item chưa học được và item đã xem.
3. `top_k` lấy `limit` item điểm cao nhất; điểm bằng nhau xếp theo index. Hàm dùng `np.partition` để không phải sắp xếp toàn bộ 198k item.
4. `rank_items` tạo `{"item_id", "score", "source"}` rồi bổ sung bằng popularity nếu thiếu.

`recommend_batch` nhân ma trận cho cả lô user một lần, nhanh hơn nhiều so với từng user.

## 13. Popularity

`Popularity` đếm số tương tác rating ≥ 4 của mỗi item trong dữ liệu fit, xếp giảm dần rồi theo index. `fill` lấy các item phổ biến nhất chưa xem và chưa được chọn. Đánh giá baseline này bằng `recsys-cf evaluate ... --popularity`.

## 14. Metric: `evaluate`

- Đọc `user_id, target_item_id, target_rating` từ file query; `--query-limit` lấy mẫu ngẫu nhiên cố định (seed 42) để các model được so trên cùng user.
- Với mỗi query: lấy 50 item, tìm hạng của target, ghi HR@10, NDCG@10 $= 1/\log_2(rank+1)$ và Recall@50.
- `fallback_rate`: tỉ lệ query mà toàn bộ danh sách là popularity.
- `positive_target`: cùng các metric nhưng chỉ với target rating ≥ 4.

## 15. Chọn tham số

Thử vài cấu hình trên validation bằng `--query-limit 20000`, chọn theo NDCG@10 (hòa thì Recall@50 cao hơn), rồi train lại ở pha test và đánh giá test **một lần**. Bảng thử của ItemKNN nằm ở mục 6 của [overview.md](../overview.md). Cấu hình BPR hiện tại (64 factors, 60 epoch) lấy từ lần tune trước trên validation.

## 16. Chạy chương trình

```bash
uv run recsys-cf train --model bpr --splits artifacts/movies-tv-2023/splits.parquet \
  --catalog artifacts/movies-tv-2023/catalog.parquet --phase test \
  --output-dir runs/bpr-movies-tv-2023/test
uv run recsys-cf evaluate --model-dir runs/bpr-movies-tv-2023/test \
  --queries artifacts/movies-tv-2023/queries_test.parquet
uv run recsys-cf recommend --model-dir runs/bpr-movies-tv-2023/test --user-id <USER_ID>
```

## 17. Debug theo triệu chứng

| Triệu chứng | Kiểm tra trước | Ở đâu |
|---|---|---|
| Mọi item đều `source = popularity` | User có rating ≥ 4 trong split được phép không? | `BPR._rank` |
| Có item đã xem trong kết quả | `seen_items` và `exclude` có đúng không? | `common.seen_items` |
| Kết quả không tái lập | Cùng seed, `num_threads=1`, cùng dữ liệu fit? | `BPR.fit` |
| Recall@50 thấp | Target có nằm trong catalog và có positive nào không? | `BPR.learned_items` |

## 18. Test quan trọng

`tests/test_cf.py`:

| Hành vi | Test |
|---|---|
| Pha validation/test chỉ đọc split được phép | `test_fit_data_uses_only_allowed_splits_in_time_order` |
| Popularity đếm positive, loại item đã xem | `test_popularity_counts_positive_ratings_and_skips_seen` |
| BPR không gợi ý item đã xem hay item chưa học | `test_bpr_recommends_only_unseen_learned_items` |
| Metric tính đúng trên ví dụ biết trước | `test_evaluate_reports_ranking_metrics` |

## Kết luận

```text
Lý thuyết                                Code
positive + sampled negative          ->  BPR.fit (implicit)
BPR học thứ tự                       ->  user_factors, item_factors, item_bias
loại seen, lấy top 50                ->  BPR._rank -> top_k -> rank_items
đo ranking                           ->  evaluate
```
