# Khảo sát dữ liệu: Amazon Reviews 2023 – Movies & TV

Tài liệu này khảo sát bộ dữ liệu mới thay cho Digital Music 2018, đánh giá mức độ phù hợp với đề tài, và chọn
tập con dùng cho đồ án.

> **Lưu ý:** khảo sát này được viết ngoài repo. Các file nó nhắc tới (`docs/plan_final.md`, `docs/progress.md`,
> `backend/scripts/explore_movies.py`, `preprocess.py`, `backend/data/processed/stats.json`) không có trong repo, nên số
> liệu ở đây chưa được kiểm chứng lại. Tập con "tác phẩm có title, 20-core" ở mục 4 là tập *Dense-Works* trong
> [overview.md](../overview.md); hiện đồ án vẫn chạy trên tầng 5-core (xem [contract.md](contract.md)).

## 1. Nguồn dữ liệu

Amazon Reviews 2023 (McAuley Lab, UCSD), nhóm Movies & TV. <https://amazon-reviews-2023.github.io/>

| Tệp | Dung lượng | Nội dung |
|---|---|---|
| `Movies_and_TV.jsonl` | 8,4 GB | 17.328.314 review của 6.503.429 user về 747.764 sản phẩm, 08/1997 → 09/2023 |
| `meta_Movies_and_TV.jsonl` | 1,3 GB | 748.224 bản ghi metadata, **khớp 100%** sản phẩm có review |

Hai tệp này **không được commit** vì GitHub giới hạn 100 MB/tệp. Đã thêm `dataset/*.jsonl` vào `.gitignore`.

### Trường dữ liệu

**Review**: `user_id`, `parent_asin` (mã sản phẩm, dùng làm item), `asin`, `rating` (1–5), `timestamp` (mili giây),
`title` (tiêu đề review), `text`, `helpful_vote`, `verified_purchase`, `images`.

**Metadata**:

| Trường | Ví dụ | Dùng cho |
|---|---|---|
| `title` | "The Big Lebowski [Blu-ray]" | Tên hiển thị, nhận diện tác phẩm |
| `main_category` | "Movies & TV" (đĩa) / "Prime Video" (streaming) | Phân loại nguồn |
| `categories` | "Drama", "Science Fiction", "Emotional", "Blu-ray", "Warner Home Video" | Thể loại và tâm trạng (phải lọc nhãn định dạng/hãng) |
| `description` | Tóm tắt nội dung phim | Nội dung item, bằng chứng |
| `details` | Directors, Starring/Actors, Studio, Genre, MPAA rating, Run time, Content advisory | Đạo diễn, diễn viên, độ tuổi |
| `features` | "IMDb 6.8", "2013", "TV-14" | Rất thưa (IMDb ~5%) |
| `average_rating`, `rating_number` | 4.7, 27.200 | Điểm và số lượt đánh giá trên Amazon (gộp mọi phiên bản) |

Ví dụ một review:

```
user_id: AGKASBHYZPGTEPO6LWZPVJWB2BVA  parent_asin: B00CB6VTDS  rating: 5  verified: true
title: "Five Stars"   text: "My Kiddos LOVE this show!!"
```

## 2. Toàn bộ dữ liệu thô

| Chỉ số | Giá trị |
|---|---|
| Review / user / item | 17,3 triệu / 6,5 triệu / 748 nghìn |
| Rating 1★ / 2★ / 3★ / 4★ / 5★ | 8,1% / 4,4% / 7,3% / 15,0% / **65,1%** |
| Review / user | trung vị **1**, p90 5. **58,5% user chỉ có 1 review** |
| Review / item | trung vị 3, p90 35, max 62.445 |
| Số từ / review | trung vị 17; 21,4% dưới 5 từ; 20,2% từ 50 từ trở lên |
| Tiêu đề review dạng "Five Stars" | 26,9% |
| Verified purchase | 79,4% |
| Cặp (user, item) lặp lại | 169.795 (1,0%) |
| Theo năm | tập trung 2013–2016 (55%), còn dữ liệu đều tới 2022 |

Dữ liệu thô quá thưa và quá lớn (density ~0,0004%) để dùng trực tiếp: phải lọc k-core.

### Thử các mức lọc

| Tập con | Tương tác | User | Item | Density | Có title |
|---|---|---|---|---|---|
| 5-core | 7,44 tr | 657 k | 198 k | 0,006% | 58% |
| 10-core | 3,73 tr | 180 k | 85 k | 0,024% | 61% |
| 20-core | 794 k | 21,4 k | 17,1 k | 0,22% | 73% |
| Chỉ item có title, 20-core | 431 k | 10,4 k | 10,1 k | 0,41% | 100% |
| **Gộp phiên bản thành tác phẩm, 20-core (chọn)** | **633 k** | **14,8 k** | **12,2 k** | **0,35%** | **100%** |

EASE cần ma trận item × item dày, nên số item phải ở khoảng ≤ 15 nghìn để chạy được trên máy 15 GB RAM.

## 3. Ba vấn đề của metadata và cách xử lý

### 3.1 Item Prime Video thường không có title

Khoảng 42% item trong 5-core **không có title**, và gần như toàn bộ là Prime Video. Những item này chỉ có đạo diễn,
diễn viên và hãng phim; không có mô tả hay thể loại. Ví dụ `B001OGWY1W` chỉ có
*Directors: Christopher Nolan · Starring: Christian Bale, Michael Caine, Heath Ledger*, tức là phim *The Dark Knight*.

**Xử lý:** chỉ giữ item có title. Phần lớn các phim này vẫn có mặt qua phiên bản DVD/Blu-ray có title.

### 3.2 Một phim có nhiều phiên bản

Cùng một phim có nhiều mã sản phẩm: DVD, Blu-ray, bản kỷ niệm, Prime Video. Ví dụ *The Big Lebowski* có 14 mã.
Nếu không gộp, hệ thống sẽ gợi ý bản Blu-ray cho người đã xem bản DVD.

**Xử lý:** gộp thành **tác phẩm (work)** theo khóa *title chuẩn hóa + đạo diễn đầu tiên*. Title chuẩn hóa bỏ các phần
trong ngoặc như `[Blu-ray]`, `(Widescreen Edition)`, `(Two-Disc Collector's Set)`.

- 115.101 item có title gộp thành 95.929 tác phẩm (giảm 17%). Trong tập được chọn, 49,7% tác phẩm có hơn 1 phiên bản.
- Thêm đạo diễn vào khóa để không gộp nhầm các bản làm lại cùng tên (*Halloween*, *King Kong*).
- Hạn chế: các mùa của một chương trình TV có cùng title (*Survivor*) bị gộp thành một tác phẩm. Có thể chấp nhận,
  vì gợi ý ở mức chương trình vẫn hợp lý.
- Review không bị nhân bản giữa các phiên bản: chỉ 0,01% review trùng (user, thời điểm, độ dài) trên hai mã khác nhau.
  `rating_number` gần bằng nhau giữa các phiên bản (16.440 / 16.583 / 16.645) là số đếm của trang Amazon, không phải
  số review có trong dữ liệu.

### 3.3 `categories` lẫn nhiều nhãn không phải thể loại

Top category gồm cả nhãn định dạng (Blu-ray, DVD), hãng phim (Warner Home Video), khuyến mãi (Today's Deals) và nhãn
chung (Movies, All Titles). Cần **danh sách trắng**:

- **Thể loại:** Drama, Comedy, Action & Adventure, Science Fiction, Fantasy, Horror, Suspense, Mystery & Thrillers,
  Kids & Family, Animation, Romance, Westerns, Musicals, Documentary, TV, …
- **Tâm trạng** (chủ yếu có ở Prime Video): Emotional, Cerebral, Compelling, Intense, Passionate, Charming, Exciting, Tense, Fun, …

## 4. Bộ dữ liệu được chọn: tác phẩm có title, 20-core

> **Cập nhật 11/10 (sau khi viết `preprocess.py`):** cách gộp phiên bản tốt hơn (chuẩn hóa title mạnh hơn, gộp theo đạo diễn
> chung) nên tập lõi thực tế là **787.176 tương tác · 17.956 user · 13.332 tác phẩm**, 4,8 phiên bản/tác phẩm, kho mở rộng
> 617.641 review. Số liệu bên dưới là của lần khảo sát; số chính thức xem `backend/data/processed/stats.json` và
> `docs/progress.md` (cả hai chưa có trong repo).

### 4.1 Quy mô và tương tác

| Chỉ số | Giá trị |
|---|---|
| Tương tác / user / tác phẩm | **633.316 / 14.814 / 12.153**, density 0,35% |
| Tương tác / user | trung bình 42,8 · trung vị 30 · p90 72 · max 1.157 |
| Tương tác / tác phẩm | trung bình 52,1 · trung vị 36 · p90 99 · max 694 |
| Đuôi dài | 22,4% tác phẩm chiếm 50% tương tác |
| Rating 1★ / 2★ / 3★ / 4★ / 5★ | 3,3% / 3,8% / 8,8% / 19,3% / **64,8%** (≥ 4★: 84,1%) |
| Cùng user, cùng tác phẩm, khác phiên bản | 4.512 (0,12%), giữ review sớm nhất |

### 4.2 Metadata (tính ở mức tác phẩm, gộp từ mọi phiên bản)

| Trường | Độ phủ |
|---|---|
| Title | 100% |
| Mô tả | 96,7% (trung vị 152 từ) |
| Category (sau khi gộp) | 100% (cần lọc theo danh sách trắng) |
| Diễn viên | 95,1% |
| Trường Genre | 83,0% |
| Đạo diễn | 75,6% |
| Điểm IMDb | ~3% (không dùng) |

### 4.3 Văn bản review

| Chỉ số | Giá trị |
|---|---|
| Số từ / review | trung vị **29** (Digital Music: 10) |
| Review dưới 5 từ / dưới 10 từ | 21,5% / 31,8% |
| Tiêu đề review dạng "X Stars" | 23,4% |
| Review ≥ 10 từ nói về cốt truyện, diễn xuất, nhân vật, tập phim | ~38% (đếm bằng regex trên mẫu 4.485 review) |
| Review ≥ 10 từ nói về đĩa, giao hàng, giá, hư hỏng | ~8%, **phải lọc khỏi bằng chứng** |
| Text có thẻ HTML (`<br />`) | Có, `clean()` hiện có đã xử lý |

**Kho review mở rộng.** Ngoài 633k review của user trong tập lõi, còn **3,12 triệu review** (2,06 triệu review ≥ 10 từ)
về cùng các tác phẩm, do user ngoài tập lõi viết. Những user này không tham gia đánh giá, nên review của họ dùng làm
bằng chứng được, miễn là vẫn lọc `ts < t_query`. Số review/tác phẩm (trung vị) tăng từ **36 lên 181**, p10 = 78.
Ví dụ *The Big Lebowski*: 163 review trong tập lõi, thêm 984 review từ ngoài tập lõi.

### 4.4 Thời gian và chia tập

Chia leave-last-out theo thời gian cho từng user: tương tác cuối vào test, áp chót vào valid, còn lại vào train.

| Tập | Số tương tác | Ngày trung vị |
|---|---|---|
| train | 603.688 | 2014-12-15 |
| valid | 14.814 (1/user) | 2017-05-20 |
| test | 14.814 (1/user) | 2018-02-15 |

- 0% item của valid/test chưa từng xuất hiện trong lịch sử, tức không có cold-start ở phía item.
- Tương tác phân bố từ 2000 đến 2023, tập trung 2013–2016.

## 5. Đánh giá: so với Digital Music 2018

| Tiêu chí | Digital Music 2018 (cũ) | Movies & TV 2023 (mới, tập chọn) |
|---|---|---|
| Quy mô | 123 k tương tác, 12,4 k user, 9,9 k item | 633 k tương tác, 14,8 k user, 12,2 k tác phẩm |
| Lịch sử / user (trung vị) | 7 | **30** |
| Metadata | 1% item có title; không có thể loại | **100% có title**, 97% có mô tả, 95% có diễn viên |
| Review | trung vị 10 từ, 37% dưới 5 từ | trung vị **29 từ**, 21% dưới 5 từ |
| Kho bằng chứng | ~145 k review | 633 k + **3,1 triệu** review mở rộng |
| Rating ≥ 4★ | 93% | 84%, có **7% review ≤ 2★** làm tín hiệu "không thích" |
| Item trùng lặp | Không đáng kể | **Có**: phải gộp phiên bản thành tác phẩm |
| Tài nguyên | Nhẹ | Dữ liệu thô 9,7 GB, phải xử lý theo luồng |

### Điểm mạnh
- **Metadata đầy đủ.** Có thể làm content-based thật (thể loại, đạo diễn, diễn viên, mô tả) và có nguồn *sự thật*
  cho giải thích, bên cạnh nguồn *ý kiến* từ review.
- **Lịch sử dài hơn.** Hồ sơ người dùng giàu thông tin hơn, và CF có nhiều tín hiệu hơn.
- **LLM hiểu được item.** Tên phim, đạo diễn và diễn viên là những thứ LLM biết, nên bước xếp hạng lại có cơ sở để
  vượt Hybrid.
- **Có tín hiệu tiêu cực** (≤ 2★) để đưa vào hồ sơ dưới dạng "không thích".

### Rủi ro
- **Kiến thức sẵn có của LLM.** LLM biết các phim nổi tiếng, nên có thể giải thích bằng kiến thức riêng thay vì bằng
  chứng, và có thể thiên về phim phổ biến. Validator phải kiểm tra cả tên người và thể loại được nhắc tới, không chỉ mã [E#].
- **Gộp phiên bản dùng luật heuristic.** Có thể gộp nhầm (các mùa TV) hoặc sót (title viết khác nhau).
- **Lọc 20-core chọn ra người dùng tích cực** (≥ 20 review), nên kết quả không đại diện cho người dùng thông thường
  (58% user thô chỉ có 1 review). Cần nêu trong báo cáo.
- **Metadata là ảnh chụp năm 2023**, mới hơn thời điểm của nhiều tương tác. Không gây rò rỉ nhãn, nhưng cần ghi chú.
- **Review về đĩa và giao hàng** (~8%) và review ngắn chung chung vẫn cần bộ lọc.
