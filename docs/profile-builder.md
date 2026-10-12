# Kế hoạch thiết kế Profile Builder

> **Trạng thái:** profile core P1 đã được cài đặt trong `backend/profile/` và kiểm thử tại `tests/test_profile_builder.py`. Tích hợp CF P2, BM25 P3, semantic search P4 và fusion P5 chưa được cài đặt. Profile Builder không phải model cần huấn luyện.
>
> Tài liệu này chi tiết hóa và thay thế contract `Profile`/Q1 hiện có trong `architecture.md`. Trước khi triển khai, P0 phải đồng bộ lại tài liệu kiến trúc tổng thể để không còn dùng schema cũ có `positive_terms`.

## 1. Mục tiêu và quyết định kiến trúc

Profile Builder nhận một ngữ cảnh truy vấn đã được server xác thực, đọc lịch sử hợp lệ và tạo một `UserProfile` bất biến cho đúng một request. Nó phải:

- dùng cùng một định nghĩa lịch sử cho mọi retriever;
- chặn leakage theo split và thời gian tại một nơi duy nhất;
- giữ toàn bộ `seen_item_ids` để lọc candidate;
- biểu diễn tín hiệu thích/không thích mà không phụ thuộc tokenizer hoặc embedding model;
- cung cấp provenance ổn định cho giải thích và debug;
- cho phép từng retriever tự quyết định cách biến profile thành query của nó.

Ranh giới quan trọng:

```text
HistoryStore + CatalogSnapshot
             │
             ▼
       ProfileBuilder
             │
             ▼
        UserProfile
       ┌─────┼──────────────┐
       ▼     ▼              ▼
   CF view  BM25 adapter  Semantic adapter
```

Profile Builder **không** tokenize cho BM25, không gọi embedding model, không tính score candidate và không truy cập target. Các thao tác đó thuộc adapter/retriever tương ứng. Vì vậy, trường `positive_terms` trong thiết kế cũ bị loại khỏi schema lõi.

## 2. Nguồn dữ liệu và chính sách chống leakage

### 2.1 Input tin cậy

```python
@dataclass(frozen=True)
class ProfileRequest:
    user_id: str
    as_of: int
    phase: Literal["validation", "test"]

    artifact_version: str


@dataclass(frozen=True)
class ProfilePolicy:
    allowed_splits: tuple[str, ...]
    positive_threshold: float = 4.0
    negative_threshold: float = 2.0

    max_positive_signals: int = 20
    max_negative_signals: int = 20
    recency_half_life_days: float = 90.0
```

`phase` và `allowed_splits` được lấy từ config/artifact đã khóa ở phía server, không nhận trực tiếp từ client:

| Phase | Split lịch sử được phép |
|---|---|
| Validation | `train` |
| Test | `train`, `validation` |

Nguồn chuẩn là các interaction đã chuẩn hóa trong history store. Không dựng profile từ cột `history` của CSV nguồn vì trường đó không chứa rating, timestamp và provenance đầy đủ. Không đọc `fit_history.json` của CF làm nguồn chuẩn vì artifact này đã mất timestamp.

### 2.2 Filter bắt buộc

Một interaction chỉ hợp lệ khi đồng thời thỏa:

1. `user_id` khớp chính xác;
2. `split` thuộc `allowed_splits`;
3. `timestamp < as_of`;
4. `item_id` hợp lệ.

Sự kiện có `timestamp == as_of` luôn bị loại. Profile Builder không nhận `target_item_id`, nên không thể vô tình dùng target để lọc hoặc tạo đặc trưng. Item thiếu trong catalog vẫn phải nằm trong `seen_item_ids`; builder chỉ để text rỗng và ghi warning. Như vậy, lỗi coverage catalog không làm mất dấu một tương tác đã xảy ra.

Store của từng phase nên chỉ chứa split được phép để có hai lớp bảo vệ: giới hạn vật lý trong artifact và filter logic khi query. SQLite cần index `(user_id, timestamp)`; nếu dùng Parquet theo batch thì partition/bucket theo user và split.

### 2.3 Trùng lặp và thứ tự xác định

- `seen_item_ids` là hợp của **mọi** interaction hợp lệ, kể cả rating 1–3.
- Nếu có nhiều interaction cho cùng item, tất cả vẫn chứng minh item đã được thấy nhưng chỉ interaction mới nhất quyết định sentiment hiện tại.
- Khi timestamp bằng nhau, phá hòa bằng `(review_id, item_id)` tăng dần; không coi thứ tự phá hòa là thứ tự nhân quả.
- Trước khi chọn interaction mới nhất, thứ tự chuẩn là `(timestamp, review_id, item_id)`.
- Catalog lookup phải là snapshot thuộc cùng bundle; không truy cập metadata thay đổi bên ngoài lúc runtime.

## 3. Schema lõi

```python
Sentiment = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True)
class ProfileTextField:
    name: Literal[
        "title",
        "subtitle",
        "description",
        "feature",
        "main_category",
        "category",
        "details",
    ]
    text: str
    source_id: str


@dataclass(frozen=True)
class PreferenceSignal:
    history_id: str
    item_id: str
    rating: float

    timestamp: int
    sentiment: Sentiment
    weight: float

    text_fields: tuple[ProfileTextField, ...]


@dataclass(frozen=True)
class ProfileDiagnostics:
    valid_interaction_count: int
    duplicate_item_count: int
    positive_count_before_limit: int

    negative_count_before_limit: int
    text_covered_positive_count: int
    missing_catalog_item_count: int

    warnings: tuple[str, ...]


@dataclass(frozen=True)
class UserProfile:
    schema_version: str
    user_id: str
    as_of: int

    phase: Literal["validation", "test"]
    artifact_version: str
    seen_item_ids: frozenset[str]

    positive_signals: tuple[PreferenceSignal, ...]
    negative_signals: tuple[PreferenceSignal, ...]

    diagnostics: ProfileDiagnostics
```

### 3.1 Ý nghĩa từng nhóm trường

- `seen_item_ids`: tập đầy đủ, không giới hạn 20 item. Đây là nguồn duy nhất để filter candidate sau retrieval.
- `positive_signals`: tối đa 20 tín hiệu gần nhất có rating từ 4 trở lên.
- `negative_signals`: tối đa 20 tín hiệu gần nhất có rating từ 2 trở xuống.
- Rating 3 chỉ đóng góp vào `seen_item_ids`; không đi vào query dương hoặc âm.
- `text_fields`: các field catalog đã chuẩn hóa kèm tên và `source_id`: title, subtitle, description, feature, main category, category và details nội dung được chọn lọc. Giữ ranh giới field để BM25 có thể áp dụng field boost. `store`, `average_rating` và `rating_number` không trở thành tín hiệu sở thích văn bản.
- `diagnostics`: phục vụ trace và kiểm thử coverage; không dùng để học từ target.

Không lưu vector hoặc token trong `UserProfile`. Chúng phụ thuộc phiên bản retriever, làm profile khó cache đúng và ghép cứng builder với model.

### 3.2 Trọng số sở thích

Trọng số là tín hiệu chung, còn retriever có quyền chuẩn hóa lại. Với tuổi sự kiện theo ngày:

$$
age_i = \max\left(0, \frac{as\_of - timestamp_i}{86\,400\,000}\right)
$$

$$
recency_i = 2^{-age_i / half\_life}
$$

Độ mạnh rating:

$$
rating\_strength_i = \frac{|rating_i - 3|}{2}
$$

Và:

$$
weight_i = rating\_strength_i \times recency_i
$$

Builder chọn tối đa 20 tín hiệu theo thời gian giảm dần để quy tắc dễ giải thích, sau đó tính `weight`. Các giá trị mặc định phải nằm trong config có version và được tune chỉ trên validation. Không thay đổi công thức sau khi khóa cấu hình test.

## 4. Cách ba retriever sử dụng profile

### 4.1 CF retriever

Với BPR-MF hiện tại, CF không cần text hoặc embedding từ profile:

```python
CFQuery(
    user_id=profile.user_id,
    excluded_item_ids=profile.seen_item_ids,
)
```

CF chỉ hợp lệ khi:

- user tồn tại trong mapping của đúng bundle;
- user thuộc `positive_users` của model;
- có ít nhất một positive signal hợp lệ trước `as_of`.

Điều kiện cuối ngăn dùng factor được học từ một tín hiệu nằm sau cutoff của request. Tuy nhiên, nếu artifact BPR đã học interaction sau `as_of`, Profile Builder không thể gỡ ảnh hưởng đó khỏi factor. Thí nghiệm chính chỉ an toàn vì validation dùng model fit trên train và test dùng model fit trên train + validation theo thứ tự đã khóa. Query tùy ý nằm giữa tập fit cần snapshot model theo thời gian hoặc phải được từ chối.

Nếu sau này thêm item-item CF cho user mới, adapter có thể dùng `positive_signals` và `weight` mà không đổi schema profile.

### 4.2 BM25 adapter

BM25 adapter nhận các `positive_signals` có text, dùng đúng analyzer/version của index và tạo `BM25Query`:

1. tokenize từng field bằng analyzer của BM25;
2. áp dụng field boost đã khóa, ví dụ title/category cao hơn description;
3. cộng term weight từ các item dương theo `signal.weight`;
4. loại stopword và term quá phổ biến bằng thống kê của chính index;
5. giữ tối đa `max_query_terms`, sắp theo weight giảm dần rồi term tăng dần;
6. nếu không còn term thì trả `not_applicable`, không trả ngẫu nhiên các item score 0.

```python
@dataclass(frozen=True)
class WeightedTerm:
    term: str
    weight: float


@dataclass(frozen=True)
class BM25Query:
    terms: tuple[WeightedTerm, ...]
    excluded_item_ids: frozenset[str]
```

`negative_signals` không được biến thành term dương. Ở phiên bản đầu, chúng chỉ được đưa vào trace và có thể dùng cho rule down-rank ở fusion nếu rule đó được đánh giá riêng.

### 4.3 Semantic search adapter

Offline, semantic retriever encode một canonical document cho mỗi catalog item bằng encoder đã khóa. Canonical document dùng cùng quy tắc field và phiên bản catalog cho cả validation/test. Manifest phải lưu ít nhất:

- embedding model/revision;
- text-template version;
- vector dimension và normalization;
- catalog hash;
- index type và seed nếu index gần đúng.

Runtime không cần encode một chuỗi profile dài. Adapter lấy vector đã index của các item dương và tạo centroid có trọng số:

$$
q^+ = \operatorname{normalize}\left(
\frac{\sum_{i \in P} w_i e_i}{\sum_{i \in P} w_i}
\right)
$$

Trong đó $e_i$ là vector đã chuẩn hóa của item dương. Item thiếu embedding bị bỏ qua và được đếm trong diagnostics của semantic adapter, không phải diagnostics lõi của profile. Nếu không còn vector dương hoặc norm bằng 0, nhánh semantic trả `not_applicable`.

Phiên bản đầu **không trừ centroid âm**. Phần mở rộng có thể thử:

$$
q = \operatorname{normalize}(q^+ - \beta q^-)
$$

nhưng chỉ bật sau ablation trên validation vì negative feedback có thể nói về chất lượng sản phẩm, không nhất thiết phủ định toàn bộ chủ đề của item. `beta` phải bằng 0 trong baseline và không được tune trên test.

```python
@dataclass(frozen=True)
class SemanticQuery:
    vector: NDArray[np.float32]
    contributing_item_ids: tuple[str, ...]
    excluded_item_ids: frozenset[str]

    encoder_version: str
```

Semantic retriever xếp theo cosine similarity, bỏ toàn bộ `seen_item_ids`, loại score không hữu hạn và phá hòa bằng `item_id`. ANN và exact search phải dùng cùng contract output để có thể kiểm tra recall của index.

## 5. Luồng build chi tiết

```text
1. Validate ProfileRequest và policy từ manifest
2. Query history theo user_id và timestamp < as_of
3. Kiểm tra lại split, timestamp, ID và catalog lookup/coverage
4. Tạo seen_item_ids từ toàn bộ hàng hợp lệ
5. Gom theo item_id, chọn interaction mới nhất
6. Phân loại positive / neutral / negative
7. Sắp positive và negative theo thời gian giảm dần
8. Áp limit, tính weight, lấy text snapshot
9. Gán H1..Hn theo thứ tự ổn định
10. Tạo diagnostics, đóng băng UserProfile và trả kết quả
```

`H1..Hn` chỉ được cấp cho các signal thực sự xuất hiện trong profile rút gọn. Nội dung nguồn nên là dữ liệu có cấu trúc được renderer tạo ra, ví dụ `Rated item <id> 5/5 before the recommendation cutoff`; không dùng LLM để tạo history source. Nguồn H chỉ chứng minh hành vi/rating của user, không được dùng để chứng minh thuộc tính nội dung của item; nhận định về item vẫn cần evidence E từ catalog/review.

Builder trả profile rỗng hợp lệ thay vì ném lỗi khi user không có history. Lỗi store, bundle không khớp hoặc policy không hợp lệ là lỗi hệ thống và không được biến thành cold-start im lặng.

## 6. Eligibility và fallback

Mỗi nhánh tự báo một trong `ready`, `not_applicable`, `failed`:

| Trường hợp | CF | BM25 | Semantic | Hành vi tổng |
|---|---|---|---|---|
| User lạ, không history | Không áp dụng | Không áp dụng | Không áp dụng | Popularity chưa seen, `cold_start` |
| Có positive, thiếu text | Có thể dùng | Không áp dụng | Chỉ dùng nếu có embedding | Dùng các nhánh sẵn sàng |
| Có text nhưng item chưa có embedding | Có thể dùng | Dùng | Bỏ item thiếu vector | Ghi coverage warning |
| Chỉ có rating 1–3 | Không áp dụng | Không áp dụng | Không áp dụng | Popularity chưa seen |
| Một adapter lỗi | Nhánh khác vẫn chạy | Nhánh khác vẫn chạy | Nhánh khác vẫn chạy | Ghi warning, không giả thành `not_applicable` |
| Không còn item chưa seen | Không có kết quả | Không có kết quả | Không có kết quả | Trả danh sách rỗng hợp lệ |

Popularity luôn phải lọc bằng `profile.seen_item_ids`. Negative item đã nằm trong seen nên không thể được gợi ý lại.

## 7. Interface module đề xuất

```text
backend/profile/
├── builder.py             # schema + ProfileBuilder
├── history_store.py       # protocol + SQLite/Parquet implementation
├── catalog_store.py       # catalog snapshot protocol
├── bm25_query.py          # UserProfile -> BM25Query
├── semantic_query.py      # UserProfile -> SemanticQuery
├── cf_adapter.py          # UserProfile -> CF call
└── eligibility.py         # ready/not_applicable/failed
```

Interface tối thiểu:

```python
class HistoryStore(Protocol):
    def get_before(
        self,
        user_id: str,
        as_of: int,
        allowed_splits: tuple[str, ...],
    ) -> Sequence[Interaction]: ...


class ProfileBuilder:
    def build(
        self,
        request: ProfileRequest,
        policy: ProfilePolicy,
    ) -> UserProfile: ...
```

Profile không nên được ghi vào CF bundle như một artifact dài hạn. Đây là dữ liệu theo request. Nếu cần cache, khóa cache phải gồm ít nhất `(schema_version, artifact_version, policy_hash, user_id, as_of)`; phiên bản đầu nên chưa cache để ưu tiên tính đúng.

## 8. Kế hoạch triển khai theo TDD

Tuân theo quy ước TDD của dự án: với mỗi symbol mới, tạo skeleton import được trước; sau đó viết test thấy thất bại ở assertion; cuối cùng chỉ cài đặt hành vi tối thiểu để test qua.

### P0 — Khóa contract

- Chốt schema `Interaction`, `ProfileRequest`, `ProfilePolicy`, `PreferenceSignal`, `UserProfile`.
- Chốt canonical catalog text và version của analyzer/encoder.
- Ghi `profile_schema_version` và `profile_policy` vào manifest/config.

### P1 — Profile core

- Tạo skeleton schema và `ProfileBuilder.build`.
- Test filter split, `timestamp < as_of`, rating boundary và user lạ.
- Test seen đầy đủ, deduplicate latest-event và thứ tự ổn định.
- Test limit 20, weight, history ID và diagnostics.
- Cài đặt in-memory store trước, sau đó adapter store thật.

### P2 — CF integration

- Thay đường truyền `seen_ids` rời rạc bằng `UserProfile` tại orchestration layer.
- Giữ `CFBundle.recommend` hiện tại làm primitive; adapter chịu trách nhiệm eligibility và exclusions.
- Regression test để ranking CF không đổi với query hiện có.
- Test request cutoff không tương thích với snapshot model bị từ chối hoặc đánh dấu rõ.

### P3 — BM25 query adapter

- Tạo skeleton analyzer/query schema trước test.
- Test dùng riêng positive text, cùng analyzer với index và deterministic top terms.
- Test query rỗng, metadata thiếu, stopword-only và không đưa negative term vào query.
- Đo BM25 Recall@50 độc lập trước fusion.

### P4 — Semantic query adapter

- Tạo embedding index offline với manifest và exact-search fixture nhỏ.
- Test weighted centroid bằng vector biết trước, normalization, missing vector và zero norm.
- Test loại seen trước top-K và tie-break ổn định.
- So sánh exact với ANN bằng ANN recall; đo Semantic Recall@50 độc lập.

### P5 — Fusion và ablation

- Hợp nhất output CF/BM25/semantic bằng rank-based fusion; không cộng raw score khác thang đo.
- Tune trọng số trên validation; khóa trước test.
- Chạy tối thiểu: CF, BM25, semantic, CF+BM25, CF+semantic, cả ba nhánh.
- Báo Candidate Recall@20/50, NDCG@10, coverage theo user và latency từng adapter.
- Negative-centroid chỉ là ablation tùy chọn sau baseline dương ổn định.

Với nhánh $b$ trả $N_b$ item, score theo rank của item $i$ là:

$$
s_b(i) =
\begin{cases}
\frac{N_b-rank_b(i)+1}{N_b}, & i \text{ có trong nhánh } b \\
0, & \text{ngược lại}
\end{cases}
$$

Fusion ba nhánh dùng trọng số khóa trên validation:

$$
s(i) = \frac{\sum_{b \in A}\lambda_b s_b(i)}{\sum_{b \in A}\lambda_b}
$$

Trong đó $A$ chỉ gồm các nhánh `ready`. Chuẩn hóa theo tổng trọng số nhánh đang hoạt động giúp request không bị giảm toàn bộ score khi một nhánh `not_applicable`. Nhánh `failed` vẫn bị loại khỏi phép tính nhưng phải sinh warning; phá hòa cuối bằng `item_id`.

## 9. Ma trận test bắt buộc

| Nhóm | Ca kiểm thử |
|---|---|
| Leakage | split cấm; timestamp trước/bằng/sau `as_of`; target không có trong input |
| Rating | 1, 2 là negative; 3 neutral; 4, 5 positive; rating không hữu hạn bị từ chối khi import |
| Seen | positive, neutral, negative đều bị loại khỏi candidate |
| Duplicate | cùng item nhiều rating; cùng timestamp; kết quả độc lập thứ tự đọc |
| Limits | hơn 20 positive/negative; seen vẫn không bị cắt |
| Text | metadata rỗng; một phần field rỗng; Unicode; analyzer version không khớp |
| Semantic | vector thiếu; sai dimension; NaN; zero norm; encoder/catalog hash không khớp |
| CF | user ngoài mapping; user không thuộc `positive_users`; cutoff không tương thích snapshot |
| Fallback | profile rỗng; từng nhánh rỗng/lỗi; catalog nhỏ hơn K; mọi item đã seen |
| Determinism | cùng input/config cho profile, query và thứ tự candidate giống hệt |

## 10. Điều kiện nghiệm thu

Profile Builder được xem là hoàn thành khi:

1. mọi retriever nhận cùng một `UserProfile` và không tự đọc history;
2. fixture leakage chứng minh không signal nào có split cấm hoặc `timestamp >= as_of`;
3. `seen_item_ids` không bị giới hạn theo số signal;
4. BM25 và semantic adapter có thể rỗng độc lập mà không làm hỏng CF/fallback;
5. profile và query đầu ra xác định hoàn toàn với cùng bundle/config;
6. trace ghi số interaction, positive/negative, text/embedding coverage và lý do từng nhánh không chạy;
7. đánh giá tách riêng chất lượng từng retriever trước khi kết luận về hybrid;
8. manifest từ chối analyzer, encoder, catalog hoặc profile-policy không cùng phiên bản.

## 11. Các giá trị mặc định ban đầu

| Tham số | Giá trị khởi đầu | Ghi chú |
|---|---:|---|
| Positive threshold | 4 | Khớp BPR hiện tại |
| Negative threshold | 2 | Rating 3 trung tính |
| Positive signal limit | 20 | Giới hạn profile/prompt |
| Negative signal limit | 20 | Chủ yếu cho trace và ablation |
| Recency half-life | 90 ngày | Tune trên validation |
| BM25 query terms | 32 | Sau analyzer và IDF filter |
| Semantic negative coefficient | 0 | Baseline không dùng centroid âm |
| Retriever pool | 50 mỗi nhánh | Trước fusion |

Các giá trị trên là điểm xuất phát, không phải kết quả nghiên cứu. Chúng phải được ghi trong config và mọi thay đổi phải đi qua validation.

## 12. Quan hệ với code hiện tại

Profile core hiện có schema, `ProfileBuilder`, `HistoryStore`/`CatalogStore` protocol, in-memory store để kiểm thử và adapter Parquet. Nó thực thi cutoff `timestamp < as_of`, policy split theo phase, deduplicate latest-event, sentiment, recency weight, text snapshot, history source và diagnostics. Runtime Movies and TV phải dùng `artifacts/movies-tv-2023/catalog_enriched.parquet`; catalog chỉ có ID vẫn được hỗ trợ cho CF và tạo coverage warning.

CF hiện vẫn dùng trực tiếp `seen_by_user` và `positive_by_user` trong `backend/cf/bundle.py`. Chúng là dữ liệu phục vụ BPR bundle; bước P2 sẽ thêm adapter để CF nhận `UserProfile` mà không thay đổi primitive scoring hiện tại.

P1 và bước enrich metadata đã hoàn thành. Thứ tự tiếp theo là P2 để bảo toàn CF hiện có, sau đó P3 và P4 có thể phát triển độc lập trên catalog enriched. Không dùng metadata Digital Music hoặc text bịa để làm pipeline có vẻ hoàn chỉnh.
