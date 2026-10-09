`PROFILE BUILDER` không phải model cần train. Nó là bước tổng hợp hồ sơ người dùng tại runtime, được tạo lại cho mỗi request từ `user_id` và `as_of`.

Luồng build dự kiến:

1. Đọc lịch sử của `user_id` từ history store.
2. Chỉ giữ tương tác thuộc split được phép:

   - Validation: `train`
   - Test: `train + validation`

3. Chỉ giữ tương tác có `timestamp < as_of`. Tương tác đúng bằng `as_of` cũng bị loại để tránh leakage.
4. Chia lịch sử thành:

   - `seen_ids`: tất cả item đã tương tác, kể cả rating thấp hoặc trung tính.
   - `positive_items`: rating ≥ 4, lấy tối đa 20 item gần nhất.
   - `negative_item_ids`: rating ≤ 2.
   - Rating = 3: chỉ nằm trong `seen_ids`, không phải thích hoặc ghét.
   - `positive_terms`: từ khóa lấy từ metadata/review của các item thích, dùng tokenizer giống BM25.
   - `history_sources`: các đoạn lịch sử được gắn ID như `H1`, `H2` để LLM giải thích.

Ví dụ logic:

```python
def build_profile(user_id, as_of, history_rows, catalog):
    history = [
        row
        for row in history_rows
        if row["user_id"] == user_id
        and row["timestamp"] < as_of
    ]

    history.sort(key=lambda row: row["timestamp"], reverse=True)

    seen_ids = {row["item_id"] for row in history}
    positive_rows = [row for row in history if row["rating"] >= 4][:20]
    negative_item_ids = {
        row["item_id"]
        for row in history
        if row["rating"] <= 2
    }

    return Profile(
        user_id=user_id,
        history_ids=[row["item_id"] for row in positive_rows],
        seen_ids=seen_ids,
        positive_terms=extract_terms(positive_rows, catalog),
        negative_item_ids=negative_item_ids,
        history_sources=make_history_sources(positive_rows),
    )
```

Kết quả minh họa:

```text
Profile(
    user_id="u17",
    history_ids=["movie_A", "series_B"],
    seen_ids={"movie_A", "series_B", "movie_C", "series_D"},
    positive_terms=["drama", "mystery", "miniseries"],
    negative_item_ids={"series_D"},
    history_sources={
        "H1": "User rated movie_A 5 stars",
        "H2": "User rated series_B 4 stars",
    },
)
```

Các thành phần sau sử dụng profile như sau:

- CF dùng các item tích cực để tìm item tương tự.
- BM25 dùng `positive_terms` để tìm theo metadata.
- `negative_item_ids` là tín hiệu tránh, không được đưa vào BM25 như từ khóa dương.
- `seen_ids` dùng để loại toàn bộ item người dùng đã tương tác.
- Nếu không có tín hiệu tích cực thì chuyển sang popularity/cold-start.

Thiết kế chi tiết đã được mô tả tại [docs/architecture.md](/Users/ledinhquan/Hust/IT4613-project/docs/architecture.md:102) và hợp đồng Q1 tại [docs/architecture.md](/Users/ledinhquan/Hust/IT4613-project/docs/architecture.md:165).

Điểm quan trọng: hiện tại dự án **chưa cài đặt `Profile` hoặc `build_profile` riêng**. Code CF mới chỉ tạo gần tương đương:

- `seen_by_user` cho toàn bộ item.
- `positive_by_user` cho rating ≥ 4.

Phần này nằm ở [backend/cf/bundle.py](/Users/ledinhquan/Hust/IT4613-project/backend/cf/bundle.py:225). Nó chưa hỗ trợ `as_of`, item ghét, `positive_terms` và `history_sources`, nên khối `PROFILE BUILDER` trong `qa.md` hiện vẫn là kiến trúc dự kiến, chưa phải module hoàn chỉnh.
