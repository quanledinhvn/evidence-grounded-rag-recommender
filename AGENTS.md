# Agent Instructions

## Rule Identifier Convention

Mỗi rule sử dụng mã `AGENT_RULE_<SECTION_ID>_<RULE_ID>`, trong đó:

- `SECTION_ID` là mã hai chữ số của section chứa rule.
- `RULE_ID` là mã hai chữ số, duy nhất trong section đó.
- Mã đã cấp là bất biến và không được đánh lại.
- Rule mới trong một section sử dụng `RULE_ID` khả dụng tiếp theo.
- Section mới sử dụng `SECTION_ID` khả dụng tiếp theo.
- Mã của rule hoặc section đã xóa không được tái sử dụng.

## 1. Project Context

- **AGENT_RULE_01_01 — Academic context:** Đây là dự án nghiên cứu thuộc học phần hệ gợi ý trong chương trình Thạc sĩ Khoa học máy tính tại HUST.

## 2. Documentation Conventions

- **AGENT_RULE_02_01 — Mathematics in Markdown files:** Khi soạn tài liệu Markdown với mục đích ghi vào file, viết công thức nội dòng bằng `$...$` và công thức khối bằng `$$...$$`; không đặt biểu thức toán học trong backtick.
- **AGENT_RULE_02_02 — Mathematics in conversation:** Khi trả lời trực tiếp trong hội thoại, không dùng ký tự `$` để bao công thức hoặc ký hiệu toán học. Dùng cú pháp hiển thị toán học phù hợp với giao diện hỏi–đáp, chẳng hạn `\(...\)` cho công thức nội dòng và `\[...\]` cho công thức khối.

## 3. Code Style

- **AGENT_RULE_03_01 — Logical grouping:** Khi định dạng code, mỗi nhóm logic chỉ được có tối đa 3 câu lệnh liên tiếp mà không có dòng trống và thực sự cần thiết, nhưng luôn ưu tiên thêm dòng trống.
- **AGENT_RULE_03_02 — Related declarations:** Có thể giữ liền tối đa 3 khai báo biến liên quan; phải dùng dòng trống để tách các nhóm logic còn lại.
- **AGENT_RULE_03_03 — Multiline statements:** Các dòng tiếp nối thuộc cùng một câu lệnh nhiều dòng không tính là câu lệnh riêng.

## 4. TDD Conventions

- **AGENT_RULE_04_01 — Skeleton before tests:** Khi phát triển theo TDD, trước khi viết test cho bất kỳ thành phần mới nào (hàm, phương thức, lớp hoặc symbol khác), luôn tạo skeleton tối thiểu để thành phần đó có thể được import và test có thể được thu thập thành công.
- **AGENT_RULE_04_02 — Expected interface:** Skeleton phải khai báo đúng tên, tham số và kiểu trả về dự kiến, nhưng chưa cài đặt hành vi thực tế.
- **AGENT_RULE_04_03 — Function placeholders:** Hàm hoặc phương thức trong skeleton phải trả về một giá trị placeholder đơn giản, đúng kiểu, chẳng hạn `0` hoặc `1` cho `int`, `0.0` cho `float`, `False` cho `bool`, chuỗi rỗng cho `str`, tuple rỗng cho `tuple`, hoặc `None` khi kiểu trả về cho phép.
- **AGENT_RULE_04_04 — Class skeletons:** Lớp mới phải có định nghĩa tối thiểu và các phương thức cần thiết ở dạng skeleton để tránh lỗi import hoặc lỗi collection.
- **AGENT_RULE_04_05 — Meaningful test failure:** Chọn placeholder sao cho test hành vi vẫn thất bại ở assertion vì chức năng chưa được cài đặt. Skeleton chỉ giúp hoàn tất import hoặc collection, không được làm test vượt qua ngoài ý muốn.
- **AGENT_RULE_04_06 — Minimal implementation:** Chỉ cài đặt phần hành vi tối thiểu để test vượt qua sau khi đã xác nhận test thất bại đúng lý do.
