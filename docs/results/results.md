# Kết quả và tiến độ

Số liệu của mọi thí nghiệm, kể cả các kết quả âm. Cách chạy lại từng thí nghiệm: [../guide.md](../guide.md).

## Quy ước báo cáo

- Metric viết dạng **%**: NDCG@10 4,91%, HR@10 7,68%, Recall@50 11,9% (dấu phẩy thập phân). Mức thay đổi tương đối ghi rõ là tương đối, ví dụ "+14% so với ItemKNN".
- Trọng số fusion viết dạng **% tổng 100%**, ví dụ ItemKNN 47,6%, BM25 4,8%, popularity 47,6%. Chỉ tỉ lệ giữa các trọng số ảnh hưởng thứ hạng; CLI nhận giá trị tương đối (`--weight itemknn=1 --weight bm25=0.1 --weight popularity=1` tương đương 47,6% / 4,8% / 47,6%). % trọng số **không** phải tỉ lệ ảnh hưởng thực tế lên thứ hạng; muốn đo đóng góp thì dùng ablation.
- Mỗi bảng ghi rõ tập và cỡ mẫu (toàn bộ test, mẫu 50k test, mẫu 10k/20k validation).
- Ghi rõ loại metric: **đúng mã** (target đúng `item_id`, metric gốc) hay **cùng phim** (bất kỳ phiên bản nào của phim target, cần `--editions`).

## Tiến độ (cập nhật 2026-10-18)

| Hạng mục | Tác dụng trong đồ án | Trạng thái |
|---|---|---|
| Nhập 3 split | Biến CSV gốc thành bảng chuẩn và **tách target đánh giá ra file query riêng**, để model không bao giờ thấy đáp án. Là nền cho mọi bước sau. | Xong; sắp theo user để đọc nhanh |
| Nối metadata, `genres`, `item_doc` | Gắn nội dung phim vào từng item và tạo một văn bản mô tả chung. Không có bước này thì không làm được BM25, embedding, bằng chứng [E#]. | Xong |
| Gom phiên bản (`editions.parquet`) | Biết item nào là cùng một phim (DVD/Blu-ray/Prime): không gợi ý lại phim đã xem dưới bản khác, không lặp một phim trong danh sách, đo đúng "trúng phim", cho phim Prime mượn title. | Xong 17/10, sửa 18/10 (Twilight); 133.694 phim từ 197.943 item |
| ① Hồ sơ người dùng | Tóm tắt sở thích của user **tại thời điểm gợi ý**; đầu vào cho bằng chứng [H#]. | Xong, ~0,3 s/user |
| ② ItemKNN | Nhánh CF chính: "người xem A thường xem B", ưu tiên phim vừa xem; `because` cho sẵn bằng chứng. | Xong, đã tune và đánh giá |
| ② BPR-MF | Baseline học máy cổ điển (run B). | Xong |
| Popularity | Mốc thấp nhất (run A) và nhánh trong fusion. | Xong |
| ② BM25 | Phim chung **từ hiếm** (đạo diễn, diễn viên, thể loại) với phim user thích (run C). | Xong; test NDCG@10 1,35% |
| ② Embedding (Dense) | Phim giống về **ý nghĩa** dù khác từ (run D). | Xong; test NDCG@10 1,16% (mẫu 50k); không dùng trong fusion |
| ③ Fusion | Trộn các nhánh thành 50 ứng viên; kỳ vọng tốt hơn nhánh tốt nhất (run E). | Xong; test NDCG@10 5,60% (+14% so với ItemKNN); **hệ thống cuối** = fusion + gom phiên bản |
| Đánh giá | Đo khách quan, cùng một giao thức cho mọi run A–G. | Xong |
| Kho review | 9,25 triệu review làm nguồn bằng chứng ý kiến. | Xong |
| ④ Gói bằng chứng | Đưa cho LLM chỉ những gì có nguồn: [H#] lịch sử, [E#] metadata và review trước `as_of`. | Xong; đã chạy trên 6 user thật (ví dụ bên dưới), < 2 s cho 3 ứng viên |
| Xếp lại top 20 (LLM, cross-encoder, learning-to-rank) | Thử xem có cách nào xếp lại pool của fusion tốt hơn không (run F). | **Bỏ khỏi đề tài 18/10**: không cách nào vượt thứ tự fusion (kết quả âm, bảng bên dưới) |
| ⑤ LLM giải thích top 3 có trích dẫn, ⑥ validator, giám khảo | Lý do kèm [E#]/[H#] cho từng gợi ý; validator loại câu không kiểm được; LLM giám khảo đo tỉ lệ câu đúng nguồn. **Đóng góp chính.** | Xong code 18/10 (Gemini `gemini-3.5-flash-lite`, giám khảo `gemini-3.5-flash`); smoke test 10 record; chưa chạy 200 user |
| Demo end-to-end (`recsys-demo`) | Một lệnh: hồ sơ → fusion → bằng chứng → giải thích → validator. | Xong 18/10 |
| Run G 200 user, chấm tay, báo cáo, slide | Số liệu chính cho câu hỏi "giải thích bắt buộc trích dẫn có đáng tin hơn"; viết báo cáo. | Chưa làm |

Kết quả trên toàn bộ 657.203 query test:

| Phương pháp | NDCG@10 | HR@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity (A) | 0,91% | 1,71% | 4,5% |
| Dense — MiniLM-L6 (D), **mẫu 50k test** | 1,16% | 2,26% | 4,3% |
| BPR-MF, 64 factors, 60 epoch (B) | 1,31% | 2,27% | 4,5% |
| BM25 — 50 từ truy vấn (C) | 1,35% | 2,73% | 6,2% |
| ItemKNN, 400 hàng xóm, $\gamma = 0{,}7$ (B) | 4,91% | 7,68% | 11,9% |
| **Fusion theo điểm** — ItemKNN 47,6%, BM25 4,8%, popularity 47,6% (E) | **5,60%** | **8,99%** | **14,2%** |

Mọi dòng là toàn bộ 657.203 query test, trừ Dense (mẫu cố định 50.000 query, seed 42; sai số NDCG@10 ~±0,1%). Nhánh nội dung đứng riêng chỉ nhỉnh hơn Popularity; Dense còn có Recall@50 thấp hơn Popularity (4,3% so với 4,5%).

Fusion so với ItemKNN trên toàn bộ test: NDCG@10 +14%, HR@10 +17%, Recall@50 +19% (tương đối) — khớp mức tăng trên validation (+15% NDCG, +19% Recall@50), nên cấu hình tune trên validation đứng vững. Đánh đổi: độ phủ catalog trong top 50 giảm từ 99,8% (ItemKNN) xuống 92,6% vì prior popularity kéo gợi ý về phim phổ biến.

**Gom phiên bản (17/10, số chạy lại 18/10 sau khi sửa gom Twilight)** — mẫu 50.000 query test (seed 42), model pha test; "đúng mã" = target đúng `item_id` (metric gốc), "cùng phim" = bất kỳ phiên bản nào của phim target (bước 10):

| Phương pháp | NDCG@10 đúng mã | HR@10 đúng mã | Recall@50 đúng mã | NDCG@10 cùng phim | HR@10 cùng phim | Recall@50 cùng phim |
|---|---:|---:|---:|---:|---:|---:|
| ItemKNN | 4,91% | 7,72% | 11,9% | 5,51% | 8,72% | 13,6% |
| Fusion (E) | 5,63% | 9,02% | 14,2% | 6,33% | 10,18% | 16,2% |
| **Fusion + gom phiên bản (hệ thống cuối)** | 5,57% | 8,81% | 13,6% | **6,46%** | **10,40%** | **16,6%** |

- **Metric gốc đánh giá thấp mọi phương pháp ~12%** (fusion 5,63% → 6,33% khi tính cùng phim): gợi ý bản DVD khi target là bản Blu-ray vẫn là gợi ý đúng. Từ nay báo cả hai; metric cùng phim là chính cho hệ thống cuối.
- Gom phiên bản trong fusion: **+2,0% NDCG@10, +2,2% HR@10, +2,2% Recall@50** (tương đối, metric cùng phim) — chỗ của các bản trùng được nhường cho phim khác. Target rating ≥ 4: 7,20% → 7,35%. Theo metric đúng mã thì giảm nhẹ (5,63% → 5,57%) vì chỉ giữ một bản mỗi phim, không nhất thiết đúng bản target — hệ quả mong muốn, không phải lỗi.
- Validation (mẫu 10k, cùng phim): fusion 7,64% / 11,91% / 17,9% → có gom 7,69% / 12,04% / 18,2% — cùng chiều với test.
- Giá phải trả: loại mọi phiên bản của phim đã xem làm mất các target "mua lại" (target là bản khác của phim đã xem) — 0,12% query test.
- Độ phủ catalog (mẫu 50k): 69,2% → 69,4%.
- Lần đầu (ngưỡng 2% cho mọi khóa) cho 6,43% / 10,38% / 16,6%; sửa 18/10 (cùng title cho lệch 5%, chuẩn hóa "X, The:", "Part One", năm) chỉ đổi kết quả ở chữ số thứ hai nhưng bỏ được lỗi gợi ý lại phim đã xem thấy trong smoke test.

**Phân tích theo nhóm (18/10)** — mẫu 50k test, metric cùng phim, NDCG@10 / Recall@20 / Recall@50 (`runs/analysis/segments_test50k.json`):

| Nhóm | Số query | ItemKNN | BM25 | Fusion + gom phiên bản |
|---|---:|---|---|---|
| Tất cả | 50.000 | 5,51% / 10,8% / 13,6% | 1,57% / 4,6% / 6,7% | 6,46% / 13,0% / 16,6% |
| Lịch sử 3–5 phim | 20.533 | 6,36% / 12,1% / 15,0% | 1,81% / 5,4% / 7,8% | 7,31% / 14,3% / 18,2% |
| Lịch sử 6–20 | 25.276 | 5,17% / 10,3% / 13,2% | 1,46% / 4,2% / 6,1% | 6,10% / 12,7% / 16,2% |
| Lịch sử > 20 | 4.191 | 3,44% / 6,7% / 8,5% | 1,06% / 3,1% / 4,6% | 4,49% / 8,7% / 10,7% |
| Target phổ biến (top 1%) | 13.150 | 9,56% / 20,6% / 26,6% | 1,44% / 5,4% / 8,7% | 11,20% / 25,2% / 34,2% |
| Target trung bình (9% tiếp) | 15.619 | 5,34% / 9,4% / 11,4% | 1,71% / 4,7% / 6,5% | 6,67% / 12,3% / 14,5% |
| Target hiếm (90% còn lại) | 21.231 | 3,13% / 5,7% / 7,1% | 1,55% / 4,0% / 5,5% | 3,37% / 6,0% / 7,2% |
| Target có title | 21.115 | 5,90% / 10,4% / 12,6% | 2,11% / 5,6% / 7,8% | 6,80% / 12,4% / 14,7% |
| Target không có title | 28.885 | 5,23% / 11,0% / 14,3% | 1,17% / 3,9% / 5,8% | 6,21% / 13,5% / 18,0% |

- Tập 5-core nên không có user 1–2 phim. User có lịch sử dài lại khó đoán hơn (gu đa dạng, target cuối cùng khó).
- Fusion giúp nhiều nhất với **phim phổ biến** (Recall@50 26,6% → 34,2%, do prior popularity), gần như không giúp **phim hiếm** (7,1% → 7,2%). Phim hiếm là chỗ BM25 tương đối mạnh nhất (gần một nửa ItemKNN, so với ~1/7 ở phim phổ biến) nhưng vẫn thua ItemKNN → không có nhóm nào nhánh nội dung thắng nhánh CF.
- BM25 tốt gấp đôi khi target có title (2,11% so với 1,17%): `item_doc` của phim không title chỉ có tên người.
- **Recall@20 = 13,0%**: chỉ khoảng 1/8 user có phim đúng trong 20 ứng viên đầu — trần của mọi bước xếp lại sau fusion.

**Xếp lại pool top 20 — kết quả âm, bỏ khỏi đề tài (18/10).** Smoke test trên validation, 50 user có target trong pool (50 user đầu của mẫu 5.000 query, seed 42), metric cùng phim; mã đã xóa khỏi repo, số liệu ở `runs/llm/dev-inpool-50/`, `runs/rerank/`:

| Cách xếp lại | NDCG@10 | HR@10 | P(tốt hơn fusion)* | Chi phí |
|---|---:|---:|---:|---|
| **Giữ thứ tự fusion** | **53,7%** | **86%** | — | 0 |
| Learning-to-rank (HistGradientBoosting, 11 đặc trưng, train 29.747 user validation khác) | 51,9% | 80% | 16% | vài phút CPU |
| LLM `gemini-3.5-flash-lite`, có thẻ bằng chứng | 48,7% | 76% | 15% | ~2.500 token/user |
| LLM, chỉ title | 44,6% | 72% | 2% | ~1.500 token/user |
| Cross-encoder `BAAI/bge-reranker-base` | 31,3% | 58% | 0% | 15,8 s/user (CPU) |

\* bootstrap 2.000 lần trên 50 user. Learning-to-rank trên 690 user có target trong pool: 54,2% so với 53,7% của fusion; trên cả 5.000 user 7,47% so với 7,41% — ngang nhau. Permutation importance: bỏ "hạng fusion" làm NDCG giảm 13,7 điểm, mọi đặc trưng khác < 1 điểm → thứ tự fusion đã gần tối ưu với các tín hiệu có sẵn. Cross-encoder và LLM chỉ đọc nội dung nên không thấy tín hiệu "người xem phim A xem tiếp phim B". Điểm đáng ghi: thẻ bằng chứng giúp LLM xếp tốt hơn (48,7% so với 44,6%). LLM rerank bị fallback 1/50 lần (lặp mã C4), dù đã ép JSON schema.

**Smoke test 10 record, hệ thống cuối (18/10)** — chi tiết từng bước ở [docs/results/smoke_test_10_record.md](smoke_test_10_record.md); 5 user test ngẫu nhiên (1/5 trúng) + 5 user có target trong top 10:

| Giải thích (30 phim) | Số câu | Phim có giải thích | supported | unsupported |
|---|---:|---:|---:|---:|
| Tự do (cùng thông tin, không mã nguồn) | 60 | 30 | 72% | 17% |
| Bắt buộc trích dẫn, sau validator | 33 | 27 | 88% | 3% |

- Bắt buộc trích dẫn giảm câu sai rõ rệt, đổi lại nói ít hơn và từ chối 3/30 phim (`evidence_unavailable`). Câu sai của bản tự do chủ yếu là gán sở thích không có trong lịch sử.
- Validator: kiểm ID/phim/tên không bắt lỗi nào lần này; lọc độ giống (0,3) loại nhầm 1 câu đúng và giữ 1 câu sai → chưa hữu ích, cần hiệu chỉnh hoặc bỏ.
- Vấn đề định tính: câu đúng nguồn nhưng không phải lý do (review kể đọc sách Tolkien hồi nhỏ); trích ý khen nhỏ trong review chê; dẫn thừa nguồn (2,1 nguồn/câu).
- Chi phí hệ thống cuối: 1 lời gọi LLM/user, ~1.600 token vào + ~300 ra, p50 1,6 s.
- Smoke test tìm ra lỗi gom phiên bản (fan Twilight được gợi ý lại *Eclipse* và *Breaking Dawn Part 1* đã xem) → đã sửa `editions.py`.

**Các smoke test LLM trước đó (validation, 18/10)** tìm và sửa: giám khảo `gemini-3.5-flash` dùng hết ngân sách output cho "thinking" → JSON bị cắt (nay thinking low, 4.096 token, không cache câu trả lời bị cắt); model trả mảng thay vì object (nay ép `responseSchema`); model chèn `[E#]` vào câu (nay tách ra); validator bắt nhầm tên chính phim đang giải thích (nay cho phép).

**Ví dụ gợi ý + bằng chứng (hệ thống cuối, 17/10)** — 5 query test cố định (`Random(42).sample(…, 5)`) và user Harry Potter; đầy đủ ở `runs/examples/recommend_5_users_editions.{txt,json}` và `runs/examples/evidence_example.json`:

| User | Lịch sử (gần nhất) | Target (rating) | Top 3 hệ thống cuối | Kết quả |
|---|---|---|---|---|
| `AGXV…Q7Q` | Harry Potter 2 (xem 18 giây trước target), Mr. Church… | Harry Potter 3 (5★) | Harry Potter 3, Harry Potter 4, Harry Potter 1 | **Trúng hạng 1** |
| `AHKI…GJN3Q` | Heavy Metal, Akira, Ice Age, A Bug's Life (33 phim) | Three Amigos (5★) | Krull, Fire and Ice, Ice Age: Collision Course | Trượt; top 10 đúng gu anime/fantasy (Ghost in the Shell, Paprika, Vampire Hunter D), target là phim hài |
| `AFUC…D7P4Q` | Groundhog Day, Best Exotic Marigold Hotel, Elf | The Peanut Butter Falcon (3★) | It Happened on 5th Avenue, Holiday Inn, The Hundred-Foot Journey | Trượt; đúng gu phim nhẹ nhàng, Giáng sinh |
| `AFAG…TBWQ` | The Walking Dead, Frozen, Pixels (4 phim) | Một mùa The Walking Dead (5★), bản Prime | Wings, Ant-Man, **cùng mùa The Walking Dead (bản khác)** | Trượt theo đúng mã, **trúng hạng 3 theo cùng phim** |
| `AFM7…Z6A` | Forrest Gump, Sling Blade, O Brother Where Art Thou? | Battleship (3★) | Cast Away, Fargo, National Lampoon's Christmas Vacation | Trượt; gợi ý hợp lý |
| `AEOU…LHA` | Thích Cast Away (4★), ghét 3 phim | The Jackal (**1★**) | The People vs. Larry Flynt, Detachment, Ikiru | Trượt; top 10 trước khi gom có **4 mục là Cast Away (3 bản) và bản trùng Polar Express**, nay đã hết |

Bằng chứng mẫu (user Harry Potter, top 3):

| ID | Nguồn (`source_id`) | Nội dung |
|---|---|---|
| H1 | `history:B002DQDV6Y` | 5★ "Harry Potter and the Chamber of Secrets (Widescreen Edition)" — title mượn từ bản DVD |
| E1 | `meta:B01KW24RCC:title` | "HP & the Prisoner of Azkaban SE" — title mượn cho bản Prime `B00271DNP4` |
| E2 | `meta:B00271DNP4:people` | "Directors: Alfonso Cuarón. Cast: Daniel Radcliffe, Rupert Grint, Emma Watson" |
| E3 | `review:5340230` | Review của một phụ huynh: con xem 3 lần trong 2 ngày, "the best of the Potter movies she'd seen yet" |
| E5 | `meta:B000E6EK38:title` | "Harry Potter and the Goblet of Fire" (ứng viên 2) |
| E9 | `meta:B075W1L85B:title` | "Harry Potter and the Sorcerer's Stone" (ứng viên 3) |

Sửa được nhờ chạy ví dụ (so với lần chạy đầu): review nói về sách ("this book was full of surprises") bị loại; "Directors: Greg Nicotero, Michael E. Cast:…" bị cắt tên → đủ tên; dòng "Directors: Various. Cast: Various" của Fargo bị loại; `B002DQLGHU` mượn nhầm title "Harry Potter: 8-Film Collection" → chặn gộp bộ sưu tập, nay đúng "Sorcerer's Stone"; [H#] của phim Prime nay có title.

Vấn đề bằng chứng còn lại: review nói về dịch vụ chứ không về phim vẫn lọt ("Used a 24 hour rental… No issues with play", "will not purchase… until CC is included"); hiếm hơn là review xếp nhầm phim (một review về video sửa mặt bếp nằm ở The Walking Dead); 57.036 item vẫn không có title (không phiên bản nào có title) — bằng chứng chỉ có tên người.

Kết quả trên 20.000 query validation (seed 42), model pha validation:

| Nhánh (run) | NDCG@10 | HR@10 | Recall@50 | Thời gian/user |
|---|---:|---:|---:|---:|
| Popularity (A) | 1,11% | 2,14% | 5,2% | 0,3 ms |
| Dense — MiniLM-L6, 64 token (D) | 1,40% | 2,86% | 5,4% | 8,4 ms |
| BM25 — 50 từ truy vấn (C) | 1,65% | 3,44% | 7,6% | 3,4 ms |
| ItemKNN (B) | 5,86% | 8,95% | 13,4% | ~3 ms |
| Fusion RRF, tốt nhất trong 64 tổ hợp | 5,86% | 8,95% | 13,4% | — |
| **Fusion theo điểm (E)**: ItemKNN 47,6%, BM25 4,8%, popularity 47,6% — mẫu 10k | **6,96%** | — | **16,0%** | — |

Đứng riêng, hai nhánh nội dung chỉ nhỉnh hơn Popularity; ItemKNN vượt trội vì "phim xem tiếp" phụ thuộc mạnh vào hành vi đám đông và thứ tự thời gian. Giá trị của BM25/Dense nằm ở chỗ **bổ sung** phim mà ItemKNN bỏ sót, đo bằng fusion.

**Fusion RRF (kết quả âm):** grid 64 tổ hợp trọng số (bm25, dense, popularity ∈ {0; 0,25; 0,5; 1}, itemknn = 1) — tổ hợp tốt nhất là ItemKNN đứng riêng; thêm nhánh nào cũng làm NDCG@10 giảm (bm25 0,25 → 5,74%; dense 0,25 → 5,66%; popularity 0,25 → 5,44%) và Recall@50 gần như không đổi. Hai lý do: RRF chỉ dùng hạng nên bỏ mất độ tự tin của ItemKNN; mỗi nhánh chỉ góp top 50 nên phim mới hầu như không vào được top 50 (hạng 1 của BM25 ở trọng số 0,25 được 0,25/61 < 1/110 của hạng 50 ItemKNN). Ngược lại, cộng prior popularity vào **điểm** ItemKNN trên toàn bộ ứng viên từng tăng NDCG 18% → thử fusion theo điểm, danh sách sâu hơn.

**Fusion theo điểm (cấu hình chốt):** điểm mỗi nhánh chia cho điểm cao nhất của nhánh đó với user, cộng $w_{pop} \cdot (1 - \text{hạng popularity}/N)$, mỗi nhánh góp 200 ứng viên. Grid 45 tổ hợp trên 10k query validation: ItemKNN đứng riêng 6,03% / 13,5% → + popularity 1,0: 6,93% / 16,1% → + BM25 0,1: **6,96% / 16,0%** (chọn). Gần như toàn bộ mức tăng (+15% NDCG, +19% Recall@50) đến từ prior popularity; BM25/Dense chỉ thêm ~0,03%, nhỏ hơn sai số mẫu. Kết luận cho báo cáo: với bài toán "phim xem tiếp", nhánh nội dung hầu như không tăng độ chính xác; giá trị của chúng là bằng chứng và giải thích.

**Embedding xếp lại trong pool (kết quả âm, 17/10):** thay vì tìm trên toàn catalog bằng vector trung bình, cộng $w \cdot \cos(\text{phim vừa thích}, j)$ (hoặc max có trọng số trên 3 phim thích gần nhất) vào điểm fusion của các ứng viên. Cùng mẫu 10k validation: mốc 6,96% / 16,04%; tốt nhất $w = 0{,}25$ (phim vừa thích) 6,99% / 15,94% — tăng 0,03%, nhỏ hơn sai số mẫu (~±0,2%), Recall@50 giảm; $w \ge 0{,}5$ giảm rõ. Không đưa vào fusion. Lý do: ItemKNN đã nắm tín hiệu "phim vừa xem"; `item_doc` của nhiều phim chỉ có tên người. Ví dụ user mẫu: embedding trung bình đặt Harry Potter 3 ở hạng 5.298, dùng riêng phim vừa xem → hạng 37, trong khi ItemKNN xếp hạng 1. → Embedding dùng cho **bằng chứng và kiểm tra lời giải thích** ở bước LLM.

BM25: chỉ giữ 50 từ mạnh nhất (trọng số × idf) trong truy vấn thay vì toàn bộ ~380 từ: nhanh hơn 5 lần (25 → 4,9 ms/user), NDCG@10 1,56% → 1,65%, Recall@50 8,0% → 7,6%.

**Quy ước đánh giá trên test (quyết định 17/10):**

- Các run nằm trong hệ thống cuối hoặc là baseline chính (Popularity, BPR, ItemKNN, BM25, Fusion) được đánh giá trên **toàn bộ** 657.203 query test.
- Dense **không** nằm trong fusion cuối (trọng số 0 trên validation), nên chỉ được đánh giá trên **mẫu cố định 50.000 query test** (`--query-limit 50000`, `Random(42)` như mọi mẫu khác), kết quả ở `runs/dense-movies-tv-2023/test_evaluation_50k.json`. Lý do: đánh giá toàn bộ mất ~90 phút trên máy này (mỗi user chấm cả 198k phim), trong khi mẫu 50k mất ~7 phút và sai số NDCG@10 chỉ ~±0,1% — đủ để báo cáo một nhánh yếu. Trong bảng kết quả phải ghi rõ "mẫu 50k" cạnh dòng Dense; không so sánh trực tiếp nó với các dòng toàn bộ test ở mức chênh lệch nhỏ hơn ~0,2%.
- Thử nghiệm và tune luôn làm trên **validation** (mẫu 10k/20k); test chỉ chạy với cấu hình đã chốt.

Test thấp hơn validation khoảng 13–16% với **mọi** phương pháp (target test là tương tác cuối cùng, khó hơn), nên không phải dấu hiệu tune quá khớp.

Phát hiện khác (validation):

- Trọng số độ gần đây được giữ ($\gamma = 0{,}7$). Ablation không dùng nó ($\gamma = 1$) làm NDCG@10 của ItemKNN giảm khoảng 17%.
- BPR yếu vì bỏ qua thứ tự thời gian, chưa hội tụ (AUC còn tăng ở epoch 60) và dữ liệu thưa.
- Embedding chạy CPU chậm: bge-small (~29 tài liệu/s) được thay bằng MiniLM-L6 với 64 token (~51 tài liệu/s); ghi là hạn chế trong báo cáo.
