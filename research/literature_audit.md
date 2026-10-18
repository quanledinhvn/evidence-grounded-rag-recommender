# Literature audit cho đề tài giải thích gợi ý top-$K$

> Ngày chốt tìm kiếm: 17/10/2026. Tài liệu này ghi lại phạm vi tìm kiếm, tiêu chí sàng lọc và các công trình tạo ranh giới novelty cho [qa.md](qa.md) và [research_design.md](research_design.md). Đây là một literature audit có phạm vi, không phải tuyên bố đã bao phủ mọi công trình từng công bố.

## 1. Phạm vi và quy trình tìm kiếm

### 1.1 Nguồn

- ACM Digital Library và chương trình RecSys, SIGIR, KDD, WSDM, CIKM;
- ACL Anthology;
- AAAI và IJCAI proceedings;
- arXiv cho preprint mới chưa vào proceedings;
- trang nhà xuất bản của Elsevier, Springer và MIT Press khi paper không có bản proceedings mở;
- survey và reference chaining từ các paper biên.

Blog, trang tổng hợp và kết quả tìm kiếm chỉ được dùng để phát hiện paper. Kết luận về novelty phải dựa trên abstract, full text hoặc trang proceedings chính thức.

### 1.2 Nhóm truy vấn

Các truy vấn được chạy bằng biến thể của các nhóm sau:

1. `explainable recommendation` kết hợp `faithfulness`, `factuality`, `citation`, `provenance`;
2. `counterfactual recommendation` kết hợp `top-k`, `list-level`, `history deletion`, `pairwise`;
3. `exact additive decomposition`, `sufficiency`, `comprehensiveness`, `necessary and sufficient` trong recommendation;
4. `selective explanation`, `abstention`, `risk coverage`;
5. `hybrid recommender`, `multi-retriever`, `candidate generation`, `ranking pipeline`, `BM25`, `semantic` và `provenance`;
6. `fixed-length fidelity`, `removed history`, `retained history`, `insertion`, `deletion`, `path integration` trong recommendation;
7. `group influence`, `coalition interaction`, `non-additive influence`, `counterfactual group` trong recommendation;
8. backward/forward chaining từ PRAG, CountER, LXR, SPINRec, PROVE-Rec, EviSnap, HARE-Rec và benchmark counterfactual top-$K$ năm 2026.

### 1.3 Tiêu chí đưa vào

Một công trình được đưa vào ma trận biên nếu thỏa ít nhất một điều kiện:

- giải thích recommendation bằng raw evidence hoặc citation;
- nối explanation với scoring/ranking mechanism;
- dùng deletion, masking, counterfactual hoặc sufficiency test;
- đánh giá explanation ở mức top-$K$ hoặc list-level;
- có abstention/selective policy liên quan trực tiếp tới explanation;
- có kiến trúc hybrid collaborative–lexical–semantic hoặc multi-stage retrieval ảnh hưởng tới cách định nghĩa intervention.

### 1.4 Giới hạn của audit

- Preprint có thể thay đổi sau ngày chốt; venue và kết quả phải được kiểm tra lại trước khi nộp luận văn.
- Không dùng cụm “lần đầu tiên” nếu chưa có systematic review theo PRISMA và forward-citation search trên cơ sở dữ liệu thư mục đầy đủ.
- Novelty được phát biểu dưới dạng “trong phạm vi audit này chưa thấy công trình đồng thời thỏa...” và phải hạ xuống nếu tìm thấy prior work mới.

## 2. Kết luận sau sàng lọc

Thiết kế cũ không thể nhận các thành phần sau là mới:

- evidence-cited recommendation explanation;
- exact additive score decomposition;
- deletion, sufficiency hoặc comprehensiveness test;
- minimal history deletion và counterfactual explanation;
- learned counterfactual attribution, removed/retained-history metrics và fixed-length fidelity evaluation;
- tính không cộng được của group influence khi tìm counterfactual group;
- pairwise “why A rather than B” explanation;
- top-$K$/list-level counterfactual evaluation;
- grounding–influence gap;
- collaborative–semantic hybrid hoặc semantic-atom perturbation;
- selective explanation, abstention hoặc risk–coverage khi xét độc lập.

Khoảng trống có thể bảo vệ hơn là **pipeline-aware evidence selection**: một raw history source đồng thời tạo tín hiệu ở nhiều nhánh, thay đổi query, candidate union, mẫu số chuẩn hóa và tập competitor khi bị xóa. Vì vậy, contribution cộng đúng vào score của lần chạy gốc có thể không đồng nhất với influence của source lên quyết định top-$K$ sau khi chạy lại toàn pipeline.

## 3. Các paper tạo ranh giới trực tiếp

| Công trình | Phần đã có | Hệ quả đối với đề tài |
|---|---|---|
| [PRAG](https://doi.org/10.1609/aaai.v37i11.26618) | Personalized retrieval, review-grounded generation, entailment/factuality | Không nhận retrieval grounding hoặc factual generation là mới |
| [ERASER](https://aclanthology.org/2020.acl-main.408/) | Rationale comprehensiveness và sufficiency | Không nhận keep/remove evidence evaluation là mới |
| [LOGER](https://aclanthology.org/2021.naacl-main.245/) | Model-linked reasoning path cho recommendation | Không nhận model-faithful path explanation là mới |
| [CountER](https://doi.org/10.1145/3459637.3482420) | Minimal aspect changes, necessity/sufficiency, recommendation reversal | Không nhận minimal counterfactual hoặc necessary/sufficient explanation là mới |
| [PRINCE](https://doi.org/10.1145/3336191.3371824) | Minimal user actions làm recommendation thay đổi | Không nhận history deletion explanation là mới |
| [ACCENT](https://arxiv.org/abs/2105.05008) | Pairwise influence và counterfactual set cho neural recommender | Không nhận boundary-gap deletion là mới |
| [Yao et al.](https://arxiv.org/abs/2211.09752) | Học surrogate cho ảnh hưởng của xóa tập con history | Không nhận coalition history deletion hoặc tăng tốc bằng surrogate là mới |
| [LXR](https://doi.org/10.1145/3589334.3645560) | Model-agnostic attribution học bằng self-supervised counterfactual loss và recommender-specific fidelity metrics | Không nhận learned counterfactual attribution hoặc perturbation fidelity metric là mới |
| [Refining Fidelity Metrics](https://doi.org/10.1145/3726302.3730242) | Tách explaining/contradictory features và đánh giá ở fixed explanation length | Mọi selector phải so ở cùng source budget; không nhận fixed-budget fidelity evaluation là mới |
| [SPINRec](https://doi.org/10.1609/aaai.v40i17.38465) | Stochastic path attribution; removed/retained history; top-$K$, deletion và insertion metrics | Không nhận removal–retention hoặc fixed-length top-$K$ fidelity test là mới |
| [MGIF](https://doi.org/10.1007/s40747-024-01547-4) | Chỉ ra tổng individual influence có thể sai group influence; iterative counterfactual-group search | Không nhận non-additive group effect hoặc greedy group search là mới |
| [Are We Explaining the Same Recommenders?](https://doi.org/10.1145/3640457.3691709) | Chất lượng recommender làm thay đổi kết luận khi đánh giá counterfactual explainer | Ranking utility là prerequisite và phải được báo cùng explainer result |
| [Xu et al.](https://doi.org/10.1145/3605357) | Model-agnostic faithful explanation và scrutability cho top-$N$ | Cần so với faithfulness framework, không chỉ generative baselines |
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | Citation correctness và completeness | Không nhận citation metrics là mới |
| [RecExplainer](https://doi.org/10.1145/3637528.3671802) | Căn chỉnh LLM với black-box recommender | Không nhận việc đưa collaborative signal vào LLM là mới |
| [XRec](https://aclanthology.org/2024.findings-emnlp.22/) | Collaborative adaptor cho LLM explanation | Latent alignment không thay cho raw-source trace |
| [Selective Explanations](https://arxiv.org/abs/2405.19562) | Chọn case cần explainer chất lượng cao theo coverage budget | Không nhận selective explanation là khái niệm mới |
| [FIRE](https://arxiv.org/abs/2508.05225) | SHAP-guided structured language generation | Exact trace có thể khác SHAP nhưng attribution-guided generation đã có |
| [EviSnap](https://arxiv.org/abs/2604.06172) | Verbatim evidence, exact additive per-concept contribution, deletion và sufficiency | Đây là biên gần nhất với tổ hợp citation + exact accounting + intervention |
| [From Top-1 to Top-K](https://arxiv.org/abs/2604.19663) | Benchmark 11 counterfactual explainers ở item/list level | Không nhận top-$K$ counterfactual evaluation là mới |
| LIME-RS, SHAP, LXR và GREASE trong benchmark trên | Implicit attribution, explicit interaction/graph perturbation, sparsity và runtime | PAEC phải được đặt cạnh cả attribution lẫn explicit-set counterfactual, không chỉ score selector |
| [TRACE](https://doi.org/10.1145/3773078.3831796) | Targeted ranking-aware counterfactual search cho sequential recommendation | Không nhận rank-aware targeted counterfactual search là mới |
| [Why This, Not That?](https://arxiv.org/abs/2608.21662) | Pairwise interpretation từ user-profile counterfactuals | Không nhận contrastive “A thay vì B” là mới |
| [Cao et al.](https://arxiv.org/abs/2608.30333) | Cross-model feature masking cho LLM-cited rationale | Không nhận cited-feature masking là mới |
| [PROVE-Rec](https://arxiv.org/abs/2610.02968) | Evidence–claim grounding và claim–ranking influence | Không nhận grounding–influence gap là mới |
| [HARE-Rec](https://doi.org/10.1016/j.eswa.2026.133631) | Collaborative–semantic routing và semantic-atom perturbation | Không nhận hybrid audit hoặc semantic perturbation là mới |
| [StaR](https://arxiv.org/abs/2604.03724) | Xếp hạng atomic explanatory statements thay vì sinh tự do | Cần có baseline extractive/ranking, không chỉ LLM vs template |

## 4. Ma trận so sánh của các biên gần nhất

Ký hiệu: `✓` là trọng tâm rõ ràng; `◐` là có một phần; `—` là không phải trọng tâm; `P` là phần đề tài phải triển khai và kiểm chứng.

| Công trình | Raw evidence | Exact accounting | History/feature set intervention | Top-$K$ list-level | Chạy lại candidate generation | Group raw source xuyên nhiều nhánh | Claim support | Selective risk–coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CountER | ◐ | — | ✓ | ◐ | — | — | — | — |
| PRINCE / ACCENT | ✓ | — | ✓ | ◐ | — | — | — | — |
| Yao et al. | ✓ | — | ✓ | ◐ | — | — | — | — |
| LXR / Refined Metrics / SPINRec | ◐ | ◐ | ✓ | ✓ | — | — | — | — |
| EviSnap | ✓ | ✓ | ✓ | — | — | — | ◐ | — |
| Top-$K$ CE benchmark | ◐ | — | ✓ | ✓ | — | — | — | — |
| TRACE | ✓ | — | ✓ | ✓ | — | — | — | — |
| Why This, Not That? | ✓ | — | ✓ | ◐ | — | — | — | — |
| PROVE-Rec | ✓ | — | ✓ | ◐ | — | — | ◐ | — |
| HARE-Rec | ◐ | ◐ | ✓ | ✓ | — | — | — | — |
| Đề tài dự kiến | P | P | P | P | P | P | P | P |

Hàng cuối không chứng minh novelty chỉ bằng việc có nhiều dấu `P`. Benchmark top-$K$ năm 2026 giả định recommender chấm các unseen item từ interaction vector/graph; nó không đánh giá việc một intervention dựng lại nhiều candidate generators và query-local fusion. Novelty phụ thuộc vào kết quả thực nghiệm cho thấy pipeline-aware coalition khác thực chất so với contribution, leave-one-out và counterfactual baseline; nếu không, đề tài phải được báo như replication/negative result.

## 5. Khoảng trống nghiên cứu được giữ lại

### 5.1 Attribution–intervention gap ở multi-stage hybrid ranker

Với lần chạy gốc, một additive trace có thể cộng chính xác về fused score. Tuy nhiên, xóa một raw history item có thể đồng thời:

1. bỏ ItemKNN neighbors;
2. đổi BM25 query và term allocation;
3. đổi semantic query/chunk matches;
4. làm item vào hoặc ra candidate union;
5. đổi mẫu số chuẩn hóa của mọi candidate;
6. đổi competitor tại biên top-$K$.

Do đó, exact accounting là ground truth của **score đã quan sát**, không phải ground truth của **decision influence sau intervention**. Audit chưa tìm thấy paper recommender nào lấy chính độ lệch này trong một hybrid candidate-generation pipeline làm đối tượng nghiên cứu chính.

### 5.2 Raw-source coalition thay vì atom độc lập

Một history item sinh nhiều derived atoms ở nhiều nhánh. Xóa riêng một term hoặc một semantic pair tạo input không tương ứng với hành động dữ liệu hợp lệ. Đơn vị intervention chính phải là raw source group. Nhiều source có thể dư thừa hoặc tương tác, nên leave-one-out từng source không nhất thiết tìm được tập giải thích tốt. MGIF đã chỉ ra vấn đề tổng individual influence đối với counterfactual group; đề tài chỉ nhận phần mới hẹp hơn là đo tương tác do source đi xuyên nhiều stage của frozen serving pipeline.

Đề tài sẽ nghiên cứu một **pipeline-aware evidence coalition** được kiểm tra hai chiều:

- bỏ coalition khỏi history phải làm recommendation mất tư cách top-$K$ trong strict test; độ giảm signed boundary margin chỉ là metric mềm phụ;
- chỉ giữ coalition phải bảo toàn tư cách top-$K$ và có retention loss trong tolerance đã khóa.

Necessity/sufficiency, removed/retained history và fixed-length top-$K$ fidelity nói chung không mới. Phần chưa được xác nhận trong audit là việc **tối ưu strict dual-valid coalition ở cấp raw source**, rồi chạy cả hai test xuyên candidate generation, query reconstruction, normalization và fusion của một hybrid ranker.

Cơ chế triển khai phải dùng source-consistent intervention closure trên dependency DAG và xuất decision certificate của factual/removal/retention run. Đây là cách làm cho “chạy lại toàn pipeline” kiểm toán được; bản thân provenance graph hoặc run hash riêng lẻ không được nhận là novelty.

## 6. Quy tắc cập nhật audit

Trước proposal defense và trước test freeze:

1. chạy lại đúng các nhóm truy vấn ở Mục 1.2;
2. kiểm tra forward citations của LXR, SPINRec, EviSnap, PROVE-Rec, HARE-Rec, TRACE và benchmark top-$K$;
3. thêm mọi paper mới có ít nhất hai trong ba thuộc tính: multi-stage pipeline intervention, cross-branch raw-source grouping, dual top-$K$ sufficiency/comprehensiveness;
4. cập nhật ma trận và hạ novelty claim nếu có prior work gần hơn;
5. lưu ngày tìm kiếm, query và quyết định include/exclude để audit có thể tái lập.
