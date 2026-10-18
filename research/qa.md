# Định vị đề tài và câu hỏi nghiên cứu

> Ngày rà soát gần nhất: 17/10/2026. Tài liệu này chốt khoảng trống, cơ chế nghiên cứu, câu hỏi nghiên cứu và phạm vi đóng góp. Mô hình hóa chi tiết nằm trong [research_design.md](research_design.md); quy trình và ma trận literature review nằm trong [literature_audit.md](literature_audit.md).

## 1. Tên đề tài đề xuất

**Tiếng Việt:** *Liên minh bằng chứng nhận biết pipeline cho giải thích gợi ý top-$K$: kiểm toán khoảng cách attribution–intervention trong ranker ItemKNN–BM25–Semantic*.

**Tiếng Anh:** *Pipeline-Aware Evidence Coalitions for Top-$K$ Recommendation: Auditing the Attribution–Intervention Gap in an ItemKNN–BM25–Semantic Ranker*.

Tên này phản ánh đúng đóng góp dự kiến: đề tài không phát minh recommender mới và không nhận RAG/citation là novelty.

## 2. Vấn đề khoa học

Một ranker cộng tính có thể lưu contribution của từng history item, BM25 term và semantic match sao cho tổng contribution đúng bằng fused score của lần chạy gốc. Tuy nhiên, equality đó chưa trả lời raw evidence nào thực sự quyết định một item nằm trong top-$K$.

Khi xóa một history item và chạy lại hệ thống, source đó có thể đồng thời:

1. biến mất khỏi ItemKNN neighborhood;
2. thay đổi BM25 query và term allocation;
3. thay đổi semantic matches;
4. làm candidate vào hoặc ra candidate union;
5. thay đổi mẫu số chuẩn hóa của mọi candidate;
6. thay đổi competitor tại biên top-$K$.

Vì thế, một source có contribution lớn trong trace tĩnh có thể có intervention effect nhỏ do tín hiệu dư thừa; ngược lại, source có contribution nhỏ có thể quyết định candidate được retrieve hoặc quyết định biên top-$K$.

Đề tài gọi độ lệch vận hành này là **attribution–intervention gap**:

- **attribution:** phần score của lần chạy gốc được gán cho raw source;
- **intervention:** thay đổi top-$K$ khi raw source hoặc một coalition source bị xóa và toàn bộ pipeline được chạy lại.

Đây là tên cho đại lượng nghiên cứu trong setting cụ thể, không phải tuyên bố phát hiện một khái niệm faithfulness hoàn toàn mới.

## 3. Vì sao thiết kế cũ chưa đủ mới?

Literature audit cho thấy từng mảnh của thiết kế cũ đã có:

- PRAG đã có retrieval-grounded explanation;
- EviSnap đã có verbatim evidence, exact additive contribution, deletion và sufficiency;
- CountER đã nghiên cứu explanation vừa necessary vừa sufficient;
- PRINCE, ACCENT và Yao et al. đã tìm history subset làm recommendation thay đổi;
- LXR đã học attribution từ counterfactual loss; Refining Fidelity Metrics và SPINRec đã đánh giá removed/retained history ở explanation length cố định;
- MGIF đã chỉ ra group influence không thể luôn suy ra bằng tổng individual influence;
- TRACE và benchmark năm 2026 đã có ranking-aware/top-$K$ counterfactual evaluation;
- Why This, Not That? đã có pairwise “vì sao A hơn B”;
- PROVE-Rec đã đặt grounding–influence gap;
- HARE-Rec đã có collaborative–semantic audit;
- Selective Explanations đã có selective policy theo coverage.

Do đó, đề tài không thể lấy “exact provenance + citation + deletion + retention + fixed-budget evaluation + abstention” làm novelty chỉ vì ghép chúng trong một hệ thống.

## 4. Cơ chế nghiên cứu mới: PAEC

### 4.1 Đơn vị giải thích

Đề tài sử dụng **Pipeline-Aware Evidence Coalition (PAEC)**. Đơn vị cơ sở không phải term hoặc semantic atom độc lập mà là một raw history source $g_i$, gồm interaction trước thời điểm dự đoán và metadata hợp lệ của history item $i$.

Mọi derived atom do $g_i$ tạo ra ở ItemKNN, BM25 và semantic branch được gom vào cùng source group. Đây là đơn vị intervention hợp lệ: xóa $g_i$ nghĩa là dựng lại profile, query, candidate pools, normalization và ranking mà không có interaction đó.

Pipeline được biểu diễn như một dependency DAG từ raw source đến profile signal, branch query, candidate membership, score, normalization và rank. Khi can thiệp coalition $E$, **source-consistent intervention closure** bắt buộc tính lại mọi descendant của $E$ trong DAG; model/index artifact và factual `seen_ids` là context cố định. Cách định nghĩa này ngăn việc xóa source ở một nhánh nhưng vô tình giữ tín hiệu dẫn xuất của nó ở nhánh khác.

### 4.2 Hai phép kiểm tra của một coalition

Với coalition $E$ và recommendation $j$:

- **removal/comprehensiveness:** chạy pipeline với $H_u^+(t)\setminus E$; coalition strict-removal-valid khi việc bỏ nó làm $j$ rời top-$K$;
- **retention/sufficiency:** chạy pipeline chỉ với $E$; coalition retention-valid khi $j$ vẫn ở top-$K$ và margin không suy giảm quá tolerance đã khóa.

Một coalition đạt **dual validity** khi đồng thời qua strict removal và retention test. Độ giảm margin khi target chưa rời top-$K$ chỉ là soft comprehensiveness metric, không được tính là dual-valid. Nếu không có coalition trong evidence budget đạt điều kiện, hệ thống không được phát biểu rằng tập evidence đã “quyết định” recommendation; nó chỉ được đưa mechanistic trace thận trọng hoặc abstain.

Mỗi explanation hợp lệ phải kèm một **decision certificate** có hash/config của factual, removal và retention run; target membership; boundary competitor; candidate-set delta; score margin và source IDs. Certificate, không phải câu chữ do LLM sinh, là sản phẩm kiểm chứng chính của cơ chế.

Removal run không được tạo rank flip tầm thường do pipeline trả ít hơn $K+1$ candidate hoặc sập về empty query. Cả removal và retention run phải trả đủ candidate bằng cùng fallback policy đã khóa; nếu không, case được đánh dấu non-evaluable thay vì dual-valid.

### 4.3 Selector được đề xuất

PAEC tìm coalition nhỏ dưới budget $m$ theo thứ tự mục tiêu:

1. đạt dual validity;
2. qua raw-support gate cho loại claim dự kiến;
3. tối thiểu số raw sources;
4. tối đa removal effect;
5. tối thiểu retention loss.

Với history ngắn, dùng level-wise exhaustive search để có nghiệm chuẩn: đánh giá hết từng kích thước tăng dần và dừng sau tầng đầu tiên có coalition hợp lệ. Việc bỏ qua tầng lớn hơn là exact theo objective ưu tiên coalition nhỏ hơn; không giả định full-pipeline effect đơn điệu. Branch-and-bound theo effect chỉ được dùng nếu chứng minh bound admissible. Với history dài, dùng beam search với cùng stopping rule và báo validity gap, size gap cùng tie-break regret trên subset có thể giải exact. Không được gọi nghiệm heuristic là “minimal” nếu chưa chứng minh tối ưu.

### 4.4 Vai trò của LLM

LLM chỉ diễn đạt coalition và không nằm trên ranking path. Đóng góp chính phải đứng vững ngay cả khi bỏ LLM. Ba renderer nhận cùng coalition:

- extractive statement hoặc raw-span presentation;
- deterministic template;
- constrained LLM verbalizer.

So sánh này kiểm tra LLM có tăng usefulness/readability mà không giảm claim support hay không. Nếu không, template/extractive output là kết luận hợp lệ.

## 5. Câu hỏi nghiên cứu duy nhất

### RQ

**Trong một frozen multi-stage ItemKNN–BM25–Semantic top-$K$ recommender, exact score contribution sai lệch với full-pipeline decision influence ở mức nào, và liệu raw-source coalition selection được kiểm tra đồng thời bằng removal và retention có tạo explanation nhỏ hơn, có top-$K$ validity cao hơn và vẫn được raw evidence hỗ trợ tốt hơn các selector contribution, leave-one-out, semantic và counterfactual dưới cùng evidence budget hay không?**

RQ có hai vế gắn chặt với nhau:

1. đo sự tồn tại và nguyên nhân của attribution–intervention gap;
2. kiểm tra PAEC có xử lý được gap đó hay không.

Không tách “xây recommender” hoặc “LLM có viết hay không” thành RQ khác. Ranking quality là prerequisite; renderer utility là secondary outcome.

## 6. Giả thuyết có thể bác bỏ

- **H1 — Gap:** thứ tự raw source theo exact contribution và theo full-pipeline removal effect không đồng nhất hoàn toàn; disagreement tăng ở case có candidate-set change hoặc normalization shift.
- **H2 — Coalition:** PAEC đạt strict dual-validity rate cao hơn top-contribution, leave-one-out greedy, semantic, recency và random selector dưới cùng budget; trong các case cùng dual-valid, PAEC cần không nhiều source hơn counterfactual set-greedy.
- **H3 — Interaction:** kế thừa cảnh báo về group influence từ prior work, tổng individual leave-one-out effect dự đoán kém coalition effect ở một tỷ lệ case đáng kể do redundancy, synergy hoặc candidate-regeneration externality giữa history sources.
- **H4 — Support:** semantic-support gate kết hợp abstention làm giảm unsupported-claim risk khi coverage giảm; kết quả phải được báo bằng risk–coverage/AURC.
- **H5 — Renderer:** constrained LLM chỉ được xem có giá trị khi tăng blind usefulness so với template/extractive renderer mà không làm giảm full-support rate quá ngưỡng đã khóa.

Nếu H1 không được hỗ trợ, PAEC không có lý do khoa học mạnh hơn contribution selector và đề tài phải báo null result. Nếu H2 không được hỗ trợ, không được tuyên bố cơ chế đề xuất tốt hơn prior selectors.

## 7. Các công trình gần nhất và phần khác còn lại

| Công trình | Phần gần với đề tài | Khác biệt còn lại của đề tài |
|---|---|---|
| [EviSnap](https://arxiv.org/abs/2604.06172) | Exact additive contribution, evidence citation, deletion và sufficiency | Fixed linear rating model; không nghiên cứu cross-branch raw-source group, candidate regeneration hoặc attribution–intervention gap của hybrid top-$K$ pipeline |
| [CountER](https://doi.org/10.1145/3459637.3482420) | Minimal, necessary/sufficient aspect counterfactual | Can thiệp item aspect trên black-box scorer; không gom một raw history source xuyên candidate-generation branches |
| [PRINCE](https://doi.org/10.1145/3336191.3371824), [ACCENT](https://arxiv.org/abs/2105.05008), [Yao et al.](https://arxiv.org/abs/2211.09752) | Xóa user-history subset để đổi recommendation | Không đối chiếu exact static accounting với full-pipeline influence trong hybrid retriever/fusion có query-local normalization |
| [LXR](https://doi.org/10.1145/3589334.3645560), [Refining Fidelity Metrics](https://doi.org/10.1145/3726302.3730242) và [SPINRec](https://doi.org/10.1609/aaai.v40i17.38465) | Attribution học bằng counterfactual loss; removal/retention, top-$K$ và fixed-length fidelity metrics | Giả định recommender nhận interaction vector và trả score trên item space; không dựng lại lexical/dense candidate generators, cross-branch source group hoặc query-local fusion |
| [MGIF](https://doi.org/10.1007/s40747-024-01547-4) | Chứng minh tổng individual influence có thể biểu diễn sai group influence; tìm counterfactual group | Can thiệp training samples và cập nhật model; đề tài can thiệp input history của frozen multi-stage pipeline và phân rã externality theo stage |
| [From Top-1 to Top-K](https://arxiv.org/abs/2604.19663) và [TRACE](https://doi.org/10.1145/3773078.3831796) | List-level/top-$K$ counterfactual validity | Tập trung benchmark hoặc search trên recommender model; không nghiên cứu cross-stage provenance và candidate-set externality |
| [Why This, Not That?](https://arxiv.org/abs/2608.21662) | Pairwise rank explanation từ profile items | Contrastive explanation đã có; đề tài dùng boundary comparison như metric, không nhận pairwise explanation là mới |
| [PROVE-Rec](https://arxiv.org/abs/2610.02968) | Grounding và ranking influence | LLM proof-gated recommender; đề tài kiểm toán frozen non-LLM multi-stage pipeline và raw-source coalition |
| [HARE-Rec](https://doi.org/10.1016/j.eswa.2026.133631) | Collaborative–semantic architecture và perturbation | Cold-start routing và semantic atoms; không có lexical branch, dual coalition validation, raw claim support hoặc candidate regeneration |

Không có một paper duy nhất gần nhất trên mọi trục. **EviSnap gần nhất về tổ hợp evidence + exact accounting + intervention; SPINRec gần nhất về removed/retained history và top-$K$ fidelity; PROVE-Rec gần nhất về việc nối evidence với ranking influence.** Khác biệt phải bảo vệ của đề tài không nằm ở các phép đo này riêng lẻ, mà ở raw-source coalition được tối ưu và chạy lại xuyên ba candidate generators, union, normalization và boundary của cùng một frozen pipeline.

## 8. Novelty có thể tuyên bố một cách bảo thủ

Trong phạm vi [literature audit](literature_audit.md), chưa thấy công trình đồng thời:

1. giữ một frozen hybrid ItemKNN–BM25–Semantic pipeline có candidate generation và query-local fusion;
2. nhóm mọi derived signal theo raw history source xuyên ba nhánh;
3. đo trực tiếp disagreement giữa exact realized-score attribution và full-pipeline top-$K$ intervention;
4. tối ưu source coalition theo strict dual validity bằng removal và retention reruns của toàn pipeline;
5. chỉ verbalize khi coalition có raw-span support, đồng thời báo support–coverage và same-coalition renderer control.

Đây là **methodological mechanism và controlled empirical contribution**. Không gọi đây là thuật toán recommender nền tảng mới hoặc quan hệ nhân quả về sở thích thật.

## 9. Đóng góp dự kiến

1. Định nghĩa và đo attribution–intervention gap trong một hybrid top-$K$ recommendation pipeline.
2. Một schema cross-branch provenance gom ItemKNN, BM25 và semantic atoms về raw history source.
3. Source-consistent intervention closure trên dependency DAG và decision certificate ba trạng thái factual/removal/retention.
4. PAEC selector với dual removal–retention validation và exact-search reference trên history ngắn.
5. Một protocol so sánh contribution, leave-one-out, counterfactual, semantic, recency và random selector dưới cùng source budget.
6. Claim-support gate và selective reporting, với extractive/template/LLM renderer nhận cùng coalition.
7. Stagewise audit tách score-only, normalization và candidate-regeneration effect để giải thích nguồn gốc của gap.
8. Phân tích failure mode theo candidate change, normalization shift, redundancy, synergy, history length và branch mixture.

## 10. Những điều không tuyên bố

- Không tuyên bố exact additive decomposition, counterfactual deletion, retention/sufficiency, fixed-budget fidelity metric hoặc top-$K$ evaluation là mới.
- Không tuyên bố pairwise/contrastive recommendation explanation là mới.
- Không gọi raw-source deletion là causal effect ngoài đời thực; đó là phản ứng của pipeline đã cài đặt.
- Không gọi dense similarity là semantic support nếu không có raw span hai phía.
- Không gọi PAEC coalition là minimal khi chỉ dùng heuristic và chưa có exact reference.
- Không gọi intervention là full-pipeline nếu không dựng lại query, ba candidate pools, union, normalization và ranking.
- Primary intervention phải giữ factual `seen_ids`; nếu xóa cả eligibility thì phải báo riêng để tránh rank flip do item đã xem quay lại candidate set.
- Không dùng LLM-as-a-judge làm ground truth duy nhất.
- Không dùng cụm “lần đầu tiên”; mọi novelty claim phải kèm phạm vi literature audit và ngày chốt.

## 11. Go/no-go gate

| Gate | Điều kiện đạt | Nếu không đạt |
|---|---|---|
| Ranking prerequisite | Hybrid ranker có utility hoặc thắng có ý nghĩa trên validation | Loại nhánh không hữu ích hoặc đổi ranker chính |
| Trace identity | Contribution cộng đúng score từng nhánh và fused score | Dừng explanation experiment |
| Intervention fidelity | Rerun thật sự dựng lại query, pools, union, normalization và ranking | Chỉ gọi metric là fixed-pool influence |
| Gap existence | Có disagreement/interaction đủ lớn và CI không chỉ ra hiệu ứng gần zero | Chuyển thành replication/null-result study, không nhận PAEC là cần thiết |
| Exact reference | Có subset history ngắn giải exact để kiểm tra heuristic | Không nhận near-minimality |
| PAEC value | PAEC cải thiện primary strict dual-validity metric so với baseline đã khóa | Bác bỏ H2 và chọn baseline tốt nhất |
| Claim reliability | Human agreement đạt ngưỡng khóa trước | Sửa rubric và chạy lại pilot |
| Renderer value | LLM tăng usefulness mà không giảm support quá tolerance | Dùng template/extractive renderer |

## 12. Phát biểu ngắn cho proposal

Đề tài kiểm tra một giả định thường bị bỏ qua trong explainable recommendation: contribution cộng đúng vào score của lần chạy gốc có thực sự chỉ ra raw evidence quyết định một item nằm trong top-$K$ hay không. Trên một frozen ItemKNN–BM25–Semantic pipeline, nghiên cứu nhóm mọi tín hiệu dẫn xuất theo history source, đo attribution–intervention gap khi chạy lại toàn bộ retrieval và ranking, rồi so sánh một selector coalition-aware có dual removal–retention validation với các selector hiện có dưới cùng evidence budget. Natural-language generation chỉ là renderer thứ cấp và phải qua raw-evidence support gate.
