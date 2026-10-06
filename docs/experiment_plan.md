# Experiment Plan: SPEC-CIR
## Structured Preserve-Edit Contrastive Training for Composed Image Retrieval

**Ngày:** 06-10-2026  
**Tác giả:** [Your name]  
**Status:** Draft v1.0

---

## 1. Scenarios — Bức tranh toàn cảnh

### 1.1 CIR là gì và người dùng gặp vấn đề gì

Composed Image Retrieval (CIR) là tác vụ: cho một **ảnh tham chiếu** và một **câu chỉnh sửa** (modification text), tìm ảnh đích từ gallery thỏa mãn cả hai.

**Ví dụ điển hình (FashionIQ):**
- Ảnh tham chiếu: áo khoác đen, cổ tròn
- Câu chỉnh sửa: *"change to red and make it longer"*
- Ảnh đích đúng: áo khoác đỏ, dáng dài — **giữ nguyên kiểu dáng, thay màu và độ dài**

Tác vụ này đòi hỏi model thực hiện **đồng thời** hai việc:
1. **Preserve** — giữ nguyên các thuộc tính không được đề cập (kiểu dáng, chất liệu, số cúc...)
2. **Edit** — thay đúng thuộc tính được chỉ định (màu → đỏ, độ dài → dài hơn)

### 1.2 Ứng dụng thực tế

| Scenario | Người dùng | Query | Expected behavior |
|---|---|---|---|
| **E-commerce fashion** | Mua sắm online | "Cái này nhưng màu xanh và không có họa tiết" | Giữ form, chất liệu; đổi màu + pattern |
| **Interior design** | Thiết kế nội thất | Ảnh ghế + "make it more minimalist, same material" | Giữ chất liệu; đơn giản hóa form |
| **Product search** | B2B procurement | Ảnh sản phẩm + "smaller version in metal" | Giữ function/design; đổi size + material |
| **Creative iteration** | Designer | Ảnh concept + "warmer tones, keep the composition" | Giữ bố cục; đổi màu sắc |

### 1.3 Benchmark landscape

| Benchmark | Domain | Metric chính | Đặc điểm |
|---|---|---|---|
| **FashionIQ** | Fashion (shirt/dress/toptee) | R@10, R@50 | Có **attribute labels** — nguồn data chính của SPEC-CIR |
| **CIRR** | Ảnh tự nhiên open-domain | R@1, R@5, R@10; R_subset@K | Subset recall kiểm tra fine-grained discrimination |
| **CIRCO** | COCO open-domain | mAP@5, mAP@10 | Multi-GT per query; nghiêm ngặt nhất |
| **GeneCIS** | Controlled attribute/object | Group R@K | 4 subsets tách biệt; test generalization |

---

## 2. Current Problem — Vấn đề hiện tại

### 2.1 Training signal không phân biệt loại lỗi

Tất cả 90+ papers trong CIR literature đều dùng **random in-batch negatives** hoặc **random hard negatives** trong contrastive loss. Điều này tạo ra một vấn đề cơ bản: model không được buộc phân biệt **hai loại lỗi hoàn toàn khác nhau**:

```
Loại A — Wrong-attribute / Right-identity:
  Query:   áo khoác đen + "change to red"  
  Target đúng:   áo khoác đỏ ✓
  Negative A: áo khoác ĐEN (cùng model, sai màu) ← model cần reject vì sai attribute
              [nhưng cực kỳ giống reference về shape/form]

Loại B — Right-attribute / Wrong-identity:
  Query:   áo khoác đen + "change to red"
  Target đúng:   áo khoác đỏ ✓  
  Negative B: áo phông đỏ (đúng màu, sai product type) ← model cần reject vì sai identity
              [nhưng có đúng target attribute]
```

Với random negatives, cả hai loại lỗi này đều rất hiếm xuất hiện trong một batch — model không có tín hiệu đủ để học phân biệt chúng.

### 2.2 Evidence từ literature (90+ papers)

| Paper | Cách xử lý negatives | Vấn đề |
|---|---|---|
| CLIP4CIR (ACM TOMM 2023) | Random in-batch | Không có cấu trúc |
| LinCIR (CVPR 2024) | Text-only; random | Không khai thác attribute |
| SPN4CIR/CSCL (ACM MM 2024) | MLLM-generated text negatives | Scale nhưng vẫn không structured theo attribute type |
| SRAIN (ECCV 2026) | Memory bank hard negatives | Theo rank, không theo attribute type |
| RankVR (ICMR 2026) | Noisy triplet handling | Giải quyết data quality, không phải negative structure |
| MRA, ConText-CIR (CVPR 2025) | Tạo data từ unlabeled images | Không khai thác annotation sẵn có |

**Kết luận:** Không có paper nào explicitly mine structured counterfactual negatives theo 2 loại lỗi kể trên.

### 2.3 Hệ quả: model học shortcut

Do thiếu tín hiệu phân biệt, model hiện tại có xu hướng:
- Dùng **dominant visual features** (shape distribution, color histogram) thay vì hiểu instruction
- Overfit vào **dataset-specific bias** (FashionIQ thiên về color/pattern; CIRR thiên về object type)
- **Generalize kém cross-domain**: LinCIR train separate checkpoint cho từng benchmark

### 2.4 Metric không phản ánh được vấn đề

Recall@K tổng hợp không tách biệt:
- Model có **giữ đúng** những gì cần giữ không? (Preserve precision)
- Model có **thay đúng** những gì cần thay không? (Edit accuracy)

MulVec (arXiv 2026) đề xuất 4 semantic roles (Global/Desired/Preserve/Forbidden) nhưng vẫn evaluate bằng standard Recall — bản thân paper thừa nhận mismatch này.

---

## 3. Proposed Approach — SPEC-CIR

### 3.1 Core Idea

**SPEC-CIR** (Structured Preserve-Edit Contrastive training for CIR) khai thác **attribute labels sẵn có** trong FashionIQ để tự động mine hai loại structured counterfactual negatives, sau đó thêm chúng vào contrastive loss để buộc model học tường minh sự khác biệt giữa preserve và edit.

Không cần:
- Data annotation mới
- LLM/VLM inference nặng
- Thay đổi kiến trúc backbone

Chỉ cần:
- Offline negative mining pipeline từ FashionIQ metadata
- Một contrastive loss term bổ sung
- ~1–2 tuần train trên 1–2 A100

### 3.2 Structured Negative Mining

**Input:** FashionIQ triplet `(r, t, p)` — reference image `r`, modification text `t`, positive target `p`  
**Output:** Hai negative hard images per triplet

#### Loại A — Wrong-attribute / Right-identity (Type-A negative)

```
Mục tiêu: tìm ảnh a_A sao cho:
  - same_product_category(a_A, r) = True  [right identity]
  - target_attribute(a_A) ≠ target_attribute(p)  [wrong attribute]
  - target_attribute(a_A) ≈ attribute(r)  [giống reference về attribute bị sửa]

Ví dụ:
  r = áo khoác đen, t = "make it red"
  p = áo khoác đỏ  ✓
  a_A = áo khoác ĐEN khác model  [nhử model vì giống r về màu]
```

**Mining procedure:**
1. Group FashionIQ images theo `(category, attribute_value)` pair
2. Với mỗi `(r, t, p)`: identify attribute được modify trong `t` (dùng keyword matching: màu → color, dài/ngắn → length, họa tiết → pattern)
3. Tìm ảnh cùng category với `r`, cùng attribute value với `r` về attribute bị modify, khác identity

#### Loại B — Right-attribute / Wrong-identity (Type-B negative)

```
Mục tiêu: tìm ảnh a_B sao cho:
  - target_attribute(a_B) = target_attribute(p)  [right attribute]
  - category(a_B) ≠ category(r)  [wrong identity / wrong category]

Ví dụ:
  r = áo khoác đen, t = "make it red"
  p = áo khoác đỏ  ✓
  a_B = áo phông ĐỎ  [nhử model vì có đúng target attribute]
```

**Mining procedure:**
1. Với mỗi `(r, t, p)`: lấy target attribute value của `p`
2. Tìm ảnh có attribute value này nhưng category khác với `r`

### 3.3 Training Objective

Tổng loss:

```
L_total = L_standard + λ_A · L_A + λ_B · L_B
```

Trong đó:

**L_standard** — standard CIR contrastive loss (InfoNCE với random in-batch negatives):
```
L_standard = -log [ exp(sim(q, p) / τ) / Σ_j exp(sim(q, n_j) / τ) ]
```

**L_A** — Type-A contrastive loss (wrong-attribute negatives):
```
L_A = -log [ exp(sim(q, p) / τ) / (exp(sim(q, p)/τ) + exp(sim(q, a_A)/τ)) ]
```

**L_B** — Type-B contrastive loss (right-attribute/wrong-identity negatives):
```
L_B = -log [ exp(sim(q, p) / τ) / (exp(sim(q, p)/τ) + exp(sim(q, a_B)/τ)) ]
```

Trong đó `q = f(r, t)` là composed query embedding.  
`λ_A, λ_B ∈ {0.1, 0.5, 1.0}` — hyperparameters cần ablate.  
`τ` = temperature (default 0.07, theo CLIP).

### 3.4 Base Model

Train SPEC-CIR loss **trên LinCIR** (CVPR 2024, arXiv:2312.01998) làm backbone:
- LinCIR là ZS-SV SOTA trên FashionIQ/CIRR, code công khai
- CLIP ViT-L/14 hoặc ViT-G/14 — chọn ViT-L/14 cho feasibility
- Tái dùng toàn bộ kiến trúc và training recipe LinCIR; chỉ thay đổi phần negative sampling

### 3.5 Evaluation Protocol

#### Primary benchmarks (required)

| Benchmark | Metric | Split |
|---|---|---|
| FashionIQ | R@10, R@50 (avg 3 categories) | Test |
| CIRR | R@1, R@5, R@10, R_subset@1, R_subset@2, R_subset@3 | Test (public leaderboard) |
| CIRCO | mAP@5, mAP@10 | Val (test labels private) |

#### Secondary benchmarks (generalization check)

| Benchmark | Mục đích |
|---|---|
| GeneCIS (4 subsets) | Kiểm tra attribute/object edit generalization |
| CIRR → FashionIQ transfer | Cross-domain test (zero-shot transfer không fine-tune) |

#### Proposed diagnostic metric — ADR (Attribute-Disentangled Recall)

Đề xuất thêm metric mới trên FashionIQ test set:

```
ADR_preserve = Recall@10 chỉ tính trên queries mà model cần PRESERVE attribute X
               (X không được đề cập trong modification text)
               → đo xem model có bị "distracted" sửa những gì không nên sửa không

ADR_edit = Recall@10 chỉ tính trên queries mà model cần EDIT attribute Y
           (Y được đề cập trong modification text)
           → đo xem model có thực sự thay đúng attribute không
```

FashionIQ cung cấp đủ metadata để compute ADR mà không cần annotation mới.

---

## 4. Experiments

### 4.1 Ablation Studies

| Exp | Mô tả | Mục đích |
|---|---|---|
| **E0** | LinCIR baseline (replicate) | Establish baseline số |
| **E1** | + L_A only (Type-A negatives) | Isolate contribution của wrong-attr negatives |
| **E2** | + L_B only (Type-B negatives) | Isolate contribution của right-attr/wrong-id negatives |
| **E3** | + L_A + L_B (λ_A=λ_B=0.5) | Full SPEC-CIR |
| **E4** | λ sweep: λ ∈ {0.1, 0.5, 1.0} cho cả λ_A, λ_B | Hyperparameter sensitivity |
| **E5** | Replace structured mining bằng random hard negatives (cùng count) | Verify structured > random |
| **E6** | CLIP ViT-L/14 vs ViT-G/14 | Backbone scaling |

### 4.2 Comparison with Baselines

| Baseline | Venue | Lý do so sánh |
|---|---|---|
| CLIP4CIR | ACM TOMM 2023 | Supervised foundation |
| LinCIR | CVPR 2024 | Direct base model |
| CIReVL | ICLR 2024 | ZS VLM-based SOTA |
| KEDs | CVPR 2024 | ZS knowledge-enhanced |
| SPN4CIR | ACM MM 2024 | Negative scaling method |
| RTD | ICCV 2025 | Post-hoc text-only framework |
| PrediCIR | CVPR 2025 | World model ZS |

### 4.3 Cross-domain Generalization Experiment

| Setting | Train | Eval | Mục đích |
|---|---|---|---|
| In-domain | FashionIQ | FashionIQ | Standard |
| Cross-domain A | FashionIQ | CIRR | Transfer từ fashion → natural |
| Cross-domain B | CIRR | FashionIQ | Transfer từ natural → fashion |
| Zero-shot C | FashionIQ | GeneCIS | Attribute generalization |

**Hypothesis:** SPEC-CIR sẽ generalize tốt hơn baseline ở cross-domain settings vì structured negatives buộc model học attribute semantics thay vì dataset-specific shortcuts.

---

## 5. Timeline

| Tuần | Milestone |
|---|---|
| 1–2 | Replicate LinCIR trên FashionIQ/CIRR (E0) — confirm baseline numbers |
| 3 | Code negative mining pipeline; verify quality bằng manual inspection 50 samples |
| 4–5 | Train E1, E2, E3 (ablation chính) |
| 6 | Train E4 (λ sweep), E5 (random vs structured comparison) |
| 7 | Evaluate ADR metric; cross-domain experiments |
| 8 | E6 (ViT-G nếu compute cho phép); tổng hợp results |
| 9–10 | Viết paper |

---

## 6. Expected Results và Success Criteria

### 6.1 Primary hypothesis

SPEC-CIR sẽ cải thiện so với LinCIR baseline:

| Metric | Baseline (LinCIR ViT-L/14) | Expected gain |
|---|---|---|
| FashionIQ Avg R@10 | ~43–45 (ZS) | +2–4 điểm |
| CIRR R@1 | ~22–25 (ZS) | +1–3 điểm |
| CIRCO mAP@5 | ~12–15 (ZS) | +1–2 điểm |
| ADR_preserve | (baseline TBD) | Cải thiện rõ |
| ADR_edit | (baseline TBD) | Cải thiện rõ |

> Số baseline ước tính từ LinCIR paper; cần xác minh bằng replication (E0) trước.

### 6.2 Secondary hypothesis (cross-domain)

SPEC-CIR sẽ giảm gap giữa in-domain và cross-domain performance so với LinCIR.

### 6.3 Failure modes cần chuẩn bị

| Scenario | Xử lý |
|---|---|
| Mining ra quá ít Type-A negatives cho một số categories | Fallback: dùng CLIP similarity để tìm nearest neighbors trong cùng category |
| λ quá lớn gây unstable training | Grid search λ ∈ {0.05, 0.1, 0.3, 0.5, 1.0} |
| Gain chỉ thấy trên FashionIQ (vì mining dùng FashionIQ attributes) | Interpret là domain-specific contribution; test CIRR để confirm robustness |
| Baseline numbers không khớp LinCIR paper | Debug data pipeline; verify split và preprocessing |

---

## 7. Differentiation — Tại sao SPEC-CIR khác prior work

| Work | Approach | Khác SPEC-CIR ở đâu |
|---|---|---|
| SPN4CIR/CSCL (ACM MM 2024) | Scale positives/negatives với MLLM text | Negatives không structured theo attribute type; cần MLLM inference |
| SRAIN (ECCV 2026) | Memory bank hard negatives | Theo rank similarity, không theo semantic attribute |
| RankVR (ICMR 2026) | Noisy triplet handling | Giải quyết annotation noise, không phải negative structure |
| MulVec (arXiv 2026) | 4-role decomposition at inference | Inference-only, không có training objective cho roles; vẫn dùng random negatives |
| ConText-CIR (CVPR 2025) | Text concept-consistency loss | Align text→image region; không mine counterfactual negatives |
| MRA (arXiv 2025) | Triplets từ unlabeled images | Không khai thác existing attribute annotation |

**SPEC-CIR là paper đầu tiên explicitly mine structured counterfactual negatives theo hai loại lỗi (wrong-attr/right-id và right-attr/wrong-id) từ attribute annotation sẵn có để dạy model preserve vs. edit disentanglement.**

---

## 8. Resources

### Compute
- 1–2 GPU A100 (40GB hoặc 80GB)
- LinCIR ViT-L/14: ~6 giờ train/epoch trên FashionIQ (ước tính từ paper); full training ~2 ngày
- SPEC-CIR: overhead nhỏ từ negative mining (offline, 1 lần); train time tương đương LinCIR

### Data
- FashionIQ: download từ official repo; attribute labels sẵn có trong metadata
- CIRR: download từ official repo (no attribute labels — dùng cho cross-domain eval)
- CIRCO: dùng val set (test labels private, phải submit leaderboard)
- GeneCIS: download từ official repo

### Code dependencies
- LinCIR official code: https://github.com/navervision/lincir
- CLIP: openai/clip (PyTorch)
- FashionIQ data loader: có sẵn trong LinCIR repo

---

## 9. Open Questions

1. **Attribute keyword extraction:** FashionIQ modification text không always explicit về attribute type — cần rule-based parser hay fine-tuned classifier?
2. **Mining ratio:** Tỷ lệ Type-A : Type-B : random negative nào tối ưu? (ablate trong E4)
3. **Extend sang CIRR:** CIRR không có attribute labels — có thể dùng CLIP-based attribute detection để tạo pseudo-labels không?
4. **ADR metric chuẩn hóa:** Cần define rõ "attribute X không được đề cập" — dùng keyword matching hay NLI model?

---

*File này sẽ được cập nhật sau khi baseline replication (E0) hoàn thành.*
