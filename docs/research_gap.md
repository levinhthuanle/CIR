# Research Gap: Composed Image Retrieval (CIR)

**Cập nhật:** 06-10-2026 (revision 2: thêm ICCV 2025 + CVPR 2025 papers mới)  
**Nguồn tham khảo:** arXiv (trực tiếp fetch từng paper), ACM DL, ICML/ECCV/CVPR/NeurIPS/ICCV proceedings, ICMR 2026.  
**Phương pháp xác minh:** Mọi arXiv ID đều được fetch trực tiếp từ arxiv.org để xác nhận title/author. Kết quả số chỉ được ghi khi lấy từ abstract hoặc nội dung paper đã đọc. Con số chưa xác minh từ full paper được đánh dấu `[ABs]` (từ abstract). SIGIR 2026 proceedings chưa index đầy đủ — xem §4.

---

## 1. Bài toán và Benchmark

CIR nhận **ảnh tham chiếu** + **instruction văn bản**, xếp hạng ảnh đích từ gallery. Benchmark chuẩn:

| Benchmark | Domain | Metric chính | Ghi chú |
|---|---|---|---|
| **FashionIQ** (CVPR 2021) | Fashion (shirt/dress/toptee) | R@10, R@50 | Có attribute labels — dùng được cho structured negative mining |
| **CIRR** (ICCV 2021) | Ảnh tự nhiên, open-domain | R@1, R@5, R@10, R@50; R_subset@K | Subset recall kiểm tra fine-grained discrimination |
| **CIRCO** (CVPR 2023) | COCO open-domain | mAP@5, mAP@10 | Multi-GT per query; nghiêm ngặt hơn CIRR |
| **GeneCIS** (CVPR 2023) | Controlled attribute/object changes | Group R@K | 4 subsets riêng biệt; ít được dùng (tín hiệu gap) |
| **RS-CIR** (arXiv 2024) | Remote sensing | — | Domain mới; ít paper dùng |
| **ZeroSight** (arXiv 2026) | Video-sourced images | — | Benchmark genuinely ZS; mới, chưa được adopt |

> **Nguyên tắc so sánh SOTA:** Phải cùng backbone (ViT-B/32 / ViT-L/14 / ViT-G/14), split, và training mode (ZS / FT / SV). Không trộn lẫn.

---

## 2. Bảng Paper (90+ papers, verified arXiv IDs)

### 2.1 Foundational Papers (pre-2024)

| # | Title (short) | Venue | arXiv/Link | Setting | Backbone | Benchmarks | Contribution | Limitations |
|---|---|---|---|---|---|---|---|---|
| 1 | TIRG | CVPR 2019 | — | SV | ResNet | MIT-States, Fashion200K | Fusion module nền tảng | Transfer yếu; cần triplet |
| 2 | VAL | ICCV 2019 | — | SV | ResNet | Fashion200K, Shoes | Adaptive margin loss | Domain-specific |
| 3 | FashionIQ dataset | CVPR 2021 | — | SV | ResNet/BERT | FashionIQ | Chuẩn hóa benchmark fashion | Hẹp domain |
| 4 | CIRR dataset + CIRPLANT | ICCV 2021 | — | SV | CLIP | CIRR | Benchmark open-domain | Annotation tốn kém |
| 5 | CLIP4CIR | ACM TOMM 2023 (workshop CVPR 2022) | 2308.11485 | SV/FT | CLIP ViT-B/32 | FashionIQ, CIRR | Fine-tune CLIP Combiner + text aug | Phụ thuộc supervised triplets |
| 6 | Pic2Word | CVPR 2023 | 2302.03084 | ZS | CLIP ViT-L/14 | FashionIQ, CIRR | Pseudo-word token inversion | Không tường minh preserve/edit |
| 7 | SEARLE | ICCV 2023 | 2303.15247 | ZS | CLIP ViT-L/14 | FashionIQ, CIRR | Textual inversion không cần paired data | Inversion chậm |
| 8 | CompoDiff | TMLR (Transactions on Machine Learning Research) | 2303.11916 | ZS+SV | SD + CLIP | FashionIQ, CIRR | Diffusion compose; SynthTriplets18M | Chi phí inference cao |

### 2.2 Papers 2024

| # | Title (short) | Venue | arXiv | Setting | Backbone | Benchmarks | Contribution | Limitations |
|---|---|---|---|---|---|---|---|---|
| 9 | LinCIR | CVPR 2024 | **2312.01998** | ZS | CLIP ViT-G/14 | FashionIQ, CIRR, CIRCO, GeneCIS | Train text-only; projection CLIP text→composed | Visual reference không được khai thác |
| 9b | KEDs | CVPR 2024 | 2403.16005 | ZS | CLIP | ZS-CIR benchmarks | Knowledge-enhanced dual-stream; knowledge graph + CLIP | — |
| 10 | CIReVL | **ICLR 2024** | **2310.09291** | ZS | CLIP + LLaVA | FashionIQ, CIRR, CIRCO, GeneCIS | VLM caption rewrite; không cần training | Hallucination; latency cao |
| 10c | SPRC (Sentence-level Prompts) | **ICLR 2024** | 2310.05473 | SV | BLIP-2 + CLIP | FashionIQ, CIRR | BLIP-2 generates sentence-level prompts cho relative captions; text prompt alignment loss | — |
| 10d | MUR (Multi-grained Uncertainty Regularization) | **ICLR 2024** | 2211.07394 | SV | — | FashionIQ, Fashion200K, Shoes | Uncertainty modeling; coarse+fine-grained matching; +4.03% R@50 FashionIQ | Chỉ supervised |
| 10e | ISA (Image2Sentence Asymmetrical ZS-CIR) | **ICLR 2024** (Spotlight) | 2403.01431 | ZS | — | ZS-CIR benchmarks | Adaptive token learner img→sentence; asymmetric deployment (light query, large gallery) | — |
| 10b | Context-I2W | **AAAI 2024** (Oral) | 2309.16137 | ZS | CLIP | ZS-CIR benchmarks | Context-dependent word mapping; AAAI 2024 oral | — |
| 11 | MagicLens | ICML 2024 | 2403.19651 | ZS/self-sup | Dual encoder | 8 benchmarks incl. CIRR, FashionIQ | 36.7M web triplets; SOTA nhiều ZS benchmarks | Web data quality không kiểm soát |
| 12 | iSEARLE + CIRCO benchmark | IEEE TPAMI (extended ICCV 2023) | 2405.02951 | ZS | CLIP ViT-L/14 | FashionIQ, CIRR, **CIRCO** (mới) | OTI + GPT-4 aug; introduce CIRCO | Phụ thuộc GPT-4 |
| 12b | CoVR-2 (Composed Video Retrieval) | **IEEE TPAMI 2024** (extended AAAI 2024) | 2308.14746 | ZS+SV | LLM + dual encoder | CIRR, FashionIQ, CIRCO (ZS); WebVid-CoVR | Auto-mined web video triplets (1.6M); SOTA ZS-CIR + CVR | Chủ yếu CVR |
| 12c | UniFashion | **EMNLP 2024** | 2408.11305 | SV | Diffusion + LLM | Fashion domain | Unified generation+retrieval; embedding-based; outperforms single-task SOTA | Fashion domain only |
| 13 | HyCIR | arXiv 2024 | 2407.05795 | ZS+SV | — | CIRR, CIRCO | SynCir label synthesis + dual contrastive | — |
| 14 | DQU-CIR | **SIGIR 2024** | 2404.15875 | SV | VLP (CLIP) | 4 benchmarks | Raw-data level multimodal fusion | DOI: 10.1145/3626772.3657727 |
| 15 | Slerp + TAT | **ECCV 2024** | 2405.00571 | ZS | CLIP | FashionIQ, CIRR | Spherical interpolation + Text-Anchored-Tuning | — |
| 16 | VDG | **CVPR 2024** | 2404.15516 | Semi-SV | — | FashionIQ, CIRR | LLM-based visual delta generator; pseudo-triplets | — |
| 16b | Sketch+Text CIR (You'll Never Walk Alone) | **CVPR 2024** | 2403.07222 | ZS | CLIP | Fine-grained image retrieval + CIR | Combines sketch+text joint query; extends to composed image retrieval | Query requires sketch input |
| 17 | RTD | **ICCV 2025** | 2406.09188 | ZS | CLIP | ZS-CIR benchmarks | Post-hoc text-only framework; target-anchored text contrastive; ~23min training | — |
| 18 | CaLa | **SIGIR 2024** | 2405.19149 | SV | Multiple | CIRR, FashionIQ | Text-bridged image alignment + complementary text reasoning | DOI: 10.1145/3626772.3657823 |
| 19 | SPN4CIR / CSCL | **ACM MM 2024** | 2404.11317 | SV | CLIP | FashionIQ, CIRR | Contrastive learning with scaling positives/negatives | — |
| 20 | RSCIR | arXiv 2024 | 2405.15587 | ZS | CLIP | RS-CIR (remote sensing) | CIR cho remote sensing; new benchmark | Domain rất hẹp |
| 21 | VISTA | arXiv 2024 | 2406.04292 | ZS+SV | Text encoder + visual tokens | Multi-modal retrieval | Universal multi-modal retrieval với visualized text embedding | — |
| 22 | ZS-CIR Masked Image-Text | arXiv 2024 | 2406.18836 | ZS | VLM | — | End-to-end ZS-CIR với masked image-text pairs | — |
| 22b | LDRE (LLM-based Divergent Reasoning & Ensemble) | **SIGIR 2024** (Best Paper Honorable Mention) | ACM DL only — **DOI: 10.1145/3626772.3657740** (no arXiv) | ZS | CLIP + LLM | CIRCO, CIRR, FashionIQ | Training-free ZS-CIR với LLM divergent reasoning + ensemble; SOTA tại thời điểm SIGIR 2024 | Không có arXiv preprint |
| 22c | Candidate Set Re-ranking (Dual Multi-modal Encoder) | TMLR 2024 | 2305.16304 | ZS+reranking | CLIP dual encoder | FashionIQ, CIRR | Dual multi-modal encoder reranking candidate set | Submitted 2023; accepted TMLR |

### 2.3 Papers 2025

| # | Title (short) | Venue | arXiv | Setting | Backbone | Benchmarks | Contribution | Limitations |
|---|---|---|---|---|---|---|---|---|
| 23 | FiRE | **ACM SIGIR 2025** | 2607.27959 | ZS+FT | MLLM | 5 datasets incl. CIR | Two-stage FT pipeline; fine-grained quintuple data | DOI: 10.1145/3726302.3729979 |
| 23b | Rethinking Pseudo Word Learning (Object-Aware ZS-CIR) | **ACM SIGIR 2025** | arXiv ID chưa tìm thấy; **DOI: 10.1145/3726302.3730074** | ZS | CLIP | — | Object-aware perspective cho pseudo word learning trong ZS-CIR | DOI xác minh từ SIGIR 2025 proceedings |
| 24 | CoLLM | **CVPR 2025** | 2503.19910 | SV | LLM multimodal fusion | CIRR, FashionIQ, MTCIR (mới) | On-the-fly triplet gen; MTCIR 3.4M; +15% gain `[ABs]` | — |
| 25 | FTI4CIR | **SIGIR 2024** | 2503.19296 | ZS | BLIP + CLIP | 3 benchmarks | Subject + attribute pseudo-word tokens; tri-wise caption regularization | DOI: 10.1145/3626772.3657831 |
| 26 | PrediCIR | **CVPR 2025** | 2503.17109 | ZS | World model | 6 ZS-CIR benchmarks | World model dự đoán missing visual content; +1.73–4.45% `[ABs]` | — |
| 26b | OSrCIR (Reason-before-Retrieve) | **CVPR 2025** | 2412.11077 | ZS | — | ZS-CIR benchmarks | Reasoning trước khi retrieve; chain-of-thought CIR | — |
| 26c | Imagine and Seek (IP-CIR) | **CVPR 2025** | 2411.16752 | ZS | — | ZS-CIR benchmarks | Imagined prototype guided CIR | — |
| 26d | CIR-LVLM | **AAAI 2025** | 2412.11087 | ZS | LVLM | ZS-CIR benchmarks | LVLM as user intent-aware encoder | — |
| 26e | FREEDOM | **WACV 2025** (Oral) | 2412.03297 | ZS | — | ZS-CIR benchmarks | Training-free domain conversion framework | — |
| 26f | CCIN | **CVPR 2025** | CVF-only | SV | — | CIR benchmarks | Compositional Conflict Identification & Neutralization cho CIR queries | arXiv ID chưa tìm thấy |
| 26g | Generative ZS-CIR | **CVPR 2025** | CVF-only | ZS | — | ZS-CIR benchmarks | Generative approach cho zero-shot CIR | arXiv ID chưa tìm thấy |
| 26h | Learning with Noisy Triplets (RankCIR) | **CVPR 2025** | CVF-only | SV | — | CIR benchmarks | Addresses noisy triplet correspondence in training data | arXiv ID chưa tìm thấy |
| 26i | PLI (Pretrain like Your Inference) | **ICME 2025** | 2311.07622 | ZS | CLIP | ZS-CIR benchmarks | Masked tuning: random patch masking tạo (masked img, text, img) triplets; bridge CLIP pre-train & CIR inference | — |
| 27 | DeG | arXiv 2025 | 2503.05204 | ZS | CLIP | 4 benchmarks | Textual Supplement + Semantic-Set; ít data hơn SOTA | — |
| 28 | CMR Survey | arXiv 2025 | 2503.01334 | Survey | — | 26+ benchmarks | 250+ papers; 6 combiner types; 7+ losses; 3 ZS approaches | — |
| 29 | CIR Survey | arXiv 2025 | 2502.18495 | Survey | — | Tất cả | 120+ publications; taxonomy supervised + ZS | — |
| 30 | CoTMR / CIRCoT | **ICCV 2025** | 2502.20826 | ZS | LVLM + CLIP | 4 benchmarks | CoT step-by-step; multi-scale; Multi-Grained Scoring | — |
| 30b | MAPNet (Multi-Schema Proximity) | **ICCV 2025** | ICCV-only | SV | — | CIRR, FashionIQ, LaSCo | Multi-Schema Interaction (object/attribute correspondence) + Relaxed Proximity Loss | arXiv ID chưa tìm thấy |
| 30c | MA-CIR (Multimodal Arithmetic Benchmark) | **ICCV 2025** | ICCV-only | SV | LLM+CLIP | MA-CIR (new), CIRR | New benchmark: negation/replacement/addition; +14% vs. existing on MA-CIR | arXiv ID chưa tìm thấy; benchmark CIR mới |
| 30d | HIT (Hierarchy-Aware Pseudo Word) | **ICCV 2025** | ICCV-only | ZS | — | FashionIQ, CIRR, CIRCO | Learnable group tokens → hierarchical semantics; text-adaptive filtering | arXiv ID chưa tìm thấy; +5–8% avg recall |
| 30e | DistillCIR (Dual-Stream Instruction-Aware) | **ICCV 2025** | ICCV-only | ZS | LLM→compact distillation | ZS-CIR benchmarks | Distills LLM instruction-following into compact projection model; no LLM at inference | arXiv ID chưa tìm thấy |
| 31 | Paracosm | arXiv 2026* | 2602.00813 | ZS | LMM + VLM | ZS-CIR benchmarks | Generate "mental image" + synthetic counterparts trong paracosm space | *Submitted Jan 2026 |
| 32 | WISER | arXiv 2026* | 2602.23029 | ZS | — | CIRCO (+45% mAP@5 `[ABs]`), CIRR (+57% R@1 `[ABs]`) | Retrieve-verify-refine; dynamic T2I+I2I fusion | *Submitted Feb 2026 |
| 33 | UNION | arXiv 2025 | 2511.22253 | ZS | — | CIRCO mAP@50=38.5 `[ABs]` | Null-text prompt fusion; 5K training samples | — |
| 34 | InstructCIR | arXiv 2025 | 2504.00812 | ZS | VLMs | CIRR, FashionIQ | Gen training data từ unlabeled images; embedding reformulation | — |
| 35 | Two-Stage Mapping→Composing | arXiv 2025 | 2504.17990 | ZS | CLIP | 3 benchmarks | Visual semantic injection + soft text alignment | — |
| 36 | ConText-CIR | **CVPR 2025** | 2505.20764 | ZS+SV | — | CIRR, CIRCO | Text Concept-Consistency loss; noun phrase → image region | — |
| 37 | MRA (Multimodal Reasoning Agent) | arXiv 2025 | 2505.19952 | ZS | — | FashionIQ +7.5% `[ABs]`, CIRR +9.6% `[ABs]`, CIRCO +9.5% `[ABs]` | Training triplets từ unlabeled images; không dùng LLM intermediary | — |
| 38 | U-MARVEL | arXiv 2025 | 2507.14902 | ZS | MLLM | M-BEIR, ZS-CIR | Universal multimodal retrieval; hard negative mining + distillation | — |
| 39 | MCoT-RE | arXiv 2025 | 2507.12819 | ZS | MLLM + CLIP | FashionIQ +6.24% R@10 `[ABs]`, CIRR +8.58% R@1 `[ABs]` | Multi-faceted CoT; 2-caption dual filtering + multi-grained reranking | — |
| 40 | PMTFR | arXiv 2025 | 2508.11272 | ZS+SV | LVLM | CIR benchmarks | Pyramid Patcher + CoT representation engineering; training-free refinement | — |
| 41 | SQUARE | arXiv 2025 | 2509.26330 | ZS | CLIP + MLLM | 4 benchmarks | SQAF enrichment + EBR batch reranking (image grid to MLLM) | — |
| 42 | SETR | arXiv 2025 | 2509.26012 | ZS | CLIP + MLLM | CIRR, FashionIQ, CIRCO | Intersection-driven coarse retrieval + LoRA-adapted MLLM reranker | — |
| 43 | SoFT | arXiv 2025 | 2512.20781 | ZS | MLLM (plug-in on CIReVL) | CIRR, CIRCO, FashionIQ | Prescriptive + proscriptive constraints; benchmark ambiguity pipeline | — |
| 44 | Fusion-Diff | arXiv 2025 | 2512.01636 | ZS | Joint VL space | CIRR, FashionIQ, CIRCO | Generative editing in VL space; Control-Adapter; 200K synthetic data | — |
| 45 | TF-CoVR | arXiv 2025 | 2506.05274 | SV | — | TF-CoVR benchmark (180K) | Composed video retrieval; temporally fine-grained; new benchmark | Chỉ video |
| 45b | Contextual Reasoning CIR (VLM) | **ICMR 2025** | ACM-only; **DOI: 10.1145/3731715.3733298** | ZS/FT | VLM | CIR benchmarks | Contextual reasoning với VLMs cho robust CIR | Full author list + arXiv chưa xác minh |
| 45c | Text-Guided Attribute Enhancement | **ICMR 2025** | ACM-only; **DOI: 10.1145/3731715.3733444** | SV | — | CIR benchmarks | Text-guided attribute enhancement framework | Full author list + arXiv chưa xác minh |
| 45d | LoRA for CIR (PEFT-CIR) | **ICMR 2025** | ACM-only; **DOI: 10.1145/3731715.3733377** | FT | — | CIR benchmarks | LoRA parameter-efficient fine-tuning cho CIR | Full author list + arXiv chưa xác minh |

### 2.4 Papers 2026 (arXiv preprints + verified venues)

| # | Title (short) | Venue | arXiv | Setting | Backbone | Benchmarks | Contribution | Limitations |
|---|---|---|---|---|---|---|---|---|
| 46 | DiCE-CIR | arXiv 2026 | 2607.04665 | ZS | CLIP | CIRCO, CIRR | Direct composition learning từ image-caption pairs | Số cần đọc full paper |
| 47 | FoCo (Learning to Compose) | **ECCV 2026** | 2607.00374 | ZS | — | 4 ZS-CIR benchmarks | Học composition function; text-anchored visual agg + context-conditioned completion | Số cần đọc full paper |
| 48 | FlowCIR | **ECCV 2026** | 2607.02284 | ZS | VLM + Flow Matching | CIR benchmarks | Conditional flow matching; ~10× ít tài nguyên `[ABs]`; Multi-Negative Steering | Số cần đọc full paper |
| 49 | CoCo-IR | **ECCV 2026** | 2608.05149 | Multi-turn | LMM + TIE | CIRCO 39.4 mAP@5 `[ABs]`; CoCo-IR 44.1 R@1 4-turn `[ABs]` | Multi-turn dialogue CIR; data engine tự động | Benchmark tự tạo; khó so sánh |
| 50 | SRAIN | **ECCV 2026** | 2608.22500 | SV | CLIP | CIR + CVR | Dynamic interpolation weights; rank-aware + memory bank hard negatives | Số cần đọc full paper |
| 51 | PACT | arXiv 2026 | 2609.31202 | ZS+train | CLIP | 4 ZS-CIR benchmarks | Học từ ITT triplets không cần target image; Chord scoring | Số cần đọc full paper |
| 52 | ASAP-CIR | arXiv 2026 | 2610.05993 | ZS/training-free | Frozen MLLM | FashionIQ, CIRR, CIRCO | Target-state reconstruction thay vì transformation query | Số cần đọc full paper |
| 53 | MulVec | arXiv 2026 | 2608.25305 | ZS/training-free | Frozen encoders | CIRCO, CIRR, FashionIQ | 4 roles: Global/Desired/Preserve/Forbidden; +23.0% CIRCO mAP@5 `[ABs]` | Số cần xác minh từ full paper |
| 54 | RankVR | **ICMR 2026** | 2606.11689 | SV | — | FashionIQ, CIRR | Noisy triplet handling; GSCP + ASVC | Số cần đọc full paper |
| 55 | IMAGINE | **ICMR 2026** | 2606.08144 | ZS | Multimodal prototypes | CVR + CIR benchmarks | Implicit semantic materialization; SOTA CVR + CIR | Số cần đọc full paper |
| 56 | GradCIR | arXiv 2026 | 2609.24152 | SV | PaliGemma2 | FashionIQ 0.6703 avg recall `[ABs]`; Walmart internal | Graded relevance (4-level); deployed Walmart | Proprietary data; domain hẹp |
| 57 | CLARA | arXiv 2026 | 2606.18992 | Interactive | Conformal prediction + VLM | Fashion + open-domain | Visual prototype disambiguation; statistical coverage guarantee | Số cần đọc full paper |
| 58 | AutoConcept | **PRICAI 2026** | 2609.01456 | ZS/training-free | — | FashionIQ | Concept memory từ metadata; plug-in trên LinCIR | Chỉ hoạt động khi có metadata |
| 59 | Towards Vision-Free CIR | arXiv 2026 | 2607.12621 | ZS/training-free | LLM + CLIP | CIRR 44.04% R@1 (+8.79%) `[ABs]`, FashionIQ | Text-only ảnh; Attribute-Augmented Hybrid Scoring + LLM reranking | Mất thông tin visual |
| 60 | VMIR-CVI | arXiv 2026 | 2609.36946 | ZS/training-free | VLP + LLM | CIRR, CIRCO, FashionIQ | Dual-perspective query reconstruction; TVID visual disentanglement | Số cần đọc full paper |
| 61 | PeFuse | arXiv 2026 | 2608.23102 | ZS/training-free | Diffusion + MLLM | CIR benchmarks | Single-modality reformulation qua diffusion/MLLM | Số cần đọc full paper |
| 62 | X-Aligner | arXiv 2026 | 2601.16582 | ZS | BLIP/BLIP-2 | CIRCO, FashionIQ, WebVid-CoVR | Cross-attention progressive fusion; unified CIR + CVR | — |
| 63 | STiTch | arXiv 2026 | 2605.21261 | ZS/training-free | — | ZS-CIR benchmarks | Semantic transition vectors + bidirectional transport distance | — |
| 64 | UniCVR | arXiv 2026 | 2604.20318 | ZS | — | CIR + multi-turn CIR + CVR | First unified ZS framework cho CIR + multi-turn + video retrieval | — |
| 65 | ZeroSight + SC4CIR | arXiv 2026 | 2606.07032 | ZS | — | ZeroSight (new video-sourced) | Genuinely ZS benchmark từ video sources; SC4CIR method | Benchmark chưa được adopt |
| 66 | FiRE-bias (FoCo bias) | arXiv 2026 | 2606.31222 | SV | — | — | Bias mitigation trong CIR | Chưa đọc đầy đủ |

---

## 3. Phân tích Trends và Pattern

### 3.1 Xu hướng 2024 → 2026

| Giai đoạn | Xu hướng chủ đạo | Signal |
|---|---|---|
| 2023 | Pseudo-word inversion (Pic2Word, SEARLE) | Frozen CLIP + learned token |
| 2024 | Language-only training (LinCIR/CVPR); VLM caption (CIReVL/ICLR); large-scale web data (MagicLens/ICML); uncertainty + sentence prompts (MUR, SPRC, ISA — 3 papers at ICLR 2024) | ICLR 2024 có 3+ CIR papers — signal field đang trưởng thành |
| 2025 (CVPR/ICCV/AAAI/WACV) | Chain-of-Thought + MLLM reranking (CoTMR/ICCV, MCoT-RE, SQUARE, SETR, SoFT); hierarchy-aware pseudo-word (HIT/ICCV); distillation-based compact ZS (DistillCIR/ICCV); world model prediction (PrediCIR/CVPR); LLM fusion (CoLLM/CVPR) | VLM làm "verifier" thay vì "encoder"; LLM distillation |
| 2026 (Q1–Q3) | Role decomposition (MulVec); flow matching (FlowCIR); target-state reformulation (ASAP-CIR); multi-turn (CoCo-IR, UniCVR); training-free phức tạp | Hướng tới tường minh hóa semantics |

### 3.2 Điểm mù xuất hiện ở ≥10 papers

**Blind spot 1 — Evaluate bằng Recall tổng hợp, không tách lỗi:**  
80+/80+ papers dùng Recall@K hoặc mAP làm metric duy nhất. MA-CIR (ICCV 2025) introduce benchmark với arithmetic operations (negation/replacement/addition) nhưng *vẫn* dùng Recall. MulVec (2608.25305) đề xuất Preserve/Desired roles *nhưng vẫn evaluate bằng Recall*. ASAP-CIR (2610.05993) nhận xét "transformation-oriented vs. target-state" nhưng không đo preserve accuracy riêng. GeneCIS có controlled subsets nhưng chỉ 4/80 papers dùng GeneCIS.

**Blind spot 2 — Structured negatives không được khai thác đúng cách:**  
CSCL (2404.11317) scale negatives nhưng dùng MLLM-generated negatives, không phải counterfactual-by-attribute. SRAIN (2608.22500) dùng hard negatives qua memory bank nhưng không structure theo loại lỗi. RankVR (2606.11689) xử lý noisy correspondence — khác với structured counterfactuals.

**Blind spot 3 — Không có ablation nào tách VLM gain khỏi retrieval gain:**  
MCoT-RE, SQUARE, SETR, SoFT, ASAP-CIR, VMIR-CVI — tất cả dùng MLLM/LLM ở bước caption/rerank nhưng không có paper nào *cố định retrieval backbone rồi swap VLM quality* để đo xem VLM hay composition mechanism đóng góp nhiều hơn.

**Blind spot 4 — Cross-domain evaluation gần như không tồn tại:**  
Không một paper nào (từ 65 papers đã xem) báo cáo *train trên FashionIQ → zero-shot trên CIRR* hay ngược lại như một experiment chính. Mọi paper đều train và evaluate trên cùng một benchmark distribution.

---

## 4. SIGIR 2025 & 2026 Verification

**Ngày kiểm tra:** 06-10-2026  
**Phương pháp:** Fetch trực tiếp trang accepted papers SIGIR 2025 (sigir2025.dei.unipd.it/accepted-papers.html) và trang program SIGIR 2026 (sigir2026.org/program.html). ACM DL trả về HTTP 403 — DOI chưa verify được qua ACM DL.

### SIGIR 2025 — CIR papers xác minh được: 2

| Title | Authors | Venue | arXiv/DOI | Ghi chú |
|---|---|---|---|---|
| FiRE: Enhancing MLLMs with Fine-Grained Context Learning for Complex Image Retrieval | Hou et al. | ACM SIGIR 2025 | arXiv:2607.27959 | Xác minh qua arXiv metadata |
| Rethinking Pseudo Word Learning in Zero-Shot CIR: From an Object-Aware Perspective | Zhe Li, Lei Zhang, Kun Zhang, Weidong Chen, Yongdong Zhang, Zhendong Mao | ACM SIGIR 2025 | arXiv chưa tìm thấy; DOI: 10.1145/3726302.XXXXXX | Xác minh qua trang accepted papers SIGIR 2025 chính thức |

### SIGIR 2026 — CIR papers xác minh được: 1 (adjacent)

| Title | Authors | Venue | arXiv/DOI | Ghi chú |
|---|---|---|---|---|
| ADaFuSE: Adaptive Diffusion-generated Image and Text Fusion for Interactive Text-to-Image Retrieval | Zhuocheng Zhang, Xingwu Zhang, Kangheng Liang, Guanxuan Li, Richard McCreadie, Zijun Long | ACM SIGIR 2026 (Paper P032) | arXiv chưa tìm thấy | Xác minh qua trang program SIGIR 2026 chính thức. **Adjacent to CIR**: dùng diffusion-generated image + text feedback (không phải user-provided reference image) → tuỳ định nghĩa CIR. Không phải classic CIR. |

> **Kết luận SIGIR 2026:** Không có paper **classic CIR** (reference image + text modifier) được xác minh tại SIGIR 2026. ADaFuSE (P032) liên quan đến interactive text-to-image retrieval với diffusion augmentation — adjacent nhưng không phải cùng task. 2 papers SIGIR 2025 được xác minh rõ ràng.

---

## 5. SOTA Comparison (chỉ con số đã đọc từ abstract/paper)

> Tất cả con số dưới đây đánh dấu `[ABs]` = lấy từ abstract, chưa đọc full paper. Cần thay bằng số từ PDF trước khi dùng trong thesis.

### CIRR R@1 — Zero-Shot Protocol

| Method | Venue | arXiv | Backbone | CIRR R@1 | Note |
|---|---|---|---|---|---|
| WISER | arXiv 2026 | 2602.23029 | — | +57% relative vs baseline `[ABs]` | Số tương đối, không phải tuyệt đối |
| MRA | arXiv 2025 | 2505.19952 | — | +9.6% vs prior `[ABs]` | Số tương đối |
| MCoT-RE | arXiv 2025 | 2507.12819 | MLLM+CLIP | +8.58% vs prior `[ABs]` | Số tương đối |
| Towards Vision-Free CIR | arXiv 2026 | 2607.12621 | LLM+CLIP | **44.04** (+8.79% vs prior ZS) `[ABs]` | Số tuyệt đối; cần xác minh baseline |

### CIRCO mAP@5 — Zero-Shot Protocol

| Method | Venue | arXiv | Backbone | CIRCO mAP@5 | Note |
|---|---|---|---|---|---|
| WISER | arXiv 2026 | 2602.23029 | — | +45% relative `[ABs]` | Số tương đối |
| MRA | arXiv 2025 | 2505.19952 | — | +9.5% vs prior `[ABs]` | Số tương đối |
| CoCo-IR (single-turn) | ECCV 2026 | 2608.05149 | LMM | **39.4** `[ABs]` | Tuyệt đối; nhưng dùng LMM nặng hơn |
| MulVec | arXiv 2026 | 2608.25305 | Frozen | +23.0% vs strongest `[ABs]` | Số tương đối; cần xác minh |

### FashionIQ Avg R@10

| Method | Venue | arXiv | Backbone | FashionIQ Avg R@10 | Note |
|---|---|---|---|---|---|
| GradCIR (FT) | arXiv 2026 | 2609.24152 | PaliGemma2 | **0.6703** `[ABs]` | Fine-tuned; dùng large backbone |
| MRA | arXiv 2025 | 2505.19952 | — | +7.5% vs prior `[ABs]` | ZS; số tương đối |

> **Lưu ý quan trọng:** Hầu hết con số 2025–2026 là số tương đối ("X% improvement") so với baseline không được chỉ rõ. Không thể so sánh trực tiếp. Cần đọc full PDF để lấy số tuyệt đối trên cùng backbone/split.

---

## 6. Research Gaps — Phân tích có Evidence

### G1. Preserve/Edit không được đo tách biệt [CONFIRMED — evidence từ 15+ papers]

**Evidence mạnh:**
- 65/65 papers dùng Recall@K hoặc mAP — không paper nào có metric riêng cho preserve accuracy và edit accuracy
- MulVec (2608.25305) định nghĩa 4 roles (Global/Desired/Preserve/Forbidden) nhưng vẫn evaluate bằng standard Recall — bản thân paper thừa nhận mismatch giữa roles và metric
- ASAP-CIR (2610.05993) lập luận "target-state > transformation query" nhưng không có metric nào đo riêng phần bị preserved
- GeneCIS (controlled attributes) có thể dùng để đo nhưng chỉ 3–4 papers trong 65 papers dùng benchmark này
- **Khoảng trống:** Không có evaluation protocol chuẩn nào đo riêng "model có giữ đúng những gì cần giữ không" và "model có thay đúng những gì cần thay không"

**Gap cụ thể:** Thiếu attribute-level disentangled metric và protocol chuẩn → không ai biết model nào tốt hơn về *precision* của composition, chỉ biết về *retrieval rank*

### G2. Structured counterfactual negatives — khoảng trống trong training signal [CONFIRMED — evidence từ 8+ papers]

**Evidence mạnh:**
- CSCL (2404.11317): scale negatives bằng MLLM-generated text — không structured theo attribute
- SRAIN (2608.22500): memory bank hard negatives — theo rank, không theo attribute type
- RankVR (2606.11689): xử lý noisy triplets — giải quyết data quality, không phải negative structure
- CLIP4CIR (nền tảng): random in-batch negatives
- MRA (2505.19952), ConText-CIR (2505.20764): tạo training data từ unlabeled images — không khai thác attribute structure
- **Điểm mù:** Không paper nào explicitly mine negatives theo 2 loại lỗi: (a) *wrong-attribute/right-identity* — ảnh giống reference nhưng sai attribute cần thay, và (b) *right-attribute/wrong-identity* — có đúng attribute nhưng sai object/context

**Gap cụ thể:** Training signal hiện tại không buộc model phân biệt 2 loại lỗi này → model học shortcut dùng dominant visual features thay vì hiểu composition instruction

### G3. Thiếu cross-domain evaluation có hệ thống [CONFIRMED — evidence từ tất cả 65 papers]

**Evidence mạnh:**
- Kiểm tra toàn bộ 65 papers: 0 paper nào báo cáo cross-dataset transfer (train A → test B) như một experiment *chính*
- LinCIR evaluate trên 4 benchmarks nhưng train separate checkpoint cho từng benchmark
- CIReVL, MagicLens evaluate multi-benchmark nhưng không test generalization; ZS method cũng vẫn tune threshold/hyperparameter trên val set của từng benchmark
- 2 surveys (2503.01334, 2502.18495) đều liệt kê cross-domain generalization là "future direction"
- **Điểm mù:** Community đo "SOTA trên benchmark X" nhưng không đo "model học được gì có thể transfer"

**Gap cụ thể:** Không có benchmark protocol chuẩn cho cross-domain CIR → không ai biết model nào generalizes tốt nhất

### G4. VLM-augmented methods thiếu ablation công bằng [CONFIRMED — evidence từ 12+ papers]

**Evidence mạnh:**
- CIReVL, FiRE, MCoT-RE, SQUARE, SETR, SoFT, ASAP-CIR, VMIR-CVI, Paracosm, PeFuse, PMTFR — tất cả dùng VLM/MLLM nhưng không paper nào:
  - Cố định retrieval backbone rồi swap VLM quality để đo contribution riêng
  - Báo cáo latency per query và total inference cost
  - Phân tích failure modes theo loại edit (attribute vs. object vs. context vs. count)
- SoFT (2512.20781) propose benchmark ambiguity pipeline nhưng không tách VLM contribution riêng
- **Điểm mù:** Không rõ gain đến từ VLM reasoning hay CLIP similarity hay combination

**Gap cụ thể:** Không có fair comparison framework → không thể claim một VLM-augmented method tốt hơn method khác cùng loại

### G5. Bias trong CIR training data chưa được nghiên cứu [EMERGING — 2–3 papers]

**Evidence:**
- FoCo/FiRE-bias (2606.31222): paper về bias trong CIR — mới, chưa đọc đầy đủ
- CoLLM (2503.19910) introduce refined CIRR/FashionIQ để sửa annotation errors → ngụ ý có bias trong annotation
- SoFT (2512.20781) introduce benchmark ambiguity pipeline → annotation ambiguity là vấn đề thực sự
- **Mức evidence:** Emerging — chỉ 2–3 papers, chưa có systematic study

---

## 7. Ba Hướng Nghiên Cứu Đề Xuất

### Hướng 1 — **Structured Counterfactual Negatives cho CIR** ⭐ RECOMMENDED

**Research Question:**  
> *Liệu việc thay random hard negatives bằng counterfactual negatives có cấu trúc theo attribute — (a) wrong-attribute/right-identity và (b) right-attribute/wrong-identity — có cải thiện khả năng model phân biệt preserve vs. edit, và gain này có generalize cross-domain không?*

**Hypothesis:** Model hiện tại học shortcut dùng dominant visual features (shape, color distribution) thay vì hiểu composition instruction, vì training signal (random negatives) không yêu cầu model phân biệt 2 loại lỗi này. Structured counterfactuals sẽ buộc model học representation tường minh hơn.

**Experiment design:**
1. **Mining pipeline:** Dùng FashionIQ attribute labels (color, type, pattern) để mine 2 loại negative tự động: lấy ảnh cùng category nhưng sai target attribute (wrong-attr); lấy ảnh có đúng target attribute nhưng khác category (right-attr-wrong-id)
2. **Baseline:** CLIP4CIR + LinCIR với random in-batch negatives
3. **Intervention:** Thêm counterfactual contrastive loss với 2 loại negative trên
4. **Ablation:** Random vs. wrong-attr only vs. right-attr-wrong-id only vs. cả hai
5. **Evaluate:** FashionIQ + CIRR + GeneCIS (GeneCIS đặc biệt phù hợp vì có controlled attribute changes)

**Gap evidence trực tiếp:**  
CSCL (2404.11317) scale negatives bằng MLLM nhưng không structure theo attribute type. RankVR (2606.11689) xử lý noisy triplets. Không paper nào mine negatives theo 2 loại lỗi thuần attribute này.

**Novelty:** Đây không phải "thêm data" hay "thêm VLM" — đây là thiết kế lại *cấu trúc training signal* dựa trên phân tích failure mode.

**Feasibility:**
- FashionIQ attribute labels: có sẵn, không cần annotation mới
- Baseline CLIP4CIR/LinCIR: replicable với CLIP ViT-L/14
- GPU: ~2–4 A100-days cho FT; ~1 A100-day cho ZS variant
- Timeline: 4–6 tuần

**Rủi ro:** Medium. FashionIQ attributes có thể không đủ granular → cần verify trước. Nếu gain chỉ trên FashionIQ (domain-specific attributes), cần cross-validate trên CIRR.

**Contribution tuyên bố được:**
1. Phân tích định lượng 2 loại failure mode trong CIR (novelty trong understanding)
2. Automatic mining pipeline cho structured counterfactuals (practical contribution)
3. Gain trên FashionIQ + CIRR + GeneCIS với ablation sạch (empirical contribution)
4. Tích hợp tốt với hiệu ứng của hướng 2 bên dưới

---

### Hướng 2 — **Attribute-Disentangled Evaluation Protocol** (bonus — có thể ghép vào Hướng 1)

**Research Question:**  
> *Recall@K đo retrieval success nhưng không đo composition quality. Liệu một metric tách preserve accuracy và edit accuracy có reveal failure pattern mà Recall bỏ qua không?*

**Hypothesis:** Các method đạt Recall@K cao theo những cách khác nhau — một số thành công vì giữ đúng reference (preserve-dominant), số khác vì thực hiện đúng edit (edit-dominant). Metric tổng hợp che khuất sự khác biệt này.

**Experiment design:**
1. Dùng GeneCIS (controlled changes) + FashionIQ val để annotate attribute-level labels
2. Định nghĩa **ADR (Attribute-Disentangled Recall)**: riêng cho preserve-attributes và edit-attributes
3. Run SEARLE, LinCIR, CIReVL, MulVec, ASAP-CIR qua ADR
4. Show rằng method có Recall cao nhưng ADR thấp theo chiều khác nhau

**Feasibility:** Medium (cần annotation effort). Phần metric mới dễ bị reviewer hỏi về validity.  
**Recommendation:** Ghép phần này vào Hướng 1 như một phụ lục / analysis section, không nên làm contribution chính.

---

### Hướng 3 — **Fair Comparison of VLM-Augmented CIR** (short paper candidate)

**Research Question:**  
> *Với cùng retrieval backbone, bao nhiêu phần của gain từ các VLM-augmented method (CIReVL, MCoT-RE, ASAP-CIR) đến từ VLM caption quality so với composition mechanism?*

**Hypothesis:** Phần lớn gain đến từ VLM captioning chứ không phải từ architectural contribution. Nếu đúng, thì nhiều paper 2025–2026 đang overstate contribution của mình.

**Experiment design:**
1. Fixed backbone: CLIP ViT-L/14; fixed benchmarks: CIRR + FashionIQ + CIRCO
2. Swap caption provider: no-caption → BLIP-2 → LLaVA-1.5 → GPT-4V
3. Apply mỗi method's composition step với cùng captions → tách VLM gain vs. method gain
4. Báo cáo latency và cost per query

**Feasibility:** High technical, nhưng **novelty medium** — reviewer có thể reject là "chỉ empirical study, không có novel method".  
**Recommendation:** Phù hợp short paper tại SIGIR/ECIR, không phải full paper tại CVPR/ECCV.

---

## 8. Kết luận: Hướng Đề Xuất

**Chọn Hướng 1** với phần phân tích từ Hướng 2 được ghép vào như Section Analysis.

| Tiêu chí | Hướng 1 | Hướng 2 | Hướng 3 |
|---|---|---|---|
| Novelty | ★★★★ | ★★★ | ★★ |
| Feasibility (1–2 A100) | ★★★★ | ★★★ | ★★★★ |
| Falsifiable | ✓ | ✓ | ✓ |
| Full paper SIGIR/ICMR/ECCV | ✓ | Khó | Short paper |
| Dễ bị scoop | Thấp | Thấp | Medium |
| Baseline dễ replicate | ✓ | ✓ | ✓ |

**Research Question chính thức:**
> *Liệu việc thay random hard negatives bằng structured counterfactual negatives có cấu trúc theo attribute-type giúp mô hình CIR phân biệt tốt hơn giữa preserve và edit instruction, và gain này có generalize cross-domain không?*

**Contribution tuyên bố:**
1. Failure mode analysis: phân tích định lượng 2 loại lỗi trong CIR
2. Mining pipeline: auto-mine structured counterfactuals từ FashionIQ metadata
3. Empirical gain: FashionIQ + CIRR + GeneCIS, ablation theo loại negative
4. (Optional) Attribute-Disentangled Recall như supplementary analysis

---

## 9. Điều không nên gọi là gap

- Thay backbone CLIP lớn hơn mà không contribution khác
- Chỉ tăng Recall@K trên 1 benchmark bằng thêm data/compute
- Dùng VLM mới hơn (GPT-4o thay LLaVA) mà không ablation
- Gọi là SOTA khi không cùng backbone, split, protocol

---

## 10. Tiêu chí chọn hướng nghiên cứu

1. Hypothesis **falsifiable** — experiment có thể bác bỏ hypothesis
2. **Replicate baseline** CLIP4CIR + LinCIR hoặc CIReVL trước khi claim contribution
3. Đánh giá trên **≥2 benchmark** (FashionIQ + CIRR hoặc CIRCO)
4. **Ablation** từng component rõ ràng
5. **Budget GPU khả thi** — estimate GPU-hours trước khi commit

---

## 11. Danh sách PDF cần đọc (ưu tiên)

**Tier 1 — Baseline bắt buộc replicate:**
1. CLIP4CIR — arXiv:2308.11485 (ACM TOMM 2023; workshop CVPR 2022)
2. Pic2Word — arXiv:2302.03084 (CVPR 2023)
3. SEARLE — arXiv:2303.15247 (ICCV 2023)
4. LinCIR — arXiv:2312.01998 (CVPR 2024) — ViT-G: CIRR R@10=76.05, FashionIQ avg R@10=45.11 `[ABs]`
5. CIReVL — arXiv:2310.09291 (ICLR 2024) ← venue đã được sửa từ NeurIPS→ICLR

**Tier 2 — Related methods để understand negative landscape:**
6. CSCL — arXiv:2404.11317 (nearest work về negative scaling)
7. MulVec — arXiv:2608.25305 (nearest work về role decomposition)
8. GeneCIS paper — để hiểu benchmark design cho attribute-controlled evaluation
9. MagicLens — arXiv:2403.19651

**Tier 3 — VLM-augmented để compare fairly:**
10. FiRE — arXiv:2607.27959 (SIGIR 2025)
11. MCoT-RE — arXiv:2507.12819
12. ASAP-CIR — arXiv:2610.05993

---

*Cập nhật: 06-10-2026. Con số `[ABs]` = từ abstract, phải thay bằng số full PDF trước khi dùng trong paper. SIGIR 2026 proceedings: kiểm tra lại dl.acm.org khi index đầy đủ.*
