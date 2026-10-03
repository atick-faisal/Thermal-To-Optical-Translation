# Narrow the claim before label transfer does

**The research goal fits an active literature, and a real gap remains. But the gap is much narrower than the proposal says, and the experiment most likely to sink the paper is not in the plan yet.** The core mechanism is already published and peer-reviewed. That mechanism is training a thermal→visible translator against a frozen visible detector's loss, so the unchanged detector runs on translated thermal. HalluciDet (WACV 2024) and ModTr (ECCV 2024 main conference) both do this on LLVIP and FLIR, with public code ([ModTr](https://arxiv.org/abs/2404.01492), [HalluciDet](https://arxiv.org/abs/2310.04662)). Using LWIR→RGB translation so that "no labelled LWIR imagery" is needed dates to 2020 ([Abbott et al.](https://mlanthology.org/cvprw/2020/abbott2020cvprw-unsupervised/)). An IR→visible diffusion translator for power-line inspection was published in 2025 ([Li et al., EPEE](https://api.openalex.org/works/https://doi.org/10.1109/epee67527.2025.11428694)). Four sub-claims still survive. First, a *visible-faithful* translator (paired reconstruction plus frozen-detector loss). Second, a *one-step SD-Turbo* thermal→visible translator trained with exact detector gradients. Third, a cross-dataset map of *when* translation pays, including annotation-equivalence. Fourth, an object-level insertion/deletion audit. The decisive threat is **cross-modal label transfer**. Your loop already needs boxes on the paired training images, and those same boxes can train a thermal YOLO11 directly. Published label transfer matches or beats manual labels ([BUDIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/), [Bouzoulas et al.](https://arxiv.org/abs/2507.02513)). Your own budget curve shows a thermal detector reaching 0.93 mAP50 at 600 labels, against the loop's 0.85. So "annotation-free" does not, on its own, set the method apart. The draft Margin criterion, measured against "the strongest baseline that uses no thermal annotations", will probably fail as written. The paper becomes publishable if four things change. It must reframe around *frozen-detector reuse* and *when translation pays*. It must treat FLIR's negative result as data. It must run label transfer, ModTr and zero-shot open-vocabulary detection as baselines. And it must pick a venue framing deliberately: IEEE TIM desk-rejects papers whose main novelty is computer vision ([TIM author guide](https://ieee-ims.org/publication/ieee-tim/information-authors)).

## HalluciDet and ModTr already own detection-in-the-loop translation

Any "first" claim about putting a frozen visible detector inside a thermal→visible translation loop is gone. **ModTr** trains "a small transformation network trained to directly minimize the detection loss", so "the original RGB model can then work on the translated inputs without any further changes or fine-tuning to its parameters" ([arXiv:2404.01492](https://arxiv.org/abs/2404.01492)). Its details are as follows ([arXiv HTML v3](https://arxiv.org/html/2404.01492v3)):

| Aspect | What ModTr does |
| --- | --- |
| Detectors | Frozen COCO-weight Faster R-CNN, RetinaNet and FCOS |
| Training data | IR images plus boxes; no RGB images at all |
| Translator output | Fused back into the IR input by element-wise product |
| FLIR result | **35.5 / 34.3 / 37.2 AP**, against 28.0 / 28.5 / 30.9 for full fine-tuning |
| Authors on their outputs | "not visually pleasant" |

ModTr appears in the main ECCV proceedings ([ECVA PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12401.pdf)), not a workshop. **HalluciDet** came a year earlier. It fine-tunes the detector on target RGB, freezes it, then trains an attention U-Net on detection loss alone ([arXiv HTML](https://arxiv.org/html/2310.04662)). It reports **90.92 AP50 on LLVIP against 84.94 for IR fine-tuning** (Faster R-CNN). The same group has since extended the idea in two directions. ModPrompt (ICCV 2025) applies it to frozen open-vocabulary detectors (YOLO-World, Grounding DINO) and to depth ([arXiv:2412.00622](https://arxiv.org/abs/2412.00622)).

Three neighbouring precedents widen what reviewers will treat as "already done". First, the mechanism is older than thermal work. CyCADA used a frozen source task model as a semantic-consistency loss on translated images in 2018 ([arXiv:1711.03213](https://arxiv.org/abs/1711.03213)). Task-driven super-resolution traded a detection loss against a reconstruction loss the same year ([arXiv:1803.11316](https://arxiv.org/abs/1803.11316)). That second paper is structurally the closest analogue to your pix2pix-plus-detector objective. Second, the diffusion side has moved. **InfraredIR (CVPR 2026)** uses single-step SD-Turbo, task-aware LoRA and a detection loss with YOLOv8 on M3FD ([GitHub](https://github.com/csmty/InfraredIR)). It comes from the TarDAL/M3FD group, and it is IR→IR restoration rather than cross-modal translation. NOLA-IR backpropagates task losses into a one-step SD-2.1 LoRA ([arXiv:2607.25390](https://arxiv.org/html/2607.25390)). DRaFT already used an OWL-ViT detector as a differentiable reward ([arXiv:2309.17400](https://arxiv.org/html/2309.17400)). Third, the power domain is not virgin territory. Li et al. translate IR→visible with a diffusion model to improve power-line instance segmentation ([EPEE 2025](https://api.openalex.org/works/https://doi.org/10.1109/epee67527.2025.11428694)). Only the abstract was read, and it shows no task loss in training. Gomes et al. run a visible detector and transfer its boxes to stereo-rectified IR, explicitly to avoid expert IR annotation ([BABT 2025](https://api.crossref.org/works/10.1590/1678-4324-2025240843)).

| Claim in the current proposal | Status | Pre-empting work |
| --- | --- | --- |
| Detection loss trains an IR→RGB translator | **Taken** | [HalluciDet](https://arxiv.org/abs/2310.04662), [ModTr](https://arxiv.org/abs/2404.01492) |
| Frozen detector, no forgetting, reused across modalities | **Taken** | [ModTr](https://arxiv.org/abs/2404.01492) |
| Annotation-free thermal detection via translation | **Taken** | [Abbott et al. 2020](https://mlanthology.org/cvprw/2020/abbott2020cvprw-unsupervised/) |
| Annotation-free thermal detection, in general | **Taken (other route)** | [Thermal-Det, CVPR 2026](https://arxiv.org/html/2605.10130v1), [BUDIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/) |
| Task loss into a one-step SD-Turbo generator | **Taken for IR restoration** | [InfraredIR](https://github.com/csmty/InfraredIR), [NOLA-IR](https://arxiv.org/html/2607.25390) |
| Thermal→visible translation for power inspection | **Taken (segmentation, no loop in abstract)** | [Li et al. 2025](https://api.openalex.org/works/https://doi.org/10.1109/epee67527.2025.11428694) |
| Task-driven IR–visible processing | **Taken (two-input fusion)** | [TarDAL](https://arxiv.org/abs/2203.16220), SeAFusion |
| A detector-based score of translation quality | **Taken as a concept** | [Kim et al. ECCV 2024](https://arxiv.org/abs/2404.05980), [Konz et al.](https://arxiv.org/abs/2404.07318) |
| A paired thermal–visible detection benchmark protocol | **Taken for public data** | [WiSE-OD](https://arxiv.org/abs/2507.18925), LLVIP/FLIR/M3FD suites |

One more title deserves attention before submission. "TIRDet: Mono-Modality Thermal InfraRed Object Detection Based on Prior Thermal-To-Visible Translation" (ACM MM 2023) was seen by title only, because the page returned 403 ([ACM DL](https://dl.acm.org/doi/10.1145/3581783.3613849)). Its content is unknown.

## Four sub-claims survive, and the "when it pays" map is the strongest

**The first surviving claim is visible-faithful detection supervision.** HalluciDet uses detection loss only, and ModTr deliberately produces non-visual, detector-friendly representations ([ModTr HTML](https://arxiv.org/html/2404.01492v3)). Neither combines a *paired visible-reconstruction* objective with the frozen-detector loss. Your translator outputs an image that is meant to look visible. That matters for three reasons: operator inspection, overlaying thermal faults, and reusing *other* visible detectors. It also opens a clean empirical question that nobody has answered: **how much detection accuracy does faithfulness cost?** Your FLIR result makes the question urgent. ModTr beats full fine-tuning on FLIR by about 7 AP. Your loop only climbs back to the raw-thermal floor. A summariser reading ModTr's tables reported zero-shot RGB detectors on raw FLIR IR at roughly 23–25 AP. That is unverified, but if it holds, ModTr clears the raw-thermal floor where you do not. There are three plausible reasons, all from [ModTr's design](https://arxiv.org/html/2404.01492v3):

- ModTr fuses the IR input back in, so the detector always sees thermal structure.
- ModTr has no visible-reconstruction constraint.
- ModTr reports COCO AP with two-stage and anchor-free detectors, not mAP50 with YOLO.

A head-to-head run of ModTr turns this from an embarrassment into a measured faithfulness–accuracy frontier.

**The second is a one-step diffusion thermal→visible translator with exact frozen-detector gradients.** About 30 searches found no paper or repo applying pix2pix-turbo or CycleGAN-Turbo to thermal→visible, and none adding a task loss to img2img-turbo. The only thermal uses found run in the reverse direction ([img2img-turbo](https://github.com/GaParmar/img2img-turbo), [arXiv:2609.32944](https://arxiv.org/html/2609.32944)). Diffusion work in your direction is mostly about faces (T2V-DDPM, DiffTV). The closest scene-level paper, TC-PDM, *evaluates* detection (+6.1% AP50) without training on it ([arXiv:2408.14227](https://arxiv.org/pdf/2408.14227)). EDTR is a useful foil. It found that backpropagating a task loss into a multi-step diffusion restorer "causes instability", so it updates only the task network ([arXiv:2507.22459](https://arxiv.org/html/2507.22459)). Stable exact gradients through a one-step translator with a *frozen*, foreign-modality detector is a small but citable methodological finding. Treat this claim as **conditional on InfraredIR's full text**, which was not readable (HTTP 403). If InfraredIR freezes its YOLOv8 and frames itself as cross-domain, the margin shrinks to "cross-modal versus same-modal".

**The third, and the strongest, is the "when does translation pay" map with annotation-equivalence.** No paper found regresses translation gain against each dataset's visible–thermal gap. None states "translation is worth N thermal labels" either. This is absence of evidence from a handful of targeted searches, not proof. The closest analogues are a cost-equivalence framing in synthetic-data work ([Hinterstoisser et al.](https://arxiv.org/pdf/1902.09967)) and HalluciDet's snippet-level claim that 30% of training data matches fine-tuning ([arXiv HTML](https://arxiv.org/html/2310.04662)). That claim is about translator training data, not thermal labels. Your numbers already tell this story. On power lines, the visible detector sees almost nothing in raw thermal (0.19 against 0.92 on visible), and translation recovers most of the gap (0.80 without the loop, 0.85 with it). On FLIR cars and people, raw thermal is already readable, and translation adds nothing beyond the floor. The same map answers the reviewer's obvious question, "why not just label 150–214 thermal images?", with a number instead of an assertion. With only five datasets, present it as a descriptive scatter with confidence intervals, not a fitted law.

**The fourth is an object-level insertion/deletion audit.** Judging translation with a downstream task is old. pix2pix's FCN-score is the ancestor, though it was recalled from memory and not re-verified. A diffusion-hallucination paper measured hallucination through downstream tasks because "most hallucinated features are small and localized" ([Kim et al., ECCV 2024](https://arxiv.org/abs/2404.05980)). Perceptual metrics "do not generally correlate with segmentation metrics" ([Konz et al.](https://arxiv.org/abs/2404.07318)). What no thermal→visible paper reports is a *decomposed* audit: separate invented-object and erased-object rates against both ground truth and real-visible detections, with uncertainty. A detection-loop paper *needs* this audit. DRaFT documents reward over-optimisation, where the generator collapses toward "a certain high-reward image" ([arXiv:2309.17400](https://arxiv.org/html/2309.17400)). An AAAI 2026 oral shows that hallucinated target-class objects degrade downstream mAP ([arXiv:2602.15383](https://arxiv.org/pdf/2602.15383)). To avoid the circularity critique, score the audit with a detector family that never sat in the loop.

The **power-line domain** is a fifth, supporting claim. It is not a headline. No public dataset provides registered thermal–visible pairs of line components with boxes. But private paired power sets are common: VISED with 500 pairs ([Processes 2025](https://doi.org/10.3390/pr13092720)), MSIS with 5,938 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8752308/)), and an insulator set with 1,000 ([Sensors 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12074302/)). VITLD offers 400 registered transmission-line pairs with conductor masks, available on request ([arXiv:2501.15099](https://arxiv.org/html/2501.15099)). "Near-unique data" overstates the case. "No public component-level box equivalent" is accurate.

## Label transfer on the same pairs is the experiment that decides the paper

**Label transfer is the most dangerous competitor because your method already contains it.** The loop backpropagates a detection loss on the paired training images, so those images already carry boxes. They come either from visible ground truth or from the visible detector's predictions on the visible half. Training YOLO11 on the thermal half with those same boxes takes zero manual thermal labels. The published evidence says this works:

- **BUDIR** auto-labelled about 300K thermal images through LiDAR reprojection with no manual labels. It reached **0.544 mAP50, against 0.523 for a detector trained on manually labelled FLIR**. Adding 150 human labels lifted it to 0.661 ([PMC7926581](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/)).
- **Bouzoulas et al.** found models trained on transferred KAIST labels beat ground-truth-trained models in **5 of 6 cases** on mAP50, across DETR, YOLO and R-CNN ([arXiv:2507.02513](https://arxiv.org/abs/2507.02513)).
- **Thermal-Det (CVPR 2026)** makes "frozen RGB teacher on paired RGB-T → annotation-free thermal detector" a top-venue method ([arXiv:2605.10130](https://arxiv.org/html/2605.10130v1)).

Your own budget curve shows a thermal YOLO11 matching the loop at about 150–214 labels and reaching 0.93 at 600. It follows that transferred labels on your 600 training pairs would plausibly land near 0.9, above the loop's 0.85. **This is an inference, not a measurement, and it is the first thing to run.** There is a sharper possibility too. If the custom-set boxes were drawn on the visible images and mirrored to thermal (the goal states FLIR's thermal labels were produced exactly that way), then the 0.93-at-600 point is *already* ground-truth label transfer.

Several arguments do not set translation apart from label transfer. **Thermal-only deployment is not a differentiator.** A thermal detector trained on transferred labels also runs on thermal-only survey footage. Deployment shift between the pairs and thermovision-car footage hurts both routes, and which one degrades less is an empirical question. Thermal-only deployment *does* separate both routes from fusion, and from Gomes-style stereo transfer, which needs a visible camera at test time ([Gomes et al.](https://api.crossref.org/works/10.1590/1678-4324-2025240843)).

Four properties *do* set translation apart, and each must be demonstrated rather than asserted:

1. **Frozen detector.** Label transfer always produces a *new* detector. It cannot serve a certified, vendor-locked or shared visible detector. ModTr's "service-based pipeline" argument makes this case ([project page](https://heitorrapela.github.io/ModTr/)).
2. **Detector-agnostic reuse.** A translator trained with YOLO11 in the loop helps a *different* visible detector. This is your E7, and it is also where visible-faithful outputs should beat ModTr's detector-specific representations.
3. **Classes absent from the paired set.** A translator is in principle class-agnostic. Label transfer only yields labels for classes that appear in the pairs.
4. **Pixel outputs for inspectors.** The fault-overlay use in your goal. This is weak as a detection argument but meaningful for a measurement or inspection audience.

The cleanest way to present this to Information Fusion turns the threat into a component: fuse, at decision level, the detections of the frozen visible detector on translated images with those of a thermal detector trained on transferred labels. Whether the two routes are complementary is untested. It is cheap to try.

The mandatory baseline set follows from this analysis. Arms that retrain the detector must be labelled as such in every table.

| Priority | Baseline | Why a reviewer demands it | Status |
| --- | --- | --- | --- |
| Must | **Label transfer**: YOLO11 trained on thermal with boxes from the visible half (detector-predicted, plus a GT-transfer variant), plus a label-transfer-plus-k-manual hybrid curve | Same pairs, zero manual labels, likely beats 0.85 ([BUDIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/)) | Not run |
| Must | **ModTr**, retrained on the same boxes, on power lines, FLIR and LLVIP | Literally your setting with a different translator; public code ([GitHub](https://github.com/heitorrapela/ModTr)) | Not run |
| Must | **HalluciDet** | Direct prior art; public code ([GitHub](https://github.com/heitorrapela/HalluciDet)) | Not run |
| Must | **Zero-shot Grounding DINO / YOLO-World**, on raw thermal *and* translated thermal | G-DINO reaches 0.636 AP50 zero-shot on FLIR-Aligned, against 0.749 fine-tuned ([Thermal-Det](https://arxiv.org/html/2605.10130v1)); power lines unmeasured | Not run |
| Must | Thermal-supervised YOLO11 budget curve | Annotation-equivalence | Have it |
| Must | Raw thermal, translation without the loop, CycleGAN/pix2pix lineage | Floor and causal control | Have most |
| Should | **D3T** unpaired UDA | Strongest RGB→thermal UDA: 69.3 mAP on FLIR against 34.7 source-only ([arXiv:2403.09359](https://arxiv.org/html/2403.09359)) | Not run |
| Should | **RGB→thermal synthesis** plus a thermal detector, if public labelled visible power data exist | Booming reverse paradigm; SD3.5+ControlNet lifted mAP 25.6→38.4 on VTUAV ([arXiv:2609.02556](https://arxiv.org/abs/2609.02556)) | Not run |
| Nice | Cross-modal knowledge distillation, Thermal-Det-style | Annotation-free top-venue route | Not run |
| Argue out | Two-input fusion (SeAFusion, TarDAL) | Needs visible input at test time; cite rather than run, unless targeting Information Fusion | — |

The proposal's current §8 list misfiles three entries. arXiv:2002.06770 generates *fake thermal from visible* ([arXiv](https://arxiv.org/abs/2002.06770)). Kieu et al. (ECCV 2020) contains no translation ([ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)). Meta-UDA contains no translation either ([arXiv:2110.03143](https://arxiv.org/abs/2110.03143)). They remain valid domain-adaptation references, but they are weaker threats than label transfer and ModTr.

## RESEARCH_FINDINGS.md needs ten corrections

The corrections below are factual or positioning fixes. Each one, left in, hands a reviewer an easy rejection.

| Section | Current text | Correction |
| --- | --- | --- |
| §8 | "ModTr (ECCV 2024 workshops)" | ECCV 2024 **main conference** ([ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12401.pdf)); promote it from baseline to closest prior art |
| §8 | Baseline list | Add HalluciDet, label transfer, zero-shot open-vocabulary detection, D3T and RGB→thermal synthesis. Refile arXiv:2002.06770 as visible→thermal synthesis, and Kieu (ECCV 2020) and Meta-UDA as no-translation domain adaptation |
| §1 | C1: "Closed-loop detection-consistency feedback for diffusion IR→VIS translation" as the headline | The mechanism is prior art. Headline the visible-faithful, one-step diffusion variant, with causal evidence, as a *successor* to HalluciDet/ModTr |
| §1 | C2: "novel" faithfulness metric | Lineage exists ([Kim et al.](https://arxiv.org/abs/2404.05980), [Konz et al.](https://arxiv.org/abs/2404.07318)); call it an object-level insertion/deletion audit |
| §1, §5 | C3 "near-unique data — no public equivalent exists" | "No public registered component-level box dataset." Private paired power sets exist (VISED, MSIS, Yang et al.); VITLD is paired but conductor-only |
| §5 | Yetgin & Gerek: "400 IR + 400 VL, wire masks… the only public paired… resource" | 4,000 + 4,000 images at 128×128 with image-level labels, plus 200 + 200 at 512×512 with masks; pairing not stated, not registered ([Mendeley](https://data.mendeley.com/datasets/n6wrv4ry6v/8), [GT set](https://data.mendeley.com/datasets/twxp8xccsw/6)). Cite VITLD as the closest paired line resource |
| §5 | HIT-UAV as "overhead/aerial-inspection domain proximity" | Thermal-only, persons and vehicles, **not power-related** ([Sci. Data](https://www.nature.com/articles/s41597-023-02066-6)) |
| §10 | Margin: "comparable published work reports gains of this order" | Unsupported, and the baseline set changed. Rewrite as below |
| §10 | Consistency: "gain holds on ≥2–3 datasets" | FLIR already contradicts it for translation versus raw thermal. Split the loop effect from the translation effect (below) |
| §13 | TIM "Primary — measurement/inspection applications" | TIM desk-rejects AI/CV-novelty papers, dataset-only papers and papers without GUM uncertainty ([author guide](https://ieee-ims.org/publication/ieee-tim/information-authors)) |

The goal itself inherits one more framing problem. Its Problem section calls a large annotated thermal set "impractical". Your own curve shows about 150–214 thermal labels match the loop. Keep the annotation-difficulty argument, which the power literature supports: annotating IR "requires specialists… harder and more expensive" ([Gomes et al.](https://api.crossref.org/works/10.1590/1678-4324-2025240843)). But shift the stakes from "impossible" to "zero labels, zero detector retraining, existing detectors reused".

## Information Fusion fits only with a fusion arm; TIM only as a measurement paper

**IEEE TIM is the riskiest of the two named venues, not the safest.** Its author guide says it typically rejects without review papers "whose main novel contribution falls within another discipline (e.g. artificial intelligence, computer vision…)". It also rejects papers presenting new image-processing or machine-learning algorithms "without clearly demonstrating how the algorithm improves measurement performance". It requires compliance with the GUM, a clear distinction between "algorithmic performance metrics" and measurement-system characteristics, and the authors' own experimental validation rather than "analysis of existing datasets" ([TIM author guide](https://ieee-ims.org/publication/ieee-tim/information-authors)). The current framing, "an instrument for generating a results table", is a computer-vision framing, and the first rule applies almost word for word. TIM remains viable only as a *measurement* paper, which needs four things:

1. The 753-pair FLIR acquisition presented as your own experimental validation: camera, NETD, band, distance, emissivity, ambient conditions, and registration residual.
2. Every mAP reported as a measurement with Type-A uncertainty across seeds and images.
3. C2 recast as characterising the instrument's systematic errors: insertion and deletion rates.
4. Comparison against the I&M literature.

A measurement journal has already shown the topic is in scope. *Measurement Science and Technology* published a 2026 topical review of thermal-infrared-to-visible translation ([IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae7822)). That makes it a realistic fallback with a lower bar on CV novelty.

**Information Fusion is the better primary of the two, provided you make the fusion link explicit.** Its scope is "multi-sensor, multi-source, multi-process information fusion" ([guide for authors](https://www.sciencedirect.com/journal/information-fusion/publish/guide-for-authors)). It is the home of task-driven IR–visible fusion. No verified Information Fusion paper on single-input thermal→visible translation was found: a search summary attributed one, but its ISSN belongs to *Optics and Lasers in Engineering* ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0143816623002749)). Frame the loop as **cross-modal knowledge fusion**: a visible-domain detector prior fused with a thermal sensor stream at training time, in the task-driven lineage of SeAFusion and TarDAL. Then add one real fusion experiment: decision-level fusion of the translation route with the label-transfer route (and/or raw-thermal detections). That experiment answers the scope question and the label-transfer threat at once. Expect Information Fusion reviewers to know Thermal-Det and TarDAL. Expect them to want a fusion upper bound on the paired set too, such as a VISED-style feature-fusion YOLO ([Processes 2025](https://doi.org/10.3390/pr13092720)).

If neither framing feels natural, two other options exist. Method-first venues (TCSVT, Pattern Recognition) would carry C1 plus the "when it pays" map; their scope was not verified in this review. CVPR PBVS or WACV, where ModTr's group publishes ([WiSE-OD](https://arxiv.org/abs/2507.18925)), would give fast exposure. On rigour, your plan is ahead of the field. At least 3 seeds, sign-flip tests and a same-model causal ablation exceed what the closest CV papers report. Five datasets is at or above the 2–4 the competitors use ([ModTr](https://arxiv.org/html/2404.01492v3), [YOLOv11-RGBT suite](https://arxiv.org/pdf/2506.14696)). Lead with that rigour in either venue.

## Rewrite the goal around when translation pays

The current central question asks whether the loop "improve[s] downstream power-line component detection". That question invites "compared with what?", and the honest answer today is "not compared with label transfer". A question your data can win is this:

> **When, and by how much, does training a thermal→visible translator against a frozen visible detector let that unchanged detector work on thermal-only imagery — compared with the other zero-manual-label routes — and does it do so without inventing or erasing components?**

Your existing measurements answer this question without contradiction. The loop is causal on both datasets that have a no-loop control: +0.05 on power lines and +0.35 on FLIR. Translation beats raw thermal only where the thermal–visible gap is large.

### Success Criteria

| Criterion | Current or draft | Proposed |
| --- | --- | --- |
| **Margin** | Draft: "on the power-line held-out test split, the loop beats the strongest baseline that uses no thermal annotations by ≥ +2 mAP@50" | "On the held-out power-line test split, the loop beats the strongest **frozen-detector** baseline by ≥ +2 mAP@50, with a 95% bootstrap CI over test images that excludes zero, and a non-negative mAP@50:95 delta. The frozen-detector baselines are raw thermal, translation without the loop, ModTr and HalluciDet retrained on the same boxes, and zero-shot Grounding DINO. Label transfer and the thermal-supervised budget curve are reported beside it as references that **retrain** the detector." With only about 100 test pairs, a +2 point gap may not clear its CI, so compute the bootstrap on validation first. If ModTr matches the loop, restate Margin as non-inferiority to ModTr plus a win on Transfer and Faithfulness. |
| **Transfer** *(new)* | — | "A translator trained with YOLO11 in the loop improves at least one visible detector never used in the loop (a different family, e.g. a DETR-type or Grounding DINO) over the no-loop translator, across ≥3 seeds." This is the property label transfer structurally lacks. Without it, the frozen-detector framing is an assertion. |
| **Consistency** | "Gain holds on public benchmarks: custom + LLVIP and/or M3FD + one more, ≥3 datasets" | "On ≥3 of the 5 datasets, the loop beats its own no-loop control (sign-flip test, ≥3 seeds). Whether translation beats raw thermal is **not** a criterion. It is C4's dependent variable, reported for all five datasets, failures included." FLIR then counts *for* the paper. |
| **Causality / Stability** | λ_det ablation; ≥3 seeds, significance-tested | Keep both. This rigour exceeds the norm in the closest prior work. |
| **Faithfulness** | "False-object and missed-object rates are low… can affirmatively claim no invention or erasure" | "Insertion and deletion rates, scored against ground truth and by a detector family never used in the loop, with CIs, are **non-inferior** to the no-loop translator within a pre-registered margin. Reported for every arm, including ModTr." An absolute "no invention" claim cannot be proven and invites attack. |
| **Fallback** | "If the loop helps only at low annotation, pivot to data-efficiency" | Delete it as a fallback. Annotation-equivalence is now part of the headline, because the 150–214-label break-even and the 0.93-at-600 crossover are already measured. |

### Non-Goals to add

Write these down so that reviewers, and future you, do not drift into them:

- **Not** claiming to be first to use detection loss for thermal→visible translation, or first to reuse a frozen visible detector on thermal. HalluciDet and ModTr are prior art.
- **Not** beating supervised thermal detection at full annotation. Your own data concedes 0.93 against 0.85 at 600 labels.
- **Not** state of the art on LLVIP or FLIR against ModTr's AP. The public datasets map *when* translation helps; they are not a leaderboard.
- **Not** a dataset-release paper. The power-line set is a private domain testbed. Release splits, configs and code even if the images stay private.
- **No** two-input fusion at inference. The only exception is a decision-level fusion ablation if Information Fusion is the target.

### Contribution list

| # | Proposed contribution | Change from now |
| --- | --- | --- |
| C1 | **Visible-faithful, detection-supervised thermal→visible translation** (paired reconstruction plus frozen YOLO11 loss) for pix2pix and a one-step SD-Turbo translator with exact gradients. Causal evidence: the same model with and without the detector loss, ≥3 seeds, sign-flip tests. Cross-detector transfer included. | Positioned as a successor to HalluciDet/ModTr, not as a new mechanism. Adds Transfer. |
| C2 | **When translation pays.** A cross-dataset map of translation gain against each dataset's thermal–visible gap, over 5 datasets, plus annotation-equivalence against thermal-supervised and label-transfer curves. | Promoted from "defends the premise" to co-headline. Absorbs the old C3 protocol. |
| C3 | **Object-level insertion/deletion audit** for detector-trained translators, scored by a held-out detector, with uncertainty. | Renamed from "novel faithfulness metric". Cites the FCN-score and medical-hallucination lineage. |
| C4 | **A power-line component testbed and reproducible protocol.** Frozen splits, plus a per-dataset bracket of visible ceiling, raw-thermal floor, thermal-supervised and label-transfer references. | Downgraded from a standalone benchmark contribution. |

## Checks to do by hand before committing to the goal

These are ordered by how much each could change the decision. The first three are experiments or local facts. The rest are papers whose full text this review could not read, or claims that rest on snippets.

| Priority | Check | Why it matters |
| --- | --- | --- |
| 1 | **Find out how the custom-set boxes were drawn**: on visible, on thermal, or on both | If on visible, the 0.93-at-600 thermal detector *is* ground-truth label transfer, and it already beats the loop by about 8 points |
| 2 | **Run label transfer** on the 600 training pairs: visible-detector boxes moved to thermal, YOLO11 trained, evaluated on val, then on held-out test | The single result most likely to reshape Margin and the framing |
| 3 | **Run ModTr (public code) on power lines and FLIR**, plus zero-shot Grounding DINO on raw and translated power-line thermal | Measures the faithfulness–accuracy trade-off and explains the FLIR discrepancy |
| 4 | Read **InfraredIR (CVPR 2026)** in full: is YOLOv8 frozen, and what is the loss form? | Decides whether the one-step-diffusion novelty survives |
| 5 | Read **Li et al. (EPEE 2025)** in full: is a segmentation loss in the loop, which dataset, which baselines? | Closest same-domain, same-direction prior work |
| 6 | Read **TIRDet (ACM MM 2023)** | Title suggests thermal→visible translation as a prior for thermal detection; content unknown |
| 7 | Read ModTr's and HalluciDet's full PDFs: zero-shot raw-IR rows, FLIR AP50, CycleGAN/FastCUT numbers, low-label experiments, HalluciDet's "30% of data" claim | Several cited numbers are summariser-level |
| 8 | Run a Google Scholar "cited by" crawl of HalluciDet, ModTr and img2img-turbo; an IEEE Xplore query on TIM for thermal/infrared translation; a check for a ModTr journal extension; and a pass over Chinese-language power journals | No systematic forward-citation search was possible; a 2026 competitor could be missing |
| 9 | Check visible-label quality for thin structures (conductors, insulators) under your registration error | Determines how strong the label-transfer baseline will be |
| 10 | Venue due diligence: walk TIM's desk-reject list line by line; read 2–3 recent Information Fusion papers to see whether single-input translation got in; verify TCSVT/PR scope | The venue choice changes how the manuscript is written |
| 11 | Verify memory-only citations before use: FCN-score (pix2pix/CycleGAN), Cohen et al. MICCAI 2018, Bhadra et al. TMI 2021, IA-YOLO, DUNIT, ForkGAN, ReFL, AlignProp, DDPO | Cited from prior knowledge, not fetched |
| 12 | Resolve snippet-level items: Abbott et al.'s datasets and loss; Thermal-Det's ablation (0.261) against its main-table (0.372) discrepancy; D3T's 69.30 against an ablation snippet's 68.46; the availability of Cantón & Lazaro's numbers, VITLD, VISED and the Gomes stereo frames | Numbers a reviewer may check |

## Conclusion

The literature does not make the project redundant. It moves the paper's centre of gravity. The novelty is no longer *that* a frozen detector can train a translator. ModTr won that argument in 2024. It is *what kind* of translator survives the loop, and *when* the whole route is worth taking instead of simply moving labels across the pair. That reframing turns your two apparent weaknesses into evidence. The FLIR null becomes the low-gap end of a map whose high-gap end is power lines. The 150–214-label break-even becomes a deployment number an engineer can act on. The loop's causal effect is positive wherever it was measured, which supports the mechanism even where translation itself does not pay.

The single most informative next step is cheap: find out how the custom boxes were drawn, then run label transfer. If transferred labels land near 0.9, the paper's honest claim is "frozen-detector reuse with visible-faithful outputs that transfer across detectors", and it needs the Transfer criterion to hold. If they land below 0.85, the original annotation-free framing comes back strengthened, with a baseline most competitors never ran.
