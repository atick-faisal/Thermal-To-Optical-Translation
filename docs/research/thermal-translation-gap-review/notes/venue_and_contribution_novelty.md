# Venue fit and novelty of secondary contributions (C2 faithfulness, C3 benchmark protocol, C4 "when translation helps")

Scope: thermal→visible translation trained with a frozen visible detector's loss (pix2pix, pix2pix-turbo/SD-Turbo), power-line components on a private 753-pair FLIR set plus LLVIP, FLIR-aligned, M3FD, MSRS. Target venues in `RESEARCH_FINDINGS.md` §13: IEEE TIM and Information Fusion (primary), TGRS, TII, TPWRD, PR/TIP, ESWA/Neurocomputing.

Research date: 2026-10-03. Labels used below:
- **[verified]**: read from the primary page this session.
- **[snippet]**: seen only in a search-result summary.
- **[memory, unverified]**: from prior knowledge and not re-fetched this session. The report writer should treat these with care.

---

## Q1. Which venues fit, and what have IEEE TIM and Information Fusion published on IR/visible translation, fusion, thermal detection and power-line inspection? Is translation in scope for Information Fusion? Which alternatives fit?

### Takeaway
IEEE TIM is the **riskiest** primary venue, not the safest. Its author guide says it typically desk-rejects papers whose main novelty is AI or computer vision. It also desk-rejects papers built only on existing datasets, and papers that do not express uncertainty to GUM standards. The paper can only go there if it is reframed as a *measurement* paper: the translator is an instrument, and its error is quantified with uncertainty. Information Fusion's stated scope is multi-sensor/multi-source fusion. A translator that takes one input at inference only fits if it is framed as fusing cross-modal knowledge (a visible-trained detector prior plus a thermal sensor). Measurement-oriented journals (Measurement Science and Technology, Elsevier Measurement) and CV venues (TCSVT, WACV, CVPR PBVS) are realistic alternatives.

### Cited Findings
**IEEE TIM scope and desk-rejection rules [verified]**
- TIM lists the kinds of papers that are "typically rejected without peer review". The list includes papers "whose main novel contribution falls within another discipline (e.g. artificial intelligence, computer vision, communication systems, navigation, robotics, healthcare...)" — [IEEE IMS, TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- It also desk-rejects papers that "present novel algorithms for signal/image processing, image enhancement, image segmentation or machine learning (including classification), without clearly demonstrating how the algorithm improves measurement performance" — [TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- It desk-rejects papers that "do not comply ... with established international metrology guidelines, such as the Guide to the Expression of Uncertainty in Measurement, GUM". It also desk-rejects papers that "fail to distinguish algorithmic performance metrics from the performance characteristics of a measurement system" — [TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- It desk-rejects papers "based only on simulation or analysis of existing datasets, without adequate discussion of Instrumentation & Measurement aspects and without a strong justification for the lack of authors' own experimental validation". It also desk-rejects papers lacking "detailed information about the experimental setup and the considered measurement chain" — [TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- Papers must compare against "the existing literature in Instrumentation & Measurement". Regular papers have a 5-page minimum, with over-length charges. IEEE DataPort is mentioned for datasets, but code and data release are not mandated — [TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- TIM's scope covers "theory, methodology, and practice of measurement; design, development and evaluation of instrumentation and measurement systems ... analysis, representation, display, and preservation of the information obtained from a set of measurements" — [IEEE IMS TIM page](https://ieee-ims.org/publication/ieee-tim) [snippet]

**What TIM has published in this area**
- TIM 2024 has IR–visible *fusion* papers, e.g. "SigFusion: Semantic information guided infrared and visible image fusion" (TIM 2024) [snippet], and an IR/visible fusion paper in TIM vol. 73, 2024, art. 9507320 — [Buffalo-hosted TIM 2024 PDF](https://cse.buffalo.edu/~wenyaoxu/papers/journal/xu-tim2024.pdf) [snippet]
- I found **no** TIM paper (2023–2026) on thermal→visible *translation for detection*. The searches surfaced fusion papers only. This is a search gap, not proof of absence.
- Power-equipment infrared detection appears in sister IEEE journals. A search snippet attributes a 2025 IR insulator/power-equipment detection paper to IEEE TII vol. 21, pp. 2829–2838 [snippet, unverified], cited via a review preprint — [Preprints.org review](https://www.preprints.org/frontend/manuscript/12d13bbd515557a0ff5ff323583c2c98/download_pub). Other venues: [IEEJ Trans. 2025, thermal defect detection in transmission lines](https://onlinelibrary.wiley.com/doi/10.1002/tee.70060). IEEE published a new standard in 2025, IEEE 3326-2025, a recommended practice for infrared online substation inspection — [en-standard.eu listing](https://www.en-standard.eu/ieee-3326-2025-ieee-recommended-practice-for-the-use-of-infrared-online-systems-for-substation-inspection/) [snippet]

**Information Fusion scope [snippet of official guide]**
- Its aim is to present "all of the developments in the field of multi-sensor, multi-source, multi-process information fusion". Articles should emphasize "architectures, algorithms, and applications". Topics include "Data/Image, Feature, Decision, and Multilevel Fusion" — [Information Fusion Guide for Authors](https://www.sciencedirect.com/journal/information-fusion/publish/guide-for-authors)
- Submission requirements: a 250-word unstructured abstract, Highlights, a graphical abstract, a CRediT statement and a data availability statement — [Information Fusion Guide for Authors](https://www.sciencedirect.com/journal/information-fusion/publish/guide-for-authors)
- Information Fusion is the home of task-driven IR–visible fusion: SeAFusion (2022, "semantic-aware real-time IR-VIS fusion"), PIAFusion (2022), and progressive semantic injection (2023) — listed in a [search summary](https://www.sciencedirect.com/journal/information-fusion) [snippet; SeAFusion venue matches memory]. A 2025 "target-aware unregistered IR–VIS fusion" paper backpropagates a detection loss into the fusion network. Its venue was **not** confirmed [snippet].
- One search summary claimed that "Contrastive learning with feature fusion for unpaired thermal infrared image colorization" appeared in Information Fusion. **This is likely wrong.** The URL's ISSN prefix (S0143-8166) belongs to *Optics and Lasers in Engineering* — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0143816623002749). I found no verified Information Fusion paper on single-input thermal→visible translation.

**Alternative venues, with evidence of fit**
- *Measurement Science and Technology* (IOP) published a 2026 topical review, "A review of thermal infrared-to-visible image translation algorithms" (Liu et al., vol. 37). It covers pseudo-colour, reference-based, fusion, CNN, GAN and transformer I2V methods — [IOPscience](https://iopscience.iop.org/article/10.1088/1361-6501/ae7822) [verified]. This is a measurement journal that explicitly treats TIR→VIS translation as in scope.
- *Measurement* (Elsevier) published TIR road-object detection in 2025 — [ScienceDirect S0263224125003306](https://www.sciencedirect.com/science/article/abs/pii/S0263224125003306) [snippet]
- *WACV 2026* accepted WiSE-OD, which benchmarks robustness of RGB→IR detector adaptation. It comes from the same group as ModTr — [arXiv 2507.18925](https://arxiv.org/abs/2507.18925) [verified]
- *ECCV 2024* accepted ModTr, the closest prior work to C1 — [arXiv 2404.01492](https://arxiv.org/abs/2404.01492) [verified]
- *ACM MM 2023* accepted "TIRDet: Mono-Modality Thermal InfraRed Object Detection Based on Prior Thermal-To-Visible Translation" — [ACM DL](https://dl.acm.org/doi/10.1145/3581783.3613849) [title only; page returned 403]
- *CVPR PBVS workshop*: PBVS 2025 (21st edition) ran a "Multi/Cross Modal Aerial Imagery Translation Challenge (MAVIC-T)" covering SAR/EO/RGB/IR translation — [PBVS 2025 MAVIC-T paper](https://openaccess.thecvf.com/content/CVPR2025W/PBVS/papers/Bowald_3rd_Multi-modal_Aerial_View_Image_Challenge_Sensor_Domain_Translation_-_CVPRW_2025_paper.pdf); [PBVS 2025 workshop page](https://cvpr.thecvf.com/virtual/2025/workshop/32351). "Borrow from Anywhere" (translation-aided thermal detection) is a PBVS 2019 paper — [CVF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Devaguptapu_Borrow_From_Anywhere_Pseudo_Multi-Modal_Object_Detection_in_Thermal_Imagery_CVPRW_2019_paper.pdf)

### Inferences
- **TIM is viable only with a measurement reframing.** The reframing would include:
  - the 753-pair FLIR acquisition as the authors' own experimental validation (camera model, NETD, spectral band, distance, emissivity and ambient conditions, registration procedure and its residual error);
  - detection performance reported as a measurement with Type-A uncertainty across seeds and images (e.g. bootstrap CIs), following GUM vocabulary;
  - C2 framed as quantifying the instrument's systematic errors: false insertion and false deletion rates, with uncertainty.

  Without this, the "AI/CV main novelty" desk-reject rule applies almost word for word. The current framing (§1 of RESEARCH_FINDINGS, "an instrument for generating a results table") is a CV framing.
- **For Information Fusion, the fusion link must be made explicit.** For example: "decision/knowledge-level fusion of a visible-domain detector prior with a thermal sensor stream". C1's loop is task-driven fusion training in the SeAFusion lineage, with a translator instead of a fuser. An optional add-on would directly strengthen scope fit: an extra experiment that fuses translated-visible and raw-thermal detections, or uses the translator inside an IR–VIS fusion pipeline. As it stands, a single-input translator is borderline.
- The ranking below is an inference from scope evidence:

  | Rank | Venue | Reason |
  | --- | --- | --- |
  | 1 | IEEE TCSVT or Pattern Recognition | Method-first; C1 + C2 carry it [scope not verified this session] |
  | 2 | Information Fusion | Needs the fusion framing |
  | 3 | Measurement Science and Technology / Measurement | Topical precedent; lower bar on CV novelty |
  | 4 | IEEE TIM | Only with a full measurement reframing |
  | Short version / fallback | CVPR PBVS workshop or WACV | Quick community exposure |

  TPWRD/IJEPES fit only if the power-engineering fault-diagnosis angle leads. I did not verify their recent CV content.

### Gaps
- I could not enumerate TIM 2023–2026 or Information Fusion 2023–2026 papers on IR→VIS translation via IEEE Xplore or ScienceDirect search. Those sites were not fetched; this needs a manual Xplore query (`"Publication Title": "IEEE Transactions on Instrumentation and Measurement"` AND (thermal OR infrared) AND (translation OR colorization)).
- I did not verify the aims and scope of IEEE TCSVT, TGRS/JSTARS, Pattern Recognition, ESWA, EAAI, IJEPES or TPWRD this session.
- I did not verify the TII 2025 power-equipment IR paper.

---

## Q2. What do typical accepted papers include (datasets, baselines, ablations, statistical tests, code/data policy)? Does TIM want measurement/uncertainty framing?

### Takeaway
TIM explicitly requires GUM-compliant uncertainty, a described measurement chain, and the authors' own experimental data. Information Fusion requires a data availability statement. Neither mandates code release. The closest prior works evaluate on 2–4 public paired sets (LLVIP, FLIR, M3FD, plus DroneVehicle or KAIST) against fine-tuning or translation baselines. I found no evidence that seed-level significance testing is common. That makes our ≥3-seed sign-flip tests a differentiator, not a requirement.

### Cited Findings
- **TIM** requires GUM compliance, a distinction between algorithmic metrics and measurement-system characteristics, and enough measurement-chain detail to replicate the results. Without own experimental validation, authors must give a "strong justification" — [TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- **Information Fusion** requires a data availability statement, Highlights and a graphical abstract — [Information Fusion Guide for Authors](https://www.sciencedirect.com/journal/information-fusion/publish/guide-for-authors) [snippet]
- **ModTr (ECCV 2024)** trains a small translator against a frozen RGB detector's loss. It evaluates IR→RGB on "two well-known datasets", compares against standard fine-tuning, and releases code — [arXiv 2404.01492](https://arxiv.org/abs/2404.01492); [GitHub](https://github.com/heitorrapela/ModTr) [verified]. The dataset names were not stated in the abstract; the searched literature associates the group's work with LLVIP and FLIR.
- **WiSE-OD (WACV 2026)** uses LLVIP-C and FLIR-C (corrupted versions), M3FD as real out-of-distribution data, four RGB-pretrained detectors and two robust baselines — [arXiv 2507.18925](https://arxiv.org/abs/2507.18925) [verified]
- **Multispectral detection papers** in 2025 routinely use four benchmarks: FLIR, LLVIP, DroneVehicle and M3FD. FLIR-aligned is described as 4,129 train / 1,013 test pairs over 3 classes — [arXiv search summary incl. 2506.14696 YOLOv11-RGBT](https://arxiv.org/pdf/2506.14696) [snippet]
- **LLVIP**: 16,836 visible–IR pairs (33,672 images) — [LLVIP, ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/papers/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.pdf) [snippet]

### Inferences
- **Four datasets (custom + 3 public) is at or above the norm.** Seed-level significance tests plus a causal ablation (C1) exceed what the closest CV papers report. Lead with this rigour; it is especially relevant to TIM's uncertainty requirement.
- **Required baselines for any venue**, given ModTr's precedent:
  - ModTr itself (code is public);
  - fine-tuning the detector on thermal;
  - unpaired and paired translation without the detection loss.
- **The "frozen detector, no forgetting" claim is already ModTr's.** Our method must be positioned as a successor: a diffusion/one-step translator, a safety-critical domain, a faithfulness guarantee and causal evidence. It must not be presented as introducing detection-loss translation.

### Gaps
- I did not systematically count datasets, baselines or statistical tests in recent TIM or Information Fusion papers. That would require full-text access to a sample.
- ModTr's exact datasets, seeds and splits were not confirmed from full text (the PDF was not parsed).

---

## Q3. C2: do object-level faithfulness / hallucination / semantic-consistency metrics already exist for image translation? Is a detector-based invent/erase metric already proposed?

### Takeaway
The *idea* of judging translation by a downstream task is old and well established: FCN-score in pix2pix/CycleGAN, and downstream segmentation or classification as a hallucination measure in medical translation. A generic "detector-consistency" metric is therefore **not novel**. What appears unclaimed in thermal→visible translation is a **decomposed, paired, object-level invent/erase rate**. That means separate false-insertion and false-deletion rates, measured against both real-visible detections and ground truth, with uncertainty and a safety-relevance weighting. Frame C2 as a protocol or decomposition for safety-critical translation, not as a new metric family.

### Cited Findings
- **ECCV 2024, Kim et al.**: "Tackling Structural Hallucination in Image Translation with Local Diffusion". Diffusion translators hallucinate on out-of-distribution regions (e.g. unseen tumours). The paper measures hallucination via downstream tasks (digit classification, tumour segmentation Dice, anomaly detection) because "most hallucinated features are small and localized" and barely move PSNR/SSIM. It reports a 40% / 25% reduction in misdiagnosis on medical / natural datasets — [arXiv 2404.05980](https://arxiv.org/abs/2404.05980) [abstract verified; metric details from search snippet of the ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10498.pdf)
- **Konz et al. 2024**, "Rethinking Perceptual Metrics for Medical Image Translation". Perceptual metrics (FID, LPIPS, etc.) "do not generally correlate with segmentation metrics", and FID is particularly inconsistent. Tested on breast MRI and lumbar MRI→CT — [arXiv 2404.07318](https://arxiv.org/abs/2404.07318) [verified]
- **VIGIL (2026)** is a hallucination-detection pipeline for image recontextualization. It checks object-level fidelity, background consistency and *omission* (i.e., both insertion and erasure) — [arXiv 2602.14633](https://arxiv.org/html/2602.14633v1) [snippet]
- **CHAIR** (object hallucination in captioning) measures the fraction of mentioned objects absent from the image. It is the canonical "invented object" metric in vision-language work and is a useful analogy — [Rohrbach et al., EMNLP 2018](https://aclanthology.org/D18-1437.pdf) [snippet]
- **"Hallucination Index: An Image Quality Metric for Generative Reconstruction Models"** (2024) — [ResearchGate](https://www.researchgate.net/publication/382331785_Hallucination_Index_An_Image_Quality_Metric_for_Generative_Reconstruction_Models) [title only]
- **[memory, unverified]**
  - pix2pix introduced the "FCN-score": run an off-the-shelf segmenter on translated images and score against labels — [arXiv 1611.07004](https://arxiv.org/abs/1611.07004). CycleGAN reused it — [arXiv 1703.10593](https://arxiv.org/abs/1703.10593). This is the direct ancestor of "detector-consistency".
  - Cohen et al., MICCAI 2018, "Distribution Matching Losses Can Hallucinate Features in Medical Image Translation" shows that CycleGAN-type losses add or remove tumours — [arXiv 1805.08841](https://arxiv.org/abs/1805.08841)
  - Bhadra et al., IEEE TMI 2021, "On Hallucinations in Tomographic Image Reconstruction" defines hallucination maps — [arXiv 2012.00646](https://arxiv.org/abs/2012.00646)
  - "FID is not enough" line: Kynkäänniemi et al., ICLR 2023 — [arXiv 2203.06026](https://arxiv.org/abs/2203.06026); Jayasumana et al., CVPR 2024 (CMMD) — [arXiv 2401.09603](https://arxiv.org/abs/2401.09603); Stein et al., NeurIPS 2023 — [arXiv 2306.04675](https://arxiv.org/abs/2306.04675)
- **Thermal→visible translation papers typically report image-quality metrics plus downstream detection mAP.** The 2026 MST review summarises trends "in terms of translation quality". Its abstract does not mention hallucination or faithfulness checks — [IOPscience review](https://iopscience.iop.org/article/10.1088/1361-6501/ae7822) [verified abstract]

### Inferences
- **What C2 can honestly claim:**
  1. A *paired*, *object-level* decomposition into invention (detections on translated images matched neither to GT nor to real-visible detections) and erasure (GT/real-visible objects lost after translation). Prior downstream-task metrics give one aggregate score.
  2. Its use as a *guardrail on the detection loop itself*. The loop could improve mAP by "inventing" plausible objects (reward hacking), so a loop paper *needs* C2 to be credible.
  3. Uncertainty-quantified rates, which fit TIM's GUM language.
- **Watch the circularity criticism.** The same frozen detector trains the loop and judges faithfulness. Use a *different*, held-out detector family to score C2, plus GT-anchored rates. Reviewers who know the FCN-score critique (the scorer shares biases with the trained model) will raise this.
- **Rename it.** Call it something like "object-level insertion/deletion rates", and cite FCN-score, Kim et al. 2024, Konz et al. 2024 and Cohen et al. 2018 as lineage. Do not call it a "novel faithfulness metric", which invites a novelty attack.

### Gaps
- I found no paper that defines a detector-based invent/erase rate specifically for IR→VIS translation. That is absence of evidence from ~4 searches, not a systematic search. A targeted Google Scholar check is advised (e.g. "hallucination" + "thermal" + "translation" + "detection", "object insertion deletion rate image translation").
- I did not parse full-text metric definitions from Kim et al. (ECCV 2024); the PDF text extraction failed.

---

## Q4. C3: do benchmark protocols for paired thermal–visible detection (fixed splits, cross-dataset evaluation) already exist?

### Takeaway
Yes, for public data. LLVIP and FLIR-aligned have de-facto fixed splits. M3FD (from TarDAL) ships a detection protocol. WiSE-OD (WACV 2026) adds corruption and cross-dataset out-of-distribution benchmarks (LLVIP-C, FLIR-C, M3FD). Multispectral detection papers use a standard 4-dataset suite. A generic "paired thermal–visible detection benchmark protocol" is therefore **not novel**. The novel part is the **paired power-line component set and its frozen splits**, even if the data cannot be released.

### Cited Findings
- **WiSE-OD (WACV 2026)** introduces LLVIP-C and FLIR-C, cross-modality out-of-distribution robustness benchmarks, and uses M3FD for real out-of-distribution evaluation across four RGB-pretrained detectors — [arXiv 2507.18925](https://arxiv.org/abs/2507.18925) [verified]
- **The 2025 multispectral detection suite** is FLIR, LLVIP, DroneVehicle and M3FD. FLIR-aligned is 4,129/1,013 pairs over 3 classes — [YOLOv11-RGBT, arXiv 2506.14696](https://arxiv.org/pdf/2506.14696) and the [FD2-Net, arXiv 2412.09258](https://arxiv.org/pdf/2412.09258) search summary [snippet]
- **LLVIP** has 16,836 pairs and an official dataset paper — [LLVIP ICCVW 2021](https://openaccess.thecvf.com/content/ICCV2021W/RLQ/papers/Jia_LLVIP_A_Visible-Infrared_Paired_Dataset_for_Low-Light_Vision_ICCVW_2021_paper.pdf) [snippet]
- **ModTr** releases code and an evaluation pipeline for IR→RGB translation-for-detection — [GitHub heitorrapela/ModTr](https://github.com/heitorrapela/ModTr) [verified existence]
- **[memory, unverified]**
  - M3FD was introduced with TarDAL (Liu et al., CVPR 2022), which also proposed detection-driven IR–VIS fusion — [arXiv 2203.16220](https://arxiv.org/abs/2203.16220)
  - VIFB is an IR–VIS image-fusion benchmark — [arXiv 2002.03322](https://arxiv.org/abs/2002.03322)
- **Power-line inspection reviews** exist, e.g. "Deep Learning in Automated Power Line Inspection: A Review" (2025) — [arXiv 2502.07826](https://arxiv.org/pdf/2502.07826) [snippet]. I found no public *paired* thermal–visible power-component *detection* benchmark.

### Inferences
- **Downgrade C3.** Make it a "domain benchmark + reproducible protocol". The core: frozen splits, seeds, a stated detector and training budget, and paired real-visible upper bound vs raw-thermal lower bound for every dataset. Release split files, configs and code even if the images stay private. The fallback in RESEARCH_FINDINGS §11 (protocol only, if release is blocked) is consistent with this.
- **The one protocol element that seems uncommon** is reporting, per dataset, three numbers under one detector and budget:
  1. the *visible upper bound*,
  2. the *raw-thermal (zero-shot RGB detector) floor*,
  3. the *thermal-fine-tuned* reference.

  This is what makes C4 computable, so C3 and C4 should be presented together.
- **Watch FLIR-aligned overlap.** Its splits are widely used but come from several "aligned" variants. State which one is used.

### Gaps
- I did not verify a dedicated "translation-for-detection" benchmark paper beyond ModTr/WiSE-OD; there may be one in 2025–2026.
- I did not verify MSRS's detection protocol (MSRS is primarily a fusion/segmentation set).

---

## Q5. C4: are there prior analyses of when translation helps downstream detection vs training on the target modality, annotation-budget equivalence ("worth N labels"), or modality-gap characterisation?

### Takeaway
Pieces exist, but no unified "map" was found:
- ModTr compares translation against full fine-tuning.
- "Borrow from Anywhere" claims translation helps with less thermal data.
- WiSE-OD and SS-DC characterise the RGB–IR gap for adaptation.
- Thermal-Det (CVPR 2026) contrasts zero-shot and fine-tuned thermal detection.
- A synthetic-data analogue of label-equivalence exists ("An Annotation Saved is an Annotation Earned").

What was **not found**: a cross-dataset regression of translation gain against each dataset's visible–thermal detection gap, together with a "translation is worth ≈N thermal annotations" break-even estimate. C4 is the **most novel** secondary contribution, and it directly defends the premise.

### Cited Findings
- **ModTr**: a translator against a frozen RGB detector performs "comparably or better than standard fine-tuning, without forgetting" on two IR datasets — [arXiv 2404.01492](https://arxiv.org/abs/2404.01492) [verified]
- **Borrow from Anywhere (CVPRW PBVS 2019)**: pseudo-RGB from CycleGAN/UNIT feeds a two-branch detector. It "has the ability to learn with less data from the thermal domain". Translated images were "perceptually far from natural domain images" — [arXiv 1905.08789](https://arxiv.org/abs/1905.08789) [snippet]
- **TIRDet (ACM MM 2023)**: mono-modality thermal detection built on prior thermal→visible translation — [ACM DL](https://dl.acm.org/doi/10.1145/3581783.3613849) [title only]
- **Thermal-Det (CVPR 2026)**: zero-shot open-vocabulary thermal detection without thermal annotations, using a large synthetic thermal set made by F-ViTA translation of GroundingCap-1M. It compares zero-shot against fully fine-tuned results — [arXiv 2605.10130](https://arxiv.org/abs/2605.10130); [CVPR 2026 supplementary](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ranasinghe_Thermal-Det_Language-Guided_Cross-Modal_CVPR_2026_supplemental.pdf) [snippet]. This is the *reverse* direction (visible→thermal data synthesis), and a competing paradigm for "no thermal annotations".
- **F-ViTA (2025)**: foundation-model-guided visible→thermal translation — [arXiv 2504.02801](https://arxiv.org/abs/2504.02801) [snippet]
- **WiSE-OD**: weight-space ensembling of RGB zero-shot and IR fine-tuned weights, motivated by the RGB–IR modality gap — [arXiv 2507.18925](https://arxiv.org/abs/2507.18925) [verified]
- **SS-DC (2025)**: spatial-spectral decoupling "across the visible-infrared gap" for domain-adaptive detection — [arXiv 2507.12017](https://arxiv.org/pdf/2507.12017) [snippet]
- **Hinterstoisser et al. 2019**: purely synthetic training compared against ~185 h of real labelling (+6 h correction) vs ~5 h of synthetic setup. This is a cost-equivalence framing in the synthetic-data literature — [arXiv 1902.09967](https://arxiv.org/pdf/1902.09967) [snippet]
- **"Exploring Thermal Images for Object Detection in Underexposure Regimes"** studies thermal vs visible detection across illumination — [arXiv 2006.00821](https://arxiv.org/pdf/2006.00821) [title/snippet only]

### Inferences
- **C4 is the strongest secondary contribution for novelty.** It turns the obvious reviewer question ("why not just annotate 200 thermal images?") into a result. The quantitative statement "translation without thermal labels matches a thermal-trained detector with ~150–214 power-line annotations" is the kind of operator-facing number that TIM-style (measurement and cost) and application venues value. It also matches the fallback framing in RESEARCH_FINDINGS §10.
- **Framing for the gap analysis:** a scatter of Δ(translation − raw thermal) against Δ(visible − raw thermal) over the 4–5 datasets, with CIs. It is novel as a cross-dataset characterisation, but with n≈5 datasets it must be presented as descriptive, not as a fitted law.
- **Cite Thermal-Det/F-ViTA as the competing route** (synthesise thermal training data instead of translating at test time) and explain why test-time translation is preferable when the visible detector is fixed or certified.

### Gaps
- I found no paper that explicitly computes a "translation ≡ N target-modality labels" break-even curve for IR detection. Absence of evidence from 3 searches; a Google Scholar check for "label efficiency" + "modality translation" + "thermal" is advised.
- The full-text results of "Borrow from Anywhere" on reduced thermal data fractions were not verified.

---

## Q6. Bottom line: which framing and contributions are most publishable; which should be downplayed?

### Takeaway
Lead with **C1 + C4** as one story: a detection-supervised translator lets a frozen visible detector work on thermal imagery, and here is *when* and *how much* it is worth in thermal labels. Use **C2** as the necessary safety guardrail, framed as object-level insertion/deletion rates in the lineage of FCN-score and medical-hallucination work. Downplay **C3** to a reproducible protocol plus a domain dataset. Choose the venue framing deliberately: measurement/uncertainty for TIM, knowledge/decision fusion for Information Fusion, or method-first for TCSVT/PR.

### Cited Findings
- TIM desk-rejects AI/CV-novelty papers and dataset-only papers, and requires GUM-style uncertainty — [TIM Information for Authors](https://ieee-ims.org/publication/ieee-tim/information-authors)
- Information Fusion's scope is "multi-sensor, multi-source, multi-process information fusion" — [Information Fusion Guide for Authors](https://www.sciencedirect.com/journal/information-fusion/publish/guide-for-authors)
- ModTr already established detection-loss-trained IR→RGB translation with a frozen detector at ECCV 2024 — [arXiv 2404.01492](https://arxiv.org/abs/2404.01492)
- Downstream-task hallucination evaluation of diffusion translation is established — [Kim et al. ECCV 2024, arXiv 2404.05980](https://arxiv.org/abs/2404.05980); [Konz et al. 2024, arXiv 2404.07318](https://arxiv.org/abs/2404.07318)
- Public paired-detection benchmarks and out-of-distribution protocols exist — [WiSE-OD arXiv 2507.18925](https://arxiv.org/abs/2507.18925)

### Inferences
- **Most publishable:**
  - C1, with causal evidence (same model ± detector loss, ≥3 seeds, sign-flip tests), positioned as "ModTr-style supervision extended to one-step diffusion translators, with causal attribution and faithfulness control".
  - C4, which is novel and premise-defending.
- **Keep, but reframe:** C2 as the "insertion/deletion audit" that rules out reward hacking. Score it with a *held-out* detector to avoid circularity.
- **Downplay:** C3 as a stand-alone contribution, and any claim of being "first" to use detection loss in IR→VIS translation (ModTr and TIRDet pre-empt it). Also any "novel metric" language for C2.
- **Venue-specific edits:**

  | Venue | Edits |
  | --- | --- |
  | TIM | Section on the measurement chain (FLIR camera specs, registration error, acquisition conditions). GUM-style uncertainty on all detection metrics. C2 as instrument error characterisation. Cite I&M literature (TIM IR–VIS fusion papers, MST review). Highlight the 753 pairs as the authors' own acquisition. |
  | Information Fusion | Recast the loop as task-driven cross-modal knowledge fusion (SeAFusion lineage). Ideally add a fusion variant (translated-visible ⊕ thermal at decision level). |
  | TCSVT / PR | Method-first. Expect "why translate?", answered by C4. |

- **Biggest risk:** the TIM desk-reject criteria. If TIM stays primary, run a pre-submission check against the nine rejection criteria. Otherwise move TIM to secondary and submit first to Information Fusion (with the fusion framing) or TCSVT/PR.

### Gaps
- Acceptance rates and review times for these venues were not collected.
- Whether Information Fusion editors have accepted single-input translation papers is unverified. Checking 2–3 recent Information Fusion "colorization/translation" papers on ScienceDirect would settle it.
