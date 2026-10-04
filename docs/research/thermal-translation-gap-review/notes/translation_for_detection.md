# Translation-for-detection: IR/thermal → visible translation trained or judged by a downstream detector (non-diffusion)

Scope: 2017–2026 methods that translate thermal/IR imagery toward the visible domain with detection (or another task) as the training signal or evaluation target. Diffusion-specific work is covered by another researcher and excluded here, except where noted. Research date: 2026-10-03. Every claim has a URL. Claims taken only from search snippets or auto-summaries (not the primary PDF) are marked **[snippet-level]**.

## Q1. Which papers train an IR→RGB translator with a detector loss, especially a FROZEN pretrained visible detector? Ranked by overlap with our method

### Takeaway
The core idea, "train an IR→RGB translator by backpropagating the loss of a frozen RGB detector, then run the unchanged RGB detector on the translated image," **is already published** by the ÉTS Montréal / LIVIA group. HalluciDet (WACV 2024) and ModTr (ECCV 2024, apparently main conference, not workshop) do exactly this on LLVIP and FLIR, and both have public code. A reviewer will reject any claim that "detection-in-the-loop thermal→visible translation" is new in itself. What these papers do *not* have: a paired visible-reconstruction term (so their outputs are not visible-looking images), a one-step diffusion generator, YOLO-family detectors, power-line or industrial-inspection data, a faithfulness study, or a comparison against thermal-trained detectors as a function of the annotation budget.

### Cited Findings

**Rank 1 (closest): ModTr: "Modality Translation for Object Detection Adaptation Without Forgetting Prior Knowledge"** (Medeiros, Aminbeidokhti, Guerrero Peña, Latortue, Granger, Pedersoli)
- arXiv v1 1 Apr 2024, v3 31 Jul 2024. The arXiv comment says "ECCV 2024" — [arXiv:2404.01492](https://arxiv.org/abs/2404.01492). The paper is in the main ECCV proceedings on ECVA ([PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12401.pdf)) and has an ECCV 2024 slide deck ([slides](https://eccv.ecva.net/media/eccv-2024/Slides/816.pdf)). **Correction:** RESEARCH_FINDINGS §8 calls it "ECCV 2024 workshops". The evidence points to the **main conference**.
- Method: "a small transformation network trained to directly minimize the detection loss. The original RGB model can then work on the translated inputs without any further changes or fine-tuning to its parameters" — [arXiv abstract](https://arxiv.org/abs/2404.01492).
- Generator: a U-Net (ResNet34 default; MobileNetv2/v3 variants down to about 3.1M params) with a 3-channel sigmoid output. That output is **fused with the IR input by a non-parametric operation** (element-wise product works best; addition and attention were also tried). Detector: Faster R-CNN, RetinaNet or FCOS with **frozen COCO weights**. Training uses only IR images plus boxes. It needs **no RGB images** and no source data — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3).
- Datasets: LLVIP (pedestrians) and FLIR-aligned (cars, bicycles, people). No M3FD, no MSRS — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3).
- Reported AP (COCO-style AP, not AP50) — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3):
  - **LLVIP, FCOS / RetinaNet / Faster R-CNN:**
    - ModTr⊙: 57.63 / 54.83 / 57.97
    - full fine-tuning: 57.37 / 53.79 / 59.62
    - HalluciDet: 28.00 / 19.95 / 57.78
    - FastCUT: 19.39 / 18.11 / 22.91
  - **FLIR, same detectors:**
    - ModTr⊙: 35.49 / 34.27 / 37.21
    - fine-tuning: 27.97 / 28.46 / 30.93
    - HalluciDet: 23.74 / 22.29 / 29.91
    - FastCUT: 24.02 / 22.00 / 26.68
- Forgetting: a single shared detector with N ModTr translators keeps the original COCO AP (38.41 for FCOS). Joint fine-tuning collapses it to 0.33 AP — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3).
- The authors say outright that the outputs are not visible-looking: "Although the obtained intermediate representations are not visually pleasant, they prove more efficient for incorporating the knowledge necessary for the OD" — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3).
- Stated contributions: (1) adapt RGB detectors to a new modality without source data, (2) preserve detector knowledge for multi-modality reuse, (3) broad empirical validation — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3). Code: [github.com/heitorrapela/ModTr](https://github.com/heitorrapela/ModTr).
- No low-data / annotation-budget experiment is reported — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3) **[snippet-level: auto-summary of the full text]**.

**Rank 2: HalluciDet: "Hallucinating RGB Modality for Person Detection Through Privileged Information"** (Medeiros, Guerrero Peña, Aminbeidokhti, Dubail, Granger, Pedersoli), WACV 2024
- Venue and code: [CVF open access](https://openaccess.thecvf.com/content/WACV2024/html/Medeiros_HalluciDet_Hallucinating_RGB_Modality_for_Person_Detection_Through_Privileged_Information_WACV_2024_paper.html), [arXiv:2310.04662](https://arxiv.org/abs/2310.04662), [code](https://github.com/heitorrapela/HalluciDet), [project page](https://heitorrapela.github.io/HalluciDet/).
- Method: an IR→RGB translation network that, "instead of focusing on reconstructing the original image on the IR modality, seeks to reduce the detection loss of an RGB detector" — [project page](https://heitorrapela.github.io/HalluciDet/).
- Generator: a U-Net with attention blocks (ResNet34 default; MobileNet and ResNet18 ablations).
- Detector training happens in two stages. First, the COCO-pretrained detector is **fine-tuned on the target dataset's RGB images**. Then it is **frozen** while the translator trains.
- Loss: L_hall = L_cls + λ·L_reg. This is detection loss only, with **no RGB reconstruction term**.
- Detectors: FCOS, RetinaNet, Faster R-CNN. Data: LLVIP and FLIR ADAS paired sets; RGB is used only to fine-tune the detector.
- Source for the three points above: [arXiv HTML](https://arxiv.org/html/2310.04662).
- Reported AP50 on LLVIP — [arXiv HTML](https://arxiv.org/html/2310.04662):
  - Faster R-CNN: HalluciDet 90.92 vs IR fine-tuning 84.94
  - Faster R-CNN, pixel-inversion baseline: 71.83
  - FCOS: HalluciDet 63.28 vs FastCUT 46.87
- Data-efficiency claim: "only 30% of LLVIP training data needed to match full fine-tuned baseline" — [arXiv HTML](https://arxiv.org/html/2310.04662) **[snippet-level: extracted by an auto-summary; check the exact figure/table in the PDF before citing]**.
- ModTr's own comparison: "HalluciDet requires RGB images for an initial fine-tuning of the model, while ModTr can work without that fine-tuning by reusing the detector's zero-shot knowledge" — [ModPrompt arXiv HTML](https://arxiv.org/html/2412.00622v1) **[snippet-level]**.

**Rank 3: Abbott et al., "Unsupervised Object Detection via LWIR/RGB Translation"**, CVPRW (PBVS) 2020
- [CVF page](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Abbott_Unsupervised_Object_Detection_via_LWIRRGB_Translation_CVPRW_2020_paper.html), [mlanthology](https://mlanthology.org/cvprw/2020/abbott2020cvprw-unsupervised/).
- A modified CycleGAN for unpaired LWIR↔RGB. In experiment 1, LWIR→RGB lets them "use an RGB trained detection algorithm… remov[ing] the need for labelled LWIR imagery." Combining synthetic RGB with real LWIR raised F1 on the RGB-trained detector. The best result (F1 85.6%) came from the opposite direction: RGB→LWIR followed by fine-tuning — [mlanthology abstract](https://mlanthology.org/cvprw/2020/abbott2020cvprw-unsupervised/).
- A Faster R-CNN supplies boxes for an **object-specific loss** that keeps ROI pixels consistent during the cycle — [Semantic Scholar/ResearchGate snippets](https://www.researchgate.net/publication/343270129_Unsupervised_object_detection_via_LWIRRGB_translation) **[snippet-level]**. This is detector-*guided* (ROI consistency), not detector-*loss* backprop.
- **Why it matters for us:** this is the earliest explicit "translate thermal to RGB so an unchanged RGB detector works, with no thermal labels" paper we found. It should be cited as the origin of the annotation-free framing.

**Rank 4: Devaguptapu et al., "Borrow from Anywhere: Pseudo Multi-modal Object Detection in Thermal Imagery"**, CVPRW (PBVS) 2019
- [arXiv:1905.08789](https://arxiv.org/abs/1905.08789), [CVF PDF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Devaguptapu_Borrow_From_Anywhere_Pseudo_Multi-Modal_Object_Detection_in_Thermal_Imagery_CVPRW_2019_paper.pdf), [code MMTOD](https://github.com/tdchaitanya/MMTOD).
- CycleGAN or UNIT creates pseudo-RGB from thermal. A multimodal Faster R-CNN then fuses thermal and pseudo-RGB features. The detector is trained on thermal labels; translation is unpaired and is **not** trained with the detector loss — [arXiv abstract](https://arxiv.org/abs/1905.08789).

**Rank 5: Herrmann, Ruf, Beyerer, "CNN-based thermal infrared person detection by domain adaptation"**, SPIE 10643, 2018
- [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10643/2304400/CNN-based-thermal-infrared-person-detection-by-domain-adaptation/10.1117/12.2304400.full).
- A hand-designed preprocessing step "transforms the IR data as close as possible to the RGB domain" so that RGB-pretrained detectors work. The remaining gap is closed by fine-tuning on limited thermal data. Evaluated on KAIST — [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10643/2304400/CNN-based-thermal-infrared-person-detection-by-domain-adaptation/10.1117/12.2304400.full). This is the non-learned ancestor of the idea; HalluciDet's "pixel inversion" baseline plays the same role.

**Rank 6: IR2VI** (Liu, John, Blasch, Liu, Huang), CVPRW 2018
- [arXiv:1806.09565](https://arxiv.org/abs/1806.09565), [CVF](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w21/html/Liu_IR2VI_Enhanced_Night_CVPR_2018_paper.html).
- Unsupervised GAN (CycleGAN-style) thermal→visible translation with a structure-connection module and an **ROI focal loss**. It is motivated by night-vision perception, not by a detector loss — [arXiv](https://arxiv.org/abs/1806.09565). This is the "CycleGAN pseudo-RGB" baseline lineage named in RESEARCH_FINDINGS §8.

**Thermal colorization works (no detector loss; at most a task-based evaluation)**
- **TIC-CGAN** (Kuang et al.): a conditional GAN for thermal colorization with content, adversarial, perceptual and TV losses — [arXiv:1810.05399](https://arxiv.org/abs/1810.05399) **[snippet-level]**.
- **PearlGAN** (Luo et al., IEEE T-ITS): top-down attention plus structured gradient-alignment loss for nighttime TIR→daytime colour. Semantic preservation is evaluated with **pixel-level annotations on FLIR/KAIST subsets**; no task loss appears in the abstract — [arXiv:2104.14374](https://arxiv.org/abs/2104.14374).
- **FoalGAN** (Luo et al., 2023): occlusion-aware mixup, appearance-consistency loss and traffic-light appearance loss. These are visual-quality losses, not detection losses — [arXiv:2310.15688](https://arxiv.org/abs/2310.15688).
- **MUGAN**: a Mixed-Skipping UNet plus GAN for TIR colorization, focused on image quality — [IEEE Xplore](https://ieeexplore.ieee.org/document/9935274/) **[snippet-level]**.
- **I2V-GAN**: unpaired IR→visible *video* translation — [arXiv:2108.00913](https://arxiv.org/abs/2108.00913) **[snippet-level]**.

**Items in the original proposal that point the wrong way (visible→thermal) or contain no translation**
- **arXiv:2002.06770** (Liu, Li, Li, "Unsupervised Image-generation Enhanced Adaptation for Object Detection in Thermal images") generates **fake *thermal* images from labelled visible data** with CycleGAN plus intensity inversion, then applies Domain-Adaptive Faster R-CNN — [arXiv](https://arxiv.org/abs/2002.06770). This is **visible→thermal data synthesis**, not thermal→visible translation. It is a valid "annotation-transfer" baseline, but it is a different pipeline from ours.
- **ThermalGAN** (Kniaz et al., ECCVW 2018) does **colour→thermal** translation for cross-modal person re-ID — [CVF](https://openaccess.thecvf.com/content_eccv_2018_workshops/w35/html/Kniaz_ThermalGAN_Multimodal_Color-to-Thermal_Image_Translation_for_Person_Re-Identification_in_Multispectral_ECCVW_2018_paper.html).
- **InfraGAN** does visible→IR — [Pattern Recognition Letters](https://www.sciencedirect.com/science/article/abs/pii/S0167865522000332).
- **DR-AVIT** does aerial visible→IR with disentangled representations — [ResearchGate](https://www.researchgate.net/publication/380918863_DR-AVIT_Towards_Diverse_and_Realistic_Aerial_Visible-to-Infrared_Image_Translation) **[snippet-level]**.
- **Task-conditioned DA (Kieu et al., ECCV 2020)** has **no translation**. It adapts an RGB-trained YOLOv3 to thermal, with an auxiliary day/night classification head that conditions the detector; the detector is fine-tuned. Evaluated on KAIST — [ECVA PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf), [code](https://github.com/mrkieumy/task-conditioned).
- **Meta-UDA (VS et al., WACV 2022)** has **no translation**. It meta-learns the detector's initialization for feature-level UDA from visible to thermal. Evaluated on KAIST and DSIAC — [arXiv:2110.03143](https://arxiv.org/abs/2110.03143), [CVF](https://openaccess.thecvf.com/content/WACV2022/html/VS_Meta-UDA_Unsupervised_Domain_Adaptive_Thermal_Object_Detection_Using_Meta-Learning_WACV_2022_paper.html).

**Follow-ups by the same LIVIA / ÉTS group (2024–2026)**
- **ModPrompt**, "Visual Modality Prompt for Adapting Vision-Language Object Detectors": an encoder–decoder **visual prompt** (input-space, image-dependent) that adapts **frozen** YOLO-World and Grounding DINO to IR (LLVIP, FLIR) and depth (NYUv2). Presented at ICCV 2025 — [arXiv:2412.00622](https://arxiv.org/abs/2412.00622), [ICCV 2025 slides](https://iccv.thecvf.com/media/iccv-2025/Slides/1037.pdf). This is effectively ModTr extended to open-vocabulary detectors and depth.
- **MiPa**, "Mixed Patch Visible-Infrared Modality Agnostic Object Detection", WACV 2025: one shared encoder for RGB or IR. No translation — [CVF PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Medeiros_Mixed_Patch_Visible-Infrared_Modality_Agnostic_Object_Detection_WACV_2025_paper.pdf).
- **WiSE-OD**: a robustness benchmark for IR object detection (Medeiros, Belal, Aminbeidokhti, Granger, Pedersoli). No translation — [arXiv:2507.18925](https://arxiv.org/abs/2507.18925).

### Inferences
Overlap ranking, from the method-design point of view:

| Rank | Work | Generator | Detector during translator training | Paired / RGB target used by translator? | Data | Code |
|---|---|---|---|---|---|---|
| 1 | ModTr (ECCV'24) | U-Net + fusion with IR input | Frozen, COCO zero-shot (FRCNN / RetinaNet / FCOS) | No RGB at all; detection loss only | LLVIP, FLIR | Yes |
| 2 | HalluciDet (WACV'24) | Attention U-Net | Frozen, after fine-tuning on target RGB | RGB only to fine-tune the detector; detection loss only | LLVIP, FLIR | Yes |
| 3 | Abbott et al. (CVPRW'20) | Modified CycleGAN | Faster R-CNN gives ROI consistency, not loss backprop (snippet-level) | Unpaired | 2 datasets (names not verified) | Not found |
| 4 | Borrow-from-Anywhere (CVPRW'19) | CycleGAN / UNIT | Detector trained on thermal labels | Unpaired | FLIR, KAIST (not re-verified) | Yes |
| 5 | Herrmann et al. (SPIE'18) | Hand-crafted preprocessing | RGB-pretrained, then fine-tuned | — | KAIST | — |
| 6 | IR2VI (CVPRW'18) | CycleGAN-style | None | Unpaired | — | — |

- Our method differs from Rank 1–2 in five ways:
  - (a) We **combine** a paired pix2pix reconstruction or perceptual objective with the detector loss. Our output is meant to be a faithful *visible image*; ModTr's is a detector-friendly "not visually pleasant" representation, and HalluciDet uses no reconstruction term at all.
  - (b) Our generators are pix2pix and a one-step SD-Turbo translator (pix2pix-turbo). Theirs are small U-Nets.
  - (c) Our detector is YOLO11, a modern one-stage detector with DFL/CIoU losses.
  - (d) Our domain is power-line components.
  - (e) We run an annotation-budget comparison against thermal-trained detectors.
- Our own FLIR result (the loop only recovers to the raw-thermal floor) sits awkwardly next to ModTr's FLIR result, where ModTr beats full fine-tuning by about 7 AP. A reviewer will ask why. Plausible reasons:
  - ModTr fuses the IR input back in (element-wise product), so the detector always "sees" the thermal structure.
  - ModTr has no visible-reconstruction constraint.
  - ModTr uses AP rather than mAP50, and FRCNN/RetinaNet/FCOS rather than YOLO.
  The paper should include ModTr as a baseline on FLIR and LLVIP using their released code. That is the single most important missing baseline.
- HalluciDet and ModTr should be added to RESEARCH_FINDINGS §8 as **mandatory baselines**. They are more direct competitors than CycleGAN-DA, arXiv:2002.06770 or Meta-UDA.

### Gaps
- Whether the HalluciDet WACV version reports FLIR AP50 numbers, and the exact location of the "30% of data" result. Only an auto-summary of the HTML was read; the PDF exceeded the fetch size limit.
- Abbott et al. (2020): exact datasets, whether the Faster R-CNN is frozen, and whether a detection loss is backpropagated. The CVF page returned 403, so only the abstract was read.
- No 2025–2026 paper was found that trains a *thermal→visible GAN/U-Net* with a frozen-detector loss beyond the LIVIA line. Search coverage of Chinese-language journals (Infrared Physics & Technology, Acta Optica Sinica) and MDPI venues is weak; there could be uncited regional work. Google Scholar "cited by" lists for HalluciDet and ModTr could not be retrieved directly; the coverage here comes from keyword searches.

## Q2. For close competitors: generator, frozen vs fine-tuned detector, paired/unpaired, datasets, mAP, code, claimed contribution

### Takeaway
The two close competitors (ModTr, HalluciDet) use small U-Net translators, frozen two-stage and anchor-free detectors (not YOLO), only LLVIP and FLIR (no M3FD, no MSRS, no industrial data), and release code. They claim adaptation without forgetting and without RGB/source data, not visual fidelity.

### Cited Findings
- ModTr: U-Net (ResNet34 / MobileNet) with output fused to the IR input by ⊙; frozen COCO FRCNN / RetinaNet / FCOS; IR plus boxes only; LLVIP and FLIR; AP as listed in Q1; code released. Claim: "adapts RGB detectors to new modalities without source data access; preserves detector knowledge" — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3).
- HalluciDet: attention U-Net; detector COCO-pretrained, then fine-tuned on target RGB, then frozen; paired LLVIP (24,050 training images) and FLIR ADAS (8,258); AP50 on LLVIP up to 90.92 (FRCNN); code released. Claim: task-specific translation using privileged RGB-detector information, with no RGB at inference — [arXiv HTML](https://arxiv.org/html/2310.04662), [GitHub](https://github.com/heitorrapela/HalluciDet).
- ModPrompt (ICCV 2025): an input-space visual prompt (encoder–decoder) for frozen YOLO-World and Grounding DINO; LLVIP, FLIR, NYUv2-depth; code at heitorrapela/ModPrompt — [arXiv:2412.00622](https://arxiv.org/abs/2412.00622).
- In ModTr's tables, the generic unpaired translator FastCUT reaches only 19–23 AP on LLVIP vs about 57 for ModTr and fine-tuning — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3). Reviewers will expect us to show the same gap between "plain translation" and "translation plus detector loss". Our 0.80 → 0.85 mAP50 on power lines is that comparison.

### Inferences
- None of the competitors use YOLO-family detectors. Getting exact YOLO loss gradients through the generator is an engineering contribution worth a sentence, but it is not a scientific novelty.
- None evaluate on M3FD or MSRS. Our use of four public datasets plus a private industrial set is broader coverage than the closest competitors.

### Gaps
- No per-class or mAP50 numbers for ModTr on FLIR were extracted; only COCO-style AP. Read Table 1/2 of the PDF if a like-for-like mAP50 comparison is needed.
- Whether ModTr reports the zero-shot RGB detector on raw IR. The summary says "not reported"; this is unverified.

## Q3. Does any competitor evaluate faithfulness (not inventing/erasing objects) or study WHEN translation beats a thermal-trained detector / annotation budgets?

### Takeaway
We found **no** thermal→visible translation-for-detection paper that measures object-level faithfulness (hallucinated or erased objects). We also found none that maps, against annotation count, when a frozen visible detector on translated images beats a thermal-trained detector. ModTr even admits its outputs are not visually plausible. HalluciDet's "30% of data matches fine-tuning" claim is the closest thing to a data-budget statement, and it concerns translator training data, not a thermal-annotation-equivalence curve.

### Cited Findings
- ModTr: the outputs "are not visually pleasant" and no faithfulness metric is reported — [arXiv HTML v3](https://arxiv.org/html/2404.01492v3).
- HalluciDet: the translator is trained only on detection loss, so nothing constrains non-object regions toward real RGB. It claims 30% of LLVIP data suffices to match the fully fine-tuned baseline — [arXiv HTML](https://arxiv.org/html/2310.04662) **[snippet-level for the 30% figure]**.
- PearlGAN measures "semantic preservation" with pixel-level annotations on FLIR/KAIST subsets — a segmentation-consistency proxy, not detection faithfulness — [arXiv:2104.14374](https://arxiv.org/abs/2104.14374).
- CyCADA names the failure mode "label flipping" (translation changing semantics) and counters it with a semantic-consistency loss from a frozen source classifier — [arXiv:1711.03213](https://arxiv.org/abs/1711.03213). This is the canonical prior concept a faithfulness section should cite.
- Abbott et al. compare LWIR→RGB with an RGB detector against RGB→LWIR with fine-tuning, and find the latter better (F1 85.6%). This is an early "which route wins" comparison, but it varies the pipeline, not the annotation count — [mlanthology](https://mlanthology.org/cvprw/2020/abbott2020cvprw-unsupervised/).
- Herrmann et al. combine IR→RGB-like preprocessing with fine-tuning on "a limited set of thermal IR data", but no budget curve is described in the abstract — [SPIE](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10643/2304400/CNN-based-thermal-infrared-person-detection-by-domain-adaptation/10.1117/12.2304400.full).
- Work on data-constrained IR detection that borrows RGB knowledge exists, e.g. "Tensor Factorization for Leveraging Cross-Modal Knowledge in Data-Constrained Infrared Object Detection" — [arXiv:2309.16592](https://arxiv.org/abs/2309.16592) **[snippet-level: title only, not read]**. Visible→thermal synthesis for low-data regimes also exists, e.g. "Partially fake it till you make it" — [arXiv:2106.13603](https://arxiv.org/abs/2106.13603) **[snippet-level]**. More recently, RGB→IR generation with SD 3.5 + ControlNet raised RF-DETR mAP from 50.8 to 60.1 (Kust4K) and from 25.6 to 38.4 (VTUAV) — [arXiv:2609.02556](https://arxiv.org/abs/2609.02556) (SPIE 2026; diffusion, reverse direction, detector used only for evaluation).

### Inferences
- An **annotation-equivalence analysis** looks unclaimed: "a frozen visible detector on translated thermal equals a thermal-trained detector with ~150–214 labels; the thermal detector wins at 600." So does an **object-level faithfulness audit**: hallucinated or erased objects measured with a detector run on the paired real visible image, or with box-level precision/recall against ground truth on translated vs real visible. Both are defensible novelty claims.
- The FLIR negative result ("the loop only recovers to the raw-thermal floor") is a genuine "when does translation help" finding. It disagrees with ModTr's FLIR gains, so it needs careful explanation (see Q1 inferences). Framed as a mapping study, it is a contribution rather than a weakness.

### Gaps
- We could not confirm whether HalluciDet's or ModTr's supplementary material contains low-label experiments; their PDFs were too large to fetch.
- No hallucination-specific benchmark for IR→RGB translation was found. The search returned only face-hallucination and VLM-hallucination literature.

## Q4. Analogous task-in-the-loop translation in neighbouring modalities that a reviewer could cite as "already done"

### Takeaway
Training an image-to-image or enhancement network through a frozen or joint downstream task network is a well-established pattern: CyCADA (2018), task-driven super-resolution (2018), SAR→optical guided by segmentation (2024), detector-friendly enhancement modules, and task-driven IR–visible *fusion* (SeAFusion, TarDAL, DetFusion). The *mechanism* is not novel. Novelty has to come from the setting, the generator class and the analyses.

### Cited Findings
- **CyCADA** (Hoffman et al., ICML 2018): CycleGAN pixel-level adaptation plus a **semantic-consistency loss from a pre-trained source task model** used as a noisy labeller; feature-level adaptation; unpaired — [arXiv:1711.03213](https://arxiv.org/abs/1711.03213), [PMLR](https://proceedings.mlr.press/v80/hoffman18a/hoffman18a.pdf), [code](https://github.com/jhoffman/cycada_release).
- **Task-Driven Super-Resolution** (Haris, Shakhnarovich, Ukita): the SR sub-network "explicitly incorporates a detection loss in its training objective, via a tradeoff with a traditional [reconstruction] loss" — [arXiv:1803.11316](https://arxiv.org/abs/1803.11316). Structurally this is the closest analogue to our pix2pix + detector-loss objective (reconstruction + detection trade-off). Whether its detector is frozen was not confirmed from the abstract.
- **Seg-CycleGAN** (Zhang et al., 2024): SAR→optical translation for ships, "guided by" a **pre-trained** ship-segmentation model — [arXiv:2408.05777](https://arxiv.org/abs/2408.05777). Frozen status not explicit in the abstract.
- **ModPrompt** applies the same input-space adaptation idea to **depth** (NYUv2) — [arXiv:2412.00622](https://arxiv.org/abs/2412.00622).
- **Low-light enhancement driven by detection loss**: e.g. a task-driven enhancement front-end "supervised directly by the final detection loss" — [search snippet, Springer LDWLE / related](https://link.springer.com/article/10.1007/s40747-024-01681-z) **[snippet-level; attribution of that quote to a specific paper not confirmed]**. "Dynamic Low-Light Image Enhancement for Object Detection via End-to-End Training" — [ResearchGate](https://www.researchgate.net/publication/351406133_Dynamic_Low-Light_Image_Enhancement_for_Object_Detection_via_End-to-End_Training) **[snippet-level]**.
- **Task-driven IR–visible fusion** (two inputs at inference; already in RESEARCH_FINDINGS §4): TarDAL with the M3FD benchmark — [arXiv:2203.16220](https://arxiv.org/abs/2203.16220). SeAFusion and DetFusion are listed in RESEARCH_FINDINGS §4 (not re-verified here).
- **Thermal day/night IR→IR translation with detector loss**: multi-level GANs that translate nighttime to daytime *thermal* for vehicle detection, with a detector loss in training per the auto-summary — [arXiv:2209.09808](https://arxiv.org/abs/2209.09808) **[snippet-level for the detector-loss detail]**.
- **Not verified in this session (cite only after checking)**: IA-YOLO (differentiable image processing trained jointly with YOLO, AAAI 2022), DUNIT (detection-based unsupervised I2I, CVPR 2020), ForkGAN (night→day, ECCV 2020), "Dirty Pixels" (ISP trained with a classification loss). These are well-known analogues, but no primary source was fetched here.

### Inferences
- Reviewers at TIM or Information Fusion will probably cite SeAFusion / TarDAL ("task-in-the-loop is standard in IR–visible fusion") and CyCADA / Task-Driven SR ("task-loss-guided translation is old"). The paper should say up front that it **inherits** this mechanism (as RESEARCH_FINDINGS §4 already says for SeAFusion). The contribution is applying it to *single-input thermal at inference* with an *off-the-shelf visible detector*, plus the analyses.

### Gaps
- Fetches for IA-YOLO, DUNIT, ForkGAN and NIR→RGB task-guided colorization were not done because of the tool budget. NIR→RGB colorization papers found ([arXiv:2404.16685](https://arxiv.org/abs/2404.16685), [arXiv:2312.16040](https://arxiv.org/abs/2312.16040), [arXiv:2107.09237](https://arxiv.org/abs/2107.09237)) appear to optimise image quality, not a task loss **[snippet-level]**.

## Q5. What sub-claim remains genuinely novel for us, and what would a reviewer reject as already done?

### Takeaway
"Detection-in-the-loop thermal→visible translation with a frozen visible detector" is **not** a gap; HalluciDet and ModTr own it, and the general mechanism dates back to CyCADA and Task-Driven SR. The defensible novelty is a combination of five things:
1. A **visible-faithful** translator (paired reconstruction plus detector loss) rather than a detector-only "hallucinated" representation.
2. A **one-step diffusion (SD-Turbo) generator** with exact detector gradients. The diffusion researcher must confirm that no diffusion/turbo IR→RGB detection-loop paper exists.
3. An **annotation-equivalence analysis** against thermal-trained detectors.
4. An **object-level faithfulness audit**.
5. A **new application domain**: power-line component inspection on paired, registered FLIR data, plus a cross-dataset "when does it help" map that includes a negative result on FLIR.

### Cited Findings
- Already done (a reviewer would reject it as new):
  - Translating IR→RGB with a translator trained on a frozen RGB detector's loss, so the unchanged detector runs on translated IR — [HalluciDet](https://arxiv.org/abs/2310.04662), [ModTr](https://arxiv.org/abs/2404.01492).
  - Doing it without fine-tuning the detector and without forgetting — [ModTr](https://arxiv.org/abs/2404.01492).
  - Doing it with open-vocabulary detectors and for depth — [ModPrompt](https://arxiv.org/abs/2412.00622).
  - Annotation-free thermal detection via LWIR→RGB translation plus an RGB detector — [Abbott et al. 2020](https://mlanthology.org/cvprw/2020/abbott2020cvprw-unsupervised/).
  - Pseudo-RGB from thermal to borrow RGB knowledge — [Borrow from Anywhere 2019](https://arxiv.org/abs/1905.08789).
  - Task-loss-guided translation in general — [CyCADA](https://arxiv.org/abs/1711.03213), [Task-Driven SR](https://arxiv.org/abs/1803.11316), [Seg-CycleGAN](https://arxiv.org/abs/2408.05777).
- Not found in any work reviewed:
  - (i) A frozen-detector loss **combined with paired visible supervision** for thermal→visible (HalluciDet and ModTr use detection loss only) — [HalluciDet HTML](https://arxiv.org/html/2310.04662), [ModTr HTML](https://arxiv.org/html/2404.01492v3).
  - (ii) YOLO-family in-loop detectors — same sources.
  - (iii) M3FD or MSRS evaluation, or any power-line / industrial data — same sources.
  - (iv) Faithfulness or annotation-budget studies (see Q3).

### Inferences
- **Recommended claim wording:** "We study *when* detection-in-the-loop translation lets an off-the-shelf visible detector replace thermal annotation. On power-line inspection, translation with a frozen YOLO11 in the loop reaches 0.85 mAP50, equivalent to a thermal detector trained on ~150–214 labels. We extend the loop to a one-step diffusion translator under a visible-faithfulness constraint and audit object hallucination." Position HalluciDet and ModTr as the direct prior art, and as baselines.
- **Claims to avoid:** "first to use detection loss for IR→RGB translation"; "first to reuse a frozen RGB detector on thermal"; "annotation-free thermal detection is new"; "task-driven translation is new".
- **Risk:** ModTr outperforms fine-tuning on FLIR, while our loop only reaches the raw-thermal floor there. Without a head-to-head ModTr run, a reviewer may conclude our generator choice (visible-faithful) costs accuracy. That trade-off — faithfulness vs detector-optimality — can itself be framed as a finding, but only if measured: ModTr's outputs vs ours on the faithfulness metric and on mAP.
- **Risk:** the 753-pair power-line set is small and private. HalluciDet and ModTr used public data with code, so reproducibility expectations are set by them.

### Gaps
- Whether a 2025–2026 *diffusion* or *turbo* IR→RGB detector-loop paper exists is outside this scope (the diffusion researcher covers it). That result determines whether sub-claim 2 survives.
- No systematic "cited by" crawl of HalluciDet and ModTr was possible with the available tools. A manual Google Scholar check before submission is recommended to catch 2026 follow-ups.
