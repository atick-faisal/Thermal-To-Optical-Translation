# Competing (non-translation) paradigms for label-scarce thermal object detection

Scope: baselines a reviewer could say make thermal→visible translation unnecessary. Research date 2026-10-03. Starting point: RESEARCH_FINDINGS.md §8 lists thermal-trained detector, CycleGAN pseudo-RGB, arXiv:2002.06770, Meta-UDA, ModTr, task-conditioned DA (ECCV 2020), fusion (SeAFusion/TarDAL/DetFusion).

Verification legend: numbers marked (verified) were read from the paper's own HTML/abstract page via fetch. Items marked (UNVERIFIED) come from search snippets, secondary summaries, or prior knowledge, and must be checked against the PDF before citing.

**Headline findings, up front**
1. **The label-transfer threat is real, and the literature backs it.** Three independent lines of evidence show that labels transferred across registered pairs train detectors as well as, or better than, manual labels. These are BUDIR (Sensors 2021), Bouzoulas et al. (KAIST, 2025–26) and Cantón & Lazaro (IEEE, 2024). A 2026 CVPR paper (Thermal-Det) does annotation-free thermal detection with a **frozen RGB teacher on paired RGB-T data**. That is exactly the "no translator needed" pipeline.
2. **ModTr (ECCV 2024) is the closest prior art, not just a baseline.** It trains a small IR→RGB translator to minimise a **frozen** RGB detector's detection loss, so the detector never changes. It reports beating full fine-tuning on FLIR. However, it **uses IR ground-truth boxes**. Our novelty against ModTr must rest on where our boxes come from and on the zero-thermal-annotation regime.
3. **UDA (unpaired RGB→thermal) is well developed.** The best number is D3T at 69.3 mAP on FLIR (VGG-16 Faster R-CNN). None of the UDA papers found compare against image translation.
4. **Zero-shot open-vocabulary detectors are not useless on thermal.** Grounding DINO reaches 0.636 AP50 on FLIR-Aligned zero-shot, against 0.749 when fine-tuned. Nothing was found on power-line components.
5. **No paper found directly compares translation with label transfer and cross-modal KD on the same paired benchmark,** and none gives an annotation-budget crossover analysis. This is a genuine gap, and it is also the experiment reviewers will ask for.

---

## Q1. Unsupervised domain adaptation (RGB-labelled source → unlabelled thermal target)

### Takeaway
RGB→thermal UDA for detection is an active line, mostly built on Faster R-CNN and mean-teacher self-training. The state of the art is D3T (CVPR 2024): 69.30 mAP on FLIR and 48.96 on KAIST, against 34.68 and 9.09 source-only. These methods need no pairs and no thermal labels. They do need retraining the detector, and none of them benchmark against translation.

### Cited Findings
- **D3T (CVPR 2024)**, Do et al. Uses dual teachers (an RGB-domain EMA teacher and a thermal-domain EMA teacher) plus "zigzag" learning that shifts epochs from RGB to thermal over training. Code: https://github.com/EdwardDo69/D3T — [arXiv 2403.09359](https://arxiv.org/html/2403.09359); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Phat_D3T_Distinctive_Dual-Domain_Teacher_Zigzagging_Across_RGB-Thermal_Gap_for_Domain-Adaptive_CVPR_2024_paper.pdf)
- D3T FLIR results, VGG-16 Faster R-CNN, mAP over person/bicycle/car (verified). Source-only 34.68; DANN 37.14; SWDA 38.29; EPM 44.60; HT (Harmonious Teacher) 65.81; **D3T 69.30**. Per class for D3T: person 70.77, bicycle 57.44, car 79.68 — [D3T arXiv](https://arxiv.org/html/2403.09359)
- D3T KAIST results, ResNet-50, person (verified). Source-only 9.09; DANN 9.17; SWDA 31.30; EPM 39.55; HT 43.45; **D3T 48.96** — [D3T arXiv](https://arxiv.org/html/2403.09359)
- D3T's protocol deliberately uses **disjoint, non-paired** RGB and thermal images, to prevent "overfitting" on pairs. FLIR uses 2,064 RGB + 2,064 thermal images; KAIST uses 4,446 + 4,446. No image-translation baseline appears in the comparison (verified) — [D3T arXiv](https://arxiv.org/html/2403.09359)
- An ablation snippet says the dual teachers alone reach 66.93 and adding zigzag learning reaches 68.46. This differs from the 69.30 in the main table, possibly because of different runs or settings (UNVERIFIED snippet) — [IEEE page](https://ieeexplore.ieee.org/document/10658394/)
- **Meta-UDA (WACV 2022)**, VS, Poster, You, Hu, Patel. Online meta-learning of the detector's initial condition for unsupervised thermal domain adaptation. Evaluated on KAIST (RGB→thermal) and DSIAC (vehicles). Exact mAP numbers were not retrieved because the CVF PDF returned 403 — [CVF PDF](https://openaccess.thecvf.com/content/WACV2022/papers/VS_Meta-UDA_Unsupervised_Domain_Adaptive_Thermal_Object_Detection_Using_Meta-Learning_WACV_2022_paper.pdf); [arXiv 2110.03143](https://arxiv.org/abs/2110.03143)
- **SSTN (2021)**: self-supervised domain adaptation for thermal object detection in autonomous driving — [arXiv 2103.03150](https://arxiv.org/pdf/2103.03150) (details UNVERIFIED)
- **Unsupervised RGB-to-thermal DA via multi-domain attention network (2022)** — [arXiv 2210.04367](https://arxiv.org/html/2210.04367) (details UNVERIFIED)
- **Domain Adaptive Thermal Object Detection with Unbiased Granularity Alignment** (ACM TOMM, 2024) — [ACM DL](https://dl.acm.org/doi/10.1145/3665892) (numbers UNVERIFIED)
- **MS-DAYOLO / Integrated MS-DAYOLO (TIP 2023)**: multiscale GRL domain classifiers on YOLOv4. It was demonstrated on weather shift (Cityscapes→Foggy, 35.64→43.04 mAP); no thermal results were found. It is relevant only as a YOLO-family DA template — [arXiv 2106.01483](https://arxiv.org/abs/2106.01483); [arXiv 2202.03527](https://arxiv.org/abs/2202.03527); [GitHub](https://github.com/Mazin-Hnewa/MS-DAYOLO)
- Thermal-Det (CVPR 2026) groups SSTN, Meta-UDA and domain-adaptive pedestrian detection as the DA related work for thermal — [arXiv 2605.10130](https://arxiv.org/html/2605.10130v1)

### Inferences
- D3T's "target-only/oracle" row was not retrieved, so the remaining gap to supervised training on FLIR is unknown from this source. HT and D3T reaching ~66–69 mAP suggests that unpaired UDA on FLIR is already far above source-only.
- UDA is a weaker threat to us than label transfer. It cannot exploit our pairs, it retrains the detector (which violates the frozen-detector premise), and it has never been shown on power lines. But it is the standard family reviewers in detection DA expect. **D3T (code public) or HT is the representative to include.**
- D3T avoids pairs to prevent "overfitting". A reviewer may turn this around and argue that methods using pairs get an unfair advantage. Report UDA with and without access to pairs.
- CMT (Contrastive Mean Teacher) and DA-Faster variants appear as generic DA baselines. No thermal results for CMT were found in this pass (UNVERIFIED that CMT reports FLIR/KAIST).

### Gaps
- Meta-UDA's exact KAIST/DSIAC numbers (403 on the CVF PDF).
- Oracle (thermal-supervised) numbers under the D3T protocol.
- No RGB→thermal UDA work on power-line or industrial components was found.
- "Revisiting domain adaptation for IR" as a named paper could not be located. It may not exist under that title.

---

## Q2. Cross-modal pseudo-labelling / label transfer, and RGB-teacher → thermal-student distillation on paired data

### Takeaway
This is the **most dangerous competitor**. Label transfer across registered pairs is old (2019), simple and empirically strong. Auto-labelled thermal training has **matched or beaten manual labels** in at least three studies. CVPR 2026's Thermal-Det makes "frozen RGB teacher + paired RGB-T → annotation-free thermal detector" a top-venue method. **No study found compares label transfer against translation on the same data.**

### Cited Findings
- **Thermal-Det (CVPR 2026)**, Ranasinghe et al. Billed as "the first zero-shot open-vocabulary detection framework for thermal imagery... without any thermal annotations" (verified):
  - A frozen RGB detector (teacher) runs on paired RGB-T frames and produces boxes, class logits and object–text similarity scores. A thermal student is distilled with three losses: GIoU box, InfoNCE semantic, and KL confidence.
  - Paired data: ~250K pairs from MMPD, Multi-Spectral Stereo and M3FD.
  - It also pretrains on GroundingCap-1M converted to synthetic thermal by F-ViTA.
  - Source: [arXiv 2605.10130](https://arxiv.org/html/2605.10130v1); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ranasinghe_Thermal-Det_Language-Guided_Cross-Modal_Distillation_for_Open-Vocabulary_Thermal_Object_Detection_CVPR_2026_paper.pdf)
- Thermal-Det zero-shot results (verified, Swin-T backbone for all methods):
  - FLIR-Aligned: **Thermal-Det 0.372 AP / 0.664 AP50**, against G-DINO 0.337/0.636 and YOLO-World 0.266/0.487.
  - G-DINO **fine-tuned** (supervised) on FLIR-Aligned reaches 0.414/0.749.
  - Source: [arXiv 2605.10130](https://arxiv.org/html/2605.10130v1)
- Thermal-Det ablation on FLIR-Aligned: zero-shot baseline 0.200 AP; +scene caption +0.012; +object caption +0.028; **+knowledge distillation +0.021**; final 0.261. This ablation table's final 0.261 differs from the 0.372 in the main table, likely because of different training scale or settings (verified as reported; the discrepancy is unexplained in the summary). The paper has **no thermal→RGB translation baseline** — [arXiv 2605.10130](https://arxiv.org/html/2605.10130v1)
- **BUDIR**, Ligocki, Jelinek, Zalud, Rahtu (*Sensors*, 2021). RGB-pretrained detectors annotate RGB frames, and the boxes are transferred to thermal through LiDAR-based 3D frustum reprojection. This produced ~300K thermal images with 1.7M boxes and **zero manual labels** (verified):
  - A detector trained on the auto-labelled BUDIR scored **mAP50 0.544**, against **0.523** for one trained on manually labelled FLIR. mAP50:95 was 0.254 vs 0.213.
  - Fine-tuning with **150 human-annotated images** raised it to **0.661 / 0.336**.
  - Failure mode: transferred boxes "slightly underfitting the real dimension".
  - Source: [PMC7926581](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/)
- **Automatic Labelling for Low-Light Pedestrian Detection**, Bouzoulas, Alamikkotervo, Ojala (arXiv 2507.02513, v5 July 2026). Transfers labels across KAIST pairs in the IR→RGB direction. Models trained on auto labels **beat models trained on ground-truth labels in 5 of 6 cases** for mAP50/LAMR and in all cases for mAP50:95, across DETR, YOLO and R-CNN (verified, abstract level) — [arXiv 2507.02513](https://arxiv.org/abs/2507.02513)
- **Automatic Labeling for Thermal Imaging Datasets Generation**, Cantón & Lazaro (IEEE, Nov 2024). YOLO is run on the RGB half of homography-aligned RGB-IR pairs and the labels are transferred to IR to train thermal models, evaluated against manually annotated sets. Numbers UNVERIFIED (only an abstract snippet was seen) — [IEEE Xplore 10797435](https://ieeexplore.ieee.org/document/10797435/); [ResearchGate](https://www.researchgate.net/publication/387348670_Automatic_Labeling_for_Thermal_Imaging_Datasets_Generation)
- **Guan et al., UDA for Multispectral Pedestrian Detection (CVPRW 2019, MULA)**. Iteratively generates pseudo-annotations, exploiting "similarity and complementarity between well-aligned visible and infrared image pairs". It claims to be "competitive with supervised multispectral pedestrian detectors". The target is multispectral (two-input), not thermal-only (verified, abstract) — [arXiv 1904.03692](https://arxiv.org/abs/1904.03692); [CVF PDF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/MULA/Guan_Unsupervised_Domain_Adaptation_for_Multispectral_Pedestrian_Detection_CVPRW_2019_paper.pdf)
- Related multispectral pedestrian UDA with illumination-aware label fusion from both modalities (Sensors 2022) — [MDPI](https://www.mdpi.com/1424-8220/22/12/4416) (details UNVERIFIED)
- **FreqKD (arXiv 2606.11572, 2026)**. Frequency-decoupled RGB-teacher → IR-student feature distillation on paired data. Stage 1 is label-free distillation; **stage 2 fine-tunes with IR ground truth**. DINO-DETR mAP50: IR baseline 61.7 → **64.1** with FreqKD; the RGB teacher scores 68.0 as a reference. Naive response-level KD *hurt* (58.8) (verified) — [arXiv 2606.11572](https://arxiv.org/html/2606.11572)
- **Contrast-Guided Cross-Modal Distillation for Thermal Object Detection**, Kim & An (arXiv 2511.01435, Nov 2025). An RGB-trained teacher's semantics are injected into thermal-student FPN features with contrastive objectives, as training-only losses. Whether thermal labels are needed was not stated in the abstract (UNVERIFIED) — [arXiv 2511.01435](https://arxiv.org/abs/2511.01435)
- **Learning Cross-Modal Deep Representations for Robust Pedestrian Detection** (Xu et al., CVPR 2017). Uses paired RGB-thermal data at training time to learn thermal-like features for RGB-only testing. This is the reverse direction, an early use of pairs as privileged information — [arXiv 1704.02431](https://arxiv.org/pdf/1704.02431)

### Inferences
- **Direct implication for our paper.** We train the translator with a detection loss on paired data, so we must already have boxes for the paired training images. These come from visible ground truth, or from the visible detector's predictions on the visible half. **Exactly those boxes can train YOLO11 directly on the 753 thermal images.**
  - Our own budget curve shows a thermal-trained YOLO11 reaches 0.93 with 600 manual labels and matches our method at ~150–214.
  - BUDIR and Bouzoulas show transferred labels perform about as well as manual ones.
  - So label transfer on 753 pairs would plausibly land near 0.9 mAP50 on in-domain test data, **above** our 0.85. This is an inference, not a measurement. **Run it before submission.**
- Where label transfer is structurally weaker than translation (candidate defensible conditions):
  1. **Frozen detector requirement.** Label transfer produces a *new* thermal detector. It cannot reuse a certified, vendor-locked or API-served visible detector, and it must be retrained whenever the visible detector or its class set changes. A translator trained once serves any future visible detector. That last claim holds only if we show cross-detector transfer (our E7).
  2. **Classes absent from the paired set.** A translator is in principle class-agnostic: a visible detector for class X could work on translated thermal even if X never appeared in the pairs. Label transfer only yields labels for classes present in the pairs. This is testable, and it is likely our strongest framing if it holds.
  3. **Pixel-level outputs for humans.** Translated images are inspectable by operators. This matters for an Instrumentation & Measurement audience, but it is a weak argument for detection accuracy.
  4. **Thermal-only content.** Here label transfer *and* translation both fail where the visible half is dark or occluded (night, smoke). Neither has an advantage, unless the translator generalises better than the pseudo-labels.
- Deployment domain shift (paired training set ≠ deployment footage) hurts both approaches. It does not by itself favour translation, unless an experiment shows translators generalise better than thermal detectors trained on transferred labels. **Treat this as an empirical question, not an argument.**
- Thermal-Det's two-mechanism design (synthetic thermal + paired KD) is the 2026 top-venue answer to "annotation-free thermal detection". A reviewer at Information Fusion is likely to know it.

### Gaps
- No paper found compares (a) label transfer, (b) cross-modal KD and (c) thermal→RGB translation for a frozen detector on the same pairs. This is the key missing comparison and a publishable experiment in itself.
- Cantón & Lazaro's quantitative results were not retrieved.
- No study found on label-transfer quality for thin structures (power lines, insulators), where small registration errors could matter more than for pedestrians or cars.

---

## Q3. Zero-shot / open-vocabulary / foundation models on thermal

### Takeaway
Open-vocabulary detectors transfer surprisingly well to *some* thermal benchmarks. Grounding DINO gets 0.636 AP50 zero-shot on FLIR-Aligned, against 0.749 fine-tuned. They collapse on others (FLIR-V2: 0.144 AP50). Thermal-specific foundation pretraining exists (InfMAE, UNIP), but it targets fine-tuning, not zero-shot detection. **No zero-shot results on power-line or insulator thermal imagery were found.**

### Cited Findings
- Zero-shot AP / AP50, Swin-T backbone (verified):
  - **FLIR-Aligned**: GLIP 0.251/0.471; T-Rex2 0.276/0.514; YOLO-World 0.266/0.487; **G-DINO 0.337/0.636**; MM-GDINO 0.354/0.619; LLMDet 0.359/0.628; Thermal-Det 0.372/0.664.
  - **FLIR-V2**: G-DINO 0.081/0.144; YOLO-World 0.029/0.044; Thermal-Det 0.096/0.173.
  - **CAMEL**: G-DINO 0.482/0.729; YOLO-World 0.197/0.336; Thermal-Det 0.511/0.758.
  - Source: [Thermal-Det arXiv 2605.10130](https://arxiv.org/html/2605.10130v1)
- Supervised fine-tuning upper bound on FLIR-Aligned: G-DINO fine-tuned 0.414 AP / 0.749 AP50 (verified) — [Thermal-Det](https://arxiv.org/html/2605.10130v1)
- Thermal-Det also evaluates SMOD, Utokyo, MFAD and LLVIP, with reported 2–4 AP gains over RGB open-vocabulary detectors (per-dataset numbers not retrieved) — [Thermal-Det](https://arxiv.org/html/2605.10130v1)
- **Open-vocabulary detector robustness under distribution shift** (2024). Evaluates OWL-ViT, YOLO-World and Grounding DINO under covariate shifts. Possibly includes modality shift (UNVERIFIED) — [arXiv 2405.14874](https://arxiv.org/html/2405.14874v3)
- **InfMAE (2024)**: an infrared foundation model with the Inf30 dataset and information-aware masking. It reports state of the art after fine-tuning on segmentation, detection and small-target detection — [arXiv 2402.00407](https://arxiv.org/pdf/2402.00407)
- **UNIP (ICLR 2025)**: unified infrared pretraining via hybrid-attention distillation on InfMix, 859,375 infrared images from 25 datasets. Up to +13.5 mIoU on infrared segmentation; evaluated by fine-tuning, not zero-shot detection — [arXiv 2502.02257](https://arxiv.org/pdf/2502.02257); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/d3ee2816ae19c9e689c3352397c93a22-Paper-Conference.pdf)
- **DuGI-MAE (2025)**: infrared MAE with dual-domain guidance — [arXiv 2512.04511](https://arxiv.org/html/2512.04511) (details UNVERIFIED)
- **ThermEval (2026)**: a benchmark of vision-language models on thermal imagery (VQA-style, not detection) — [arXiv 2602.14989](https://arxiv.org/pdf/2602.14989) (details UNVERIFIED)
- SAM is reported to be sensitive to photometric change, with performance falling from colour jitter to grayscale to thermal. Other work uses SAM zero-shot on thermal for masks. This is search-snippet level only (UNVERIFIED; no primary source pinned) — [search context; see Caltech Aerial RGB-T dataset arXiv 2403.08997](https://arxiv.org/pdf/2403.08997)
- High-voltage equipment defect detection with vision large models has been reviewed in *High Voltage* (IET, 2025/26). Whether it covers thermal zero-shot detection was not verified — [Wiley](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/hve2.70228?af=R)

### Inferences
- A reviewer will ask: "Does Grounding DINO / YOLO-World on raw thermal already beat 0.19?" This is cheap to run and **must be included**: Grounding DINO with prompts such as "insulator", "power line" and "tower" on raw thermal power-line test images.
  - On FLIR-Aligned, raw-thermal G-DINO reaches 85% of its fine-tuned AP50 (0.636/0.749).
  - Power-line components are out-of-distribution for its training text and images, so zero-shot performance is likely much lower. This is an inference and needs measuring.
- Translation and open-vocabulary detection compose. Running G-DINO on our *translated* images is a natural extra row. If translation also lifts zero-shot open-vocabulary detectors, that strengthens the "reuse any visible detector" framing.
- Thermal foundation backbones (InfMAE, UNIP) lower the number of labels a thermal detector needs. They strengthen the "just annotate ~150 thermal images" counter-argument rather than replacing translation.

### Gaps
- No zero-shot open-vocabulary results on thermal power-line, insulator or transmission-tower imagery were found.
- Per-dataset Thermal-Det numbers for LLVIP and M3FD-like sets were not retrieved.
- No large thermal pretraining work evaluated in a strictly zero-shot detection setting was found, other than Thermal-Det.

---

## Q4. Few-shot / semi-supervised thermal detection, and synthetic thermal (RGB→thermal generation) as a label source

### Takeaway
The *reverse* direction (RGB→thermal generation to make labelled synthetic thermal data) is booming in 2025–26: F-ViTA, ThermalGen, TherA, and ControlNet/SD3.5 pipelines. It is a direct competitor, because it reuses large *public labelled visible datasets* without any thermal labels. Few-shot thermal detection exists but is thin (FSMODNet, 2025).

### Cited Findings
- **RGB-to-IR translation for infrared vehicle detection in unseen UAV domains**, Eker, Fokkinga, van Woerden et al. (arXiv 2609.02556, SPIE 2026). Trained on paired RGB-IR source data, then generates synthetic IR from labelled RGB. SD3.5 + ControlNet worked best: mAP **50.8 → 60.1 on Kust4K** and **25.6 → 38.4 on VTUAV** over source-only baselines. Multiple seeds add +1.1 and prompt variation +3.3. It compares against no pseudo-labelling or UDA baselines (verified, abstract) — [arXiv 2609.02556](https://arxiv.org/abs/2609.02556)
- **F-ViTA (2025)**: foundation-model-guided visible→thermal translation. Used by Thermal-Det to convert GroundingCap-1M (>1M RGB images with boxes) into synthetic thermal. Replacing up to 50% of real thermal data with F-ViTA synthetic data roughly preserves segmentation performance (snippet-level, UNVERIFIED) — [arXiv 2504.02801](https://arxiv.org/pdf/2504.02801)
- **ThermalGen**: described as the first RGB-T translation model that synthesises thermal images across large viewpoint, sensor and environment variation, with a detection mAP figure of 23.9% in a snippet (context UNVERIFIED) — cited in [Thermal-Det](https://arxiv.org/html/2605.10130)
- **TherA (2026)**: thermal-aware visual-language prompting for controllable RGB→thermal translation — [arXiv 2602.19430](https://arxiv.org/html/2602.19430v1) (details UNVERIFIED)
- **Training with synthetic data for drone detection in thermal imagery** (2026) — [arXiv 2608.17799](https://arxiv.org/pdf/2608.17799) (details UNVERIFIED)
- **Synthetic thermal image generation for real-time animal detection** (2026) — [arXiv 2609.32944](https://arxiv.org/html/2609.32944) (details UNVERIFIED)
- **FSMODNet (2025)**: a reproducible few-shot multispectral detection baseline on FLIR and M3FD — [arXiv 2509.20905](https://arxiv.org/pdf/2509.20905) (details UNVERIFIED)
- **Borrow from Anywhere** (Devaguptapu et al., CVPRW/PBVS 2019). Thermal→pseudo-RGB via unpaired I2I translation, then pseudo multi-modal fusion in the detector. Claims to "learn with less data from thermal domain". This is a translation-lineage baseline, not a competitor — [arXiv 1905.08789](https://arxiv.org/abs/1905.08789); [CVF PDF](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Devaguptapu_Borrow_From_Anywhere_Pseudo_Multi-Modal_Object_Detection_in_Thermal_Imagery_CVPRW_2019_paper.pdf)
- BUDIR shows auto-labels plus **150 manual thermal labels** gives a large jump (0.544 → 0.661 mAP50). A few manual labels on top of transferred labels is a strong hybrid — [PMC7926581](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/)

### Inferences
- Synthetic-thermal generation is the mirror image of our method. It moves *data* into the thermal domain so a thermal detector can be trained from public visible labels, whereas we move *images* into the visible domain so a frozen visible detector can be reused. For power lines, a reviewer could suggest:
  - training an RGB→thermal generator on our 753 pairs;
  - pushing public labelled visible power-line datasets (e.g. insulator datasets) through it;
  - training a thermal YOLO11 on the result.
  This is a credible baseline if public visible power-line labels exist. Coordinate with the data-layer notes.
- Semi-supervised thermal detection (mean-teacher with a few labels plus unlabelled thermal) was not specifically found. In practice it overlaps with D3T/HT-style self-training.

### Gaps
- Few-shot thermal-only (not multispectral) detection numbers were not found.
- No semi-supervised thermal detection paper with annotation-budget curves was found.
- ThermalGen's primary URL and its detection numbers were not verified.

---

## Q5. Direct comparisons of translation against these paradigms; "when does translation pay off" analyses

### Takeaway
The only head-to-head found is inside ModTr, which compares a detection-loss-trained translator with full fine-tuning and with CycleGAN/FastCUT. ModTr wins on FLIR and ties on LLVIP, **using full IR labels**. No paper compares translation with label transfer, cross-modal KD, UDA or open-vocabulary detection on the same benchmark. No annotation-budget crossover analysis was found.

### Cited Findings
- **ModTr (ECCV 2024)**, Medeiros, Aminbeidokhti, Guerrero Peña, Latortue, Granger, Pedersoli. A small transformation network turns IR images into RGB-like inputs and is trained to minimise the detection loss of a **frozen** RGB detector (Faster R-CNN, RetinaNet, FCOS). It motivates a "service-based pipeline": one unaltered RGB detector server queried by several modalities, each through its own translator. Sources: [arXiv 2404.01492](https://arxiv.org/abs/2404.01492); [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12401.pdf); [project page](https://heitorrapela.github.io/ModTr/); [code](https://github.com/heitorrapela/ModTr)
  - It **requires IR ground-truth boxes** for the detection loss (verified via HTML).
  - LLVIP AP: fine-tuned 57.37 / 53.79 / 59.62 vs **ModTr 57.63 / 54.83 / 57.97** (FCOS / RetinaNet / Faster R-CNN).
  - FLIR-Aligned AP: fine-tuned 27.97 / 28.46 / 30.93 vs **ModTr 35.49 / 34.27 / 37.21**.
  - Zero-shot RGB detectors on IR scored roughly 32–38 (LLVIP) and 23–25 (FLIR). These are approximate values reported by the fetch summariser and must be checked against the PDF tables.
  - It reports outperforming CycleGAN and FastCUT translation baselines (verified, numbers not extracted).
- D3T compares no translation baselines (verified) — [D3T](https://arxiv.org/html/2403.09359). Thermal-Det compares no thermal→RGB translation (verified) — [Thermal-Det](https://arxiv.org/html/2605.10130v1)
- Eker et al. compare synthetic-IR generators against source-only training, but not against UDA or pseudo-labelling (verified) — [arXiv 2609.02556](https://arxiv.org/abs/2609.02556)

### Inferences
- **ModTr directly undercuts a "first to put a frozen detector's loss in the translation loop" claim.** Our differences must be stated precisely:
  - (a) zero thermal annotations: boxes from the visible half of pairs, versus ModTr's IR ground truth;
  - (b) a full image-to-image generator (GAN/diffusion) rather than a lightweight input adapter;
  - (c) an industrial power-line domain and YOLO11;
  - (d) the annotation-budget analysis.
  If our loop actually uses visible-half labels, then ModTr trained on transferred labels is *literally* our setting with a different translator. **Include ModTr as a baseline.** Code is public.
- ModTr's FLIR result (translator beats full fine-tuning) contrasts with our finding that translate-then-fine-tune was worse than thermal-only at every budget. The settings differ: ModTr compares against fine-tuning the RGB detector on IR, not against a translated-then-fine-tuned detector. Reconcile this explicitly in the paper.
- Our annotation-budget curve (a thermal-supervised detector matches us at ~150–214 labels) is itself novel analysis. To be honest it needs a **label-transfer curve** next to it: label transfer gives N noisy labels "for free" from N pairs.

### Gaps
- No study found asking "when does translation pay off versus label transfer or KD".
- ModTr's CycleGAN/FastCUT numbers and exact zero-shot numbers still need extracting from the PDF.
- A journal extension of ModTr (e.g. in TPAMI/TIP) was not found. Worth a check because the same group (ÉTS Montréal, Granger/Pedersoli) is active.

---

## Q6. Bottom line: minimal baseline set, and which framing survives

### Takeaway
"Annotation-free reuse of visible-trained detectors on thermal-only imagery" survives only if it is narrowed. The defensible version is: *a frozen visible detector, unchanged and unretrained, works on thermal-only deployment imagery with no thermal labels, using a translator trained once on pairs*. It must then be benchmarked against **label transfer on the same pairs** and against **ModTr**. If label transfer beats us in-domain, which is likely, the contribution must shift to the frozen-detector setting and to properties label transfer lacks. Those properties are: detector-agnostic reuse, classes not present in the pairs, and robustness to deployment shift. Each must be demonstrated empirically, not asserted.

### Cited Findings (supporting the baseline list)
- Label transfer and auto-labelling match or beat manual labels: [BUDIR, PMC7926581](https://pmc.ncbi.nlm.nih.gov/articles/PMC7926581/); [Bouzoulas et al., arXiv 2507.02513](https://arxiv.org/abs/2507.02513); [Cantón & Lazaro, IEEE 10797435](https://ieeexplore.ieee.org/document/10797435/)
- Paired frozen-teacher KD as an annotation-free top-venue method: [Thermal-Det, CVPR 2026](https://arxiv.org/html/2605.10130v1)
- Frozen-detector translator with detection loss (closest prior art): [ModTr, ECCV 2024](https://arxiv.org/abs/2404.01492)
- Strongest unpaired UDA: [D3T, CVPR 2024](https://arxiv.org/html/2403.09359) (FLIR 69.30, KAIST 48.96)
- Zero-shot open-vocabulary detection on thermal is non-trivial (G-DINO 0.636 AP50 on FLIR-Aligned): [Thermal-Det](https://arxiv.org/html/2605.10130v1)
- Synthetic-thermal generation from labelled RGB as a label source: [Eker et al., arXiv 2609.02556](https://arxiv.org/abs/2609.02556); [F-ViTA, arXiv 2504.02801](https://arxiv.org/pdf/2504.02801)

### Inferences — minimal baseline set (priority order)
1. **Cross-modal label transfer (must).** Run YOLO11 (the same visible detector) on the visible half of the 753 pairs, transfer the boxes to thermal, and train YOLO11 on thermal with zero manual labels. If visible ground truth exists, also report the GT-transfer variant. Evaluate in-domain and on a held-out deployment-shift split. This is the reviewer's first question.
2. **ModTr (must)**, trained on the same transferred boxes, so the only difference is the translator and loop design. Also report it with ground-truth thermal boxes if available, for parity with the original paper.
3. **Thermal-supervised YOLO11 budget curve** (already have it: ~150–214 to match, 0.93 at 600). Add label-transfer + k manual labels as a hybrid curve, the BUDIR-style +150.
4. **Zero-shot open-vocabulary (must, cheap).** Grounding DINO (and/or YOLO-World) on raw thermal **and** on translated thermal.
5. **Raw thermal → frozen visible detector** (already have it: 0.19) and **translation without the loop** (have 0.80). Include a CycleGAN/pix2pix lineage row as the §8 list requires.
6. **One UDA method (should).** D3T (public code) or HT, using the visible images as the labelled source and thermal as the unlabelled target, on public FLIR/KAIST at least. Note that UDA retrains the detector.
7. **Cross-modal KD (nice-to-have).** A response- or feature-level RGB-teacher → thermal-student baseline on the same pairs, Thermal-Det-style, if time allows.
8. **Fusion (SeAFusion/TarDAL/DetFusion)**: argue it out of scope, because deployment is thermal-only and fusion needs a visible input at test time. Cite briefly rather than run.

### Inferences — which framing survives against each competitor
- **Against label transfer.** Survives only under: (i) the detector must stay frozen (certified, vendor or API, or shared across modalities as in ModTr's service argument); (ii) detector-agnostic reuse, where the translator works with a *different* visible detector than the one in the loop (needs our E7 result); (iii) classes or objects not annotated in the pairs. Without at least one of these demonstrated, "annotation-free" is not a distinguishing property, because label transfer is also annotation-free.
- **Against ModTr.** Survives on zero thermal annotations (ModTr needs IR ground truth), on the generator class, on the industrial domain, and on the budget analysis. It does **not** survive as "first detector-in-the-loop translation for frozen detectors".
- **Against UDA.** Survives easily on the frozen-detector premise and on thermal-only deployment. Concede that UDA needs no pairs.
- **Against open-vocabulary zero-shot.** Survives if G-DINO on raw power-line thermal is far below 0.85, and is strengthened if translation also lifts G-DINO.
- **Against synthetic thermal generation.** Survives if no large labelled visible power-line dataset exists, or if a translator generalises better. Otherwise it is a peer method worth acknowledging.
- **Against "just label 150 thermal images".** Our own data concedes this is cheap. The framing must be about *zero*-label reuse of existing detectors and deployment logistics, not about beating supervised thermal training.
- Deployment conditions where the framing is strongest:
  - a frozen or unretrainable detector;
  - thermal-only test-time sensing;
  - a detector set that changes or expands over time;
  - large public visible datasets and detectors that outnumber any thermal resource.
  Deployment shift between the pairs and the field footage is **not** automatically in translation's favour. It must be measured for both translation and label transfer.

### Gaps
- Whether visible-half pseudo-labels on our power-line pairs are accurate (thin conductors, registration error) is unknown. It determines how strong baseline 1 will be.
- No evidence was found either way on whether translators generalise better than transferred-label thermal detectors under deployment shift.
- No primary source was verified for zero-shot foundation-model performance on industrial or power-line thermal imagery.
