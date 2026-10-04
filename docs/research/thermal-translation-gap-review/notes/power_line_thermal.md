# Deep-learning thermal/IR inspection of power lines and electrical equipment — cross-modal methods and datasets (2018–2026)

Scope note: searched 2026-10-03 via web search, Crossref, OpenAlex, arXiv, Mendeley, Zenodo, Hugging Face. MDPI, IEEE Xplore, ResearchGate, figshare and SciELO full texts were mostly blocked (403/JS), so several claims rest on abstracts (Crossref/OpenAlex) or search snippets — flagged inline as **[abstract only]** or **[snippet only]**.

## Q1. Is there a public paired/registered thermal–visible power dataset with bounding boxes?

### Takeaway
No public dataset found that provides **registered thermal–visible pairs of power-line components with bounding-box labels**. But the proposal's wording is wrong in two ways: (a) **private** paired IR–VIS power datasets are common (at least 5 groups report 400–5,938 registered pairs), so "near-unique" must be narrowed to "no *public* equivalent at component/box level"; (b) Yetgin & Gerek is **not** the only public-ish paired power resource — **VITLD** (Choi et al., IEEE TII 2022) is a genuinely matched RGB–thermal transmission-line set (conductor masks, 256×256), used as a benchmark by several later papers, though its availability is "on request".

### Cited Findings
**Public or semi-public power datasets that contain thermal imagery**
- **Yetgin & Gerek "Powerline Image Dataset (IR and VL)"**, Mendeley Data, 2019, DOI 10.17632/n6wrv4ry6v.8, MIT licence: 4,000 IR + 4,000 VL images scaled to 128×128 (2,000 with / 2,000 without lines per modality); frames from TEIAS aircraft video, originals 576×325 (IR) and full HD (VL), 21 regions of Turkey; labels are image-level only (TV = line, TY = no line). The page does not state that IR and VL images are paired captures of the same scene. — [Mendeley](https://data.mendeley.com/datasets/n6wrv4ry6v/8)
- Companion **"Ground Truth of Power Line Dataset (IR and VL)"**, CC BY 4.0: 200 IR + 200 VL images at 512×512 with line ground-truth masks; pairing again not stated. — [Mendeley](https://data.mendeley.com/datasets/twxp8xccsw/6). There is also an "Extra … Classified (Easy and Hard)" version. — [Mendeley](https://data.mendeley.com/datasets/4zkszx3398/2)
  - **Correction for RESEARCH_FINDINGS §5:** the proposal says "400 IR + 400 VL, wire masks". The verified numbers are 4,000 + 4,000 (128×128, classification) and 200 + 200 (512×512, masks). The "400" probably conflates the GT subset or VITLD.
  - **Conflict:** Aboalia et al. (AJSE 2024) describe this dataset as "paired infrared–visible power line datasets" and fuse the two streams (99.37% accuracy, line/no-line classification). — [Springer](https://link.springer.com/article/10.1007/s13369-024-09043-0). Mendeley does not confirm the pairing. Treat it as "weakly paired at best, not registered".
- **VITLD (Visible and Infrared Transmission Line Detection)**, Choi, Yun, Kim, Jang, Kim, "Attention-Based Multimodal Image Feature Fusion Module for Transmission Line Detection", IEEE TII 18(11), 2022, DOI 10.1109/TII.2022.3147833 (94 citations per OpenAlex). Abstract: a real-world dataset "constructed by visible light and infrared images", tested under augmented day/night/fog/snow. — [OpenAlex record](https://api.openalex.org/works/https://doi.org/10.1109/tii.2022.3147833)
  - Per Zhang et al. (arXiv 2501.15099): 400 matched RGB–IR pairs from a DJI Phantom; RGB 1920×1080, IR 640×512; originally misaligned, registered with MATLAB, cropped/resized to 256×256; **binary transmission-line masks** (not boxes); "data will be made available on request". — [arXiv HTML](https://arxiv.org/html/2501.15099)
  - Other sources give 420 matched pairs at 256×256 **[snippet only]**. — [ESWA 2025, Guo/Zhou/Liu](https://www.sciencedirect.com/science/article/abs/pii/S0957417424022735)
  - Used as a benchmark by: MAINet (Guo, Zhou, Liu, ESWA 2025, DOI 10.1016/j.eswa.2024.125406) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417424022735); Zhou, Wang, Qian, IEEE TCSI 2025, DOI 10.1109/TCSI.2024.3521933 — [IEEE](https://ieeexplore.ieee.org/document/10820014/); HMMEN (Zhang et al., arXiv 2501.15099) — [arXiv](https://arxiv.org/abs/2501.15099).
- **Gomes et al. 2025 (Brazil)**, "Infrared Detection of Substation Objects Using the Visible Light Spectrum", Brazilian Archives of Biology and Technology, DOI 10.1590/1678-4324-2025240843. They use **stereo visible+IR cameras**, run a visible-trained detector on the visible image, and **transfer the boxes to the rectified paired IR image**. They argue that annotating IR "requires specialists … harder and more expensive" and that visible detectors "require modifications" for IR. — [Crossref abstract](https://api.crossref.org/works/10.1590/1678-4324-2025240843); [ResearchGate](https://www.researchgate.net/publication/390425770_Infrared_Detection_of_Substation_Objects_Using_the_Visible_Light_Spectrum)
  - The same Brazilian project released a **visible-only** 15-class substation dataset: 7,539 images and 213,566 YOLO boxes from one distribution substation, captured partly with stereo FLIR cameras and AGVs, curated for an IR-thermography project. — [figshare](https://figshare.com/articles/dataset/A_YOLO_Annotated_15-class_Ground_Truth_Dataset_for_Substation_Equipment/24060960) **[snippet only]**. A semantic-segmentation version has 1,660 images and 15 classes. — [Data 2023, DOI 10.3390/data8070118](https://doi.org/10.3390/data8070118). The Hugging Face mirror (Apache-2.0) shows visible images only. — [HF](https://huggingface.co/datasets/AndrzejDD/15-class-Substation-Equipment)
  - Whether the paired IR frames were released could NOT be verified. This is the closest thing found to "public substation IR+VIS with boxes", and it is substation equipment, not line components.
- **Thermal-only power datasets (public):**
  - Zenodo "Insulator Infrared Rotated Object Detection Dataset" (Jia, Peng; Changsha Univ. of Sci. & Tech.), 2026-06-26, 1,737 IR images, YOLO-OBB labels, CC-BY-4.0, no visible. — [Zenodo](https://zenodo.org/records/20922151)
  - Zenodo "Rotating infrared insulator", 2026-04, IR transmission-line insulators with rotated boxes **[snippet only]**. — [Zenodo](https://zenodo.org/records/19485450)
  - Kaggle "Infrared Thermal Image Dataset" of HV equipment (transformers, breakers, insulators) **[snippet only; size and labels unverified]**. — [Kaggle](https://www.kaggle.com/datasets/s3programmerlead/infrared-thermal-image-dataset/data)
  - ScienceDB "Infrared Thermal Image Dataset of High Voltage Electrical Power Equipment under Different Operating Conditions" (details not retrievable). — [scidb.cn](https://www.scidb.cn/en/detail?dataSetId=e416c488169f484485ad7575dcfc43ce)

**Private paired IR–VIS power datasets (shows "near-unique" overstates the case)**
- **VISED** (Visible-Infrared Substation Equipment Dataset): 500 synchronously registered visible/IR pairs from multiple industrial sites and labs. Zhang, Kuang, Teng, Xiang, Li, Zhou, *Processes* 13(9):2720, 2025, DOI 10.3390/pr13092720; CBAM-YOLOv4 feature-level fusion, +1.61% mAP, 40.8 FPS on Jetson Nano. — [Crossref/DOI](https://doi.org/10.3390/pr13092720) **[abstract + snippet only]**. **Conflict:** the abstract says "validation on public datasets", while the snippets describe VISED as newly built. Public availability is unverified.
- Hsieh & Yu, *Nondestructive Testing and Evaluation* 2026, DOI 10.1080/10589759.2026.2732240: "registered visible–infrared substation inspection dataset containing 500 image pairs" (possibly VISED); RA-TPAF-YOLO reaches 88.42% mAP@0.5 and 281.7 FPS. — [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1080/10589759.2026.2732240) **[abstract only]**
- Yang et al., *Sensors* 25(9):2858, 2025: 1,000 paired IR/visible images of sensing insulators, 5,163 labels, proprietary; improved-SIFT registration, Dual-ResNet50 fusion, then YOLOv5. Fused 0.865 mAP vs IR 0.836 vs visible 0.813. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12074302/)
- MSIS, Shu, He, Li, *Comput. Intell. Neurosci.* 2022: 5,938 paired IR/visible images (2,940 arresters + 2,998 current transformers), Fluke Ti480 PRO, modified-SIFT registration, instance masks, "available on request". MSIS 40.06 AP, +7.5 pp over IR-only SOLOv2. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8752308/)
- Power-specific IR–VIS **registration** has its own literature, implying many private paired collections: Jiang et al., CAO-C2F, IEEE TPWRD 36(4), 2021, DOI 10.1109/TPWRD.2020.3011962 ("self-established images dataset", 107 citations) — [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1109/tpwrd.2020.3011962); two-stage substation thermal–visible registration, *Appl. Sci.* 2024 — [DOI](https://doi.org/10.3390/app14031158); gradient-distribution IR–VIS alignment in electric power scenes, *J. Imaging* 2025 — [DOI](https://doi.org/10.3390/jimaging11010023).

**Verification of the proposal's single-modality claims**
- **HIT-UAV** is thermal-only *and not power-related*: 2,898 IR images, classes Person/Car/Bicycle/OtherVehicle/DontCare, 60–130 m altitude. — [Sci. Data 2023](https://www.nature.com/articles/s41597-023-02066-6). Fine as an "aerial thermal proximity" set, but it should not be described as power-line data.
- **CPLID** (Tao et al., IEEE TSMC 2018, DOI 10.1109/TSMC.2018.2871750) is aerial visible imagery; its defect scarcity was handled by augmentation (affine transforms, insulator segmentation and background fusion…). — [OpenAlex abstract](https://api.openalex.org/works/https://doi.org/10.1109/tsmc.2018.2871750). Visible-only is confirmed by the abstract.
- **TTPLA** (arXiv 2010.10032) and **InsPLAD** (arXiv 2311.01619): visible-only. These were not re-verified this session; known from prior literature. — [TTPLA](https://arxiv.org/abs/2010.10032); [InsPLAD](https://arxiv.org/abs/2311.01619)
- The 2025 power-line DL review (Applied Energy) lists the Yetgin & Gerek set as the **only** IR dataset among the public power-line datasets it catalogues, and says utility datasets are "not available to the public due to privacy and data protection laws". — [arXiv 2502.07826](https://arxiv.org/html/2502.07826v1); [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306261925002375)
- A 2026 PRISMA review (148 studies, 2018–2026) reports "persistent gaps in dataset availability, domain generalization". — [Sensors 26(15):4819](https://doi.org/10.3390/s26154819) **[abstract only]**

### Inferences
- A safe claim for the paper: "To our knowledge, no public dataset provides registered thermal–visible pairs of transmission-line components with box annotations. Existing public IR–VIS power data are image-level or conductor-mask only (Yetgin & Gerek; VITLD on request), and the registered substation sets (VISED, MSIS, Yang et al.) are private." Avoid "near-unique".
- The proposal's §5 row "Yetgin & Gerek … the only public paired thermal-visible power resource" should be rewritten. Cite VITLD as the closest paired transmission-line resource (lines only, 256×256, on request), and cite Yetgin & Gerek as large, unregistered, and classification-only.
- Releasing even a subset of the 753-pair set with boxes would be a real contribution (C3), since every paired power set found is either private or conductor-only.

### Gaps
- Could not confirm whether VISED, VITLD, or the Gomes et al. stereo IR frames can be downloaded. Next step: email the authors or check the TII supplementary material.
- IEEE DataPort and Roboflow Universe were not searched successfully. A paired power IR–VIS project could exist there unindexed.
- Could not read the ScienceDB HV-equipment dataset or the Kaggle IR dataset metadata (resolution, labels, pairing).

## Q2. Has anyone translated thermal↔visible for power equipment, used GAN/diffusion translation for power detection, transferred visible detectors to IR, or used IR–VIS fusion for power detection?

### Takeaway
Yes, partially, so "first to translate thermal→visible for power lines" is **not** safe. The closest prior art is **Li et al., EPEE 2025**: a diffusion model that translates IR→visible to improve **instance segmentation** in power-line inspection. Its abstract shows no detector/segmenter loss in the loop; quality comes from semantic and structural modulation modules. Visible→IR CycleGAN augmentation for power-equipment detection also exists (Yin et al., ICPICS 2023), as does visible-detect-then-transfer-boxes via stereo (Gomes et al. 2025). **No work found that trains a thermal→visible translator with a frozen visible-trained detector's loss in the loop in the power domain.** IR–VIS fusion for power detection is a crowded competitor space.

### Cited Findings
**Translation (closest threats)**
- **Li, Li, Zheng, Yuan, "Dual Modulation-Based Infrared-Visible Image Translation Diffusion Model for Power Line Inspection", 2025 5th Int. Conf. on Energy, Power and Electrical Engineering (EPEE), 2025-09-19, DOI 10.1109/EPEE67527.2025.11428694** (0 citations).
  - Abstract: infrared images have "lower data interpretability", which degrades existing instance-segmentation methods, "therefore, infrared to visible image translation presents an effective solution".
  - A semantic modulation module aligns IR/VIS feature distributions and a structural modulation module preserves structure, to reduce object distortion. Experiments use "the power grid dataset" (unnamed, presumably private).
  - Source: [OpenAlex abstract](https://api.openalex.org/works/https://doi.org/10.1109/epee67527.2025.11428694) **[abstract only — full text not accessed; whether a downstream loss is used in training is unverified]**
- **Yin, Li, Cui, Zhang, Wang, Si, "CycleGAN-Based Visible-Infrared Image Enhancement Method for Infrared Power Equipment Object Detection", IEEE ICPICS 2023, DOI 10.1109/ICPICS58376.2023.10235612** (9 citations). Translates visible power-equipment images into IR with CycleGAN + LSGAN to enlarge the IR pool; detection "significantly improved". This is the opposite direction and offline augmentation, with no detector in the loop. — [OpenAlex abstract](https://api.openalex.org/works/https://doi.org/10.1109/icpics58376.2023.10235612); [IEEE](https://ieeexplore.ieee.org/document/10235612/)
- Liu & Huang, *Sensors* 24(2):428, 2024: an "efficient cross-modality insulator augmentation" with a high-resolution insulator cross-modality translation (HICT) module, evaluated on visible UPID/SFID insulator-defect sets. "Modality" here seems to mean visual domains, not thermal (unverified). — [DOI](https://doi.org/10.3390/s24020428)
- Within-IR GAN work (not cross-modal): SA-CycleGAN IR enhancement for substation equipment, *Electronics* 2024 — [DOI](https://doi.org/10.3390/electronics13173376); DET-CycleGAN IR enhancement for cable terminals, *Infrared Phys. & Tech.* 2025 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1350449525003913); CycleGAN illumination style transfer of transformer IR images (95.2% identification) — [ADS](https://ui.adsabs.harvard.edu/abs/2024JPhCS2741a2016D/abstract).

**Visible→IR detector transfer without IR labels**
- Gomes et al. 2025 (above): visible detector plus stereo rectification and box transfer to IR. They report results "comparable to other works that performed detection directly on the infrared domain". — [Crossref](https://api.crossref.org/works/10.1590/1678-4324-2025240843). This needs a co-located visible camera at test time, which is exactly what thermal-only utility footage lacks. That is a clean differentiator.
- General-domain (not power): DOD-SA (Jin et al., arXiv 2508.10445, rev. 2026-08) does IR–VIS detection with **single-modality annotations** via teacher–student pseudo-label transfer across modalities. — [arXiv](https://arxiv.org/abs/2508.10445)

**IR–VIS fusion for power detection (competitors reviewers may ask for)**
- DSEFusion, Guo et al., *Knowledge-Based Systems* 2026, DOI 10.1016/j.knosys.2025.114792: scene/equipment feature decoupling plus a temperature-texture constraint loss; improves substation equipment detection. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0950705125018301) **[snippet only]**
- CBAM-YOLOv4 fusion on VISED, *Processes* 2025. — [DOI](https://doi.org/10.3390/pr13092720)
- RA-TPAF-YOLO thermal-prior-guided fusion, NDT&E 2026. — [DOI](https://doi.org/10.1080/10589759.2026.2732240)
- Dual-ResNet fusion + YOLOv5 on insulators, *Sensors* 2025. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12074302/)
- MSIS multispectral instance segmentation, 2022. — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8752308/)
- Transmission-line (conductor) RGB-T segmentation on VITLD: AMIFFM (TII 2022), MAINet (ESWA 2025), SSRNet with KD + contrastive learning (TCSI 2025), HMMEN (arXiv 2025). — links in Q1
- GAN-based IR–VIS fusion for substation fault points, IEEE Access 2020, DOI 10.1109/ACCESS.2020.2990539. — [DOI](https://doi.org/10.1109/access.2020.2990539)
- Others: "A Thermal-Aware Infrared–Visible Image Fusion Method for Defect Detection in Power Transmission Equipment" (Preprints.org 2026-07, takes spatially registered pairs) — [DOI](https://doi.org/10.20944/preprints202607.0999.v1); Yu, Shi, Li, REPE 2024 substation detection by IR+visible fusion — [DOI](https://doi.org/10.1109/repe62578.2024.10809697); "Fusion-Guided Recognition: A Staged Multi-modal Method for Power Line Detection" (Springer SIST 2026) — [DOI](https://doi.org/10.1007/978-981-92-0607-0_14); "Research on Overheating Identification … Based on Infrared and Visible Light Fusion" (IEEE 2025) — [IEEE](https://ieeexplore.ieee.org/abstract/document/10957634/); a lightweight substation detector via VIS–IR fusion (Chinese journal, 2025) — [jgcm.ac.cn](https://www.jgcm.ac.cn/dqgcxb/en/article/doi/10.11985/2025.06.003).
- The 2025 Applied Energy review found **no** documented IR–VIS fusion, domain-adaptation or GAN thermal–RGB synthesis work among the power-line papers it reviewed, and lists multimodal fusion only as a future direction. — [arXiv 2502.07826](https://arxiv.org/html/2502.07826v1). This undercounts the substation and fusion literature above, so it is not reliable as a gap citation on its own.

### Inferences
- **Must-cite and must-differentiate:** Li et al. EPEE 2025 (IR→VIS diffusion for power-line instance segmentation) is the direct prior-art hit. The likely differences are no frozen-detector loss in the loop, a segmentation rather than detection target, an unnamed dataset, and a short conference paper. All of these need confirmation from the full text before the paper claims novelty.
- Novelty should rest on **detection-in-the-loop with a frozen, visible-trained detector** and on the **thermal-only-at-test-time** deployment setting. "First thermal→visible translation for power equipment" is not safe.
- Fusion methods need both modalities at inference, so they do not solve thermal-only footage. Still, reviewers at TIM / Information Fusion will likely want one fusion upper-bound baseline on the paired set, e.g., a VISED-style feature-fusion YOLO or a generic TarDAL-type fusion method.
- Gomes et al. 2025 is a natural "visible-detect-and-transfer" reference/oracle. With registered pairs, our own pipeline can reproduce it as an upper bound that needs the visible camera.

### Gaps
- Full text of Li et al. EPEE 2025 not accessed. Unknown: dataset, classes, whether the segmentation loss enters training, and baselines (CycleGAN? CM-Diff?).
- Full text of DSEFusion (KBS) not accessed. Unknown: its dataset and whether it is public.
- Chinese-language-only journals (e.g., Power System Technology, Proc. CSEE, High Voltage Engineering) were not searched. IR→visible "colourisation" for power may exist there untranslated.

## Q3. How does the power-inspection community handle scarce thermal labels?

### Takeaway
Mostly by (i) training IR detectors directly on in-house labelled IR sets (thousands of images) with YOLO variants, (ii) GAN/diffusion **augmentation** (visible→IR CycleGAN; diffusion for anomalies and backgrounds; classic augmentation), (iii) synthetic/CAD rendering, (iv) weak supervision and few-shot fusion, and (v) occasionally box transfer from a co-registered visible camera. Reusing visible-trained detectors on translated thermal imagery is rare (only the EPEE 2025 segmentation paper).

### Cited Findings
- Visible→IR CycleGAN augmentation for power equipment detection, motivated by "scarcity of infrared image datasets". — [Yin et al. 2023](https://doi.org/10.1109/icpics58376.2023.10235612)
- Condition-controllable diffusion (Canny-edge guided) creates pseudo-anomalous substation samples for self-supervised anomaly detection (CARe). — [Xu et al., IET Cyber-Systems & Robotics 2025](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/csy2.70032) **[snippet only]**
- DDPM used to create complex-background substation samples; an 8,371-image in-house IR substation dataset with 5 classes (arresters, breakers, bushings, CTs, VTs). — [Zhu et al., IET Image Processing 2025](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70210) **[snippet only; attribution of the two claims to this one paper is uncertain]**
- Synthetic images rendered from manufacturer datasheet dimensions to train VJ/YOLO detectors when real target images are lacking (visible). — [Santos et al., Sensors 24(13):4219, 2024](https://doi.org/10.3390/s24134219)
- Weakly supervised IR fault identification for substation equipment. — [arXiv 2311.11214](https://arxiv.org/pdf/2311.11214)
- Few-shot VIS–IR fusion detection for power equipment (500 registered pairs). — [Hsieh & Yu, NDT&E 2026](https://doi.org/10.1080/10589759.2026.2732240)
- Box transfer from visible to IR through stereo rectification, explicitly to avoid expert IR annotation. — [Gomes et al. 2025](https://api.crossref.org/works/10.1590/1678-4324-2025240843)
- A 2026 *Sensors* review of IR power inspection cites "transfer learning and few-shot learning" as the general answer to "blurred target features and scarce samples in infrared images" and says "fault detection accuracy in small-sample scenarios remains low". It names no specific datasets or translation works. — [Guo et al., Sensors 2026, PMC13258969](https://pmc.ncbi.nlm.nih.gov/articles/PMC13258969/)
- Classic augmentation (rotation, flipping, brightness, noise) is the default expansion strategy for IR substation sets **[snippet only]**. — [Applied Sciences 15(1):328, 2025](https://www.mdpi.com/2076-3417/15/1/328)
- DCGAN used to rebalance IR thermal-defect training sets. — [Electronics 10(16):1986, 2021](https://doi.org/10.3390/electronics10161986) **[snippet only]**

### Inferences
- Our E8 low-annotation regime should be framed against these baselines: (a) a thermal detector fine-tuned on k labelled thermal images, (b) visible→thermal CycleGAN augmentation (Yin et al. style), (c) zero-label transfer by translation without the loop. Together these are the community-standard alternatives.
- The annotation-difficulty argument already appears in the power literature (Gomes et al. 2025 is quotable). Citing it strengthens the motivation.

### Gaps
- No power-domain semi-supervised (Mean-Teacher-style) IR detection paper was verified, although such papers likely exist in Chinese venues.
- Unknown whether any utility has published results on vehicle-mounted thermal video for component detection.

## Q4. Which venues publish this, and what claims/baselines do they accept?

### Takeaway
Power-domain IR/VIS work appears across IEEE TII, IEEE TPWRD, IEEE TCSI, IEEE TIM (generic IR–VIS fusion), Knowledge-Based Systems, ESWA, Sensors, Processes, Electronics, Applied Sciences, IET journals, NDT&E and Chinese-utility conference proceedings (EPEE, ICPICS, REPE). The typical accepted claim is "+x mAP / +y FPS over a YOLO baseline on a private dataset, plus an edge-device benchmark". Baselines are usually single-modality detectors plus 3–8 generic fusion or segmentation networks.

### Cited Findings
- IEEE TII: AMIFFM, compared against single-modal input and SOTA fusion methods across several baselines under day/night/fog/snow. — [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1109/tii.2022.3147833)
- IEEE TPWRD publishes power IR–VIS registration (CAO-C2F) evaluated on a self-built dataset, compared with SOTA registration on precision, recall and RMSE. — [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1109/tpwrd.2020.3011962)
- IEEE TCSI: VIS–IR transmission-line detection with KD and contrastive learning. — [IEEE](https://ieeexplore.ieee.org/document/10820014/)
- IEEE TIM regularly publishes generic IR–VIS fusion (GANMcC, TIM vol. 70, 2021; SIGFUSION, 2024; FAFusion, TIM vol. 73, 2024; CGTF, TIM vol. 71, 2022) **[snippet only]**. No TIM paper found that does power-domain IR→VIS translation. — [search listing via arXiv 2410.22837 refs](https://arxiv.org/pdf/2410.22837)
- Baseline patterns in power fusion papers: HMMEN compares U-Net, SegNet, MFNet, Lite-HRNet, BiSeNet, BGRNet plus fusion modules FuseNet, MMTM, AMIFFM — [arXiv 2501.15099](https://arxiv.org/html/2501.15099); MSIS compares Mask R-CNN, MS R-CNN, TensorMask, PANet, PolarMask, YOLACT++, SOLOv2 and RFN-fusion variants — [PMC8752308](https://pmc.ncbi.nlm.nih.gov/articles/PMC8752308/); *Processes* / NDT&E papers report mAP, FPS, model size and Jetson Nano / INT8 deployment — [Processes](https://doi.org/10.3390/pr13092720), [NDT&E](https://doi.org/10.1080/10589759.2026.2732240).

### Inferences
- For TIM, the "measurement" angle matters. Expect reviewers to ask for a quantified link between translation fidelity and downstream detection, robustness across capture conditions, and runtime on edge hardware. Power papers routinely report FPS and edge deployment.
- For Information Fusion, reviewers will likely see translation as "fusion-free cross-modal transfer" and ask for comparison to fusion upper bounds (DSEFusion-type, TarDAL-type) on the paired set.
- Power-domain baselines reviewers may expect: CycleGAN visible↔IR (Yin et al. 2023 as the power precedent), the IR→VIS diffusion of Li et al. 2025 (if code exists — probably not), visible-detect-and-transfer (Gomes et al. 2025) as an oracle, and a thermal-trained YOLO.

### Gaps
- Did not verify any Information Fusion paper specific to power equipment. Did not check TIM/TPWRD author guidelines on dataset release.

## Q5. Bottom line: is the power-line domain a genuine gap for detection-guided thermal→visible translation? Closest existing work?

### Takeaway
**Mostly yes, with one important caveat.** No paper was found that trains thermal→visible translation with a frozen visible-trained detector in the loop for power-line components, and no public registered thermal–visible component-level box dataset exists. But **IR→visible translation for power-line inspection has been published** (Li et al., EPEE 2025, diffusion, for instance segmentation). Visible→IR translation for power-equipment detection (Yin et al., ICPICS 2023) and visible-detection-to-IR box transfer (Gomes et al., BABT 2025) also exist. The novelty claim must be "detection-in-the-loop, thermal-only at test time, component-level, paired benchmark", not "first thermal→visible for power".

### Cited Findings
- Closest #1 (same direction, same domain, different supervision/task): Li et al., EPEE 2025, IR→VIS diffusion for power-line instance segmentation. — [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1109/epee67527.2025.11428694)
- Closest #2 (same goal of reusing visible detectors on IR without IR labels, but needs a co-registered visible camera at test time): Gomes et al., BABT 2025. — [Crossref](https://api.crossref.org/works/10.1590/1678-4324-2025240843)
- Closest #3 (translation for power detection, opposite direction, offline augmentation): Yin et al., ICPICS 2023. — [OpenAlex](https://api.openalex.org/works/https://doi.org/10.1109/icpics58376.2023.10235612)
- Closest paired power data: VITLD (TII 2022; lines only, on request), VISED (Processes 2025; substation, availability unclear), MSIS (2022; arresters/CTs, on request), Yang et al. (Sensors 2025; insulators, proprietary). — links in Q1
- A domain review confirms the general gap in public IR power data and multimodal methods for power lines. — [arXiv 2502.07826](https://arxiv.org/html/2502.07826v1); [Sensors 2026 PRISMA review](https://doi.org/10.3390/s26154819)

### Inferences
- Recommended edits to RESEARCH_FINDINGS §5:
  1. Replace "Near-unique — no public equivalent exists" with "No public registered thermal–visible component-level box dataset exists. Paired power sets are private (VISED, MSIS, Yang et al.) or conductor-only (VITLD)".
  2. Fix the Yetgin & Gerek row: 4,000 IR + 4,000 VL at 128×128 (classification) plus 200 + 200 at 512×512 (masks), not registered, MIT / CC BY 4.0.
  3. Add VITLD and VISED.
  4. Mark HIT-UAV as non-power.
- Related Work must cite and differentiate Li et al. 2025, Yin et al. 2023 and Gomes et al. 2025.
- Risk: Li et al.'s EPEE work comes from a power-grid group (affiliation not verified) and may become a journal paper. Monitor for a 2026 journal version.

### Gaps
- Full texts of Li et al. 2025, DSEFusion 2026, VISED 2025 and the Gomes et al. dataset release are unverified. These four are the highest-priority reads before submission.
- Chinese-language venues and IEEE DataPort/Roboflow were not covered, so a hidden paired power dataset or a CN-language IR→VIS work cannot be ruled out.
