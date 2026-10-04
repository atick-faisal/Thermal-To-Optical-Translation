# Diffusion-based IR/thermal→visible translation and task-reward fine-tuning of diffusion generators (gap review, as of 2026-10-03)

Scope note: GAN-based translation-for-detection (HalluciDet, ModTr, FoalGAN, etc.) is covered by another researcher and only mentioned here for context. Verification level is marked per item: **[abs]** = verified from the arXiv/CVF abstract page; **[full]** = verified from full text/HTML or official repo; **[snip]** = only seen in a search-result snippet or secondary page, not confirmed from a primary source; **[known]** = cited from prior knowledge with a stable arXiv ID, not re-fetched in this session.

## Q1. Which diffusion-based IR→RGB / thermal colorization / TIR→VIS methods exist (2022–2026), what do they evaluate, and do they use a task loss?

### Takeaway
Diffusion work in the **thermal→visible** direction is thin and dominated by **faces** (T2V-DDPM, DiffTV, MTVDiff) plus a few scene-level colorization/video papers (TC-PDM, CM-Diff, a Stable-Diffusion colorizer); none found trains the translator with a downstream detector loss, and none found uses a one-step (SD-Turbo / pix2pix-turbo) translator for thermal→visible. The **visible→thermal** direction is far more active (see Q3).

### Cited Findings
**Thermal→visible (our direction), diffusion-based**
- **T2V-DDPM** (Nair & Patel, IEEE FG 2023): DDPM learning the conditional distribution of visible faces given thermal faces, with a custom faster inference strategy; evaluated on face datasets; no detection, no task loss in abstract. [abs] — [arXiv 2209.08814](https://arxiv.org/pdf/2209.08814); [code](https://github.com/Nithin-GK/T2V-DDPM)
- **DiffTV** (ACM MM 2024): described as "the first Latent Diffusion Model specifically designed for T2V facial image translation" with identity preservation via feature alignment and dual-stage conditions; face-verification oriented, not detection. [snip] — [ACM DL](https://dl.acm.org/doi/10.1145/3664647.3680635); [OpenReview PDF](https://openreview.net/pdf/77645301257396ed47b36b35868055d8e4ab38de.pdf)
- **MTVDiff** (arXiv 2607.19886, 2026; Springer chapter): multimodal latent diffusion fusing thermal, depth and text (Dual-Branch Cross-Attention Fusion, Gated Text-to-Visual alignment) for thermal→visible faces. [snip] — [arXiv 2607.19886](https://arxiv.org/html/2607.19886)
- **Multi-attribute guided thermal face translation** (arXiv 2512.21032): latent-diffusion thermal→visible faces preserving identity. [snip] — [arXiv 2512.21032](https://arxiv.org/pdf/2512.21032)
- **TC-PDM** (Doan et al., arXiv 2408.14227, Aug 2024): patch diffusion for **infrared→visible video** with semantic-guided denoising (foundation-model features) and temporal blending; reports +35.3% FVD and **+6.1% AP50 on day-to-night object detection** — i.e. detection is *evaluated*, but no detection loss is described in the abstract. [abs] — [arXiv 2408.14227](https://arxiv.org/pdf/2408.14227)
- **CM-Diff** (Hu et al., arXiv 2503.09514, rev. Aug 2025): single diffusion network for **bidirectional** IR↔VIS with Bidirectional Diffusion Training (direction labels) and Statistical Constraint Inference; abstract mentions no downstream detection/segmentation and no task loss. [abs] — [arXiv 2503.09514](https://arxiv.org/html/2503.09514)
- **PTSD colorizer** (Cheng et al., *Engineering Research Express* 8(4), 2026): pre-trained Stable Diffusion + "shadow network" + Gram-matrix fusion module for TIR→RGB colorization; evaluated on KAIST and IRVI with SSIM/PSNR only; **no detection evaluation**. [abs] — [IOP](https://iopscience.iop.org/article/10.1088/2631-8695/ae4445); [code](https://github.com/jinxinhuo/PTSD-for-Infrared-Image-Colorization)
- A 2026 topical review of TIR→visible translation (Liu et al., *Measurement Science and Technology* 37(26), 2026) lists future directions as "cross-domain semantic consistency modeling, model generalization and robustness improvement, and optimization for lightweight and real-time applications" — task-driven/detection-in-the-loop training is not named as a direction in the accessible abstract. [abs] — [IOP](https://iopscience.iop.org/article/10.1088/1361-6501/ae7822)
- GAN-era TIR colorization for driving scenes (e.g. top-down guided attention, arXiv 2104.14374; FoalGAN "feedback-based object appearance learning", arXiv 2310.15688; memory-guided attention, arXiv 2208.02960) exists but is GAN-based — hand-off to the GAN reviewer. [snip] — [2104.14374](https://arxiv.org/pdf/2104.14374); [2310.15688](https://arxiv.org/pdf/2310.15688); [2208.02960](https://arxiv.org/pdf/2208.02960)

**One-step diffusion (pix2pix-turbo / CycleGAN-Turbo) and thermal**
- **img2img-turbo** (Parmar et al., arXiv 2403.12036): pix2pix-turbo and CycleGAN-Turbo adapt SD-Turbo to new paired/unpaired tasks via adversarial learning in a single step; demonstrated on sketch2photo, edge2image, day2night, weather — **no thermal task in the paper**. [abs] — [arXiv 2403.12036](https://arxiv.org/abs/2403.12036); [code](https://github.com/GaParmar/img2img-turbo)
- CycleGAN-Turbo has been applied in the **reverse** direction: RGB wildlife images → synthetic thermal for real-time animal detection, evaluated with YOLO variants and RT-DETR (arXiv 2609.32944, 2026). [snip] — [arXiv 2609.32944](https://arxiv.org/html/2609.32944)
- An aerial V→IR paper (AerialIRGAN, Sci. Rep. 2024) reports CycleGAN-Turbo outputs "maintain good structural information but fail to effectively reflect the infrared features." [snip] — [Nature Sci. Rep.](https://www.nature.com/articles/s41598-024-73381-0)
- I found **no paper or repo applying pix2pix-turbo/CycleGAN-Turbo to thermal→visible**, and none adding a task loss to img2img-turbo. Searches for "pix2pix-turbo thermal", "img2img-turbo infrared", "CycleGAN-Turbo infrared" returned only the reverse-direction uses above. — [search evidence: img2img-turbo repo](https://github.com/GaParmar/img2img-turbo)

**Not found under the names given in the brief**
- "DiffusionIR" as a thermal→visible method: not found. "PID" exists but is RGB→IR (see Q3). "ControlNet-based IR colorization": only generic ControlNet colorizers (e.g. ColorizeNet) found, nothing thermal-specific. — [ColorizeNet](https://github.com/rensortino/ColorizeNet)

### Inferences
- The thermal→visible diffusion literature is small and mostly face-centric; for scene-level TIR→VIS with detection, TC-PDM is the closest diffusion paper, and it evaluates (not trains with) detection.
- Using pix2pix-turbo for thermal→visible appears to be unclaimed in itself; the one-step backbone is what makes exact detector-loss backprop cheap, which differentiates us from multi-step DDPM/LDM translators that would need DRaFT/ReFL-style truncation.

### Gaps
- Full texts of DiffTV, MTVDiff and CM-Diff were not read; their dataset lists and whether any report detection mAP on translated images are unverified.
- The 2026 MST review's body (which diffusion methods it lists) was not accessible.
- Google Scholar "cited by" for img2img-turbo was not crawled (no tool access); a thermal→visible pix2pix-turbo paper in a low-visibility venue could have been missed.

## Q2. Who fine-tunes diffusion generators with differentiable downstream rewards, and has anyone used an object detector as the loss for an image-to-image diffusion translator?

### Takeaway
Differentiable-reward fine-tuning (DRaFT, ReFL, AlignProp, ControlNet++) and **task-driven one-step diffusion restoration** (EDTR, NOLA-IR, InfraredIR/CVPR 2026) are established; DRaFT already used a **detector (OWL-ViT) as a differentiable reward** for text-to-image, and InfraredIR (CVPR 2026) uses **SD-Turbo single-step + task-aware LoRA with detection losses (YOLOv8) on infrared images**. I found **no** work that backpropagates a **frozen, visible-trained detector's loss into a one-step diffusion cross-modal (thermal→visible) translator**. The mechanism is therefore not new; the specific cross-modal, frozen-foreign-detector, annotation-free-reuse setting appears to be.

### Cited Findings
**General differentiable-reward / RL fine-tuning of diffusion**
- **DRaFT** (Clark et al., ICLR 2024): backprop through the sampler, truncated to last K steps (DRaFT-K, DRaFT-LV); includes an **object-detection reward experiment with OWL-ViT** that adds or removes object classes (e.g. remove people, add strawberries) by maximizing/minimizing detection scores; notes the model "loses diversity and collapses to generate a certain high-reward image," mitigated by LoRA scaling, early stopping, KL; truncation also helps because full backprop suffers gradient explosion; DRaFT-LV ~2× faster than ReFL. [full] — [arXiv 2309.17400](https://arxiv.org/html/2309.17400); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/13e8be77982beb73d7ed0bbf122f9f3c-Paper-Conference.pdf)
- **ReFL / ImageReward** (Xu et al., NeurIPS 2023): reward-model gradient on a one-step x̂₀ from a random late timestep. [known] — [arXiv 2304.05977](https://arxiv.org/abs/2304.05977)
- **AlignProp** (Prabhudesai et al., 2023): end-to-end reward backprop with LoRA + gradient checkpointing. [known] — [arXiv 2310.03739](https://arxiv.org/abs/2310.03739)
- **DDPO** (Black et al., ICLR 2024) and **DPOK** (Fan et al., NeurIPS 2023): policy-gradient RL fine-tuning of diffusion on (possibly non-differentiable) rewards. [known/snip] — [DDPO arXiv 2305.13301](https://arxiv.org/abs/2305.13301); [DPOK NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/fc65fab891d83433bd3c8d966edde311-Paper-Conference.pdf)
- **Detection-based feedback for T2I** (Niu et al., arXiv 2412.00122, Nov 2024): uses object detection to count categories/quantities, turns it into a matching-score reward, and "fine-tune[s] the diffusion model by backpropagation the reward gradients" (gradient-based, not RL). Text-to-image, not translation. [abs] — [arXiv 2412.00122](https://arxiv.org/abs/2412.00122)
- **ControlNet++** (Li et al., ECCV 2024): frozen pre-trained discriminative models (e.g. UperNet for segmentation, depth estimators, edge detectors) as reward in a cycle-consistency loss; efficient reward by noising the input image and using a **single-step denoised** prediction; +11.1% mIoU (seg), +13.4% SSIM (lineart), +7.6% RMSE (depth) over ControlNet. Conditional generation, no detector reward. [abs] — [arXiv 2404.07987](https://arxiv.org/abs/2404.07987)
- **Reward tuning of one/few-step generators** is an active 2024–2026 thread: LaSRO latent surrogate reward for two-step models (arXiv 2411.15247), Wasserstein-gradient-flow reward fine-tuning of one-step generators (arXiv 2608.29647), Diff-Instruct with diffused reward (arXiv 2605.24001), DOLLAR (arXiv 2412.15689). The WGF paper states that multi-step fine-tuning methods "cannot be directly applied to one-step generation" because they rely on intermediate trajectory states. [snip] — [2411.15247](https://arxiv.org/html/2411.15247v1); [2608.29647](https://arxiv.org/html/2608.29647); [2605.24001](https://arxiv.org/html/2605.24001v1); [2412.15689](https://arxiv.org/pdf/2412.15689)

**Task-driven diffusion image *restoration* (closest mechanism to ours; same-modality I2I)**
- **EDTR** (Kim, Oh, Lee, ICCV 2025): SD-2.1 + trainable ControlNet, 1 or 4 denoising steps from a pre-restored LQ image; tasks: classification, segmentation, detection (Faster R-CNN on PASCAL VOC 2012); degradations: downsampling+JPEG, and downsampling+blur+noise+compression. **Crucially, "ℒtask updates only the task network ℋ and does not update EDTR"** because "backpropagating the task loss … causes instability to the EDTR"; task net and restorer are trained alternately (task net is *not* frozen). [full] — [arXiv 2507.22459](https://arxiv.org/html/2507.22459); [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Exploiting_Diffusion_Prior_for_Task-driven_Image_Restoration_ICCV_2025_paper.pdf)
- **NOLA-IR** (Kim & Lee, arXiv 2607.25390, Jul 2026): noise-free **one-step** LoRA (rank 64) on SD-2.1; task losses (classification, segmentation, detection, OCR) guide the LoRA; detector Faster R-CNN-MobileNetV3 on VOC 2012 / COCO 2017; task network jointly trained in alternation (not frozen); beats EDTR (detection 30.2→32.0 mAP); the paper does **not** analyze instability, reward hacking or hallucination. [full] — [arXiv 2607.25390](https://arxiv.org/html/2607.25390)
- **TaskTok** (arXiv 2606.26615): token-selective task-driven restoration reported 8.3× faster than EDTR. [snip] — [arXiv 2606.26615](https://arxiv.org/html/2606.26615v1)
- **UniRestore** (arXiv 2501.13134): unified perceptual + task-oriented restoration with diffusion prior. [snip] — [arXiv 2501.13134](https://arxiv.org/html/2501.13134)
- **InfraredIR — "Taming Generative Diffusion Model for Task-Oriented Infrared Imaging"** (Ma, …, Jinyuan Liu, Risheng Liu, Dalian Univ. of Tech., **CVPR 2026**): reformulates infrared imaging as a **single-step diffusion** on **SD-Turbo** with dynamic timestep estimation, a spectral (thermal-radiation) regularizer, and **task-aware LoRA via dynamic prompting**; downstream tasks: detection (YOLOv8 on M3FD), segmentation (SegFormer on FMB), small-target detection (SCTransNet on SIRST3). The loss includes a "semantic alignment loss … instantiated according to the supervision used by the downstream task such as detection (classification and box regression losses)". **Input and output are both infrared (degraded IR → restored IR), not thermal→visible.** [full for repo; snip for loss text — CVF PDF returned 403] — [GitHub](https://github.com/csmty/InfraredIR); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Taming_Generative_Diffusion_Model_for_Task-Oriented_Infrared_Imaging_CVPR_2026_paper.pdf)

**Detector-in-the-loop for diffusion in other modalities**
- **DetDiffusion** (Wang et al., CVPR 2024): SD-1.5 layout-to-image for detection data; a pre-trained detector labels each GT object easy/hard (IoU) as a "perception-aware attribute", plus a segmentation-based perception-aware loss. Detector informs conditioning, not a backpropagated detection loss into a translator. [snip/full-text page seen] — [arXiv 2403.13304](https://arxiv.org/html/2403.13304); [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_DetDiffusion_Synergizing_Generative_and_Perceptive_Models_for_Enhanced_Data_Generation_CVPR_2024_paper.pdf)
- **Dual-Critic Diffusion Alignment (DCDA)** (Li et al., ECCV 2026, arXiv 2607.01983): 4D-radar-conditioned diffusion refines weather-degraded LiDAR **features**, guided by a "detection-guided critic … anchored by a pre-trained clean-weather model" and a weather-adversarial critic. Feature-level, 3D, not image translation — but it is a "frozen clean-domain detector guides diffusion" precedent. [abs] — [arXiv 2607.01983](https://arxiv.org/pdf/2607.01983)
- **KeypointDiff** (You, Jia, Xu, arXiv 2503.19798): unpaired object-level SAR→optical diffusion for aircraft; a pre-trained keypoint detector is used **during sampling** (search snippets also mention a "detector-based supervision loss", unconfirmed from abstract). [abs + snip] — [arXiv 2503.19798](https://arxiv.org/abs/2503.19798)
- **Contrastive-SDXL** (arXiv 2605.16406): LoRA-tuned SDXL day→night augmentation with DINOv2 patch contrastive loss and an "object consistency loss" for pedestrian preservation (annotation-preserving augmentation). [snip] — [arXiv 2605.16406](https://arxiv.org/html/2605.16406)
- **Diffusion-based Image Translation with Label Guidance** (Peng et al., ICCV 2023): pixel-classifier loss gradient as sampling guidance for domain-adaptive *segmentation* (guidance, not fine-tuning). [snip] — [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Peng_Diffusion-based_Image_Translation_with_Label_Guidance_for_Domain_Adaptive_Semantic_ICCV_2023_paper.pdf)

**Ranked by overlap with "one-step diffusion thermal→visible translator, frozen visible detector loss backpropagated, off-the-shelf detector at test time"**

| Rank | Work | One-step diffusion | Cross-modal T→V | Detector loss into generator | Detector frozen & foreign-domain | Threat |
|---|---|---|---|---|---|---|
| 1 | InfraredIR (CVPR 2026) | Yes (SD-Turbo) | No (IR→IR) | Yes (cls+box loss, per snippet) | Unclear ("YOLOv8 fine-tuned on M3FD") | **High** on mechanism |
| 2 | NOLA-IR (2607.25390) | Yes (SD-2.1 LoRA) | No (RGB restoration) | Yes | No (jointly trained) | High on mechanism |
| 3 | EDTR (ICCV 2025) | 1–4 steps | No | **No** (explicitly unstable) | No | Medium; useful foil |
| 4 | DRaFT (ICLR 2024) | No (multi-step, truncated) | No (T2I) | Yes (OWL-ViT reward) | Yes | Medium (detector-as-reward precedent) |
| 5 | ControlNet++ (ECCV 2024) | Single-step reward trick | No | Frozen seg/depth reward | Yes | Medium |
| 6 | Detection-feedback T2I (2412.00122) | No | No | Yes (reward gradient) | Yes | Low–medium |
| 7 | TC-PDM (2408.14227) | No | **Yes (IR→VIS video)** | No (detection only evaluated) | — | Medium on task/domain |
| 8 | DCDA (ECCV 2026) | No | No (LiDAR features) | Detection-guided critic | Yes | Low–medium |
| 9 | DetDiffusion (CVPR 2024) | No | No | Detector-derived attributes | Yes | Low |
| 10 | CoVisIT / SC-Diff / UAV RGB→IR | No | Reverse (V→IR) | No | — | Low (paradigm competitor, Q3) |

### Inferences
- A reviewer will say "task-loss backprop into a one-step diffusion restorer is done (InfraredIR CVPR 2026, NOLA-IR 2026)" and "detector-as-differentiable-reward is done (DRaFT 2024)". Our defensible delta is the **combination**: cross-modal (thermal→visible, a large appearance gap, not denoising), a **frozen detector trained only on the other modality** (no thermal labels, no detector adaptation at test time), and the deployment claim of reusing off-the-shelf visible detectors.
- EDTR reports that backpropagating task loss into the diffusion restorer is unstable; NOLA-IR and InfraredIR do it in one-step LoRA settings. Our success with exact gradients through a one-step SD-Turbo translator with a *frozen* detector is consistent with "one-step makes it stable" — worth stating explicitly and citing EDTR as the counterpoint.
- InfraredIR must be cited and distinguished; it is the single most dangerous prior work because it shares backbone (SD-Turbo), single-step, LoRA, YOLOv8 detection loss and infrared data, and comes from the TarDAL/M3FD group (Jinyuan Liu, Risheng Liu).

### Gaps
- InfraredIR's CVF PDF returned HTTP 403; whether its YOLOv8 is frozen during generator training, and the exact form of the semantic alignment loss, are unverified beyond a search snippet and the repo README.
- AlignProp, ImageReward/ReFL and DDPO were cited from known arXiv IDs, not re-fetched.
- No primary source found where any detector loss trains an image-to-image diffusion translator across modalities (thermal, SAR, night→day, sim2real, medical). This is an absence-of-evidence finding from ~30 searches, not proof.

## Q3. Diffusion-based thermal data synthesis (RGB→thermal) for label-scarce thermal detection — the reverse paradigm

### Takeaway
The reverse paradigm (synthesize thermal from labelled RGB, then train a thermal detector) is the **dominant** use of diffusion in IR translation in 2024–2026, with strong reported gains (e.g. SD3.5+ControlNet lifting UAV IR vehicle mAP from 25.6 to 38.4). Reviewers will raise it as the obvious alternative to annotation-free reuse; our counter is deployment-time: no thermal detector training, reuse of an existing visible detector.

### Cited Findings
- **DiffV2IR** (Ran, Wang, et al., arXiv 2503.19012, 2025; SSRN version): V→IR with Progressive Learning Module and Vision-Language Understanding Module; releases **IR-500K** (500k IR images). [abs] — [arXiv 2503.19012](https://arxiv.org/abs/2503.19012); [code](https://github.com/LidongWang-26/DiffV2IR)
- **PID** (arXiv 2407.09299): physics-informed diffusion for infrared image generation (RGB→IR). [snip] — [arXiv 2407.09299](https://arxiv.org/pdf/2407.09299)
- **ECDM** (Zhu et al., ACM MM 2024): edge-guided conditional diffusion + modality-adversarial training to make pixel-aligned pseudo-thermal from visible on LLVIP for thermal-detection data. [abs] — [arXiv 2408.03748](https://arxiv.org/abs/2408.03748)
- **F-ViTA** (Paranjape, de Melo, Patel; WACV 2026): InstructPix2Pix conditioned on SAM + Grounded-DINO masks/labels for visible→LWIR/MWIR/NIR, five datasets. [abs] — [arXiv 2504.02801](https://arxiv.org/abs/2504.02801); [WACV PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Paranjape_F-ViTA_Foundation_Model_Guided_Visible_to_Infrared_Translation_WACV_2026_paper.pdf)
- **TherA** (Lee, …, Ukcheol Shin, Ayoung Kim; arXiv 2602.19430; listed under CVPR 2026 by a paper-notes site): VLM-produced thermal-aware embedding + latent-diffusion translator for controllable RGB→thermal; SOTA on FLIR and M3FD, up to 33% zero-shot improvement. [abs; venue snip] — [arXiv 2602.19430](https://arxiv.org/html/2602.19430); [notes](https://en.papernotes.org/CVPR2026/image_generation/thera_thermal-aware_visual-language_prompting_for_controllable_rgb-to-thermal_in/)
- **SC-Diff** (Zhang, …, Chenqiang Gao; arXiv 2608.08555, Aug 2026): SAM3 semantic masks as condition plus Semantic-Guided Self-Attention Calibration; "produces more effective synthetic training data for downstream infrared object detection." [abs] — [arXiv 2608.08555](https://arxiv.org/abs/2608.08555)
- **CoVisIT** (Linfeng Tang et al., NeurIPS 2026; code not released): V→IR diffusion guided by aligned low-res IR; evaluated on LLVIP, M3FD, FMB, MSRS; downstream fusion, segmentation (SegNeXt/FMB) and detection (YOLOv8/LLVIP); losses = diffusion + reconstruction + thermal-distribution consistency (no detection loss). [full repo README] — [GitHub](https://github.com/Linfeng-Tang/CoVisIT)
- **RGB-to-IR for UAV vehicle detection** (Eker et al., TNO, arXiv 2609.02556, Sep 2026): compares supervised GANs, ControlNet diffusion and LoRA editing; **SD3.5 + ControlNet best, improving RF-DETR mAP 50.8→60.1 on Kust4K and 25.6→38.4 on VTUAV**; translators trained independently of the detector. [abs] — [arXiv 2609.02556](https://arxiv.org/abs/2609.02556)
- **Inference-time scaling for IR data generation** (Horstmann et al., NeurIPS 2025 Workshop): FLUX.1-dev PEFT-tuned on IR, CLIP-based verifier at inference; 10% FID reduction on KAIST. [abs] — [arXiv 2511.07362](https://arxiv.org/abs/2511.07362)
- **ThermalDiffusion** (ICRA 2025 TIRO workshop, arXiv 2506.20969): conditional DDPM RGB→thermal to augment robotics datasets. [snip] — [arXiv 2506.20969 summary](https://pith.science/paper/2506.20969)
- **ThermalGen** (arXiv 2509.24878): style-disentangled flow-based RGB→thermal. [snip] — [arXiv 2509.24878](https://arxiv.org/pdf/2509.24878)
- **Text2Thermal** (arXiv 2609.03585): physics-aware thermal synthesis from text. [snip] — [arXiv 2609.03585](https://arxiv.org/pdf/2609.03585)
- **ThermalDiff** (J. Vis. Commun. Image Represent., 2025) and improved U-Net / conditional diffusion V→TIR (Neurocomputing, 2025). [snip] — [ThermalDiff](https://www.sciencedirect.com/science/article/abs/pii/S1047320325001385); [Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231225016789)
- **Thermal-Det** (arXiv 2605.10130): open-vocabulary thermal detector using synthetic thermal supervision + thermal-text alignment + RGB→thermal distillation for "true zero-shot" thermal detection — a direct competitor to the *annotation-free* framing from the detector-adaptation side. [snip] — [arXiv 2605.10130](https://arxiv.org/html/2605.10130)
- **FusionProxy** (Guo et al., arXiv 2605.06010, May 2026): distilled diffusion to add thermal awareness to RGB systems in real time, evaluated on recognition and closed-loop driving. [abs] — [arXiv 2605.06010](https://arxiv.org/abs/2605.06010)

### Inferences
- Reviewers' likely objection: "Why not synthesize thermal training data from your visible labels (SD3.5+ControlNet, SC-Diff, DiffV2IR) and train a thermal detector?" Our answer has to be empirical (E8-style) or practical: we need **no labels at all** in thermal *and* no detector retraining, and our power-line visible detector already exists. An RGB→thermal synthesis baseline on power lines would pre-empt this.
- None of the reverse-direction papers puts the detector in the generator's loss either, so "detection-in-the-loop" is also open in the V→IR direction.

### Gaps
- Full texts of PID, DiffV2IR, ThermalDiffusion and ThermalGen not read; whether each reports detection mAP is unverified.
- TherA's CVPR 2026 acceptance is from a secondary notes site only.

## Q4. Hallucination in diffusion translators and metrics for object-level faithfulness

### Takeaway
Hallucination (inventing/erasing objects) is a recognised failure of generative translators, and 2024–2026 papers propose region-local generation, semantic calibration, and segmentation-discriminator suppression; but there is **no standard object-level faithfulness metric for translation** — papers use downstream mAP, FID/KID/LPIPS/SSIM, or domain-specific indices.

### Cited Findings
- **Target-class hallucination suppression** (Li, Tan, Tan; **AAAI 2026 oral**, arXiv 2602.15383): in unpaired day→night translation, "objects from target classes such as traffic signs and vehicles … are incorrectly synthesized. These hallucinations significantly degrade downstream performance." A Schrödinger-bridge translator with a dual-head discriminator that segments hallucinated regions plus class prototypes; +15.5% mAP day→night adaptation on BDD100K, +31.7% on traffic lights. [abs] — [arXiv 2602.15383](https://arxiv.org/pdf/2602.15383)
- **Local diffusion for structural hallucination** (ECCV 2024, arXiv 2404.05980): frames hallucination as error in object-structure estimation, says "there is no widely accepted method to evaluate hallucination magnitude," and reduces it by region-wise diffusion. [snip] — [arXiv 2404.05980](https://arxiv.org/pdf/2404.05980); [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10498.pdf)
- **Hallucination Index** (Springer LNCS 2024): an image-quality metric for generative reconstruction models (medical). [snip] — [Springer](https://link.springer.com/chapter/10.1007/978-3-031-72117-5_42)
- **SC-Diff** motivates its design by noting diffusion V→IR methods that use semantic priors only as external conditions find it "difficult to preserve object locations, shapes, and semantic layouts"; evaluated with FID/KID/LPIPS/SSIM. [abs/snip] — [arXiv 2608.08555](https://arxiv.org/html/2608.08555)
- **DRaFT** reports reward over-optimization: the model "loses diversity and collapses to generate a certain high-reward image." [full] — [arXiv 2309.17400](https://arxiv.org/html/2309.17400)
- **NOLA-IR** does not analyze reward hacking or hallucination under task-driven optimization. [full] — [arXiv 2607.25390](https://arxiv.org/html/2607.25390)
- **Contrastive-SDXL** adds an explicit object-consistency loss to keep pedestrians through day→night diffusion augmentation. [snip] — [arXiv 2605.16406](https://arxiv.org/html/2605.16406)

### Inferences
- With a detector in the loss, the specific risk is that the generator **invents detector-pleasing objects** (false positives) or adversarial textures that only fool the training detector. A held-out detector (the plan's E7 detector-identity control) plus per-image false-positive / hallucinated-box rates against thermal-side GT boxes would be a novel, cheap object-level faithfulness protocol; no prior translation paper found reports it.
- The AAAI 2026 paper gives a citable precedent that hallucinated target-class objects degrade mAP — useful to justify a faithfulness stress test (E10).

### Gaps
- No paper found that measures object-level hallucination specifically for IR→VIS translation.
- I did not find a published study on adversarial/"reward-hacking" transfer when a frozen detector trains a translator and a different detector evaluates it.

## Q5. What sub-claim remains novel, and what will a reviewer say is already done?

### Takeaway
Already done: diffusion IR↔VIS translation (mostly V→IR and faces), one-step diffusion + task/detection loss for **restoration** (InfraredIR CVPR 2026, NOLA-IR 2026), detectors as differentiable diffusion rewards (DRaFT 2024), and task-in-the-loop for IR–VIS **fusion** (SeAFusion/TarDAL, per RESEARCH_FINDINGS §4). Apparently still open: a **one-step diffusion thermal→visible translator fine-tuned with exact gradients from a frozen, visible-only detector** so that off-the-shelf visible detectors can be reused on thermal-only input without thermal labels, measured on power-line components.

### Cited Findings
- Closest mechanism (one-step SD-Turbo + task LoRA + YOLOv8 detection loss) is IR→IR restoration, not cross-modal translation. — [InfraredIR GitHub](https://github.com/csmty/InfraredIR)
- Closest one-step task-driven restorer trains the task network jointly, not frozen. — [NOLA-IR](https://arxiv.org/html/2607.25390)
- Prior multi-step diffusion TDIR found task-loss backprop into the diffusion model unstable and avoided it. — [EDTR](https://arxiv.org/html/2507.22459)
- Detector-as-differentiable-reward exists for text-to-image. — [DRaFT](https://arxiv.org/html/2309.17400); [2412.00122](https://arxiv.org/abs/2412.00122)
- Diffusion IR→VIS work evaluates detection at most (TC-PDM, +6.1 AP50) without training on it. — [TC-PDM](https://arxiv.org/pdf/2408.14227)
- The V→IR data-synthesis paradigm reports large detector gains without detector-in-the-loop. — [arXiv 2609.02556](https://arxiv.org/abs/2609.02556); [CoVisIT](https://github.com/Linfeng-Tang/CoVisIT)
- One-step generator reward fine-tuning is recognised as needing different machinery than multi-step methods. — [arXiv 2608.29647](https://arxiv.org/html/2608.29647)

### Inferences
- **Likely reviewer statements:** (1) "Task-driven one-step diffusion with detection loss = InfraredIR (CVPR 2026) / NOLA-IR; incremental." (2) "Detector reward fine-tuning = DRaFT/ReFL; you just apply it." (3) "Why not RGB→thermal synthesis + thermal detector (DiffV2IR, SC-Diff, SD3.5-ControlNet)?" (4) "Is the gain just adversarial overfitting to the training YOLO?" (5) "Gain 0.80→0.85 on 753 pairs — significance?"
- **Defensible novelty claims (ordered by strength):**
  1. Cross-modal (thermal→visible) one-step diffusion translator trained with a **frozen detector from the target (visible) modality only** — no thermal labels, no detector adaptation — for annotation-free reuse of deployed visible detectors. Not found anywhere.
  2. Evidence that exact (non-truncated) detector gradients through a one-step translator are stable and useful, contrasting EDTR's reported instability for multi-step diffusion.
  3. First power-line/transmission-component thermal→visible translation benchmark with detection-in-the-loop (domain novelty; no diffusion IR→VIS paper on power lines found).
  4. An object-level faithfulness protocol (held-out detector, hallucinated-box rate) for detector-trained translators — absent from all surveyed translation papers.
- **Must-cite and must-distinguish:** InfraredIR (CVPR 2026), NOLA-IR, EDTR, DRaFT, ControlNet++, TC-PDM, CM-Diff, CoVisIT/SC-Diff/DiffV2IR (reverse paradigm), AAAI 2026 hallucination paper, img2img-turbo.
- Baselines a reviewer may request: pix2pix-turbo without loop (already have), a reverse-paradigm baseline (RGB→thermal synthesis + YOLO11 trained on synthetic thermal), and possibly CM-Diff or TC-PDM as diffusion IR→VIS baselines.

### Gaps
- Given 2026 arXiv volume, a direct competitor posted after ~Sep 2026 or in a non-indexed venue (e.g. Chinese journals, IEEE TGRS/TIM early access) may have been missed; a final check of IEEE Xplore for "thermal visible translation detection loss diffusion" before submission is advisable.
- InfraredIR's detector-freezing and loss details need confirming from the full CVPR PDF, which was not retrievable here (403).
