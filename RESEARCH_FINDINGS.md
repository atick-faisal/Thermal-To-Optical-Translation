# Thermal→Visible Translation with Detection-in-the-Loop
## Research Implementation Plan

**Target:** Q1 journal (primary: IEEE TIM or Information Fusion)
**Hardware:** 2× NVIDIA A100 40GB, Windows Server 2022 — **native Windows, WSL2 unavailable**
**Scope of this document:** what we are building and why, which external codebases we draw from, and which experiments we run. Low-level implementation detail is deliberately excluded.

---

## 1. Objective

Build a research codebase that produces a defensible answer to one question:

> Does closing a training-time detection-consistency loop around a diffusion-based thermal→visible translator improve downstream power-line component detection, without hallucinating or erasing safety-relevant components?

Everything in the repo exists to support that claim or to defend it against the obvious reviewer attacks. The repo is not a product, not a deployment artifact, and not an inspection tool. It is an instrument for generating a results table.

### Contribution stack

The paper stands on four legs, not one. The naive "BBDM + SeAFusion" combination is not sufficient for Q1 on its own.

| #   | Contribution                                                                    | Role                                  |
| --- | ------------------------------------------------------------------------------- | ------------------------------------- |
| C1  | Closed-loop detection-consistency feedback for **diffusion** IR→VIS translation | Headline method                       |
| C2  | Faithfulness / hallucination metric for safety-critical translation             | Co-contribution, cheap, novel         |
| C3  | Paired thermal-visible power-component detection benchmark protocol             | Domain contribution, near-unique data |
| C4  | Analysis of *when* translation beats direct thermal detection                   | Defends the premise                   |

---

## 2. Compute and environment strategy

Native Windows imposes real constraints on a Linux-assuming research stack. These are accommodated by design rather than fought.

### Parallelism: no DDP

NCCL does not exist on Windows; PyTorch falls back to Gloo, which for GPU tensors is slow enough that distributed training is generally not worth it. **The second A100 is throughput, not scale.** One experiment per GPU via `CUDA_VISIBLE_DEVICES`, used for parallel seeds and configs. Every method must remain trainable on a single 40GB card — which the phased plan already assumes.

### Memory: assume no triton, no xformers, no torch.compile

Triton support on Windows is unreliable-to-absent, and `torch.compile` depends on it. xformers memory-efficient attention should likewise not be assumed. Mitigations, in order of value:

- PyTorch native `scaled_dot_product_attention` — works on Windows, recovers most memory-efficient-attention benefit
- Gradient checkpointing
- bf16 (A100 supports it natively)

**Expect tighter headroom than published Linux figures.** Measure actual VRAM in Phase 0 before committing to batch sizes and resolution.

### Data loading: spawn, not fork

Windows spawns worker processes. Dataset classes must be importable at module level, everything picklable, entry points guarded. This is the most common source of silent hangs on this platform — budget for it explicitly in Phase 0 rather than discovering it mid-run.

### Repo scripts and filesystem

- BBDM and CUT ship bash launchers that will not run. Invoke Python directly; drive everything through our own config layer.
- Enable Developer Mode (symlink permission) and long-path support before setup. Some research configs assume symlinked dataset directories — use junctions or absolute config paths instead.
- `pycocotools` may need prebuilt wheels or Visual C++ build tools.

### Revised budget assumptions

- **LBBDM-f4 @ 256²** — one A100 40GB, batch size to be determined empirically (Linux reference: 8–16; assume lower). Days, not weeks.
- **CUT / pix2pix** — hours to ~2 days, single card.
- **Full-sampler gradient backprop** — will not fit. This drives the entire coupling design (§6).

---

## 3. Repository architecture

Five layers. The boundary that matters most is that `core` never contains method-specific code.

```
core/          data contract, metrics, config schema, W&B logging, checkpointing
translators/   uniform wrapper per backbone; third_party/ pinned underneath
detection/     in-loop detector and evaluation detector, strictly separated
coupling/      detection-consistency loss and its schedule
experiments/   configs and results only
```

### Invariants

1. **One evaluation path.** Every method computes every metric through the same code. Non-negotiable — it is what makes the comparison table defensible.
2. **Frozen data contract.** Splits decided once, hashed, version-controlled. No method sees its own split logic.
3. **Backbones interchangeable.** A translator is anything that can `fit()` and `translate(batch) → tensor`. Swapping CUT for LBBDM is a config change.
4. **Third-party code vendored at pinned commits, never edited in place.** All adaptation lives in wrappers we own.
5. **The loop is a first-class component**, switchable off cleanly — because switching it off *is* the central ablation.
6. **An experiment is a config file.** Results carry their config hash.
7. **Two detectors, never conflated.** The in-loop detector guides training and receives generator gradients. The evaluation detector is trained independently and never does. Conflating them is the mistake reviewers catch.

### Dependency strategy

Attempt one `uv sync` with all dependencies first — the conflict may not exist. Note that uv dependency groups resolve into a *single* lockfile and venv; they document grouping but do not isolate. If resolution genuinely fails, fall back to per-adapter uv projects with the shared harness as an editable path dependency, and cross-venv subprocess invocation.

---

## 4. Source codebases

What we take from each. Nothing is forked wholesale except where noted.

### Translation backbones

| Repo                   | URL                                                    | Take                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BBDM / LBBDM**       | `github.com/xuekt98/BBDM`                              | Primary diffusion backbone. Brownian-bridge formulation, LBBDM-f4 config, VQGAN f4/f8/f16 integration. ~405★, actively issue-tracked. No pretrained IR2VIS weights — train from your paired data. |
| **BiBBDM**             | `github.com/xuekt98/BiBBDM`                            | Optional later, if bidirectional thermal↔visible becomes interesting.                                                                                                                             |
| **CUT / FastCUT**      | `github.com/taesungp/contrastive-unpaired-translation` | Phase-1 GAN baseline. Patch-NCE contrastive loss. ~2.3k★. **Pin versions** — repo issues document FID discrepancies and CUDA/Torch breakage.                                                      |
| **pix2pix / CycleGAN** | `github.com/junyanz/pytorch-CycleGAN-and-pix2pix`      | Paired baseline (pix2pix) and the CycleGAN-DA baseline for §8.                                                                                                                                    |
| **UNSB**               | `github.com/cyclomon/UNSB`                             | Unpaired diffusion translation. Reserved for IR-only benchmark data and data-efficiency ablations.                                                                                                |

### Loop / coupling mechanics

| Repo           | URL                                 | Take                                                                                                                                                                                                                                                 |
| -------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SeAFusion**  | `github.com/Linfeng-Tang/SeAFusion` | The task-in-the-loop recipe: task-loss backprop into the generator, joint low-level/high-level adaptive training schedule, warmup discipline. Information Fusion Best Paper 2024. We inherit the *mechanism*, not the two-input fusion architecture. |
| **DetFusion**  | `github.com/SunYM2020/DetFusion`    | Detection-loss-driven cascade (ACM MM 2022), object-aware content loss. mmdetection-based — useful reference for how detection loss is wired as a training signal.                                                                                   |
| **TarDAL**     | `github.com/JinyuanLiu-CV/TarDAL`   | Bilevel optimization formulation; also ships the M3FD benchmark and a bundled YOLOv5.                                                                                                                                                                |
| **MetaFusion** | `github.com/wdzhao123/MetaFusion`   | Meta-feature embedding to bridge the generation/detection feature gap. The reference for resolving gradient conflict if naive cascading proves unstable.                                                                                             |
| **IVIF_ZOO**   | `github.com/RollingPlain/IVIF_ZOO`  | Survey repo — catalog of task-driven fusion methods and their coupling taxonomies. Use for related-work coverage and baseline discovery.                                                                                                             |

### Reward / gradient tractability

Not repos we vendor, but the techniques that make detection-loss backprop through a diffusion sampler feasible on 40GB:

- **DRaFT** (arXiv:2309.17400, ICLR 2024) — truncated backprop to last K sampling steps; DRaFT-LV for K=1. LoRA rather than full-parameter updates.
- **ReFL** — evaluate the detector on a one-step-predicted x̂₀ from a randomly sampled late timestep. Cheapest option; our default.
- **AlignProp** — LoRA + gradient checkpointing if deeper backprop is needed.
- **LCM / consistency distillation** — distill to 1–4 steps so the whole sampler becomes differentiable cheaply. Escape hatch if ReFL signal proves too weak.

### Detection

- **Ultralytics YOLOv8/v11** — in-loop detector (differentiable detection loss, fast iteration) and evaluation detector. AGPL-3.0, confirmed acceptable.
- **mmdetection** — optional, if a Faster R-CNN comparison is wanted to match DetFusion's protocol.

---

## 5. Data layer

All datasets enter through adapters normalizing to one internal representation. Adding a dataset never touches training code.

### Primary

| Dataset                              | Modality                                       | Role                                                                    |
| ------------------------------------ | ---------------------------------------------- | ----------------------------------------------------------------------- |
| **Your custom paired power dataset** | Thermal + visible, paired, component-annotated | Headline testbed. Near-unique — no public equivalent exists. Drives C3. |

### Public paired IR-VIS (translation credibility)

| Dataset          | Size                            | Role                                                                                  |
| ---------------- | ------------------------------- | ------------------------------------------------------------------------------------- |
| **LLVIP**        | 15,488 aligned pairs, 1024×1280 | Primary public translation + detection benchmark. Strictly aligned in time and space. |
| **M3FD**         | 4,200 aligned pairs, 6 classes  | Detection-driven translation benchmark. Ships with TarDAL.                            |
| **MSRS**         | 1,444 aligned pairs             | Semantic-aware translation benchmark.                                                 |
| **FLIR-aligned** | 5,142 pairs                     | Cross-domain detection generalization.                                                |
| **HIT-UAV**      | 2,898 aerial thermal            | Overhead/aerial-inspection domain proximity. Thermal only.                            |

### Public power-line (domain grounding)

| Dataset                  | Modality                                   | Role                                                                                                                                  |
| ------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **InsPLAD**              | 10,607 UAV RGB, 17 asset classes           | Large visible detection/anomaly benchmark. Strongest support for the pretrained-detector-reuse argument (C4).                         |
| **CPLID**                | ~848 images, visible + *synthetic* defects | Visible-domain detection benchmark / pretraining. **No thermal** — verify before relying on it.                                       |
| **TTPLA**                | 1,100 aerial, instance seg                 | Aerial component benchmark.                                                                                                           |
| **Yetgin & Gerek IR+VL** | 400 IR + 400 VL, wire masks                | The only public paired thermal-visible power resource. Conductor-focused, not registered. Cite as motivation for why C3 is necessary. |

**Minimum for Q1 acceptance:** custom dataset + LLVIP and/or M3FD + one public power dataset. Reviewers will expect ≥3 total.

---

## 6. Method progression

Deliberately staged so that each phase de-risks the next and each produces a usable result even if the following phase fails.

### Phase 0 — Instrument
Establish the evaluation harness and the two reference points that bracket every claim: detector on raw thermal, detector on real visible. Reproduce LBBDM-f4 on LLVIP purely to validate tooling.

### Phase 1 — GAN loop (the go/no-go)
CUT or pix2pix + YOLOv8 detection loss, SeAFusion recipe. **This is the decision gate.** If translated-image mAP does not beat raw-thermal mAP on at least one class, stop and rethink the framing before escalating to diffusion.

### Phase 2 — Diffusion loop
LBBDM-f4 behind the extracted translator interface. Latent detection-consistency via ReFL/K=1 on the one-step-predicted x̂₀, LoRA-scaled, warmed up after the translator alone is stable.

Guardrails against generator collapse and reward hacking:
- LoRA scaling on the translator (most effective anti-hacking lever per DRaFT)
- λ_det warmup from near-zero; never start high
- Fidelity floor — reject updates that raise LPIPS past a threshold
- Independent evaluation detector

### Phase 3 — Defend
Full baseline suite, low-annotation sweeps, cross-dataset generalization, faithfulness stress tests.

### Phase 4 — Harden
Multi-seed runs, significance testing, complete ablation grid.

---

## 7. Experiment matrix

### E1 — Reference bracket
Detector on {raw thermal, real visible} × {detector trained on thermal, on visible}. Establishes floor and ceiling.

### E2 — Backbone comparison
{CUT, pix2pix, LBBDM-f4, UNSB} × {λ_det = 0}. Isolates translation quality from loop effects.

### E3 — The core ablation (most important experiment in the project)
{CUT, LBBDM} × {λ_det = 0, λ_det > 0} × 3 seeds. If the loop is not the causal driver of the gain, there is no paper.

### E4 — Coupling mechanism
Cascaded (SeAFusion/DetFusion style) vs bilevel (TarDAL style) vs meta-feature (MetaFusion style). Determines whether gradient conflict needs explicit handling.

### E5 — Gradient tractability
ReFL/K=1 vs DRaFT-K (K=2,4) vs LCM-distilled full-sampler. Quantifies the fidelity/compute/signal tradeoff of the tractability trick — a genuine methodological finding.

### E6 — Schedule
Warmup vs no-warmup; joint vs alternating; λ_det sweep. Establishes stability boundaries and where reward hacking begins.

### E7 — Detector-identity control
In-loop detector = evaluation detector vs independent. Directly answers the "you overfit to your own detector" critique.

### E8 — Low-annotation regime (defends the premise)
Sweep thermal annotation fraction {1%, 5%, 10%, 25%, 50%, 100%}. Hypothesis: translation's advantage is largest where thermal labels are scarce.

### E9 — Cross-dataset generalization
Train translation on A, evaluate detection on B, across all dataset pairs.

### E10 — Faithfulness stress test
False-object and missed-object rates as functions of λ_det. Expected to reveal the safety/performance frontier.

---

## 8. Baselines

Reviewers will demand these. Missing any one is a likely rejection.

- Detector trained directly on thermal (YOLO and Faster R-CNN)
- CycleGAN thermal→pseudo-RGB + detector (the IR2VI lineage)
- Unsupervised image-generation-enhanced adaptation (arXiv:2002.06770)
- **Meta-UDA** (arXiv:2110.03143) — meta-learning thermal domain adaptation
- **ModTr** (ECCV 2024 workshops) — modality translation for detection adaptation
- Task-conditioned domain adaptation for thermal (ECCV 2020)
- SeAFusion / TarDAL / DetFusion where a fusion comparison is fair (note: two-input, so framing matters)

---

## 9. Metrics

**Fidelity:** PSNR, SSIM, LPIPS, FID, KID — reported, but explicitly argued as insufficient. PSNR/SSIM reward blur; FID/KID use ImageNet backbones insensitive to domain-specific structure and are unreliable on small sets.

**Task:** mAP@50, mAP@50:95, per-class AP.

**Faithfulness (C2):**
- False-object rate — detections present in translated output, absent in ground truth
- Missed-object rate — ground-truth components erased by translation
- Detection-consistency between translated and real-visible
- Adapted Hallucination Index (Hellinger distance to a zero-hallucination reference, MICCAI 2024)

**Rigor:** ≥3 seeds, mean ± std, paired t-test or bootstrap CIs on mAP.

---

## 10. Acceptance criteria for drafting

Begin drafting when **all five** hold:

1. **Margin** — ≥ +2–4 mAP@50 over the strongest baseline on the custom dataset, with positive mAP@50:95 delta. Comparable published work reports gains of this order, so this is credible rather than cherry-picked.
2. **Consistency** — gain holds on ≥2–3 datasets.
3. **Causality** — λ_det ablation (E3) shows the loop drives the gain, seed-stable.
4. **Stability** — ≥3 seeds, significance-tested, no collapse.
5. **Faithfulness** — hallucination rates low and reported; can affirmatively claim no invention/erasure of safety-relevant components.

**Fallback framing:** if the loop helps only in low-annotation regimes, that remains a strong honest Q1 story — pivot the paper to data-efficiency and operator interpretability rather than raw SOTA.

---

## 11. Risks and mitigations

| Risk                                                | Signal                         | Mitigation                                                                             |
| --------------------------------------------------- | ------------------------------ | -------------------------------------------------------------------------------------- |
| Phase 1 fails — translation never beats raw thermal | E1 vs E3 at λ_det=0            | Stop before diffusion. Re-frame around low-annotation regime.                          |
| Reward hacking — mAP rises, images degrade          | LPIPS/FID rising with λ_det    | LoRA scaling, fidelity floor, early stopping, E6 sweep                                 |
| Gradient conflict — training unstable               | Loss oscillation, collapse     | Escalate cascaded → meta-feature (E4)                                                  |
| ReFL signal too weak at K=1                         | No mAP movement with λ_det > 0 | E5 escalation path to DRaFT-K or LCM distillation                                      |
| Direct thermal detection wins at full annotation    | E8 at 100%                     | Expected outcome — pivot to E8's low-label regime as headline                          |
| Third-party repo reproduction failure               | Setup phase                    | Pin commits; budget time explicitly for CUT/BBDM env issues on Windows                 |
| VRAM tighter than expected (no xformers/triton)     | Phase 0 OOM                    | SDPA + gradient checkpointing + bf16; reduce batch, then resolution; latent space only |
| DataLoader hangs on Windows spawn                   | Phase 0, silent stalls         | Module-level dataset classes, guarded entry points, low `num_workers` until stable     |
| Dataset release blocked                             | Any time                       | C3 degrades to a benchmark *protocol* contribution rather than data release            |

---

## 12. Out of scope

No inference service, no deployment tooling, no UI, no labeling tools, no real-time optimization, no edge deployment. If it does not contribute to the results table or its defense, it does not belong in this repo.

---

## 13. Target venues

| Venue                         | Fit                      | Values                                                          |
| ----------------------------- | ------------------------ | --------------------------------------------------------------- |
| **IEEE TIM**                  | Primary                  | Measurement/inspection applications, rigorous evaluation        |
| **Information Fusion**        | Primary                  | Multimodal, task-in-the-loop framing (SeAFusion's home)         |
| **IEEE TGRS**                 | Strong                   | UAV/aerial inspection + translation                             |
| **IEEE TII**                  | Strong                   | Industrial inspection framing                                   |
| **IEEE TPWRD**                | Conditional              | If power-engineering fault diagnosis leads over vision method   |
| **Pattern Recognition / TIP** | Only if C1+C2 generalize | Method over application; will press hardest on "why translate?" |
| **ESWA / Neurocomputing**     | Fallback                 | Broad scope, faster turnaround                                  |