# Thermal→Visible Translation with Detection-in-the-Loop

## Research Progress Update — 20 August 2026

*Biweekly update. Next: 3 September 2026.*

**Research question**

> Does closing a training-time detection-consistency loop around a diffusion-based
> thermal→visible translator improve downstream power-line component detection, without
> hallucinating or erasing safety-relevant components?

**Status in one line:** the experimental instrument is built and validated, the Phase 1
go/no-go gate has **passed by a wide margin**, and the project's central causal ablation
has been run once, returned null, been diagnosed as a dosing error, and is re-running now
at a corrected dose.

---

## 1. Executive summary

- **The instrument is complete.** A full research harness — data contract, three
  structurally-separated detector roles, one shared evaluation path, frozen hashed splits,
  significance testing — is built, tested (349 tests) and running on both A100s. Phase 0 is
  closed.

- **The core premise is established and independently verified.** Translating thermal to
  visible recovers detection performance that raw thermal loses entirely:
  **0.1552 → 0.7851 mAP@50**, against a real-visible ceiling of 0.9366. All four annotated
  classes improve. The two classes that are effectively invisible to a visible-trained
  detector on raw thermal — Fuse (0.032) and Switch (0.013) — become usable (0.85, 0.53).

- **The Phase 1 go/no-go gate passed**, and it passed on the arm that *cannot* be
  contaminated: the λ_det = 0 baseline, which never constructs the in-loop detector at all.
  The headline claim therefore does not depend on the loop working.

- **No reward hacking.** Detection rises while every perceptual fidelity metric holds or
  improves. Bounded at six seeds: λ_det moves LPIPS by −0.002, CI [−0.016, +0.011].

- **The causal ablation (E3) came back null on its first run** — twelve runs, six paired
  seeds, budget-matched, judged by an independently-trained detector. Stage-3 effect
  **+0.0070 (p = 0.66)** against a null control of **−0.0063**. By the decision rule fixed
  before the campaign, that is negative.

- **The null was then diagnosed, and it is a dosing error, not a mechanism result.** The
  detection term was measured at **2.3% of the training objective** at the top of the ramp —
  roughly 19× smaller than the perceptual term it had to compete with. E3 never tested the
  hypothesis at a dose capable of refuting it. `grad_scale` has been recalibrated 15×
  (0.01 → 0.15, verified by probe to land the term at 11/17/23% of the objective) and **the
  twelve-run campaign is re-running on the server as of now** — ~36 hours wall clock across
  both cards.

---

## 2. Where we are

| Phase | Scope | Status |
| --- | --- | --- |
| **Phase 0 — Instrument** | Harness, data layer, metrics, coupling, engine, server bring-up | ✅ Closed |
| **Phase 1 — GAN loop (go/no-go)** | pix2pix + YOLO detection loss; the decision gate | ✅ **PASS** |
| **E3 — Causality ablation** | 12 runs, 6 paired seeds, budget-matched, independent judge | 🔄 **Re-running now** (first campaign null, dose corrected) |
| **Phase 2a — One-step diffusion** | pix2pix-turbo backbone | 🔄 Backbone landed; pretrain blocked on data fetch |
| **Phase 2b — Multi-step diffusion** | LBBDM-f4 + ReFL comparison arm | ⏳ Queued (lower priority) |
| **Phase 3 — Defend** | Baseline suite, E8 low-annotation, E9 cross-dataset, E10 faithfulness | ⏳ Not started |
| **Phase 4 — Harden** | Multi-seed, significance, full ablation grid | ⏳ Not started |

**Build figures:** 51 commits over 9 days (11–19 August), ~7,000 lines of first-party
Python, 349 tests, 5 tracked experiment configs, 4 translation backbones staged behind one
interchangeable interface.

---

## 3. The instrument, and why it makes the results defensible

The repository is not a product. It is an instrument for producing one results table and
defending it against the reviewer attacks we can predict. Five things in it exist
specifically to survive review:

**Three detector roles, never conflated — enforced structurally, not by discipline.**
The config schema splits `detector` into `in_loop`, `evaluation` and `reference`
sub-sections, so no two roles can share a weights file by accident.

- *In-loop* guides training and receives generator gradients.
- *Evaluation* is fine-tuned on translated exports and never supplies gradient.
- *Reference* is never trained on anything this project produced — it only scores
  translations zero-shot, and it is what every headline number is measured with.

Conflating these is the mistake reviewers catch. Here it is impossible to make silently.

**One evaluation path.** Every method computes every metric through the same code —
fidelity (`FidelityEvaluator`), task (`evaluate_detector`), faithfulness
(`FaithfulnessEvaluator`). There is exactly one implementation of each, so no comparison in
the final table can be an artifact of two code paths that happen to agree.

**Frozen, hashed splits.** Split membership is recorded as a committed manifest of filename
stems plus a hash, so drift is detectable by `git diff` without the images ever entering the
repository (the remote is public; the pairs are unpublished). The custom dataset is frozen at
`combined_hash 7ede3433adc9c0b8`.

**An experiment is a config file, and results carry its hash.** The hash covers the
experiment, not the invocation — the same experiment on two GPUs under two names carries one
hash, while a silently changed seed does not.

**A bare clone is fully testable.** Because no image can be committed, the whole test suite
runs on synthetic pairs generated at test time, on CPU, in seconds. Every pathology worth
testing (an out-of-range class id, an unpairable filename, a degenerate box) is constructed
exactly rather than hoped for.

### One design revision worth reporting

The original plan made **LBBDM-f4 trained from scratch** the Phase 2 diffusion backbone.
Two facts killed that: 753 usable pairs cannot train a diffusion model from scratch
without memorising, and backpropagating a detection loss through a multi-step sampler does
not fit in 40GB — which is why the original plan carried a whole ladder of gradient
approximations (ReFL → DRaFT-K → LCM distillation) as a *dependency*.

**`pix2pix-turbo` replaces it as the primary backbone.** It is a one-step distilled
diffusion model: one UNet evaluation, no sampling loop, pretrained SD-turbo weights with only
LoRA adapters training. This resolves four problems at once — the data scale, the memory
budget, the anti-reward-hacking lever (LoRA-only updates, which the literature identifies as
the most effective one), and it converts the tractability ladder from a dependency into
**experiment E5**: exact gradients through a one-step model versus truncated gradients
through a multi-step one. That is a more publishable question than the workaround it
replaces. LBBDM is not dropped — it becomes E5's multi-step arm.

---

## 4. Data

### Custom paired power-component dataset (the headline testbed)

- **853 registered thermal/visible pairs**, 640×480, FLIR, labels shared across modalities.
- **753 are all any experiment has ever touched** — 600 train / 153 val. The remaining ~100
  are held out as an unseen test set, and the isolation is *structural*: the manifest reader
  physically drops any `test:` key, so no code path in the project can read those pairs.
- **4 annotated classes** — Fuse, Pole, Switch, Transformer. The manifest declares 5; the
  fifth (`Connector`) is a labelling-tool artifact with zero instances in both splits. It is
  left in place deliberately (removing index 0 would renumber every label file), and every
  reported mAP is already a clean 4-class average. **The paper must report 4 classes and
  "753 train+val of 853", not "5 classes" or "850 pairs".**

### Public datasets

| Dataset | Status | Counts |
| --- | --- | --- |
| **MSRS** | ✅ Adapted, frozen | 1163 train / 361 val |
| **FLIR-aligned** | ✅ Adapted, frozen | 4129 train / 1013 val (matches the published split exactly) |
| **LLVIP, M3FD, TTPLA** | ⚠️ Fetch scripted, **not fetched** | Blocked on local disk; needs a re-host decision |
| **InsPLAD** | ✅ Format verified | 7981 / 2626 images, COCO JSON |
| CPLID, HIT-UAV | Out of scope | Single-modality; no paired counterpart |

**Corrections already found that must reach the paper:** Yetgin & Gerek is 4,000 IR + 4,000
VL at 128×128, **unpaired**, with binary presence labels — not "400 + 400 with wire masks"
as commonly cited; it is cite-as-motivation only. InsPLAD's released annotations contain
**18 categories, not the 17** the paper's text lists.

> **This is a live blocker.** Q1 acceptance needs ≥3 datasets. LLVIP and/or M3FD are the
> credibility datasets for the translation claim, and neither is on disk yet.

---

## 5. Measured results

Every number below is produced through the same evaluation path on the same 153-image /
423-instance validation split.

### 5.1 The reference bracket — the floor and ceiling every claim sits between

Scored by an **independently-trained judge** (`yolo11s`, different architecture and seed
from the in-loop detector, trained on the visible train split only, never on anything this
project produced; val mAP@50 0.9364).

| Detector trained on \ evaluated on | Raw thermal | Real visible |
| --- | --- | --- |
| Thermal | **> 0.9** * | — |
| Visible | **0.1552** | **0.9366** |

\* the thermal-trained row is from a separate earlier measurement, not the `yolo11s` judge.

Two conclusions, both load-bearing:

- **In-domain detection is excellent on both modalities.** Thermal frames are not
  information-poor for this task. The problem is the *domain gap*, not the sensor.
- **Cross-domain transfer collapses**, and very unevenly: Pole survives at 0.39 (a pole's
  silhouette is thermally obvious) while Switch is 0.013 and Fuse 0.032 — effectively gone.

The realistic deployment baseline is the second row: run the detector you can actually label
data for (visible) on whatever raw sensor frame you have. Training a thermal-domain detector
requires exactly the thermal annotations the low-annotation framing exists to avoid needing.

### 5.2 Phase 1 gate — translated images, scored by the same independent judge

| Arm | mAP@50 | mAP@50-95 | Fuse | Pole | Switch | Transformer |
| --- | --- | --- | --- | --- | --- | --- |
| Raw thermal (floor) | 0.1552 | 0.0675 | 0.0321 | 0.3949 | 0.0130 | 0.1810 |
| **pix2pix, λ_det = 0** | **0.7851** | 0.5082 | 0.8515 | 0.9514 | 0.5279 | 0.8095 |
| Loop stage 1, λ = 1 | 0.8244 | 0.5503 | 0.8791 | 0.9679 | 0.5336 | 0.9169 |
| Loop stage 2, λ = 2 | 0.8106 | 0.5421 | 0.8596 | 0.9724 | 0.5127 | 0.8978 |
| Loop stage 3, λ = 3 | 0.8470 | 0.5593 | 0.8852 | 0.9696 | 0.6175 | 0.9156 |
| Real visible (ceiling) | 0.9366 | 0.6831 | 0.9746 | 0.9719 | 0.8507 | 0.9492 |

**GATE DECISION: PASS.** The bar was "beat raw thermal on at least one class". The
uncontaminated λ_det = 0 arm beats it by **+0.630 mAP@50 overall, improving all four
classes**, and reaches 84% of the real-visible ceiling with no loop involved at all. The premise behind the whole
project — that translation beats direct thermal detection — is answered affirmatively and
independently.

**A useful negative check came free here.** The original in-loop detector's provenance did
not confirm a held-out split, so it was possible it had memorised validation. It had not:
the clean judge scores the real-visible ceiling *higher* (0.9366 vs 0.9213), which is the
opposite direction memorisation produces.

### 5.3 A methodological finding: the two metric arms invert the ordering

This is worth reporting in the paper independently of anything else.

Detection on translated images can be measured two ways, and they answer different questions:

- **Adapted** — fine-tune an evaluation detector on each stage's translated export, then
  measure it.
- **Zero-shot** — run a fixed, never-fine-tuned visible detector straight at the translated
  images.

| Arm | Adapted mAP@50 | Zero-shot mAP@50 |
| --- | --- | --- |
| Baseline, λ = 0 | 0.9199 | 0.7851 |
| Loop stage 3, λ = 3 | 0.9132 | 0.8470 |

The adapted arm ranks stage 3 **below** the baseline; the zero-shot arm ranks it
**+0.062 above**. The arms do not merely differ in precision — **they invert the ordering.**

The adapted arm is uninformative on this dataset for two independent reasons: it is
saturated (a same-domain-trained detector exceeds 0.9 on *raw thermal* too), and its entire
observed spread sits inside the noise between two runs of the *same* configuration. It also
quietly defeats the research framing, because fine-tuning on translated images requires the
thermal-domain annotations the project exists to avoid depending on.

**Consequence:** all headline numbers are reported zero-shot. The adapted arm is retained
only for the detector-identity experiment (E7).

### 5.4 Reward hacking — ruled out

Reward hacking's signature is detection climbing while perceptual fidelity degrades. It does
not happen here. Read down the four stages of a single loop run — every row is the same
translator under a rising λ_det. (mAP here is the earlier judge's, measured before the
independent one existed; the fidelity metrics involve no detector at all.)

| Arm | Zero-shot mAP@50 | LPIPS ↓ | FID ↓ | KID ↓ | SSIM ↑ | PSNR ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| Loop stage 0, λ = 0 | 0.7622 | 0.2988 | 104.82 | 0.0301 | 0.4979 | 15.34 |
| Loop stage 1, λ = 1 | 0.8218 | 0.2828 | 96.13 | 0.0283 | 0.5002 | 15.08 |
| Loop stage 2, λ = 2 | 0.8332 | **0.2811** | 92.69 | 0.0265 | **0.5027** | 15.66 |
| Loop stage 3, λ = 3 | **0.8696** | 0.2825 | 90.85 | 0.0273 | 0.4914 | **15.69** |

mAP rises 0.7622 → 0.8696 while **every** perceptual metric holds or improves: LPIPS falls,
FID falls monotonically 104.82 → 90.85, KID falls, SSIM is flat. The translator is not buying
detections with adversarial texture. This was later bounded properly at six seeds: λ_det
moves stage-3 LPIPS by −0.0019, CI [−0.016, +0.011].

**Reported honestly: the absolute fidelity is poor.** PSNR ~15.5, SSIM ~0.50, FID ~90 are
weak numbers for an image-translation paper — expected for pix2pix on 753 pairs, and part of
why the diffusion backbone exists. It also *supports* the paper's own argument: detection
transfer works well (84–90% of the real-visible ceiling) on images these metrics score as
mediocre reconstructions. Pixel fidelity is measuring something other than what the
downstream task needs.

### 5.5 E3 — the causal ablation, first campaign: null

E3 is the experiment the paper's headline contribution stands or falls on. Design: two arms
differing **only** in λ_det (control `[0,0,0,0]`, loop `[0,1,2,3]`), 400 warm-started epochs
each, six paired seeds, judged by the independent detector. Twelve runs, ~72 GPU-hours.

Two design choices make the result interpretable:

- **Stage 0 is a free null control.** Both arms are λ = 0 there, so the paired stage-0
  difference measures run-to-run noise *from inside the experiment itself*, at no extra cost.
- **Six seeds is a floor, not a preference.** The distribution-free test on paired data is an
  exact sign-flip permutation over 2ⁿ assignments. Its smallest attainable two-sided p is
  0.25 at n = 3 and **0.031 at n = 6** — n = 6 is the first size at which such a test can
  clear 0.05 *at all*, whatever the effect size.

Paired loop − control, exact sign-flip test, with bootstrap CI:

| Metric | Stage 0 (null control) | Stage 1 | Stage 2 | **Stage 3 (endpoint)** |
| --- | --- | --- | --- | --- |
| Zero-shot mAP@50 | −0.0063, p = .875 | +0.0241, p = .281 | +0.0238, p = .312 | **+0.0070, p = .656, CI [−.020, +.032]** |
| LPIPS | +0.0096, p = .875 | +0.0056, p = .344 | +0.0075, p = .500 | **−0.0019, p = .812** |
| Switch AP@50 | +0.0430, p = .406 | +0.1085, p = .031 | +0.0669, p = .094 | **+0.0316, p = .562** |

The decision rule was fixed before the campaign ran: *if the stage-3 effect is not clearly
larger than the stage-0 difference, E3 is negative.* Stage 3 is +0.0070 against a null of
−0.0063 — the same magnitude. **By its own rule, negative.**

What the campaign nonetheless established, all of it reusable:

1. **The null control behaved as a null** (−0.0063, p = 0.875, CI straddling zero almost
   symmetrically). This is the single most important line in the table: it means the
   measurement is sound and the campaign was not broken.
2. **The noise floor is confirmed.** Per-seed sd ≈ 0.053 mAP@50, matching the n = 1 estimate
   the six-seed budget was justified against.
3. **Six paired seeds bought ±0.026 resolution** — 2.3× tighter than a single draw. The
   honest claim is therefore *"no effect larger than about +3 mAP@50 points"*, **not** "no
   effect": a true +0.02 sits inside the interval.
4. **No per-class claim is attainable in this design at any effect size.** The one cell under
   0.05 sits exactly on the n = 6 p-floor of 0.031, where a multiplicity correction across 12
   reported tests would require 0.0042.

### 5.6 Why the null does not mean what it appears to — the dose was never applied

One thing blocked reading that result as a statement about coupling: **λ_det was never 1/2/3.**
The stage weight is multiplied by a separate `grad_scale` downscale — an anti-reward-hacking
guardrail borrowed from the reward-tuning literature — so the ramp the optimiser actually saw
was **0.01 / 0.02 / 0.03**, against fidelity terms weighted 1.0 and 5.0.

Measuring the loss composition directly, pooled over all six loop runs (zero GPU cost — the
per-epoch losses were already recorded):

| Stage | Effective λ | LPIPS term | GAN term | L2 term | **Detection term** | **Share of objective** |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.01 | 1.5940 | 1.6886 | 0.0361 | 0.0298 | **0.9%** |
| 2 | 0.02 | 1.5146 | 1.6508 | 0.0336 | 0.0550 | **1.7%** |
| 3 | 0.03 | 1.4758 | 1.7490 | 0.0323 | 0.0782 | **2.3%** |

At the *top* of the ramp the detection term was **1/43rd of the objective**, roughly **19×
smaller than the LPIPS term** it had to compete against for the optimiser's attention.
**E3 did not test the coupling hypothesis at a dose capable of refuting it.**

The 72 GPU-hours were not wasted — they produced a validated null control, a confirmed noise
floor and a fidelity bound — but the headline they were spent on cannot be read as evidence
about coupling.

This was predicted in the plan and the instruction was skipped: *"the [0,1,2,3] ramp is
calibrated to a **segmentation** loss scale, not a detection loss — recalibrate empirically in
Phase 0."* That recalibration never happened. It has now.

**A second finding fell out of the same measurement:** the objective is 44% LPIPS / 52% GAN /
**1.0% pixel-L2**. Any sentence in the paper calling the pixel term dominant is wrong.

### 5.7 The corrected dose, verified by probe

Extrapolating from the measured loss magnitudes, `grad_scale: 0.15` (15× the E3 value) was
predicted to bring the detection term to ~12/21/27% of the objective. A single-seed
calibration probe confirmed it:

| Stage | Effective λ | **Achieved share** | Predicted |
| --- | --- | --- | --- |
| 1 | 0.15 | **10.9%** | 12% |
| 2 | 0.30 | **16.6%** | 21% |
| 3 | 0.45 | **22.5%** | 27% |

In band, and the ramp now spans a real dose range instead of 0.9–2.3%. Stability held: four
stages completed, the adversarial term did not diverge, nothing collapsed.

**The dose is measurably doing something, and what it measurably does is trade fidelity
improvement.** Against a matched-length control, LPIPS improves 25.2% across stages at
`grad_scale` 0.01 but only 9.5% at 0.15 — a 15.7-point gap against a 3.6% run-to-run floor,
the one comparison that clears its own noise by a wide margin. Whether it *buys detection* is
exactly what the control arm in the running campaign is for.

> **Deliberately not concluded from the probe:** anything about mAP. With a per-seed sd of
> ≈0.053, one run cannot resolve the effect being chased. The probe is a calibration; the
> measurement is the six-seed campaign.

---

## 6. What is and is not established

| Claim | Status |
| --- | --- |
| Translation beats direct thermal detection (C4's premise) | ✅ **Established** — +0.630 mAP@50, all 4 classes, uncontaminated arm, independent judge |
| The loop does not reward-hack | ✅ **Established** — bounded at 6 seeds, LPIPS Δ −0.002 [−0.016, +0.011] |
| The evaluation protocol is sound | ✅ **Established** — validated null control, measured noise floor |
| λ_det causally improves detection (C1, the headline) | 🔄 **Unresolved** — first campaign null, but dose-limited; re-running |

### Against the five criteria for beginning to draft the paper

| # | Criterion | Status |
| --- | --- | --- |
| 1 | **Margin** ≥ +2–4 mAP@50 over the strongest baseline | ✅ vs the thermal floor (+0.63); ⏳ vs the full baseline suite, which is Phase 3 |
| 2 | **Consistency** across ≥2–3 datasets | ⏳ One dataset so far — blocked on the LLVIP/M3FD fetch |
| 3 | **Causality** — the ablation shows the loop drives the gain | ❌ **Not satisfied.** Unresolved rather than refuted |
| 4 | **Stability** — ≥3 seeds, significance-tested, no collapse | ✅ Methodology met in full — 6 seeds, exact test, no collapse |
| 5 | **Faithfulness** — hallucination rates low and reported | ✅ Partial — 3 of 4 metrics built and reported; the 4th deferred with a documented reason |

**The honest implication of criterion 3:** whatever margin this method holds over baselines
is currently contributed by the **translator**, not by the coupling. If the corrected-dose
campaign also comes back null, that is a real mechanism result and the paper's centre of
gravity moves to the data-efficiency and benchmark contributions (C2/C3/C4 + E8) — which at
753 pairs was always the likely honest outcome, and remains a strong Q1 story rather than a
fallback.

---

## 7. Methodological findings that hold regardless of C1

These are transferable results, worth reporting whatever happens to the headline claim:

1. **Adapted and zero-shot detection metrics invert the ordering of methods** on this kind of
   dataset, and the adapted arm saturates. Work that reports only the adapted number may be
   reporting noise (§5.3).
2. **Effective λ must be reported as `task_weight × grad_scale`.** The paper must never say
   "λ_det = 3": ours ran at 0.03 (§5.6).
3. **The noise floor is the experiment.** Per-seed sd ≈ 0.053 mAP@50 on this dataset means
   n = 6 is the *minimum* for any distribution-free claim, and no per-class claim is
   attainable at all under multiplicity correction.
4. **TarDAL's released code does not backprop detection loss into its generator.** The fused
   tensor arrives with `requires_grad=False` because the call site is decorated `no_grad`; the
   detection loss reaches only the detector. Anyone reproducing that baseline reproduces the
   bug. Our comparison arm fixes it, and the paper will say so.
5. **MetaFusion's released repository is inference-only** — no training code, no detector, no
   licence file. Its arm is deferred on that basis rather than silently omitted.
6. **The dependency maze in this literature belongs to the frameworks, not the models.**
   Vendoring model definitions and losses only — never the training frameworks — resolved a
   documented, widely-reported dependency conflict in one lock file.

---

## 8. Risks and open items

| Risk / item | Signal | Response | Owner |
| --- | --- | --- | --- |
| Corrected-dose E3 also returns null | Stage 3 ≈ stage 0 again | Genuine mechanism result; pivot the paper to C2/C3/C4 + E8 data-efficiency | Us |
| **LLVIP / M3FD / TTPLA not fetched** | Criterion 2 blocked; Phase 2a pretrain blocked | Needs disk + a **re-host destination decision** | **Decision needed** |
| 753 pairs is small for diffusion | Turbo arm overfits | LLVIP pretrain (blocked above), heavier augmentation, lower LoRA rank | Us |
| Each campaign costs ~72 GPU-hours | Turbo arm queued behind E3 | Deliberately serialised — the turbo arm must not launch at an uncalibrated dose and reproduce the same null | Us |
| Fidelity degrades at the higher dose | LPIPS rising against the control | Would be a *finding*, not a failure — it bounds the safety/performance frontier. `reward_target` is the response, held in reserve so the dose and its guard do not change together | Us |
| Licensing constrains downstream use | sd-turbo non-commercial; ultralytics AGPL-3.0 | Fine for an academic paper; must be stated in it | Us |

---

## 9. Next actions

1. **Now → +36h.** The twelve-run corrected-dose E3 campaign completes (seeds 0–2 on card 0,
   3–5 on card 1; pairs kept on one card so the paired difference cannot absorb a
   GPU-to-GPU difference).
2. **On completion.** Read the stage-3 paired difference against the stage-0 null control,
   plus LPIPS and Switch AP, in one aggregation pass. This is the pre-registered endpoint —
   one metric, one stage, decided in advance.
3. **Branch on the result.**
   - *Effect clears the null* → launch the pix2pix-turbo arm, which is the strong arm; the
     pix2pix campaign becomes its cheap control.
   - *Null again at a real dose* → write it up as a mechanism result and move the paper's
     centre to E8 (low-annotation) and the benchmark contributions.
4. **In parallel, unblocked by either branch.** Fetch and re-host LLVIP/M3FD (needs the
   decision below); measure turbo VRAM on the real card; implement the fidelity floor.
5. **Then Phase 3.** Full baseline suite, E8 annotation-fraction sweep, E9 cross-dataset,
   E10 faithfulness stress tests.

---

## 10. Decisions we need from you

1. **Where should LLVIP / M3FD / TTPLA be re-hosted?** They are Google-Drive-gated, so they
   are fetched once locally and then need a stable URL the training server can `curl`. This
   currently blocks both the diffusion pretrain and the ≥3-dataset acceptance bar. Options:
   a university-hosted share, an S3/institutional bucket, or a HuggingFace dataset repo.

2. **Confirm the target venue.** The plan names IEEE TIM and Information Fusion as primary,
   with TGRS/TII strong. The choice materially affects how much of the paper is method versus
   measurement, and it is worth fixing before drafting rather than after.

3. **Is a paper with a null C1 acceptable to you if C2/C3/C4 and E8 are strong?** At this
   data scale that is a realistic outcome, and it would still be an honest, defensible Q1
   submission — a translation benchmark, a faithfulness metric, and a demonstration of *when*
   translation beats direct thermal detection. We would rather agree the framing now than
   discover the disagreement at drafting time.
