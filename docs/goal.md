# Thermal→Visible Translation with Detection-in-the-Loop

## Problem

Thermal images of power-line components are hard to annotate. A component with no
temperature difference from its background barely shows; different components look alike
without colour and texture; and boundaries are unclear, so every bounding box is a guess.
Labels drawn on a paired visible photo can be moved onto the thermal image, but only where
registered pairs exist, and only as accurately as the registration — itself an open problem
for small, thin components. And some surveys exist only in thermal: high-quality thermal
video shot from a thermovision car, with no optical counterpart, sits unused because it
cannot be annotated.

Translation closes that gap differently. A translator needs registered pairs once, to train;
after that it needs only a thermal image. Its output is an optical representation that any
visible-trained detector can read — one trained on any visible data, including detectors and
classes that did not exist when the translator was trained — and that an inspector can read
too. Detection is the primary use. Overlaying thermal faults on the translated image is a
second use, which this repo does not build.

The gap is large. A detector trained on visible photos does not transfer to the thermal
image of the same scene: on the 153 validation pairs it scores 0.1887 mAP50 on raw thermal
against 0.9213 on the real visible photo. A translator trained for image fidelity is not
trained for the detector that reads its output — and one that *is* trained for a detector
can learn to invent or erase components, which here are safety-relevant.

Training a translator against a frozen detector is not new: HalluciDet (WACV 2024) and ModTr
(ECCV 2024) do it, with detection loss alone, producing detector-friendly rather than
visible-faithful images. What is open is whether a *visible-faithful* translator can carry
the detector loss, whether its gain transfers to detectors outside the loop, and *when* the
translation route beats direct thermal detection or registration-based label transfer.

Our own power-line set is small (753 pairs) and unpublished, so it cannot carry the paper
alone: a claim that holds only on data nobody else can see is not one a reviewer can check.
Every claim is therefore benchmarked on public paired thermal–visible datasets as well.

This repo is an instrument for answering one question with one defensible results table:

> When does a visible-faithful thermal→visible translator, trained against a frozen visible
> detector, let visible-trained detectors work on thermal-only imagery better than direct
> thermal detection or registration-based label transfer — and does it do so without
> inventing or erasing safety-relevant components?

## Success Criteria

Drafting begins when **all six** hold:

- **Margin** — on the held-out power-line test split, the loop beats the strongest baseline
  that keeps the visible detector frozen (raw thermal; the same translator without the loop;
  ModTr and HalluciDet trained on the same data; zero-shot Grounding DINO) by ≥ +2 mAP@50,
  with a 95% bootstrap CI over test images that excludes zero and a non-negative mAP@50:95
  delta. If the validation split shows a CI at the test split's size cannot resolve +2,
  Margin becomes non-inferiority to ModTr, with Transfer and Faithfulness carrying the claim.
- **Transfer** — the loop's gain over the no-loop translator holds for at least one visible
  detector from a different family, never used in the loop, across ≥3 seeds.
- **Consistency** — on ≥2 of the 4 datasets, the loop beats its own no-loop control
  (sign-flip test, ≥3 seeds). Whether translation beats raw thermal is *not* a criterion: it
  is C2's measured outcome, reported for every dataset, failures included.
- **Causality** — the λ_det ablation (E3) shows the loop drives the gain, seed-stable.
- **Stability** — ≥3 seeds, significance-tested, no collapse.
- **Faithfulness** — insertion and deletion rates (objects invented or erased), scored
  against ground truth by a detector family never in the loop, with CIs, are no worse than
  the no-loop translator's within a margin fixed before the test split is read. Reported for
  every arm.

Alongside them:

- **Reproducible on public data.** Every headline number has a public-dataset counterpart on
  frozen, committed splits, so a reviewer can rerun it without our data.
- **C1** — visible-faithful detection-in-the-loop translation: paired reconstruction plus a
  frozen detector's loss, for pix2pix and one-step pix2pix-turbo, positioned as a successor
  to HalluciDet and ModTr.
- **C2** — a map of *when* translation pays, along three axes: headroom (per class: the gap
  between a visible ceiling and a thermal-trained detector, and between that detector and the
  visible detector on raw thermal), annotation budget (the thermal labels a direct detector
  needs to match the loop, E8), and registration error (how fast label transfer and
  translation each degrade as alignment worsens). Label transfer is compared with the visible
  detector trained both on the pairs and on visible data the translator never saw.
- **C3** — an object-level insertion/deletion audit for translators trained against a
  detector.
- **C4** — a power-line testbed and reproducible protocol: frozen splits, and per dataset a
  bracket of visible ceiling, raw-thermal floor, thermal-supervised and label-transfer
  references. If the second power-line set (below) is annotated, it is released as the
  public power-line benchmark, with a cross-camera split (train on one camera model, test
  on the other) and a cross-country test against the Brazilian set.
- **A decision-level fusion experiment** combining the translation route with the
  label-transfer route.
- A submission-ready manuscript for **Information Fusion**, framing the loop as cross-modal
  knowledge fusion.

## Non-Goals

- **Nothing outside the publication.** The goal is one paper: a translation → detection
  loop, with the emphasis on power-line components and benchmarking on public datasets.
  Work that does not serve that paper is a non-goal, however useful.
- **Not a product, not an inspection tool.** No inference service, deployment tooling, UI,
  labelling tools, real-time optimisation or edge deployment. If it does not feed the
  results table or its defence, it does not belong here.
- **No thermal fault detection or overlay tool.** Overlaying faults on translated images is a
  use the translations serve, not something this repo builds; detecting the faults
  themselves is out of scope too.
- **No "first" claims.** Training a translator against a frozen detector (HalluciDet, ModTr)
  and reusing a frozen visible detector on thermal (ModTr) are prior art.
- **Not beating supervised thermal detection at full annotation.** Measured: 0.93 mAP50 at
  600 labels against the loop's 0.85. Conceded.
- **No public-benchmark leaderboard.** Public datasets test the mechanism and map its
  limits. Fusion methods that need both images at inference are not competitors.
- **The 753-pair custom set is never released.** Splits, configs and code are.
- **Not solving registration.** Cross-modal registration is the sister project's problem
  (Thermal-Image-Registration). Here it is an input, and its error is one of C2's axes.

## Constraints

- **Two machines, git as the only transport.** A Mac (CPU/MPS) with no dataset and no
  detector weights; a native Windows Server 2022 box with 2× A100 40 GB and no WSL2. No
  shared filesystem: configs reach the server by `git pull`, run outputs (`runs/`) stay
  there. W&B is self-hosted.
- **Native Windows rules.** No DDP (no NCCL): one experiment per GPU, and every method must
  train on a single 40 GB card. Assume no triton, xformers or `torch.compile`. Spawn, not
  fork. No bash launchers, no symlinks.
- **Every component runs end-to-end on a synthetic fixture built at test time**, on CPU, in
  seconds, as a pytest.
- **Public benchmarks, adapted and frozen** (`splits/*.json`: sorted stems + hash):
  FLIR-aligned (4,129 / 1,013), M3FD (3,369 / 831) and MSRS (1,083 / 361; daytime-only
  `msrs-day` 536 / 179) train / val pairs, fetched by `scripts/fetch_datasets.py`. LLVIP is
  dropped as a benchmark: people only, all warm, mostly low-light, so it cannot test thermally
  passive objects. FLIR ships no thermal labels; they are mirrored from visible and de-rolled.
  No public dataset provides registered thermal–visible pairs of power-line components with
  boxes: CPLID and TTPLA are visible-only, HIT-UAV is thermal-only (people and vehicles),
  Yetgin & Gerek states no pairing at 128×128, and VITLD (~400 registered transmission-line
  pairs, conductor masks only) is available on request.
- **Custom data.** ~850 annotated, registered 640×480 FLIR pairs, taken in Brazil.
  Experiments touch only 600 train / 153 val; the ~100 held-out test pairs are read only at
  reporting time. Report "753 train+val of 853" and 4 classes — the manifest
  declares `nc: 5`, but `Connector` (index 0) has no instances. Boxes were drawn on the
  visible images and moved to thermal with the sister project's registration, so the
  thermal-trained baseline trains on transferred labels.
- **Second power-line set.** ~2,500 paired images taken in Bangladesh, releasable, not yet
  labelled, from two camera models: one with higher thermal and lower optical resolution,
  one with lower thermal and much higher optical resolution — a built-in cross-camera test,
  and against the Brazilian set a cross-country one (different grid hardware and scenery).
  Annotating it is about a week of detector-assisted work; its test-split labels must be
  checked by hand. **Atick provides it** (images, registration and labels); no task depends
  on it until then.
- **`dataset/` is never committed.** The remote is public; the pairs are unpublished
  research data.
- **ultralytics is pinned `>=8.4.108,<8.5`** — the Detect head's output format changed
  between 8.3 and 8.4.
- **One `uv.lock` for both machines.** `cpu` and `gpu` are conflicting extras; exactly one
  is chosen at sync time.
- **`third_party/`** stays at pinned commits, never edited in place.
- **Licences constrain downstream use:** sd-turbo and CUT are non-commercial; ultralytics
  is AGPL-3.0.
- Python ≥3.12, `src/` layout, pyright `standard`, ruff line-length 100.
