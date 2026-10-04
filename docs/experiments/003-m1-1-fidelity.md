# 003 — M1.1 fidelity metrics on M1's exports: the reward-hacking check

**Date:** 2026-08-13 · **Task:** M1.1 · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in e7c2e59 (fidelity table) and e48a4b7 (Connector) · **W&B:** not recorded · **Log:** none saved — transcribed from TASKS.md 1366-1510 @ pre-spec-migration

## Command

As written in `TASKS.md`. The `<data.yaml>` path actually passed on the server is not
recorded.

```
uv run t2o fidelity --translated runs/pix2pix-baseline/stage0/translated \
    --data <real paired data.yaml> --device cuda:0
uv run t2o fidelity --translated runs/pix2pix-loop/stage<N>/translated \
    --data <real paired data.yaml> --device cuda:0      # N = 0..3
```

## Configuration

- Post-hoc scoring of M1's two completed runs (record 002): `pix2pix-baseline` (λ_det=0,
  one stage) and `pix2pix-loop` (stages 0–3, λ_det 0/1/2/3, warm-started). No retraining.
- Scored artifact: the exported translated val split, the same PNGs M1's zero-shot arm scored,
  against the real visible frames. Pools are paired by filename stem (`export.py` writes
  `.png`, the sources are `.jpg`).
- Metrics: all five from the existing `FidelityEvaluator` (M0.5) through
  `metrics/fidelity.py::evaluate_fidelity` — LPIPS, FID, KID, SSIM, PSNR.
- Epoch budgets are cumulative and not matched: baseline and loop stage 0 at 100 epochs, loop
  stages 1/2/3 at 200/300/400.
- Evaluation set: the frozen custom val split, 153 images.

## Results

Exported val split (153 images) against the real visible frames, with M1's zero-shot mAP50
alongside, as recorded in e7c2e59:

| arm | zero-shot mAP50 | LPIPS ↓ | FID ↓ | KID ↓ | SSIM ↑ | PSNR ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| baseline, λ=0, 100 ep | 0.7751 | 0.3059 | **87.24** | **0.0210** | 0.4528 | 15.50 |
| loop stage 0, λ=0, 100 ep | 0.7622 | 0.2988 | 104.82 | 0.0301 | 0.4979 | 15.34 |
| loop stage 1, λ=1, 200 ep cum. | 0.8218 | 0.2828 | 96.13 | 0.0283 | 0.5002 | 15.08 |
| loop stage 2, λ=2, 300 ep cum. | 0.8332 | **0.2811** | 92.69 | 0.0265 | **0.5027** | 15.66 |
| loop stage 3, λ=3, 400 ep cum. | **0.8696** | 0.2825 | 90.85 | 0.0273 | 0.4914 | **15.69** |

No results file was saved; there is nothing to re-render from.

## Findings

### F06 — No reward hacking: zero-shot mAP50 climbs across the loop while every perceptual metric improves or holds

The signature of reward hacking is a detection metric climbing while perceptual fidelity
degrades. Across the loop's four stages mAP50 rises 0.7622 → 0.8696 while LPIPS falls
0.2988 → 0.2825, FID falls monotonically 104.82 → 90.85, KID falls 0.0301 → 0.0273, and SSIM
is flat. The translator is not buying detections with adversarial texture. This answers the
reward-hacking half of F04. `reward_target` stays `null` and `grad_scale` stays `1e-2` —
nothing here argues for retuning them before M2.

### F07 — λ_det's gain is not causally established: confounded with epoch budget, and inside run-to-run variance

1. **The stages are not budget-matched and not independent.** Stage 3 is the same translator
   after 400 cumulative warm-started epochs; stage 0 after 100. Every stage-to-stage
   improvement is confounded with training longer.
2. **Run-to-run variance is much larger than the mAP noise floor suggested.** The baseline
   and loop stage 0 are the same computation — same config bar `task_weights`, same seed,
   same 100 epochs — yet differ by **17.6 FID** (87.24 vs 104.82), 0.009 KID and 0.045 SSIM.
   On mAP50 they differed by only 0.0129 (F04's noise floor). 100 epochs of GAN training on
   a GPU is not reproducible run-to-run even at a fixed seed (cuDNN non-determinism in conv
   backward), and fidelity metrics expose that far more than mAP does. **The loop's whole
   14-point FID improvement is inside one sample of that variance.** The baseline has the
   best FID and KID of any arm and the worst LPIPS and SSIM: two draws from a noisy process,
   not a better or worse translator.

Resolving it needs E3 as designed (PLAN.md §11): budget-matched arms, ≥3 seeds, and an
independently-trained reference detector. Given this variance, ≥3 seeds is the minimum to
say anything at all.

### F08 — Absolute fidelity is poor, yet detection transfers: pixel fidelity measures something other than what the task needs

PSNR ~15.5, SSIM ~0.50, LPIPS ~0.28 and FID ~90 are weak numbers for an image-translation
paper — expected for pix2pix on ~850 pairs, and part of why M2's diffusion backbone exists.
On the same images, detection transfer reaches 0.87 mAP50, 94% of the real-visible ceiling
(0.9213, record 001). This supports PLAN.md §12's argument that pixel fidelity measures
something other than what the downstream task needs.

### F09 — The custom dataset declares 5 classes but only 4 are annotated: `Connector` is an unused Label Studio class

`DatasetManifest` logged `5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer']`
off the server's real `data.yaml`; PLAN.md §9 recorded 4 and omitted Connector. Connector
appears in no per-class AP table from any M1 evaluation. Resolved with the user: the class
was created in the labelling project and never used, so it has zero instances in train *and*
val. Not a split problem.

Left at `nc: 5` deliberately. Connector is index **0**, so dropping it would renumber every
other class in every label file on the server — a migration with real corruption risk for no
measurable gain. The consequences are benign:

- `v8DetectionLoss` carries one class logit that never receives positive supervision.
- `metrics/task.py::_extract_per_class_ap` omits zero-instance classes rather than reporting
  `0.0` (M0.5's decision), which is why this surfaced as an absence.
- ultralytics averages mAP over classes actually present, so no reported mAP is diluted.
  Every number in M1's gate table is a 4-class average.
- **The paper must report 4 classes**, not the `nc: 5` the manifest declares.

## Provenance caveats

- Fidelity is scored on the exported PNGs, the exact bytes both detectors saw, not the
  translator's float output. The numbers include `to_uint8`'s quantisation and are very
  slightly pessimistic relative to what the translator emitted.
- The λ_det > 0 mAP50 column is F04's contaminated judge (`optical_best_n_1.pt` is both the
  in-loop detector and the judge); this record lifts it, not re-measures it. F06 rests on the
  fidelity columns, which no detector touches.
- The 17.6 FID gap is one pair of identical-config runs, not a variance estimate. Single
  seed, 153 val images, no confidence interval.
- The source's closing paragraph (TASKS.md 1492) still reads "the experiment is not run"; it
  was written with the code in b296882, before e7c2e59 added the results above it.
- F09 was resolved in conversation with the user (e48a4b7); no instance count was saved.

## Next

Separate λ_det's effect from epoch budget and run variance: E3 as designed (PLAN.md §11) —
budget-matched arms, ≥3 seeds, an independently-trained judge. The judge comes first
(M1.2, record 004).
