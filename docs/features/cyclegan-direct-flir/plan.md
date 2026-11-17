---
slug: cyclegan-direct-flir
title: CycleGAN as the direct translator on FLIR
status: in-progress
created: 2026-11-17
---

# CycleGAN as the direct translator on FLIR

## Summary

The professor's hypothesis is that **unpaired** learning fits thermal → visible translation better
than **paired** learning. The frozen detector only needs a plausible-looking object. It does not
care whether a transformer or car is painted the right colour, and possibly not whether its
structure matches the registered visible image pixel for pixel. If that is true, pix2pix's paired
L2 + LPIPS terms are spending capacity on accuracy the detector never rewards.

This feature tests the idea by putting CycleGAN in pix2pix's place in the e3 loop on **FLIR**,
with thermal as the direct input, and comparing it against the existing FLIR pix2pix campaign,
**e3f** (record 016). It runs **2 stages** (`task_weights` `[0, 0]` control and `[0, 1]` loop),
not e3f's 4, within a budget of **3–5 days** of server time. The seed count follows from the
measured wall clock.

**Standing.** This is **paper-2 exploration**, not a TIM-paper claim. The human chose that on
2026-11-17, knowing that `docs/goal.md`'s first Non-Goal ("nothing outside the publication")
reserves paired → unpaired work for paper 2. It has the same standing as the enhancer side
quest. No TIM criterion depends on it, and it must not take GPU time that a TIM criterion needs.

**How to read the result.** The read is CycleGAN vs pix2pix only. It is never CycleGAN vs raw
thermal (T), because the passive claim was closed negative on 2026-10-15. If CycleGAN beats
pix2pix but stays below T's 0.4499, the result is "unpaired beats paired", nothing more.

**The nearest prior result** is record 039 (feature `cyclegan-cascade`). There, CycleGAN ran as
stage 2 on top of a frozen pix2pix, on the custom set, seed 0. It added nothing: control −0.0005
and loop −0.0216 zero-shot mAP50. This feature is a different question (CycleGAN as the *only*
translator, on thermal, on FLIR), but that negative read is the prior.

**This is not "trained the same way".** CycleGAN's builder refuses the paired loss terms
(`_require_unpaired_objective`). So the objective, batch size (1 vs 8), learning-rate schedule
and precision (float32 vs bf16) all differ from pix2pix. The comparison is recipe A vs recipe B,
not a controlled paired-vs-unpaired test with everything else equal. Any write-up says so.

## Tech Stack / Approach

There is **no new backbone code.** Interchangeable backbones are design.md §5 invariant 3, and
`cyclegan-cascade` already shipped everything this needs:

- `src/t2o/translators/cyclegan.py` is the backbone. It is wired in through the `translator`
  union in `src/t2o/config/schema.py` (`CycleGANTranslatorConfig`, around `:425`) and one branch
  of `build_translator` in `src/t2o/translators/__init__.py`.
- `unpaired_visible`, `infrared_channels`, `max_train_images`, `subset_seed` and
  `val_loss_images` already exist in `src/t2o/config/schema.py:56-110`.
- `experiments/cascade_cyclegan_loop.yaml` is a working CycleGAN loop config and the template
  for the new configs.
- `experiments/e3_pix2pix_{control,loop}.yaml` set the e3 shape: 100 epochs per stage, coupling
  through `coupling.task_weights`, and `grad_scale` as the dose.

What gets built:

1. **Two new experiment configs**, `experiments/e3_cyclegan_{control,loop}.yaml`. They are copied
   from the cascade CycleGAN config and reshaped to e3: 2 stages, `task_weights` `[0, 0]` and
   `[0, 1]`, 100 epochs per stage to match e3f's stages 0–1, and thermal as input. The two files
   differ only in their design fields. A pinning test enforces that, modelled on
   `tests/test_cascade_cyclegan_experiments.py`.
2. **A FLIR dose probe.** CLAUDE.md requires a `scripts/loss_share.py` run (25 epochs) before
   every new backbone's campaign, with the detection term held at 20–30% of the objective. The
   custom-set probe from `cyclegan-cascade` (ADR-02: 8.2% at 0.15, raised to 0.6) used a
   different input (pix2pix RGB output, 3 channels, with identity). It does not cover FLIR
   thermal, so this feature runs a fresh probe and fixes `grad_scale` from it.
3. **A seed-0 timing and collapse probe** for both arms. It measures the per-run wall clock,
   which decides the seed count, and shows whether e3f's stage-0 collapse (F140) also hits
   CycleGAN.
4. **The comparison.** The pipeline is `t2o loop`, then `t2o faithfulness --write-back`, then
   `t2o aggregate` / `scripts/campaign_report.py`, with FLIR's 3-class primary metric
   (bicycle/car/person). Everything matches e3f: the 600-image train subset
   (`--max-train-images 600`), `--val-loss-images 153`, the same seeds, the same judge and the
   same de-rolled labels. The pix2pix side reads e3f's stage-0 and stage-1 numbers from the
   per-stage `runs/e3f-*/metrics.json` already on the server (subject to Q1).
5. **Loss curves on W&B, read before any number is trusted.** No new code is needed:
   `src/t2o/engine/trainer.py:305-321` already logs every term CycleGAN's `fit()` returns, as
   per-epoch means prefixed by stage (`train/loss_gan`, `loss_cycle`, `loss_det`, `loss_total`,
   `loss_d`, plus `val/pixel_l2`). `src/t2o/engine/loop.py:249-251` adds each stage's zero-shot
   and fidelity metrics. Every server run passes `--wandb --group e3f-cyclegan`, mirroring
   e3f's `e3-pix2pix-flir600`, and the record names the group. The self-hosted URL and API key
   come from the server's environment (`schema.py:549-551`) and are never written to a file.
6. **Faithfulness on every arm, reported whatever mAP says.** CycleGAN with no paired term is the
   setting most likely to hallucinate, and faithfulness is goal criterion C3 (legacy C2).

Server runs follow the usual rules. Runbooks stay untracked. `--workers` stays at 4–8 after the
commit-limit crash of 2026-11-10. There is no DDP: one arm per card.

## Open Questions

## Resolved

### Q1: What is the pix2pix side of the comparison?

- [x] Reuse e3f's stage-0 and stage-1 numbers (recommended): zero GPU cost, and six seeds are
  already recorded per stage in `metrics.json`. The cost: e3f's stage 1 sat inside a 4-stage
  run and was trained on full epochs, so the schedule is matched only up to stage 1.
- [ ] Re-run pix2pix under the exact 2-stage recipe used for CycleGAN: a true like-for-like on
  schedule and sampling, but it spends roughly half the 3–5 day budget on a baseline that
  already exists.
- [ ] Run CycleGAN for all 4 stages to match e3f directly: no fairness caveat, but double the
  run length, so fewer seeds fit in the budget, against the noisiest baseline in the repo.

### Q2: How should each epoch be made cheaper?

You asked for a fresh random sample of images every epoch. Over a run, N random images per epoch
for E epochs costs the same number of steps as fewer full passes over all 600. The saving comes
from the total number of steps, not from the sampling itself.

- [x] Fewer epochs over the full 600 images (recommended): the same GPU saving with **no new
  code**. CycleGAN's learning-rate decay (counted per batch in `_count_training_batches`) stays
  correct, and every image is seen equally often.
- [ ] A new per-epoch resampling setting: exactly what you described. It touches the sampler,
  `_count_training_batches`, the `unpaired_visible` permutation and per-epoch seeding in
  `src/t2o/engine/trainer.py:234-270`. It must stay hash-neutral at its default and apply to
  translator training only. This is the riskiest code change in the feature.
- [ ] A fixed smaller subset through the existing `max_train_images`: no new code, but CycleGAN
  would see fewer distinct images than e3f's 600. That confounds the comparison.

### Q3: Should thermal go into CycleGAN as 1 channel or 3?

- [x] 1 channel, with `lambda_identity: 0` (recommended): the same input pix2pix got in e3f, so
  the input is not a confound. Identity mostly preserves colour, which is what the hypothesis
  says does not matter. Unit tests cover it (`tests/test_cyclegan_translator.py:227`, `:247`),
  but no loop run has used it yet; the seed-0 probe is the first.
- [ ] 3 channels (thermal repeated) with identity on: the textbook CycleGAN recipe, and identity
  may hold down hallucination. But the input differs from e3f's, and identity on a
  thermal → visible pair has no clear meaning.

### Q4: How should the seeds be committed?

- [ ] Seed-0 probe for both arms first, then decide the seed count from the measured wall clock
  (recommended): the run length on FLIR with 2 stages is unknown (record 039's 10.5–13.5 h per
  200 epochs is the only anchor), and the probe also shows whether CycleGAN collapses.
- [x] Commit up front to 3 seeds × 2 arms (6 runs): a fixed plan that likely fits in 3–5 days on
  two cards. But it commits before knowing whether one run takes 6 h or 14 h, or whether seed 0
  collapses.
- [ ] Loop arm only, as many seeds as fit: twice the seeds on the arm that matters most. But
  without a control arm, the loop's effect cannot be separated from the backbone's.

### Q5: How many epochs per stage, given Q2 and Q4 together?

Six runs on two cards means three per card. At record 039's 10.5–13.5 h per 200 epochs, e3f's
full schedule (2 stages × 100 epochs) costs about 40 h per card, which fits the budget without a
cut. Resolved 2026-11-17, after Q1–Q4.

- [x] e3f's 100 epochs per stage, cut only if the dose probe's timing projects the 6 runs past
  5 days (recommended): an exact epoch match for Q1, and Q2's cut stays in reserve
- [ ] Fix a reduced count now (e.g. 50 per stage): a predictable wall clock, but it gives up the
  epoch match with e3f even when the budget would have allowed it

## Out of Scope

- **No TIM-paper claim.** This is paper-2 exploration (see Summary). Nothing here changes the
  six criteria, `docs/goal.md`, or the 2027-01-31 submission plan.
- **No CycleGAN-vs-raw-thermal read.** The passive claim is closed. The question is unpaired vs
  paired, not translation vs T.
- **No custom-set (e3b) arm.** FLIR only, compared against e3f. Record 039 already covers
  CycleGAN on custom as a cascade stage.
- **No new backbone code and no edits to `third_party/`.** CycleGAN already exists in `src/`.
- **No cascade.** CycleGAN is the only translator here. It never sits on top of pix2pix.
- **No retuning after a faithfulness fail.** A false-object fail is reported as a fail, as
  e3t's was (record 033): no margin retune and no extra seeds to rescue it.
