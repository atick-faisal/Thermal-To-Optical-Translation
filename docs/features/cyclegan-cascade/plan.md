---
slug: cyclegan-cascade
title: Paired→unpaired cascade (pix2pix → CycleGAN)
status: in-progress
created: 2026-11-13
---

# Paired→unpaired cascade (pix2pix → CycleGAN)

## Summary

The professor's multi-stage idea is to chain a **paired** translator with an **unpaired** one.
Stage 1 is pix2pix, trained on aligned thermal/visible pairs. Stage 2 is CycleGAN, trained with
unpaired losses to move stage 1's output closer to real optical images. The earlier attempt
(`docs/features/cascade-gate/`) put a *frozen, zero-shot* enhancer (PiSA-SR) in stage 2 and
failed the faithfulness gate, passing 2 of 14 tests (record 035). That was not what the
professor asked for. This feature *trains* the unpaired stage 2.

It is a quick side quest. It is built while the registration-axis sweep finishes, and it runs
once both cards are free. There are two arms on the custom power-line set, one seed each:

- **control (cuda:0):** thermal → frozen pix2pix (`e3b-control-s0`) → CycleGAN → optical.
- **loop (cuda:1):** thermal → frozen pix2pix (`e3b-loop-s0`) → CycleGAN → optical → frozen
  detector. The detection loss is propagated back into CycleGAN.

Stage 1 already exists. It is record 007's E3 pix2pix campaign (custom set,
`grad_scale: 0.15`). This feature adds only the stage-2 CycleGAN on top.

It is framed as **paper-2 exploration** (Q1). It makes no claim in the TIM paper:
`docs/goal.md`'s Non-Goals reserve the paired→unpaired cascade for a second paper unless it
passes a faithfulness gate fixed before its first score.

## Tech Stack / Approach

- **Stage 1 is precomputed, not wrapped** (Q4, Q8). `t2o export --checkpoint`
  (`src/t2o/cli.py:110-112`, `src/t2o/engine/export.py:137`) already writes a translator's
  train and val translations plus mirrored labels and a `data.yaml`. Run it once per arm on that
  arm's frozen stage-1 checkpoint (`runs/e3b-{control,loop}-s0/stage3/translator_last.pt`;
  filename from `src/t2o/engine/trainer.py:78`) to get two stage-1 trees. Each tree is then
  stage 2's "thermal" input. No composite translator, no new checkpoint loading, and the data
  contract is unchanged: it is just another dataset tree. Open detail for the task list: the
  exported tree must still let the loader find each frame's real visible image, either copied
  alongside or reached through the original manifest.
- **New backbone `cyclegan`** (Q7: plain CycleGAN), following the existing backbone pattern:
  - a wrapper in `src/t2o/translators/cyclegan.py`, modelled on
    `src/t2o/translators/pix2pix.py:57-169`, whose `fit()` adds the detection term the same way
    (`:161-164`);
  - a `CycleGANTranslatorConfig` in the `src/t2o/config/schema.py` translator union, next to
    `Pix2PixTranslatorConfig` (`:297-337`);
  - one branch in `build_translator` (`src/t2o/translators/__init__.py:46-99`).
- **Networks are reused, not written.** `define_G`, `define_D` and `GANLoss` come from
  `third_party/pix2pix/networks.py:132,163,209`, the same upstream repo as CycleGAN. `ImagePool`
  is not vendored. Either add it under `third_party/` at a pinned commit, or leave it out with
  the deviation cited in the module docstring. `third_party/` is never edited in place.
- **The translator protocol stays as it is.** `src/t2o/translators/protocol.py:29-53`:
  `fit(batch, task_loss, task_weight)`, and `translate()` stays connected to the gradient graph,
  so in the loop arm the detection loss reaches the forward generator (stage-1 output → optical).
- **Unpaired sampling** (Q6). `TranslationPairDataset` (`src/t2o/data/dataset.py:51-69,190-366`)
  pairs input and visible by filename. CycleGAN's real-visible sample is drawn from the custom
  set's own visible train frames, independently of the input (a seeded shuffle). Detection
  labels stay on the input's frame, so the loop arm's detection loss is unchanged.
- **Experiment configs** `experiments/cascade_cyclegan_{control,loop}.yaml`. One stage of 200
  epochs (Q9): control `task_weights: [0.0]`, loop `task_weights: [λ]`. Following the E3
  pattern, they differ only in `coupling.task_weights` and `runtime.name`; a test pins that, like
  `tests/test_e3_experiments.py::test_control_and_loop_configs_differ_only_by_design`. The
  stage-1 `data.manifest` path also differs per arm (Q5), so that test's allow-list has to name
  it.
- **Dose calibration comes first.** As `CLAUDE.md` and cascade-gate ADR-02
  (`docs/features/cascade-gate/decisions.md:200-203`) require, the new backbone runs a 25-epoch
  `scripts/loss_share.py` probe on the loop config, and sets `grad_scale` and λ so the detection
  term is 20–30% of the objective.
- **Scoring reuses what exists:** `t2o aggregate` on `zero_shot.map50` and `fidelity.lpips`, and
  `t2o faithfulness` for a hallucination read. The result goes to a `/log-experiment` record
  that states the one-arm-per-card deviation (Q3) and the confound from matched stage-1
  checkpoints (Q5).
- **Testing:** every piece runs on `tmp_path_factory` synthetic pairs on CPU in seconds. Any
  end-to-end loop run gets the `slow` marker.
- **CycleGAN's losses and layers** (CYCLEGAN-CASCADE-03). Adversarial weight is the shared
  `loss.gan`, both directions summed and logged as `loss_gan`; cycle (10) and identity (0.5)
  weights are `CycleGANTranslatorConfig` fields at upstream's defaults. Paired `loss.l2`/
  `loss.lpips` mean nothing once -02 decouples the train target, so `build_translator` refuses
  `cyclegan` with either above 0 or with `loss.gan` at 0, as it does for ModTr. Upstream
  CycleGAN's `instance` norm and `lsgan`, and unconditional discriminators (D_B on visible, D_A
  on the stage-1 domain). `ImagePool` is vendored from upstream's `util/image_pool.py` at the
  same pinned `2a7afba`, beside `networks.py`, pool size 50. `fit` also returns `loss_cycle` and
  `loss_idt`, and `loss_share.py`'s display keys gain both. The identity term needs equal
  channel counts, so ADR-01's loader change (-06) lands first. Upstream's identity weight
  multiplies the cycle weight (`models/cycle_gan_model.py:162,165`), so 0.5 is an effective 5,
  and `build_translator` refuses identity above 0 at `infrared_channels: 1` (upstream's assert,
  `:87-88`). The D names here follow the paper (D_X judges domain X), the reverse of upstream's
  `netD_A`; `fit` keeps upstream's order, one Adam over both generators and then one over both
  discriminators fed detached from the pools, so the detection gradient never reaches a D.
- **Dose and probe** (CYCLEGAN-CASCADE-04). The loop arm's λ is fixed at 1.0, so `grad_scale`
  alone sets the dose, and -05 changes it in both files (the differ-only test holds them
  equal). CycleGAN's schedule starts decaying halfway through whatever epoch count it is given,
  so the probe runs at `--epochs 50` and is read with `loss_share.py --first-epochs 25`, the
  window that matches the campaign's own constant-rate start. The runbook stops after the probe
  until -05 is pulled; both campaign runs start after that. The record also states that stage 2
  trains on pix2pix's in-sample train outputs but is scored on its out-of-sample val outputs.

## Open Questions

## Resolved

### Q1: How is this side quest framed?

- [x] Paper-2 exploration (recommended) — a quick side quest with no TIM claim. One seed per arm is acceptable, and no gate ADR is needed.
- [ ] A second faithfulness-gate attempt for the TIM paper — would need a new ADR fixing margins before any score (like cascade-gate ADR-04) and paired seeds, which one seed per arm cannot satisfy.
- [ ] Drop it until the registration sweep finishes — it would cost nothing now, but the human wants to test the idea while the cards are busy.

### Q2: When do the runs go on the GPUs, given the registration sweep holds both cards?

- [ ] Share the cards with the sweep now, with `runtime.workers` lowered (recommended) — matches "in the meantime". CycleGAN plus a yolo11n fits next to a pix2pix-sized sweep job on 40 GB. Fewer loader processes protect the RAM limit behind the 2026-11-10 crash (error 1455).
- [x] Wait for the sweep to finish (~53 h on cuda:0, ~62 h on cuda:1) — no risk to the sweep, but the side quest stalls for about 2.5 days. *Human's answer: the sweep has already run more than 2 days; cuda:0 is free and cuda:1 is expected free by the morning of 2026-11-14. Build tonight, start both runs then on a clean slate.*
- [ ] Pause the sweep, run this, then resume — the cards are free, but it delays the sweep, which serves the January paper.

### Q3: How are the two arms placed on the GPUs?

- [x] One arm per card, as asked: control on cuda:0, loop on cuda:1 (recommended) — fastest wall clock. It breaks record 007's split-by-seed rule, but with one seed there is no paired test for a GPU difference to bias. The record states the deviation.
- [ ] Both arms on one card, one after the other — keeps the house rule, but doubles wall clock and leaves the other card's spare capacity unused.
- [ ] Both arms on one card, in parallel — keeps the rule at the same wall clock, but puts three jobs on one card next to the sweep, which is the worst case for RAM.

### Q4: Is stage 1 (pix2pix) frozen or trainable during stage 2?

- [x] Frozen; only CycleGAN trains, and the detection gradient reaches stage 2 only (recommended) — this is "one more stage on top", it isolates what stage 2 adds, and it allows the precomputed-input shortcut in Q8.
- [ ] Trainable, with stage 1 fine-tuned under its paired loss plus the detection loss — closer to end-to-end joint training, but both stages move, so a gain cannot be attributed to stage 2. It also needs the composite translator.
- [ ] Frozen in control, trainable in loop — gives the loop the most freedom, but the arms then differ in more than `task_weights`, which breaks the control/loop design.

### Q5: Which stage-1 checkpoint feeds each arm?

- [ ] `e3b-control-s0` stage 3 for both arms (recommended) — the only difference between arms is stage 2's detection loss, and control stays detector-free end to end.
- [ ] `e3b-loop-s0` stage 3 for both arms — starts from the stronger stage 1 (record 007's positive result), but the control arm then already carries detector influence.
- [x] Matched: control ← `e3b-control-s0`, loop ← `e3b-loop-s0` — tells the cleanest "detector everywhere vs nowhere" story, but the arms differ in two places, so stage 2's share is not separable.

### Q6: Where do stage 2's unpaired visible images come from?

- [x] The custom set's own visible train frames, shuffled independently of the input (recommended) — no new data, same domain as the labels and the detector, and no licence work. Caveat: the pairs exist, so "unpaired" is a property of the loss, not of the data.
- [ ] An external visible power-line image set — truly unpaired and possibly more data, but nothing is in the repo, it needs a licence check under `goal.md` Constraints, and it adds a domain shift.

### Q7: Which CycleGAN?

- [x] Plain CycleGAN built from the vendored `third_party/pix2pix` networks (recommended) — same upstream as stage 1, fits on a 40 GB A100, and adds no new dependency.
- [ ] CycleGAN-Turbo — stronger on paper, but needs about 48 GB for unpaired training (`docs/design.md:536`), more than one card can give.

### Q8: How does stage 2 get its stage-1 input?

- [x] Precompute stage 1: export each arm's frozen stage-1 translations of train and val into a YOLO tree with the same labels, then run the `cyclegan` backbone on it as if it were thermal (recommended, needs Q4 = frozen) — no composite translator and no checkpoint loading. The data contract stays intact, since it is just another dataset tree.
- [ ] A composite `cascade` translator that loads `translator_last.pt` and wraps stage 1 plus CycleGAN — needed if stage 1 trains (Q4). It means new pretrained-weights config fields and a second checkpoint path into `build_translator`.

### Q9: What training schedule do the two runs use?

- [ ] E3's schedule: `task_weights` `[0,1,2,3]` × 100 epochs, with the evaluation-detector fine-tune each stage (recommended) — comparable to record 007, and stage 0 is a free null control inside each run.
- [ ] Two stages, control `[0,0]` and loop `[0,λ]` × 100 epochs — about half the GPU time, but there is no λ ramp and it is not comparable to record 007.
- [x] One stage of 200 epochs, CycleGAN's usual length — the quickest to read, but it drops the stage-0 null control entirely.

### Q10: What learning-rate schedule does CycleGAN train with?

- [x] Upstream's: constant for the first half of the run, then linear decay to zero over the second (recommended) — this is what Q9's "CycleGAN's usual length" means upstream (`n_epochs` 100 + `n_epochs_decay` 100). Built as ModTr does (`translators/modtr.py:65-114`): the step count from `_count_training_batches`, the schedule stepped per batch, and both optimizers and schedules saved through `get_extra_state`, so a resume continues the decay instead of restarting it.
- [ ] Constant rate throughout, like this repo's pix2pix — less code, but the exported final checkpoint comes from a GAN still at full LR, a resume resets Adam's moments, and the deviation must be cited in the module docstring.

### Q11: What batch size do both arms train at?

- [x] Batch 1, upstream CycleGAN's (recommended) — keeps Q9/Q10's upstream schedule at upstream's 120k updates over 200 epochs, and removes the risk of running out of memory with four fp32 networks plus the detector. Estimated at about twice the wall clock of batch 8. *Human's answer, 2026-11-14.*
- [ ] Batch 8, E3's — faster and the same as the pix2pix campaigns, but only ~15k updates, with the decay starting at 7.5k, so the GAN may stop before it settles.
- [ ] Batch 4 — ~30k updates, but it matches neither upstream nor E3 and needs its own justification in the record.

## Out of Scope

- **Any claim in the TIM paper.** Per `docs/goal.md` Non-Goals, the paired→unpaired cascade is
  paper 2. Entering this paper would need a separate gate feature with margins fixed before the
  first score, by 2027-01-10.
- **More than one seed per arm**, and any significance test on the result. This is a reading,
  not a measurement.
- **Training or fine-tuning stage 1.** It stays frozen at record 007's checkpoints.
- **CycleGAN-Turbo, StegoGAN, CUT/UNSB** or any other stage-2 candidate.
- **Public datasets (FLIR, MSRS).** The custom set only.
- **Editing `third_party/`, `docs/goal.md`, `docs/design.md`, or any existing record.**
- **Touching the registration-axis sweep.** These runs start only after it frees the cards.
