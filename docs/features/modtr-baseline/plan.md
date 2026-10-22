---
slug: modtr-baseline
title: ModTr baseline — the non-inferiority anchor for Margin
status: in-progress
created: 2026-10-22
---

# ModTr baseline — the non-inferiority anchor for Margin

## Summary

Margin, one of `docs/goal.md`'s six drafting criteria, can no longer be met as "+2 mAP@50 over
the strongest baseline". On 153 validation images the 95% CI is too wide to resolve +2 points
(F189, F190, record 027). Under `goal.md:52-53`, Margin therefore becomes **non-inferiority to
ModTr**, with Transfer and Faithfulness carrying the claim. ModTr does not exist in this repo, so
Margin is the one criterion that cannot move.

This feature builds ModTr (Medeiros et al., ECCV 2024) as a translator backbone in `t2o`. It trains
against the same frozen in-loop `yolo11n`, on the same transferred boxes, as the loop. Its exports
go through the single evaluation path, so they get the zero-shot `yolo11s` Margin score and the
RT-DETR Faithfulness rates "for every arm" (`goal.md:65`). The feature also fixes the
non-inferiority margin δ and its test **before the first ModTr number exists**. Margin cannot be
scored without δ, and no document defines it today.

Why this is a fair comparison, and why it matters for the paper: ModTr is the closest prior art.
It is a translator trained only on a frozen detector's loss
(`docs/research/thermal-translation-gap-review/notes/competing_paradigms.md:169-185`). Trained on
the transferred boxes, it is "literally our setting with a different translator" (`:183`). Once
δ is fixed, this comparison either supports or rules out the paper's positioning of the loop as
ModTr's successor (C1).

## Tech Stack / Approach

**A new backbone through the existing extension point.** pix2pix_turbo (M2a) was added the same
way:

- a `MODTR` member in the `Backbone` StrEnum, and a `ModTrConfig` in the `TranslatorConfig`
  discriminated union (`src/t2o/config/schema.py:41-50`, `:372-377`);
- one branch in `build_translator` (`src/t2o/translators/__init__.py:41-85`);
- a wrapper `src/t2o/translators/modtr.py` that satisfies the `fit`/`translate` protocol
  (`src/t2o/translators/protocol.py`). `translate` stays grad-connected. The wrapper is modelled on
  `src/t2o/translators/pix2pix.py:57-169`: it owns the optimizer and device placement, builds a
  loss term only when its weight is above 0, and adds `task_loss` the same way (`:161-164`).

**Upstream network vendored, never edited.** ModTr's transformation network goes under
`third_party/` at a pinned commit, network definitions only, the way pix2pix was vendored
(design.md §7, `:213-250`). Its training framework and its torchvision detectors are not used.
The first task is a spike that reads upstream at a pinned commit and answers six things:

- the architecture;
- whether the output is fused with the input;
- the dependencies;
- the licence;
- the training recipe;
- the memory footprint on one 40 GB A100 under native Windows (no DDP, no triton or
  `torch.compile`, spawn not fork: `goal.md:115-121`, design.md:60-80).

If vendoring needs a new package, that is a dependency conversation before any code.

**The objective already exists.** The detection term is `yolo11n`'s `v8DetectionLoss` with
`grad_scale` (`src/t2o/coupling/detection_loss.py`, `coupling/schedule.py`). The schema already
allows a detection-only stage. `src/t2o/config/schema.py:162-192` only requires non-negative
weights; the "stage 0 must be λ=0" rule is a comment, not a check.

**The same boxes by construction.** The loop's training boxes are the labels paired with the
visible image (`src/t2o/data/dataset.py:300`). A ModTr arm run through this loader trains on
exactly the loop's boxes, which is the fair setting (`competing_paradigms.md:210`). margin-gate's
ADR-01 binds the detector: ModTr trains against the same in-loop `yolo11n`
(`docs/features/margin-gate/decisions.md:51-53`). The paper must say that this ModTr is
reimplemented with a YOLO loss, the way design.md §11 discloses TarDAL.

**Evaluation and scoring are reused, not rebuilt.**

- `src/t2o/engine/loop.py:143-315` exports, then runs the zero-shot reference, fidelity and the
  judges. A new backbone gets all of this unchanged.
- An experiment is a config file. ModTr gets `experiments/` YAMLs copied from
  `experiments/e3_pix2pix_loop.yaml`.
- The Margin score reuses `scripts/margin_resolvability.py`'s paired image bootstrap (record 027),
  with ModTr's exports in the comparison arm. What is new is the non-inferiority decision, "lower
  bound > −δ", read against the δ fixed in this feature.

**Tests.** `tests/test_modtr_translator.py` runs on CPU synthetic pairs from `tmp_path_factory`.
An experiments test, in the style of `tests/test_e3_experiments.py`, loads the new YAMLs. No ModTr
weights or dataset are needed on the Mac.

**Runs are not tasks.** The ModTr campaign is a server run between two tasks. It needs at least
3 seeds, because F192 shows seed spread as large as one seed's image noise. Its result goes
through `/log-experiment`.

**What the wrapper owns beyond the protocol (ADR-01, ADR-04).**

- Upstream's optimiser: AdamW at `train.lr` with torch's default weight decay (0.01), gradient
  accumulation 2 with the loss halved as Lightning does, and a per-step cosine stepped once per
  optimiser step. `T_max` comes from Q8.
- `fit()` therefore steps the optimiser on every second call. That bends the protocol's "one
  full optimisation step", and the module docstring says so. The wrapper cannot see epoch ends,
  so unlike Lightning it never flushes a half-accumulated gradient at the end of an epoch; that
  deviation is named too.
- The AdamW state, the scheduler state and the accumulation counter go in `get_extra_state`, as
  `pix2pix_turbo.py:358-394` does, so a resume continues the cosine instead of restarting it at
  full lr.
- `translate()` repeats thermal to 3 channels and returns `clamp(x * h(x), 0, 1)`.
- ModTr has no reconstruction term. `build_translator` raises `ConfigError` if any `loss` weight
  is non-zero or any `coupling.task_weights` entry is 0; otherwise the defaults (`lpips: 5.0`)
  would be silently ignored.
- `ModTrConfig` has one field, `pretrained`, so tests build the network without downloading
  ImageNet weights. `train.amp` is ignored, as pix2pix ignores it.

**Scoring Margin.** The non-inferiority bootstrap draws n-out-of-n over the whole val pool;
`paired_image_bootstrap`'s `n=100` was sized for a test split that no longer exists. Arms are
keyed by seed and paired seed for seed under one `--seeds` list, so ModTr's campaign seeds must be
a subset of every loop campaign it is scored against: 0, 1, 3 covers e3b (0–5) and e3t (0, 1, 3).
δ is not a CLI flag; the script uses ADR-02's default and writes δ to the CSV. The δ boundary is
tested on `non_inferior` directly; bootstrapped test arms get clear-cut shifts.

**The calibration run (ADR-04).** The inner split is a seeded 80/20 copy of the training pairs,
written beside `data/budget.py` into a new tree with its own `data.yaml` (absolute `path:`, a
non-empty output folder refused). The calibration is then an ordinary `t2o loop --no-detector`
against that manifest, so validation is not in it and Q8 counts 480 images unchanged. A
config-level split was rejected because `Trainer`, `export_translated` and
`_count_training_batches` would each need the same cut. A `runtime` flag (hash-excluded;
ultralytics' validator never reseeds global RNG) makes `Trainer` export its validation split and
score it with the in-loop detector through `evaluate_detector` each epoch. It appends rows to a
per-stage file, because `metrics.json` is written only at stage end and a resumed stage would drop
the epochs before the crash. The trigger works in mAP points (×100), uses full windows only,
breaks ties to the earliest window, and sets the budget to the best window's last epoch index
plus 1.

## Open Questions

### Q8: Where does ModTr's cosine `T_max` come from?

ADR-04 keeps upstream's per-step cosine, `T_max = epochs × train batches per epoch`.
`build_translator(config)` runs before any `Trainer` or loader exists, so the wrapper cannot
see the batch count.

- [x] Count it in `build_translator`: build the training `TranslationPairDataset` from the
      manifest and take `ceil(len / batch_size) × epochs_per_stage × len(task_weights)`
      (recommended) — no new field, no protocol change, and it cannot go stale: the
      calibration run's 480 images and the campaign's 600 each get their own `T_max`
      automatically. Cost: `build_translator` now reads the manifest.
- [ ] A `ModTrConfig` field holding `T_max`, set by hand per YAML — the most explicit, and the
      hash carries it. But it goes stale silently when the dataset, batch size or epochs change,
      and the calibration and campaign configs need different values.
- [ ] A `Trainer` hook that hands the translator its loader length before training — exact, and
      it could flush accumulation at epoch end the way Lightning does. But it moves the
      `Trainer`/`Translator` boundary for one backbone, and the loop's stage count still has
      to reach it.

### Q9: What does "resumable" mean for a bootstrap that needs every seed at once?

- [x] Not resumable: compute everything, then write the per-seed rows and the summary row once
      at the end (recommended) — a rerun repays about 6 val runs, roughly 20 minutes, and a
      joint CI cannot resume from CSV rows without cached records.
- [ ] Cache each (arm, seed)'s `ImageStats` on disk, so a resume skips val runs whose cache
      exists and always reruns the cheap bootstrap — a real resume, but a new on-disk record
      format to own, and the cache cannot tell when an export or a `labels.cache` changed under
      it, the staleness CLAUDE.md warns about.

## Resolved

### Q1: Which training regime counts as "ModTr"?

- [x] Upstream's recipe, reimplemented: one detection-only stage, ModTr's network and its
      input-fusion operator, ModTr's optimiser and schedule where the spike finds them
      (recommended) — the closest thing to the published method. Only the detector changes, which
      ADR-01 forces. Reviewers can recognise it as ModTr.
- [ ] The repo's staged 4-stage schedule with every reconstruction weight at 0 — reuses the
      loop's exact schedule and epochs, so only the network differs. But it is not ModTr's
      recipe, and a reviewer can call it a weakened ModTr.
- [ ] ModTr's network plus a reconstruction term — this is our loop with a smaller network. It
      tests a different question (generator capacity) and stops being the prior-art baseline.

### Q2: If upstream's network cannot be vendored cleanly, what is the fallback?

- [x] Reimplement the network from the paper, in `src/t2o/translators/`, with a module docstring
      citing the paper's section and noting the deviation (recommended) — ModTr's network is small
      (a "small transformation network"), so a faithful reimplementation is cheap, and the paper
      already has to disclose a YOLO-loss reimplementation.
- [ ] Stop and reconsider the feature — the safest for fidelity to the original, but it leaves
      Margin blocked with no alternative anchor.
- [ ] Use the vendored pix2pix generator, trained on detection loss only — zero new code. But the
      arm becomes an ablation of our own loop, not ModTr, and Margin as `goal.md` words it would
      still be unmet.

### Q3: What is the non-inferiority margin δ?

All numbers below are already measured in record 027, before any ModTr number exists. That is
what makes fixing δ now honest rather than a guess. On the custom set:

- the loop's mean gain over its own no-loop control is **5.12 mAP@50 points** (sd 3.84, six seeds);
- the loop sits about **68 points above raw thermal**;
- the image-level 95% half-width is about **4.9 points** at n = 100 (about 4 at n = 153).

- [x] **δ = 5 mAP@50 points, absolute** (recommended) — loose but defensible. It is about the
      loop's own measured gain over its no-loop control. It is also under 8% of the ~65-point
      gain over raw thermal, so a passing loop keeps at least ~92% of ModTr's improvement over
      the raw-thermal floor. And it sits just above the image noise, so a loop that truly ties
      ModTr can pass on 153 images.
- [ ] δ = 2.5 points (half the loop's gain over no-loop, the "preserve 50%" convention) — the
      stricter, more conventional rule. With a ~4-point half-width, though, a loop that truly
      ties ModTr fails it most of the time: it is close to unwinnable on 153 images.
- [ ] δ as a fraction of ModTr's own score (e.g. 10% of ModTr's mAP@50) — reads naturally in a
      table, but δ then depends on the arm being tested and is only known after ModTr runs,
      which a reviewer will read as post hoc.
- [ ] Fix δ after ModTr's first seed — the most information, but it is exactly the post-hoc
      choice a pre-set margin exists to prevent. Rejected for the paper's sake.

### Q4: Where is δ recorded?

- [x] An ADR in this feature's `decisions.md`, committed before the first ModTr server run
      (recommended) — `goal.md` already delegates this kind of number. Faithfulness's margin is
      also "fixed before the final score" without being written into `goal.md`. A dated commit
      before the run is the evidence a reviewer needs.
- [ ] A `goal.md` amendment, as a conversation — the most visible place. But `goal.md` is
      immutable by design, and δ fills a parameter it left open rather than changing a criterion.

### Q5: Over what unit is the non-inferiority interval computed?

- [x] ADR-01's unit: a paired 95% bootstrap over validation images, on mAP averaged across
      seeds, with the per-seed deltas and their sd reported beside it (recommended) — consistent
      with how margin-gate already defined Margin's interval and with `goal.md`'s wording. Seed
      spread is shown, not hidden.
- [ ] A hierarchical bootstrap over seeds and images — the most honest about F192's seed spread.
      But it is wider: with 3 seeds, a tied loop likely fails even δ = 5. It would also reopen
      ADR-01.
- [ ] The seed-level sign-flip test (design.md:660-666) — matches the rule for seed intervals.
      But it tests superiority, not non-inferiority. With 3–6 seeds it cannot express a margin
      well, and ADR-01 already rejected it for Margin.

### Q6: Should `scripts/loss_share.py` be run for ModTr?

The repo rule (CLAUDE.md) is to re-run it for every new backbone and keep detection at 20–30% of
the objective. ModTr's objective is detection only, so the share is 100% by definition
(`scripts/loss_share.py:1-30`).

- [x] No; record why in an ADR (recommended) — the rule calibrates the balance between
      reconstruction and detection, and there is no balance here. With Adam and one loss term,
      `grad_scale` likely just rescales the step and cancels out (inferred, not verified). The
      ADR also says what replaces the check: a short server smoke run showing that the loss falls.
- [ ] Yes, as written — follows the rule literally. It costs 25 epochs and returns 100% by
      construction. Its only value would be as a smoke run, which a shorter run gives more cheaply.

### Q7: Which datasets get a ModTr arm?

- [x] The custom set plus FLIR (recommended) — the custom set decides Margin. FLIR is the one
      public set ModTr itself reports on (ModTr beat fine-tuning there:
      `competing_paradigms.md:173`). So FLIR is both the public counterpart `goal.md` wants for
      headline numbers and a sanity check on our reimplementation.
- [ ] The custom set only — the cheapest, and the only set Margin needs. But Margin's headline
      number would then have no public counterpart (`goal.md`, "Reproducible on public data").
- [ ] All four datasets — the fullest parity with Consistency's grid. Roughly double the GPU time
      of the recommended option, for cells no criterion needs.

## Out of Scope

- **HalluciDet and zero-shot Grounding DINO.** They no longer carry Margin (`docs/roadmap.md:30-32`).
  Each, if wanted, is its own feature.
- **Decision-level fusion** (`goal.md:86-87`). It is its own feature, as margin-gate already said.
- **C2's registration-error axis and the label-transfer reference.** They are a separate
  campaign.
- **Anything that chases the closed passive claim**, that translation beats thermal-only T.
- **Editing `goal.md` or `design.md`.** δ is fixed in this feature's ADR unless Q4 decides
  otherwise, and design.md's seed-level rule stays as written.
- **ModTr with ground-truth thermal boxes**, for parity with the original paper. The custom set's
  thermal boxes are themselves transferred from visible, so no such arm exists here.
- **margin-gate's leftover tasks** (MARGIN-GATE-07, -08) and **roadmap housekeeping** (the
  Transfer status line, applying Faithfulness's pass rule). Each is a small edit or task that
  needs no feature.
- **The ModTr server campaign itself.** A run is not a task. It happens between this feature's
  tasks, and its results go through `/log-experiment`.
