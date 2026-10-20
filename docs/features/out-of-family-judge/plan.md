---
slug: out-of-family-judge
title: Out-of-family RT-DETR judge
status: shipped
created: 2026-10-18
---

# Out-of-family RT-DETR judge

## Summary

Two of `docs/goal.md`'s six drafting criteria need a detector from a family the loop has never
used:

- **Transfer** (`goal.md:54-55`): the loop's gain over the no-loop translator holds for such a
  detector, across ≥3 seeds.
- **Faithfulness** (`goal.md:61-64`): insertion and deletion rates are scored by such a detector,
  and stay within a margin fixed before the test split is read.

Every score so far comes from `yolo11s`. That is the same family as the in-loop `yolo11n`
(`docs/roadmap.md:30-31`, `:49-51`), so neither criterion has been measured.

This feature adds one visible-trained **RT-DETR** judge per dataset. It then re-scores the
stage-3 control and loop exports that already exist on the server, for zero-shot mAP and for
the three faithfulness rates. No translator is retrained. The yolo11s scores behind the existing
findings stay exactly where they are.

**What it delivers:**
- **Transfer:** a direct test, run through `t2o aggregate`'s paired sign-flip and bootstrap.
- **Faithfulness:** the val-split rates under an out-of-family judge, plus a pre-registered
  margin.

**What it does not deliver:** it does not close Faithfulness. That needs the held-out test split,
which is not pinned yet (`roadmap.md:226-230`).

## Tech Stack / Approach

**No new dependency and no loader seam.** Under the pinned ultralytics (8.4.117), `YOLO(path)`
checks the checkpoint's last layer. If that layer is an `RTDETRDecoder`, `YOLO` turns itself into
`RTDETR`, which brings RT-DETR's own predictor, validator and trainer
(`ultralytics/models/yolo/model.py:79-86`, `rtdetr/model.py:56-62`). The explorer read this in the
source; it has not been run. If it holds, every existing scoring call already accepts an RT-DETR
`.pt`:

- `src/t2o/metrics/task.py:130-132` (`evaluate_detector`);
- `src/t2o/metrics/faithfulness.py:223,246` (`evaluate_faithfulness`);
- `src/t2o/detection/evaluation.py:37-39`;
- `src/t2o/engine/detector_stage.py:136,156` (`train_detector`).

The first task proves this with a test before anything builds on it. On this Mac,
`RTDETR('rtdetr-l.yaml')` already builds an `RTDETRDetectionModel` on CPU.

**Training the judge reuses `t2o train-detector`** (`src/t2o/cli.py:168-194`, `:538-558`). Its
`--init-weights` help already says "use an architecture other than the in-loop detector's". The
runbook is record 022's (`docs/experiments/022-msrs-day-detector-training.md`), with
`--init-weights rtdetr-l.pt` in place of `yolo11s.pt`. The judge trains on the same visible train
split as the existing yolo11s reference. Under invariant 7 (`docs/design.md:166-171`) the judge is
a second *reference* detector: it is never trained on project outputs and only scores zero-shot.
No fourth detector role is needed.

**The code change is where a judge's scores land in `metrics.json`.** Three facts force it:

- `t2o evaluate` (`cli.py:101-108`, `:431-443`) only logs its result. It has no `--write-back`, so
  a re-scored zero-shot mAP can't reach `metrics.json`, and therefore can't reach `t2o aggregate`.
- `record_faithfulness` (`src/t2o/engine/loop.py:370-394`) always writes the fixed key
  `"faithfulness"`. Scoring with RT-DETR would overwrite the yolo11s rates behind F49, F76, F143
  and F171.
- `StageResult` (`loop.py:98-120`) only round-trips named fields through `_stage_result_from_json`
  (`:350-367`). A key without a field is silently dropped the next time a run resumes, as the
  comment at `:114-119` warns.

So the plan is:

- **A judge name on both re-scoring commands.** `t2o evaluate` and `t2o faithfulness` both take
  `--write-back` and a `--judge` name. Without `--judge`, `faithfulness` keeps writing
  `"faithfulness"` exactly as it does today.
- **Shared write-back logic.** The zero-shot write-back copies the existing
  `_export_run_and_stage` (`cli.py:512-543`) and `record_faithfulness` pattern.
- **A key shape that round-trips through `StageResult`** and stays reachable by
  `t2o aggregate`'s dotted metric paths (`src/t2o/analysis/aggregate.py:215-237`). The exact shape
  is Q1.
- **The primary-class mean reaches the judge's section.** `add_primary_mean`
  (`aggregate.py:358-391`) only scans top-level sections. A nested section would silently get no
  primary mean, which is how a passive-class headline goes missing without an error.

**Tests** extend the existing suites on synthetic pairs from `tmp_path_factory`, on CPU. A random
RT-DETR checkpoint fixture follows `tests/conftest.py:81-94`, which builds a random
`DetectionModel("yolo11n.yaml")`. RT-DETR-l is much heavier than yolo11n on CPU, so a test that
predicts with it may need the `slow` marker or a smaller image size. The first task finds out.

**Pre-registering the Faithfulness margin is in scope.** Once the val re-scoring run lands,
one task writes an ADR in `decisions.md`. It fixes the margin under the rule chosen in Q4,
computed from the control arm's noise and never from the loop's observed advantage, and it states
that the test split is still unread.

**The server runs are not tasks.** They happen between tasks and come back as logs for
`/log-experiment`:

1. **A short RT-DETR training smoke run.** RT-DETR's `grid_sample` backward pass is not
   deterministic, so this checks how `deterministic=True` behaves before the real runs.
2. **One judge training run per dataset** (Q2, Q3).
3. **Re-scoring every existing stage-3 export** with `--write-back --judge <name>`, then
   `t2o aggregate`.

The exports already exist on the server and are never deleted:
- custom set: `runs/e3b-*` for pix2pix (record 007) and `runs/e3t-*` for pix2pix-turbo
  (record 010);
- FLIR: `runs/e3f-*` (record 016);
- msrs-day: `runs/e3md-*` (record 024).

**Every judge is gated before it scores anything** (Q5). A judge that is weak on real visible
frames gives noise, not a Transfer signal. msrs-day has only 536 visible training images, which
makes this a real risk there.

## Open Questions

## Resolved

### Q1: What shape do a judge's scores take in `metrics.json`?

- [x] A `judges` map, `judges.<name>.zero_shot` and `judges.<name>.faithfulness` (recommended) —
  one new `StageResult` field holding a small value object per judge. The yolo11s keys stay
  untouched, and a second judge later needs no schema edit. `add_primary_mean` has to learn to
  descend one level, which is a few lines with a test.
- [ ] Fixed flat fields, `zero_shot_rtdetr` and `faithfulness_rtdetr` — no aggregator change
  because `add_primary_mean` already sees them. But the family is hard-coded into the schema, and
  every further judge means another field plus another branch in `_stage_result_from_json`.
- [ ] A separate `judges.json` beside `metrics.json` — avoids the resume problem entirely. But
  `t2o aggregate` would have to learn to read a second file, and one run's scores would live in
  two places.

### Q2: Which datasets get a judge?

- [x] All three where Consistency holds: custom set, FLIR and msrs-day (recommended) — the
  exports exist for all three, so the only extra cost is two more judge trainings. A Transfer
  result on one dataset only would be the weakest version of the claim.
- [ ] Custom set only — the power-line set is the paper's headline and the cheapest single
  answer. But goal.md's "Reproducible on public data" line (`goal.md:68-69`) asks every headline
  number to have a public-dataset counterpart, so FLIR or msrs-day would be needed anyway.
- [ ] Custom set first, then FLIR and msrs-day only if the custom-set judge passes Q5's gate —
  saves GPU time if RT-DETR proves hard to train. It costs a second round-trip before any public
  number exists.

### Q3: How many judge seeds per dataset?

- [x] One (recommended) — goal.md's "≥3 seeds" for Transfer means translator seeds, and six
  paired seeds already exist per dataset. A judge-seed sensitivity check can follow if the result
  turns out borderline.
- [ ] Three, matching record 022's yolo11s reference on msrs-day (s1–s3) — shows the result does
  not hinge on one judge's training luck. It triples the judge-training GPU time and the
  re-scoring passes.

### Q4: What rule sets the Faithfulness margin?

- [x] The control arm's seed-to-seed standard deviation of each rate, across its six val seeds
  (recommended) — the noise scale of exactly the unit being compared (paired seeds). It is simple
  to state and cannot be tuned toward the loop's result.
- [ ] The half-width of the control arm's 95% bootstrap CI over val images — closer to the
  per-image CI goal.md asks for at test time. But it shrinks with split size, so a val-derived
  number transfers poorly to a test split of a different size.
- [ ] A fixed absolute number chosen now, e.g. 0.02 on each rate — the purest form of
  pre-registration. It is arbitrary, though, and an arbitrary margin is hard to defend to a
  reviewer.

### Q5: What must a judge reach before it scores anything?

- [x] Its visible-val mAP50 is at least 90% of the yolo11s reference's on the same split
  (recommended) — relative to a judge already accepted for that dataset, so one rule works on all
  three. A miss is recorded as a finding, not retried silently.
- [ ] An absolute floor, e.g. mAP50 ≥ 0.5 — easy to state, but the datasets differ widely in
  difficulty, so one floor is either too loose on msrs-day or too strict on FLIR.
- [ ] No gate; report the judge's visible mAP beside every score — the least machinery. It leaves
  "is the judge any good?" to the reader, after the result is already known.

## Out of Scope

- **Closing Faithfulness.** That needs the held-out test split.
- **Pinning the test split** (`roadmap.md:226-230`), including the custom set's missing
  `splits/*.json`. Both belong to the Margin baseline suite.
- **The Margin baselines:** ModTr, HalluciDet and zero-shot Grounding DINO.
- **Retraining any translator,** and any new translator backbone. The `loss_share.py` rule
  therefore does not apply.
- **Any comparison against the thermal-trained detector T.** The passive claim is closed negative
  (`roadmap.md:36-42`). Every comparison here is loop against no-loop.
- **A non-ultralytics family** such as Faster R-CNN or mmdetection. RT-DETR is the chosen
  family, so no new dependency is added.
- **Changing the in-loop detector** or the yolo11s reference keys, and editing any landed record.
- **The roadmap's Consistency wording** ("3 of 5", with LLVIP). That is a roadmap edit, outside
  this feature's directory.
