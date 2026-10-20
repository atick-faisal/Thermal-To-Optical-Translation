---
slug: margin-gate
title: Margin gate — resolvability check and test-split pin
status: in-progress
created: 2026-10-20
---

# Margin gate — resolvability check and test-split pin

## Summary

Margin is the only one of `docs/goal.md`'s six drafting criteria with no evidence yet
(`goal.md:48-53`, `roadmap.md:25-29`). It asks the loop to beat ModTr, HalluciDet,
zero-shot Grounding DINO, raw thermal and the no-loop translator by ≥ +2 mAP@50 on the
~100-image held-out test split, with a 95% bootstrap CI over test images. `goal.md` lists
those baselines, and that list is the one in scope, not `RESEARCH_FINDINGS.md` §8's. The
goal also builds in a gate: if the validation split shows that a CI at the test split's
size cannot resolve +2, Margin becomes non-inferiority to ModTr. In that case, HalluciDet
and Grounding DINO no longer carry the margin claim.

This feature builds that gate and nothing past it. It answers one question before any
baseline is built: **can ~100 images resolve +2 mAP@50?** The answer sets the scope of the
baseline suite, which is a separate, later feature. It also pins the held-out test split's
membership, so a reshuffle inside the ~100 pairs cannot go unnoticed before the split is
first read (`roadmap.md:226-230`).

## Tech Stack / Approach

**The resolvability check: per-image stats, resampled.**

- **Why mAP needs a different bootstrap.** mAP is not a sum over images, so you can't
  bootstrap it from one number per image. Each resample has to recompute AP from the
  matched predictions of the images it drew.
- **Capture the per-image stats.** ultralytics 8.4.117 already records them during a
  val run.
  - `DetectionValidator.update_metrics` calls `metrics.update_stats({...})` once per image.
    Each call carries `tp`, `conf`, `pred_cls`, `target_cls` and `im_name`.
  - `get_stats` then concatenates them and clears them.
  - The capture hooks in before that point, so each val run stays a single
    `YOLO(...).val()` call, exactly as `evaluate_detector` does it
    (`src/t2o/metrics/task.py:114-140`).
- **Recompute AP per resample.** Q2 decides how.
- **Resample at the test split's size.** Each resample draws 100 images with replacement
  from the 153 custom val images (an m-out-of-n bootstrap). The images are paired across
  the two arms being compared, and the statistic is the paired mAP@50 delta.
- **Seeding and seeds.** The bootstrap is seeded, as `bootstrap_ci` is
  (`src/t2o/analysis/aggregate.py:330-347`), but it resamples images, not seed
  differences. It runs per training seed and reports every seed's interval. The gate reads
  the widest interval.
- **Where it runs.** A standalone `scripts/` entrypoint following
  `scripts/annotation_sweep.py`: its own `basicConfig`, every row through the one
  evaluation path, and a resumable CSV. It runs on the server over the E3 custom-dataset
  exports that already exist (`experiments/e3_*`). No new training.
- **The run is not a task.** Its log comes back through `/log-experiment`, and the
  verdict lands in the findings ledger.
- **A rule that must be squared.** `docs/design.md:660-666` says "No claim rests on a
  bootstrap interval excluding zero". That rule is about percentile bootstraps over n=6
  *seeds*. `goal.md`'s Margin CI is over ~100 *images*, which is a different unit with a
  different sample size. An ADR records that distinction, so the two rules don't read as
  contradicting each other.
- **Which detector scores it.** The check uses the same detector checkpoint, in the same
  role, that Margin will be scored with. The ADR above names it, so the gate and the final
  number can't drift apart.
- **How the capture hooks in.** An `on_val_start` callback wraps the validator's
  `metrics.update_stats`, because `DetMetrics.stats` never stores `im_name`
  (`utils/metrics.py:1136-1145`). Using a callback rather than a validator subclass keeps
  `RTDETRValidator` intact for whichever scorer -03 names. Records are keyed by stem:
  exports are `.png` and raw thermal is `.jpg`, so filenames never match across arms. -01
  owns the records-to-mAP@50 helper, and -02 calls it per resample. The untrained fixture
  detector outputs only its biases — one confidence for every anchor, every box past the
  image edge — so it scores 0.0 on any labels, and an mAP equality against it proves
  nothing. The AP math is therefore checked against ultralytics' own `DetMetrics` on
  hand-built records with true positives, and the capture by each record carrying its own
  image's label classes. Both stay out of `slow`, so `just verify` runs them.

**The test-split pin.**

- **What exists.** `freeze_split` (`src/t2o/data/splits.py:85`) hashes train and val only.
  `DatasetManifest` cannot see a `test:` key by design (`design.md:398-402`).
- **What it adds.** The pin records the held-out images' sorted stems and their hash. It
  does this without making the test images readable by any training or evaluation path.
- **Where the record lives.** The custom dataset's split record stays on the server
  (`design.md` §9; `TASKS.md` M0.9), so the test pin does too. Only the code reaches git.
- Q1 decides whether this feature also makes the split reachable for scoring.

**Every piece runs on a synthetic fixture.** That means a `tmp_path_factory` pair set and a
stub detector, on CPU, in seconds, gated on `just verify`.

## Open Questions

## Resolved

### Q1: How much of the test split does this feature touch?

- [x] Pin only (recommended) — the split's only reader is the final Margin and Faithfulness scoring, which happens after the baselines exist. Adding `--split test` now opens an early-read path months before anything needs it.
- [ ] Pin and make it reachable (`--split test` on `evaluate` and `faithfulness`) — finishes the prerequisite in one go, but the reachable path sits idle and unguarded until the baselines land.
- [ ] Neither; leave the pin to its own feature — keeps this feature to the check alone, but the roadmap row stays open, and the pin must still land before any test read.

### Q2: How does each bootstrap resample recompute mAP@50?

- [x] ultralytics' own `ap_per_class` over the captured per-image stats (recommended) — the same AP math (interpolation, IoU thresholds, class averaging) as `evaluate_detector`, so invariant 1's single evaluation path holds. One val run per arm, and thousands of resamples are cheap.
- [ ] One `model.val()` per resample over a resampled image list — the purest match to the evaluation path, but 1,000+ val runs per arm and per seed. Also, the list of resampled images has duplicates, which ultralytics does not handle as a weighted set.
- [ ] pycocotools or torchmetrics over saved predictions — standard and well tested, but it uses a different AP interpolation from ultralytics. That is a second scoring path, which invariant 1 forbids.

### Q3: What counts as "a CI at the test split's size can resolve +2"?

- [x] The 95% CI's half-width at n=100 is ≤ 2 mAP@50 (recommended) — the direct reading: if the true delta were +2, an interval that narrow would exclude zero. It needs only the spread, not the val effect size.
- [ ] Power ≥ 80% to exclude zero at a true +2 — the textbook test, but mAP isn't additive per image, so "shift every image by +2" is undefined. It would need a simulation model this repo doesn't have.
- [ ] The observed val delta's CI at n=100 excludes zero — cheap, but it answers "is the loop already ahead on val?", which is a different question from resolving +2. It also reads an effect size the gate isn't supposed to depend on.

### Q4: Which arm pair sets the CI width?

- [x] Both loop vs no-loop control and loop vs raw thermal, gating on the wider (recommended) — the real baselines don't exist yet, so neither pair is a perfect proxy. Taking the wider interval is the conservative choice, and the second pair costs only one more val run.
- [ ] Loop vs no-loop control only — translator vs translator under one detector is the closest proxy for loop vs ModTr. It may understate the spread against a baseline whose errors are less correlated with the loop's.
- [ ] Loop vs raw thermal only — a different input domain, so its per-image errors are likely less correlated and its interval wider. But it is the weakest proxy for the translator baselines that decide Margin.

## Out of Scope

- **The baselines themselves.** ModTr, HalluciDet and zero-shot Grounding DINO each get
  built in a later feature. That feature's scope is set by this one's verdict.
- **Reading or scoring the test split.** Margin and Faithfulness close only when the
  baselines exist. The pin exists so that first read is clean.
- **Decision-level fusion** (`goal.md:86-87`). It combines the translation and
  label-transfer routes rather than being a baseline, so it belongs to its own feature.
- **Changing the seed-level rule** in `design.md:660-666`. The ADR adds a scope note
  alongside it and does not edit it.
- **Anything that chases the closed passive claim**, that translation beats thermal-only T.
- **New training runs.** The check reuses the existing E3 exports.
