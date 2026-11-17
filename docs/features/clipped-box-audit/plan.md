---
slug: clipped-box-audit
title: Clipped-box audit
status: shipped
created: 2026-11-16
---

# Clipped-box audit

## Summary

The custom set's boxes were drawn on the optical image, which has a wider field of view than
the thermal camera. The sister build script (`alignment-scripts/build_rgbt_dataset.py:235-295`,
`warp_box`) moves each box onto the thermal frame in four steps:

1. it passes the box's four corners through the image's RoMA homography;
2. it takes the axis-aligned box around the warped corners;
3. it clamps that box to the thermal frame;
4. it drops the box if less than 25% of its area survives (`--min-visibility 0.25`).

Step 3 cuts only the axis that crosses the frame edge. The other axis keeps its full extent. A
thin diagonal object, such as a pole, therefore ends up in a **loose** box: it no longer reaches
the box's new corners. Step 2 adds a second, smaller looseness, because the rotated corners are
re-boxed.

For a thin diagonal object clipped on one side, the box's retained-area fraction *equals* the
best IoU any detector could reach against a tight box. A box kept at 0.4 visibility cannot
be matched at IoU 0.5, even by a perfect detector.

This repo's registration-axis perturbation re-boxes and clamps the same way. It does this
through `deroll_label_lines` (`src/t2o/data/misalignment.py:145`, `src/t2o/data/mirror.py:99`).
As the misalignment level grows, more looseness comes in, so part of record 040's degradation
slope could come from the label clamp rather than from the misalignment.

This feature **measures the problem before changing any label**. It does four things:

- it sizes the looseness in the frozen custom labels;
- it sizes the looseness the registration axis adds at each level;
- it re-scores the existing detectors on a val set with the worst boxes removed;
- it states, before any score is seen, the rule that decides between disclosing and fixing.

Custom val is the paper's reporting split, so its validity is what is at stake.

## Tech Stack / Approach

Three standalone scripts plus one server re-score. All are shaped like
`scripts/registration_report.py` and `scripts/annotation_sweep.py`: each has its own
`basicConfig` and CSV output, and each has a pytest on synthetic boxes from `tmp_path_factory`.
No frozen tree is touched, no record is edited, and no detector is retrained.

1. **Frozen-label audit** (server run, Mac-tested). The script reads three inputs:
   - the original optical labels (`yolo_original`);
   - the per-image homographies (`alignment/<stem>.json`, written by
     `alignment-scripts/align_flir.py:331-336`);
   - the frozen `yolo_rgbt_29_jul` labels.

   It writes one CSV row per box: split, stem, class, retained-area fraction, kept or dropped,
   which edge cut the box, and the roll inflation (the area of the warped corner quad divided
   by the area of the box around it). As a fidelity check, every recomputed kept box must
   match the frozen label to six decimals, and the run fails if any does not. Before writing
   anything, check whether the original build log (the `report()` line at
   `build_rgbt_dataset.py:458`) already holds the clipped and dropped counts. Also check
   whether the Label Studio export has polygons.
2. **Registration-axis looseness** (runs on the Mac, then on the server on real val
   boxes). For each level in `REGISTRATION_LEVELS_PX` (`misalignment.py:68`), the script
   compares two boxes per object:
   - the box `misalign_label_lines` writes;
   - the tight box of the same object warped by `Misalignment.homography()`.

   It does this for two bounding object shapes. A thin diagonal segment is the worst case. A
   box-filling rectangle is the best case, and it gives zero looseness by construction. The
   script reports the mean IoU and the clamp rate per level. This yields a band of label
   looseness per level, to set beside record 040's curve.
3. **Val sensitivity re-score** (server run). Build copies of the val tree that leave out boxes
   below the visibility thresholds. Follow `misalignment.write_misaligned_tree`'s copy-a-tree
   pattern, call `labels.invalidate_label_cache`, and write `LABELS_PROVENANCE.json`. Two
   thresholds are planned:
   - **0.5**, the limit for matching at mAP50;
   - **0.75**, where mAP50-95's stricter IoU levels start to suffer.

   Re-score the existing checkpoints behind the paper's custom numbers through the one
   evaluation path (`detection/evaluation.py`, ultralytics `val`). Those are the E8 arm A
   thermal baseline (record 011), the label-transfer sweep (036), the translation arms (037)
   and the registration-axis levels (040).

   Translation arms score on their own stage export (`<run>/stage<N>/translated/data.yaml`),
   a plain YOLO tree whose val labels are a verbatim copy of the frozen labels
   (`engine/export.py` `_mirror_label`). The drop-tree builder therefore gets an export mode
   that drops by the same audit line index from `val/labels`, and the re-score takes each
   arm's original and drop `data.yaml` explicitly. Lost matches are the fall in true positives
   at IoU 0.5 from the original to the drop tree, from `collect_image_stats` on the same
   `model.val()`. The count is near-exact, since removing a box can rarely let a prediction
   re-match another.
4. **Record it** with `/log-experiment`, then apply the decision rule from Q3. The effect on
   earlier claims goes in the findings ledger's `Status` and `Acts on` cells, never in an
   edited record.

Existing code reused: `t2o.data.misalignment` (`Misalignment`, `misalign_label_lines`,
`REGISTRATION_LEVELS_PX`), `t2o.data.mirror` (`deroll_label_lines`, `PROVENANCE_FILENAME`),
`t2o.data.labels.invalidate_label_cache`, `t2o.detection.evaluation.EvaluationDetector`.

## Open Questions

## Resolved

### Q1: Where does the frozen-label audit's warp code live?

- [x] A `t2o` script that re-implements `warp_box`, with a byte-match fidelity check (recommended) — it gates on `just verify` like every other task, and the server-side match against `yolo_rgbt_29_jul` shows it reproduces the sister script exactly, so drift can't hide
- [ ] A script in `alignment-scripts`, importing `warp_box` directly — exact by construction, but the sister repo has no `just verify` gate, and this repo's tasks could not test it
- [ ] Re-run `build_rgbt_dataset.py` with extra logging — the least new code, but it edits a sister script, and a per-box CSV is a bigger change than a log line

### Q2: How are low-visibility boxes removed for the val re-score?

- [x] Drop them from a copied val tree, and score through the one evaluation path (recommended) — this keeps invariant 1. The loose labels themselves bias mAP down through missed matches, while dropping biases it down through false positives on the dropped objects. If both disagree only a little, the effect is small. Count the predictions that land on dropped boxes so that second bias is visible.
- [ ] Mark them as ignore regions, and remove matching predictions before scoring — COCO's `iscrowd` semantics and the fairest number, but ultralytics `val` has no ignore support, so this would be a second scoring path, which invariant 1 forbids
- [ ] Report both — the most complete answer, but it builds the second scoring path anyway

### Q3: What result means the labels must be fixed rather than disclosed? (Fixed before the first re-score.)

- [x] Any change of arm ranking, or any change larger than that arm's reported seed spread (recommended) — it ties "moved" to noise this project has already measured, and it targets the comparisons the paper actually claims
- [ ] A fixed absolute change, e.g. 0.01 mAP50 on any headline number — simple and easy to state, but it ignores how noisy each arm already is
- [ ] No preset rule; decide after seeing the numbers — flexible, but a threshold picked after scoring is exactly what the goal's "gate fixed before its first score" habit guards against

> Superseded by ADR-01 (2026-11-17)

## Out of Scope

- **Relabelling by hand**, or any annotation tooling. `docs/goal.md` Non-Goal: "no labelling tools".
- **Changing the frozen `yolo_rgbt_29_jul` tree or the sister build script.** Registration is
  the sister project's problem (Non-Goal "Not solving registration"). Any corrected tree is
  a follow-up feature, opened only if Q3's rule says so.
- **Shrinking boxes along an assumed diagonal.** Each object's orientation is unknown, so this
  would add a guess on top of a label that is already a transfer.
- **Rebuilding boxes from polygons.** This feature only checks whether polygons exist. Using
  them belongs to the follow-up.
- **Retraining any detector or translator.** The re-score uses existing checkpoints only.
- **Changing the registration axis's box transform** (e.g. `mirror.py`'s `centre` mode). Task 2
  measures how much looseness the transform adds. Changing the transform would change the experiment.
- **The random-crop augmentation's clamp** (`src/t2o/data/dataset.py:96-134`). It matters only
  when `train.crop` is set, and all 20 configs under `experiments/` set `crop: null`.
- **Editing records 011, 036, 037 or 040.** What this feature finds goes in the ledger.
