---
slug: registration-axis
title: Registration-error axis (C2)
status: in-progress
created: 2026-11-08
---

# Registration-error axis (C2)

## Summary

Goal C2 (legacy C4, "when translation pays") maps where translation beats its alternatives.
Since the 2026-11-03 amendment it must include a **registration-error axis**: "how fast label
transfer and translation each degrade as alignment worsens" (`docs/goal.md:77-85`). Nothing in
the repo measures this yet. `modtr-baseline/plan.md:279` deferred it as "a separate campaign",
and finding F119 (`docs/experiments/index.md:165`, record 014:328) names the translation half as
still open. This feature builds the code for both halves. It injects a controlled, known
misalignment into the **training** pairs at a few magnitudes, then scores each level on the
untouched validation split through the one evaluation path. The output is two degradation
curves per dataset, label transfer and translation, plotted against misalignment in pixels.
They are reported pass or fail, like every criterion. The server runs happen between tasks and
land as records via `/log-experiment`.

## Tech Stack / Approach

**The perturbation.** A synthetic misalignment applied on top of each dataset's existing
alignment. It is an input to the experiment, not a registration method (Non-Goal "Not solving
registration", `goal.md:131-132`). `ResidualCalibration` in `src/t2o/data/calibration.py:104-165`
already models a corner field, its homography and `magnitude_px()`, and its docstring's alpha
table (lines 24-34) scales a corner field. The perturbation reuses that type rather than adding
a new one. Each level is a frozen dataclass plus a digest, following `calibration.py` and
`data/splits.py`.

**Label-transfer half (cheap: ~10-minute detector fine-tunes).**
- Push the train-split boxes through the perturbing homography with the existing
  `deroll_label_lines` / `mirror_labels_onto_infrared` seam (`src/t2o/data/mirror.py:99-148`),
  which also writes `LABELS_PROVENANCE.json`.
- Write each level to its **own** label tree, never over `infrared/labels`. This avoids the
  `labels.cache` trap (ultralytics keys its cache on size and path, not content) and the
  provenance-state trap.
- Train with `train_detector` (`src/t2o/engine/detector_stage.py:111`) and score with
  `evaluate_detector` (`src/t2o/metrics/task.py:140`) on the unperturbed val split.
- Drive it with a sweep script shaped like `scripts/annotation_sweep.py`: a CSV it appends to,
  resume support, cheap cells first, its own `basicConfig`. Every row is stamped with its
  magnitude and perturbation digest, as `scripts/gate_table.py:102-105` stamps
  `calibration_digest`.
- The zero point on the custom set is E8 arm A at N=600 (0.9300, record 011:268-272). On FLIR
  the zero point is a yolo trained on the de-rolled mirrored labels. That detector is also
  FLIR's missing C4 thermal-supervised and label-transfer reference (`docs/roadmap.md:260-262`),
  because FLIR has no thermal labels and the two references are the same detector. If FLIR is
  in scope, the C4 bracket folds into this feature as one task.
- FLIR's sweep runs on the FLIR judge's recipe (`--init-weights yolo11s.pt --epochs 100`,
  record 013), so the bracket's V − T compares sensors rather than recipes. The custom set keeps
  E8's yolo11n at 50 epochs. At about 1.43 h per FLIR cell, FLIR's 15 cells cost about 21
  GPU-hours, not minutes.

**Translation half (expensive: one pix2pix run is about 10.5 GPU-hours).**
- Add a misalignment field to `DataConfig` (`src/t2o/config/schema.py:48-77`). It warps the
  visible target of each **train** pair at load time. It sits inside `config_hash()`, so a
  level is a config file (design.md §5, invariant 6), not a hidden flag.
- The field defaults to "no perturbation" and must leave every existing config's hash
  unchanged. Otherwise the existing aligned runs stop counting as the zero point.
- Val pairs, val labels and split membership are never perturbed (invariant 2, the frozen
  data contract).
- One experiment YAML per level under `experiments/`, plus a runbook with the server
  provenance block. The runs are not tasks.

**Disclosure carried into the record.** The custom set's val labels were themselves moved onto
thermal by the sister repo's registration. Every custom-set curve is therefore scored against
boxes that already carry that error, so its "0 px" point is not truly zero. FLIR's measured
residual is 5.90 px, with about 4-5 px of scatter that can't be removed (record 012:47-58).
On the levels' own measure, mean corner displacement, FLIR's calibration reads 9.32 px; the
5.90 px is per-pixel EPE. About 2.25 px of median per-pair scatter survives calibration, so
FLIR's 0 px is not truly zero either.

**FLIR's translation half (ADR-02).** Thermal-side boxes are
`pairing.label_path(pairing.infrared_path(visible))`, the path `data/inner_split.py:96` already
derives. The target warp is one homography, the level's composed with
`calibration.visible_to_infrared()`, applied in one `warpPerspective` so the target is resampled
once; at 0 px the export copies the thermal label byte for byte. Only the export reads val boxes
(the trainer's val loss never touches them, `engine/trainer.py:208`), so the label switch lives in
the train dataset and `engine/export.py`. The digest check reads both splits' `infrared/labels`
with `read_label_provenance`, as `scripts/gate_table.py:229-256` does for val. The faithfulness
metrics keep visible-side boxes; they are not on this curve.

**Tests.** Synthetic `tmp_path_factory` pairs on CPU, as always: the warp moves boxes and
pixels by the expected amount; level zero is the identity, byte for byte; the default config
hash is unchanged; val is untouched; a tiny sweep runs end to end under the `slow` marker.

## Open Questions

### Q6: Which boxes score FLIR's translation half at val, once its train targets are de-rolled?

- [ ] De-rolled thermal-side labels, the ones the label-transfer half and the gate floor already use (recommended) — both halves are scored against the same val boxes, so 0 px means the calibrated alignment on both; costs a label-side switch in the export, and the new FLIR runs are not comparable with record 016's
- [ ] Visible-side labels, as every translation run so far — no evaluation change, but a translator trained on de-rolled targets is then scored against boxes ~9.3 px away at every level, so the de-roll makes the curve worse, not cleaner
- [ ] Drop the de-roll; train on rolled pairs with `e3f-loop-s0` as the zero point, as Q4 planned — no new code or run, but the real roll partly cancels the injected one (totals 9.3 / 7.6 / 7.1 / 9.6 / 19.1 px), so the curve is not against the stated levels

## Resolved

### Q1: What shape does the injected misalignment take?

- [x] Scaled corner field (recommended) — reuses `ResidualCalibration` and its alpha scaling, and matches the real failure (FLIR's −1.18° roll is a corner field, not a shift), so the curve speaks to real rigs
- [ ] Uniform translation (dx, dy) — simplest to explain and plot, but no real rig fails this way, and it adds a second perturbation type alongside the existing corner model
- [ ] Random shift per image — models jitter from per-frame registration, but adds a sampling seed to every level and blurs "magnitude" into a distribution

### Q2: Which misalignment magnitudes (corner-field px at training resolution)?

- [x] 0 / 3 / 6 / 12 / 24 px (recommended) — brackets FLIR's measured 5.90 px, reaches clear failure, and four non-zero levels draw a curve, not two points
- [ ] 0 / 6 / 24 px — halves the translation GPU cost, but two non-zero levels can't show *how fast* it degrades, which is the axis's own wording
- [ ] 0 / 1.5 / 3 / 6 / 12 / 24 / 48 px — the finest curve, but six translator retrains per dataset is beyond the January budget

### Q3: Which datasets carry the axis?

- [x] Custom + FLIR (recommended) — "reproducible on public data" (`goal.md:72`) needs a public counterpart, FLIR has a measured registration anchor, and it folds in FLIR's missing C4 bracket
- [ ] Custom only — half the cost, but the headline axis would have no public counterpart, and FLIR's C4 bracket stays a separate item
- [ ] Custom + FLIR + msrs-day — a third curve, but msrs-day's registration is "assumed, not measured" (`scripts/campaign_report.py:143`), so its zero point means nothing

### Q4: How is the translator trained at each misalignment level?

- [x] From scratch, 1 seed per non-zero level, reusing the existing 3-seed aligned runs as the zero point (recommended) — clean, about 4 levels × 2 datasets × ~10 h ≈ 80 GPU-hours, about two days on both A100s; the single seed per level is disclosed
- [ ] From scratch, 3 seeds per level — seed-stable at every point, but about 250 GPU-hours competes with the remaining paper runs before 2027-01-31
- [ ] Fine-tune the existing aligned checkpoints on warped pairs — cheapest, but the starting weights already learned the aligned mapping, so it likely understates the degradation and confounds the curve

> Superseded by ADR-02 (2026-11-10) for FLIR's zero point; the custom set still reuses its aligned runs

### Q5: Where is FLIR's C4 bracket built?

- [x] Extend `scripts/gate_table.py` (recommended) — a `--label-transfer-csv` reads the sweep's FLIR 0 px rows, refuses a `calibration_digest` that differs from the gate's own, scores their checkpoints as the T row, and labels it "thermal-supervised = label transfer" with the reason; the sweep row gains `init_weights` so the recipe shows. One renderer, one command, and the de-roll check the E9 gate already learned
- [ ] New report script joining `gate.csv` and the sweep CSV — no re-scoring, but a second renderer of a table the gate already prints, and the digest check would have to be written twice
- [ ] No code: the runbook passes the 0 px checkpoints to the existing `--thermal-weights`, and the record states the coincidence — cheapest, but nothing checks the checkpoints are the de-rolled 0 px cells, and the row's "tests on a synthetic sweep CSV" has nothing to test

## Out of Scope

- Building or improving a registration method. Registration is the sister project's problem and only an input here (`goal.md:131-132`).
- Perturbing val images, val labels or split membership. Only the train side is perturbed (design.md §5, invariant 2).
- The never-seen-visible comparison. It is the axis's optional sibling in C2 (`goal.md:83-85`) and stays a separate roadmap row.
- A pix2pix-turbo curve. pix2pix carries the translation half, which keeps the GPU budget inside January.
- Measuring the sister repo's own registration residual on the custom set. It is disclosed, not measured here.
- Running the server campaigns. They happen between tasks and land as records via `/log-experiment`.
