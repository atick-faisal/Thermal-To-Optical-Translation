# Registration-error axis (C2) — Decisions

## ADR-01 — Training boxes move with the misaligned visible target

**Date:** 2026-11-10
**Status:** accepted
**Tasks:** REGISTRATION-AXIS-05, REGISTRATION-AXIS-06, REGISTRATION-AXIS-07

**Context**
The translation half warps the visible target of each train pair by a misalignment level
(plan, Approach). The plan was silent on the boxes that travel with it. They are read from the
visible-side labels and reach two places: the loop's detection loss (stages 1–3) and the
exported train labels the adapted arm's evaluation detector fine-tunes on. The zero-shot arm
never reads them (`engine/loop.py:231`); it sees them only through the translator the loop
shaped. Two readings of "warp the visible target" therefore build different experiments, and
the axis exists to compare translation with label transfer under the *same* registration error.

**Decision**
At a non-zero `data.misalignment_px`, each train pair's boxes move with its warped visible
image, through the same `misalign_label_lines` helper the label-transfer tree uses: in the
loop (`TranslationPairDataset`) and in the exported train labels (`export_split`). Val images,
val labels and split membership never move (design.md §5, invariant 2).

**Alternatives considered**
- Warp the image only; boxes stay aligned everywhere — hands translation correctly placed
  thermal-frame boxes that no misaligned rig has (they would need the same broken registration
  to produce), so translation gets labels label transfer is denied. The loop's detection loss
  would pull the translator back toward alignment and the adapted arm would learn to undo the
  shift, flattering translation on the axis meant to test it.
- Move boxes in the loop only; export aligned train labels — stays inside the task row's
  wording, but the adapted arm then fine-tunes on correctly placed boxes, so its stage mAP mixes
  in the same oracle supervision.
- Moved boxes plus one image-only ablation cell (custom set, 12 px, ~10.5 GPU-h) to separate the
  pixel effect from the box effect — offered and declined: not needed for C2, and it needs a
  code switch and a plan change.

**Consequences**
- Both halves of the curve face one registration error, and a level means identical boxes in
  the label-transfer tree, the loop and the export.
- The loop's detection loss now agrees with the shifted target, so the loop arm may degrade
  faster than a no-loop arm. That is a real effect of the loop on misregistered pairs, to be
  reported as one.
- REGISTRATION-AXIS-06 runs the loop arm only, so the curve cannot say how much of the loop
  arm's drop the loop itself causes.
- The pixel and box effects are measured together and cannot be told apart.
- `export_split` and `TranslationPairDataset` each gain a `misalignment_px` argument that only
  a train split may set.

## ADR-02 — FLIR's translation half is de-rolled before injection and scored on de-rolled boxes

**Date:** 2026-11-10
**Status:** accepted
**Tasks:** REGISTRATION-AXIS-06, REGISTRATION-AXIS-07

**Context**
Supersedes plan Q4 for FLIR's zero point. FLIR's training images still carry the real residual
roll; only its thermal labels were de-rolled (F119). `calibration/flir.json` reads 9.32 px of
mean corner displacement at 512×640, on the same measure the levels use (the 5.90 px quoted
elsewhere is the sister repo's per-pixel EPE). The real field turns counter-clockwise and the
injected one clockwise, so on rolled pairs the two partly cancel. Every translation box, train
and val, is read from the visible side (`Pairing.label_path`), while the label-transfer half
and the raw-thermal floor are scored on the de-rolled thermal-side labels. The planned
translation curve on FLIR would therefore not stand on the stated levels, and its two halves
would be scored against different val boxes.

**Decision**
On FLIR, each train pair's visible target is warped by the calibration's visible-to-thermal
map and then by the injected level, so the total misalignment is exactly 0 / 3 / 6 / 12 / 24 px.
Its train boxes are the de-rolled thermal-side labels moved by the level, through the same
helper the label-transfer tree uses (ADR-01). Its translated val images are scored against those
same de-rolled thermal-side labels. The run refuses to start unless the labels' provenance digest
matches the calibration's. FLIR's translation zero point is a new 0 px de-rolled run, not
`e3f-loop-s0`. The custom set is unchanged.

**Alternatives considered**
- Rolled training as planned, with `e3f-loop-s0` as the zero point — no new code or run, but the
  real roll cancels part of the injected one: totals of 9.32 / 7.62 / 7.12 / 9.55 / 19.11 px, so
  the curve is not against its stated levels and is not monotone in them.
- Reverse the injected field's sense on FLIR so it adds to the real roll — monotone, but the
  totals become 9.32 / 11.57 / 14.09 / 19.51 / 30.93 px, the curve never sees 0 px, and the
  two halves still measure different errors (the label-transfer half is exactly 0 to 24 px).
- De-roll the train targets but keep scoring on the visible-side boxes — no evaluation change,
  but a translator trained in the thermal frame is then scored against boxes ~9.3 px away at
  every level, so the de-roll makes the curve worse rather than cleaner.

**Consequences**
- On FLIR, both halves of the curve face the same error at every level and are scored against
  the same val boxes.
- One extra FLIR translation run at 0 px, about 10.5 GPU-h.
- A new code task before REGISTRATION-AXIS-06: a calibration config field kept out of
  `config_hash()` while unset, the composed image warp, thermal-side train and val boxes, and
  the digest check.
- FLIR's translation numbers on this axis are not comparable with record 016's twelve-run cell,
  which trained on rolled pairs and scored on visible-side boxes.
- FLIR's 0 px point is still not truly zero: about 2.25 px of median per-pair scatter remains
  after the calibration, and the report states it as it does the custom set's.
- Translation on FLIR now reads a different label side from translation on the custom set.
