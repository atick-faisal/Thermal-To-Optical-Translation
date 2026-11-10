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
