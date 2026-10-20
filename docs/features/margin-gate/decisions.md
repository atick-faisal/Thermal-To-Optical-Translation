# Margin gate — resolvability check and test-split pin — Decisions

## ADR-01 — Margin's CI is over test images, scored by the zero-shot yolo11s reference

**Date:** 2026-10-20
**Status:** accepted
**Tasks:** MARGIN-GATE-03, MARGIN-GATE-04

**Context**
`goal.md`'s Margin criterion asks for a 95% bootstrap CI over ~100 held-out test images.
`design.md:660-666` says "No claim rests on a bootstrap interval excluding zero", a rule written
because a percentile bootstrap over n = 6 training seeds is anti-conservative and disagreed with
the exact sign-flip test on three E3 cells. Read literally, the two contradict each other.
Separately, the resolvability gate (MARGIN-GATE-04) must score with the same detector that the
final Margin number will use. Otherwise the gate measures one detector's spread and the claim
rests on another's. No document named that detector.

**Decision**
- The seed-level rule governs intervals over training seeds. Margin's interval is over test
  images (n ≈ 100, an m-out-of-n draw from the 153 val images for the gate), a different unit
  with a different sample size, so it sits outside that rule. `design.md` is not edited.
- Both the gate and the final Margin score use the zero-shot reference detector,
  `detector.reference`, at `runs/reference-yolo11s/weights/best.pt`, the checkpoint every E3
  config names. It is never fine-tuned on anything this project produced and never receives a
  generator gradient.

**Alternatives considered**
- Hold Margin to the seed rule (claim rests on a sign-flip p over seeds) — contradicts
  `goal.md`'s wording, which names a CI over test images, and `goal.md` is immutable.
- Edit `design.md:660-666` to carve out the exception — the plan's Out of Scope keeps that rule
  as written; a scope note beside it is enough.
- Score with the out-of-family RT-DETR-l judge (`runs/reference-rtdetr-l`) — `goal.md` already
  gives that role to Transfer and Faithfulness. It was gated at only 90% of the yolo11s
  reference, and its numbers would not compare with E1's 0.1887 raw-thermal floor or with F35
  and F72, all measured by yolo11s.
- Score with both and gate on the wider — Margin is one number from one scorer. The
  out-of-family check is Transfer's job, and running both doubles every val run for a criterion
  that already has a conservative gate (Q4).
- Score with the in-loop `yolo11n` — it receives the loop's gradient, so a gain it reports may be
  reward hacking (design.md §5, invariant 7; every E3 config keeps it apart from the reference).
- Score with the adapted evaluation detector — it is fine-tuned on each export, which breaks
  `goal.md`'s "keeps the visible detector frozen", and it saturates above 0.9 mAP@50 on every arm
  (design.md §12).

**Consequences**
- Two kinds of interval now appear side by side. Every reported CI must say whether it is over
  seeds or over images.
- An image-level CI from one training seed ignores seed-to-seed variance, which M1.2 put at
  0.059 mAP@50 between two runs of one configuration. The gate copes by reading the widest
  interval across seeds; the final Margin number still needs Stability's ≥3 seeds beside it.
- Margin is scored by a detector from the same family as the in-loop one. So ModTr and
  HalluciDet must be trained against the same in-loop `yolo11n`, or the family match favours
  the loop. This binds the later baselines feature.
- Zero-shot Grounding DINO is a detector, not a translator, so this scorer cannot score it. The
  baselines feature has to decide how its number is made comparable.
- The checkpoint lives only on the server. MARGIN-GATE-04 takes it as a path, and the Mac tests
  stand in a fixture for it.
