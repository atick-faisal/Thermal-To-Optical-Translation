# Clipped-box audit — Decisions

## ADR-01 — Disclose the clamp's loose boxes rather than fix them, despite Q3's "fix" verdict

**Date:** 2026-11-17
**Status:** accepted
**Tasks:** CLIPPED-BOX-AUDIT-04, CLIPPED-BOX-AUDIT-05

**Context**
Plan Q3 fixed a rule before any score: "fix" the labels when, on mAP50, an arm's rank
changes or its delta exceeds its seed spread. Record 041 applied it and got `fix` on all
36 verdicts. But the deltas it fired on are not what the rule assumed they measure:

- 63–71 of the 72 val boxes under 0.5 retained area were already true positives at
  IoU 0.5, on every checkpoint (F246). The detectors were trained on the same clamped
  boxes and predict them back.
- Dropping those boxes turns their matched predictions into false positives. On every
  drop tree, precision falls 0.07–0.39 while recall moves −0.03 to +0.11.
- Plan Q2 named this bias in advance and said that if both biases disagree only a
  little, the effect is small. The loose labels' own bias, missed matches, is at most
  1–9 boxes out of 72.

So Q2 and Q3, both settled, point opposite ways on the same run.

**Decision**
The custom labels are not corrected. The frozen `yolo_rgbt_29_jul` tree stays the
paper's custom set, and every custom number stands as reported. The paper discloses the
looseness with record 041's numbers: 72 of 423 val boxes (17.0%) under half their area,
mostly poles (F245), already matched by the detectors (F246), and the route order intact
on the 0.5 drop tree (F248). No corrected-tree feature is opened.

**Alternatives considered**
- Honour Q3 and open a corrected-tree feature — there is no source of tight boxes. No
  polygon labels exist (F250), hand relabelling is a goal Non-Goal, and shrinking boxes
  along an assumed diagonal is out of scope in the plan. Per F246, a tightened val set
  scored by the current checkpoints would also score lower, so every custom checkpoint
  would need retraining on tightened train boxes.
- Measure better before deciding, e.g. retrain one label-transfer detector on tightened
  boxes and compare — costs GPU time before the January paper, and needs the same
  tight boxes that do not exist.

**Consequences**
- A gate fixed before its first score was set aside after the score was seen. That is
  the habit the goal guards against. The defence is Q2's own pre-stated reading, and a
  reviewer may still read it as post-hoc.
- The val set keeps boxes that a detector boxing the visible object tightly could not
  match at IoU 0.5. Custom numbers measure agreement with the clamp, not with the objects.
- mAP50-95 on custom is untested against the looseness. Record 041 measured matching at
  IoU 0.5 only, and the 0.75 drop tree removed 152 of 153 val poles.
- The registration axis's share of record 040's label-transfer slope stays unapportioned:
  F249 bounds the looseness per level but does not split the slope.
- The paper carries a disclosure paragraph on the custom set's labels.
