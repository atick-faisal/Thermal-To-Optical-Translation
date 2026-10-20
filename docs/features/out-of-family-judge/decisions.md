# Out-of-family RT-DETR judge — Decisions

## ADR-01 — Faithfulness margin: the control arm's seed spread under the RT-DETR judge

**Date:** 2026-10-20
**Status:** accepted
**Tasks:** OUT-OF-FAMILY-JUDGE-06

**Context**
goal.md's Faithfulness criterion asks two things of the loop. Its insertion and deletion rates,
scored by a detector family never used in the loop, must be no worse than the no-loop translator's,
and "within a margin fixed before the test split is read". The test split is not pinned yet and has
not been read by any run. Record 026 has now scored every E3 stage-3 validation export with an
RT-DETR-l judge (`rtdetr`). That gives each cell's control-arm spread, which plan Q4 chose as the
margin, ahead of the alternatives listed below. The record also surfaced three things the rule had
not settled:

- **FLIR's control spread is about 10× the other datasets'** (F186). Two of its six control seeds
  collapsed.
- **The turbo cell has three control seeds**, not six.
- **Nothing said what "within the margin" means** when the test is run.

**Decision**
- **The margin is fixed per cell and per rate.** It is the stage-3 control arm's seed-to-seed std
  under the `rtdetr` judge, copied from record 026 as `t2o aggregate` prints it:

  | cell | control seeds | false-object | missed-object | consistency |
  | --- | --- | --- | --- | --- |
  | e3b (custom, pix2pix) | 6 | 0.0179 | 0.0094 | 0.0075 |
  | e3t (custom, turbo) | 3 | 0.0315 | 0.0112 | 0.0038 |
  | e3f (FLIR) | 6 | 0.0169 | 0.1708 | 0.1222 |
  | e3md (msrs-day) | 6 | 0.0171 | 0.0143 | 0.0107 |

- **The pass rule.** On the test split, a rate passes when the 95% bootstrap CI of the seed-paired
  loop − control difference (as `t2o aggregate` prints it) lies entirely on the safe side of the
  margin:
  - below +margin for false-object and missed-object;
  - above −margin for consistency.
- **FLIR's margins are used as computed.** Its missed-object and consistency margins are stated to
  barely constrain the result.
- **Turbo's margin comes from its own three seeds**, flagged as a three-seed estimate.
- **The preconditions, stated as fact.** All three judges passed the 90%-of-yolo11s gate (F181),
  and the test split is unread as of this entry.
- **Scope.** These margins belong to the `rtdetr` checkpoints scored in record 026, and to no other
  judge.

**Alternatives considered**
- **Half-width of the control's 95% bootstrap CI over val images** (plan Q4): it shrinks with split
  size, so a number taken on val transfers poorly to a test split of a different size.
- **A fixed absolute number, e.g. 0.02 on each rate** (plan Q4): the purest pre-registration, but
  arbitrary and hard to defend to a reviewer.
- **Capping FLIR at the widest spread elsewhere** (0.0143 missed-object, 0.0107 consistency):
  tighter, but the cap would be chosen after the spreads were seen, which is the post-hoc tuning a
  pre-registered margin exists to prevent.
- **A robust spread, such as the median absolute deviation, on every cell**: collapsed seeds could
  not inflate it, but it changes the rule after seeing the data and moves every cell's margin.
- **One spread per rate, pooled across cells**: simple, but it mixes datasets of very different
  difficulty, and FLIR still inflates the pool.
- **Borrowing e3b's six-seed margin for turbo**: a steadier estimate, but it comes from a different
  translator, which breaks "the noise of exactly the unit being compared".
- **No margin for turbo**: honest about n = 3, but it leaves the custom set's Faithfulness resting
  on pix2pix alone.
- **A per-image bootstrap CI at test time**: matches the Margin criterion's style, but the margin is
  a seed spread, so the units would not match.
- **The point estimate alone at test time**: the simplest option, but it ignores goal.md's "with
  CIs".

**Consequences**
- **FLIR's missed-object and consistency tests are weak** (margins 0.17 and 0.12). On FLIR,
  Faithfulness leans on the false-object rate, whose margin is 0.0169. Its paired difference on val
  was −0.1487.
- **Turbo's margins rest on three seeds** and are unstable estimates. Its consistency margin
  (0.0038) is the tightest in the table, so it may fail on noise alone.
- **The margins are tied to these judge checkpoints.** RT-DETR training does not reproduce (F188),
  and absolute rates differ between judges (F185). Retraining or replacing a judge voids its cell's
  margin, so that judge's checkpoint must never be overwritten.
- **Test-time scoring must match the val scoring:** seed-paired, at stage 3, under `rtdetr`, with
  the same `t2o faithfulness` settings.
- **Faithfulness is not closed by this entry.** It closes when the pinned test split is scored
  against these numbers.
