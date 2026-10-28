# Out-of-family RT-DETR judge — Decisions

## ADR-01 — Faithfulness margin: the control arm's seed spread under the RT-DETR judge

**Date:** 2026-10-20
**Status:** superseded by ADR-02
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

## ADR-02 — Faithfulness is decided on val, under ADR-01's margins and rule

**Date:** 2026-10-28
**Status:** accepted
**Tasks:** none — the verdict is applied by a `/log-experiment` record, not a task

**Context**
This supersedes ADR-01's split and closure terms. ADR-01 applied its pass rule "on the test split"
and said Faithfulness closes only when that split is scored. No test split exists: the custom set
is 600 train / 153 val, frozen, and `goal.md:61-65` now scores Faithfulness on val, with a margin
"drawn from the same validation data it is scored on, and the paper says so". Applying the rule
also runs into `design.md:664`: "No claim rests on a bootstrap interval excluding zero." The rule
uses a seed-level percentile bootstrap CI, which is exactly the unit that line was written about.
Margin-gate ADR-01 avoided that line because its interval is over images, but this interval is
over seeds, so that route is closed. One more fact forced the choice: on val, e3t's (turbo)
false-object difference is +0.0040, well inside its 0.0315 margin, while its n = 3 interval
reaches +0.0538 (record 026, `026-out-of-family-rtdetr-judge.md:1248`). The handling of the CI
decides turbo's verdict.

**Decision**
- **The split is val.** ADR-01's pass rule applies to record 026's stage-3 val scores under
  `rtdetr`, with nothing else changed. "On the test split" now reads "on val".
- **ADR-01's margins and rule stand word for word.** The per-cell, per-rate table, the seed-paired
  95% bootstrap CI, its direction per rate, and the scope to record 026's checkpoints all stay.
  Nothing is retuned.
- **A scoped exception to `design.md:664`, stated by direction.**
  - **A fail is robust.** The percentile bootstrap at n = 3–6 is anti-conservative, i.e. too
    narrow. An interval that crosses the margin while too narrow would also cross it at its
    honest width.
  - **A pass rests on that anti-conservative interval** and is reported that way, beside the
    rate's exact sign-flip p.
  - `design.md` is not edited.
- **The verdict is per cell and per rate.** A fail is reported as a fail. It is not outvoted by
  another cell on the same dataset.
- **Faithfulness closes** when one record applies this rule to record 026's numbers.
- **ModTr's rates are reported**, as `goal.md:65` asks ("for every arm"), under the same judge
  checkpoints. ModTr sits outside the pass rule, which compares loop against control only.

**Alternatives considered**
- **An exact one-sided sign-flip test of (loop − control − margin)**: it is design.md's own test,
  but it changes the rule after the numbers are seen. At n = 3 its smallest p is 0.125, so turbo
  could never pass any rate by construction.
- **The point estimate inside the margin, with the CI shown as descriptive spread**: turbo would
  pass (+0.0040 vs 0.0315). That is a rule change toward a pass, made after seeing a fail.
- **Margin-gate ADR-01's scoping argument (the interval is over images, not seeds)**: this
  interval is over seeds, so the argument does not hold here.
- **Editing `design.md:664` to carve out the exception**: the line is right about passes, which
  are the claims it guards. A scope note in this entry is enough, as it was for margin-gate.
- **Retuning turbo's margin, e.g. borrowing e3b's six-seed spread**: ADR-01 already rejected this
  because it breaks "the noise of exactly the unit being compared". Doing it after seeing the fail
  would be the post-hoc tuning a fixed margin exists to prevent.
- **More turbo seeds to tighten its interval**: a costly server campaign, and it reads as chasing
  a pass after a fail.
- **Resting the custom set's Faithfulness on e3b alone**: it breaks consistency with how similar
  experiments were run before. Earlier campaigns reported e3b and e3t each in their own right
  (records 007 and 010), and this verdict follows the same practice.
- **Waiting for a held-out test split**: none exists or is planned. `goal.md` now scores on val.

**Consequences**
- **Turbo's false-object rate fails**, and the paper reports it. The custom set's Faithfulness is
  split: pix2pix passes and turbo fails on one rate.
- **The turbo fail shows an unresolved rate, not more invented objects.** Its point estimate is
  +0.0040 (p = .75); the fail comes from n = 3's interval width. The paper has to say so in those
  terms.
- **Passes are weaker evidence than fails.** They rest on an interval `design.md` calls
  anti-conservative. e3b's false-object pass has a sign-flip p of .156, which shows no difference
  either way, not a demonstrated equivalence.
- **The margins and the verdict come from the same data.** Record 026 printed the paired
  differences on 2026-10-19, and ADR-01 fixed the margins from the control spread on 2026-10-20.
  This is disclosed, as `goal.md:63-64` requires.
- **Nothing reopens Faithfulness** once the record lands, because no test split will arrive to
  re-score it. ADR-01's checkpoint tie and its scoring-match rule still bind any re-score.
- **The ModTr rates need a server run** using these same judge checkpoints. Reporting them with a
  CI needs code, because `t2o aggregate` cannot pair ModTr runs (`aggregate.py:285-298`).
