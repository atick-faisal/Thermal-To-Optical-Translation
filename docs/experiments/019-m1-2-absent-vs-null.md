# 019 — M1.2 step 9: an absent metric and a null one are different facts, and a resume has to know it

**Date:** 2026-08-23 · **Task:** M1.2
**git_sha:** ddb9cd2 — `t2o faithfulness --write-back`, which also wrote the source text below

## Evidence

Lifted verbatim from `TASKS.md:2711-2738` @ pre-spec-migration (M1.2 step 9):

> **`--write-back` is what makes the twelve invocations answerable.** C2's question is not
> "what is s0's false-object rate" but the paired loop-minus-control contrast across six seeds
> — the same instrument the +0.0512 rests on, and `t2o aggregate` is where it lives. So the
> pass folds its three rates into the scored run's `metrics.json` (`engine/loop.py`'s
> `record_faithfulness`), and they are then reachable as
> `--metric faithfulness.false_object_rate` with **no aggregator change at all**: `RunRecord`
> keeps stage entries as raw JSON and pulls dotted leaf floats out of them. The run directory
> and stage come from `--translated` rather than a second flag, since `run_loop` already
> encodes both in the export path; an export not sitting in place is refused rather than
> guessed at.
>
> Three consequences worth recording, none of them obvious before building it:
>
> 1. **A post-hoc metric exists at one stage, not four.** `aggregate` now picks each metric's
>    stages independently — every stage where *all* runs record it — instead of demanding one
>    stage set for the whole report. So `zero_shot.map50` still gets its four-stage table and
>    its trajectory while `faithfulness.*` gets a single stage-3 row and, correctly, no
>    trajectory: a gain needs a baseline, and one scored stage is a level, not a movement.
> 2. **Absent and null had to be separated.** `metric_value` deliberately raises on an explicit
>    `null` — that is what `--no-detector` writes, and averaging around a stage that computed
>    nothing is the hole it refuses to paper over. "Not scored yet" is a different fact, so
>    `metric_is_recorded` keys on the key's *presence*, and `_stage_result_to_json` omits
>    `faithfulness` entirely when empty rather than writing `null`. Without that split, every
>    run written after this change would have looked like twelve holes.
> 3. **`StageResult` needed the field, or a resume would eat the result.** `resume` rebuilds
>    every completed stage from disk and writes them all back, so a key with no field to land
>    in is silently dropped by the next stage — losing a scoring pass that cost GPU time, with
>    nothing in the file to show it ever happened.

Record 008 lifts consequences 1 and 3 as design context without F-IDs and leaves them to this
record. Consequence 1 stays design context: picking each metric's stages independently is how
`aggregate` reads a post-hoc metric, not a limit or a trap. Consequence 3 is minted below as F155.

The code as it stands (`src/` and `tests/` are unchanged since the tag):

- `src/t2o/analysis/aggregate.py:215-237` — `metric_value` raises on a `None` anywhere along the
  path; `:240` — `metric_is_recorded` keys on presence and counts an explicit `null` as recorded.
- `src/t2o/engine/loop.py:326-338` — `_stage_result_to_json` deletes an empty `faithfulness` rather
  than writing `null`, and says why.
- `src/t2o/engine/loop.py:114-120` — the comment on `StageResult.faithfulness`: the field "exists
  so that pass's numbers survive a resume".
- Tests: `tests/test_aggregate.py:189` and `:602` (null raises), `:551` and `:621` (absence is
  skipped), `tests/test_loop.py:381` (`test_recorded_faithfulness_survives_a_resume`).

Both behaviours were reasoned about while building the write-back and pinned by tests in the same
commit. Neither was observed as a failure on the server.

## Findings

### F154 — Absent means "not scored here yet" and null means "computed nothing"; `aggregate` must tell them apart, so an unscored `faithfulness` is omitted, never written as `null` (legacy: M1.2 step 9, consequence 2, "Absent and null had to be separated")

Compared against a single "missing" state. `metric_value` raises on an explicit `null` because
that is what `--no-detector` writes, and averaging around a stage that computed nothing papers over
a hole in a paired comparison. A post-hoc metric is absent from every stage it has not yet scored.
Had absence been read as null, or had the loop written `null` for an unscored `faithfulness`, every
run written after ddb9cd2 "would have looked like twelve holes". So `metric_is_recorded` keys on the
key's presence, null still reaches `metric_value` and still raises, and `_stage_result_to_json`
omits the empty field.

### F155 — A post-hoc result needs a `StageResult` field, or the next resume silently drops it (legacy: M1.2 step 9, consequence 3, "`StageResult` needed the field, or a resume would eat the result")

Compared against writing the key into `metrics.json` alone. A resume rebuilds every completed stage
from disk and writes them all back, so a key with no field to land in is dropped by the next resumed
stage. The scoring pass that cost GPU time would be lost, with nothing in the file to show it ever
ran. `StageResult.faithfulness` exists for this, and `test_recorded_faithfulness_survives_a_resume`
guards it.

## Provenance caveats

- Both findings were reasoned out and tested while building; neither loss was observed on a real
  run. The tests run on CPU on synthetic `metrics.json` files.
- Record 008 calls consequence 2 "a build-time finding that SPEC-MIGRATION-19 records". That is this
  record.
- The source does not say whether F155's drop was hit during development or foreseen. Its wording
  ("would eat") reads as foreseen.

## Next

None of its own. Any future post-hoc metric written back into `metrics.json` needs both: a
`StageResult` field, and omission rather than `null` when it is empty.
