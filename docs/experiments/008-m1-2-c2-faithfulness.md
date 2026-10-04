# 008 — M1.2 step 9: C2's faithfulness pass on record 007's twelve stage-3 exports, scored by the reference `yolo11s`

**Date:** 2026-08-23 · **Task:** M1.2 · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in eb61cd1 · **W&B:** none — no command below carries `--wandb` · **Log:** none saved — transcribed from TASKS.md 2614-2762 @ pre-spec-migration

## Command

As written in `TASKS.md`, in PowerShell since the server is native Windows. The `$DATA` path
actually passed on the server is not recorded. The first two commands belong to step 9 but feed
record 007's findings (see Provenance caveats); the `foreach` and the last `aggregate` are this
record's run.

```powershell
uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 `
  --metric zero_shot.map50 fidelity.lpips zero_shot.per_class_ap50.Switch `
  --csv runs/e3b-tidy.csv
uv run python scripts/loss_share.py --runs 'runs/e3b-control-*' --terms-only
foreach ($arm in 'control','loop') { foreach ($s in 0,1,2,3,4,5) {
  uv run t2o faithfulness --translated "runs/e3b-$arm-s$s/stage3/translated" `
    --data $DATA --weights runs/reference-yolo11s/weights/best.pt --device cuda:0 --write-back
} }
uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 `
  --metric faithfulness.false_object_rate faithfulness.missed_object_rate `
           faithfulness.detection_consistency
```

## Configuration

**Why this pass exists.** `metrics/faithfulness.py` has been complete since M0.5 and was wired
to **nothing** — no engine, no analysis, no CLI. Finding 9 (F43) is what makes it load-bearing:
an LPIPS cost that coincides with a detection gain is either a fidelity trade or hallucination,
and only a count of invented and erased objects separates them. Every stage's export is still on
disk, so this needs no retraining.

- **Inputs:** the twelve stage-3 exports of record 007's campaign,
  `runs/e3b-{control,loop}-s{0..5}/stage3/translated`, `grad_scale: 0.15`. No retraining.
- **Judge:** record 004's reference `yolo11s`. **`--weights` is deliberately required and
  un-defaulted.** It must be the reference `yolo11s`. Scoring hallucination with the in-loop
  `yolo11n` measures how well the translator learned to please the checkpoint that trained it,
  which is the confound step 1's judge exists to remove.
- **Matching:** the command passes no thresholds, so `t2o faithfulness`'s defaults apply —
  `--split val`, `--iou-threshold 0.5`, `--conf-threshold 0.25`, `--imgsz 640`
  (`src/t2o/cli.py:148-151` at ddb9cd2 and eb61cd1; a reading of the code, see caveats).
- **Test:** paired loop − control at stage 3, exact two-sided sign-flip over 2⁶ assignments,
  with a bootstrap CI — record 007's instrument.

**Pre-registered read**, written before the pass ran, now that finding 9's training-loss half
is in: false-object rate flat or falling while mAP50 is +0.0512 closes the reward-hacking
question outright, and C2 gets its first real number. False objects rising with λ would be the
one result that still contradicts the loss-space evidence — the objective shows no perceptual
sacrifice, so hallucination would have to be arriving through the detection term's own gradient
rather than by outbidding LPIPS. That would make `reward_target` live and **hold the turbo
campaign** — the same gate step 6 applied to the dose question, for the same reason. The prior
after finding 9 is that it comes back clean.

**`--write-back` is what makes the twelve invocations answerable.** C2's question is not "what
is s0's false-object rate" but the paired loop-minus-control contrast across six seeds — the
same instrument the +0.0512 rests on, and `t2o aggregate` is where it lives. So the pass folds
its three rates into the scored run's `metrics.json` (`engine/loop.py`'s `record_faithfulness`),
and they are then reachable as `--metric faithfulness.false_object_rate` with **no aggregator
change at all**: `RunRecord` keeps stage entries as raw JSON and pulls dotted leaf floats out of
them. The run directory and stage come from `--translated` rather than a second flag, since
`run_loop` already encodes both in the export path; an export not sitting in place is refused
rather than guessed at.

Three consequences worth recording, none of them obvious before building it:

1. **A post-hoc metric exists at one stage, not four.** `aggregate` now picks each metric's
   stages independently — every stage where *all* runs record it — instead of demanding one
   stage set for the whole report. So `zero_shot.map50` still gets its four-stage table and
   its trajectory while `faithfulness.*` gets a single stage-3 row and, correctly, no
   trajectory: a gain needs a baseline, and one scored stage is a level, not a movement.
2. **Absent and null had to be separated.** `metric_value` deliberately raises on an explicit
   `null` — that is what `--no-detector` writes, and averaging around a stage that computed
   nothing is the hole it refuses to paper over. "Not scored yet" is a different fact, so
   `metric_is_recorded` keys on the key's *presence*, and `_stage_result_to_json` omits
   `faithfulness` entirely when empty rather than writing `null`. Without that split, every
   run written after this change would have looked like twelve holes.
3. **`StageResult` needed the field, or a resume would eat the result.** `resume` rebuilds
   every completed stage from disk and writes them all back, so a key with no field to land
   in is silently dropped by the next stage — losing a scoring pass that cost GPU time, with
   nothing in the file to show it ever happened.

These three are lifted as the design the numbers were produced under. None carries an F-ID
here; SPEC-MIGRATION-19's no-run records own the build-time findings.

## Results

As recorded in eb61cd1. Stage 3, `runs/e3b-*`, twelve exports scored with the reference
`yolo11s`, n = 6 paired seeds. Per-arm mean ± sd:

| metric | control | loop | loop − control | p |
| --- | --- | --- | --- | --- |
| false-object rate (lower better) | 0.1918 ± 0.0245 | 0.1629 ± 0.0169 | **−0.0289** | 0.156 |
| missed-object rate (lower better) | 0.1962 ± 0.0226 | 0.1592 ± 0.0118 | **−0.0370** | **0.031** |
| detection-consistency (higher better) | 0.7677 ± 0.0220 | 0.7968 ± 0.0091 | **+0.0291** | **0.031** |

Re-render from the run directories without re-scoring: `uv run t2o aggregate --runs 'runs/e3b-*'
--stage 3 --metric faithfulness.false_object_rate faithfulness.missed_object_rate
faithfulness.detection_consistency`

## Findings

### F49 — The pre-registered discriminator is satisfied: false objects did not rise, they fell −0.0289 (−15.1% relative) while mAP50 rose +0.0512 (legacy: M1.2 step 9 finding 12)

The read written above was "flat or falling closes the reward-hacking question outright". It
fell, −0.0289 (−15.1% relative): control 0.1918, loop 0.1629. That is the criterion, and it is
met. Reward hacking predicts the opposite sign — a translator buying mAP50 with invented
components — and every seed's mAP50 went up by +0.0512 (F35) while this went down.

### F50 — False-object rate is the only one of the three independent of the gain, and the one that does not reach significance: p = .156, CI [−.0559, +.0039] (legacy: M1.2 step 9 finding 13)

Missed-object rate (−0.0370) and detection-consistency (+0.0291) both sit exactly on the 2/2⁶
floor, p = 0.031, so all six seeds agreed on both — but "the detector finds more of the real
objects on the translated image" is close to a restatement of "+0.0512 mAP50" in different
units. They corroborate the endpoint; they are not independent evidence about hallucination.
The genuinely independent number is false objects, and its honest reading is **directionally
favourable, not established** (CI [−0.0559, +0.0039] crosses zero, consistent with p = 0.156 —
the two agree here, unlike F46's three cells).

### F51 — No per-metric claim survives a multiplicity correction, and none is being made: Bonferroni α = 0.0167 is below the n = 6 sign-flip floor of 0.031 (legacy: M1.2 step 9 finding 14)

Three metrics at stage 3 gives Bonferroni α = 0.0167, below the sign-flip floor of 0.031 — so
at n = 6 *nothing* can clear it, the same structural limit the per-class note records (F45).
This is why false-object rate is designated the primary and the other two are reported as
corroboration: that split was written down before the numbers arrived (the pre-registered read
under Configuration), not chosen after seeing which ones cleared.

### F52 — The loop arm is markedly less variable on all three rates: sd ratios loop/control 0.69, 0.52, 0.41 — an observation, untested (legacy: M1.2 step 9 finding 15)

False-object, missed-object and detection-consistency respectively. Recorded as an observation
only: no test was run on the variances, n = 6 gives almost no power for one, and this is exactly
the kind of pattern that invites a story after the fact. Worth a pre-registered variance
comparison in the turbo campaign, where it can be a prediction rather than a retrofit — the same
treatment F42 gives the loop arm's `zero_shot.map50` spread.

### F53 — F43's +0.0097 LPIPS gap is bounded from both sides, so `reward_target` stays null and the hold on the turbo campaign is released (legacy: M1.2 step 9, consequences for finding 9 and for the plan)

**Consequence for finding 9 (F43) / §16 caveat 2.** The +0.0097 LPIPS gap is now bounded from
two sides: no cost in the training objective (F43) and *better* object-level faithfulness here
(F49, F50). What is left is a perceptual-texture difference that does not reach the objects,
which is a reportable property of the method rather than an open threat. LPIPS remains in the
tables.

**Consequence for the plan.** `coupling.reward_target` **stays null** — there is no measured
problem for it to respond to, and step 8's reasoning about not changing the dose and its guard
together still applies. The hold on the turbo campaign is **released**.

### F54 — `t2o faithfulness` crashed on its first server image, CUDA predictions against CPU ground truth; every faithfulness test built CPU tensors, so the case was unreachable from the suite (legacy: M1.2 step 9, unnumbered)

**The first server run of this pass crashed on the first image**, in `_greedy_match`:
`RuntimeError: Expected all tensors to be on the same device ... cuda:0 and cpu`. The detector
returns predictions on `--device`, ground truth is read off disk by `load_yolo_labels` on CPU,
and `box_iou` will not mix them. `detections_from_result` now moves predictions to CPU — the
one place device-resident tensors enter the module — and the matcher's masks are allocated on
their inputs' device rather than the default one. CPU is also the right side to match on: the
loop is Python-level over a handful of boxes, so every `.tolist()`/`int()`/`bool()` in it would
otherwise be a device sync.

**Why the test suite did not catch it.** All of M0.5's faithfulness tests build tensors with a
bare `torch.tensor(...)`, so the whole module had only ever been exercised on one device — the
cross-device condition was unreachable from the tests, not merely untested. The fix (cf72c9c)
adds three tests parametrised over `cuda`/`mps`, skipping whichever is absent. `mps` is present
on the dev machine, so the case that only appeared on the server now runs locally on every
commit; reverting the fix reproduces the server's error verbatim (`mps:0 and cpu`). A mismatch
also raises a named `ValueError` at our boundary now instead of surfacing from inside
torchvision.

## Provenance caveats

- **Step 9's first two commands and its checklist rows 2617–2621 live in record 007, not
  here.** Re-running `t2o aggregate` with the trajectory block rewrote step 8 finding 7, weakened
  findings 9 and 10 and added finding 11, and `loss_share.py --terms-only` on the control arm
  gave finding 9's training-loss half. Record 007 lifted those findings at the tag's wording, so
  they are F41, F43, F44, F45 and F46 there and are not re-minted here.
- **The `--csv`'s answer to step 8 finding 6 (F40) is not recorded.** The source says the
  `runs/e3b-tidy.csv` table "answers finding 6 directly" — one outlier seed or spread across all
  six — but states no answer in lines 2614–2762. The CSV sits in the server's `runs/`, not in
  `docs/results/`.
- **Absent vs null** (Configuration, consequence 2) is a build-time finding that
  SPEC-MIGRATION-19 records as a no-run record. Consequences 1 and 3 are lifted here as design
  context without F-IDs, and left for SPEC-MIGRATION-19 to judge.
- **Timeline, from `git log`:** the crashing first run came before cf72c9c (the device fix,
  15:54); `--write-back` landed in ddb9cd2 (16:17); the result was committed in eb61cd1 (16:42),
  all on 2026-08-23. The twelve scored invocations use `--write-back`, so they ran at or after
  ddb9cd2. The exact SHA, the wall clock and the `$DATA` path are not recorded.
- **Matching thresholds are a reading of the code, not a record of the run.** The command
  passes none; the CLI defaults (IoU 0.5, confidence 0.25, `val`, 640) are identical at ddb9cd2
  and eb61cd1, but the source never states the values that ran.
- **Detection-consistency's formula is the module's own judgement call**
  (`src/t2o/metrics/faithfulness.py` docstring): the fraction of what the detector finds on the
  real visible image that it also finds on the translated one. `RESEARCH_FINDINGS.md §9` names
  the metric without a formula.
- **Only the false-object rate's CI is in the source.** The other two rates' CIs, the bootstrap
  method and resample count, and per-seed values are not recorded. The p-values are exact
  sign-flip values at n = 6.
- **F54's code was re-checked at the tag**: the `.cpu()` move at
  `src/t2o/metrics/faithfulness.py:93-101`, the named `ValueError` at `:110-113`, the masks on
  the inputs' device at `:119-120`, and the three parametrised tests at
  `tests/test_faithfulness.py:106-135`. That reverting the fix reproduces `mps:0 and cpu` is
  the source's claim (and cf72c9c's message), not re-executed for this record.
- **F53's "§16 caveat 2" is `PLAN.md §16` as it stood**, graded against the five legacy
  criteria; `docs/goal.md` now holds six (spec-migration Q12). eb61cd1 also edited `PLAN.md`;
  that text migrates with `docs/design.md`, not here.
- **Cross-references are re-keyed to F-IDs**: the source's "finding 9" is F43, "finding 11" is
  F46, "the per-class note" is F45, and "+0.0512" is F35.
- **Nothing later is spliced in.** `git blame` at the tag shows lines 2614–2762 written only by
  42ebf92, 5ee297c, 627963e, 5d1e1b6, cf72c9c, ddb9cd2 and eb61cd1. The first four wrote step 9's
  plan text during step 8, before this pass ran.

## Next

The turbo campaign (M2a), whose hold this record releases. Pre-register F52's variance
comparison there beside F42's, as a prediction rather than a retrofit, and keep false-object rate
as C2's designated primary (F51).
