# 005 — M1.2 steps 5–6: E3's pix2pix arm at `grad_scale: 1.0e-2`, six paired seeds

**Date:** 2026-08-19 · **Task:** M1.2 · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in 6f47aba · **W&B:** group `e3-pix2pix`; run paths not recorded · **Log:** none saved — transcribed from TASKS.md 1966-2119 @ pre-spec-migration

## Command

As written in `TASKS.md`. The `DATA` and `OPTICAL` paths actually passed on the server are not
recorded, nor is how the seed range was split across the two GPUs.

```
DATA=<real paired data.yaml>
OPTICAL=<visible-trained yolo11n.pt>

for s in 0 1 2 3 4 5; do
  for arm in control loop; do
    uv run t2o loop --config experiments/e3_pix2pix_$arm.yaml \
      --data "$DATA" --in-loop-weights "$OPTICAL" --eval-init-weights "$OPTICAL" \
      --seed $s --name e3-$arm-s$s --wandb --device cuda:0
  done
done
uv run t2o aggregate --runs 'runs/e3-*' --stage 3 --metric zero_shot.map50
```

`--name` is mandatory per run: `runs/<runtime.name>` has no seed component, so a bare
`--seed 1` would write straight over seed 0's `metrics.json`, `config.yaml` and every
`stage*/translator_*.pt`. Both arms were launched with identical flags — a difference in any
of them is a confound, the same one `test_control_and_loop_configs_differ_only_by_design`
guards inside the files. `detector.reference.weights` is already correct in both files and
needs no flag. With two GPUs the seed range is split across two shells via
`CUDA_VISIBLE_DEVICES`, keeping each *pair* on one card, since the comparison is per seed.

## Configuration

E3 exactly as designed in record 004: `experiments/e3_pix2pix_control.yaml` against
`experiments/e3_pix2pix_loop.yaml`, differing only in `coupling.task_weights` (`[0, 0, 0, 0]`
vs `[0, 1, 2, 3]`) and `runtime.name`.

- 12 runs: 2 arms × 6 paired seeds (0–5), 400 warm-started translator epochs in both arms.
- Judge: record 004's independent `yolo11s` (`detector.reference.weights`), never the in-loop
  `yolo11n`.
- `coupling.grad_scale: 1.0e-2` in both files. Loss weights `l2: 1.0` + `lpips: 5.0` +
  `gan: 1.0`.
- Test: paired loop − control per stage, exact two-sided sign-flip over 2⁶ assignments, with a
  bootstrap CI.
- Decision rule, fixed before the campaign ran (record 004): *"If the stage-3 effect is not
  clearly larger than the stage-0 difference, E3 is negative and gets reported that way."*

## Results

As recorded in 6f47aba. Per-arm mean ± sd:

| metric | arm | stage 0 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- | --- | --- |
| `zero_shot.map50` | control | 0.7632 ± .0408 | 0.7679 ± .0560 | 0.7913 ± .0352 | 0.7987 ± .0309 |
| `zero_shot.map50` | loop | 0.7568 ± .0438 | 0.7920 ± .0483 | 0.8151 ± .0390 | 0.8057 ± .0324 |
| `fidelity.lpips` | control | 0.3104 ± .0098 | 0.2964 ± .0115 | 0.2893 ± .0124 | 0.3009 ± .0139 |
| `fidelity.lpips` | loop | 0.3200 ± .0293 | 0.3021 ± .0211 | 0.2968 ± .0196 | 0.2989 ± .0099 |
| Switch AP50 | control | 0.4131 ± .1091 | 0.4434 ± .1488 | 0.5064 ± .0710 | 0.5310 ± .0693 |
| Switch AP50 | loop | 0.4561 ± .1076 | 0.5519 ± .0398 | 0.5733 ± .0539 | 0.5626 ± .0939 |

Paired loop − control, exact two-sided sign-flip over 2⁶ assignments, with bootstrap CI:

| metric | stage 0 (null) | stage 1 | stage 2 | **stage 3 (headline)** |
| --- | --- | --- | --- | --- |
| `zero_shot.map50` | −0.0063, p=.875, [−.045, +.040] | +0.0241, p=.281, [−.011, +.058] | +0.0238, p=.312, [−.014, +.055] | **+0.0070, p=.656, [−.020, +.032]** |
| `fidelity.lpips` | +0.0096, p=.875, [−.011, +.040] | +0.0056, p=.344, [−.003, +.015] | +0.0075, p=.500, [−.003, +.023] | **−0.0019, p=.812, [−.016, +.011]** |
| Switch AP50 | +0.0430, p=.406, [−.050, +.136] | +0.1085, p=.031, [+.016, +.214] | +0.0669, p=.094, [+.019, +.114] | **+0.0316, p=.562, [−.046, +.115]** |

Re-render from the results files without re-running, on the server:
`uv run t2o aggregate --runs 'runs/e3-*' --stage 3 --metric zero_shot.map50`

## Findings

### F15 — E3's pix2pix arm is negative on its pre-registered endpoint (legacy: M1.2 step 6, headline)

Stage 3 is **+0.0070** (p = .656, CI [−.020, +.032]) against a stage-0 null of **−0.0063**
(p = .875) — the same magnitude. By the rule fixed before the campaign ran, E3's pix2pix arm is
negative, and PLAN.md §16's causality criterion is not satisfied.

### F16 — The stage-0 null control behaved as a null, so this is a measurement and not a broken campaign (legacy: M1.2 step 6 finding 1)

−0.0063 at p = .875, with a CI straddling zero almost symmetrically ([−.045, +.040]). Step 2's
RNG audit and step 2b's `workers` fix were exactly the work that makes the stage-0 contrast
interpretable, and it came out where it had to.

### F17 — Step 1's noise floor, taken at n = 1, was right: per-seed sd ≈ 0.053 (legacy: M1.2 step 6 finding 2)

Step 1 measured 0.0591 from one paired λ=0 draw (F13). Stage 0's CI half-width here
(0.042 ≈ 1.96·SE) implies a per-seed sd of **≈0.053**. The number the whole six-seed budget was
justified against holds.

### F18 — Six paired seeds bought ±0.026 resolution: the claim is "no effect larger than about +3 mAP50 points", not "no effect" (legacy: M1.2 step 6 finding 3)

±0.026 is 2.3× tighter than a single paired draw. A true +0.02 sits inside the stage-3 CI
([−.020, +.032]). An unmeasurable effect cannot establish causality, so §16's criterion is
still not satisfied — but the writeup must state the bound rather than assert a zero.

### F19 — There is no dose-response in λ at this `grad_scale` (legacy: M1.2 step 6 finding 4)

The paired zero-shot difference goes 0 → +0.024 → +0.024 → +0.007 across stages 0–3, peaking
at the *smallest* nonzero λ and decaying as λ triples. Both arms converge over stages (control
0.7632 → 0.7987, loop 0.7568 → 0.8057), which is what a shared 400-epoch budget plus a
per-stage detector fine-tune produces on its own. Extends step 1's finding (F12): λ_det's effect
is not monotone in the *stage*, and now not monotone in the *weight* either.

### F20 — λ_det is fidelity-neutral, with a bound: M1.1's reward-hacking question is closed at n = 6 (legacy: M1.2 step 6 finding 5)

λ_det moves stage-3 LPIPS by **−0.0019**, CI [−0.016, +0.011], p = .812. The detection gradient
at the `grad_scale` PLAN.md §8 prescribes neither degrades nor improves perceptual fidelity.
That is worth reporting whatever happens to C1.

### F21 — Switch, where step 1 said the gain lived, does not survive; no per-class claim is attainable in this design (legacy: M1.2 step 6 finding 6)

Switch's own stage-0 null is **+0.043**, larger than its stage-3 effect of **+0.032**; the loop
arm is simply noisier on the rarest class. The one cell under 0.05 (stage 1, +0.1085) is 1 of 12
reported tests sitting *exactly* on the n = 6 p-floor of 0.031, where Bonferroni would need
0.0042 — so **no per-class claim is attainable in this design at any effect size.** Logged as a
hypothesis to pre-register for the turbo arm, not as a finding. One observation, not a result:
at stage 1 the loop arm's Switch sd is 0.0398 against the control's 0.1488, which may be
variance reduction on the hardest class rather than mean improvement — but it does not
reappear at stages 2 or 3.

### F22 — E8 is not an escape route: lowering the annotation fraction makes E3's arms more alike, not less (legacy: M1.2 step 6 finding 7)

`data.annotation_fraction` gates only the batch's `cls`/`bboxes` (`data/dataset.py:188-191`) —
that is, only the *loop* arm's own supervision, since the control never reads annotations at
all. A low-annotation E3 therefore cannot recover a causality claim that failed at full
annotation. E8 remains a valid question about the *translator's* data efficiency (PLAN.md §11),
but it answers a different question than C1's. The source notes that an earlier note in
`TASKS.md` saying otherwise was wrong.

### F23 — λ_det was never 1/2/3: the ramp the optimiser saw was 0.01/0.02/0.03, so the campaign cannot separate a mechanism null from a dose null (legacy: M1.2 step 6, unnumbered)

`translators/pix2pix.py:161-163` adds `task_weight * detection` to the total, while
`coupling/detection_loss.py:103` has *already* multiplied by `grad_scale`. Both E3 configs set
`grad_scale: 1.0e-2`, so the effective ramp was **0.01 / 0.02 / 0.03**, against `l2: 1.0` +
`lpips: 5.0` + `gan: 1.0`. That downscale is PLAN.md §8's anti-reward-hacking guardrail — and
M1.1, then F20, established there was no hack to guard against. The guardrail, set aggressively
against a problem that did not materialise, is the prime suspect for F15's null. So the campaign
cannot yet distinguish "coupling does not help pix2pix at this data scale" from "λ_det was too
small to move the optimiser".

### F24 — W&B silently dropped stages 1–3's generator loss curves; no number E3 is read off was affected (legacy: M1.2 step 5, unnumbered)

Fourteen hours in, the server logged thousands of `Tried to log to step 46 that is less than the
current step 258`. `Trainer.train()` logged with an explicit `step=epoch`, and `epoch` restarts
at 0 in every stage while W&B's counter only increases; everything else in a loop run logs
implicitly and rides the auto-increment, so the counter was already at 151 when stage 1's
epoch 0 arrived, and all 100 epochs were rejected, and so on per stage.

Diagnosed as cosmetic, on three grounds: the per-epoch history is in `metrics.json` regardless
(`asdict(StageResult)` carries every `EpochStats`, written after each stage); `t2o aggregate`
reads `metrics.json` plus the `config.yaml` snapshot and never touches W&B; and the stage-level
metrics in W&B (`stage*/zero_shot/*`, `stage*/fidelity/*`, `stage*/detector/*`) all log
implicitly and were all accepted. The campaign was not stopped.

The fix (accfe56) drops the explicit `step=`, carries the epoch as a value, and adds
`Trainer(metric_prefix=...)`, set to `stage{N}` by `run_loop`. The prefix repairs a second
latent defect the step problem masked: the trainer's keys were the only ones in a loop run not
namespaced by stage, so stage 1's `train/l2` would have plotted over stage 0's even with
monotonic steps. `test_epoch_metrics_are_namespaced_and_never_carry_an_explicit_step` pins
both halves. It was not pulled onto the server until the campaign finished: logging touches no
RNG, but pulling mid-campaign would have left seeds 0–2 and 3–5 on different trees.

## Provenance caveats

- **Three annotations in the source postdate this result** and are not lifted here. All three
  were added in 5ee297c (2026-08-23, M1.2 step 8): step 6's checkbox text "Superseded by step 8
  — this campaign's dose was 2.3% of the objective (step 7) …"; the headline's aside "at
  `grad_scale: 1.0e-2` — read step 7 and step 8 before quoting anything below"; and finding 4's
  parenthetical giving step 8's 0 → +0.028 → +0.036 → +0.051 ramp at `grad_scale: 0.15`. Their
  outcomes belong to records 006 and 007. This record carries the 6f47aba wording.
- The source records no wall clock. "Fourteen hours in" (F24) and record 004's ~72 GPU-hour
  estimate are not a measured total.
- The SHA the campaign executed at is not recorded. It ran on a tree **before** accfe56
  (2026-08-17), since that fix was held off the server until the campaign finished; both arms
  and all seeds ran on the same tree.
- F22 and F23 are readings of the source code at the time, cited by the line numbers the source
  gave, not observations of a run. They were not re-checked against the code at this record's
  writing.
- F17's sd ≈ 0.053 is back-derived from a CI half-width, not computed from the per-seed values,
  which the source does not list.
- The p-values are exact sign-flip values at n = 6; the CIs are bootstrap, method and resample
  count not recorded.
- The re-render command assumes `runs/e3-*` still exists on the server; that is not checked.
  Step 8's runs used an `e3b-` prefix so as not to overwrite these, but step 8's probe directory
  `runs/e3-probe-g015` also matches the glob.

## Next

Settle F23 before spending more GPU time: measure the detection term's actual share of the
objective at `grad_scale: 1.0e-2` (M1.2 step 7). **Do not launch M2a's turbo arm until it is
settled**: same `grad_scale`, same possible null, another ~72 GPU-hours. The two readings of F15
— "coupling does not help pix2pix at this data scale" and "λ_det was too small to move the
optimiser" — have very different consequences, and step 7 exists to tell them apart.
