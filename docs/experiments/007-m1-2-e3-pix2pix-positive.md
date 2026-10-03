# 007 — M1.2 step 8: the dose recalibrated to `grad_scale: 0.15`, then E3's pix2pix arm re-run at it, six paired seeds

**Date:** 2026-08-23 · **Task:** M1.2 · **Machine:** Windows server, 2× A100 40 GB · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in 5ee297c · **W&B:** group `e3-pix2pix-g015`; run paths not recorded; the probe ran without `--wandb` · **Log:** none saved — transcribed from TASKS.md 2224-2613 @ pre-spec-migration

## Command

As written in `TASKS.md`. Three things ran, in order: a one-seed probe, a matched-epoch read of
record 005's seed-0 loop run, and the twelve-run campaign. The `DATA` and `OPTICAL` paths
actually passed on the server are not recorded.

The probe — both knobs are already CLI overrides (`cli.py::_OVERRIDES`), and `config_hash`
covers `coupling`, so a probe run cannot be confused with a campaign run on resume:

```bash
uv run t2o loop --config experiments/e3_pix2pix_loop.yaml \
  --data "$DATA" --in-loop-weights "$OPTICAL" --eval-init-weights "$OPTICAL" \
  --grad-scale 0.15 --epochs 25 --seed 0 --name e3-probe-g015 --device cuda:0
uv run python scripts/loss_share.py --runs 'runs/e3-probe-g015'
```

The matched-epoch read, putting record 005's `e3-loop-s0` on the probe's 25-epoch window:

```bash
uv run python scripts/loss_share.py --runs 'runs/e3-loop-s0' --first-epochs 25
```

The campaign, in PowerShell since the server is native Windows. Split across two cards by seed,
never by arm — a pair must stay on one card or the paired difference absorbs whatever differs
between the GPUs:

```powershell
# shell A
foreach ($s in 0,1,2) { foreach ($arm in 'control','loop') {
  uv run t2o loop --config "experiments/e3_pix2pix_$arm.yaml" `
    --data $DATA --in-loop-weights $OPTICAL --eval-init-weights $OPTICAL `
    --seed $s --name "e3b-$arm-s$s" --group e3-pix2pix-g015 --wandb --device cuda:0
} }

# shell B -- identical but for the seed range and the card
foreach ($s in 3,4,5) { foreach ($arm in 'control','loop') {
  uv run t2o loop --config "experiments/e3_pix2pix_$arm.yaml" `
    --data $DATA --in-loop-weights $OPTICAL --eval-init-weights $OPTICAL `
    --seed $s --name "e3b-$arm-s$s" --group e3-pix2pix-g015 --wandb --device cuda:1
} }
```

```bash
uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 \
  --metric zero_shot.map50 fidelity.lpips zero_shot.per_class_ap50.Switch
uv run python scripts/loss_share.py --runs 'runs/e3b-loop-*'
```

and `scripts/loss_share.py --terms-only` on the control arm, whose exact invocation the source
does not write out.

Concurrency is safe on disk: `run_loop` passes `project=stage_dir / "detector"`, so every
ultralytics write is scoped to one run's own stage directory, and each stage's detector trains
on that stage's own exported images — there is no shared label `.cache` for the two processes to
race on. The one machine-global file in play is ultralytics' `settings.json`, so shell B was to
start a minute after shell A. Both shells must carry identical `$DATA`/`$OPTICAL`: those flags
are machine-specific paths rather than config, so a difference between the shells is a confound
that `test_control_and_loop_configs_differ_only_by_design` cannot see.

## Configuration

**The probe** — a calibration, not a measurement. `experiments/e3_pix2pix_loop.yaml`, seed 0,
`--grad-scale 0.15` (record 006's candidate, F29), `--epochs 25` against the campaign configs'
`epochs_per_stage: 100`, all four stages kept because the ramp is what needs measuring. No
control arm. Read three things off it, **never mAP**:

1. **Share** per stage. In the 20–30% band → proceed. Still under ~10% → raise `grad_scale` by
   the shortfall ratio and re-probe. Over ~40% → lower it.
2. **Fidelity**, from the run's own `stage*/fidelity/lpips`, read against its own stage 0 and
   never against E3's ~0.30. Rising sharply within the run → set `coupling.reward_target` and
   re-probe.
3. **Stability**: `loss_det` should fall rather than oscillate, and `loss_gan` should not
   diverge.

**The campaign** — record 005's E3 verbatim except for two things: `coupling.grad_scale: 0.15`
in **both** `experiments/e3_pix2pix_{control,loop}.yaml` (5f2c9a5; the control's value is inert,
since no detector is built at weight 0, but must match, which
`test_control_and_loop_configs_differ_only_by_design` enforces), and the run names, `e3b-`
rather than `e3-` (F48).

- 12 runs: 2 arms × 6 paired seeds (0–5), 400 warm-started translator epochs in both arms.
  Seeds 0–2 on `cuda:0`, 3–5 on `cuda:1`.
- Judge: record 004's independent `yolo11s`.
- `coupling.reward_target` stays `null`. Setting it now would change the dose and its guard
  together, and any fidelity result would then be unattributable; it is the response to a
  measured fidelity cost, not a precaution against one.
- Test: paired loop − control per stage, exact two-sided sign-flip over 2⁶ assignments, with a
  bootstrap CI. Decision rule, unchanged from record 005: the stage-3 effect must be clearly
  larger than the stage-0 difference.
- Added after the result was seen: the within-arm trajectory (`TrajectoryResult`,
  `analysis/aggregate.py`), each arm's gain over its own stage 0, differenced. A sensitivity
  analysis, never the endpoint (F41).

## Results

As recorded in efead46 and 5f2c9a5 (the probe, 2026-08-19) and in 5ee297c, 627963e and 5d1e1b6
(the campaign, 2026-08-23).

### The probe, `runs/e3-probe-g015`

```
stage runs epochs   w lambda_eff  loss_l2 loss_lpips loss_gan loss_det loss_total   w*det  share
    0    1     25 0.0     0.0000   0.0492     2.1006   1.7234       --     3.8732  0.0000   0.0%
    1    1     25 1.0     0.1500   0.0430     1.9252   1.6868   0.4462     4.1012  0.4462  10.9%
    2    1     25 2.0     0.3000   0.0461     2.0174   2.4022   0.4456     5.3570  0.8913  16.6%
    3    1     25 3.0     0.4500   0.0431     1.9002   2.3294   0.4136     5.5136  1.2409  22.5%
```

### Matched-epoch comparison: probe vs `e3-loop-s0 --first-epochs 25`

Same seed, same arm, same length; `grad_scale` 0.01 vs 0.15. Stage 0 is λ-inert, so its residual
is the run-to-run noise floor in loss space:

| stage 0, λ inert | `loss_l2` | `loss_lpips` | `loss_gan` | fidelity total |
| --- | --- | --- | --- | --- |
| gap between two nominally identical runs | +3.4% | +3.6% | +11.8% | +7.1% |

### The campaign, `runs/e3b-*`

Twelve runs, six paired seeds, 400 warm-started epochs in both arms, same independent `yolo11s`
judge, `grad_scale: 0.15`. Per-arm mean ± sd:

| metric | arm | stage 0 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- | --- | --- |
| `zero_shot.map50` | control | 0.7579 ± .0372 | 0.7519 ± .0238 | 0.8071 ± .0263 | 0.7975 ± .0330 |
| `zero_shot.map50` | loop | 0.7182 ± .0907 | 0.7799 ± .0887 | 0.8427 ± .0215 | 0.8487 ± .0140 |
| `fidelity.lpips` | control | 0.3304 ± .0364 | 0.3165 ± .0150 | 0.2992 ± .0101 | 0.2905 ± .0054 |
| `fidelity.lpips` | loop | 0.3272 ± .0299 | 0.3230 ± .0414 | 0.2997 ± .0205 | 0.3002 ± .0106 |
| Switch AP50 | control | 0.4000 ± .0943 | 0.4340 ± .0726 | 0.5601 ± .0442 | 0.5271 ± .0787 |
| Switch AP50 | loop | 0.4511 ± .1534 | 0.5007 ± .1620 | 0.6211 ± .0579 | 0.6306 ± .0335 |

Paired loop − control, exact two-sided sign-flip over 2⁶ assignments, with bootstrap CI:

| metric | stage 0 (null) | stage 1 | stage 2 | **stage 3 (headline)** |
| --- | --- | --- | --- | --- |
| `zero_shot.map50` | −0.0397, p=.469, [−.127, +.037] | +0.0280, p=.500, [−.051, +.078] | +0.0357, p=.062, [+.013, +.055] | **+0.0512, p=.031, [+.025, +.081]** |
| `fidelity.lpips` | −0.0033, p=.844, [−.049, +.037] | +0.0065, p=.844, [−.030, +.045] | +0.0005, p=.938, [−.017, +.018] | **+0.0097, p=.125, [+.002, +.017]** |
| Switch AP50 | +0.0511, p=.594, [−.119, +.201] | +0.0667, p=.406, [−.098, +.178] | +0.0610, p=.062, [+.024, +.093] | **+0.1035, p=.094, [+.025, +.183]** |

Within-arm trajectory of `zero_shot.map50`, gain over stage 0 (sensitivity analysis):

| stage | control gain | loop gain | difference | p | 95% CI |
| --- | --- | --- | --- | --- | --- |
| 1 | −0.0060 | +0.0617 | +0.0677 | .312 | [−.061, +.184] |
| 2 | +0.0491 | +0.1245 | +0.0754 | .156 | [+.003, +.160] |
| **3** | **+0.0396** | **+0.1305** | **+0.0909** | **.094** | **[+.019, +.181]** |

The detection term's share at full length, `scripts/loss_share.py --runs 'runs/e3b-loop-*'`:

```
stage runs epochs   w lambda_eff  loss_l2 loss_lpips loss_gan loss_det loss_total   w*det  share
    0    6    100 0.0     0.0000   0.0419     1.8288   1.6460       --     3.5167  0.0000   0.0%
    1    6    100 1.0     0.1500   0.0362     1.6131   1.8033   0.3834     3.8359  0.3834  10.0%
    2    6    100 2.0     0.3000   0.0334     1.5358   1.7464   0.3181     3.9518  0.6362  16.1%
    3    6    100 3.0     0.4500   0.0317     1.4905   1.8211   0.2758     4.1707  0.8274  19.8%
```

Training-loss fidelity terms, both arms (`--terms-only` supplies the control; 6 runs, 100
epochs, pooled):

| stage | arm | `loss_l2` | `loss_lpips` | `loss_gan` | fidelity total |
| --- | --- | --- | --- | --- | --- |
| 0 | control | 0.0423 | 1.8422 | 1.6799 | 3.5645 |
| 0 | loop | 0.0419 | 1.8288 | 1.6460 | 3.5167 |
| 3 | control | 0.0331 | 1.5028 | 1.8686 | 3.4045 |
| 3 | loop | 0.0317 | 1.4905 | 1.8211 | 3.3433 |

| term | stage-0 gap (null) | stage-3 gap | λ-attributable |
| --- | --- | --- | --- |
| `loss_l2` | −0.95% | −4.23% | −3.28 pp |
| `loss_lpips` | −0.73% | −0.82% | **−0.09 pp** |
| `loss_gan` | −2.02% | −2.54% | −0.52 pp |

Re-render from the results files without re-running, on the server:
`uv run t2o aggregate --runs 'runs/e3b-*' --stage 3 --metric zero_shot.map50 fidelity.lpips zero_shot.per_class_ap50.Switch`
(the trajectory block prints beside the paired one) and
`uv run python scripts/loss_share.py --runs 'runs/e3b-loop-*'`.

## Findings

### F30 — `grad_scale: 0.15` lands the dose: the probe's share is 10.9 / 16.6 / 22.5% against a predicted 12 / 21 / 27 (legacy: M1.2 step 8, probe result)

**In band.** The ramp now spans a real dose range rather than step 7's 0.9–2.3% (F25), and it
lands close to F29's extrapolation. Stability held: four stages completed, `loss_gan` did not
diverge, nothing collapsed. `grad_scale: 0.15` is the calibrated value.

**Nothing about the probe's mAP is usable**, and not only for F17's reason (per-seed sd ≈ 0.053
cannot resolve the +0.02 being chased at n = 1): at 25 epochs the translator is a quarter
trained, so its 0.7451 → 0.8158 stage ramp measures training length as much as anything else.
That confound is exactly what the control arm exists to subtract, and the probe has no control
arm.

### F31 — The epoch-length trap: a 25-epoch probe reads above a 100-epoch run in every term at once, which looks like a changed condition rather than a shorter one (legacy: M1.2 step 8, probe result)

The probe ran `--epochs 25`; `epochs_per_stage` in both E3 configs is **100**. Every figure
`loss_share.py` reports is a mean over epochs, and the opening epochs are the loud ones. Two
false alarms came out of that before the cause was found:

* Stage 0 is provably inert to `grad_scale` (`translators/pix2pix.py::fit` gates the detector on
  `task_weight > 0.0`), so the probe's stage 0 should reproduce `e3-loop-s0`'s. It did not —
  `loss_lpips` 2.1006 vs 1.8608, val `map50` 0.7451 vs 0.7951, val `lpips` 0.3451 vs 0.3020.
  That looked like a determinism failure large enough to invalidate step 2/2b. It was the epoch
  count. **Determinism is not in question; nothing about steps 2, 2b or 6 changes.**
* Raw detection loss (`loss_det / grad_scale`) reads 2.97 / 2.97 / 2.76 here against step 7's
  2.98 / 2.75 / 2.61 (F29) — i.e. *worse* under 15× the pressure. **Withdrawn**: a 25-epoch
  mean against a 100-epoch mean is not a comparison.

The `loss_gan` observation (this run rises 35% across stages 0→3 where `e3-loop-s0` rises 12%)
is within-run on both sides so it survives the level offset, but 25 epochs leaves less room to
re-equilibrate; logged as a watch item, not a finding (closed by F39). `--first-epochs`
(efead46) exists so this cannot recur silently: it truncates the pool, and the `epochs` column
puts the length on the row.

### F32 — Two GPU runs at one seed do not reproduce, by design: the loss-space noise floor is +3.4 / +3.6 / +11.8 / +7.1% (legacy: M1.2 step 8, matched-epoch comparison)

`--first-epochs 25` on `e3-loop-s0` puts it and the probe on equal terms. Stage 0 is λ-inert, so
the gap there between two nominally identical runs is +3.4% `loss_l2`, +3.6% `loss_lpips`,
+11.8% `loss_gan`, +7.1% fidelity total.

**This is expected and by design, not a determinism defect.** `seeding.py`'s docstring is
explicit that `torch.use_deterministic_algorithms` and the cuDNN flags are deliberately absent,
because "cuDNN non-determinism is precisely the variance E3 exists to quantify across seeds";
`test_training_is_bit_identical_at_any_worker_count` is a CPU test of the data pipeline, not of
kernels. F17 already priced this in mAP space (sd ≈ 0.053); this is its counterpart in loss
space, and the resolution limit for every `loss_share.py` comparison from here on.

### F33 — Under 15× the weight, raw detection loss does not detectably move at 25 epochs: −5.0 / +3.9 / +1.4%, every one inside the floor (legacy: M1.2 step 8, matched-epoch comparison)

`loss_det / grad_scale` goes 3.13 → 2.97, 2.86 → 2.97, 2.72 → 2.76 across stages 1–3 (0.01 →
0.15). Not "λ_det fails to reduce its own loss"; the comparison cannot resolve anything under
~10% (F32). Nor is this the endpoint: the in-loop detector is the frozen `yolo11n`, and E3's
claim rests on the independent `yolo11s` judge.

### F34 — The dose trades fidelity improvement: within-run `loss_lpips` falls 25.2% at `grad_scale` 0.01 but only 9.5% at 0.15 (legacy: M1.2 step 8, matched-epoch comparison — withdrawn in the source)

As written in 5f2c9a5: *"Within-run across stages 0→3, `loss_lpips` falls 25.2% at
`grad_scale` 0.01 but only 9.5% at 0.15 — a 15.7-point gap against a 3.6% floor, the one
comparison here that clears it by a wide margin. The fidelity total follows: −6.0% versus
+10.3%. `loss_gan` (+20.1% vs +35.2%) points the same way but sits close enough to its own
11.8% floor to stay a watch item."* And: *"So the dose is doing something measurable, and what
it measurably does is trade fidelity improvement."*

The source struck both through in 5d1e1b6: **"Withdrawn — the campaign refutes it."** At 100
epochs with six runs per arm, `loss_lpips` falls **18.42% in the control and 18.50% in the loop
arm**: no trade at all (F43). This bullet compared one 25-epoch probe against a 100-epoch run
truncated to its first 25, at one seed and across two `grad_scale` values — three confounds
where the campaign has none, and it needed a control arm it did not have. The epoch-length trap
this very section documents (F31) caught it after all.

### F35 — E3's pix2pix arm is positive at the calibrated dose: stage-3 paired zero-shot mAP50 +0.0512, p = .031, CI [+.025, +.081] (legacy: M1.2 step 8 finding 1)

The headline clears the pre-registered endpoint, at the only p this design can reach.
p = 0.0312 is exactly 2/2⁶ — the sign-flip floor — so all six seeds moved the same way *and* the
observed mean was the most extreme of all 64 assignments. Nothing about n = 6 can produce a
smaller number; step 5's six-seed budget was sized for exactly this outcome. F15 is the same
design at `grad_scale: 1.0e-2`.

### F36 — Dose-response appeared: the paired difference goes 0 → +0.0280 → +0.0357 → +0.0512, monotone in λ (legacy: M1.2 step 8 finding 2)

At `grad_scale: 1.0e-2` it went 0 → +0.024 → +0.024 → +0.007, peaking at the *smallest* nonzero
λ and decaying (F19). This is the single most persuasive line in the table, because it is not
something a lucky draw produces: the same six seeds, the same machinery, the same judge, ordered
by dose.

### F37 — The mechanism is visible in loss space: raw detection loss runs 2.56 → 2.12 → 1.84, stage 3 about 30% below the uncalibrated campaign (legacy: M1.2 step 8 finding 3)

`loss_det / grad_scale`, from `scripts/loss_share.py --runs 'runs/e3b-loop-*'`, across stages
1–3, against the uncalibrated campaign's 2.98 → 2.75 → 2.61 — stage 3 about **30% lower**, well
outside the ~10% loss-space noise floor (F32). F33 could only report that raw detection loss
"does not detectably move" under 15× the weight, measured over 25 epochs; at full length it
moves, and downward.

### F38 — The dose landed where the probe said: 10.0 / 16.1 / 19.8% at 100 epochs, and was not re-tuned (legacy: M1.2 step 8 finding 4)

Against the probe's 10.9 / 16.6 / 22.5% at 25 epochs (F30); the epoch-length trap barely bit.
The top of the ramp sits just under the 20–30% target band. **Not re-tuned**: the band was a
calibration target, not an endpoint, and moving `grad_scale` again after seeing the mAP result
would make the campaign unreportable. Record the achieved share; do not chase the band.

### F39 — `loss_gan` did not diverge: +10.6% across stages 0→3 against the uncalibrated campaign's +11.1% (legacy: M1.2 step 8 finding 5)

Indistinguishable, closing the probe's watch item. The probe's alarming +35% was the 25-epoch
artifact F31 suspected it of being, and is now confirmed as one.

### F40 — The stage-0 null drew wide, −0.0397 (p = .469, CI [−.127, +.037]); the stage-3 effect is 1.3× that, not clearly larger on that reading alone (legacy: M1.2 step 8 finding 6)

Against the uncalibrated campaign's −0.0063, with the loop arm's stage-0 sd at **0.0907 versus
the control's 0.0372**. This is the result's main soft spot. Stage 0 is provably λ-inert in
*both* arms — `build_detection_loss` returns `None` at weight ≤ 0 so no `FrozenDetector` is ever
constructed (`coupling/schedule.py`), `translators/pix2pix.py:161` gates the term on
`task_weight > 0.0`, and `config_hash` is read only by `Trainer`'s resume drift check and seeds
nothing — so the arms run the identical computation there and this is an unlucky draw, not a
code path. But the decision rule says the stage-3 effect must be *clearly larger* than the
stage-0 difference, and 0.0512 against 0.0397 is 1.3× by magnitude.

### F41 — The within-arm trajectory corroborates the effect but does not confirm it: +0.0909, p = .094, CI [+.019, +.181] (legacy: M1.2 step 8 finding 7)

Over the same 400-epoch budget the loop arm travels 3.3× further (control +0.0396, loop
+0.1305 at stage 3). Two readings, and the second is the one that must not be skipped:

* **It says the effect is not an artifact of the stage-0 offset.** If the finish-line +0.0512
  were the offset showing through, removing the offset would collapse the contrast toward zero.
  It grows instead, and grows *monotonically in dose* (+0.068 → +0.075 → +0.091) — a second
  dose-response, in a contrast the offset cannot touch, independent of F36's.
* **It cannot rescue significance, and p = 0.094 is the honest number.** Differencing two paired
  quantities adds their variances (`Var(A−B) = Var(A) + Var(B) − 2Cov`), and the stage-0
  difference is precisely the noisy one, so subtracting it injects exactly the noise F40 is
  about. **A larger point estimate at a worse p is not a stronger result.**

So the defensible sentence is *"the pre-registered endpoint is significant at p = 0.031, and a
contrast immune to the stage-0 draw points the same way, larger, at p = 0.094"* — never
"+0.0909, CI excludes zero". **It was added after seeing the data and is a sensitivity analysis,
never the endpoint.** The one mitigating argument the paper may make: the decision rule already
directs stage 3 to be read against stage 0, so this formalises the rule's own arithmetic rather
than introducing a second hypothesis.

**F40 is therefore narrowed, not closed.** The stage-0 draw is still the result's softest edge.
All three available readings agree in direction — the stage-0 null's CI [−.127, +.037] contains
zero comfortably, stage 3's [+.025, +.081] excludes it, and the trajectory grows monotonically
in dose — and F36 and F37 do not depend on stage 0 at all.

### F42 — Variance collapses in the loop arm as λ rises: `zero_shot.map50` sd 0.0907 → 0.0887 → 0.0215 → 0.0140, 2.4× tighter than the control by stage 3 (legacy: M1.2 step 8 finding 8)

The control stays in 0.024–0.037 throughout. Record 005 saw this on Switch alone at stage 1
(F21) and logged it as a hypothesis to pre-register; it now appears on the headline metric and
across the ramp. **Still not a claim** — regression from an unlucky stage-0 draw predicts some of
it — but it is the pre-registerable hypothesis for the turbo arm.

### F43 — Validation LPIPS is +0.0097 worse at stage 3 (p = .125), but training fidelity is untouched: a generalisation gap, not a trade, bounded at ~0.02 LPIPS (legacy: M1.2 step 8 finding 9)

As first written (5ee297c): *"Fidelity is no longer neutral: +0.0097 LPIPS at stage 3, CI
[+.002, +.017]. The uncalibrated campaign's −0.0019 ± 0.016 bound does not survive the dose.
Both arms still improve over the ramp (control 0.3304 → 0.2905, loop 0.3272 → 0.3002); the loop
arm improves **less** — 0.0270 against 0.0399, so **about a third** of the control's fidelity
gain is traded away. Step 8 predicted exactly this ("what the dose measurably does is trade
fidelity improvement")."* The sign-flip test gives p = 0.125 while the bootstrap CI excludes
zero, so it is *suggestive, not significant at n = 6* — report both. And detection rising while
fidelity falls **is** the reward-hacking signature (PLAN.md §8), which the independent judge
argues against but does not measure.

**The trajectory weakens it further.** Within-arm, the control's LPIPS improves −0.0399 and the
loop's −0.0270, a difference of **+0.0129, p = 0.562, CI [−.025, +.054]** — straddling zero.
Directionally consistent across both contrasts, statistically established by neither.

**And in training loss there is no cost at all.** Within-arm over the full ramp, `loss_lpips`
falls **18.42%** in the control and **18.50%** in the loop arm; the λ-attributable change is
−3.28 pp `loss_l2`, **−0.09 pp** `loss_lpips`, −0.52 pp `loss_gan`, and the loop arm's stage-3
fidelity-only total is 3.3433 against the control's 3.4045. The detection term is not competing
with the perceptual one for the optimiser's attention at this dose — it is added on top.

Identical training-loss improvement with slightly worse validation transfer is a
**generalisation gap**, not the optimiser trading the perceptual term away — which is the
specific mechanism PLAN.md §8's guardrails were written against and the specific thing
`reward_target` would fix. `reward_target` is therefore **not** indicated. The strongest
defensible claim is a bound: **λ_det at this dose costs nothing measurable in training fidelity
and at most ~0.02 LPIPS at evaluation.** It supersedes F20's −0.0019 ± 0.016 bound, which was
measured at a fifteenth of the dose, and refutes F34.

What is still open is the export path: training loss is computed on float tensors, while the
evaluation LPIPS is scored on uint8 PNGs written through `export.py`'s per-image `clamp`
normalisation (PLAN.md §12). That is the one place a difference could appear between two arms
whose training fidelity matches, and it is untested.

### F44 — A better loss-space noise floor: pooled 6-vs-6, the stage-0 gap is 0.95 / 0.73 / 2.02 / 1.34% (legacy: M1.2 step 8 finding 9, noise floor)

Supersedes F32's floor, which was measured from two single runs at one seed (+3.4% l2 / +3.6%
lpips / +11.8% gan / +7.1% total). The pooled figures are about what averaging six runs per side
predicts from the single-pair figure (1/√6 ≈ 0.41). Use the pooled numbers for any
campaign-scale comparison; the single-run floor still governs probe-vs-probe reads.

### F45 — Switch is still not claimable: its stage-3 +0.1035 (p = .094) dissolves under the trajectory to +0.0524, p = .781 (legacy: M1.2 step 8 finding 10)

+0.1035 at stage 3 stands against its own stage-0 null of +0.0511. Under the trajectory contrast
it dissolves outright: control +0.1270, loop +0.1794, difference **+0.0524, p = 0.781, CI
[−.102, +.248]**. Nearly all of Switch's apparent gain is movement *both* arms make over the
ramp. F21's arithmetic is unchanged besides: 3 metrics × 4 stages is 12 reported tests against
an n = 6 floor of 0.031, where Bonferroni needs 0.0042. **Descriptive only**, and the per-class
table in the paper carries no inferential claim at all.

### F46 — Where the bootstrap CI and the sign-flip p disagree, believe the p (legacy: M1.2 step 8 finding 11)

It happens three times above (map50 trajectory at stage 3, LPIPS at stage 3, Switch at stage 2),
always the same way: CI excludes zero, p does not clear 0.05. Both are facts about n = 6. The
sign-flip test is exact but **coarse** — 2⁶ = 64 assignments, so p can only take 0.031, 0.062,
0.094, … and there is no value between the floor and 0.062. The percentile bootstrap resamples 6
numbers with replacement and is known to be anti-conservative at that size, giving intervals
narrower than their nominal coverage. The CIs stay in the tables as descriptive spread; **no
claim in this project rests on a bootstrap interval excluding zero.** PLAN.md §12 permits
either, which is what made this ambiguity available — that permission needs narrowing in the
writeup.

### F47 — PLAN.md §16's causality criterion is satisfied for the pix2pix backbone, with F40/F41 and F43 as its two stated caveats (legacy: M1.2 step 8, consequence for §16)

The dose caveat that gated step 6's null (F25) is resolved: coupling was tested at 19.8% of the
objective and moved the endpoint (F35, F36). F07's "λ_det's gain is not causally established",
written of M1's unmatched single runs, is answered for this backbone by a paired, budget-matched
six-seed campaign.

### F48 — Without `--resume`, `run_loop` rewrites `metrics.json` from scratch: there is no `config_hash` guard on a run directory, so a reused run name destroys the earlier run (legacy: M1.2 step 8, unnumbered)

**Relaunch under new names, or the uncalibrated campaign is destroyed.** Without `--resume`,
`run_loop` starts from `results = []` (`src/t2o/engine/loop.py:149`) and rewrites `metrics.json`
stage by stage, so reusing `e3-<arm>-s<n>` would overwrite record 005's campaign seed for seed.
Those twelve run directories are the only copy of the 0.01 campaign — record 005's tables hold
its *results*, but the per-epoch loss curves behind record 006's shares and F32's noise floor
live nowhere else, and `runs/` is gitignored. Hence the `e3b-` prefix, which also leaves
`'runs/e3-*'` selecting the old campaign; `pair_runs` would refuse a mixed glob anyway (two runs
per `(arm, seed)` cell — the same guard that refused the probe against E3).

## Provenance caveats

- **Two dates in one record.** The probe, the matched-epoch read and the config bump are
  2026-08-19 (efead46 17:22, 5f2c9a5 17:29, 940ec89 17:33); the campaign's results are
  2026-08-23 (5ee297c 14:53, 627963e 15:35, 5d1e1b6 15:42). The header carries the campaign's.
- **This record carries the tag's wording, which three commits revised on 2026-08-23.** 627963e
  replaced the trajectory's point estimate with a measured test and changed "resolves the
  stage-0 draw" to "corroborates but does not confirm" (F41), and by its own commit message
  first rewrote the fidelity cost as a bound (F43). 5d1e1b6 struck F34 through and wrote F43's
  trajectory and training-loss paragraphs and F44, as they stand at the tag.
- **The tag contradicts itself inside F43.** Its first paragraph (5ee297c) still says "about a
  third of the control's fidelity gain is traded away. Step 8 predicted exactly this", quoting
  F34's sentence that 5d1e1b6 later withdrew; the paragraphs 5d1e1b6 added say training shows no
  trade. Lifted as written, per the plan's rule against correcting a frozen record; the later
  paragraphs are the source's settled reading, and the ~third is a fact about validation LPIPS
  only.
- **A mislabelled cross-reference in the source.** `TASKS.md:2317` (5f2c9a5) cites the sd ≈ 0.053
  as "Step 6 finding 1"; it is step 6 finding 2, F17. F32 cites F17.
- **F31's raw-detection-loss alarm has no F-ID of its own**: efead46 recorded it already withdrawn,
  so it never stood as a claim. F34 stood for four days (5f2c9a5 → 5d1e1b6) and has one.
- **The trajectory was built after the campaign ran**: `TrajectoryResult` landed in 2cc9a1e
  (2026-08-23 14:53:20), thirty seconds before 5ee297c. `--terms-only` landed in 627963e, also
  after the campaign; the exact control-arm invocation is not written in the source.
- **The probe's block carries an `epochs` column** that efead46 added in the same commit that
  first recorded the block; the probe was first read without it, which is how F31's trap arose.
- **F48 is a reading of the code, not an observation**: no overwrite happened, because the
  relaunch used new names to avoid one. The line was re-checked for this record at
  `src/t2o/engine/loop.py:149`, identical at the tag and today. The source's claim that
  `'runs/e3-*'` selects the old campaign alone overlooks `runs/e3-probe-g015`, which also
  matches it (record 005's caveats).
- **F40 and F47 cite code** (`coupling/schedule.py`, `translators/pix2pix.py:161`, `seeding.py`) by
  the references the source gave; they were not re-checked at this record's writing.
- **F47 grades against PLAN.md §16 as it stood**, the five legacy criteria. `docs/goal.md` now
  holds six (spec-migration Q12).
- The source records no wall clock: "~36h" is the relaunch plan's estimate, not a measurement.
  The SHA the campaign executed at is not recorded; the configs were bumped in 5f2c9a5, so it ran
  at or after that. The `$DATA`/`$OPTICAL` paths, and whether the two shells' matched, are not
  recorded.
- The p-values are exact sign-flip values at n = 6; the CIs are bootstrap, method and resample
  count not recorded. Per-seed values are not listed in the source.
- **Nothing later is spliced in.** `git blame` at the tag shows lines 2224–2613 written only by
  b98835d, efead46, 5f2c9a5, 940ec89, 5ee297c, 627963e and 5d1e1b6.

## Next

Settle F43's reward-hacking question on the evaluation side: detection rising while validation
fidelity falls is the signature PLAN.md §8 names, and `t2o faithfulness` (C2) on step 8's exports
exists to measure it (M1.2 step 9). Only if that comes back clean and the +0.0097 still needs
explaining, test the export path. Pre-register F42's variance collapse for the turbo arm, and
narrow PLAN.md §12's "either test" permission to the exact sign-flip test in the writeup (F46).
