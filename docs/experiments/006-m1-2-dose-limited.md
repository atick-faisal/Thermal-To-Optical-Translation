# 006 — M1.2 step 7: was λ_det ever large enough to matter? The detection term's share of E3's objective

**Date:** 2026-08-19 · **Task:** M1.2 · **Machine:** Windows server (CPU read of saved `metrics.json`, zero GPU cost) · **Wall clock:** not recorded
**git_sha:** not recorded — results committed in b98835d · **W&B:** none — not used for this measurement · **Log:** none saved — transcribed from TASKS.md 2120-2223 @ pre-spec-migration

## Command

As written in `TASKS.md`:

```bash
uv run python scripts/loss_share.py --runs 'runs/e3-loop-*'
```

Zero GPU cost: every loop run's `metrics.json` already holds `task_weight` per stage and
`loss_det`/`loss_total` per epoch (`StageResult` → `asdict`). `loss_det` as recorded is already
post-`grad_scale`, so `task_weight * loss_det / loss_total` *is* the detection term's share of
the objective, with no reconstruction of the weight chain needed.

It exists as a script rather than a `t2o aggregate --metric` path because
`analysis/aggregate.py::metric_value` walks dicts and `epochs` is a list. W&B is no help either:
the step bug fixed in `accfe56` dropped precisely the stages that have a detection term.
Control runs are skipped rather than pooled in (they record no `loss_det` at all), and runs that
disagree on `task_weight` or `grad_scale` are refused, so a mis-globbed `--runs` cannot quietly
average two experiments.

## Configuration

The six loop runs of record 005's campaign, read after the fact; nothing was trained.

- Input: `runs/e3-loop-*` — `experiments/e3_pix2pix_loop.yaml`, seeds 0–5, `task_weights`
  `[0, 1, 2, 3]`, `coupling.grad_scale: 1.0e-2`, loss weights `l2: 1.0` + `lpips: 5.0` +
  `gan: 1.0`.
- Pooled over epochs and over the six loop runs.
- Decision rule, written before the number was known:
  - **share ≲ 5%** → the null is *dose-limited*. Re-calibrate λ before any further six-seed
    campaign, in either backbone. The share also gives the candidate λ for free: `L_det`'s
    magnitude is roughly stable across training, so scale `grad_scale` by the ratio needed to
    bring the term to ~20–30% of the objective.
  - **share ≳ 20%** → the null is a *mechanism* result. E3's pix2pix arm is written up negative
    as it stands, and the turbo arm becomes the test of whether a stronger prior changes that.

## Results

As recorded in b98835d. Pooled over epochs and over the six loop runs:

| stage | w | λ_eff | `loss_l2` | `loss_lpips` | `loss_gan` | `loss_det` | `loss_total` | w·det | share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0.0415 | 1.8062 | 1.5747 | — | 3.4223 | 0 | 0.0% |
| 1 | 1 | 0.01 | 0.0361 | 1.5940 | 1.6886 | 0.0298 | 3.3485 | 0.0298 | **0.9%** |
| 2 | 2 | 0.02 | 0.0336 | 1.5146 | 1.6508 | 0.0275 | 3.2540 | 0.0550 | **1.7%** |
| 3 | 3 | 0.03 | 0.0323 | 1.4758 | 1.7490 | 0.0261 | 3.3354 | 0.0782 | **2.3%** |

Derived, not measured: the raw per-image detection loss `loss_det / grad_scale` = **2.98 /
2.75 / 2.61** across stages 1–3. With fidelity terms summing to F = 3.257 at stage 3 and a
target share S, the required weight is `w·g = F·S/((1−S)·L_raw)`:

| target share at stage 3 | required `w·g` | `grad_scale` at w=3 |
| --- | --- | --- |
| 20% | 0.312 | 0.104 |
| 25% | 0.416 | 0.139 |
| 30% | 0.535 | 0.178 |

## Findings

### F25 — E3's null is dose-limited, not a mechanism result: the detection term was 0.9 / 1.7 / 2.3% of the objective (legacy: M1.2 step 7, headline)

The decision rule fired on the ≲5% branch, and not marginally: at the *top* of the ramp the
detection term was **1/43rd of the objective**, roughly **19× smaller than the LPIPS term** it
had to compete against for the optimiser's attention. (It was not the objective's *smallest*
term — `loss_l2` at 1.0% is smaller still; see F27.) E3 did not test the coupling hypothesis at
a dose capable of refuting it. This settles F23's open question on the dose side.

### F26 — PLAN.md §8 predicted exactly this, and its instruction was skipped (legacy: M1.2 step 7, unnumbered)

§8's second guardrail bullet reads: *"The `[0,1,2,3]` ramp is calibrated to SeAFusion's
**segmentation** loss scale, not a detection loss — recalibrate empirically in Phase 0."* That
recalibration was never done, and `grad_scale: 1.0e-2` (AlignProp's `loss_coeff`, for a
different objective on a different task) went into E3 unexamined. The 72 GPU-hours were not
wasted — they produced a validated null-control, a confirmed noise floor and a fidelity bound —
but the headline they were spent on cannot be read as evidence about coupling.

### F27 — The objective is GAN + LPIPS, and `l2` is 1% of it (legacy: M1.2 step 7, unnumbered)

At stage 3: LPIPS 1.4758 (44%), GAN 1.7490 (52%), l2 0.0323 (**1.0%**), detection 0.0782
(2.3%). `LossConfig`'s comment calling l2 "Dominant, and the reason a translator with no other
term produces a blurred conditional mean" is true of an l2-only translator and false of this
configuration — corrected in the schema. Cross-check that the recorded losses mean what we
think: `loss_lpips / 5.0` = 0.295 against the measured val `fidelity.lpips` of 0.2989. The GAN
term is the only one that *rises* across stages (1.575 → 1.749) while LPIPS falls
(1.806 → 1.476): the discriminator is winning more as training goes on.

### F28 — Step 6's "no dose-response in λ" is superseded: it spanned shares too narrow to say anything about λ (legacy: M1.2 step 7, unnumbered)

F19's "no dose-response in λ" spanned shares of 0.9% → 2.3% — a range too narrow to be
informative about λ at all. It is not evidence against a dose-response; it is evidence that no
dose was applied.

### F29 — The candidate dose is `grad_scale: 0.15`, from the numbers rather than from taste, with two honest limits (legacy: M1.2 step 7, unnumbered)

The raw detection loss is stable and slowly declining across stages (2.98 / 2.75 / 2.61), which
is what makes a linear extrapolation usable. **`grad_scale: 0.15`** — 15× the E3 value — is the
candidate, predicting a ramp of ≈12% / 21% / 27% (λ_eff 0.15 / 0.30 / 0.45). Chosen at the
upper half of the 20–30% band deliberately: the extrapolation holds `L_raw` fixed, but a term
15× stronger will *drive the detection loss down*, which lowers its own share at equilibrium.
The achieved share will land below the prediction.

Two honest limits. (1) Loss share is a proxy for gradient influence, not a measurement of it —
`grad_scale` multiplies the detection gradient exactly linearly, so 15× more gradient is
certain, but whether that is 15× more *influence on the update* depends on the other terms'
gradient norms, which nothing here measured. (2) Raising `grad_scale` 15× removes most of §8's
anti-reward-hacking downscale. The replacement guard is already in place and quantified: LPIPS
is reported per stage, and F20 bounds λ_det's current fidelity effect at ±0.016.
`coupling.reward_target` stays `null` for the probe — with `L_raw` ≈ 2.6–3.0 a target near
1.5–2.0 would start biting, and that is the knob to reach for *if* fidelity degrades at the new
dose. Fidelity degrading would itself be a finding, not a failure: it would mean the loop can
trade fidelity for detection, which is precisely what §8 anticipated and what the guardrails
exist to bound.

## Provenance caveats

- **The decision rule was in git nine minutes before the number.** It, `scripts/loss_share.py`
  and its tests were committed in 6f47aba (2026-08-19 12:49, alongside step 6's results); the
  result landed in b98835d at 12:58.
- **One sentence was corrected the next day, and this record carries the corrected text.**
  b98835d said the detection term was "the smallest term in it by a factor of ~19". fe50cc3
  (2026-08-20) corrected it against the step's own table: `loss_l2` (0.0323, 1.0%) is smaller,
  and the ~19 is the ratio to LPIPS (1.4758 / 0.0782 = 18.9). No number and no conclusion
  changed. The tag holds the corrected wording, which F25 lifts.
- **The machine is inferred, not stated.** The source says only "zero GPU cost";
  `runs/e3-loop-*` exists only on the Windows server, so the script ran there.
- **F27's "corrected in the schema"** is b98835d's edit to `src/t2o/config/schema.py`. The same
  commit also rewrote `coupling.grad_scale`'s comment to name the too-low failure mode, which
  `TASKS.md` does not mention.
- **F29's numbers are derived, not measured**: `L_raw` from `loss_det / grad_scale`, the
  required-weight table and the ≈12 / 21 / 27% ramp from a linear extrapolation that the source
  itself says will overshoot. Record 007 measures what the probe actually achieved.
- **Nothing later is spliced in.** `git blame` at the tag shows lines 2120–2223 written only by
  6f47aba, b98835d and fe50cc3 (plus three blank lines from 42ebf92); no step 8 annotation sits
  inside this range.
- The per-seed loss values behind the pooled table are not listed in the source.

## Next

Recalibrate the dose before any further six-seed campaign, in either backbone (M1.2 step 8).
Probe first: one seed at `grad_scale: 0.15`, reading the achieved share, stability (does LPIPS
collapse, does the GAN diverge, does the detection term stay bounded) and fidelity — **never
mAP**. With a per-seed sd of ≈0.053 (F17), a single run cannot resolve the +0.02 being chased;
that is precisely the n = 1 mistake M1.2 exists to correct. The measurement is another paired
six-seed campaign, and it only earns its GPU time once the share says the dose is real.
