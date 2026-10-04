# Experiments

## Runs

| Run | Date | What ran | Task | Machine | Headline | Findings |
| --- | --- | --- | --- | --- | --- | --- |
| [001](001-e1-reference-bracket.md) | 2026-08-12 | E1 reference bracket: {thermal, visible}-trained detector × {raw thermal, real visible} | M0.10 | Windows server, 2× A100 | visible-trained detector: 0.1887 mAP50 on raw thermal vs 0.9213 on visible | F01, F02 |
| [002](002-m1-phase1-gate.md) | 2026-08-13 | M1 Phase 1 go/no-go: pix2pix λ_det=0 baseline + 4-stage λ_det loop, adapted and zero-shot arms | M1 | Windows server, 2× A100 | λ_det=0 pix2pix: 0.7751 zero-shot mAP50 vs the 0.1887 raw-thermal floor — gate PASS | F03, F04, F05 |
| [003](003-m1-1-fidelity.md) | 2026-08-13 | M1.1 fidelity (LPIPS/FID/KID/SSIM/PSNR) on M1's exported val splits, baseline + loop stages 0–3 | M1.1 | Windows server, 2× A100 | loop stages 0→3: FID 104.82 → 90.85 while zero-shot mAP50 0.7622 → 0.8696 — no reward hacking | F06, F07, F08, F09 |
| [004](004-m1-2-yolo11s-judge.md) | 2026-08-13 | M1.2 step 1: independent `yolo11s` judge (visible train only) re-scores M1's raw-thermal, baseline, loop stages 0–3 and visible arms | M1.2 | Windows server, 2× A100 | same-config λ=0 arms differ by 0.0591 mAP50 under an honest judge (0.0129 under the in-loop one) | F10, F11, F12, F13, F14 |
| [005](005-m1-2-e3-pix2pix-null.md) | 2026-08-19 | M1.2 steps 5–6: E3 pix2pix, all-zero control vs λ_det [0,1,2,3] loop at `grad_scale: 1.0e-2`, six paired seeds, judged by `yolo11s` | M1.2 | Windows server, 2× A100 | stage-3 paired zero-shot mAP50 +0.0070 (p = .656) against a −0.0063 stage-0 null — negative | F15, F16, F17, F18, F19, F20, F21, F22, F23, F24 |
| [006](006-m1-2-dose-limited.md) | 2026-08-19 | M1.2 step 7: `scripts/loss_share.py` over E3's six loop runs' saved `metrics.json` — the detection term's share of the objective per stage | M1.2 | Windows server, CPU read (zero GPU cost) | detection term 0.9 / 1.7 / 2.3% of the objective at `grad_scale: 1.0e-2` — E3's null is dose-limited | F25, F26, F27, F28, F29 |
| [007](007-m1-2-e3-pix2pix-positive.md) | 2026-08-23 | M1.2 step 8: one-seed dose probe at `grad_scale: 0.15`, then E3 pix2pix re-run at that dose — all-zero control vs λ_det [0,1,2,3] loop, six paired seeds, judged by `yolo11s` | M1.2 | Windows server, 2× A100 | stage-3 paired zero-shot mAP50 +0.0512 (p = .031, the n = 6 sign-flip floor) against a −0.0397 stage-0 null — positive | F30, F31, F32, F33, F34, F35, F36, F37, F38, F39, F40, F41, F42, F43, F44, F45, F46, F47, F48 |
| [008](008-m1-2-c2-faithfulness.md) | 2026-08-23 | M1.2 step 9: C2 faithfulness (`t2o faithfulness --write-back`) on record 007's twelve stage-3 exports, scored by the reference `yolo11s`, six paired seeds | M1.2 | Windows server, 2× A100 | stage-3 false-object rate −0.0289 loop vs control (p = .156) while mAP50 rose +0.0512 — no reward hacking | F49, F50, F51, F52, F53, F54 |
| [009](009-m2a-turbo-probes.md) | 2026-08-25 | M2a step 4: turbo VRAM probe (one stage-3 epoch, batch 2, full frame) and two 25-epoch `--no-detector` `grad_scale` probes at 0.15 and 0.75, seed 0 | M2a | Windows server, 2× A100 | `grad_scale: 0.15` puts turbo's detection term at 16.6 / 22.9 / 24.5% of the objective — in band at pix2pix's value; batch 2 peaks at 34.14 GB | F55, F56, F57, F58, F59, F60, F61 |
| [010](010-m2a-e3-turbo-replicates.md) | 2026-09-23 | M2a step 5: E3 pix2pix-turbo, all-zero control vs λ_det [0,1,2,3] loop at `grad_scale: 0.15`, cut from six paired seeds to three (0, 1, 3) after four OOM episodes, judged by `yolo11s`, plus C2 on the stage-3 exports | M2a | Windows server, 2× A100 | stage-3 paired zero-shot mAP50 +0.0306 (p = .250, the n = 3 floor, 3/3 seeds) against a level −0.0023 stage-0 null — E3 replicates on turbo | F62, F63, F64, F65, F66, F67, F68, F69, F70, F71, F72, F73, F74, F75, F76, F77, F78, F79, F80 |
| [011](011-e8-annotation-sweep.md) | 2026-09-24 | E8: direct-thermal (A) vs translated-then-fine-tuned (B) `yolo11n` at N ∈ {10 … 600}, three seeds, against the reference judge's zero-annotation anchors A0 / C / D; C and D topped up to six translator seeds | M3 E8 | Windows server, 2× A100 | arm A passes annotation-free translation (C, 0.7975) at N ≈ 150 annotated thermal images, interval [114, 185]; direct thermal wins at N = 600, 0.9300 | F81, F82, F83, F84, F85, F86, F87, F88 |
| [012](012-e9-pix2pix-wall-clock.md) | 2026-09-26 | E9 pricing of the FLIR cell's three blockers, and step 0: pix2pix per-stage wall clock from `translator_last.pt` mtimes over 24 completed runs, 72 intervals | M3 E9 | Windows server, filesystem read (zero GPU cost) | a 4-stage pix2pix run is 10.18 h control / 10.75 h loop; the twelve-run cell is ~126 GPU-h, not ~72 | F89, F90, F91, F92, F93, F94 |

## Findings

| ID | Claim | Run | Status | Acts on / acted on by |
| --- | --- | --- | --- | --- |
| F01 | In-domain detection > 0.9 mAP50 on both modalities; the domain gap, not the sensor, is the problem | 001 | open | |
| F02 | A visible-trained detector collapses to 0.1887 mAP50 on raw thermal (0.9213 on visible); Switch and Fuse near zero, Pole 0.5551 | 001 | open | |
| F03 | The Phase 1 gate passes on the uncontaminated λ_det=0 arm: 0.7751 vs 0.1887 floor, +0.586 mAP50, all four classes | 002 | confirmed | F10 |
| F04 | λ_det raises zero-shot mAP50 monotonically (+0.107 at λ=3 over the epoch-matched control, ~8x the 0.0129 noise floor), but the judge supplied the training gradient — not reportable | 002 | superseded | F06, F07, F12, F13 |
| F05 | The adapted (fine-tuned) arm is saturated (0.8984–0.9199, inside a 1.5-point same-config gap) and inverts the zero-shot ranking — uninformative | 002 | open | |
| F06 | No reward hacking: across loop stages mAP50 0.7622 → 0.8696 while LPIPS, FID and KID improve and SSIM holds | 003 | confirmed | acts on F04; F20 |
| F07 | λ_det's gain is not causally established: stages are not budget-matched, and same-config runs differ by 17.6 FID | 003 | confirmed | acts on F04; F13, F47 |
| F08 | Absolute fidelity is poor (PSNR ~15.5, FID ~90) yet detection reaches 0.87 mAP50, 94% of the visible ceiling | 003 | open | |
| F09 | The custom set declares `nc: 5`, but `Connector` (index 0) has zero instances; 4 classes are reported, no mAP is diluted | 003 | open | |
| F10 | An independent `yolo11s` judge confirms M1's gate: baseline 0.7851 vs 0.1552 floor, +0.630 mAP50, all four classes | 004 | open | acts on F03; F86 |
| F11 | No evidence the old judge was trained on val: the train-only judge scores real visible 0.9364 vs the old 0.9213; M1's ceiling stands | 004 | open | |
| F12 | Self-grading was real but small (~0.023 on λ>0 arms); λ_det's effect is not monotone (stage 2 0.8106 < stage 1 0.8244); the gain lives in Switch | 004 | superseded | acts on F04; F19, F21 |
| F13 | The noise floor is 0.0591 mAP50, not 0.0129: stage 3 vs baseline +0.062 is ~1 noise draw, so at n = 1 λ_det's gain cannot be separated from run variance | 004 | confirmed | acts on F04, F07; F17 |
| F14 | ultralytics resolves a relative `project` under a machine-global `runs_dir`, not the cwd, so outputs land in another repository; M1's numbers unaffected | 004 | open | |
| F15 | E3's pix2pix arm is negative on its pre-registered endpoint at `grad_scale: 1.0e-2`: stage 3 +0.0070 (p = .656) vs a −0.0063 stage-0 null | 005 | open | F35 |
| F16 | The stage-0 null control behaved as a null (−0.0063, p = .875, CI straddling zero), so the campaign is a measurement, not a broken run | 005 | open | |
| F17 | Step 1's n = 1 noise floor was right: stage 0's CI half-width implies a per-seed sd ≈ 0.053 | 005 | open | acts on F13 |
| F18 | Six paired seeds resolve ±0.026: the claim is "no effect larger than ~+3 mAP50 points", not "no effect" | 005 | open | |
| F19 | No dose-response in λ at `grad_scale: 1.0e-2`: paired difference 0 → +0.024 → +0.024 → +0.007, peaking at the smallest λ | 005 | superseded | acts on F12; F28, F36 |
| F20 | λ_det is fidelity-neutral at n = 6: stage-3 LPIPS −0.0019, CI [−0.016, +0.011]; reward-hacking question closed | 005 | superseded | acts on F06; F43 |
| F21 | The Switch gain does not survive (stage-0 null +0.043 > stage-3 effect +0.032); no per-class claim is attainable at n = 6 | 005 | confirmed | acts on F12; F42, F45 |
| F22 | E8 is not an escape route: `annotation_fraction` gates only the loop arm's supervision, so lowering it makes E3's arms more alike | 005 | open | |
| F23 | λ_det was never 1/2/3: `grad_scale` is applied before `task_weight`, so the effective ramp was 0.01/0.02/0.03 — dose and mechanism nulls are inseparable | 005 | confirmed | F25 |
| F24 | W&B's explicit `step=epoch` dropped stages 1–3's generator loss curves; `metrics.json` and every E3 number are unaffected (fixed in accfe56) | 005 | open | |
| F25 | At `grad_scale: 1.0e-2` the detection term was 0.9 / 1.7 / 2.3% of the objective (~19× below LPIPS at stage 3): E3's null is dose-limited, not a mechanism result | 006 | confirmed | acts on F23; F35, F47 |
| F26 | PLAN.md §8 said to recalibrate the SeAFusion-scaled ramp for a detection loss; it never happened, and AlignProp's `loss_coeff` went into E3 unexamined | 006 | open | |
| F27 | The objective is GAN (52%) + LPIPS (44%); l2 is 1.0%, so `LossConfig`'s "dominant" comment was false at these weights; only GAN rises across stages | 006 | open | |
| F28 | Step 6's "no dose-response in λ" spanned shares of 0.9–2.3%: evidence that no dose was applied, not evidence against a dose-response | 006 | confirmed | acts on F19; F36 |
| F29 | Candidate `grad_scale: 0.15`, extrapolated from a stable raw `L_det` (2.98 / 2.75 / 2.61) to a predicted ≈12 / 21 / 27% ramp; share is only a proxy for gradient influence | 006 | confirmed | F30, F38 |
| F30 | `grad_scale: 0.15` lands the dose: the 25-epoch probe's share is 10.9 / 16.6 / 22.5% against a predicted ≈12 / 21 / 27; stable, nothing collapsed | 007 | confirmed | acts on F29; F38 |
| F31 | The epoch-length trap: a 25-epoch probe reads above a 100-epoch run in every term at once, mimicking a changed condition; two false alarms, `--first-epochs` added | 007 | confirmed | F39 |
| F32 | Two GPU runs at one seed do not reproduce, by design: the single-run loss-space floor is +3.4 / +3.6 / +11.8 / +7.1% (l2 / lpips / gan / total) | 007 | superseded | F44 |
| F33 | Under 15× the weight, raw `L_det` does not detectably move at 25 epochs: −5.0 / +3.9 / +1.4%, all inside the floor | 007 | superseded | F37 |
| F34 | The dose trades fidelity improvement: within-run `loss_lpips` falls 25.2% at `grad_scale` 0.01 vs 9.5% at 0.15 (withdrawn in the source) | 007 | refuted | F43 |
| F35 | E3's pix2pix arm is positive at `grad_scale: 0.15`: stage-3 paired zero-shot mAP50 +0.0512, p = .031 (the n = 6 floor), CI [+.025, +.081] | 007 | open | acts on F15, F25; F72, F87 |
| F36 | Dose-response appeared: paired difference 0 → +0.0280 → +0.0357 → +0.0512, monotone in λ | 007 | open | acts on F19, F28; F77 |
| F37 | Raw detection loss falls 2.56 → 2.12 → 1.84 across stages 1–3, stage 3 ~30% below the uncalibrated campaign's 2.61 | 007 | open | acts on F33 |
| F38 | Achieved share at 100 epochs 10.0 / 16.1 / 19.8%, just under the 20–30% band; not re-tuned after seeing mAP | 007 | open | acts on F30 |
| F39 | `loss_gan` did not diverge: +10.6% across stages 0→3 vs +11.1% uncalibrated; the probe's +35% was the 25-epoch artifact | 007 | open | acts on F31 |
| F40 | The stage-0 null drew wide: −0.0397 (p = .469), loop-arm sd 0.0907 vs 0.0372; stage 3 is only 1.3× it — not clearly larger on that reading alone | 007 | open | F41 |
| F41 | Within-arm trajectory +0.0909, p = .094, CI [+.019, +.181], monotone in dose: corroborates but does not confirm; post-hoc sensitivity analysis | 007 | open | acts on F40 |
| F42 | Loop-arm variance collapses as λ rises: sd 0.0907 → 0.0140, 2.4× tighter than control by stage 3 — a hypothesis to pre-register for turbo | 007 | open | acts on F21; F80 |
| F43 | Val LPIPS +0.0097 at stage 3 (p = .125) but training fidelity untouched (`loss_lpips` −18.42% vs −18.50%): a generalisation gap, bounded at ~0.02 LPIPS | 007 | confirmed | acts on F20, F34; F53 |
| F44 | Pooled 6-vs-6 loss-space floor 0.95 / 0.73 / 2.02 / 1.34% governs campaign-scale comparisons; the single-run floor still governs probes | 007 | open | acts on F32 |
| F45 | Switch is not claimable: stage-3 +0.1035 (p = .094) dissolves under the trajectory to +0.0524, p = .781 | 007 | open | acts on F21 |
| F46 | Where bootstrap CI and sign-flip p disagree (three cells, always CI excludes zero), believe the p; no claim rests on a bootstrap CI | 007 | open | |
| F47 | PLAN.md §16's causality criterion is satisfied for pix2pix, with F40/F41 and F43 as its caveats; F25's dose caveat resolved | 007 | open | acts on F07, F25 |
| F48 | Without `--resume`, `run_loop` rewrites `metrics.json` from `results = []`: no `config_hash` guard on a run dir, so a reused name destroys the earlier run | 007 | open | |
| F49 | C2's pre-registered discriminator is met: stage-3 false-object rate fell −0.0289 (−15.1%), loop 0.1629 vs control 0.1918, while mAP50 rose +0.0512 | 008 | open | F76 |
| F50 | False-object rate is the only rate independent of the gain and is not significant (p = .156, CI [−.0559, +.0039]); missed-object −0.0370 and consistency +0.0291 (both p = .031) corroborate the endpoint, not hallucination | 008 | open | F76 |
| F51 | No per-metric claim survives multiplicity: Bonferroni α = 0.0167 is below the n = 6 sign-flip floor of 0.031; false-object rate was designated primary before the numbers | 008 | open | |
| F52 | The loop arm is less variable on all three rates (sd ratio loop/control 0.69 / 0.52 / 0.41); untested, an observation to pre-register for turbo | 008 | open | F80 |
| F53 | F43's +0.0097 LPIPS gap is bounded from both sides — no training-loss cost, better object-level faithfulness — so `reward_target` stays null and the turbo hold is released | 008 | open | acts on F43; F76 |
| F54 | `t2o faithfulness` crashed on its first server image (CUDA predictions vs CPU ground truth); every test built CPU tensors, so the case was unreachable; fixed in cf72c9c | 008 | open | |
| F55 | Batch 2 fits turbo at full frame: 34.14 GB peak at stage 3 with the `FrozenDetector` resident, ~5.8 GB headroom; the stage-boundary detector fine-tune is unmeasured | 009 | superseded | F66 |
| F56 | `grad_scale: 0.15` lands turbo in band: 16.6 / 22.9 / 24.5% at 25 epochs, projected ~21.6% at 100; band ≈ [0.11, 0.21]; stage-0 totals agree to 0.13% | 009 | refuted | F77 |
| F57 | The dose bites in loss space within 25 epochs: raw `loss_det` −5.2 / −9.0 / −13.1% at 0.75 vs 0.15, monotone in stage | 009 | open | |
| F58 | The dose costs fidelity: weighted `loss_lpips` +6.9 / +11.0 / +3.7% at 5× the dose, against a 0.3% stage-0 null — no case for raising `grad_scale` | 009 | open | |
| F59 | Turbo is lower than pix2pix on every objective term at stage 3 (total −34%; raw LPIPS 0.175 at 25 ep vs 0.298 at 100) — training loss, one seed, descriptive only | 009 | open | |
| F60 | Turbo's objective is GAN-heavy: detection 24.5 / GAN 50.9 / LPIPS 24.1 / l2 0.5% vs pix2pix's 19.8 / 43.7 / 35.7 / 0.8% — composition differs by backbone at identical knobs | 009 | open | |
| F61 | `loss_gan` +119% across stages reads as catch-up, not divergence: ends below pix2pix (1.8513 vs 2.3294) from half the stage-0 base; the 0.75 arm climbs alike | 009 | open | F77 |
| F62 | Turbo costs 24.1 h a stage, ~96 h a 4-stage run; the ~72 GPU-h budget was pix2pix's and 16× low — time one stage of a new backbone before its campaign | 010 | confirmed | F92 |
| F63 | `expandable_segments:True` is silently unsupported on Windows (torch 2.12.1): accepted, warned, `is_expandable` [False]; its crash-free 24 h was luck | 010 | open | |
| F64 | A mid-stage-3 resume died in FID with `CUSOLVER_STATUS_INTERNAL_ERROR`, read as cache pressure not numerics; `empty_cache()` before fidelity (b7b1627) fixed it | 010 | open | |
| F65 | A crashed `uv run` does not stop `foreach`: the first attempt started all twelve runs and finished none, 18 of 48 stages; a relaunch must `break` on <4 stages | 010 | open | |
| F66 | The turbo config never fit: ~32.1 GiB allocated + ~6.8 GiB reserved-but-unallocated ≈ 98% of the card, steady across four crashes; step 4 measured allocated, not reserved | 010 | open | acts on F55 |
| F67 | Allocator flags cannot fix it on Windows: the ~6.8 GB sits in segments holding a live tensor, which only `expandable_segments` addresses; per-epoch release could not reach a mid-epoch crash | 010 | open | |
| F68 | Every resume silently resets both turbo `AdamW` optimizers; endpoint exposure 1–4 epochs of 400, falling on both arms — a stated limitation | 010 | open | F79 |
| F69 | Turbo's control starts far above pix2pix's: stage-0 mAP50 0.8889 vs 0.7579 ± .0372 (~3.5 sd), LPIPS ~5 sd better at stage 3 — n = 1, most-disturbed run | 010 | open | |
| F70 | Prediction from one run: turbo's control declines monotonically under warm-started training (0.8889 → 0.8293); the other two controls will too | 010 | refuted | F73 |
| F71 | Globbing `runs/e3t-*` pulls in six stumps, collapses `common_stages` to [0] and yields a plausible, wrong n = 6 stage-0 table; `--stage` outside it is now fatal (f6ee35e) | 010 | open | |
| F72 | E3 replicates on turbo: stage-3 paired zero-shot mAP50 +0.0306, p = .250 (the n = 3 floor), CI [+.0121, +.0459], 9/9 coupled-stage wins, level stage-0 null −0.0023 | 010 | open | acts on F35 |
| F73 | Observation B falsified per run: only `control-s3` is monotone, its −.0596 five times the others; all three controls do finish below their start | 010 | open | acts on F70 |
| F74 | The stage-3 trajectory statistic (+0.0329, p = .75) is the s3 pair's −0.0601 baseline draw; with a level stage-0 the raw contrast governs (3/3) | 010 | open | |
| F75 | Turbo `fidelity.lpips` is null: stage-0 null −0.0117 (3/3) is not level, so the trajectory governs, −0.0080, p = .500; not a device confound | 010 | open | |
| F76 | C2 clean on turbo: false-object −0.0193 (p = .500), missed-object −0.0386 and consistency +0.0319 at the p = .250 floor — recall, not precision; `reward_target` stays null | 010 | open | acts on F49, F50, F53 |
| F77 | No dose-response on turbo (0 → +.0274 → +.0235 → +.0306): share plateaus 8.8 / 12.3 / 12.9% as backbone-intrinsic `loss_gan` inflates 2.47 → 4.51 | 010 | open | acts on F36, F56, F61 |
| F78 | Turbo loss-space figures are weak at n = 3: resumed stages pool stumps, only stage 2 is clean in both arms, and control `loss_gan` spans 1.55–4.42 across seeds | 010 | open | |
| F79 | The resume audit is balanced: ~2–3 degraded endpoint epochs on two control runs vs ~3 on one loop run, opposite pairs — M2a's caveat closed; `AdamW` now checkpointed (e210b13) | 010 | open | acts on F68 |
| F80 | Reading (a) partially replicates: loop/control sd 0.71 / 0.40 / 0.54 / 0.25× on mAP50, LPIPS, missed, consistency, but 1.07× on false-object — 2-dof, a direction | 010 | open | acts on F42, F52 |
| F81 | Annotation-free translation (C) is worth ≈150 annotated thermal images, interval [114, 185]; spending them inside the loop (D) raises it to ≈214 | 011 | open | |
| F82 | Direct thermal wins at full annotation (0.9300 vs C 0.7975, D 0.8487): criterion 1 is measured, not met — +0.642 at N = 0, −0.133 at N = 600 | 011 | open | |
| F83 | Translate-then-fine-tune (B) loses to thermal at every N ≥ 50, 3/3 seeds, by +0.032 to +0.059; translation's value is entirely annotation-free transfer | 011 | open | |
| F84 | N = 10 is degenerate: both trained arms fall below A0's 0.155 floor at precision ≈0.01 — never read as "what 10 images buy" | 011 | open | |
| F85 | A's sd collapses ±0.082 → ±0.010 from N = 50 to 600; n = 3 floors p at 0.25, so the curve reports sign consistency (3/3 at N ≥ 50), never significance | 011 | open | |
| F86 | A0's 0.1552 reproduces the clean-judge thermal floor to four decimals, and C / D land on its λ=0 / stage-3 rows — nothing drifted between campaigns | 011 | open | acts on F10 |
| F87 | D − C at n = 6 reproduces E3's +0.0512 (6/6, p = .031) to every digit — the same computation, a reproduction never counted as second evidence | 011 | open | acts on F35 |
| F88 | The crossover's width is C's translator-seed spread, not A's resolution: n = 3 → 6 nearly halved C's sem and narrowed [111, 214] to [114, 185]; infilling A not worth it | 011 | open | |
| F89 | A pix2pix stage is not a constant: interval medians 2.40 / 2.83 / 3.37 h, ~+20% a stage, ±8% within a boundary; stage 0 ≈ 1.79 h | 012 | open | |
| F90 | A 4-stage pix2pix run costs 10.18 h control / 10.75 h loop (coupling +0.57 h, +6.8%); the twelve-run cell is ~126 GPU-h, and the parts predict 62.8 h of a 63.0 h observed span | 012 | open | |
| F91 | "~6 h per 4-stage run" was low by 1.7× and "~72 GPU-h" by 1.75×; a full-corpus FLIR cell is ~870 GPU-h, not ~496 — the 600-pair matched cell holds | 012 | open | |
| F92 | pix2pix:turbo is 9.0× per complete run (10.5 h vs ~94 h), not 16×; the same method re-derives turbo at 23.46 h a stage against F62's 24.1 h | 012 | open | acts on F62 |
| F93 | FLIR has no thermal labels on disk — adapters write labels under `visible/` only — so the raw-thermal floor needs mirroring, and mirrored labels inherit the 5.90 px residual, biasing the floor down | 012 | open | |
| F94 | FLIR's dog class has 13 val instances against 4,124 cars; a 0.4 AP swing moves the 4-class mAP50 by 0.1, a decision band's width — 3-class is primary | 012 | open | |
