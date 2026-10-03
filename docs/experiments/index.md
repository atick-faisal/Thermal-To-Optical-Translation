# Experiments

## Runs

| Run | Date | What ran | Task | Machine | Headline | Findings |
| --- | --- | --- | --- | --- | --- | --- |
| [001](001-e1-reference-bracket.md) | 2026-08-12 | E1 reference bracket: {thermal, visible}-trained detector × {raw thermal, real visible} | M0.10 | Windows server, 2× A100 | visible-trained detector: 0.1887 mAP50 on raw thermal vs 0.9213 on visible | F01, F02 |
| [002](002-m1-phase1-gate.md) | 2026-08-13 | M1 Phase 1 go/no-go: pix2pix λ_det=0 baseline + 4-stage λ_det loop, adapted and zero-shot arms | M1 | Windows server, 2× A100 | λ_det=0 pix2pix: 0.7751 zero-shot mAP50 vs the 0.1887 raw-thermal floor — gate PASS | F03, F04, F05 |
| [003](003-m1-1-fidelity.md) | 2026-08-13 | M1.1 fidelity (LPIPS/FID/KID/SSIM/PSNR) on M1's exported val splits, baseline + loop stages 0–3 | M1.1 | Windows server, 2× A100 | loop stages 0→3: FID 104.82 → 90.85 while zero-shot mAP50 0.7622 → 0.8696 — no reward hacking | F06, F07, F08, F09 |
| [004](004-m1-2-yolo11s-judge.md) | 2026-08-13 | M1.2 step 1: independent `yolo11s` judge (visible train only) re-scores M1's raw-thermal, baseline, loop stages 0–3 and visible arms | M1.2 | Windows server, 2× A100 | same-config λ=0 arms differ by 0.0591 mAP50 under an honest judge (0.0129 under the in-loop one) | F10, F11, F12, F13, F14 |
| [005](005-m1-2-e3-pix2pix-null.md) | 2026-08-19 | M1.2 steps 5–6: E3 pix2pix, all-zero control vs λ_det [0,1,2,3] loop at `grad_scale: 1.0e-2`, six paired seeds, judged by `yolo11s` | M1.2 | Windows server, 2× A100 | stage-3 paired zero-shot mAP50 +0.0070 (p = .656) against a −0.0063 stage-0 null — negative | F15, F16, F17, F18, F19, F20, F21, F22, F23, F24 |

## Findings

| ID | Claim | Run | Status | Acts on / acted on by |
| --- | --- | --- | --- | --- |
| F01 | In-domain detection > 0.9 mAP50 on both modalities; the domain gap, not the sensor, is the problem | 001 | open | |
| F02 | A visible-trained detector collapses to 0.1887 mAP50 on raw thermal (0.9213 on visible); Switch and Fuse near zero, Pole 0.5551 | 001 | open | |
| F03 | The Phase 1 gate passes on the uncontaminated λ_det=0 arm: 0.7751 vs 0.1887 floor, +0.586 mAP50, all four classes | 002 | confirmed | F10 |
| F04 | λ_det raises zero-shot mAP50 monotonically (+0.107 at λ=3 over the epoch-matched control, ~8x the 0.0129 noise floor), but the judge supplied the training gradient — not reportable | 002 | superseded | F06, F07, F12, F13 |
| F05 | The adapted (fine-tuned) arm is saturated (0.8984–0.9199, inside a 1.5-point same-config gap) and inverts the zero-shot ranking — uninformative | 002 | open | |
| F06 | No reward hacking: across loop stages mAP50 0.7622 → 0.8696 while LPIPS, FID and KID improve and SSIM holds | 003 | confirmed | acts on F04; F20 |
| F07 | λ_det's gain is not causally established: stages are not budget-matched, and same-config runs differ by 17.6 FID | 003 | confirmed | acts on F04; F13 |
| F08 | Absolute fidelity is poor (PSNR ~15.5, FID ~90) yet detection reaches 0.87 mAP50, 94% of the visible ceiling | 003 | open | |
| F09 | The custom set declares `nc: 5`, but `Connector` (index 0) has zero instances; 4 classes are reported, no mAP is diluted | 003 | open | |
| F10 | An independent `yolo11s` judge confirms M1's gate: baseline 0.7851 vs 0.1552 floor, +0.630 mAP50, all four classes | 004 | open | acts on F03 |
| F11 | No evidence the old judge was trained on val: the train-only judge scores real visible 0.9364 vs the old 0.9213; M1's ceiling stands | 004 | open | |
| F12 | Self-grading was real but small (~0.023 on λ>0 arms); λ_det's effect is not monotone (stage 2 0.8106 < stage 1 0.8244); the gain lives in Switch | 004 | superseded | acts on F04; F19, F21 |
| F13 | The noise floor is 0.0591 mAP50, not 0.0129: stage 3 vs baseline +0.062 is ~1 noise draw, so at n = 1 λ_det's gain cannot be separated from run variance | 004 | confirmed | acts on F04, F07; F17 |
| F14 | ultralytics resolves a relative `project` under a machine-global `runs_dir`, not the cwd, so outputs land in another repository; M1's numbers unaffected | 004 | open | |
| F15 | E3's pix2pix arm is negative on its pre-registered endpoint at `grad_scale: 1.0e-2`: stage 3 +0.0070 (p = .656) vs a −0.0063 stage-0 null | 005 | open | |
| F16 | The stage-0 null control behaved as a null (−0.0063, p = .875, CI straddling zero), so the campaign is a measurement, not a broken run | 005 | open | |
| F17 | Step 1's n = 1 noise floor was right: stage 0's CI half-width implies a per-seed sd ≈ 0.053 | 005 | open | acts on F13 |
| F18 | Six paired seeds resolve ±0.026: the claim is "no effect larger than ~+3 mAP50 points", not "no effect" | 005 | open | |
| F19 | No dose-response in λ at `grad_scale: 1.0e-2`: paired difference 0 → +0.024 → +0.024 → +0.007, peaking at the smallest λ | 005 | open | acts on F12 |
| F20 | λ_det is fidelity-neutral at n = 6: stage-3 LPIPS −0.0019, CI [−0.016, +0.011]; reward-hacking question closed | 005 | open | acts on F06 |
| F21 | The Switch gain does not survive (stage-0 null +0.043 > stage-3 effect +0.032); no per-class claim is attainable at n = 6 | 005 | open | acts on F12 |
| F22 | E8 is not an escape route: `annotation_fraction` gates only the loop arm's supervision, so lowering it makes E3's arms more alike | 005 | open | |
| F23 | λ_det was never 1/2/3: `grad_scale` is applied before `task_weight`, so the effective ramp was 0.01/0.02/0.03 — dose and mechanism nulls are inseparable | 005 | open | |
| F24 | W&B's explicit `step=epoch` dropped stages 1–3's generator loss curves; `metrics.json` and every E3 number are unaffected (fixed in accfe56) | 005 | open | |
