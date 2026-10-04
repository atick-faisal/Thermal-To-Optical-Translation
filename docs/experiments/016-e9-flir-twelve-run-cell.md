# 016 — E9 step 5: the twelve-run FLIR cell — the loop beats its control by +0.351, and lands on the raw-thermal floor

**Date:** 2026-10-01 · **Task:** M3 E9 · **Machine:** Windows server, 2× A100 — two-up, one run per card, each pair on one card (s0–s2 `cuda:0`, s3–s5 `cuda:1`) · **Wall clock:** ~116.8 GPU-h for the cell; a run 9.51–9.61 h control, 9.94–10.44 h loop (`control-s0` 8.46 h, counted from its resume)
**git_sha:** not recorded — the launch SHA was never written down; the report ran at 7a3ca94, the result was committed in 123410b, the wall-clock fixes in 8705364 and e70760c · **W&B:** not recorded — the resolved config prints `runtime.group: e3-pix2pix-flir600`, never checked against W&B · **Log:** `runs/e3f-report-2026-10-01.txt` (blocks 1–4 and 6–8; its block 5 predates both wall-clock fixes); the re-run block 5 is none saved — transcribed from TASKS.md 5319-5328 @ pre-spec-migration

## Command

**The campaign's launch command: not recorded.** Nothing in the source records one, and
`e3f-control-s0` in step 5's concurrency budget (record 015) was the only trace of the run names.
The twelve `config.yaml` snapshots are the only record of what was asked for; the report's block 3
prints every key that differs across them and the fully resolved config of one run (under
Configuration below). The source: "Treat the printed flag set as the reproducible launch, not the
reconstruction below."

What the pre-registration fixed, as the source states it: `--max-train-images 600` (the matched
corpus), `--val-loss-images 153` (F125), `--workers 8` two-up with one run per card, each *pair* on
one card by seed, `--in-loop-weights` and `--eval-init-weights` both
`runs/inloop-flir-yolo11n/weights/best.pt`, `--reference-weights
runs/reference-flir-yolo11s/weights/best.pt`, `--resume` on every run (F135), and the ramp left to
`experiments/e3_pix2pix_loop.yaml` rather than overridden.

The readout — C2 over the twelve stage-3 exports, then the report — verbatim from the source:

```powershell
$FLIR  = "dataset/processed/flir/data.yaml"
$JUDGE = "runs/reference-flir-yolo11s/weights/best.pt"

# 1. Pre-flight. ultralytics honours `path:` literally while t2o's loader falls through a
#    stale one, and everything below reaches ultralytics (step 1's sharper statement).
Get-Content $FLIR -TotalCount 2

# 2. C2 over the twelve stage-3 exports, scored with the REFERENCE judge, never the in-loop
#    one. --write-back lands the rates in metrics.json so the paired test is the same one the
#    headline rests on, rather than twelve numbers to eyeball.
#    Needs the --batch fix below; the command itself is unchanged, 16 is the default.
foreach ($arm in 'control','loop') { foreach ($s in 0,1,2,3,4,5) {
  uv run t2o faithfulness --translated "runs/e3f-$arm-s$s/stage3/translated" `
    --data $FLIR --weights $JUDGE --write-back --device cuda:0
} }

# 3. The report. ~500 lines; redirect, open, paste back whole.
uv run python scripts/campaign_report.py --runs 'runs/e3f-*' --stage 3 `
    --primary-classes bicycle car person --data $FLIR > runs/e3f-report-2026-10-01.txt
```

Also wanted, from the W&B UI and never read: the **Runtime** column for the twelve runs and the
campaign's `--group` (open, in `## Next`).

## Configuration

**Twelve runs**: `experiments/e3_pix2pix_{control,loop}.yaml` × seeds 0–5, control at
`task_weights` 0/0/0/0 and loop at 0/1/2/3, `grad_scale: 0.15`, 100 epochs a stage, 600 FLIR train
pairs, the val loss over 153 of FLIR's 1,013 val pairs; the exported val split, zero-shot mAP, FID
and the detector fine-tune read all 1,013 (F123).

**The endpoint, fixed before the numbers were read.** 3-class mAP50 (bicycle / car / person) is the
primary, dog stated separately, 4-class `zero_shot.map50` reported beside it. This follows the dog
rule pre-registered for this cell (F94) — 13 val instances carrying 25% of a 4-class mean, where one
detection moves it by ~0.1, the width of the kill-test's whole decision band — and it puts the gain
on the scale of the **+0.2066** headroom the cell is read against (F115), itself a 3-class number.
`t2o aggregate` could not express it: `metric_value` resolves one dotted leaf. `--primary-classes`
now derives `zero_shot.primary_map50` into each record *before* aggregating, so the arm summaries,
the sign-flip test, the trajectory contrast and the tidy CSV all apply to the subset endpoint; the
mean is `metrics/task.py::primary_mean`, the same function `scripts/gate_table.py` scored the
headroom with. `zero_shot.primary_n_classes` travels beside it: std 0.0000 at every stage is the
check that the denominator never moved.

**C2 was added here, declared before any mAP was read.** It was not pre-registered for E9 — both
earlier cells ran a faithfulness pass and this one had no plan for it. Adding a metric before the
numbers are known is legitimate; after would not be. Scored with the **reference** `yolo11s` judge,
never the in-loop one, and written back into `metrics.json` so the paired test is the one the
headline rests on.

**Command 2 OOM'd on its first attempt** (`torch.OutOfMemoryError: Tried to allocate 9.89 GiB`):
`t2o faithfulness` handed ultralytics a `list[Path]`, which is decoded up front and run as one batch,
`batch=` silently dropped. FLIR's 1,013-image val split is the first that did not fit. That finding is
SPEC-MIGRATION-19's (legacy: M3 E9 step 5, "Command 2 OOM'd on the first attempt"; F-ID minted by
SPEC-MIGRATION-19). Nothing wrote back before the crash; all twelve C2 passes are a clean re-run after
the fix (7a3ca94), which is the SHA the report ran at.

**The two stop-checks, read first.** (1) Completeness — 12 runs, every run at stage 3; `aggregate`
computes on the stages shared by every run, so one stump collapses them to `[0]` (F71). (2) Config
variance — nothing unexpected varying, no pair split across cards. **Both passed** (block 2's and
block 3's verdicts below).

From the saved report, block 3 verbatim — the record of the launch:

```
keys whose value differs across the campaign, grouped by value:
  coupling.task_weights
      [0.0, 0.0, 0.0, 0.0]         e3f-control-s0 e3f-control-s1 e3f-control-s2 e3f-control-s3 e3f-control-s4 e3f-control-s5
      [0.0, 1.0, 2.0, 3.0]         e3f-loop-s0 e3f-loop-s1 e3f-loop-s2 e3f-loop-s3 e3f-loop-s4 e3f-loop-s5
  runtime.device
      'cuda:0'                     e3f-control-s0 e3f-control-s1 e3f-control-s2 e3f-loop-s0 e3f-loop-s1 e3f-loop-s2
      'cuda:1'                     e3f-control-s3 e3f-control-s4 e3f-control-s5 e3f-loop-s3 e3f-loop-s4 e3f-loop-s5
  runtime.name
      12 distinct values, one per run
  train.seed
      0                            e3f-control-s0 e3f-loop-s0
      1                            e3f-control-s1 e3f-loop-s1
      2                            e3f-control-s2 e3f-loop-s2
      3                            e3f-control-s3 e3f-loop-s3
      4                            e3f-control-s4 e3f-loop-s4
      5                            e3f-control-s5 e3f-loop-s5

verdict: 0 unexpected varying key(s): none
pairs split across cards: none (a pair must share one card, or the GPU difference lands in the paired contrast)

fully resolved config of e3f-control-s0 (the reconstructed launch):
  coupling.grad_scale                          0.15
  coupling.reward_target                       None
  coupling.task_weights                        [0.0, 0.0, 0.0, 0.0]
  data.annotation_fraction                     1.0
  data.annotation_seed                         0
  data.manifest                                'dataset\\processed\\flir\\data.yaml'
  data.max_train_images                        600
  data.subset_seed                             0
  data.val_loss_images                         153
  detector.evaluation.batch                    16
  detector.evaluation.epochs                   50
  detector.evaluation.imgsz                    640
  detector.evaluation.init_weights             'runs\\inloop-flir-yolo11n\\weights\\best.pt'
  detector.in_loop.imgsz                       640
  detector.in_loop.weights                     'runs\\inloop-flir-yolo11n\\weights\\best.pt'
  detector.reference.batch                     16
  detector.reference.imgsz                     640
  detector.reference.weights                   'runs\\reference-flir-yolo11s\\weights\\best.pt'
  export.normalize                             'clamp'
  loss.gan                                     1.0
  loss.l2                                      1.0
  loss.lpips                                   5.0
  metrics.conf_threshold                       0.25
  metrics.iou_threshold                        0.5
  metrics.kid_subset_size                      50
  metrics.lpips_net                            'alex'
  runtime.device                               'cuda:0'
  runtime.group                                'e3-pix2pix-flir600'
  runtime.name                                 'e3f-control-s0'
  runtime.run_dir                              'runs'
  runtime.wandb                                True
  runtime.wandb_project                        'thermal-to-optical'
  runtime.workers                              8
  train.amp                                    True
  train.amp_dtype                              'bfloat16'
  train.batch_size                             8
  train.crop                                   None
  train.epochs_per_stage                       100
  train.hflip                                  0.5
  train.lr                                     0.0002
  train.lr_gamma                               0.95
  train.seed                                   0
  translator.backbone                          'pix2pix'
  translator.gan_mode                          'vanilla'
  translator.ndf                               64
  translator.net_d                             'basic'
  translator.net_g                             'resnet_9blocks'
  translator.ngf                               64
```

**The caveat that travels with whatever the result is** (F119): the de-roll corrected the
**labels**, which bought an honest floor. The 5.90 px roll still sits between pix2pix's input and its
`l2: 1.0` + `lpips: 5.0` supervision target, and a generator's only way to satisfy a systematic roll
is to blur. Warping the thermal *images* is a separate and much larger job. This confound exists on
the public cell and nowhere else.

## Results

As recorded in 123410b, 8705364 and e70760c. The tables are the source's, verbatim; each is
re-checked against the saved report or `docs/results/e3f-tidy.csv` (block 4 verbatim, 48 rows, no
path columns).

Per-arm mean ± sd, 3-class mAP50 (the primary), with the gate's two anchors for scale:

| arm | stage 0 | stage 1 | stage 2 | stage 3 |
| --- | --- | --- | --- | --- |
| control | 0.0722 ± .0535 | 0.0710 ± .0572 | 0.0940 ± .0472 | 0.0902 ± .0941 |
| loop | 0.0987 ± .0158 | 0.3736 ± .0486 | 0.4220 ± .0224 | **0.4415 ± .0251** |
| *floor — the judge on raw de-rolled thermal (step 3b)* | | | | *0.4499* |
| *ceiling — the judge on visible* | | | | *0.6566* |

Paired loop − control, exact two-sided sign-flip over 2⁶ assignments, bootstrap CI descriptive only:

| metric | stage 0 (null) | stage 1 | stage 2 | **stage 3 (headline)** |
| --- | --- | --- | --- | --- |
| **3-class mAP50** | +0.0266, p=.281, [−.005, +.061] | +0.3026, p=.031 | +0.3281, p=.031 | **+0.3512, p=.031, [+.292, +.405]** |
| 4-class `zero_shot.map50` | +0.0199, p=.281 | +0.2342, p=.031 | +0.2506, p=.031 | +0.2686, p=.031, [+.229, +.306] |
| `zero_shot.map50_95` | +0.0103, p=.250 | +0.1118, p=.031 | +0.1218, p=.031 | +0.1299, p=.031, [+.111, +.148] |
| `detector.map50` | +0.1055, p=.344 | +0.0988, p=.031 | +0.1012, p=.031 | +0.0986, p=.031, [+.079, +.117] |
| `fidelity.lpips` | −0.1792, **p=.031** | −0.0482, p=.406 | −0.0656, p=.062 | −0.0764, p=.094, [−.129, −.026] |
| `fidelity.ssim` | +0.1338, **p=.031** | +0.0681, p=.156 | +0.1303, p=.062 | +0.1635, p=.031, [+.066, +.268] |

As fractions of the gate headroom:

| | control, stage 3 | loop, stage 3 |
| --- | --- | --- |
| custom set (e3b, all-class; floor 0.1887, ceiling 0.9213, headroom +0.733) | **+83%** | **+90%** |
| FLIR (3-class; floor 0.4499, ceiling 0.6566, headroom +0.2066) | **−174%** | **−4%** |

Per class (the floor column is record 014's de-rolled gate):

| class | floor | loop, stage 3 | loop − floor | loop − control |
| --- | --- | --- | --- | --- |
| car | 0.6051 | 0.6956 | **+0.0905** | +0.4947 |
| person | 0.3772 | 0.4173 | **+0.0401** | +0.3849 |
| bicycle | 0.3674 | 0.2116 | **−0.1558** | +0.1741 |
| dog (13 instances, not in the headline) | 0.0242 | 0.0208 | −0.0034 | +0.0207, p=.031 — noise, as pre-registered |

C2 at stage 3, reference judge:

| rate | control | loop | paired | worst loop vs best control |
| --- | --- | --- | --- | --- |
| false-object (primary) | 0.5198 ± .2752 | 0.1874 ± .0209 | **−0.3324, p=.031** | 0.221 vs 0.242 |
| missed-object | 0.9229 ± .0824 | 0.5671 ± .0330 | −0.3558, p=.031 | 0.600 vs 0.786 |
| detection consistency | 0.0725 ± .0776 | 0.3792 ± .0320 | +0.3068, p=.031 | 0.346 vs 0.202 |

**Dose (block 7).** Detection share 12.4 / 17.7 / **21.1%**; raw detection loss
(`loss_det / grad_scale`) 3.58 → 2.85 → 2.27 across stages 1–3; `loss_gan` 1.82 → 2.30 in the
control, 1.77 → 2.06 in the loop.

**Cost (the re-run block 5, transcribed — not in the saved report).** The campaign cost
**~116.8 GPU-h** (one run per card, so run-hours are GPU-hours). A run is **9.51–9.61 h control,
9.94–10.44 h loop**; `control-s0`'s span runs from its resume (6.52 h), so it is counted as training
plus boundaries, 8.46 h. Of that, **102.3 h is translator training** (88%). Every stage boundary —
export, zero-shot, FID and the 50-epoch fine-tune together — is **0.27–0.35 h and flat across
stages**. Training plus four boundaries reproduces every other run's span to within 0.02 h. Median
epoch: stage 0 ~50 s, stage 3 ~101–112 s; per-stage growth ×2.03; control stage 0 50.5 s against the
solo probe's 43.524 s, ×1.16.

**From the saved report**, verbatim. Block 2's verdict, block 6's headline and trajectory lines,
block 7, and one run of the pre-fix block 5 (its `TOTAL span` and `bound_h` are the buggy ones,
F147, F148; `train_h` and `median_s` are unaffected):

```
verdict: 12 runs, every run reached stage 3: True
verdict: 0 unexpected varying key(s): none
pairs split across cards: none (a pair must share one card, or the GPU difference lands in the paired contrast)
paired (loop - control), zero_shot.primary_map50
  stage 0 (null) n=6  +0.0266  p=0.2812  95% CI [-0.0052, +0.0613]
  stage 1        n=6  +0.3026  p=0.0312  95% CI [+0.2689, +0.3371]
  stage 2        n=6  +0.3281  p=0.0312  95% CI [+0.3048, +0.3567]
  stage 3        n=6  +0.3512  p=0.0312  95% CI [+0.2919, +0.4053] <-- headline
trajectory (loop - control) of the gain over stage 0, zero_shot.primary_map50 -- sensitivity analysis, not the pre-registered endpoint
  stage 1        n=6  control -0.0011  loop +0.2749  diff +0.2760  p=0.0312  95% CI [+0.2525, +0.2973]
  stage 2        n=6  control +0.0218  loop +0.3233  diff +0.3015  p=0.0312  95% CI [+0.2533, +0.3526]
  stage 3        n=6  control +0.0181  loop +0.3427  diff +0.3247  p=0.0312  95% CI [+0.2659, +0.3846] <-- headline
-- loop arm, 6 run(s)
stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
    0    6     100   0.0     0.0000      0.0642      2.1666      1.7659          --      3.9967    0.0000    0.0%
    1    6     100   1.0     0.1500      0.0481      1.8877      1.8440      0.5375      4.3172    0.5375   12.4%
    2    6     100   2.0     0.3000      0.0455      1.8324      2.1115      0.4282      4.8458    0.8564   17.7%
    3    6     100   3.0     0.4500      0.0394      1.7132      2.0635      0.3411      4.8393    1.0233   21.1%

-- control arm (terms only), 6 run(s)
stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
    0    6     100   0.0     0.0000      0.0668      2.2120      1.8154          --      4.0943    0.0000    0.0%
    1    6  95-100   0.0     0.0000      0.0560      1.9823      1.9788          --      4.0171    0.0000    0.0%
    2    6     100   0.0     0.0000      0.0540      1.9102      2.2368          --      4.2011    0.0000    0.0%
    3    6     100   0.0     0.0000      0.0493      1.8207      2.3017          --      4.1717    0.0000    0.0%
run                          stage epochs  median_s  train_h  bound_h
e3f-control-s1                   0    100    49.659     1.40     2.13
e3f-control-s1                   1    100    65.090     1.84     2.64
e3f-control-s1                   2    100    84.035     2.35     3.06
e3f-control-s1                   3    100   101.259     2.78    37.28
e3f-control-s1               TOTAL span 46.51 h (config.yaml -> metrics.json)
two-up contention -- campaign median epoch vs step 4 (b)'s SOLO probe:
  control   71.460 s   solo  43.524 s   x1.64
  loop      81.820 s   solo  51.156 s   x1.60

per-stage growth, last stage's training / stage 0's: median x2.03 over 12 runs (step 0 measured ~+20%/stage on the custom set and never explained it)
```

Re-render without re-running (server, at e70760c or later, run dirs intact): `uv run python
scripts/campaign_report.py --runs 'runs/e3f-*' --stage 3 --primary-classes bicycle car person
--data dataset/processed/flir/data.yaml`

## Findings

### F136 — The pre-registered endpoint is positive at the sign-flip floor: 3-class mAP50 loop − control +0.3512, p = .031, 13× a stage-0 null that held (legacy: M3 E9 step 5 finding 1)

+0.3512 at p = 0.0312 = 2/2⁶, all six pairs the same sign, CI [+.292, +.405]; the stage-0 null is
+0.0266 (p = .281), so stage 3 is **13× the null** — "clearly larger" with no judgement call, unlike
e3b's 1.3× (F40). Monotone in dose (+0.027 → +0.303 → +0.328 → +0.351), and the trajectory contrast,
which the stage-0 draw cannot touch, agrees: **+0.3247, p = .031, [+.266, +.385]** (a sensitivity
analysis, never the endpoint). The smallest per-pair difference is +0.228, 4.7× FLIR's own 0.048
noise floor (F134). This is 6.9× e3b's +0.0512 (F35) — and F137 is why that ratio must not be quoted
as a bigger win.

### F137 — The loop lands on the raw-thermal floor (0.4415 against 0.4499) and the control far below it (−0.360): the detector loss prevents a loss plain translation incurs, it does not improve on no translation (legacy: M3 E9 step 5 finding 2)

The floor is the same judge on the same 1,013 val frames with the same de-rolled labels, fed raw
thermal (F115). Stage-3 loop is **0.4415 against 0.4499, −0.008**, inside the noise floor; only 2 of
6 seeds (s0 0.477, s1 0.467) clear it. The control is **−0.360** below it, and no control row at any
stage ever reaches it (best: 0.249, s0 stage 3). Even stage 0 — plain pix2pix in both arms — sits at
0.07–0.10. As fractions of the gate headroom, control and loop reach **+83% / +90%** on the custom
set but **−174% / −4%** on FLIR: on FLIR, pix2pix at 600 pairs destroys detectable content and the
loop restores it to roughly what raw thermal already had. The defensible sentence is *"the detector
loss prevents a 0.36 mAP50 loss that plain translation incurs"* — never *"translation with the loop
improves detection on FLIR"*, which this cell does not show. This finding governs every sentence
written about the cell.

### F138 — Per class the floor comparison splits: car (+0.0905) and person (+0.0401) clear the floor, bicycle loses 42% of its floor score (−0.1558) (legacy: M3 E9 step 5 finding 3)

Car and person — the large, warm, thermally legible classes — do clear the floor, so the loop arm buys
something real there. Bicycle falls from 0.3674 to 0.2116. The pre-registration called bicycle the
weak class (+0.1295 headroom, F108) and person the strongest (+0.2932, F107); bicycle is confirmed,
person is not — car carries the gain. Dog (13 instances) moves +0.0207, p = .031 — noise, as
pre-registered. **Hypothesis, untested:** bicycles are thin structures, and the 5.90 px roll between
pix2pix's input and its `l2` + `lpips` target is exactly what a generator satisfies by blurring
(F119). Warping the thermal images is the test.

### F139 — The control arm collapses (7 of 24 stage-rows below 0.02 mAP50) and the loop arm never does (0 of 24); the worst loop seed (0.414) beats the best control seed (0.249) (legacy: M3 E9 step 5 finding 4)

Collapsed rows in the control: s1 stage 1; s3 stages 2–3, where SSIM falls to 0.016 and FID rises to
413; s4 stages 0–1, LPIPS 1.009 at stage 0; s5 stages 0 and 3. The loop's stage-3 sd is 0.025 against
the control's 0.094, **3.7×** tighter, against e3b's 2.4× (F42), on a second dataset. The paired
effect does not rest on the collapses: dropping the two pairs whose control ends collapsed (s3, s5)
leaves +0.312. But the plain baseline fails criterion 4's "no collapse" on this dataset, and part of
what the loop measurably does here is **stabilise GAN training**.

### F140 — Stage 0 collapses by draw, not by arm: `control-s4` collapsed where `loop-s4` (same seed, same card, same computation) did not, so a single-seed FLIR pix2pix number is close to meaningless (legacy: M3 E9 step 5 finding 5)

Stage 0 runs the identical computation in both arms (F40). 2 of 6 control seeds against 0 of 6 loop
collapse there, Fisher p ≈ 0.45. Training is not bit-reproducible across arms on this hardware, which
is known; what is new is that the non-reproducibility reaches all the way to **collapse-or-not**.

### F141 — The fidelity null failed — LPIPS, SSIM and FID all favour the loop at stage 0, p = .031 — so no fidelity claim is made from this cell; the trajectory contrast is null on every fidelity metric (legacy: M3 E9 step 5 finding 6)

At stage 0, where the arms are λ-inert, the six LPIPS pair-differences are −0.001, −0.020, −0.036,
−0.095, −0.594, −0.330: one a tie, one nearly, the magnitude from F140's two collapsed control seeds.
The sign-flip test counts a −0.001 as a vote, so this is the test's known weakness on a heavy tail,
not a code path — but stage-3 fidelity cannot be read alone. Trajectory: LPIPS +0.103, p = .406; FID
+24.6, p = .875; SSIM +0.030, p = .688; PSNR −1.44, p = .594. Net: **no λ-attributable fidelity cost
and no λ-attributable fidelity gain** — unlike e3b's finding 9 (F43), where the source has the loop
trading about a third of the control's LPIPS improvement.

### F142 — `detector.map50` is not a λ effect: stage 3 +0.0986 (p = .031) against a stage-0 null of +0.1055, trajectory −0.0070, p = 1.000 (legacy: M3 E9 step 5 finding 7)

The whole stage-3 gap was already there at λ = 0 — F140's draw, inherited. Fine-tuning a detector on
the translations absorbs whatever the loop changed; only the zero-shot judge sees it.

### F143 — C2 favours the loop on all three rates (false-object −0.3324, missed-object −0.3558, consistency +0.3068, each p = .031) and the worst loop seed beats the best control on each, but "low" is not met (legacy: M3 E9 step 5 finding 8)

Worst loop against best control: false-object 0.221 vs 0.242, missed-object 0.600 vs 0.786,
consistency 0.346 vs 0.202. C2 was measured at stage 3 only, so it has **no stage-0 null** and cannot
separate λ from F140's draw on its own — the worst-vs-best column is what carries it. The loop arm
still misses 57% of the labelled objects, and 19% of its detections are false objects. Criterion 5
cannot be claimed from this cell.

### F144 — The dose landed in the band: detection share 12.4 / 17.7 / 21.1%, inside 20–30% at stage 3, which F126's +17.5% coupling surcharge had put in doubt (legacy: M3 E9 step 5 finding 9)

Against e3b's 10.0 / 16.1 / 19.8% (F38). Raw detection loss falls 3.58 → 2.85 → 2.27 across stages
1–3, **−37%**, against e3b's −28% (F37). `loss_gan` rises 1.82 → 2.30 (+27%) in the control and
1.77 → 2.06 (+17%) in the loop, against e3b's +10.6% (F39) — the loop arm is the *less*
adversarially strained one here, consistent with F139. Not re-tuned; recorded.

### F145 — The per-stage growth is ×2.03 (~27% a stage compounded), in the control arm too, so it is not coupling; two-up contention is ×1.16, and the report's ×1.64 folds the growth in (legacy: M3 E9 step 5 finding 10)

Median epoch roughly doubles from stage 0 (~50 s) to stage 3 (~101–112 s) **in the control arm**,
which never constructs a detector. Step 0 measured ~+20%/stage on the custom set (F89); this is ~+27%
compounded and still unexplained (F129). The report's "two-up ×1.64" compares an all-stage campaign
median against a **stage-0-only** solo probe; at stage 0 alone the control ran 50.5 s against the
probe's 43.524 s, **×1.16**, which is the contention figure F128 did not carry.

### F146 — The cell cost ~116.8 GPU-h, 13% over the top of the 77–103 GPU-h floor, for the two reasons the floor said it did not carry: two-up contention and growth beyond +20% a stage (legacy: M3 E9 step 5, "the campaign cost ~116.8 GPU-h")

102.3 h of it is translator training (88%). Every stage boundary is 0.27–0.35 h and flat across
stages, so the ×2.03 growth (F145) is translator training alone. Training plus four boundaries
reproduces every other run's span to within 0.02 h.

### F147 — `campaign_report.py` ended a run at `metrics.json`'s mtime, which `t2o faithfulness --write-back` rewrites, so every span became time-since-launch (13.8–63.0 h); the end is now the newest file under the last stage's directory (legacy: M3 E9 step 5 open item, "`campaign_report.py`'s wall clock is wrong after a `--write-back`")

The `TOTAL span` and stage-3 `bound_h` columns of the saved report measure time-since-launch, in
launch order per card. **Fixed** in 8705364: `_run_end` (`scripts/campaign_report.py:266` at the tag)
takes the detector fine-tune's final write, seconds before `metrics.json`, from a directory `t2o
faithfulness` only reads. The test pins `metrics.json` 100 h late and asserts the 4.50 h span; the
old code printed 100.00 h. Nothing had written into a stage directory since, so re-running the report
recovered the true spans (F146).

### F148 — Stages 0–2's `bound_h` subtracted the next stage's training from a dict filled inside the same loop, so the lookup was always 0 and each boundary carried a whole stage of training (2.1–3.4 h printed against ~0.3 h real) (legacy: M3 E9 step 5 open item, "the re-run exposed a second bug")

`control-s1` stage 0: 2.13 = 0.28 + stage 1's 1.84. Only stage 3 and the span were right once F147
was fixed. **Fixed** in e70760c: `stage_train` is filled before the loop
(`scripts/campaign_report.py:317` at the tag). The test now pins a stage-0 boundary too; the old code
printed 1.00 for 0.98.

### F149 — `control-s0` stage 1 shows 95 epochs because it was interrupted ~5 epochs in and resumed: it trained all 100, and only the record of loss and clock is short (legacy: M3 E9 step 5 open item, "`control-s0` stage 1 ran 95 epochs, not 100")

`Trainer.resume` restarts at the checkpoint's epoch + 1 (`engine/trainer.py:356`) and `train` records
only the epochs it runs (`:208`), so the first ~5 have no `EpochStats`. Two other traces agree: its
span (6.52 h) is the remaining 95 epochs plus three stages almost exactly — `run_loop` re-snapshots
`config.yaml` on every launch (`engine/loop.py:145`), so a resumed run's span starts at the resume —
and its stage-0 boundary is 0.51 h against everyone else's ~0.30, the lost epochs plus a short
restart. Nothing here moves a metric.

## Provenance caveats

- **Commits that wrote this range** (`git blame` at the tag), all 2026-10-01 (+03:00): a6876e6
  (10:41) the launch narrative, the endpoint, the commands, the read order and the caveat; 7a3ca94
  (11:14) the OOM diagnosis and fix; 123410b (13:47) the result, findings 1–10, the cost and the open
  items; 8705364 (14:22) the first wall-clock fix; e70760c (14:36) the second.
- **The saved report predates both wall-clock fixes.** It was generated 2026-10-01T08:32:47Z at
  7a3ca94, before 8705364 and e70760c, and its block 5 is the pre-fix one: spans 13.75–63.02 h, stage
  0–2 `bound_h` 1.87–3.42 h. **What re-derives from it:** `train_h` sums to 102.28 h; the stage 0–2
  boundaries, as `bound_h` minus the next stage's `train_h`, are 0.27–0.35 h (`control-s0`'s stage 0
  0.51); training plus those three boundaries plus ~0.3 h for stage 3 lands within ~0.03 h of every
  quoted span. **What does not:** the stage-3 boundaries, the exact spans, the ~116.8 GPU-h total and
  "within 0.02 h", which exist only in the source's transcription of the unsaved re-run.
- **"2.1–3.4 h printed" excludes `control-s0`'s stage-0 1.87**, which carries only its short
  95-epoch stage 1 (F149). F148's worked example gives 0.28 for `control-s1`'s stage-0 boundary; the
  report's rounded columns give 0.29.
- **`control-s0`'s stage 2–3 medians (67.215 s, 80.465 s) sit well below the other eleven runs'**
  (~81–90 s and ~101–112 s). The source quotes "~101–112 s" without it and does not say why it ran
  faster.
- **The campaign's launch is block 3, not a command.** The report itself labels the resolved config
  "the reconstructed launch". No `t2o loop` line is given here because none was recorded.
- **"Criterion 4" and "criterion 5" are the legacy numbering** of `RESEARCH_FINDINGS.md` §10's five
  criteria — Stability and Faithfulness, both kept in `docs/goal.md`'s six (Q12). "Low" is the
  legacy wording; this cell's C2 was not scored against `goal.md`'s Faithfulness margin.
- **The custom-set +83% / +90%** re-derive from E3's stage-3 rows as E8 reproduced them (C 0.7975,
  D 0.8487; F82, F86) against F01's 0.1887 / 0.9213 bracket.
- **F43's re-keying is approximate.** The source calls e3b's finding 9 a trade of "about a third of
  the control's LPIPS improvement"; F43's own claim is a generalisation gap, not a trade.
- **The per-class floor column is record 014's**, not this report's.
- **The W&B Runtime column and the campaign group were never read**; `e3-pix2pix-flir600` is the
  resolved config's value only.
- **Derived numbers re-checked from the saved report and the tidy CSV:** every cell of the per-arm,
  paired, per-class and C2 tables; 13× = 0.3512 / 0.0266; 6.9× = 0.3512 / 0.0512; 4.7× = 0.228 /
  0.048; the six pair-differences and their minimum; −174% and −4%; the seven collapsed rows and their
  SSIM 0.016, FID 413, LPIPS 1.009; worst loop 0.414 vs best control 0.249; +0.312 without s3, s5;
  3.7× = 0.0941 / 0.0251; Fisher 0.45; the six LPIPS stage-0 differences; the trajectory values; worst
  vs best on all three C2 rates; −42% = 1 − 0.2116 / 0.3674; the shares, the raw `loss_det` (block 7's
  `loss_det` ÷ 0.15), −37%, +27%, +17%; ×2.03, ×1.64, ×1.16 = 50.5 / 43.524, ~27% = 2.03^(1/3).
- **Code anchors, checked at the tag:** `scripts/campaign_report.py:266` (`_run_end`), `:317`
  (`stage_train`), `engine/trainer.py:208` (`train` from `start_epoch`), `:356`
  (`start_epoch = epoch + 1`), `engine/loop.py:145` (`config.snapshot`), `metrics/task.py:57`
  (`primary_mean`), `cli.py:229` (`--primary-classes`). The tests the source names for F147 and F148
  were not re-run here.
- **Cross-references, re-keyed:** e3b is record 007 — its +0.0512 F35, its 1.3× null ratio and
  identical stage 0 F40, its 2.4× variance F42, its finding 9 F43, its dose F37–F39. The dog rule is
  F94; the 1,013 val count F103; the floor and headroom F115; the roll confound F119; the stump trap
  F71; step 2b's 1.36 h clean against 2.29 h wall F100; the val-loss cut F125; the surcharge F126; the
  77–103 GPU-h floor F128; the +20%/stage growth F89 and F129; the 0.048 noise floor F134; `--resume`
  F135; the reference judge and in-loop `yolo11n` record 013. The list-source OOM is a no-run finding
  SPEC-MIGRATION-19 will mint.

## Next

E9's FLIR cell is read: the loop beats its control (F136) and does not beat raw thermal (F137). Open
from this readout, carried to the roadmap under `E9`:

- **Still wanted from W&B:** the Runtime column for the twelve runs (wall time including stalls,
  beside the fixed report's span) and the campaign `--group`, which should match the resolved config's
  `runtime.group: e3-pix2pix-flir600`.
- **The ×2.03 per-stage growth** (F145), now measured on two datasets and in an arm with no detector.

Posted by this record: warping the thermal **images**, not the labels, is the test of F138's
bicycle hypothesis and of F119's confound, and it is the separate, much larger job F119 names.
