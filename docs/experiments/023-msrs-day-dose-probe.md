# 023 — msrs-day dose probe: one 25-epoch pix2pix loop run at `grad_scale` 0.15, seed 0, solo

**Date:** 2026-10-07 · **Task:** MSRS-DAY-CAMPAIGN-01 · **Machine:** not recorded (no provenance block; the log's ultralytics lines read `CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)` under `D:\Atick\GitHub\Thermal-To-Optical-Translation`) · **Wall clock:** 2.06 h by the log's timestamps (19:10:28 → 21:13:50); the `loss_share.py` read followed at 21:14:10
**git_sha:** not recorded · **W&B:** none — the log carries no W&B line · **Log:** `logs/2026-10-07-msrs-day-campaign-probe-g015.txt` (the loop run); `logs/2026-10-07-msrs-day-campaign-probe-loss-share.txt` (the share table)

## Command

Not recorded: neither log echoes its command. The campaign runbook's step 2 is what this run was
meant to follow, and every value the log *can* confirm matches it (run name `e3md-probe-g015`,
25 epochs per stage, seed 0, `workers=8`, in-loop/eval-init weights
`runs\inloop-msrs-day-yolo11n\weights\best.pt`, a `yolo11s` zero-shot judge, `lambda_eff` 0.15 at
`w` 1.0). The runbook's form, not a transcript of what was typed:

```powershell
uv run t2o loop --config experiments/e3_pix2pix_loop.yaml `
    --data $DATA --in-loop-weights $INLOOP --eval-init-weights $INLOOP --reference-weights $JUDGE `
    --epochs 25 --seed 0 --workers 8 --name e3md-probe-g015 --resume --device cuda:0 2>&1 |
  Tee-Object "logs/$d-msrs-day-campaign-probe-g015.txt"

uv run python scripts/loss_share.py --runs runs/e3md-probe-g015 2>&1 |
  Tee-Object "logs/$d-msrs-day-campaign-probe-loss-share.txt"
```

with `$DATA = dataset/processed/msrs-day/data.yaml`, `$INLOOP =
runs/inloop-msrs-day-yolo11n/weights/best.pt`, `$JUDGE =
runs/reference-msrs-day-yolo11s-s1/weights/best.pt`.

## Environment

```
not recorded — the log predates any provenance block for this campaign
```

## Configuration

- **Why it ran.** `CLAUDE.md` requires a 25-epoch `loss_share.py` probe before a campaign, with the
  detection term at 20–30% of the objective. `grad_scale: 0.15` was calibrated on the custom set
  (F30, F38) and held on FLIR (F144); on msrs-day it was unmeasured (record 021, Next).
- **The run.** `experiments/e3_pix2pix_loop.yaml` unchanged: λ ramp `[0, 1, 2, 3]`, `grad_scale`
  0.15, `l2` 1.0 + `lpips` 5.0 + `gan` 1.0. Overridden to 25 epochs per stage and one seed (0). Solo
  on `cuda:0`.
- **Data.** `dataset/processed/msrs-day`: 5 classes `['car', 'person', 'bike', 'car_stop',
  'color_cone']`, 536 train / 179 val pairs, all annotated. No `--max-train-images` and no
  `--val-loss-images`, so the whole corpus and the whole val split were used.
- **Detectors.** The in-loop and evaluation-init detector is a 5-class `yolo11n` trained on msrs-day
  visible (runbook step 1, whose own log was not passed to this record). Each stage's 50-epoch
  fine-tune warm-starts from the previous stage's `best.pt`. Zero-shot is scored by a `yolo11s`
  (`YOLO11s summary (fused): 101 layers, 9,414,735 parameters` precedes each zero-shot pass).
- **The 5-class tree ran end to end.** Four stages of training, export, zero-shot, fidelity and
  fine-tune, with no traceback and no class-count warning from `detection/frozen.py:84-95`.

## Results

`logs/2026-10-07-msrs-day-campaign-probe-loss-share.txt`, whole:

```
2026-10-07 21:14:10,610 INFO    __main__: stage runs  epochs     w lambda_eff     loss_l2  loss_lpips    loss_gan    loss_det  loss_total     w*det   share
2026-10-07 21:14:10,610 INFO    __main__:     0    1      25   0.0     0.0000      0.0627      2.3081      0.8526          --      3.2235    0.0000    0.0%
2026-10-07 21:14:10,610 INFO    __main__:     1    1      25   1.0     0.1500      0.0593      2.3231      0.9642      0.7025      4.0490    0.7025   17.4%
2026-10-07 21:14:10,610 INFO    __main__:     2    1      25   2.0     0.3000      0.0590      2.3633      1.0741      0.6330      4.7625    1.2661   26.6%
2026-10-07 21:14:10,610 INFO    __main__:     3    1      25   3.0     0.4500      0.0589      2.3946      1.2584      0.5811      5.4552    1.7433   32.0%
```

`logs/2026-10-07-msrs-day-campaign-probe-g015.txt`, lines 1516–1519 (the run's closing summary):

```
2026-10-07 21:13:50,415 INFO    t2o.cli: stage 0 (task_weight=0.0): 25 epochs | zero-shot mAP50 0.2075 | LPIPS 0.4385 | fine-tuned mAP50 0.5261
2026-10-07 21:13:50,415 INFO    t2o.cli: stage 1 (task_weight=1.0): 25 epochs | zero-shot mAP50 0.3407 | LPIPS 0.4602 | fine-tuned mAP50 0.5754
2026-10-07 21:13:50,415 INFO    t2o.cli: stage 2 (task_weight=2.0): 25 epochs | zero-shot mAP50 0.3470 | LPIPS 0.4958 | fine-tuned mAP50 0.5900
2026-10-07 21:13:50,417 INFO    t2o.cli: stage 3 (task_weight=3.0): 25 epochs | zero-shot mAP50 0.4192 | LPIPS 0.4773 | fine-tuned mAP50 0.5879
```

Derived from the run log's `epoch N took S s` lines (25 per stage) and its stage timestamps (each
stage's `epoch 24 took` line is when its translator training ended):

| stage | mean s/epoch | translator training ends | stage done | boundary (export → zero-shot → fidelity → fine-tune) |
| --- | --- | --- | --- | --- |
| 0 | 40.46 | 19:27:35 | 19:37:59 | 10.4 min |
| 1 | 49.87 | 19:58:51 | 20:08:21 | 9.5 min |
| 2 | 53.04 | 20:30:32 | 20:40:29 | 10.0 min |
| 3 | 56.50 | 21:04:07 | 21:13:50 | 9.7 min |

The translator's peak allocation was 13.99 GiB at stage 0 and 15.02 GiB at
stage 1 of 39.70 GiB.

Re-render the share table without re-running: `uv run python scripts/loss_share.py --runs
runs/e3md-probe-g015` (on the server).

## Findings

### F165 — `grad_scale` 0.15 puts msrs-day's detection term at 17.4 / 26.6 / 32.0% after 25 epochs, the hottest pix2pix probe yet and 2.0 points over the band's top at stage 3

- **Against the two 25-epoch probes at the same `grad_scale`:**

  | probe | stage 1 | stage 2 | stage 3 |
  | --- | --- | --- | --- |
  | **msrs-day pix2pix (this run)** | **17.4%** | **26.6%** | **32.0%** |
  | custom pix2pix (F30) | 10.9% | 16.6% | 22.5% |
  | custom turbo (F56) | 16.6% | 22.9% | 24.5% |

- **Against the band.** Stages 1 and 2 sit inside or below 20–30%. Stage 3 is over by 2.0 points,
  a 6.7% relative excess.
- **Against record 007's re-tune rule** (`007:72-73`): re-tune below ~10% or above ~40%. 32.0% is
  in neither zone.
- **Against the single-run noise floor.** A 6.7% relative excess sits under F32's +7.1% total-loss
  gap between two nominally identical runs. One seed cannot tell 32.0% from 30%.
- **Why it runs hot.** At stage 3, raw `loss_det` is 0.5811 against the custom probe's 0.4136 (1.4×),
  and `loss_gan` is 1.2584 against 2.3294 (0.54×). A larger detection loss over a smaller GAN term
  makes the share bigger at the same weight.

### F166 — The 25-epoch probe says nothing reliable about full-length share on its own: the two earlier probe-to-campaign pairs moved by factors of 0.88 and 0.53

- **Pix2pix on the custom set:** stage 3 went from 22.5% at 25 epochs (F30) to 19.8% at 100 (F38),
  ×0.88. Applied here: about 28%, inside the band.
- **Turbo on the custom set:** stage 3 went from 24.5% (F56) to 12.9% (F77), ×0.53, because
  `loss_gan` inflated with the backbone. F56's "~21.6% at 100" projection was refuted. Applied
  here: about 17%, below the band.
- **What both agree on:** the direction. Full-length share has come in *below* the 25-epoch probe
  every time, consistent with F31's epoch-length trap. So the probe bounds stage 3 from above at
  roughly 32%, and the pix2pix precedent, the only one on this backbone, points into the band.
- **This is a projection from n = 1 precedent per backbone,** not a measurement. The campaign's own
  section 7 (`campaign_report.py`) is where the realised share gets measured.

### F167 — A solo msrs-day loop run prices at about 6.2 h at full length, two thirds of FLIR's realised ~9.7 h per run

- **Translator training at 100 epochs per stage,** from the per-stage means: (40.46 + 49.87 +
  53.04 + 56.50) × 100 s = 5.55 h.
- **Boundaries:** 9.5–10.4 min each, 0.66 h a run. FLIR's were 0.27–0.35 h each (F146).
  msrs-day's are about half, with a 50-epoch fine-tune on 536 images.
- **So a loop run is about 6.2 h solo,** and the cell's twelve runs about 70 GPU-h if a control run
  costs no more than a loop run. That is the floor: FLIR's two-up cell came in 13% over its
  solo-measured floor (F146), and F128 carried +20%/stage that a control arm also pays.
- **Against FLIR:** 116.8 GPU-h for twelve runs is ~9.7 h each (F146).
- **Stage-to-stage epoch time rises 40.5 → 56.5 s (+40%).** Coupling (FLIR: +17.5% flat in λ,
  F126) and the unexplained per-stage growth (F128) are mixed together in this one arm, so this
  run cannot price either separately.

## Provenance caveats

- **No provenance block.** `git_sha`, the hostname, the environment and the exact command lines are
  not recorded. The machine is read off ultralytics' device line and the `D:\` paths. Whether the
  server tree was at `main` after PR #3 is unknown. Nothing this probe measures depends on PR #3's
  `campaign_report.py` change.
- **`grad_scale` is not printed in the log.** That it was 0.15 rests on the share table's
  `lambda_eff` 0.1500 at `w` 1.0.
- **One seed, 25 epochs, no control arm.** The zero-shot and fine-tuned mAP50 lines in Results are
  all-class, scored on a probe with no paired control. They are recorded verbatim and are not a
  finding. Reading them for the campaign's endpoint would be tuning on the outcome.
- **The in-loop `yolo11n`'s own training log was not passed in.** Its epochs, `best.pt` epoch and
  mAP are unrecorded here.
- **ultralytics warned `Slow image access detected` 6 times,** at 32–48 MB/s on `D:` during the
  detector fine-tunes. The zero-shot passes read "Fast image access" at 55 MB/s. Two-up, the
  detector stages may slow down.
- **The boundary column is read from log timestamps,** not from `translator_last.pt` mtimes (the
  method of record 012 and `campaign_report.py`). The two should agree to about a second.

## Next

- **Launch the twelve-run cell at `grad_scale` 0.15 unchanged** (runbook step 3). This keeps the
  method identical to the custom set and FLIR. F165 sits inside record 007's no-re-tune zone and
  under the single-run noise floor.
- **Prediction, from F166:** the campaign's pooled six-run stage-3 share (`campaign_report.py`
  section 7) lands at **20–30%**, and below this probe's 32.0%. Above 30% would mean the epoch-length
  trap does not shrink msrs-day's share. Below 20% would mean the turbo-like ×0.53, not pix2pix's
  ×0.88.
- **Prediction, from F167:** the cell costs **70–95 GPU-h** two-up, below FLIR's 116.8.
