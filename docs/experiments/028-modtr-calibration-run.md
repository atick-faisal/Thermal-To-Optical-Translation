# 028 — ModTr calibration run: the 400-epoch budget stands and the detection loss falls

**Date:** 2026-10-23 (verdict read 2026-10-25) · **Task:** MODTR-BASELINE-07 · **Machine:** VSIBMEOT05 (`cuda:0`, solo) · **Wall clock:** 6 h 38 min 18 s (13:12:36 → 19:50:54 +03:00)
**git_sha:** `pre-spec-migration-179-gac5390a` (`git describe --always --dirty`, so a clean tree at `ac5390a`) · **W&B:** none (`runtime.wandb: false` in `experiments/modtr_custom.yaml:106`; no W&B line in the log) · **Log:** `logs/2026-10-23-modtr-baseline-07-callibration-run.txt`, with `logs/2026-10-23-modtr-baseline-07-inner-split.txt`, `logs/2026-10-23-modtr-baseline-07-callibration-pilot.txt` and `logs/2026-10-23-modtr-baseline-07-callibration-verdict.txt`

## Command

From each log's provenance block, as run, in order:

```
uv run python scripts/modtr_calibration.py split --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml --out D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner
uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --seed 0 --device cuda:0 --monitor-map --no-detector --epochs 2 --name modtr-calibration-pilot
uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --seed 0 --device cuda:0 --monitor-map --no-detector --name modtr-calibration
uv run python scripts/modtr_calibration.py verdict --run runs/modtr-calibration
```

The record is the third and fourth commands. The split is their input, and the 2-epoch pilot only
proved the run starts, so neither gets a record of its own.

## Environment

The calibration run's block, verbatim:

```
=== PROVENANCE ===
captured: 2026-10-23 13:12:36 +03:00
hostname: VSIBMEOT05
os: Microsoft Windows Server 2022 Standard 10.0.20348 (build 20348)
cpu: AMD EPYC 7763 64-Core Processor
cpu_logical_cores: 64
ram_total: 128.0 GB
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16; NVIDIA A100-SXM4-40GB, 40960 MiB, 582.16
cuda_driver: 13.0
git_sha: pre-spec-migration-179-gac5390a
working_dir: D:\Atick\GitHub\Thermal-To-Optical-Translation
command: uv run t2o loop --config experiments/modtr_custom.yaml --data D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\data.yaml --in-loop-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --eval-init-weights D:\Atick\GitHub\RGBT-Fusion-Detection\optical_best_n_1.pt --seed 0 --device cuda:0 --monitor-map --no-detector --name modtr-calibration
error: Extra `matchers` is not defined in the project's `optional-dependencies` table
=== END PROVENANCE ===
```

The split, pilot and verdict blocks are the same apart from `captured:` (13:03:53, 13:07:09 on
2026-10-23; 09:04:22 on 2026-10-25) and their command lines. All four carry the same `git_sha`.
The ultralytics banner gives `Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130`.

## Configuration

modtr-baseline ADR-03 and ADR-04, as `experiments/modtr_custom.yaml` sets them at `ac5390a`.

| Axis | Value | Where it is evidenced |
| --- | --- | --- |
| Translator | ModTr (ResNet34 U-Net, fusion `clamp(x·h(x), 0, 1)`), detection-only objective | modtr-baseline ADR-01; `experiments/modtr_custom.yaml` |
| Budget | 1 stage × 400 epochs, `task_weight` 1.0, `grad_scale` 1.0, last checkpoint | `stage 0 done: task_weight=1.0 epochs=400`; `modtr_custom.yaml:33, :65` |
| Optimiser | AdamW, lr 1e-4, batch 8, accumulation 2, per-step cosine (ADR-04) | `modtr_custom.yaml:36, :38` |
| Seed | 0 | the command |
| Training data | the inner split's 480 train pairs | `480 paired samples, 480 annotated` |
| Monitor | the inner split's 120 val pairs, translated and scored every epoch | `exported 120 translated images`; every `all 120 373` row |
| Inner split | 480 / 120 of the custom set's 600 training images, fraction 0.20, seed 0; validation never read | the split log's `inner split` line |
| In-loop detector | `optical_best_n_1.pt` (`yolo11n`), the custom set's visible-trained detector (record 001) | the command's `--in-loop-weights` |
| Detector stage | off (`--no-detector`) | the command |
| Trigger | 10-epoch trailing mean of inner mAP@50 in points; fires when its final value is > 2.0 below its maximum | ADR-04; `src/t2o/analysis/epoch_budget.py` |
| Loss rule | mean `loss_det` over the last 5 epochs below the mean over the first 5 | ADR-03 |
| `config_hash` | not in the log | — |

## Results

The verdict log from the end of its provenance block to its finish stamp, verbatim:

```
2026-10-25 09:05:40,152 INFO    __main__: budget: 400 epochs (trigger did not fire) -- trailing mean peaks at 94.52 points ending epoch 360, ends at 94.10
2026-10-25 09:05:40,153 INFO    __main__: loss: PASS -- loss_det mean 6.2651 over the first 5 epochs, 1.2360 over the last 5
exit code: 0
=== FINISHED 2026-10-25 09:05:41 +03:00 ===
```

The split log's result line, verbatim:

```
2026-10-23 13:05:33,718 INFO    t2o.data.inner_split: inner split D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\data.yaml: 480 train / 120 val of D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul\data.yaml's 600 training images (fraction 0.20, seed 0)
```

The run log is 4,822 lines: one block of about twelve lines per epoch, 400 times. Its opening
through epoch 0 and its last two epochs follow. ANSI colour escapes and the per-batch
`Class Images Instances Box(P ...` progress lines are stripped; everything else is as the log
holds it.

```
2026-10-23 13:12:54,281 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner
2026-10-23 13:12:54,394 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\train\visible\images: 480 paired samples, 480 annotated
2026-10-23 13:12:55,058 INFO    t2o.data.manifest: data.yaml: 5 classes ['Connector', 'Fuse', 'Pole', 'Switch', 'Transformer'], root D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner
2026-10-23 13:12:56,036 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\train\visible\images: 480 paired samples, 480 annotated
2026-10-23 13:12:56,130 INFO    t2o.data.dataset: D:\Atick\data\dataset\Brazil-Aligned\yolo_rgbt_29_jul_inner\val\visible\images: 120 paired samples, 120 annotated
2026-10-23 13:13:44,098 INFO    t2o.engine.export: exported 120 translated images to runs\modtr-calibration\stage0\monitor\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 41.76.9 MB/s, size: 494.8 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-calibration\stage0\monitor\val\labels... 120 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 120/120 309.5it/s 0.4s
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-calibration\stage0\monitor\val\labels.cache
                   all        120        373      0.291      0.227      0.238      0.144
Speed: 1.2ms preprocess, 2.8ms inference, 0.0ms loss, 1.1ms postprocess per image
2026-10-23 13:13:54,249 INFO    t2o.metrics.task: eval optical_best_n_1.pt on data.yaml: P 0.2914  R 0.2273  mAP50 0.2385  mAP50-95 0.1439
2026-10-23 13:13:54,249 INFO    t2o.engine.trainer: epoch 0 took 58.1 s
[... epoch 0's memory line, epochs 1–396, and epoch 397 up to its memory line: elided here, in the log ...]
2026-10-23 19:48:56,026 INFO    t2o.engine.trainer: epoch 397 peak memory: 5.31 GiB allocated, 5.85 GiB reserved of 39.70 GiB
2026-10-23 19:49:43,552 INFO    t2o.engine.export: exported 120 translated images to runs\modtr-calibration\stage0\monitor\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 26.08.5 MB/s, size: 418.9 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-calibration\stage0\monitor\val\labels... 120 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 120/120 288.2it/s 0.4s
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-calibration\stage0\monitor\val\labels.cache
                   all        120        373      0.935      0.872       0.94      0.695
Speed: 1.0ms preprocess, 1.1ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-23 19:49:52,765 INFO    t2o.metrics.task: eval optical_best_n_1.pt on data.yaml: P 0.9353  R 0.8716  mAP50 0.9400  mAP50-95 0.6952
2026-10-23 19:49:52,771 INFO    t2o.engine.trainer: epoch 398 took 56.1 s
2026-10-23 19:49:52,771 INFO    t2o.engine.trainer: epoch 398 peak memory: 5.32 GiB allocated, 5.86 GiB reserved of 39.70 GiB
2026-10-23 19:50:41,134 INFO    t2o.engine.export: exported 120 translated images to runs\modtr-calibration\stage0\monitor\val (normalize=clamp)
Ultralytics 8.4.117  Python-3.12.7 torch-2.12.1+cu130 CUDA:0 (NVIDIA A100-SXM4-40GB, 40654MiB)
YOLO11n summary (fused): 101 layers, 2,583,127 parameters, 0 gradients, 6.4 GFLOPs
WARNING val: Slow image access detected (ping: 0.10.0 ms, read: 33.59.3 MB/s, size: 420.0 KB). Use local storage instead of remote/mounted storage for better performance. See https://docs.ultralytics.com/guides/model-training-tips/
val: Scanning D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-calibration\stage0\monitor\val\labels... 120 images, 3 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 120/120 306.3it/s 0.4s
val: New cache created: D:\Atick\GitHub\Thermal-To-Optical-Translation\runs\modtr-calibration\stage0\monitor\val\labels.cache
                   all        120        373      0.942      0.857      0.936      0.695
Speed: 1.2ms preprocess, 1.1ms inference, 0.0ms loss, 1.0ms postprocess per image
2026-10-23 19:50:50,782 INFO    t2o.metrics.task: eval optical_best_n_1.pt on data.yaml: P 0.9417  R 0.8568  mAP50 0.9361  mAP50-95 0.6947
2026-10-23 19:50:50,785 INFO    t2o.engine.trainer: epoch 399 took 57.4 s
2026-10-23 19:50:50,785 INFO    t2o.engine.trainer: epoch 399 peak memory: 5.32 GiB allocated, 5.86 GiB reserved of 39.70 GiB
2026-10-23 19:50:51,551 INFO    t2o.engine.loop: stage 0 done: task_weight=1.0 epochs=400
2026-10-23 19:50:51,561 INFO    t2o.cli: stage 0 (task_weight=1.0): 400 epochs
=== FINISHED 2026-10-23 19:50:54 +03:00 ===
```

Derived from all 400 `eval optical_best_n_1.pt ... mAP50` lines in the run log. The table samples
every 25th epoch plus epoch 0, epoch 9 (the first full window) and epoch 360 (the peak window's
last epoch). The trailing mean at epoch e is the mean of epochs e−9 … e.

| epoch | mAP@50 (points) | 10-epoch trailing mean |
| --- | --- | --- |
| 0 | 23.85 | — |
| 9 | 73.29 | 45.61 |
| 24 | 91.18 | 88.75 |
| 49 | 91.79 | 91.17 |
| 74 | 92.73 | 92.60 |
| 99 | 92.40 | 92.33 |
| 124 | 92.48 | 93.18 |
| 149 | 93.16 | 92.90 |
| 174 | 94.20 | 93.72 |
| 199 | 93.31 | 93.68 |
| 224 | 93.82 | 93.91 |
| 249 | 94.00 | 93.88 |
| 274 | 94.05 | 93.92 |
| 299 | 93.49 | 93.84 |
| 324 | 93.56 | 94.07 |
| 349 | 94.01 | 93.92 |
| 360 | 94.63 | 94.52 |
| 374 | 93.61 | 94.05 |
| 399 | 93.61 | 94.09 |

From the same lines: the single best epoch is 357 at 94.85, and 101 of the 400 epochs score
≥ 94.00. The trailing mean first comes within 2 points of its peak in the window ending at
epoch 64, and within 1 point in the window ending at epoch 170. Per-epoch `took` times have a
median of 59.2 s (range 53.7–73.3 s) and sum to 6.56 h. Peak memory is 5.36 GiB allocated and
5.90 GiB reserved of 39.70.

Re-render the verdict without re-running, on the server:
`uv run python scripts/modtr_calibration.py verdict --run runs/modtr-calibration`. Per-epoch
`loss_det` and `val_map50` are in `runs/modtr-calibration/stage0/monitor.jsonl` there.

## Findings

### F193 — The calibration trigger does not fire: the trailing inner mAP@50 ends 0.42 points below its peak against ADR-04's 2.0, so ModTr's campaign budget stays 400 epochs

The verdict reads a 10-epoch trailing mean that peaks at 94.52 points in the window ending epoch
360 and ends at 94.10. That is a 0.42-point drop, 4.8× under the 2.0-point trigger that ADR-04
fixed before the run. Re-deriving the curve from the log's own four-decimal `mAP50` lines,
independently of `monitor.jsonl`, gives the same peak window and 94.517 / 94.095, so the
verdict is not an artefact of the file it read. The curve has no late decline to speak of. It
climbs steeply to 91.17 by epoch 49 and 92.33 by epoch 99. From there it creeps up by about
1.8 points over the last 300 epochs, within 2 points of its eventual peak from epoch 64 onward.
400 epochs on 480 images gave no overfitting signal on this monitor. The campaign runs as
configured, with `epochs_per_stage: 400` unchanged.

### F194 — ADR-03's smoke rule passes by a wide margin: mean `loss_det` falls from 6.2651 over the first 5 epochs to 1.2360 over the last 5, −80%

The pass rule is "last-5 mean below first-5 mean". The run clears it by a factor of 5.1. So the
detection-only objective trains the translator at `grad_scale: 1.0`. The monitor shows the same
from the other side, inner mAP@50 rising 23.85 → 93.61 from epoch 0 to 399. ADR-03's worry
was that a small `grad_scale` would let Adam's ε damp training. It is not exercised: the
config runs at 1.0, not the 1e-2 default the worry was about, so this run neither confirms nor
refutes it.

### F195 — A 400-epoch ModTr run costs 6.64 h on one A100 with the per-epoch monitor on, and peaks at 5.90 GiB reserved, about a quarter of pix2pix's translator

Wall clock is 6 h 38 min from provenance to finish stamp. The 400 per-epoch times, which include
the monitor's export and val pass, sum to 6.56 h at a 59.2 s median. Against F90's 10.18 h for a
4-stage pix2pix control run, a ModTr run is about two thirds the time, before the monitor's cost
is taken out. Its memory peak, 5.36 GiB allocated and 5.90 GiB reserved of 39.70, is about a
quarter of the 22.59 GiB pix2pix's translator reserves (F131). The comparisons are those two
earlier measurements.

## Provenance caveats

- **The provenance block's Python probe failed**, as in record 027. Its `error:` line stands in
  for the `python`, `torch` and `cudnn` lines. The ultralytics banner gives Python 3.12.7 and
  torch 2.12.1+cu130. cuDNN and torch's own CUDA build are not recorded.
- **The in-loop detector has seen the monitor's images.** `optical_best_n_1.pt` was trained on
  all 600 of the custom set's training images, as visible images. The human confirmed this on
  2026-10-25; record 001 had said only "with ultralytics defaults". So the 120 inner-val stems
  are in its training set, which inflates the curve's absolute level: 94 points on translated
  inner-val, against the same detector's 0.9213 on real visible val (F02). The trigger reads the
  curve's shape, a drop from its own peak, so F193 is less exposed than the level. A lenient
  scorer could still mute a decline.
- **One seed.** Every number is seed 0, as ADR-04 fixed. How the curve varies across seeds is
  not measured.
- **The monitor's cost is not separable in the log.** Each epoch's `took` includes the export
  of 120 images and a val pass. The val pass runs about 10 s from the `exported` line to the
  `eval` line. The export's own time is not logged, so the campaign's per-epoch time without
  `--monitor-map` is not measured.
- **The pilot is not a reproducibility check.** Its epochs 0–1 read 22.08 / 26.32 against the
  run's 23.85 / 25.93 at the same seed. But `--epochs 2` shortens the cosine schedule's `T_max`,
  so the learning rate differed from the first step.
- **The run's `config_hash` is not in the log.** It is in `runs/modtr-calibration/config.yaml`
  on the server.
- **Slow image access.** ultralytics warns "Slow image access detected" on the monitor's val
  passes, as it did for e3b in record 027. That cost time, not numbers.
- **The log filenames spell "callibration".** They are cited here as saved.

## Next

ADR-04's condition is met: the result is recorded, the budget stays at 400, and the smoke rule
passed. So the ModTr campaign on the custom set can launch: seeds 0, 1 and 3, on all 600
training images, with `experiments/modtr_custom.yaml` unchanged. After that come the RT-DETR
judge per seed and `scripts/modtr_non_inferiority.py` against the e3b and e3t loops. A
falsifiable prediction from F195 for the campaign: one seed takes at most about 6.6 h on one
card, because the campaign does not pay for the monitor. ADR-04 does not say whether the FLIR
arm needs its own calibration run.
